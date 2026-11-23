"""Two-stage training (paper §5, Alg. 1-2), on the xp6 OutcomeGRU's runs, splits and loss.

Stage 1 (if cfg.pretrain): encoder pretraining with InfoNCE over horizons 1..tau (+ InfoMax), early stopping on
the validation contrastive loss. Stage 2: decoder to the end of the run, masked MSE on [y_clean_acc, y_atk] (the
paper's Gaussian NLL with fixed sigma is the same objective up to scale), encoder fine-tuned at a lower learning
rate, optional CLUB balancing (treatment head on detached br, SGD; representation minimises CLUB with a weight
rising from 0 to club_alpha). Early stopping on the validation factual MSE (fixed, evenly spaced cuts).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import RunData
from .model import CausalCPC, contrastive_mask


@dataclass
class Config:
    hidden: int = 64
    tau: int = 20
    pretrain: bool = True    # stage 1 (CPC)
    infomax: bool = True     # + InfoMax in stage 1
    club: bool = False       # CLUB balancing in stage 2
    feedback: bool = False   # decoder fed with its own previous prediction
    a_cpc: float = 1.0
    a_infomax: float = 0.2   # paper configs: alpha_infonce 0.5, alpha_recons 0.1
    cpc_times: int = 4       # sampled times per batch for the CPC loss
    pre_epochs: int = 1000   # paper: max 1000 epochs, patience 100
    pre_patience: int = 100
    epochs: int = 300
    patience: int = 40
    lr: float = 5e-4
    wd: float = 1e-2
    enc_lr_mult: float = 0.5  # encoder learning rate in stage 2, relative (only after pretraining)
    club_alpha: float = 1.0
    club_ramp: int = 50       # epochs to reach ~club_alpha
    lr_treat: float = 1e-2
    batch: int = 32
    cuts: int = 16
    seed: int = 0


# Ablation ladder: A0 = the OutcomeGRU's recipe in the CPC architecture; each rung adds one ingredient.
PRESETS = {
    "A0": dict(pretrain=False, infomax=False, club=False, feedback=False),
    "A1": dict(pretrain=True, infomax=False, club=False, feedback=False),
    "A2": dict(pretrain=True, infomax=True, club=False, feedback=False),
    "A3": dict(pretrain=True, infomax=True, club=True, feedback=False),
    "A4": dict(pretrain=True, infomax=True, club=True, feedback=True),
}


def _stage1_loss(model: CausalCPC, data: RunData, rows: torch.Tensor, cfg: Config, gen: torch.Generator,
                 device) -> torch.Tensor:
    enc = data.enc[rows].to(device)
    mask = contrastive_mask(data.seed[rows].to(device))
    z, c = model.encode(enc)
    n = enc.shape[1]
    times = torch.randint(1, n - 1, (cfg.cpc_times,), generator=gen)
    loss = cfg.a_cpc * model.cpc_loss(z, c, mask, times)
    if cfg.infomax:
        t = int(torch.randint(2, n, (1,), generator=gen))
        t0 = int(torch.randint(1, t, (1,), generator=gen))
        loss = loss + cfg.a_infomax * model.infomax_loss(z, mask, t, t0)
    return loss


def pretrain(model: CausalCPC, tr: RunData, va: RunData, cfg: Config, device, log=print) -> list[dict]:
    opt = torch.optim.AdamW(model.pretrain_params(), lr=cfg.lr, weight_decay=cfg.wd)
    gen = torch.Generator().manual_seed(cfg.seed)
    best, best_state, patience, hist = math.inf, None, 0, []
    for epoch in range(cfg.pre_epochs):
        model.train()
        losses = []
        for rows in torch.randperm(len(tr), generator=gen).split(cfg.batch):
            if len(rows) < 2:
                continue
            loss = _stage1_loss(model, tr, rows, cfg, gen, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():  # fixed sampling: a comparable validation loss across epochs
            vgen = torch.Generator().manual_seed(12345)
            v = float(np.mean([_stage1_loss(model, va, rows, cfg, vgen, device).item()
                               for _ in range(4) for rows in torch.arange(len(va)).split(cfg.batch) if len(rows) > 1]))
        hist.append({"epoch": epoch, "train": float(np.mean(losses)), "val": v})
        if v < best - 1e-4:
            best, patience = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 10 == 0 or patience >= cfg.pre_patience:
            log(f"  stage 1 epoch {epoch}: train {hist[-1]['train']:.4f} val {v:.4f} (best {best:.4f})")
        if patience >= cfg.pre_patience:
            break
    model.load_state_dict(best_state)
    return hist


def _factual(model: CausalCPC, data: RunData, rows: torch.Tensor, cuts: torch.Tensor, device):
    W = data.W[rows].to(device)
    y, br, valid, idx = model.cut_predictions(data.enc[rows].to(device), W, data.t.to(device), cuts.to(device))
    Y = data.Y[rows][:, 1:].to(device)[:, idx]
    M = data.M[rows][:, 1:].to(device)[:, idx] * valid[None, :, :, None]
    return y, br, valid, idx, W, Y, M


def val_mse(model: CausalCPC, va: RunData, cuts: torch.Tensor, device) -> float:
    model.eval()
    num = den = 0.0
    with torch.no_grad():
        for rows in torch.arange(len(va)).split(16):
            y, _, _, _, _, Y, M = _factual(model, va, rows, cuts, device)
            num += float((((y - Y) ** 2) * M).sum())
            den += float(M.sum())
    return num / den


def fit_decoder(model: CausalCPC, tr: RunData, va: RunData, cfg: Config, device, log=print) -> list[dict]:
    enc_lr = cfg.lr * (cfg.enc_lr_mult if cfg.pretrain else 1.0)
    opt = torch.optim.AdamW([{"params": model.encoder_params(), "lr": enc_lr},
                             {"params": model.decoder_params(), "lr": cfg.lr}], weight_decay=cfg.wd)
    opt_t = torch.optim.SGD(model.treat_params(), lr=cfg.lr_treat, momentum=0.9)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    n_cuts = tr.n_rows - 1
    va_cuts = torch.arange(0, n_cuts, max(1, n_cuts // 50))
    best, best_state, patience, hist = math.inf, None, 0, []
    for epoch in range(cfg.epochs):
        alpha = cfg.club_alpha * (2 / (1 + math.exp(-10 * epoch / cfg.club_ramp)) - 1) if cfg.club else 0.0
        model.train()
        ly, lc, lt = [], [], []
        for rows in torch.randperm(len(tr), generator=gen).split(cfg.batch):
            cuts = (torch.arange(n_cuts) if n_cuts <= cfg.cuts
                    else torch.randperm(n_cuts, generator=gen)[:cfg.cuts].sort().values)
            y, br, valid, idx, W, Y, M = _factual(model, tr, rows, cuts, device)
            loss_y = (((y - Y) ** 2) * M).sum() / M.sum()
            loss = loss_y
            if cfg.club:
                w_now = W[:, 1:][:, idx]  # planned W of each step (B, C, S)
                v = valid[None].expand_as(w_now)
                w = w_now[v]
                logit = model.treat(br)[..., 0][v]
                perm = torch.randperm(len(w), generator=gen).to(device)
                pos = -F.binary_cross_entropy_with_logits(logit, w, reduction="none")
                neg = -F.binary_cross_entropy_with_logits(logit, w[perm], reduction="none")
                club = (pos - neg).mean()
                loss = loss + alpha * club
                lc.append(club.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cfg.club:  # critic step: treatment head only, on the detached representation
                bce = F.binary_cross_entropy_with_logits(model.treat(br.detach())[..., 0][v], w)
                opt_t.zero_grad()
                bce.backward()
                opt_t.step()
                lt.append(bce.item())
            ly.append(loss_y.item())
        v = val_mse(model, va, va_cuts, device)
        hist.append({"epoch": epoch, "train_mse": float(np.mean(ly)), "val_mse": v, "alpha": alpha,
                     "club": float(np.mean(lc)) if lc else None, "treat_bce": float(np.mean(lt)) if lt else None})
        if v < best - 1e-6:
            best, patience = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 10 == 0 or patience >= cfg.patience:
            extra = f" club {hist[-1]['club']:.4f} treat_bce {hist[-1]['treat_bce']:.4f} a {alpha:.2f}" if cfg.club else ""
            log(f"  stage 2 epoch {epoch}: train {hist[-1]['train_mse']:.5f} val {v:.5f} (best {best:.5f}){extra}")
        if patience >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return hist


def train_model(cfg: Config, tr: RunData, va: RunData, device, log=print) -> tuple[CausalCPC, dict]:
    torch.manual_seed(cfg.seed)
    model = CausalCPC(tr.enc.shape[-1], cfg.hidden, cfg.tau, cfg.feedback).to(device)
    hist = {"stage1": pretrain(model, tr, va, cfg, device, log) if cfg.pretrain else []}
    hist["stage2"] = fit_decoder(model, tr, va, cfg, device, log)
    hist["best_val_mse"] = min(h["val_mse"] for h in hist["stage2"])
    return model, hist


def save(folder: Path, model: CausalCPC, cfg: Config, tr: RunData, hist: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "cpc.pt")
    np.savez(folder / "norm.npz", x_mean=tr.d.x_mean, x_std=tr.d.x_std)
    (folder / "config.json").write_text(json.dumps({"config": asdict(cfg), "x_cols": tr.x_cols,
                                                    "in_dim": int(tr.enc.shape[-1])}, indent=1))
    (folder / "history.json").write_text(json.dumps(hist, indent=1))


def load(folder: Path, device) -> tuple[CausalCPC, Config]:
    meta = json.loads((folder / "config.json").read_text())
    cfg = Config(**meta["config"])
    model = CausalCPC(meta["in_dim"], cfg.hidden, cfg.tau, cfg.feedback).to(device)
    model.load_state_dict(torch.load(folder / "cpc.pt", map_location=device))
    return model.eval(), cfg
