"""xp8 learning: WHEN to switch Krum on, learned from the paired branches. Models A and B, offline test.

    ../.venv/Scripts/python xp8/learn8.py                # A and B, 3 seeds each -> results/xp8/learn/{A,B}/
    ../.venv/Scripts/python xp8/learn8.py --smoke        # few setups, few epochs (pipeline check)

Data: results/xp8/part*/all_blocks.parquet + manifest.csv (trunks Krum off / on and the branches of every setup).

Inputs at step t (what a defender sees before choosing W_t; outcomes never enter):
    the 25 gradient signals of step t + validation accuracy before step t   (signed log, normalised on the train
    trunks, clipped to +-10), W_{t-1}, t / T.
Utility of a step: u = clean acc - LAMBDA * targeted attack success (backdoor / flip; 0 for ALIE, SignFlip, none).
Gain of defending at a branch point t, from a trunk in state s (s = 0 off, 1 on):
    G_t = U(Krum on over [t, t+K), then s) - U(Krum off over [t, t+K), then s),   U = mean of u over steps t..T-1.

Model A (direct gain): a GRU reads the input rows; its head at row t predicts G_t. Trained on the measured G_t of
    the train branch points (MSE), early stopping on the val branch points.
Model B (outcome model, Causal CPC): CPC + InfoMax pretraining of the encoder on the train trunks; then a decoder
    predicts [clean acc, attack success] at every future step under an on/off plan; trained on every train
    trajectory (trunks + branches; factual MSE), early stopping on val. G_hat_t = U_hat(plan on) - U_hat(plan off).
Decision rule (both): defend at step t iff G_hat_t > 0.

Splits by setup (manifest 'split'): per configuration seeds 2-5 train, seed 1 val, seed 0 test.
Per model and seed: weights, normalisation, config, training history, val / test predictions. The seed with the
lowest val regret is the model used for the offline test tables and the live test (live8.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "xp8" else HERE  # local: CPC_DP/xp8/; Kaggle: files flat in $WORK
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import xp8 as X  # noqa: E402
from cpc import train as CT  # noqa: E402  (stage 1: CPC + InfoMax pretraining)
from cpc.model import CausalCPC  # noqa: E402

T, K = X.T, X.K
LAMBDA = 1.0
INPUTS = X.SIGNALS + ["x_val_acc_pre"]
N_IN = len(INPUTS) + 2  # + W_{t-1}, t / T
TARGETED = ("backdoor_win", "flip_win")
CLIP = 10.0
SCALE = 10.0  # model A predicts SCALE * G (gains are ~0.01-0.7)
SEEDS = (0, 1, 2)
RES = ROOT / "results" / "xp8"


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def moment(fam: str, onset: int, end: int, t: int) -> str:
    if fam == "none":
        return "no attack"
    for d, name in ((0, "onset"), (10, "onset+10"), (30, "onset+30")):
        if t == onset + d:
            return name
    return "during" if onset <= t <= end else ("before" if t < onset else "after")


# ============================================================================= data
@dataclass
class Bank:
    """Every trajectory as full arrays. A branch = its trunk's rows before t + its own rows from t."""
    x: np.ndarray       # (N, n, F) inputs, signed log (not normalised)
    W: np.ndarray       # (N, n) defense per step
    y: np.ndarray       # (N, n, 2) [clean acc, targeted attack success]
    meta: pd.DataFrame  # one row per trajectory (index = position in the arrays)
    labels: pd.DataFrame  # one row per branch point: traj (its trunk), branch, t, trunk, gain, setup info

    @property
    def n(self) -> int:
        return self.x.shape[1]


SETUP_COLS = ["seed", "split", "family", "config", "duration", "n_attackers", "onset", "end", "attack"]


def read_parts(res: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["setup", "trunk", "t_branch", "step", "W", *INPUTS, "y_clean_acc", "y_atk"]
    parts = sorted(p for p in res.glob("part*") if (p / "all_blocks.parquet").exists() and "smoke" not in p.name)
    if not parts:
        raise FileNotFoundError(f"no xp8 part in {res}")
    df = pd.concat([pd.read_parquet(p / "all_blocks.parquet", columns=cols) for p in parts], ignore_index=True)
    man = pd.concat([pd.read_csv(p / "manifest.csv") for p in parts], ignore_index=True).drop_duplicates("setup")
    return df, man


def build_bank(df: pd.DataFrame, man: pd.DataFrame) -> Bank:
    man = man.set_index("setup")
    df = df[df.step >= 0].sort_values(["setup", "trunk", "t_branch", "step"], kind="stable")
    n = int(df.step.max()) + 1
    xs, ws, ys, meta, labels = [], [], [], [], []

    def arrays(rows: pd.DataFrame, targeted: bool):
        x = signed_log(rows[INPUTS].to_numpy(np.float64)).astype(np.float32)
        atk = rows.y_atk.fillna(0.0).to_numpy(np.float32) if targeted else np.zeros(len(rows), np.float32)
        return x, rows.W.to_numpy(np.float32), np.stack([rows.y_clean_acc.to_numpy(np.float32), atk], -1)

    for (setup, trunk), g in df.groupby(["setup", "trunk"], sort=True):
        r = man.loc[setup]
        targeted = r.attack in TARGETED
        tr = g[g.t_branch == -1]
        if list(tr.step) != list(range(n)):
            raise ValueError(f"setup {setup} trunk {trunk}: steps are not 0..{n - 1}")
        tx, tw, ty = arrays(tr, targeted)
        if not (tw == trunk).all():
            raise ValueError(f"setup {setup} trunk {trunk}: W is not constant")
        ti = len(meta)
        xs.append(tx); ws.append(tw); ys.append(ty)
        meta.append({"setup": setup, "trunk": trunk, "t_branch": -1})
        u_tr = ty[:, 0] - LAMBDA * ty[:, 1]
        for t, b in g[g.t_branch >= 0].groupby("t_branch"):
            t = int(t)
            if list(b.step) != list(range(t, n)):
                raise ValueError(f"setup {setup} trunk {trunk} branch {t}: steps are not {t}..{n - 1}")
            bx, bw, by = arrays(b, targeted)
            x, w, y = tx.copy(), tw.copy(), ty.copy()
            x[t:], w[t:], y[t:] = bx, bw, by
            bi = len(meta)
            xs.append(x); ws.append(w); ys.append(y)
            meta.append({"setup": setup, "trunk": trunk, "t_branch": t})
            u_b = y[:, 0] - LAMBDA * y[:, 1]
            sign = 1.0 if trunk == 0 else -1.0  # + = defended path minus undefended path
            labels.append({"traj": ti, "branch": bi, "setup": setup, "trunk": trunk, "t": t,
                           "gain": float(sign * (u_b[t:].mean() - u_tr[t:].mean())),
                           "moment": moment(r.family, int(r.onset), int(r.end), t)})
    meta = pd.DataFrame(meta).join(man[SETUP_COLS], on="setup")
    labels = pd.DataFrame(labels).join(man[SETUP_COLS], on="setup")
    return Bank(np.stack(xs), np.stack(ws), np.stack(ys), meta, labels)


def smoke_subset(df: pd.DataFrame, man: pd.DataFrame, per: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A few setups per (family, split): the smallest real dataset that exercises every code path."""
    keep = man.groupby(["family", "split"]).head(per).setup
    return df[df.setup.isin(keep)], man[man.setup.isin(keep)]


@dataclass
class Norm:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Norm":
        f = x.reshape(-1, x.shape[-1]).astype(np.float64)
        m, s = np.nanmean(f, 0), np.nanstd(f, 0)
        s[~(s > 1e-6)] = 1.0
        return cls(m.astype(np.float32), s.astype(np.float32))

    def _z(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.nan_to_num((x - self.mean) / self.std, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)

    def enc(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """x (..., n, F) signed-log inputs, W (..., n) -> encoder rows (..., n, F + 2): [z, W_{t-1}, t / T]."""
        n = x.shape[-2]
        wp = np.concatenate([np.zeros_like(W[..., :1]), W[..., :-1]], -1)
        t = np.broadcast_to(np.arange(n, dtype=np.float32) / T, W.shape)
        return np.concatenate([self._z(x), wp[..., None], t[..., None]], -1).astype(np.float32)

    def row(self, x_t: np.ndarray, w_prev: float, t: int) -> np.ndarray:
        """One encoder row from one step's signed-log inputs (live use). Same values as enc()[..., t, :]."""
        return np.concatenate([self._z(x_t), [w_prev, t / T]]).astype(np.float32)

    def save(self, f: Path) -> None:
        np.savez(f, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, f: Path) -> "Norm":
        d = np.load(f)
        return cls(d["mean"], d["std"])


# ============================================================================= models
class GainGRU(nn.Module):
    """Model A: input rows -> GRU -> SCALE * G_t at every row (row t has seen the signals of steps 0..t)."""

    def __init__(self, in_dim: int = N_IN, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(in_dim, hidden), nn.SELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SELU(), nn.Linear(hidden, 1))

    def forward(self, enc: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        c, h = self.gru(self.local(enc), h)
        return self.head(c)[..., 0], h


def plan_gain(model: CausalCPC, h0: torch.Tensor, w_prev0: torch.Tensor, t: torch.Tensor, s: torch.Tensor,
              n: int) -> torch.Tensor:
    """Model B: G_hat = U_hat(on over [t, t+K) then s) - U_hat(off over [t, t+K) then s), U over steps t..n-1.

    h0 (L, H) encoder state at row t; w_prev0 (L,) = W_{t-1}; t, s (L,) long. Returns (L,)."""
    S = n - int(t.min())
    j = torch.arange(S, device=h0.device)
    steps = t[:, None] + j[None]
    valid = (steps < n).float()
    tt = steps.clamp(max=n - 1).float() / T
    first = j[None] < K
    out = []
    for on in (1.0, 0.0):
        plan = torch.where(first, torch.full_like(tt, on), s[:, None].float().expand_as(tt))
        w_prev = torch.cat([w_prev0[:, None].float(), plan[:, :-1]], 1)
        y, _ = model.decode(h0, w_prev, plan, tt)
        u = y[..., 0] - LAMBDA * y[..., 1]
        out.append((u * valid).sum(1) / valid.sum(1))
    return out[0] - out[1]


def decode_cuts(model: CausalCPC, c: torch.Tensor, W: torch.Tensor, cuts: torch.Tensor):
    """Factual predictions from row `cut` (per trajectory) for steps cut..n-1 under the trajectory's own W.

    c (B, n, H), W (B, n), cuts (B, C) long. Returns y (B, C, S, 2), valid (B, C, S), idx (B, C, S)."""
    B, n = W.shape
    S = n - int(cuts.min())
    raw = cuts[..., None] + torch.arange(S, device=W.device)
    valid, idx = raw < n, raw.clamp(max=n - 1)
    wp = torch.cat([torch.zeros_like(W[:, :1]), W[:, :-1]], 1)
    bi = torch.arange(B, device=W.device)[:, None, None]
    h0 = c[torch.arange(B, device=W.device)[:, None], cuts]
    C = cuts.shape[1]
    y, _ = model.decode(h0.reshape(B * C, -1), wp[bi, idx].reshape(B * C, S), W[bi, idx].reshape(B * C, S),
                        (idx.float() / T).reshape(B * C, S))
    return y.reshape(B, C, S, -1), valid, idx


# ============================================================================= training
@dataclass
class HP:
    hidden: int = 64
    lr: float = 1e-3
    wd: float = 1e-2
    batch: int = 32
    dropout: float = 0.1
    a_epochs: int = 400
    a_patience: int = 40
    pre_epochs: int = 300     # B stage 1 (CPC + InfoMax)
    pre_patience: int = 30
    b_lr: float = 5e-4
    b_epochs: int = 150       # B stage 2 (decoder)
    b_patience: int = 20
    b_cuts: int = 8           # random cuts per trajectory per batch (+ the setup's branch points)


def _label_tensors(bank: Bank, norm: Norm, split: str):
    L = bank.labels[bank.labels.split == split]
    trajs = sorted(L.traj.unique())
    pos = {t: i for i, t in enumerate(trajs)}
    enc = torch.tensor(norm.enc(bank.x[trajs], bank.W[trajs]))
    G = torch.full((len(trajs), bank.n), float("nan"))
    for r in L.itertuples():
        G[pos[r.traj], r.t] = r.gain * SCALE
    return enc, G


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    m = ~torch.isnan(target)
    return ((pred - target.nan_to_num()) ** 2)[m].mean()


def train_A(bank: Bank, norm: Norm, seed: int, hp: HP, device, log=print) -> tuple[GainGRU, dict]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tr_enc, tr_G = _label_tensors(bank, norm, "train")
    va_enc, va_G = _label_tensors(bank, norm, "val")
    va_enc, va_G = va_enc.to(device), va_G.to(device)
    model = GainGRU(N_IN, hp.hidden, hp.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.wd)
    best, best_state, patience, hist = math.inf, None, 0, []
    for epoch in range(hp.a_epochs):
        model.train()
        losses = []
        for rows in torch.randperm(len(tr_enc), generator=gen).split(hp.batch):
            loss = _masked_mse(model(tr_enc[rows].to(device))[0], tr_G[rows].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            v = float(_masked_mse(model(va_enc)[0], va_G))
        hist.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": v})
        if v < best - 1e-6:
            best, patience = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 20 == 0 or patience >= hp.a_patience:
            log(f"  A seed {seed} epoch {epoch}: train {hist[-1]['train_mse']:.4f} val {v:.4f} (best {best:.4f})")
        if patience >= hp.a_patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), {"epochs": hist, "best_val_mse": best}


class _Seqs:
    """Duck-typed view for cpc.train.pretrain: .enc (N, n, F) and .seed (N,)."""

    def __init__(self, enc: torch.Tensor, seed: torch.Tensor):
        self.enc, self.seed = enc, seed

    def __len__(self) -> int:
        return len(self.enc)


def _traj_tensors(bank: Bank, norm: Norm, idx: np.ndarray):
    enc = torch.tensor(norm.enc(bank.x[idx], bank.W[idx]))
    return enc, torch.tensor(bank.W[idx]), torch.tensor(bank.y[idx])


def _branch_points(bank: Bank, idx: np.ndarray, width: int = 4) -> torch.Tensor:
    """(len(idx), width): the branch points of each trajectory's setup (padded by repetition)."""
    bps = bank.labels.groupby("setup").t.apply(lambda v: sorted(set(v))).to_dict()
    out = []
    for s in bank.meta.setup.to_numpy()[idx]:
        p = bps.get(s, [0]) or [0]
        out.append((p * width)[:width])
    return torch.tensor(out, dtype=torch.long)


def _factual_mse(model: CausalCPC, enc, W, Y, cuts, device) -> tuple[torch.Tensor, torch.Tensor]:
    c = model.encode(enc.to(device))[1]
    W, Y = W.to(device), Y.to(device)
    y, valid, idx = decode_cuts(model, c, W, cuts.to(device))
    B = len(W)
    Yt = Y[torch.arange(B, device=device)[:, None, None], idx]
    m = valid[..., None].float()
    return (((y - Yt) ** 2) * m).sum(), m.sum() * y.shape[-1]


def train_B(bank: Bank, norm: Norm, seed: int, hp: HP, device, log=print) -> tuple[CausalCPC, dict]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    meta = bank.meta
    model = CausalCPC(N_IN, hp.hidden, tau=20, feedback=False, n_out=2).to(device)
    # stage 1: CPC + InfoMax on the trunks (same-seed trajectories are never negatives of each other)
    seqs = {}
    for split in ("train", "val"):
        idx = meta.index[(meta.split == split) & (meta.t_branch == -1)].to_numpy()
        seqs[split] = _Seqs(_traj_tensors(bank, norm, idx)[0], torch.tensor(meta.seed.to_numpy()[idx]))
    cfg = CT.Config(hidden=hp.hidden, pretrain=True, infomax=True, pre_epochs=hp.pre_epochs,
                    pre_patience=hp.pre_patience, lr=hp.b_lr, batch=hp.batch, seed=seed)
    h1 = CT.pretrain(model, seqs["train"], seqs["val"], cfg, device, log=lambda m: log(f"  B seed {seed}{m}"))
    # stage 2: decoder on every trajectory (trunks + branches), factual MSE on [clean acc, attack success]
    tr = meta.index[meta.split == "train"].to_numpy()
    va = meta.index[meta.split == "val"].to_numpy()
    tr_enc, tr_W, tr_Y = _traj_tensors(bank, norm, tr)
    va_enc, va_W, va_Y = _traj_tensors(bank, norm, va)
    tr_bp, va_bp = _branch_points(bank, tr), _branch_points(bank, va)
    n = bank.n
    va_cuts = torch.cat([va_bp, torch.arange(0, n - 1, max(1, n // 10)).expand(len(va), -1)], 1)
    opt = torch.optim.AdamW([{"params": model.encoder_params(), "lr": hp.b_lr * 0.5},
                             {"params": model.decoder_params(), "lr": hp.b_lr}], weight_decay=hp.wd)
    best, best_state, patience, hist = math.inf, None, 0, []
    for epoch in range(hp.b_epochs):
        model.train()
        losses = []
        for rows in torch.randperm(len(tr), generator=gen).split(hp.batch):
            cuts = torch.cat([tr_bp[rows], torch.randint(0, n - 1, (len(rows), hp.b_cuts), generator=gen)], 1)
            num, den = _factual_mse(model, tr_enc[rows], tr_W[rows], tr_Y[rows], cuts, device)
            loss = num / den
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        num = den = 0.0
        with torch.no_grad():
            for rows in torch.arange(len(va)).split(hp.batch):
                a, b = _factual_mse(model, va_enc[rows], va_W[rows], va_Y[rows], va_cuts[rows], device)
                num, den = num + float(a), den + float(b)
        v = num / den
        hist.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": v})
        if v < best - 1e-7:
            best, patience = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 5 == 0 or patience >= hp.b_patience:
            log(f"  B seed {seed} stage 2 epoch {epoch}: train {hist[-1]['train_mse']:.5f} val {v:.5f} "
                f"(best {best:.5f})")
        if patience >= hp.b_patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), {"stage1": h1, "stage2": hist, "best_val_mse": best}


# ============================================================================= prediction
@torch.no_grad()
def predict(kind: str, model: nn.Module, bank: Bank, norm: Norm, labels: pd.DataFrame, device,
            chunk: int = 32) -> np.ndarray:
    """G_hat at each labelled branch point (from its trunk's history, the trunk's state as s)."""
    pred = np.full(len(labels), np.nan)
    L = labels.reset_index(drop=True)
    for trajs in np.array_split(np.array(sorted(L.traj.unique())), max(1, math.ceil(L.traj.nunique() / chunk))):
        enc = torch.tensor(norm.enc(bank.x[trajs], bank.W[trajs])).to(device)
        pos = {t: i for i, t in enumerate(trajs)}
        sel = L.index[L.traj.isin(trajs)].to_numpy()
        rows = torch.tensor([pos[t] for t in L.traj.to_numpy()[sel]], device=device)
        ts = torch.tensor(L.t.to_numpy()[sel], device=device)
        if kind == "A":
            pred[sel] = (model(enc)[0][rows, ts] / SCALE).cpu().numpy()
        else:
            c = model.encode(enc)[1]
            Wt = torch.tensor(bank.W[trajs], device=device)
            w_prev0 = torch.where(ts > 0, Wt[rows, (ts - 1).clamp(min=0)], torch.zeros_like(ts, dtype=Wt.dtype))
            s = torch.tensor(L.trunk.to_numpy()[sel], device=device)
            pred[sel] = plan_gain(model, c[rows, ts], w_prev0, ts, s, bank.n).cpu().numpy()
    return pred


class Controller:
    """Live use: feed one step at a time (raw signals in INPUTS order), get G_hat_t, then report the chosen W_t."""

    def __init__(self, folder: Path, device, n_steps: int = T):
        self.kind, self.model, self.norm, self.meta = load_model(folder, device)
        self.device, self.n = device, n_steps
        self.h, self.w_prev, self.t = None, 0.0, 0

    @torch.no_grad()
    def gain(self, x_raw: np.ndarray) -> float:
        row = torch.tensor(self.norm.row(signed_log(np.asarray(x_raw, np.float64)).astype(np.float32),
                                         self.w_prev, self.t), device=self.device)[None, None]
        if self.kind == "A":
            out, self.h = self.model(row, self.h)
            return float(out[0, 0]) / SCALE
        z = self.model.local(row)
        c, self.h = self.model.ar(z, self.h)
        d = self.device
        return float(plan_gain(self.model, c[:, 0], torch.tensor([self.w_prev], device=d),
                               torch.tensor([self.t], device=d), torch.tensor([int(self.w_prev)], device=d),
                               self.n)[0])

    def update(self, w: int) -> None:
        self.w_prev, self.t = float(w), self.t + 1


# ============================================================================= evaluation
def decisions(L: pd.DataFrame) -> pd.DataFrame:
    d = L.copy()
    d["defend"] = d.pred > 0
    d["right"] = d.defend == (d.gain > 0)
    d["value"] = np.where(d.defend, d.gain, 0.0)  # gain obtained relative to not defending
    d["oracle"] = d.gain.clip(lower=0.0)
    d["regret"] = d.oracle - d.value
    return d


def summarise(d: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    def one(x):
        c = np.corrcoef(x.pred, x.gain)[0, 1] if len(x) > 2 and x.pred.std() > 0 and x.gain.std() > 0 else np.nan
        return pd.Series({"n": len(x), "right_decisions": x.right.mean(), "model": x.value.mean(),
                          "always_on": x.gain.mean(), "always_off": 0.0, "oracle": x.oracle.mean(),
                          "regret": x.regret.mean(), "defend_rate": x.defend.mean(),
                          "true_defend_rate": (x.gain > 0).mean(), "corr": c, "mae": (x.pred - x.gain).abs().mean()})
    if not by:
        return one(d).to_frame().T
    return d.groupby(by).apply(one, include_groups=False)


# ============================================================================= save / load
def save_model(folder: Path, kind: str, model: nn.Module, norm: Norm, seed: int, hp: HP, hist: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "model.pt")
    norm.save(folder / "norm.npz")
    (folder / "config.json").write_text(json.dumps({
        "kind": kind, "seed": seed, "hp": asdict(hp), "inputs": INPUTS, "in_dim": N_IN, "T": T, "K": K,
        "lambda": LAMBDA, "scale": SCALE, "rule": "defend at step t iff G_hat_t > 0"}, indent=1))
    (folder / "history.json").write_text(json.dumps(hist, indent=1))


def load_model(folder: Path, device):
    meta = json.loads((folder / "config.json").read_text())
    hp = HP(**meta["hp"])
    if meta["kind"] == "A":
        model = GainGRU(meta["in_dim"], hp.hidden, hp.dropout)
    else:
        model = CausalCPC(meta["in_dim"], hp.hidden, tau=20, feedback=False, n_out=2)
    model.load_state_dict(torch.load(folder / "model.pt", map_location=device))
    return meta["kind"], model.to(device).eval(), Norm.load(folder / "norm.npz"), meta


# ============================================================================= driver
def run(kind: str, bank: Bank, norm: Norm, out: Path, seeds, hp: HP, device, log=print) -> dict:
    rows = []
    for seed in seeds:
        t0 = time.time()
        log(f"== {kind} seed {seed}")
        model, hist = (train_A if kind == "A" else train_B)(bank, norm, seed, hp, device, log)
        folder = out / kind / f"seed{seed}"
        save_model(folder, kind, model, norm, seed, hp, hist)
        res = {"seed": seed, "best_val_mse": hist["best_val_mse"], "train_minutes": round((time.time() - t0) / 60, 1)}
        for split in ("val", "test"):
            L = bank.labels[bank.labels.split == split].copy()
            L["pred"] = predict(kind, model, bank, norm, L, device)
            d = decisions(L)
            d.to_csv(folder / f"pred_{split}.csv", index=False)
            s = summarise(d).iloc[0]
            res.update({f"{split}_{k}": float(s[k]) for k in ("regret", "right_decisions", "model", "corr")})
        rows.append(res)
        log(f"   {kind} seed {seed}: val regret {res['val_regret']:.4f} right {res['val_right_decisions']:.3f} | "
            f"test regret {res['test_regret']:.4f} right {res['test_right_decisions']:.3f} "
            f"({res['train_minutes']} min)")
    seeds_df = pd.DataFrame(rows)
    seeds_df.to_csv(out / kind / "seeds.csv", index=False)
    best = int(seeds_df.sort_values(["val_regret", "best_val_mse"]).iloc[0].seed)
    (out / kind / "best.json").write_text(json.dumps({"best_seed": best, "chosen_by": "lowest val regret",
                                                      "folder": f"seed{best}"}, indent=1))
    d = pd.read_csv(out / kind / f"seed{best}" / "pred_test.csv")
    test = out / kind / "test"
    test.mkdir(exist_ok=True)
    d.to_csv(test / "branch_points.csv", index=False)
    summarise(d).to_csv(test / "overall.csv", index=False)
    for by in (["family"], ["family", "moment"], ["config"], ["trunk"], ["family", "n_attackers"]):
        summarise(d, by).round(4).to_csv(test / f"by_{'_'.join(by)}.csv")
    log(f"== {kind}: best seed {best}\n{summarise(d, ['family']).round(3).to_string()}")
    return {"kind": kind, "best_seed": best, **summarise(d).iloc[0].to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=Path, default=RES, help="folder with part1/, part2/")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--models", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true", help="1 setup per (family, split), few epochs")
    args = ap.parse_args()
    out = args.out or args.res / ("learn_smoke" if args.smoke else "learn")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    df, man = read_parts(args.res)
    hp = HP()
    seeds = args.seeds
    if args.smoke:
        df, man = smoke_subset(df, man)
        hp = HP(a_epochs=3, a_patience=2, pre_epochs=2, pre_patience=1, b_epochs=2, b_patience=1)
        seeds = seeds[:2]
    bank = build_bank(df, man)
    del df
    m = bank.meta
    norm = Norm.fit(bank.x[m.index[(m.split == "train") & (m.t_branch == -1)].to_numpy()])
    print(f"bank: {len(m)} trajectories x {bank.n} steps, {len(bank.labels)} branch points "
          f"{bank.labels.split.value_counts().to_dict()}, device {device} ({time.time() - t0:.0f} s)", flush=True)
    bank.labels.to_csv(out / "branch_points_all.csv", index=False)
    log = lambda s: print(s, flush=True)  # noqa: E731
    summary = [run(kind, bank, norm, out, seeds, hp, device, log) for kind in args.models]
    pd.DataFrame(summary).to_csv(out / "offline_test.csv", index=False)
    print(pd.DataFrame(summary).round(4).to_string(), f"\n== learn done in {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
