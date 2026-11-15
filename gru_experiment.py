"""GRU outcome model: predict future outcomes under a candidate defense schedule.

    python gru_experiment.py                                    # xp1 (frozen model in results/xp1/gru)
    python gru_experiment.py --cf runs_cf/all_blocks.csv        # xp1 + check 3 against true counterfactual runs
    python gru_experiment.py --runs results/xp2/runs_xp2/all_blocks.csv --out results/xp2/gru   # xp2

Model. An encoder GRU reads the observable history up to block b: the signals X
(incl. validation accuracy), the defense flag W, time and the defense type. A
decoder GRU, started from the encoder state, reads the candidate future
schedule W[b+1:] and outputs y_clean_acc and y_atk for every future block.
Inputs never include U (attack type / activity), Y_clean or Y_atk: the attack
must be inferred from X. Future X is not predicted (no feedback of errors).

The run length is read from the data (xp1: 50 blocks of 10 rounds; xp2: 250
blocks of 2 rounds). Training samples CUTS_PER_BATCH cut points per batch
(all of them when a run has fewer cuts).

Split by seed, stratified by attack (see assign_splits): 70% train, 10%
validation (early stopping), 20% test. For xp1 (seeds 0-19 in every
condition) this is train 0-13 / val 14-15 / test 16-19.

Check 1  Prediction error on test seeds, by horizon, vs baselines.
Check 2  (xp1-shaped data only) GRU-implied one-step effect of W vs the
         randomized Q1 estimate (analyze.py).
Check K  (windowed attacks) Kill awareness: from the history up to the attack's
         first block, does the GRU predict that defending from now on keeps the
         backdoor out, and that not defending implants it?
Check 3  (xp1, --cf) Counterfactual reruns. The model is frozen once trained
         (weights hash in report.json); --cf only loads it, never retrains.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import analyze  # noqa: E402
from analyze import DEFENSES, INK, INK2, SERIES, X_COLS  # noqa: E402

Y_COLS = ["y_clean_acc", "y_atk"]
CF_PREFIX = 3
CUTS_PER_BATCH = 16
MAX_RIDGE_PAIRS = 1_000_000


def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Split by seed, stratified by attack: rank each attack's seeds, then rank % 20 in 0-13 train, 14-15 val,
    16-19 test (70/10/20).

    xp1: each attack has seeds 0-19, so rank = seed (train 0-13, val 14-15, test 16-19, as before).
    xp2: the attack is set by seed % 10, so a plain seed % 20 rule would leave the windowed attack out of
    the test set; ranking within each attack keeps every attack in every split. The attack label is used
    for stratification only, never as a model input.
    """
    split = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby("attack"):
        rank = {s: k for k, s in enumerate(sorted(g.seed.unique()))}
        r = g.seed.map(rank) % 20
        split[g.index] = np.where(r < 14, "train", np.where(r < 16, "val", "test"))
    return split


def horizons(n_blocks: int) -> list[int]:
    return [1, 5, n_blocks // 5, n_blocks // 2]  # xp1: 1, 5, 10, 25


# ----------------------------------------------------------------------------- tensors
class Data:
    """Per-run tensors. Row i is block i-1 (row 0 = initial state); n_rows = n_blocks + 1."""

    def __init__(self, df: pd.DataFrame, x_mean: np.ndarray | None = None, x_std: np.ndarray | None = None):
        df = df.sort_values(["run_id", "block"])
        runs = df.groupby("run_id", sort=True)
        sizes = runs.size()
        if sizes.nunique() != 1:
            raise ValueError(f"runs have different lengths: {sizes.value_counts().to_dict()}")
        meta = runs.first()[["seed", "attack", "defense"]]
        self.run_ids = meta.index.to_numpy()
        self.seed = meta.seed.to_numpy()
        self.attack = meta.attack.to_numpy()
        self.defense = meta.defense.to_numpy()
        R, n = len(meta), int(sizes.iloc[0])
        self.n_rows, self.n_blocks = n, n - 1
        X = df[X_COLS].to_numpy(float).reshape(R, n, len(X_COLS))
        if x_mean is None:
            x_mean, x_std = np.nanmean(X[:, 1:], axis=(0, 1)), np.nanstd(X[:, 1:], axis=(0, 1))
        self.x_mean, self.x_std = x_mean, x_std
        X = np.nan_to_num((X - x_mean) / x_std)  # row 0 (init) -> 0, flagged below
        W = np.nan_to_num(df.W.to_numpy(float).reshape(R, n))
        t = np.broadcast_to((np.arange(n) - 1) / self.n_blocks, (R, n))
        init = np.zeros((R, n)); init[:, 0] = 1
        D = np.stack([self.defense == d for d in DEFENSES], 1).astype(float)  # (R, 6)
        Dr = np.broadcast_to(D[:, None, :], (R, n, len(DEFENSES)))
        self.W = W
        self.enc = torch.tensor(np.concatenate([X, W[..., None], t[..., None], init[..., None], Dr], -1),
                                dtype=torch.float32)
        self.D = torch.tensor(D, dtype=torch.float32)
        Y = df[Y_COLS].to_numpy(float).reshape(R, n, 2)
        self.Y = torch.tensor(np.nan_to_num(Y), dtype=torch.float32)
        self.Ymask = torch.tensor(~np.isnan(Y), dtype=torch.float32)
        # Attack activity per block: an evaluation label only (never a model input).
        self.U = df.u_active.to_numpy(float).reshape(R, n) if "u_active" in df else None

    def dec_inputs(self, W: np.ndarray | torch.Tensor, runs: slice | None = None) -> torch.Tensor:
        """Decoder inputs for rows 1..n-1 given a schedule W (R, n): [W, t, defense one-hot].

        ``runs`` selects whose defense one-hot to use; by default W has one row per run of this Data (or this
        Data holds a single run, whose one-hot is shared by every row of W, e.g. several candidate plans).
        """
        W = torch.as_tensor(np.asarray(W, float), dtype=torch.float32)
        R, n = W.shape[0], self.n_rows
        D = self.D if runs is None else self.D[runs]
        t = ((torch.arange(1, n) - 1) / self.n_blocks).expand(R, -1)
        return torch.cat([W[:, 1:, None], t[..., None], D[:, None, :].expand(R, n - 1, -1)], -1)


# ----------------------------------------------------------------------------- model
class OutcomeGRU(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, hidden: int = 64):
        super().__init__()
        self.enc = nn.GRU(enc_dim, hidden, batch_first=True)
        self.dec = nn.GRU(dec_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def encode(self, enc_x: torch.Tensor) -> torch.Tensor:
        return self.enc(enc_x)[0]  # (R, rows, H): state after each row

    def decode(self, h0: torch.Tensor, dec_x: torch.Tensor) -> torch.Tensor:
        out, _ = self.dec(dec_x, h0.unsqueeze(0).contiguous())
        return torch.sigmoid(self.head(out))  # (N, steps, 2)


def cut_predictions(model: OutcomeGRU, enc_x: torch.Tensor, dec_x: torch.Tensor,
                    cuts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predictions from history rows 0..i for each cut i in ``cuts``, for every later row i+1+j.

    Returns preds (R, C, S, 2), a validity mask (C, S) for rows <= n-1, and idx (C, S): the dec_x / Y[:, 1:]
    index of the target row (row i+1+j has index i+j).
    """
    R, n_dec = enc_x.shape[0], dec_x.shape[1]  # n_dec = n_rows - 1
    cuts = cuts.to(enc_x.device)
    h = model.encode(enc_x)[:, cuts]  # (R, C, H)
    S = n_dec - int(cuts.min())
    j = torch.arange(S, device=enc_x.device)
    raw = cuts[:, None] + j[None, :]
    valid = raw < n_dec
    idx = raw.clamp(max=n_dec - 1)
    C = len(cuts)
    seq = dec_x[:, idx]  # (R, C, S, F)
    out = model.decode(h.reshape(R * C, -1), seq.reshape(R * C, S, -1)).reshape(R, C, S, 2)
    return out, valid, idx


def fit_gru(tr: Data, va: Data, device: torch.device, epochs: int, hidden: int, seed: int,
            cuts_per_batch: int = CUTS_PER_BATCH) -> tuple[OutcomeGRU, list]:
    """Masked MSE over (cut, future block); early stopping on validation seeds (patience 25)."""
    torch.manual_seed(seed)
    model = OutcomeGRU(tr.enc.shape[-1], tr.dec_inputs(tr.W).shape[-1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    tr_dec, va_dec = tr.dec_inputs(tr.W), va.dec_inputs(va.W)
    n_cuts = tr.n_rows - 1
    va_cuts = torch.arange(0, n_cuts, max(1, n_cuts // 50))  # fixed, evenly spaced: a stable validation loss
    best, best_state, history, patience = np.inf, None, [], 0
    gen = torch.Generator().manual_seed(seed)

    def loss_on(data: Data, dec: torch.Tensor, rows: torch.Tensor, cuts: torch.Tensor) -> torch.Tensor:
        pred, valid, idx = cut_predictions(model, data.enc[rows].to(device), dec[rows].to(device), cuts)
        Y = data.Y[rows][:, 1:].to(device)[:, idx]  # (B, C, S, 2): target row i+1+j
        M = data.Ymask[rows][:, 1:].to(device)[:, idx] * valid[None, :, :, None]
        return (((pred - Y) ** 2) * M).sum() / M.sum()

    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(tr.seed), generator=gen).split(32):
            cuts = (torch.arange(n_cuts) if n_cuts <= cuts_per_batch
                    else torch.randperm(n_cuts, generator=gen)[:cuts_per_batch].sort().values)
            opt.zero_grad()
            loss_on(tr, tr_dec, batch, cuts).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([loss_on(va, va_dec, rows, va_cuts).item() for rows in torch.arange(len(va.seed)).split(16)])
        history.append(float(v))
        if v < best - 1e-6:
            best, patience = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 25:
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_cuts(model: OutcomeGRU, data: Data, cuts: torch.Tensor, device, chunk: int = 16) -> np.ndarray:
    """(R, C, S, 2) predictions under each run's own schedule, computed in chunks of runs."""
    dec = data.dec_inputs(data.W)
    parts = [cut_predictions(model, data.enc[r].to(device), dec[r].to(device), cuts)[0].cpu().numpy()
             for r in torch.arange(len(data.seed)).split(chunk)]
    return np.concatenate(parts)


# ----------------------------------------------------------------------------- baselines
def ridge_features(data: Data, cut_rows: np.ndarray, target_rows: np.ndarray, run_idx: np.ndarray) -> np.ndarray:
    """Direct multi-horizon linear baseline: X at the cut, time, horizon, defense, and future-W summaries."""
    enc = data.enc.numpy()
    Xc = enc[run_idx, cut_rows]  # (N, F): X, W, t, init, defense one-hot
    D = data.D.numpy()[run_idx]
    W = data.W[run_idx]
    h = (target_rows - cut_rows)[:, None] / data.n_blocks
    w_target = W[np.arange(len(run_idx)), target_rows][:, None]
    csum = np.cumsum(W, axis=1)
    w_frac = ((csum[np.arange(len(run_idx)), target_rows] - csum[np.arange(len(run_idx)), cut_rows])
              / (target_rows - cut_rows))[:, None]
    wf = np.concatenate([w_target, w_frac], 1)
    inter = (wf[:, :, None] * D[:, None, :]).reshape(len(run_idx), -1)  # W summaries x defense
    return np.concatenate([Xc, h, h**2, wf, inter, wf * h], 1)


def pairs(R: int, n_rows: int, cuts: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (run, cut row, target row) with target > cut, for the given cut rows (default: all)."""
    cuts = np.arange(n_rows - 1) if cuts is None else np.asarray(cuts)
    run, cut, tgt = [], [], []
    for i in cuts:
        k = np.arange(i + 1, n_rows)
        run.append(np.repeat(np.arange(R), len(k))); cut.append(np.full(R * len(k), i)); tgt.append(np.tile(k, R))
    return np.concatenate(run), np.concatenate(cut), np.concatenate(tgt)


def fit_ridge(data: Data, alpha: float = 1.0, seed: int = 0) -> list[np.ndarray]:
    run, cut, tgt = pairs(len(data.seed), data.n_rows)
    if len(run) > MAX_RIDGE_PAIRS:  # xp2: subsample (xp1's 642k pairs are all used)
        keep = np.sort(np.random.default_rng(seed).choice(len(run), MAX_RIDGE_PAIRS, replace=False))
        run, cut, tgt = run[keep], cut[keep], tgt[keep]
    F = ridge_features(data, cut, tgt, run)
    F = np.concatenate([F, np.ones((len(F), 1))], 1)
    Y, M = data.Y.numpy()[run, tgt], data.Ymask.numpy()[run, tgt]
    coefs = []
    for c in range(2):
        keep = M[:, c] > 0
        A = F[keep]
        coefs.append(np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ Y[keep, c]))
    return coefs


def predict_ridge(coefs: list[np.ndarray], data: Data, run, cut, tgt) -> np.ndarray:
    F = ridge_features(data, cut, tgt, run)
    F = np.concatenate([F, np.ones((len(F), 1))], 1)
    return np.clip(np.stack([F @ c for c in coefs], 1), 0, 1)


# ----------------------------------------------------------------------------- check 1
def eval_cuts(data: Data) -> np.ndarray:
    n_cuts = data.n_rows - 1
    return np.arange(0, n_cuts, max(1, n_cuts // 50))  # xp1: every cut; xp2: every 5th


def check1(model: OutcomeGRU, ridge: list, train: Data, test: Data, device) -> pd.DataFrame:
    cuts = eval_cuts(test)
    pred = predict_cuts(model, test, torch.as_tensor(cuts), device)  # (R, C, S, 2)
    run, cut, tgt = pairs(len(test.seed), test.n_rows, cuts)
    pos = np.searchsorted(cuts, cut)
    gru = pred[run, pos, tgt - cut - 1]
    lin = predict_ridge(ridge, test, run, cut, tgt)
    # Defense-mean trajectory (schedule- and history-blind): mean over train runs with the same defense.
    Ytr, Mtr = train.Y.numpy(), train.Ymask.numpy()
    mean_traj = {d: np.nansum(Ytr[train.defense == d] * Mtr[train.defense == d], 0)
                 / Mtr[train.defense == d].sum(0).clip(1) for d in set(train.defense)}
    dmean = np.stack([mean_traj[test.defense[r]][k] for r, k in zip(run, tgt)])
    # Oracle persistence: true Y at the cut carried forward (uses unobservable Y; reference only).
    Y, M = test.Y.numpy(), test.Ymask.numpy()
    oracle = Y[run, cut]
    truth, mask = Y[run, tgt], M[run, tgt] * M[run, cut]
    h = tgt - cut
    rows = []
    for c, y in enumerate(Y_COLS):
        for name, p in [("GRU", gru), ("ridge (direct, W-aware)", lin), ("defense-mean trajectory", dmean),
                        ("oracle persistence (true Y_b)", oracle)]:
            for hz in ["all", *horizons(test.n_blocks)]:
                sel = mask[:, c] > 0
                if hz != "all":
                    sel &= h == hz
                if not sel.any():
                    continue
                err = p[sel, c] - truth[sel, c]
                rows.append({"outcome": y, "model": name, "horizon": hz, "rmse": np.sqrt(np.mean(err**2)),
                             "mae": np.mean(np.abs(err)), "n": int(sel.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- check 2 (xp1)
def implied_effects(model: OutcomeGRU, data: Data, device) -> np.ndarray:
    """(R, n_blocks, 2): predicted Y at block b with W_b=1 minus W_b=0, from history up to block b-1."""
    with torch.no_grad():
        h = model.encode(data.enc.to(device))[:, : data.n_rows - 1]  # history rows 0..i -> predicts row i+1
        R, C, H = h.shape
        out = []
        for w in (1.0, 0.0):
            Wf = np.full((R, data.n_rows), w)
            x = data.dec_inputs(Wf).to(device)  # step input for row i+1 is x[:, i]
            out.append(model.decode(h.reshape(R * C, H), x.reshape(R * C, 1, -1)).reshape(R, C, 2))
    return (out[0] - out[1]).cpu().numpy()


def check2(model: OutcomeGRU, df: pd.DataFrame, test: Data, device, out: Path, n_boot: int) -> pd.DataFrame:
    eff = implied_effects(model, test, device)
    rng = np.random.default_rng(0)
    rows = []
    for c, y in enumerate(Y_COLS):
        _, phase, _ = analyze.q1(df, y, n_boot, rng)  # randomized estimate, all 20 seeds
        for (a, d), g in phase.groupby(["attack", "defense"]):
            sel = (test.attack == a) & (test.defense == d)
            gru = eff[sel, :, c].mean(0).reshape(5, -1).mean(1)
            for _, r in g.iterrows():
                rows.append({"outcome": y, "attack": a, "defense": d, "phase": r.phase, "randomized": r.effect,
                             "lo": r.lo, "hi": r.hi, "gru": gru[int(r.phase) - 1]})
    t = pd.DataFrame(rows)
    t["inside_ci"] = (t.gru >= t.lo) & (t.gru <= t.hi)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, y in zip(axes, Y_COLS):
        s = t[t.outcome == y]
        lim = [min(s.lo.min(), s.gru.min()) * 100 - 1, max(s.hi.max(), s.gru.max()) * 100 + 1]
        ax.plot(lim, lim, color=INK2, linewidth=0.8)
        for d in DEFENSES:
            q = s[s.defense == d]
            ax.errorbar(q.gru * 100, q.randomized * 100, yerr=[(q.randomized - q.lo) * 100, (q.hi - q.randomized) * 100],
                        fmt="o", markersize=4, color=SERIES[d], elinewidth=0.8, label=d)
        r = np.corrcoef(s.gru, s.randomized)[0, 1]
        ax.set_title(f"{analyze.OUTCOMES[y]}: r = {r:.2f}, GRU inside 95% CI in {100 * s.inside_ci.mean():.0f}% "
                     f"of {len(s)} cells", color=INK, loc="left")
        ax.set_xlabel("GRU-implied effect, test seeds (pts per defended block)")
        ax.set_ylabel("randomized estimate, 95% CI (pts)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Check 2: one-step effect of W per (attack, defense, phase) — GRU vs randomized estimate",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "check2_effects.png", dpi=130)
    plt.close(fig)
    return t


def is_xp1_shaped(df: pd.DataFrame, data: Data) -> bool:
    return data.n_blocks == 50 and df.defense.nunique() > 1 and df.attack.nunique() > 1


# ----------------------------------------------------------------------------- check K (windowed attacks)
@torch.no_grad()
def check_kill(model: OutcomeGRU, data: Data, device, out: Path) -> pd.DataFrame:
    """For each test run with a windowed attack: history up to the attack's first block (detectable in X), then
    two plans: defend from the next block on, or never. Predicted backdoor success at the window end and at
    the run end, next to what the run's own schedule actually produced.

    The window is read from u_active, an evaluation label; it is never a model input.
    """
    if data.U is None:
        return pd.DataFrame()
    rows = []
    for r in range(len(data.seed)):
        if data.attack[r] != "backdoor_win" or data.Ymask[r, :, 1].sum() == 0:
            continue
        active = np.flatnonzero(data.U[r] > 0)  # rows (block + 1)
        if len(active) == 0:
            continue
        first, last = int(active[0]), int(active[-1])
        h = model.encode(data.enc[r:r + 1, : first + 1].to(device))[:, -1]  # state after the first poisoned block
        preds = {}
        for name, w in [("defend", 1.0), ("never", 0.0)]:
            W = data.W[r:r + 1].copy()
            W[:, first + 1:] = w
            dec = data.dec_inputs(W, runs=slice(r, r + 1))[:, first:]  # rows first+1 .. n-1
            preds[name] = model.decode(h.expand(1, -1), dec.to(device))[0, :, 1].cpu().numpy()
        end_w = last - first - 1  # index of the window's last row in the decoded sequence
        W_true = data.W[r]
        on_rows = np.flatnonzero(W_true[first:] > 0)
        delay = int(on_rows[0]) if len(on_rows) else -1  # blocks after the attack's first block
        covered = len(on_rows) > 0 and bool((W_true[first + delay: last + 1] > 0).all())
        rows.append({
            "run_id": data.run_ids[r], "seed": int(data.seed[r]), "window_blocks": (first - 1, last - 1),
            "actual_delay_blocks": delay, "actual_covered_to_end": covered,
            "actual_atk_window_end": float(data.Y[r, last, 1]), "actual_atk_final": float(data.Y[r, -1, 1]),
            "gru_atk_window_end_if_defend": float(preds["defend"][max(end_w, 0)]),
            "gru_atk_window_end_if_never": float(preds["never"][max(end_w, 0)]),
            "gru_atk_final_if_defend": float(preds["defend"][-1]), "gru_atk_final_if_never": float(preds["never"][-1]),
        })
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(out / "checkK_kill.csv", index=False)
    return t


# ----------------------------------------------------------------------------- check 3 (xp1)
def score_cf(model: OutcomeGRU, ridge: list, train: Data, cf_df: pd.DataFrame, xp1: pd.DataFrame,
             device, out: Path) -> dict:
    """Score predictions on true counterfactual runs. History = the run's own first CF_PREFIX blocks (observed)."""
    res: dict = {}
    # Reproduction check: cf 'random' runs must equal the xp1 runs with the same run_id.
    rand = cf_df[~cf_df.run_id.str.contains("__cf_")]
    if len(rand):
        merged = rand.merge(xp1, on=["run_id", "block"], suffixes=("", "_xp1"))
        cols = [c for c in [*X_COLS, *Y_COLS] if c in rand.columns]
        diffs = {c: float(np.nanmax(np.abs(merged[c] - merged[f"{c}_xp1"]))) for c in cols}
        res["reproduction"] = {"runs": int(rand.run_id.nunique()), "max_abs_diff": max(diffs.values()),
                               "per_column": diffs}
    data = Data(cf_df[cf_df.run_id.str.contains("__cf_")], train.x_mean, train.x_std)
    n = data.n_rows
    sched = pd.Series(data.run_ids).str.extract(r"__cf_(\w+?)_p\d+__")[0].to_numpy()
    cut = CF_PREFIX  # rows 0..3 = blocks -1..2 observed
    with torch.no_grad():
        h = model.encode(data.enc[:, : cut + 1].to(device))[:, -1]
        pred = model.decode(h, data.dec_inputs(data.W)[:, cut:].to(device)).cpu().numpy()  # rows cut+1..n-1
    run, tgt = np.repeat(np.arange(len(data.seed)), n - 1 - cut), np.tile(np.arange(cut + 1, n), len(data.seed))
    lin = predict_ridge(ridge, data, run, np.full(len(run), cut), tgt).reshape(len(data.seed), -1, 2)
    Y, M = data.Y.numpy()[:, cut + 1:], data.Ymask.numpy()[:, cut + 1:]
    rows = []
    for r in range(len(data.seed)):
        for c, y in enumerate(Y_COLS):
            if M[r, -1, c] == 0:
                continue
            rows.append({"run_id": data.run_ids[r], "attack": data.attack[r], "defense": data.defense[r],
                         "seed": data.seed[r], "schedule": sched[r], "outcome": y,
                         "true_final": Y[r, -1, c], "gru_final": pred[r, -1, c], "ridge_final": lin[r, -1, c],
                         "gru_rmse_traj": float(np.sqrt(np.mean((pred[r, :, c] - Y[r, :, c]) ** 2))),
                         "ridge_rmse_traj": float(np.sqrt(np.mean((lin[r, :, c] - Y[r, :, c]) ** 2)))})
    t = pd.DataFrame(rows)
    t.to_csv(out / "check3_counterfactual.csv", index=False)
    # Ranking: per (seed, attack, defense, outcome), do predicted finals order the 5 schedules like the truth?
    rank = []
    for key, g in t.groupby(["seed", "attack", "defense", "outcome"]):
        if g.true_final.nunique() > 1:
            rank.append({"outcome": key[3], "attack": key[1], "defense": key[2],
                         "gru_spearman": g.true_final.rank().corr(g.gru_final.rank()),
                         "ridge_spearman": g.true_final.rank().corr(g.ridge_final.rank())})
    res["ranking"] = pd.DataFrame(rank).groupby(["outcome", "attack"])[["gru_spearman", "ridge_spearman"]].mean() \
        .round(3).reset_index().to_dict("records") if rank else []
    res["final_error"] = t.groupby(["outcome", "schedule"]).apply(
        lambda g: pd.Series({"true": g.true_final.mean(), "gru": g.gru_final.mean(), "ridge": g.ridge_final.mean(),
                             "gru_mae": (g.gru_final - g.true_final).abs().mean(),
                             "ridge_mae": (g.ridge_final - g.true_final).abs().mean()})).round(3) \
        .reset_index().to_dict("records")
    return res


# ----------------------------------------------------------------------------- support diagnostic
def support(df: pd.DataFrame) -> dict:
    """How long the defended stretches in the training schedules are (the GRU's training support)."""
    W = df[df.block >= 0].sort_values(["run_id", "block"]).groupby("run_id").W.apply(lambda s: [int(v) for v in s])
    n_blocks = len(W.iloc[0])
    longest_on = W.apply(lambda w: max(len(m) for m in re.findall("1+", "".join(map(str, w))) or [""]))
    first_off = W.apply(lambda w: w.index(0) if 0 in w else len(w))
    return {"runs": int(len(W)), "n_blocks": n_blocks, "longest_defended_streak_max": int(longest_on.max()),
            "longest_defended_streak_median": float(longest_on.median()),
            "first_undefended_block_median": float(first_off.median()),
            "runs_with_defended_streak_over_20pct": int((longest_on >= n_blocks // 5).sum())}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("results/xp1/results_collect/runs/all_blocks.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/xp1/gru"))
    ap.add_argument("--cf", type=Path, help="all_blocks.csv of the counterfactual runs (check 3)")
    ap.add_argument("--retrain", action="store_true", help="overwrite the frozen gru.pt (never with --cf)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--cuts-per-batch", type=int, default=CUTS_PER_BATCH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    if args.cf and args.retrain:
        raise SystemExit("--retrain with --cf would score check 3 with a model fitted after the runs existed.")
    sys.stdout.reconfigure(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = analyze.load(args.runs)
    split = assign_splits(df)
    train = Data(df[split == "train"])
    val = Data(df[split == "val"], train.x_mean, train.x_std)
    test = Data(df[split == "test"], train.x_mean, train.x_std)
    print(f"runs: train {len(train.seed)}, val {len(val.seed)}, test {len(test.seed)}; "
          f"{train.n_blocks} blocks per run; device {device}", flush=True)

    weights = args.out / "gru.pt"
    model = OutcomeGRU(train.enc.shape[-1], train.dec_inputs(train.W).shape[-1], args.hidden).to(device)
    if weights.exists() and not args.retrain:
        # Frozen model (check 3 must use the model trained BEFORE the counterfactual runs existed).
        model.load_state_dict(torch.load(weights, map_location=device))
        model.eval()
        history = json.loads((args.out / "report.json").read_text())["val_history"]
    elif args.cf:
        raise SystemExit(f"No frozen model at {weights}: train it first (without --cf).")
    else:
        model, history = fit_gru(train, val, device, args.epochs, args.hidden, args.seed, args.cuts_per_batch)
        model.eval()
        torch.save(model.state_dict(), weights)
    weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()[:16]
    print(f"model: {len(history)} epochs, best val MSE {min(history):.5f}, weights sha256 {weights_sha}", flush=True)
    ridge = fit_ridge(train)

    c1 = check1(model, ridge, train, test, device)
    c1.to_csv(args.out / "check1_prediction.csv", index=False)
    report = {"weights_sha256": weights_sha, "n_blocks": train.n_blocks, "support": support(df),
              "epochs": len(history), "best_val_mse": min(history), "val_history": history}
    c2 = check2(model, df, test, device, args.out, args.boot) if is_xp1_shaped(df, test) else None
    if c2 is not None:
        c2.to_csv(args.out / "check2_effects.csv", index=False)
    ck = check_kill(model, test, device, args.out)
    if args.cf:
        report["check3"] = score_cf(model, ridge, train, analyze.load(args.cf), df, device, args.out)
    (args.out / ("report_cf.json" if args.cf else "report.json")).write_text(json.dumps(report, indent=2, default=float))

    print("\n## Check 1: test-seed prediction error (RMSE, accuracy units)")
    print(c1.pivot_table(index=["outcome", "model"], columns="horizon", values="rmse", sort=False).round(4).to_string())
    if c2 is not None:
        print("\n## Check 2: GRU-implied one-step effect vs randomized estimate")
        for y, s in c2.groupby("outcome"):
            print(f"{y}: corr {np.corrcoef(s.gru, s.randomized)[0, 1]:.3f}, inside 95% CI "
                  f"{100 * s.inside_ci.mean():.0f}% of {len(s)} cells")
    if len(ck):
        killed = (ck.actual_atk_window_end < 0.1)
        print(f"\n## Check K: kill awareness on {len(ck)} windowed test runs")
        print(f"predicted backdoor at window end if defending from the attack's 2nd block: "
              f"mean {ck.gru_atk_window_end_if_defend.mean():.3f}; if never defending: "
              f"mean {ck.gru_atk_window_end_if_never.mean():.3f}")
        print(f"actual: {int(killed.sum())} runs killed (backdoor < 0.1 at window end), {int((~killed).sum())} not")
    print("\n## Training support")
    print(json.dumps(report["support"], indent=2))
    if "check3" in report:
        print("\n## Check 3")
        print(json.dumps(report["check3"], indent=2, default=float))


if __name__ == "__main__":
    main()
