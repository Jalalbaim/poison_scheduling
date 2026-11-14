"""GRU outcome model: predict future outcomes under a candidate defense schedule.

    python gru_experiment.py                       # train on xp1, checks 1-2, pre-register check-3 predictions
    python gru_experiment.py --cf runs_cf/all_blocks.csv   # + check 3 against true counterfactual runs

Model. An encoder GRU reads the observable history up to block b: the signals X
(incl. validation accuracy), the defense flag W, time and the defense type. A
decoder GRU, started from the encoder state, reads the candidate future
schedule W[b+1:] and outputs y_clean_acc and y_atk for every future block.
Inputs never include U (attack type / activity), Y_clean or Y_atk: the attack
must be inferred from X. Future X is not predicted (no feedback of errors).

Split by seed: train 0-13, validation 14-15 (early stopping), test 16-19.

Check 1  Prediction error on test seeds, by horizon, vs baselines.
Check 2  GRU-implied one-step effect of W (predict block b with W_b = 1 vs 0),
         averaged over test runs, vs the randomized Q1 estimate (analyze.py).
Check 3  Counterfactual schedules for test seeds (defended 3-block prefix, then
         on / off / on-until-20 / off-until-20 / alternate). The GRU reads the
         run's own observed prefix and predicts the rest. The model is frozen
         now (weights hash in report.json); --cf only loads it, never retrains.
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

TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS = range(0, 14), range(14, 16), range(16, 20)
Y_COLS = ["y_clean_acc", "y_atk"]
N_ROWS = 51  # block -1 .. 49
CF_PREFIX = 3
HORIZONS = [1, 5, 10, 25]


# ----------------------------------------------------------------------------- tensors
class Data:
    """Per-run tensors. Row i is block i-1 (row 0 = initial state)."""

    def __init__(self, df: pd.DataFrame, x_mean: np.ndarray | None = None, x_std: np.ndarray | None = None):
        df = df.sort_values(["run_id", "block"])
        runs = df.groupby("run_id", sort=True)
        meta = runs.first()[["seed", "attack", "defense"]]
        self.run_ids = meta.index.to_numpy()
        self.seed = meta.seed.to_numpy()
        self.attack = meta.attack.to_numpy()
        self.defense = meta.defense.to_numpy()
        R = len(meta)
        X = df[X_COLS].to_numpy(float).reshape(R, N_ROWS, len(X_COLS))
        if x_mean is None:
            x_mean, x_std = np.nanmean(X[:, 1:], axis=(0, 1)), np.nanstd(X[:, 1:], axis=(0, 1))
        self.x_mean, self.x_std = x_mean, x_std
        X = np.nan_to_num((X - x_mean) / x_std)  # row 0 (init) -> 0, flagged below
        W = np.nan_to_num(df.W.to_numpy(float).reshape(R, N_ROWS))
        t = np.broadcast_to((np.arange(N_ROWS) - 1) / 50.0, (R, N_ROWS))
        init = np.zeros((R, N_ROWS)); init[:, 0] = 1
        D = np.stack([self.defense == d for d in DEFENSES], 1).astype(float)  # (R, 6)
        Dr = np.broadcast_to(D[:, None, :], (R, N_ROWS, len(DEFENSES)))
        self.W = W
        self.enc = torch.tensor(np.concatenate([X, W[..., None], t[..., None], init[..., None], Dr], -1),
                                dtype=torch.float32)
        self.D = torch.tensor(D, dtype=torch.float32)
        Y = df[Y_COLS].to_numpy(float).reshape(R, N_ROWS, 2)
        self.Y = torch.tensor(np.nan_to_num(Y), dtype=torch.float32)
        self.Ymask = torch.tensor(~np.isnan(Y), dtype=torch.float32)

    def dec_inputs(self, W: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Decoder inputs for rows 1..50 given a schedule W (R, 51): [W, t, defense one-hot]."""
        W = torch.as_tensor(np.asarray(W, float), dtype=torch.float32)
        R = W.shape[0]
        t = ((torch.arange(1, N_ROWS) - 1) / 50.0).expand(R, -1)
        return torch.cat([W[:, 1:, None], t[..., None], self.D[:, None, :].expand(R, N_ROWS - 1, -1)], -1)


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


def all_cuts(model: OutcomeGRU, enc_x: torch.Tensor, dec_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Predictions for every cut i = 0..49 (history rows 0..i) and step j (row i+1+j).

    Returns preds (R, 50 cuts, 50 steps, 2) and a validity mask (50, 50) for row i+1+j <= 50.
    """
    R, C = enc_x.shape[0], N_ROWS - 1
    h = model.encode(enc_x)[:, :C]  # (R, C, H)
    j = torch.arange(C)
    idx = (j[:, None] + j[None, :]).clamp(max=C - 1)  # dec_x index of row i+1+j is i+j
    valid = (j[:, None] + j[None, :]) < C
    seq = dec_x[:, idx]  # (R, C, C, F)
    out = model.decode(h.reshape(R * C, -1), seq.reshape(R * C, C, -1)).reshape(R, C, C, 2)
    return out, valid.to(enc_x.device), idx


def fit_gru(tr: Data, va: Data, device: torch.device, epochs: int, hidden: int, seed: int) -> tuple[OutcomeGRU, list]:
    """Masked MSE over every (cut, future block); early stopping on validation seeds (patience 25)."""
    torch.manual_seed(seed)
    model = OutcomeGRU(tr.enc.shape[-1], tr.dec_inputs(tr.W).shape[-1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    tr_dec, va_dec = tr.dec_inputs(tr.W), va.dec_inputs(va.W)
    best, best_state, history, patience = np.inf, None, [], 0
    gen = torch.Generator().manual_seed(seed)

    def loss_on(data: Data, dec: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        pred, valid, idx = all_cuts(model, data.enc[rows].to(device), dec[rows].to(device))
        Y = data.Y[rows][:, 1:][:, idx].to(device)  # (B, cut, step, 2): target row i+1+j
        M = data.Ymask[rows][:, 1:][:, idx].to(device) * valid[None, :, :, None]
        return (((pred - Y) ** 2) * M).sum() / M.sum()

    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(tr.seed), generator=gen).split(32):
            opt.zero_grad()
            loss_on(tr, tr_dec, batch).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_on(va, va_dec, torch.arange(len(va.seed))).item()
        history.append(v)
        if v < best - 1e-6:
            best, patience = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 25:
                break
    model.load_state_dict(best_state)
    return model, history


# ----------------------------------------------------------------------------- baselines
def ridge_features(data: Data, cut_rows: np.ndarray, target_rows: np.ndarray, run_idx: np.ndarray) -> np.ndarray:
    """Direct multi-horizon linear baseline: X at the cut, time, horizon, defense, and future-W summaries."""
    enc = data.enc.numpy()
    Xc = enc[run_idx, cut_rows]  # (N, F): X, W, t, init, defense one-hot
    D = data.D.numpy()[run_idx]
    W = data.W[run_idx]
    h = (target_rows - cut_rows)[:, None] / 50.0
    w_target = W[np.arange(len(run_idx)), target_rows][:, None]
    csum = np.cumsum(W, axis=1)
    w_frac = ((csum[np.arange(len(run_idx)), target_rows] - csum[np.arange(len(run_idx)), cut_rows])
              / (target_rows - cut_rows))[:, None]
    wf = np.concatenate([w_target, w_frac], 1)
    inter = (wf[:, :, None] * D[:, None, :]).reshape(len(run_idx), -1)  # W summaries x defense
    return np.concatenate([Xc, h, h**2, wf, inter, wf * h], 1)


def pairs(R: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run, cut, tgt = [], [], []
    for i in range(N_ROWS - 1):
        for k in range(i + 1, N_ROWS):
            run.append(np.arange(R)); cut.append(np.full(R, i)); tgt.append(np.full(R, k))
    return np.concatenate(run), np.concatenate(cut), np.concatenate(tgt)


def fit_ridge(data: Data, alpha: float = 1.0) -> list[np.ndarray]:
    run, cut, tgt = pairs(len(data.seed))
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
def check1(model: OutcomeGRU, ridge: list, train: Data, test: Data, device) -> pd.DataFrame:
    with torch.no_grad():
        pred, valid, idx = all_cuts(model, test.enc.to(device), test.dec_inputs(test.W).to(device))
    pred = pred.cpu().numpy()
    run, cut, tgt = pairs(len(test.seed))
    gru = pred[run, cut, tgt - cut - 1]
    lin = predict_ridge(ridge, test, run, cut, tgt)
    # Defense-mean trajectory (schedule- and history-blind): mean over train runs with the same defense.
    Ytr, Mtr = train.Y.numpy(), train.Ymask.numpy()
    mean_traj = {d: np.nansum(Ytr[train.defense == d] * Mtr[train.defense == d], 0)
                 / Mtr[train.defense == d].sum(0).clip(1) for d in DEFENSES}
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
            for hz in ["all", *HORIZONS]:
                sel = mask[:, c] > 0
                if hz != "all":
                    sel &= h == hz
                err = p[sel, c] - truth[sel, c]
                rows.append({"outcome": y, "model": name, "horizon": hz, "rmse": np.sqrt(np.mean(err**2)),
                             "mae": np.mean(np.abs(err)), "n": int(sel.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- check 2
def implied_effects(model: OutcomeGRU, data: Data, device) -> np.ndarray:
    """(R, 50, 2): predicted Y at block b with W_b=1 minus W_b=0, from history up to block b-1."""
    with torch.no_grad():
        h = model.encode(data.enc.to(device))[:, : N_ROWS - 1]  # history rows 0..i -> predicts row i+1
        R, C, H = h.shape
        out = []
        for w in (1.0, 0.0):
            Wf = np.full((R, N_ROWS), w)
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
            gru = eff[sel, :, c].mean(0).reshape(5, 10).mean(1)
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


# ----------------------------------------------------------------------------- check 3
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
    sched = pd.Series(data.run_ids).str.extract(r"__cf_(\w+?)_p\d+__")[0].to_numpy()
    cut = CF_PREFIX  # rows 0..3 = blocks -1..2 observed
    with torch.no_grad():
        h = model.encode(data.enc[:, : cut + 1].to(device))[:, -1]
        pred = model.decode(h, data.dec_inputs(data.W)[:, cut:].to(device)).cpu().numpy()  # rows cut+1..50
    run, tgt = np.repeat(np.arange(len(data.seed)), N_ROWS - 1 - cut), np.tile(np.arange(cut + 1, N_ROWS), len(data.seed))
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
    """How often xp1 schedules contain long defended / undefended runs (the GRU's training support)."""
    W = df[df.block >= 0].sort_values(["run_id", "block"]).groupby("run_id").W.apply(lambda s: [int(v) for v in s])
    longest_on = W.apply(lambda w: max(len(m) for m in re.findall("1+", "".join(map(str, w))) or [""]))
    first_off = W.apply(lambda w: w.index(0) if 0 in w else len(w))
    return {"runs": int(len(W)), "longest_defended_streak_max": int(longest_on.max()),
            "longest_defended_streak_median": float(longest_on.median()),
            "first_undefended_block_median": float(first_off.median()),
            "runs_defended_through_block_2": int((first_off >= 3).sum()),
            "runs_defended_through_block_19": int((first_off >= 20).sum())}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("results/xp1/results_collect/runs/all_blocks.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/xp1/gru"))
    ap.add_argument("--cf", type=Path, help="all_blocks.csv of the counterfactual runs (check 3)")
    ap.add_argument("--retrain", action="store_true", help="overwrite the frozen gru.pt (never with --cf)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    if args.cf and args.retrain:
        raise SystemExit("--retrain with --cf would score check 3 with a model fitted after the runs existed.")
    sys.stdout.reconfigure(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = analyze.load(args.runs)
    train = Data(df[df.seed.isin(TRAIN_SEEDS)])
    val = Data(df[df.seed.isin(VAL_SEEDS)], train.x_mean, train.x_std)
    test = Data(df[df.seed.isin(TEST_SEEDS)], train.x_mean, train.x_std)
    print(f"runs: train {len(train.seed)}, val {len(val.seed)}, test {len(test.seed)}  device {device}", flush=True)

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
        model, history = fit_gru(train, val, device, args.epochs, args.hidden, args.seed)
        torch.save(model.state_dict(), weights)
    weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()[:16]
    print(f"model: {len(history)} epochs, best val MSE {min(history):.5f}, weights sha256 {weights_sha}", flush=True)
    ridge = fit_ridge(train)

    c1 = check1(model, ridge, train, test, device)
    c1.to_csv(args.out / "check1_prediction.csv", index=False)
    c2 = check2(model, df, test, device, args.out, args.boot)
    c2.to_csv(args.out / "check2_effects.csv", index=False)
    report = {"weights_sha256": weights_sha, "support": support(df), "epochs": len(history),
              "best_val_mse": min(history), "val_history": history}
    if args.cf:
        report["check3"] = score_cf(model, ridge, train, analyze.load(args.cf), df, device, args.out)
    (args.out / ("report_cf.json" if args.cf else "report.json")).write_text(json.dumps(report, indent=2, default=float))

    print("\n## Check 1: test-seed prediction error (RMSE, accuracy units)")
    print(c1.pivot_table(index=["outcome", "model"], columns="horizon", values="rmse", sort=False).round(4).to_string())
    print("\n## Check 2: GRU-implied one-step effect vs randomized estimate")
    for y, s in c2.groupby("outcome"):
        print(f"{y}: corr {np.corrcoef(s.gru, s.randomized)[0, 1]:.3f}, inside 95% CI {100 * s.inside_ci.mean():.0f}% "
              f"of {len(s)} cells")
    print("\n## Training support (xp1 schedules)")
    print(json.dumps(report["support"], indent=2))
    if "check3" in report:
        print("\n## Check 3")
        print(json.dumps(report["check3"], indent=2, default=float))


if __name__ == "__main__":
    main()
