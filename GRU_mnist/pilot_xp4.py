"""Pilot for xp4: which signals reveal stealthy backdoors, defended or not, once training-time drift is removed?

    python pilot_xp4.py                 # 48 runs, then the detectability tables -> results/pilot_xp4
    python pilot_xp4.py --report-only

Runs: {1/8, 2/8, 8/8 poisoned, no attack} x {always on, always off, probe every 3 blocks} x 4 seeds, windowed
backdoor + Krum, 250 blocks of 2 rounds (the xp3 setup), with the v1 signals and the new v2 signals (block mean
and block max of each).

Detectability, per signal: AUC of 'attack active' vs 'no attack', comparing like with like:
  * same defense state: blocks of always-on runs (on), probe blocks of probe runs (probe), undefended blocks of
    always-off and probe runs (off);
  * same point in training: each value becomes a z-score against no-attack blocks of the same state in the same
    25-block time bin; the score is |z| (an attack may push a signal either way).
Positives: the first 10 blocks of the attack (inside the erasure window, when defending still kills it) and the
whole window. Negatives: the same state's blocks of no-attack runs. 'combined' = logistic regression on all |z|
of one signal set, AUC by leave-one-seed-out cross-validation.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import poison_testbed  # noqa: F401  (CUBLAS_WORKSPACE_CONFIG)
from poison_testbed.runner import RunConfig, simulate
from poison_testbed.signals import X2_COLUMNS, X_COLUMNS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/pilot_xp4"
XP1_C, SIGMA, BLOCK_LEN, N_BLOCKS = 3.0552712798118593, 0.01, 2, 250
CONDITIONS = {"pf1/8": ("backdoor_win", 0.125), "pf2/8": ("backdoor_win", 0.25), "pf8/8": ("backdoor_win", 1.0),
              "none": ("none", 1.0)}
SCHEDULES = ["on", "off", "probe_P3_L1_o0"]
SEEDS = {4000: (10, 69), 4001: (40, 99), 4002: (100, 159), 4003: (25, 84)}  # seed -> attack window (blocks)
V1 = [*X_COLUMNS, "x_val_acc"]
V2 = list(X2_COLUMNS)
EARLY, TIME_BIN = 10, 25


def config(cond: str, schedule: str, seed: int) -> RunConfig:
    attack, pf = CONDITIONS[cond]
    return RunConfig(attack=attack, defense="Krum", seed=seed, C=XP1_C, sigma=SIGMA, schedule=schedule,
                     block_len=BLOCK_LEN, n_blocks=N_BLOCKS, window=SEEDS[seed], poison_frac=pf,
                     always_measure_backdoor=True, signals="v2")


def run_one(cond: str, schedule: str, seed: int) -> tuple[str, float]:
    run_id = f"{cond.replace('/', 'of')}__{schedule}__s{seed}"
    out = OUT / "runs" / run_id
    if (out / "blocks.csv").exists():
        return run_id, json.loads((out / "config.json").read_text())["wall_clock_s"]
    cfg = config(cond, schedule, seed)
    started = time.time()
    df = simulate(cfg)
    wall = time.time() - started
    df.insert(0, "condition", cond)
    df["run_id"] = run_id
    df["window_start"] = cfg.window[0]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "blocks.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "config": cfg.__dict__, "config_hash": cfg.hash(), "wall_clock_s": round(wall, 1),
        "device_name": torch.cuda.get_device_name(0) if cfg.device == "cuda" else cfg.device}, indent=2, default=str))
    return run_id, wall


# ----------------------------------------------------------------------------- analysis
def state_of(df: pd.DataFrame) -> pd.Series:
    sched = df.run_id.str.extract(r"__(on|off|probe)")[0]
    return np.where(df.W == 0, "off", np.where(sched == "on", "on", "probe"))


def zscores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """|z| of each signal against no-attack blocks of the same defense state and time bin."""
    z = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    ref_mask = df.condition == "none"
    for (state, tb), g in df.groupby(["state", "tbin"]):
        ref = g[ref_mask[g.index]]
        mu, sd = ref[cols].mean(), ref[cols].std().replace(0, np.nan)
        z.loc[g.index] = ((g[cols] - mu) / sd).abs().to_numpy()
    return z.fillna(0.0)


def detectability(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = V1 + V2
    z = zscores(df, cols)
    since = df.block - df.window_start
    rows, combined = [], []
    for pf in ["pf1/8", "pf2/8", "pf8/8"]:
        for state in ["on", "probe", "off"]:
            neg = (df.condition == "none") & (df.state == state)
            for phase, lo, hi in [("first 10 blocks", 0, EARLY), ("whole window", 0, 10_000)]:
                pos = (df.condition == pf) & (df.state == state) & (df.u_active == 1) & (since >= lo) & (since < hi)
                sel = pos | neg
                y = pos[sel].astype(int).to_numpy()
                if y.sum() < 3:
                    continue
                for c in cols:
                    rows.append({"attack": pf, "state": state, "phase": phase, "signal": c, "set": "v1" if c in V1 else "v2",
                                 "auc": roc_auc_score(y, z.loc[sel, c]), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())})
                groups = df.seed[sel].to_numpy()
                for name, feats in [("v1 combined", V1), ("v2 combined", V2), ("v1+v2 combined", cols)]:
                    pred = np.zeros(len(y))
                    for s in np.unique(groups):  # leave one seed out
                        tr, te = groups != s, groups == s
                        if len(np.unique(y[tr])) < 2:
                            pred[te] = 0.5
                            continue
                        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
                        clf.fit(z.loc[sel, feats].to_numpy()[tr], y[tr])
                        pred[te] = clf.predict_proba(z.loc[sel, feats].to_numpy()[te])[:, 1]
                    combined.append({"attack": pf, "state": state, "phase": phase, "detector": name,
                                     "auc": roc_auc_score(y, pred), "n_pos": int(y.sum())})
    return pd.DataFrame(rows), pd.DataFrame(combined)


def report() -> None:
    df = pd.concat([pd.read_csv(f) for f in sorted((OUT / "runs").glob("*/blocks.csv"))], ignore_index=True)
    df = df[df.block >= 0].reset_index(drop=True)
    df["state"] = state_of(df)
    df["tbin"] = df.block // TIME_BIN
    single, comb = detectability(df)
    single.to_csv(OUT / "auc_by_signal.csv", index=False)
    comb.to_csv(OUT / "auc_combined.csv", index=False)
    pd.set_option("display.width", 220)
    print("## Best single signal per set (AUC, time-matched, same defense state)")
    best = (single.sort_values("auc", ascending=False).groupby(["attack", "state", "phase", "set"]).head(1)
            .assign(best=lambda t: t.signal.str.replace("x2_", "") + " " + t.auc.round(2).astype(str))
            .pivot_table(index=["attack", "phase", "state"], columns="set", values="best", aggfunc="first"))
    print(best.to_string())
    print("\n## Combined detectors (logistic on |z|, leave-one-seed-out AUC)")
    print(comb.pivot_table(index=["attack", "phase", "state"], columns="detector", values="auc").round(2).to_string())
    print("\n## Signals with AUC >= 0.8 in the first 10 blocks (count of attack x state cells, of 9)")
    early = single[single.phase == "first 10 blocks"]
    print(early[early.auc >= 0.8].groupby("signal").size().sort_values(ascending=False).head(15).to_string())
    # Backdoor outcome per condition x schedule (does a probe-every-3 schedule kill anything by itself?)
    last = df[df.block == df.block.max()]
    print("\n## Final backdoor success (mean over seeds)")
    print(last.assign(s=last.run_id.str.extract(r"__(on|off|probe)")[0]).pivot_table(
        index="condition", columns="s", values="y_atk").round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        jobs = [(c, s, seed) for c in CONDITIONS for s in SCHEDULES for seed in SEEDS]
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
            futs = [pool.submit(run_one, *j) for j in jobs]
            for k, fut in enumerate(as_completed(futs), 1):
                run_id, wall = fut.result()
                print(f"[{k}/{len(jobs)}] {run_id}  {wall:.0f}s  (elapsed {time.time() - started:.0f}s)", flush=True)
    report()


if __name__ == "__main__":
    main()
