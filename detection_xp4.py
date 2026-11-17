"""Item 1 (xp5 prep): faster onset detection, evaluated offline on the xp4 logs (no new runs).

    python detection_xp4.py      -> results/xp4/detection/

Every signal is first standardised against no-attack TRAINING blocks of the same defense state (W) and 25-block
time bin: z = (x - mean) / sd. Detectors turn z into a score per block; an alarm is a block whose score exceeds a
threshold. Every detector's threshold is calibrated on no-attack training runs to the same budget: on average
ALARMS_PER_RUN alarm onsets (rising edges) per run. False alarms are then counted on held-out (val + test)
no-attack runs.

Detection: attacked runs (any split: the threshold never sees them) that are undefended for the first 10 blocks
of the attack, so the defense state does not change the signal at onset. Delay d = blocks from the attack's first
block to the first alarm; the defense then starts at block d+1. "In time" = d+1 within the erasure window measured
in measure_erasure.py: 1 block (8/8), 5 blocks (2/8), 10 blocks (1/8); 1 block = 2 SGD rounds.

Detectors:
    v1                  old signal alone (x_norm_std)
    v2                  new signal alone (x2_outlier_proj_max, block max), the deploy_xp4 alarm
    v1 OR v2            max of the two z-scores
    v2 sum of 3 blocks  sum of v2's z over the last 3 blocks
    v2 CUSUM            S_b = max(0, S_{b-1} + z_b - K), alarm when S_b > threshold (accumulates weak evidence)
    multi-signal CUSUM  CUSUM on the mean z of 4 memory/outlier signals
    OR CUSUM            CUSUM on max(z_v1, z_v2)
Per-round decisions (every SGD step instead of every 2) cannot be evaluated from block logs: they need new runs.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import INK, INK2, SERIES  # noqa: E402
from gru_experiment import assign_splits  # noqa: E402

RUNS = Path("results/xp4/runs_xp4/all_blocks.csv")
OUT = Path("results/xp4/detection")
BIN, ALARMS_PER_RUN, K = 25, 1.0, 0.5
ERASURE = {0.125: 10, 0.25: 5, 1.0: 1}  # blocks
LABEL = {0.125: "1/8", 0.25: "2/8", 1.0: "8/8"}
V1, V2 = "x_norm_std", "x2_outlier_proj_max_max"
MULTI = ["x2_outlier_proj_max_max", "x2_outlier_consistency_max", "x2_krum_top_ratio_max", "x2_sv_top_share_max"]


def standardise(df: pd.DataFrame, cols: list[str], ref_mask: pd.Series) -> pd.DataFrame:
    ref = df[ref_mask]
    stats = ref.groupby([ref.W.astype(int), ref.block // BIN])[cols]
    mu, sd = stats.mean(), stats.std()
    idx = pd.MultiIndex.from_arrays([df.W.astype(int), df.block // BIN])
    return pd.DataFrame((df[cols].to_numpy() - mu.reindex(idx).to_numpy()) / sd.reindex(idx).to_numpy(),
                        index=df.index, columns=cols)


def cusum(z: pd.Series, run: pd.Series) -> pd.Series:
    out = np.zeros(len(z))
    zv = z.to_numpy()
    for idx in z.groupby(run).indices.values():
        s = 0.0
        for i in idx:
            s = max(0.0, s + zv[i] - K)
            out[i] = s
    return pd.Series(out, index=z.index)


def rolling_sum(z: pd.Series, run: pd.Series, k: int) -> pd.Series:
    return z.groupby(run).transform(lambda s: s.rolling(k, min_periods=1).sum())


def onsets(alarm: pd.Series, run: pd.Series) -> pd.Series:
    prev = alarm.groupby(run).shift(1, fill_value=False)
    return (alarm & ~prev).groupby(run).sum()


def calibrate(score: pd.Series, run: pd.Series, mask: pd.Series) -> float:
    """Lowest threshold with at most ALARMS_PER_RUN alarm onsets per no-attack training run (bisection)."""
    s, r = score[mask], run[mask]
    lo, hi = 0.0, 1.0
    for _ in range(40):
        q = (lo + hi) / 2
        if onsets(s > s.quantile(q), r).mean() > ALARMS_PER_RUN:
            lo = q
        else:
            hi = q
    return float(s.quantile(hi))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUNS)
    df["split"] = assign_splits(df)
    df = df[df.block >= 0].sort_values(["run_id", "block"]).reset_index(drop=True)
    train_none = (df.split == "train") & (df.attack == "none")
    z = standardise(df, list(dict.fromkeys([V1, V2, *MULTI])), train_none)
    run = df.run_id
    zor = pd.Series(np.maximum(z[V1], z[V2]), index=df.index)
    scores = {
        "v1 (old signal)": z[V1],
        "v2 (new signal)": z[V2],
        "v1 OR v2": zor,
        "v2 sum of 3 blocks": rolling_sum(z[V2], run, 3),
        "v2 CUSUM": cusum(z[V2], run),
        "multi-signal CUSUM": cusum(z[MULTI].mean(axis=1), run),
        "OR CUSUM": cusum(zor, run),
    }
    start = df[df.u_active == 1].groupby("run_id").block.min()
    df["since"] = df.block - df.run_id.map(start)
    first10 = df[(df.since >= 0) & (df.since < 10)]
    clean_onset = first10.groupby("run_id").W.sum().eq(0) & first10.groupby("run_id").size().eq(10)
    eligible = clean_onset[clean_onset].index
    pf_of = df.groupby("run_id").poison_frac.first()
    held_none = df.split.isin(["val", "test"]) & (df.attack == "none")

    # Alarm-level ORs: each part calibrated to half the budget, then combined (the OR's budget is checked below).
    def half(sc: pd.Series) -> pd.Series:
        global ALARMS_PER_RUN
        full, ALARMS_PER_RUN = ALARMS_PER_RUN, ALARMS_PER_RUN / 2
        try:
            return sc > calibrate(sc, run, train_none)
        finally:
            ALARMS_PER_RUN = full
    combos = {"v1 OR v2 sum of 3 (alarm OR)": (scores["v1 (old signal)"], scores["v2 sum of 3 blocks"]),
              "v1 OR v2 (alarm OR)": (scores["v1 (old signal)"], scores["v2 (new signal)"])}
    alarms = {name: sc > calibrate(sc, run, train_none) for name, sc in scores.items()}
    thresholds = {name: float("nan") for name in combos}
    for name, (a, b) in combos.items():
        alarms[name] = half(a) | half(b)

    rows, delays = [], []
    for name, alarm in alarms.items():
        thr = thresholds.get(name, float("nan"))
        fa = onsets(alarm[held_none], run[held_none])
        fa_blocks = alarm[held_none].groupby(run[held_none]).sum()
        hits = df[alarm & (df.since >= 0)].groupby("run_id").since.min()
        for pf, window in ERASURE.items():
            runs = [r for r in eligible if pf_of[r] == pf]
            d = np.array([hits.get(r, np.inf) for r in runs], float)
            delays += [{"detector": name, "attack": LABEL[pf], "run_id": r, "delay": x} for r, x in zip(runs, d)]
            rows.append({"detector": name, "attack": LABEL[pf], "runs": len(d),
                         "caught_in_time": float(np.mean(d + 1 <= window)),
                         "caught_within_10": float(np.mean(d <= 9)), "median_delay": float(np.median(d)),
                         "false_alarms_per_heldout_run": float(fa.mean()),
                         "false_alarm_blocks_per_heldout_run": float(fa_blocks.mean()), "threshold": thr})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "detectors.csv", index=False)
    pd.DataFrame(delays).to_csv(OUT / "delays.csv", index=False)
    pd.set_option("display.width", 220)
    n_runs = {LABEL[pf]: sum(pf_of[r] == pf for r in eligible) for pf in ERASURE}
    print(f"Attacked runs undefended at onset: {n_runs}; held-out no-attack runs: {df[held_none].run_id.nunique()}; "
          f"alarm budget {ALARMS_PER_RUN} per no-attack training run\n")
    caught = t.pivot_table(index="detector", columns="attack", values="caught_in_time", sort=False).round(2)
    delay = t.pivot_table(index="detector", columns="attack", values="median_delay", sort=False).add_prefix("median delay ")
    fa = t.groupby("detector", sort=False)[["false_alarms_per_heldout_run", "false_alarm_blocks_per_heldout_run"]].first().round(2)
    print("caught in time (share):")
    print(caught.join(delay).join(fa).to_string())
    fig_detectors(t)


def fig_detectors(t: pd.DataFrame) -> None:
    attacks = ["1/8", "2/8", "8/8"]
    # Plotted: the 6 informative detectors (all of them are in detectors.csv). The plain CUSUMs are left out:
    # their alarms persist for tens of blocks, so they cost 5-55 defended blocks per no-attack run.
    names = ["v1 (old signal)", "v2 (new signal)", "v1 OR v2", "v2 sum of 3 blocks", "multi-signal CUSUM",
             "v1 OR v2 sum of 3 (alarm OR)"]
    colors = dict(zip(names, SERIES.values()))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [3, 1.2]})
    width = 0.84 / len(names)
    for k, n in enumerate(names):
        s = t[t.detector == n].set_index("attack").reindex(attacks)
        x = np.arange(len(attacks)) - 0.42 + width * (k + 0.5)
        axes[0].bar(x, s.caught_in_time, width * 0.9, color=colors[n], label=n)
        axes[1].bar(k, s.false_alarms_per_heldout_run.iloc[0], 0.7, color=colors[n])
    axes[0].set_xticks(range(len(attacks)), [f"{a} poisoned\n(erasure window {w} block{'s' * (w > 1)})"
                                             for a, w in zip(attacks, [10, 5, 1])])
    axes[0].set_ylabel("share of attacks caught in time")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    axes[1].set_xticks(range(len(names)), [n.split(" (")[0] for n in names], rotation=45, ha="right")
    axes[1].set_ylabel("false alarms per held-out\nno-attack run")
    axes[1].axhline(ALARMS_PER_RUN, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
    fig.suptitle("Onset detection on xp4 logs: same false-alarm budget for every detector (1 per no-attack training run)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "detectors.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
