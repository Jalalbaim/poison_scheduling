"""Pilot for xp5 (Fashion-MNIST, several attack families, Krum / Median / TrimmedMean). Runs on Kaggle.

    python pilot_xp5.py --jobs 4          # runs + report -> results/pilot_xp5
    python pilot_xp5.py --report-only

Part A, damage grid: every attack x every defense x {always off, always on} x 2 seeds. Attack window: blocks
30-89 (60 blocks of 2 rounds), 250 blocks. Answers: does each attack do damage on Fashion-MNIST, does each defense
stop it, and what does each defense cost in clean accuracy?
    ALIE has no targeted outcome: its harm is the clean-accuracy gap to the no-attack run of the same seed and
    schedule (the goal is to reach no-attack performance).
Part B, erasure windows: backdoor (corner patch) at 1/8, 2/8, 8/8 with each defense switched on d blocks after the
attack starts (d = 0, 2, 5, 10, 20, 40) and kept on to the end of the window; 1 seed. Sets the reaction delays of
the xp5 schedules, per defense.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import poison_testbed  # noqa: E402,F401  (CUBLAS_WORKSPACE_CONFIG)
from analyze import INK, INK2, SERIES  # noqa: E402
from poison_testbed.runner import RunConfig, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/pilot_xp5"
XP1_C, SIGMA, BLOCK_LEN, N_BLOCKS, WINDOW = 3.0552712798118593, 0.01, 2, 250, (30, 89)
DEFENSES = ["Krum", "Median", "TrimmedMean"]
# name -> RunConfig fields
ATTACKS = {
    "none": dict(attack="none"),
    "backdoor corner 2/8": dict(attack="backdoor_win", poison_frac=0.25),
    "backdoor corner 8/8": dict(attack="backdoor_win", poison_frac=1.0),
    "backdoor patch_r3_c5 2/8": dict(attack="backdoor_win", poison_frac=0.25, trigger="patch_r3_c5"),
    "backdoor blend 2/8": dict(attack="backdoor_win", poison_frac=0.25, trigger="blend"),
    "burst corner 8/8 (5 of 10)": dict(attack="backdoor_burst", poison_frac=1.0, burst=(10, 5)),
    "flip 2/8": dict(attack="flip_win", poison_frac=0.25),
    "flip 8/8": dict(attack="flip_win", poison_frac=1.0),
    "alie": dict(attack="alie_win"),
}
SEEDS_A = [7000, 7001]
ERASURE_RATES, ERASURE_DELAYS, SEED_B = [0.125, 0.25, 1.0], [0, 2, 5, 10, 20, 40], 7100


def config(fields: dict, defense: str, schedule: str, seed: int) -> RunConfig:
    return RunConfig(defense=defense, seed=seed, C=XP1_C, sigma=SIGMA, schedule=schedule, block_len=BLOCK_LEN,
                     n_blocks=N_BLOCKS, window=WINDOW, always_measure_backdoor=True, signals="v2",
                     dataset="fmnist", extended_log=True, **fields)


def jobs() -> list[tuple[str, str, dict, str, str, int]]:
    out = []
    for name, fields in ATTACKS.items():
        for d in DEFENSES:
            for sch in ["off", "on"]:
                for s in SEEDS_A:
                    out.append(("A", name, fields, d, sch, s))
    for pf in ERASURE_RATES:
        for d in DEFENSES:
            for delay in ERASURE_DELAYS:
                out.append(("B", f"erasure {pf}", dict(attack="backdoor_win", poison_frac=pf), d,
                            f"react_d{delay}_e0", SEED_B))
    return out


def run_one(part: str, name: str, fields: dict, defense: str, schedule: str, seed: int) -> tuple[str, float]:
    cfg = config(fields, defense, schedule, seed)
    run_id = f"{part}__{name.replace('/', 'of').replace(' ', '_')}__{defense}__{schedule}__s{seed}"
    out = OUT / "runs" / run_id
    if (out / "blocks.csv").exists():
        return run_id, json.loads((out / "config.json").read_text())["wall_clock_s"]
    started = time.time()
    df = simulate(cfg).copy()
    wall = time.time() - started
    df.insert(0, "part", part)
    df.insert(1, "name", name)
    df["run_id"] = run_id
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "blocks.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "config": cfg.__dict__, "config_hash": cfg.hash(), "wall_clock_s": round(wall, 1),
        "device_name": torch.cuda.get_device_name(0) if cfg.device == "cuda" else cfg.device}, indent=2, default=str))
    return run_id, wall


def report() -> None:
    b = pd.concat([pd.read_csv(f) for f in sorted((OUT / "runs").glob("*/blocks.csv"))], ignore_index=True)
    b = b[b.block >= 0]
    b.to_csv(OUT / "blocks_all.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    s, e = WINDOW
    a = b[b.part == "A"].copy()
    a["schedule"] = a.run_id.str.extract(r"__(on|off)__s")[0]
    rows = []
    for (name, d, sch, seed), g in a.groupby(["name", "defense", "schedule", "seed"]):
        g = g.sort_values("block")
        win = g[(g.block >= s) & (g.block <= e)]
        rows.append({"attack": name, "defense": d, "schedule": sch, "seed": seed,
                     "final_clean": g.y_clean_acc.iloc[-1], "clean_at_window_end": g.y_clean_acc.iloc[e],
                     "mean_clean_in_window": win.y_clean_acc.mean(),
                     "atk_peak": g.y_atk.max(), "atk_mean_in_window": win.y_atk.mean(), "atk_final": g.y_atk.iloc[-1]})
    r = pd.DataFrame(rows)
    none = r[r.attack == "none"].set_index(["defense", "schedule", "seed"])
    key = list(zip(r.defense, r.schedule, r.seed))
    r["clean_gap_vs_no_attack_final"] = r.final_clean.to_numpy() - none.final_clean.reindex(key).to_numpy()
    r["clean_gap_vs_no_attack_in_window"] = (r.mean_clean_in_window.to_numpy()
                                             - none.mean_clean_in_window.reindex(key).to_numpy())
    r.to_csv(OUT / "damage_runs.csv", index=False)
    t = r.groupby(["attack", "defense", "schedule"]).mean(numeric_only=True).drop(columns="seed").round(3)
    t.to_csv(OUT / "damage.csv")
    print("## Part A: damage per attack x defense, always off vs always on (mean of 2 seeds)")
    cols = ["atk_peak", "atk_mean_in_window", "atk_final", "final_clean", "clean_gap_vs_no_attack_in_window",
            "clean_gap_vs_no_attack_final"]
    print(t[cols].unstack("schedule").round(3).to_string())

    eb = b[b.part == "B"].copy()
    eb["delay"] = eb.run_id.str.extract(r"react_d(\d+)_e0")[0].astype(int)
    eb["pf"] = eb.name.str.replace("erasure ", "").astype(float)
    fin = eb[eb.block == eb.block.max()].pivot_table(index=["defense", "pf"], columns="delay", values="y_atk").round(3)
    fin.to_csv(OUT / "erasure.csv")
    print("\n## Part B: final backdoor success when the defense starts d blocks after the attack (corner patch)")
    print(fin.to_string())
    fig_erasure(fin)


def fig_erasure(fin: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(DEFENSES), figsize=(4.2 * len(DEFENSES), 3.6), sharey=True, squeeze=False)
    colors = dict(zip(ERASURE_RATES, list(SERIES.values())[:3]))
    for ax, d in zip(axes[0], DEFENSES):
        if d not in fin.index.get_level_values(0):
            continue
        for pf in ERASURE_RATES:
            if (d, pf) in fin.index:
                v = fin.loc[(d, pf)]
                ax.plot(v.index, v.values, marker="o", color=colors[pf], label=f"{int(pf * 8)}/8 poisoned")
        ax.set_title(d, color=INK, loc="left")
        ax.set_xlabel("blocks between attack start and defense (2 rounds each)")
        ax.axhline(0.1, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
    axes[0, 0].set_ylabel("final backdoor success")
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fashion-MNIST erasure windows: how long can each defense wait and still erase the backdoor?",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "erasure.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--limit", type=int, help="first N jobs only (smoke test)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        todo = jobs()[: args.limit] if args.limit else jobs()
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
            futs = [pool.submit(run_one, *j) for j in todo]
            for k, fut in enumerate(as_completed(futs), 1):
                run_id, wall = fut.result()
                print(f"[{k}/{len(todo)}] {run_id}  {wall:.0f}s  (elapsed {time.time() - started:.0f}s)", flush=True)
    report()


if __name__ == "__main__":
    main()
