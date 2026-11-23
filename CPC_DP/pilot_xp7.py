"""xp7 pilot, local, one run at a time: run time per run, the clean-accuracy cost of Krum under non-iid workers,
and the persistent attack (active from an early block to the end of the run).

    ../.venv/Scripts/python pilot_xp7.py            # -> results/pilot_xp7/{runs/, summary.csv}

Settings = xp6 (Fashion-MNIST, 250 blocks of 2 rounds, n = 16, f = 3, v2 signals, extended log), Krum.
"""

import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "GRU_mnist"))

import poison_testbed  # noqa: E402,F401
from poison_testbed.runner import RunConfig, run  # noqa: E402
from run_experiment import xp5_params  # noqa: E402

OUT = HERE / "results/pilot_xp7"
ALPHAS = [None, 1.0, 0.3, 0.1]
PERSISTENT = (5, 249)  # attack from block 5 to the last block


def configs() -> list[tuple[str, RunConfig]]:
    out = []
    for seed in (70_000, 70_001):
        for alpha in ALPHAS:
            for sched in ("off", "on"):
                out.append(("noniid cost", RunConfig(attack="none", defense="Krum", seed=seed, schedule=sched,
                                                     noniid_alpha=alpha, **xp5_params({}))))
    seed = 70_002
    for attack, extra in [("backdoor_win", dict(trigger="corner", poison_frac=0.25)),
                          ("flip_win", dict(poison_frac=1.0)), ("sign_flip_win", {}), ("alie_win", {})]:
        for sched in ("off", "on", "gamma_g8_c3_p10"):
            out.append(("persistent", RunConfig(attack=attack, defense="Krum", seed=seed, schedule=sched,
                                                window=PERSISTENT, **extra, **xp5_params({}))))
    return out


def main() -> None:
    rows = []
    jobs = configs()
    for i, (part, cfg) in enumerate(jobs):
        t0 = time.time()
        df, meta = run(cfg, OUT / "runs" / f"alpha_{cfg.noniid_alpha}")  # run_id does not carry alpha
        last = df.iloc[-1]
        rows.append({"part": part, "attack": cfg.attack, "alpha": cfg.noniid_alpha, "seed": cfg.seed,
                     "schedule": cfg.schedule, "share_on": df.W.iloc[1:].astype(float).mean(),
                     "clean_final": last.y_clean_acc, "clean_mean": df.y_clean_acc.iloc[1:].mean(),
                     "atk_final": last.y_atk, "atk_mean": df.y_atk.iloc[1:].mean(), "wall_s": meta["wall_clock_s"]})
        print(f"[{i + 1}/{len(jobs)}] {cfg.run_id} alpha={cfg.noniid_alpha}  clean {last.y_clean_acc:.3f}  "
              f"atk {last.y_atk:.3f}  {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
