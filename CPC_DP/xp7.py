"""xp7: when to switch Krum on, under 4 levels of confounding (notes/03_xp7_design.md).

    ../.venv/Scripts/python xp7.py --list                   # plan table, run nothing
    ../.venv/Scripts/python xp7.py --part a --jobs 4        # main runs, gamma 0 and 2     (400 runs)
    ../.venv/Scripts/python xp7.py --part b --jobs 4        # main runs, gamma 8 and hard  (400 runs)
    ../.venv/Scripts/python xp7.py --part twins --jobs 4    # onset twins, val + test seeds (EVALUATION/SELECTION ONLY)
    ../.venv/Scripts/python xp7.py --part a --smoke         # 6 runs of 30 blocks (pipeline check)

Design
- 200 base setups (seed, attack, window, target), each run under the 4 policies below: the gamma levels differ
  only in the policy, so they can be compared run by run.
- Policies: gamma_g0 (fair coin every 10 blocks), gamma_g2, gamma_g8 (P(on) = sigmoid(gamma (s - 3)), s = highest
  alarm score since the last decision), hard (xp6 alarm rule, random tau / delay / hold). Propensities are logged.
- Attacks (Krum only, iid workers): none, backdoor corner 2/8, backdoor patch 8/8, targeted flip 8/8, ALIE,
  SignFlip. 25% of attacked setups (not SignFlip) are persistent: active from block 1-10 to the last block.
- Splits by base setup (same split at every gamma level), stratified by family: 70% train, 10% val, 20% test.
- Twins: the val + test setups, rerun open loop: 'off', and Krum on over [onset + d, onset + d + 20) for
  d in {0, 5, 10}. Same seed: the true Delta_20 at that history. Placed relative to the TRUE window: val twins may
  be used for model selection, test twins for evaluation only, neither for training.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE.parent / "GRU_mnist" / "run_experiment.py").exists():  # local layout; on Kaggle the files sit side by side
    sys.path.insert(0, str(HERE.parent / "GRU_mnist"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_experiment as R  # noqa: E402
from poison_testbed.runner import RunConfig  # noqa: E402

N_BASE, SEED0 = 200, 50_000
FAMILIES = {"none": 0.20, "corner": 0.16, "patch": 0.16, "flip": 0.16, "alie": 0.16, "signflip": 0.16}
PERSISTENT = 0.25
# Pilot: under averaging the update is (13 - 3 scale) / 16 x the honest mean. Scale <= 4.33 does no harm; any scale
# above reverses training (clean accuracy to chance) and the weights grow: signals reach ~3e8 at 4.6, ~1e22 at 5 over
# 80 blocks, and a persistent window at 5 crashes the run. So: 4.6, xp6-style windows only (<= 80 blocks).
SIGNFLIP_SCALE = 4.6
GAMMAS = {"g0": "gamma_g0_c3_p10", "g2": "gamma_g2_c3_p10", "g8": "gamma_g8_c3_p10", "hard": None}
PARTS = {"a": ["g0", "g2"], "b": ["g8", "hard"]}
TWIN_PLANS = [(0, 20), (5, 20), (10, 20)]


def base_setups(overrides: dict) -> pd.DataFrame:
    """One row per base setup: everything but the policy, plus the split."""
    rows = []
    for i in range(N_BASE):
        rng = np.random.default_rng(110_000 + i)
        fam = str(rng.choice(list(FAMILIES), p=list(FAMILIES.values())))
        persistent = fam not in ("none", "signflip") and rng.random() < PERSISTENT
        if persistent:
            window = (int(rng.integers(1, 11)), R.XP2_BLOCKS - 1)
        else:
            window = R._xp5_window(rng)
        fields = {"attack": "none", "trigger": "corner", "poison_frac": 1.0, "flip_source": 1}
        if fam == "corner":
            fields.update(attack="backdoor_win", trigger="corner", poison_frac=0.25)
        elif fam == "patch":
            fields.update(attack="backdoor_win", trigger="patch_r3_c5", poison_frac=1.0)
        elif fam == "flip":
            fields.update(attack="flip_win", poison_frac=1.0, flip_source=int(rng.choice(R.XP5_FLIP_SOURCES)))
        elif fam == "alie":
            fields.update(attack="alie_win")
        elif fam == "signflip":
            fields.update(attack="sign_flip_win")
        rows.append({"base": i, "seed": SEED0 + i, "family": fam, "persistent": persistent, "window": window,
                     "target": int(rng.choice(R.XP5_TARGETS)), "hard_schedule": _hard(rng), **fields})
    df = pd.DataFrame(rows)
    rank = df.groupby("family").cumcount()
    df["split"] = np.select([rank % 10 < 2, rank % 10 == 2], ["test", "val"], "train")
    return df


def _hard(rng: np.random.Generator) -> str:
    return (f"alarm_t{rng.choice(R.XP6_ALARM_TAU)}_d{rng.choice(R.XP6_ALARM_DELAY)}"
            f"_h{rng.choice(R.XP6_ALARM_HOLD)}")


def config(row, schedule: str, overrides: dict) -> RunConfig:
    return RunConfig(defense="Krum", seed=int(row.seed), schedule=schedule, window=tuple(row.window),
                     attack=row.attack, trigger=row.trigger, poison_frac=float(row.poison_frac),
                     flip_source=int(row.flip_source), target=int(row.target), sign_flip_scale=SIGNFLIP_SCALE,
                     **R.xp5_params(overrides))


def main_configs(levels: list[str], overrides: dict) -> tuple[list[RunConfig], pd.DataFrame]:
    base = base_setups(overrides)
    out, rows = [], []
    for level in levels:
        for row in base.itertuples():
            cfg = config(row, GAMMAS[level] or row.hard_schedule, overrides)
            out.append(cfg)
            rows.append({"run_id": cfg.run_id, "base": row.base, "seed": row.seed, "level": level,
                         "family": row.family, "persistent": row.persistent, "split": row.split,
                         "onset": row.window[0], "end": row.window[1], "schedule": cfg.schedule})
    return out, pd.DataFrame(rows)


def twin_configs(overrides: dict) -> tuple[list[RunConfig], pd.DataFrame]:
    base = base_setups(overrides)
    out, rows = [], []
    for row in base[base.split != "train"].itertuples():
        n = R.xp5_params(overrides)["n_blocks"]
        plans = [("off", None, None)]
        for d, k in TWIN_PLANS:
            t = row.window[0] + d
            if t < n:
                plans.append((f"off+flip_s{t}_l{min(k, n - t)}", d, min(k, n - t)))
        for schedule, d, k in plans:
            cfg = config(row, schedule, overrides)
            out.append(cfg)
            rows.append({"run_id": cfg.run_id, "base": row.base, "seed": row.seed, "split": row.split,
                         "family": row.family, "attack": row.attack, "persistent": row.persistent,
                         "group": "none" if row.family == "none" else "attacked", "onset": row.window[0],
                         "plan": "off" if d is None else "on", "d": d, "k": k,
                         "t": None if d is None else row.window[0] + d})
    return out, pd.DataFrame(rows).astype({"d": "Int64", "k": "Int64", "t": "Int64"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["a", "b", "twins"], default="a")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="a few runs of 30 blocks")
    args = ap.parse_args()
    overrides = R.parse_overrides(args.set + (["n_blocks=30"] if args.smoke else []))
    if args.part == "twins":
        configs, manifest = twin_configs(overrides)
        print(manifest.groupby(["split", "group", "plan"]).size().unstack(fill_value=0).to_string())
    else:
        configs, manifest = main_configs(PARTS[args.part], overrides)
        print(manifest.groupby(["level", "family"]).size().unstack(fill_value=0).to_string())
        print(manifest.groupby(["split"]).base.nunique().to_string())
    if args.smoke:  # one setup per family, early onset so something happens in 30 blocks
        keep = manifest.groupby("family").head(1).index[:6]
        configs, manifest = [configs[i] for i in keep], manifest.loc[keep].reset_index(drop=True)
        configs = [replace(c, window=(min(c.window[0], 5), c.window[1])) for c in configs]
    print(f"total runs: {len(configs)}", flush=True)
    if args.list:
        return
    out = args.out or HERE / "results" / f"xp7_{args.part}{'_smoke' if args.smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "manifest.csv", index=False)
    R.execute(configs, out, args.jobs)


if __name__ == "__main__":
    main()
