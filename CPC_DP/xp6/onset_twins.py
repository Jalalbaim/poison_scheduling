"""xp6 onset twins: EVALUATION-ONLY counterfactual ground truth for Delta_k(H_t) (assessment §3, §6 step 1).

    ../.venv/Scripts/python xp6/onset_twins.py --jobs 4                       # 555 runs -> results/xp6/eval/
    ../.venv/Scripts/python xp6/onset_twins.py --list                         # print the plan table, run nothing
    ../.venv/Scripts/python xp6/onset_twins.py --smoke                        # 5 runs of 30 blocks (pipeline check)

Estimand (agreed 2026-11-21): Delta_k(H_t) = E[U(on for k blocks from t, then off) - U(off) | H_t],
U = y_clean(T) - lambda * y_atk(t + tau). Each selected xp6 seed is rerun under open-loop plans that share the
same all-off prefix up to the decision block t = onset + d:

    off                       the reference (pi_0 = off everywhere)
    off+flip_s{t}_l{k}        Krum on over blocks [t, t + k), off elsewhere

Same seed and config as the xp6 run, so the pair (plan, off) is identical before t and the difference of their
outcomes is the true Delta_k of that run at history H_t (the off run's own rows up to t).

These runs are placed RELATIVE TO THE TRUE ATTACK WINDOW. That is fine for evaluation and would be a leak in
training (the xp5 lesson): they live in their own folder (results/xp6/eval) and must never be read by a training
or model-selection script. Seeds: the xp6 TEST split (same rule as gru_experiment.assign_splits) and the held-out
groups. Nothing here changes the xp6 data, which is collected unchanged by GRU_mnist (run_kaggle.sh xp6).
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name in ("xp6", "xp7") else HERE  # local: CPC_DP/xp7/; Kaggle: files flat in $WORK
sys.path.insert(0, str(ROOT))
if (ROOT.parent / "GRU_mnist" / "run_experiment.py").exists():  # local layout; on Kaggle the files sit side by side
    sys.path.insert(0, str(ROOT.parent / "GRU_mnist"))

import pandas as pd  # noqa: E402

import run_experiment as R  # noqa: E402
from poison_testbed.runner import RunConfig  # noqa: E402

# (d, k) plans per group: d = blocks after the onset at which the defense switches on, k = blocks it stays on.
PLANS = {
    "attacked": [(0, 20), (2, 20), (5, 20), (10, 20), (2, 10), (2, 40)],
    "none": [(0, 10), (0, 20), (0, 40)],  # phantom onset: the true Delta is the cost of defending only
    "heldout": [(0, 20), (5, 20)],
}


def test_split(main: list[RunConfig]) -> list[bool]:
    """xp6 test membership, as gru_experiment.assign_splits for a single defense: within each (attack, poison
    rate) group, rank the seeds; rank % 20 in 16-19 = test. (Re-implemented to avoid importing the GRU code on
    Kaggle; tests/test_onset_twins.py checks it against assign_splits.)"""
    meta = pd.DataFrame([{"seed": c.seed, "attack": c.attack, "poison_frac": c.poison_frac} for c in main])
    rank = meta.groupby(["attack", "poison_frac"]).seed.rank(method="first").astype(int) - 1
    return list(rank % 20 >= 16)


def onset_configs(overrides: dict | None = None) -> tuple[list[RunConfig], pd.DataFrame]:
    """(configs, manifest): one 'off' reference + the group's plans per selected seed, in a fixed order."""
    configs = R.xp6_configs(overrides or {})
    main, heldout = configs[: R.XP6_MAIN], configs[R.XP6_MAIN + R.XP6_TWINS:]
    chosen = [(c, "none" if c.attack == "none" else "attacked") for c, t in zip(main, test_split(main)) if t]
    chosen += [(c, "heldout") for c in heldout]
    out, rows = [], []
    for base, group in chosen:
        onset, n = base.window[0], base.n_blocks
        plans = [("off", None, None)]
        for d, k in PLANS[group]:
            t = onset + d
            if t >= n:  # only with shortened runs (smoke): no room for the plan
                continue
            plans.append((f"off+flip_s{t}_l{min(k, n - t)}", d, min(k, n - t)))
        for schedule, d, k in plans:
            cfg = replace(base, schedule=schedule)
            out.append(cfg)
            rows.append({"run_id": cfg.run_id, "seed": cfg.seed, "group": group, "attack": cfg.attack,
                         "poison_frac": cfg.poison_frac, "trigger": cfg.trigger, "onset": onset,
                         "plan": "off" if d is None else "on", "d": d, "k": k,
                         "t": None if d is None else onset + d})
    return out, pd.DataFrame(rows).astype({"d": "Int64", "k": "Int64", "t": "Int64"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "results/xp6/eval")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (as run_experiment)")
    ap.add_argument("--list", action="store_true", help="print the plan table and exit")
    ap.add_argument("--smoke", action="store_true", help="2 seeds x 3 plans, 30-block runs")
    args = ap.parse_args()
    overrides = R.parse_overrides(args.set + (["n_blocks=30"] if args.smoke else []))
    configs, manifest = onset_configs(overrides)
    if args.smoke:  # two attacked seeds with an early onset, their 'off' run and first two plans
        seeds = manifest[(manifest.group == "attacked") & (manifest.onset <= 10)].seed.unique()[:2]
        keep = manifest.seed.isin(seeds) & (manifest.plan.eq("off") | manifest.d.isin([0, 2]).fillna(False))
        keep &= ~(manifest.k.fillna(0) > 20) & ~((manifest.d == 2) & (manifest.k.fillna(20) < 20))
        configs = [c for c, k in zip(configs, keep) if k]
        manifest = manifest[keep].reset_index(drop=True)
    summary = manifest.groupby(["group", "plan"]).size().unstack(fill_value=0)
    print(summary.to_string(), f"\ntotal runs: {len(configs)} over {manifest.seed.nunique()} seeds", flush=True)
    if args.list:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out / "manifest.csv", index=False)
    R.execute(configs, args.out, args.jobs)


if __name__ == "__main__":
    main()
