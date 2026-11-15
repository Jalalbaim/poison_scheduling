"""Data-poisoning testbed experiments.

    python run_experiment.py calibrate                 # fix C once -> calibration.json
    python run_experiment.py sanity  [--jobs 3]        # Step 2: always-off vs always-on, 3 seeds
    python run_experiment.py collect [--jobs 3]        # Step 3: random Bernoulli(0.5) schedules, 20 seeds
    python run_experiment.py counterfactual            # GRU check 3: test seeds x candidate schedules
    python run_experiment.py xp2 [--jobs 4]            # xp2: windowed backdoor + Krum, data with kills
    python run_experiment.py report                    # Step 2 table from runs_sanity/

Any RunConfig field can be overridden with --set key=value (e.g. --set sigma=0.02 --set sign_flip_scale=5).
Finished runs (same config hash) are skipped, so an interrupted command can simply be re-run.
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

import poison_testbed  # noqa: F401  (sets CUBLAS_WORKSPACE_CONFIG before torch touches CUDA)
from poison_testbed.runner import RunConfig, run, simulate

ROOT = Path(__file__).resolve().parent
CALIBRATION = ROOT / "calibration.json"
ATTACKS = ["none", "label_flip", "backdoor", "backdoor_win", "alie", "sign_flip"]
DEFENSES = ["ClipNoise", "Krum", "MultiKrum", "Median", "TrimmedMean", "Bulyan"]
DEFAULT_SIGMA = 0.01


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
        if isinstance(out[key], list):
            out[key] = tuple(out[key])
    return out


def base_params(overrides: dict) -> dict:
    if "C" not in overrides and not CALIBRATION.exists():
        raise SystemExit("No calibration.json: run `python run_experiment.py calibrate` first (or pass --set C=...).")
    params = {"sigma": DEFAULT_SIGMA}
    if CALIBRATION.exists():
        params["C"] = json.loads(CALIBRATION.read_text())["C"]
    return {**params, **overrides}


def _job(cfg: RunConfig, out_root: str) -> tuple[str, float]:
    _, meta = run(cfg, Path(out_root))
    return cfg.run_id, meta["wall_clock_s"]


def execute(configs: list[RunConfig], out_root: Path, jobs: int) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    started, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(_job, cfg, str(out_root)) for cfg in configs]
        for fut in as_completed(futures):
            run_id, wall = fut.result()  # fail fast: re-raises the run's exception
            done += 1
            print(f"[{done}/{len(configs)}] {run_id}  {wall:.1f}s  (elapsed {time.time() - started:.0f}s)", flush=True)
    frames = [pd.read_csv(out_root / c.run_id / "blocks.csv", dtype={"round_start": "Int64", "W": "Int64"})
              for c in configs]
    pd.concat(frames, ignore_index=True).to_csv(out_root / "all_blocks.csv", index=False)
    print(f"wrote {out_root / 'all_blocks.csv'}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    over = parse_overrides(args.set)
    cfg = RunConfig(attack="none", defense="Average", seed=0, schedule="off", **{"C": 1.0, "sigma": 0.0, **over})
    t = time.time()
    df = simulate(cfg)
    norms = df.loc[df.block >= 0, "x_norm_mean"]
    C = float(norms.median())
    CALIBRATION.write_text(json.dumps({
        "C": C,
        "rule": "median over blocks of the mean worker-gradient norm; attack=none, always Average, seed 0",
        "norm_first_block": float(norms.iloc[0]), "norm_last_block": float(norms.iloc[-1]),
        "final_clean_acc": float(df.y_clean_acc.iloc[-1]), "wall_clock_s": round(time.time() - t, 1),
    }, indent=2))
    print(CALIBRATION.read_text())


def cmd_sanity(args: argparse.Namespace) -> None:
    p = base_params(parse_overrides(args.set))
    seeds = range(args.seeds or 3)
    attacks, defenses = args.attacks or ATTACKS, args.defenses or DEFENSES
    configs = [RunConfig(attack=a, defense="Average", seed=s, schedule="off", **p) for a in attacks for s in seeds]
    configs += [RunConfig(attack=a, defense=d, seed=s, schedule="on", **p)
                for a in attacks for d in defenses for s in seeds]
    execute(configs, ROOT / (args.out or "runs_sanity"), args.jobs)


def cmd_collect(args: argparse.Namespace) -> None:
    p = base_params(parse_overrides(args.set))
    seeds = range(args.seeds or 20)
    attacks, defenses = args.attacks or ATTACKS, args.defenses or DEFENSES
    configs = [RunConfig(attack=a, defense=d, seed=s, schedule="random", **p)
               for a in attacks for d in defenses for s in seeds]
    execute(configs, ROOT / (args.out or "runs"), args.jobs)


CF_SEEDS = [16, 17, 18, 19]  # the GRU's held-out test seeds
CF_ATTACKS = ["none", "backdoor", "backdoor_win", "sign_flip"]
CF_DEFENSES = ["ClipNoise", "Krum", "Median", "Bulyan"]
CF_PREFIX = 3
CF_SCHEDULES = ["random", *(f"cf_{c}_p{CF_PREFIX}" for c in ["on", "off", "ontil20", "offtil20", "alt"])]


def cmd_counterfactual(args: argparse.Namespace) -> None:
    """Check 3: true counterfactuals for held-out seeds. 'random' re-runs the xp1 schedule (reproduction check)."""
    p = base_params(parse_overrides(args.set))
    seeds = range(args.seeds) if args.seeds else CF_SEEDS
    attacks, defenses = args.attacks or CF_ATTACKS, args.defenses or CF_DEFENSES
    configs = [RunConfig(attack=a, defense=d, seed=s, schedule=sch, **p)
               for a in attacks for d in defenses for s in seeds for sch in CF_SCHEDULES]
    execute(configs, ROOT / (args.out or "runs_cf"), args.jobs)


XP1_C = 3.0552712798118593  # pinned: X signals (x_frac_clipped) stay comparable with xp1
XP2_RUNS = 260
XP2_BLOCK_LEN, XP2_BLOCKS = 2, 250  # decide every 2 rounds; 500 rounds as in xp1
XP2_STICKY_P = [0.01, 0.02, 0.05, 0.1]
XP2_REACT_DELAY = [-4, -2, 0, 1, 2, 3, 5, 8, 12]  # blocks of 2 rounds after the window start
XP2_REACT_EXTRA = [0, 2, 5, 10]


def xp2_config(i: int, overrides: dict) -> RunConfig:
    """Run i of xp2: its own seed, a random window (real or phantom), and a random schedule.

    Attack mix by i % 10: 0-5 windowed backdoor, 6-7 none, 8-9 continuous backdoor. Every run draws a
    window; react schedules follow it whether or not an attack happens, so W never reveals the attack.
    """
    rng = np.random.default_rng(20_000 + i)  # per-run stream: adding runs never changes earlier ones
    attack = ["backdoor_win"] * 6 + ["none"] * 2 + ["backdoor"] * 2
    start = int(rng.integers(25, 151))
    window = (start, start + int(rng.integers(30, 81)) - 1)
    family = rng.choice(["sticky", "react", "on", "off"], p=[0.45, 0.45, 0.05, 0.05])
    if family == "sticky":
        schedule = f"sticky_p{rng.choice(XP2_STICKY_P)}"
    elif family == "react":
        schedule = f"react_d{rng.choice(XP2_REACT_DELAY)}_e{rng.choice(XP2_REACT_EXTRA)}"
    else:
        schedule = str(family)
    params = {"C": XP1_C, "sigma": DEFAULT_SIGMA, "block_len": XP2_BLOCK_LEN, "n_blocks": XP2_BLOCKS, **overrides}
    return RunConfig(attack=attack[i % 10], defense="Krum", seed=i, schedule=schedule, window=window, **params)


def cmd_xp2(args: argparse.Namespace) -> None:
    """xp2: training data that contains kills (windowed backdoor + Krum, decisions every 2 rounds)."""
    over = parse_overrides(args.set)
    configs = [xp2_config(i, over) for i in range(args.seeds or XP2_RUNS)]
    execute(configs, ROOT / (args.out or "runs_xp2"), args.jobs)


def cmd_report(args: argparse.Namespace) -> None:
    """Step 2 table: final outcomes per (attack, arm), mean ± sd over seeds, and the on − off gaps."""
    root = ROOT / (args.out or "runs_sanity")
    df = pd.read_csv(root / "all_blocks.csv")
    last = df.block.max()
    df["arm"] = df.defense.where(df.W == 1, "Average")
    final = df[df.block == last]
    win_end = df[(df.attack == "backdoor_win") & (df.block == 30)]  # end of the window (blocks 15-30)

    def agg(frame: pd.DataFrame, col: str) -> pd.DataFrame:
        return frame.groupby(["attack", "arm"])[col].agg(["mean", "std"])

    acc, atk, atk_win = agg(final, "y_clean_acc"), agg(final, "y_atk"), agg(win_end, "y_atk")
    lines = ["| attack | arm | clean acc (final) | Δ acc vs Average | backdoor success (final) | Δ success | "
             "backdoor success (block 30) |", "|" + "---|" * 7]
    for attack in ATTACKS:
        if attack not in acc.index.get_level_values(0):
            continue
        base_acc = acc.loc[(attack, "Average"), "mean"]
        base_atk = atk.loc[(attack, "Average"), "mean"]
        for arm in ["Average", *DEFENSES]:
            if (attack, arm) not in acc.index:
                continue
            a, s = acc.loc[(attack, arm)]
            t = atk.loc[(attack, arm)]
            atk_str = "–" if pd.isna(t["mean"]) else f"{t['mean']:.3f} ± {t['std']:.3f}"
            d_atk = "–" if pd.isna(t["mean"]) or arm == "Average" else f"{t['mean'] - base_atk:+.3f}"
            w = atk_win.loc[(attack, arm)] if (attack, arm) in atk_win.index else None
            w_str = "–" if w is None else f"{w['mean']:.3f} ± {w['std']:.3f}"
            d_acc = "" if arm == "Average" else f"{a - base_acc:+.3f}"
            lines.append(f"| {attack} | {arm} | {a:.3f} ± {s:.3f} | {d_acc} | {atk_str} | {d_atk} | {w_str} |")
    walls = [json.loads(p.read_text())["wall_clock_s"] for p in root.glob("*/config.json")]
    lines.append(f"\nWall clock per run: median {pd.Series(walls).median():.1f}s "
                 f"(min {min(walls):.1f}, max {max(walls):.1f}) over {len(walls)} runs.")
    text = "\n".join(lines)
    (root / "sanity_table.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in [("calibrate", cmd_calibrate), ("sanity", cmd_sanity), ("collect", cmd_collect),
                     ("counterfactual", cmd_counterfactual), ("xp2", cmd_xp2), ("report", cmd_report)]:
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if name == "report":
            sp.add_argument("--out", help="run folder name under the project root (default runs_sanity)")
        elif name != "calibrate":
            sp.add_argument("--jobs", type=int, default=3)
            sp.add_argument("--seeds", type=int)
            sp.add_argument("--attacks", nargs="+", choices=ATTACKS)
            sp.add_argument("--defenses", nargs="+", choices=DEFENSES)
            sp.add_argument("--out", help="output folder name under the project root")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the report prints "Δ" / "±"; Windows consoles default to cp1252
    args.fn(args)


if __name__ == "__main__":
    main()
