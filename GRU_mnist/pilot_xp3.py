"""Pilot before designing xp3: the threshold defender (a threshold rule that holds the defense on), and candidate attacks that might beat it.

    python pilot_xp3.py                 # all variants, then summary + figures -> results/pilot_xp3
    python pilot_xp3.py --report-only

Step 1. "threshold_defender" detector: turns defense on when the spread of worker-gradient norms in the previous block
exceeds TAU; stays on until the signal has been <= TAU for K blocks in a row. K = 12: on xp2 training runs the
signal stayed low for at most 10 blocks while a defended attack continued. Tested on the deployment seeds of
deploy_xp2 (ref_win) and on no-attack seeds (false alarms). "threshold_nohold" = the memoryless rule (K = 1).

Step 2. Candidate xp3 attacks, each with always-off, always-on, oracle (defends exactly the active blocks) and
the threshold defender. A good xp3 attack: does damage (always-off), defending only when it matters pays (oracle kills it and is
more accurate than always-on), and the threshold defender fails on it (so a learned controller has room to help).

Everything else is the xp2 setup: Krum, decisions every 2 rounds (250 blocks), xp1's C, sigma 0.01.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import poison_testbed  # noqa: E402,F401  (CUBLAS_WORKSPACE_CONFIG)
from analyze import INK, INK2, SURFACE  # noqa: E402  (also applies the shared plot style)
from poison_testbed.poisoning import make_attack  # noqa: E402
from poison_testbed.runner import RunConfig, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/pilot_xp3"
XP1_C, SIGMA, BLOCK_LEN, N_BLOCKS = 3.0552712798118593, 0.01, 2, 250
SIGNAL, TAU, K_HOLD = "x_norm_std", 3.65, 12

# variant -> (attack, poison_frac, burst, window rule, seeds, policies)
DEPLOY_SEEDS, PILOT_SEEDS = [1000, 1001, 1002, 1003, 1004], [3000, 3001, 3002]
ATTACKED = ["off", "on", "oracle", "threshold_defender"]
VARIANTS = {
    "ref_win": ("backdoor_win", 1.0, None, "xp2", DEPLOY_SEEDS, ATTACKED),
    "none": ("none", 1.0, None, "xp2", DEPLOY_SEEDS, ["off", "on", "threshold_nohold", "threshold_defender"]),
    "low_pf_0.25": ("backdoor_win", 0.25, None, "xp2", PILOT_SEEDS, ATTACKED),
    "low_pf_0.125": ("backdoor_win", 0.125, None, "xp2", PILOT_SEEDS, ATTACKED),
    "burst": ("backdoor_burst", 1.0, (10, 2), "xp2", PILOT_SEEDS, ATTACKED),
    "early": ("backdoor_win", 1.0, None, "early", PILOT_SEEDS, ATTACKED),
}
POLICY_COLOR = {"off": "#1baf7a", "on": "#eb6834", "oracle": "#e87ba4", "threshold_nohold": "#eda100", "threshold_defender": "#2a78d6"}
POLICY_LABEL = {"off": "always off", "on": "always on", "oracle": "oracle (defends exactly the active blocks)",
                "threshold_nohold": "threshold, no hold", "threshold_defender": f"threshold defender (holds {K_HOLD} blocks)"}
DRAW_ORDER = ["off", "on", "oracle", "threshold_nohold", "threshold_defender"]


def window_for(seed: int, rule: str) -> tuple[int, int]:
    """xp2 rule (same draw as deploy_xp2.window_for): start 25-150, length 30-80. early: start 2-8."""
    rng = np.random.default_rng(40_000 + seed)
    start = int(rng.integers(25, 151))
    length = int(rng.integers(30, 81))
    if rule == "early":
        start = int(np.random.default_rng(50_000 + seed).integers(2, 9))
    return start, start + length - 1


class Detector:
    """Defend when the previous block's signal exceeds TAU; stay on until it has been low for k blocks."""

    def __init__(self, k: int):
        self.k, self.on, self.low = k, False, 0

    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        value = history[-1].get(SIGNAL)
        value = float("nan") if value is None else float(value)
        if value > TAU:
            self.on, self.low = True, 0
        elif self.on:
            self.low += 1
            if self.low >= self.k:
                self.on, self.low = False, 0
        return int(self.on), {"det_signal": value, "det_low_streak": self.low}


def config(variant: str, policy: str, seed: int) -> RunConfig:
    attack, pf, burst, rule, _, _ = VARIANTS[variant]
    schedule = {"threshold_defender": "policy", "threshold_nohold": "policy"}.get(policy, policy)
    return RunConfig(attack=attack, defense="Krum", seed=seed, C=XP1_C, sigma=SIGMA, schedule=schedule,
                     block_len=BLOCK_LEN, n_blocks=N_BLOCKS, window=window_for(seed, rule), poison_frac=pf,
                     burst=burst)


def run_one(variant: str, policy: str, seed: int) -> tuple[str, float]:
    run_id = f"{variant}__{policy}__s{seed}"
    out = OUT / "runs" / run_id
    if (out / "blocks.csv").exists():
        return run_id, json.loads((out / "config.json").read_text())["wall_clock_s"]
    cfg = config(variant, policy, seed)
    controller = {"threshold_defender": lambda: Detector(K_HOLD), "threshold_nohold": lambda: Detector(1)}.get(policy, lambda: None)()
    started = time.time()
    df = simulate(cfg, controller)
    wall = time.time() - started
    spec = make_attack(cfg.attack, window=cfg.window, burst=cfg.burst)
    df.insert(0, "variant", variant)
    df.insert(1, "policy", policy)
    df["run_id"] = run_id
    df["attack_active"] = [int(spec.active(b)) if b >= 0 else 0 for b in df.block]  # evaluation label
    df["poison_frac"] = cfg.poison_frac
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "blocks.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "variant": variant, "policy": policy, "config": cfg.__dict__, "signal": SIGNAL,
        "tau": TAU, "k_hold": K_HOLD, "wall_clock_s": round(wall, 1),
        "device_name": torch.cuda.get_device_name(0) if cfg.device == "cuda" else cfg.device,
    }, indent=2, default=str))
    return run_id, wall


# ----------------------------------------------------------------------------- report
def summarize(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run_id, g in blocks.groupby("run_id"):
        tr = g[g.block >= 0].sort_values("block")
        w, a = tr.W.astype(int).to_numpy(), tr.attack_active.to_numpy().astype(bool)
        first_active = int(np.argmax(a)) if a.any() else None
        on_after = np.flatnonzero(w[first_active:]) if first_active is not None else []
        has_trigger = tr.y_atk.notna().any()
        rows.append({
            "variant": tr.variant.iloc[0], "policy": tr.policy.iloc[0], "seed": int(tr.seed.iloc[0]),
            "final_clean_acc": tr.y_clean_acc.iloc[-1],
            "final_backdoor_success": tr.y_atk.iloc[-1] if has_trigger else np.nan,
            "attack_killed": bool(tr.y_atk.iloc[-1] < 0.1) if has_trigger else np.nan,
            "defended_blocks": int(w.sum()),
            "defended_active": int(w[a].sum()), "active_blocks": int(a.sum()),
            "false_alarm_blocks": int(w[~a].sum()),
            "reaction_delay_blocks": int(on_after[0]) if len(on_after) else np.nan,
        })
    per_run = pd.DataFrame(rows).sort_values(["variant", "policy", "seed"])
    metrics = ["final_clean_acc", "final_backdoor_success", "attack_killed", "defended_blocks", "defended_active",
               "active_blocks", "false_alarm_blocks", "reaction_delay_blocks"]
    agg = per_run.groupby(["variant", "policy"])[metrics].mean().round(4).reset_index()
    return per_run, agg


def verdicts(agg: pd.DataFrame) -> pd.DataFrame:
    """Per candidate attack: damage, prize of defending only when it matters, and whether the threshold defender catches it."""
    rows = []
    for v, g in agg.groupby("variant"):
        if v == "none":
            continue
        p = g.set_index("policy")
        rows.append({
            "variant": v,
            "damage: backdoor if never defended": p.loc["off", "final_backdoor_success"],
            "oracle: backdoor": p.loc["oracle", "final_backdoor_success"],
            "prize: oracle - always-on clean acc (pts)": 100 * (p.loc["oracle", "final_clean_acc"]
                                                               - p.loc["on", "final_clean_acc"]),
            "threshold defender: kill rate": p.loc["threshold_defender", "attack_killed"],
            "threshold defender: backdoor": p.loc["threshold_defender", "final_backdoor_success"],
            "threshold defender: false-alarm blocks": p.loc["threshold_defender", "false_alarm_blocks"],
            "threshold defender: active blocks defended": f"{p.loc['threshold_defender', 'defended_active']:.0f} / {p.loc['threshold_defender', 'active_blocks']:.0f}",
        })
    return pd.DataFrame(rows)


def fig_variant(blocks: pd.DataFrame, variant: str) -> None:
    g = blocks[(blocks.variant == variant) & (blocks.block >= 0)]
    has_trigger = g.y_atk.notna().any()
    metrics = ([("y_atk", "backdoor success")] if has_trigger else []) + [("y_clean_acc", "clean accuracy")]
    seeds = sorted(g.seed.unique())
    present = [p for p in DRAW_ORDER if p in set(g.policy)]
    fig, axes = plt.subplots(len(metrics), len(seeds), figsize=(3.4 * len(seeds), 2.6 * len(metrics) + 1.0),
                             sharex=True, sharey="row", squeeze=False)
    for j, seed in enumerate(seeds):
        s = g[g.seed == seed]
        ref = s[s.policy == present[0]].sort_values("block")
        active = ref.attack_active.to_numpy().astype(bool)
        defender = s[s.policy == "threshold_defender"].sort_values("block")
        for i, (col, label) in enumerate(metrics):
            ax = axes[i, j]
            for b in np.flatnonzero(active):
                ax.axvspan(b - 0.5, b + 0.5, color=INK2, alpha=0.10, linewidth=0)
            if len(defender):
                for b in np.flatnonzero(defender.W.astype(int).to_numpy()):
                    ax.axvspan(b - 0.5, b + 0.5, ymin=0.0, ymax=0.04, color=POLICY_COLOR["threshold_defender"], linewidth=0)
            for p in present:
                q = s[s.policy == p].sort_values("block")
                ax.plot(q.block, q[col], color=POLICY_COLOR[p], linewidth=2.0 if p == "threshold_defender" else 1.3,
                        zorder=3 if p == "threshold_defender" else 2)
            if col == "y_clean_acc":
                ax.set_ylim(max(0.0, g[g.block >= 10][col].min() - 0.02), min(1.0, g[col].max() + 0.01))
            if i == 0:
                ax.set_title(f"seed {seed}", color=INK)
            if j == 0:
                ax.set_ylabel(label)
            if i == len(metrics) - 1:
                ax.set_xlabel("block (2 SGD steps each)")
    handles = [Line2D([], [], color=POLICY_COLOR[p], linewidth=2.0 if p == "threshold_defender" else 1.3, label=POLICY_LABEL[p])
               for p in present]
    handles += [Patch(color=INK2, alpha=0.10, label="attack active"),
                Patch(color=POLICY_COLOR["threshold_defender"], label="blocks defended by the threshold defender (bottom strip)")]
    attack, pf, burst, rule, _, _ = VARIANTS[variant]
    desc = {"ref_win": "windowed backdoor (xp2)", "none": "no attack", "low_pf_0.25": "windowed backdoor, 2/8 poisoned",
            "low_pf_0.125": "windowed backdoor, 1/8 poisoned", "burst": f"backdoor bursts ({burst[1] if burst else ''} of "
            f"every {burst[0] if burst else ''} blocks)", "early": "windowed backdoor starting at blocks 2-8"}[variant]
    fig.suptitle(f"Pilot: {desc}", fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    top = 1 - 0.8 / fig.get_figheight()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, top + 0.05), ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, top - 0.02))
    fig.savefig(OUT / f"pilot_{variant}.png", dpi=130)
    plt.close(fig)


def report() -> None:
    blocks = pd.concat([pd.read_csv(f) for f in sorted((OUT / "runs").glob("*/blocks.csv"))], ignore_index=True)
    blocks.to_csv(OUT / "blocks_all.csv", index=False)
    per_run, agg = summarize(blocks)
    per_run.to_csv(OUT / "summary_runs.csv", index=False)
    agg.to_csv(OUT / "summary_by_policy.csv", index=False)
    v = verdicts(agg)
    v.to_csv(OUT / "candidate_attacks.csv", index=False)
    for variant in VARIANTS:
        if (blocks.variant == variant).any():
            fig_variant(blocks, variant)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(agg.to_string(index=False))
    print("\n## Candidate xp3 attacks")
    print(v.round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        jobs = [(v, p, s) for v, spec in VARIANTS.items() for p in spec[5] for s in spec[4]]
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
            futs = [pool.submit(run_one, v, p, s) for v, p, s in jobs]
            for k, fut in enumerate(as_completed(futs), 1):
                run_id, wall = fut.result()
                print(f"[{k}/{len(jobs)}] {run_id}  {wall:.0f}s  (elapsed {time.time() - started:.0f}s)", flush=True)
    report()


if __name__ == "__main__":
    main()
