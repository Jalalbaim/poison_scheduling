"""Deploy the xp2 GRU as a defense controller on windowed backdoor attacks, against always-on / always-off.

    python deploy_xp2.py                 # 3 policies x 5 fresh seeds, horizon 10 -> results/deploy_xp2
    python deploy_xp2.py --horizon 1 --out deploy_xp2_H1   # horizon 1; reuses always-on/off from deploy_xp2
    python deploy_xp2.py --report-only   # rebuild summary + figures from finished runs (same --out/--horizon)

Case: windowed backdoor + Krum, decisions every 2 rounds (250 blocks), as in xp2. Each seed gets a random
attack window drawn like xp2's (start in blocks 25-150, length 30-80). Seeds 1000-1004 were never seen by the
GRU (xp2 used seeds 0-259).

GRU policy: short-horizon receding control. Before each block b, the GRU reads the observable history (blocks
-1..b-1: X signals, W, defense; never outcomes, attack labels or the window) and predicts the outcomes at block
b+H-1 under 4 candidate plans for the next H blocks:
    on      defend the next H blocks        off     do not defend the next H blocks
    on1     defend this block only          off1    skip this block, then defend
It applies the first action of the plan with the best score
    score = predicted clean accuracy - LAMBDA * predicted backdoor success, both at block b+H-1.
A short horizon (rather than the end of the run) lets the controller wait: scored to the end of the run,
"never defend" would always look bad, because an attack may still start later.
"""

import argparse
import hashlib
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
from analyze import INK, INK2, SURFACE, X_COLS  # noqa: E402  (also applies the shared plot style)
from gru_experiment import Data, OutcomeGRU, assign_splits  # noqa: E402
from poison_testbed.runner import RunConfig, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parent
TRAIN_RUNS = ROOT / "results/xp2/runs_xp2/all_blocks.csv"
WEIGHTS = ROOT / "results/xp2/gru/gru.pt"
WEIGHTS_SHA = "5223e37fffd8c2d0"
OUT = ROOT / "results/deploy_xp2"

XP1_C, SIGMA = 3.0552712798118593, 0.01  # as in xp2
BLOCK_LEN, N_BLOCKS = 2, 250
N_ROWS = N_BLOCKS + 1
ATTACK, DEFENSE = "backdoor_win", "Krum"
SEEDS = [1000, 1001, 1002, 1003, 1004]
POLICIES = ["gru", "threshold", "oracle", "on", "off"]
# Threshold detector: defend block b iff the spread of worker-gradient norms in block b-1 exceeds TAU.
# TAU fitted on xp2 TRAINING runs (max TPR - FPR, attack onset + defended attack blocks vs no-attack blocks):
# train TPR 0.538 / FPR 0.0013; validation TPR 0.509 / FPR 0.0005, attack onset caught 80%.
# (An undefended attack is invisible in X once implanted, so only onset and defended blocks are detectable.)
THRESHOLD_SIGNAL, TAU = "x_norm_std", 3.65
PLANS = ["on", "off", "on1", "off1"]
HORIZON = 10  # blocks (20 rounds), about twice the ~10 rounds a backdoor needs to become permanent
LAMBDA = 1.0

# Validated categorical slots 1-3 of the reference palette (see analyze.py).
POLICY_COLOR = {"gru": "#2a78d6", "on": "#eb6834", "off": "#1baf7a", "threshold": "#eda100", "oracle": "#e87ba4"}
POLICY_LABEL = {"gru": "GRU policy", "on": "always on", "off": "always off",
                "threshold": "threshold detector", "oracle": "oracle (defends exactly the attack)"}
DRAW_ORDER = ["off", "on", "oracle", "threshold", "gru"]  # GRU drawn last, on top


def window_for(seed: int) -> tuple[int, int]:
    """Random attack window (blocks, inclusive), drawn like xp2's."""
    rng = np.random.default_rng(40_000 + seed)
    start = int(rng.integers(25, 151))
    return start, start + int(rng.integers(30, 81)) - 1


# ----------------------------------------------------------------------------- controller
class GRUController:
    """Stateless per call: re-encodes the observable history and scores the candidate plans."""

    def __init__(self, x_mean: np.ndarray, x_std: np.ndarray, device: str, horizon: int = HORIZON):
        self.x_mean, self.x_std, self.device = x_mean, x_std, torch.device(device)
        self.horizon = horizon
        probe = self._data([{"run_id": "probe", "seed": 0, "defense": DEFENSE, "block": -1, "W": np.nan}])
        self.model = OutcomeGRU(probe.enc.shape[-1], probe.dec_inputs(probe.W).shape[-1]).to(self.device)
        self.model.load_state_dict(torch.load(WEIGHTS, map_location=self.device))
        self.model.eval()

    def _data(self, history: list[dict]) -> Data:
        """Pad the history to N_ROWS rows (the encoder is causal: padding never affects the state we read)."""
        df = pd.DataFrame(history)
        last = int(df.block.max())
        pad = pd.DataFrame([{"run_id": df.run_id.iloc[0], "seed": df.seed.iloc[0], "defense": df.defense.iloc[0],
                             "block": b} for b in range(last + 1, N_BLOCKS)])
        df = pd.concat([df, pad], ignore_index=True)
        df["attack"] = "unknown"  # the controller never knows the attack
        for c in ["W", "y_clean_acc", "y_atk", *X_COLS]:  # outcomes are never in the history: all NaN
            if c not in df:
                df[c] = np.nan
        df["W"] = pd.to_numeric(df.W, errors="coerce")
        return Data(df, self.x_mean, self.x_std)

    @torch.no_grad()
    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        data = self._data(history)
        h = self.model.encode(data.enc.to(self.device))[:, b]  # state after rows 0..b = blocks -1..b-1
        H = min(self.horizon, N_BLOCKS - b)  # blocks b..b+H-1
        plans = {"on": [1] * H, "off": [0] * H, "on1": [1] + [0] * (H - 1), "off1": [0] + [1] * (H - 1)}
        # With H = 1, on1 = on and off1 = off: the comparison is simply "defend this block or not".
        W = np.zeros((len(PLANS), N_ROWS))
        for k, p in enumerate(PLANS):
            W[k, b + 1: b + 1 + H] = plans[p]  # row i = block i-1
        dec = data.dec_inputs(W)[:, b: b + H]  # decoder step for row i+1 is dec index i -> blocks b..b+H-1
        pred = self.model.decode(h.expand(len(PLANS), -1), dec.to(self.device)).cpu().numpy()  # (4, H, 2)
        end = pred[:, -1]
        score = end[:, 0] - LAMBDA * end[:, 1]
        best = int(np.argmax(score))
        extra = {"gru_plan": PLANS[best]}
        for k, p in enumerate(PLANS):
            extra[f"gru_pred_clean_h_{p}"] = float(end[k, 0])
            extra[f"gru_pred_atk_h_{p}"] = float(end[k, 1])
            extra[f"gru_score_{p}"] = float(score[k])
        return plans[PLANS[best]][0], extra


class ThresholdController:
    """Defend block b iff the observable signal of block b-1 exceeds TAU (no decision data at block 0: off)."""

    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        value = history[-1].get(THRESHOLD_SIGNAL)
        value = float("nan") if value is None else float(value)
        return int(value > TAU), {"thr_signal": value, "thr_tau": TAU}


# ----------------------------------------------------------------------------- runs
def config(policy: str, seed: int) -> RunConfig:
    # gru / threshold decide online; oracle defends exactly the attack window; on / off are fixed.
    schedule = {"gru": "policy", "threshold": "policy", "oracle": "react_d0_e0"}.get(policy, policy)
    return RunConfig(attack=ATTACK, defense=DEFENSE, seed=seed, C=XP1_C, sigma=SIGMA, schedule=schedule,
                     block_len=BLOCK_LEN, n_blocks=N_BLOCKS, window=window_for(seed))


def run_one(policy: str, seed: int, x_mean: np.ndarray, x_std: np.ndarray, horizon: int, out_root: Path,
            fixed_from: Path | None) -> tuple[str, float]:
    """Horizon and output folder are passed explicitly: spawned workers re-import this module."""
    cfg = config(policy, seed)
    run_id = f"{ATTACK}_{DEFENSE}__{policy}__s{seed}"
    out = out_root / "runs" / run_id
    if (out / "blocks.csv").exists():
        return run_id, json.loads((out / "config.json").read_text())["wall_clock_s"]
    if policy != "gru" and fixed_from is not None and (fixed_from / "runs" / run_id / "blocks.csv").exists():
        # Non-GRU policies do not depend on the horizon: reuse the finished run (same config, same GPU).
        src = fixed_from / "runs" / run_id
        meta = json.loads((src / "config.json").read_text())
        if meta["config"] != json.loads(json.dumps(cfg.__dict__, default=str)):
            raise SystemExit(f"{src} has a different config; refusing to reuse it")
        out.mkdir(parents=True, exist_ok=True)
        (out / "blocks.csv").write_bytes((src / "blocks.csv").read_bytes())
        (out / "config.json").write_text(json.dumps({**meta, "horizon_blocks": None, "copied_from": str(src)},
                                                    indent=2, default=str))
        return run_id, meta["wall_clock_s"]
    controller = (GRUController(x_mean, x_std, cfg.device, horizon) if policy == "gru"
                  else ThresholdController() if policy == "threshold" else None)
    started = time.time()
    df = simulate(cfg, controller)
    wall = time.time() - started
    df.insert(0, "policy", policy)
    df["run_id"] = run_id
    df["window_start"], df["window_end"] = cfg.window  # evaluation label (never seen by the controller)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "blocks.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "policy": policy, "config": cfg.__dict__, "lambda": LAMBDA,
        "horizon_blocks": horizon if policy == "gru" else None,
        "plans": PLANS, "gru_weights_sha256": WEIGHTS_SHA, "wall_clock_s": round(wall, 1),
        "device_name": torch.cuda.get_device_name(0) if cfg.device == "cuda" else cfg.device,
    }, indent=2, default=str))
    return run_id, wall


# ----------------------------------------------------------------------------- report
def summarize(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run_id, g in blocks.groupby("run_id"):
        tr = g[g.block >= 0].sort_values("block")
        w = tr.W.astype(int).to_numpy()
        s, e = int(tr.window_start.iloc[0]), int(tr.window_end.iloc[0])
        inside = np.zeros(len(w), bool); inside[s: e + 1] = True
        defended_in = np.flatnonzero(w[s: e + 1])
        rows.append({
            "policy": tr.policy.iloc[0], "seed": int(tr.seed.iloc[0]), "run_id": run_id,
            "window_start": s, "window_end": e,
            # standard metrics first
            "final_clean_acc": tr.y_clean_acc.iloc[-1], "final_backdoor_success": tr.y_atk.iloc[-1],
            "backdoor_success_window_end": tr.y_atk.iloc[e],
            "peak_backdoor_success": tr.y_atk.max(),
            "attack_killed": bool(tr.y_atk.iloc[-1] < 0.1),
            "mean_clean_acc": tr.y_clean_acc.mean(),
            # defense usage
            "defended_blocks": int(w.sum()),
            "defended_in_window": int(w[inside].sum()), "window_blocks": int(inside.sum()),
            "defended_outside_window": int(w[~inside].sum()),
            "reaction_delay_blocks": int(defended_in[0]) if len(defended_in) else np.nan,
            "switches": int(np.abs(np.diff(w)).sum()),
            # complementary: the controller's objective evaluated on the true final outcomes
            "score_true": tr.y_clean_acc.iloc[-1] - LAMBDA * tr.y_atk.iloc[-1],
        })
    per_run = pd.DataFrame(rows).sort_values(["policy", "seed"])
    metrics = ["final_clean_acc", "final_backdoor_success", "backdoor_success_window_end", "peak_backdoor_success",
               "attack_killed", "mean_clean_acc", "defended_blocks", "defended_in_window",
               "defended_outside_window", "reaction_delay_blocks", "switches", "score_true"]
    by_policy = per_run.groupby("policy")[metrics].agg(["mean", "std"]).round(4)
    by_policy.columns = [f"{m}_{s}" for m, s in by_policy.columns]
    return per_run, by_policy.reindex(POLICIES).reset_index()


def legend_handles(present: set[str]) -> list:
    handles = [Line2D([], [], color=POLICY_COLOR[p], linewidth=2.0 if p == "gru" else 1.4, label=POLICY_LABEL[p])
               for p in DRAW_ORDER if p in present]
    return handles + [
        Patch(color=POLICY_COLOR["gru"], alpha=0.15, label="block defended by GRU"),
        Line2D([], [], color=INK2, linestyle=(0, (3, 2)), linewidth=1.0, label="attack window (start / end)"),
    ]


def policy_grid(blocks: pd.DataFrame, metrics: list[tuple[str, str]], title: str, path: Path,
                xlim: tuple[int, int] | None = None) -> None:
    """One column per seed, one row per metric; GRU-defended blocks shaded, attack window marked."""
    seeds = sorted(blocks.seed.unique())
    present = set(blocks.policy)
    fig, axes = plt.subplots(len(metrics), len(seeds), figsize=(3.4 * len(seeds), 2.7 * len(metrics) + 1.0),
                             sharex=xlim is None, sharey="row", squeeze=False)
    for j, seed in enumerate(seeds):
        s = blocks[blocks.seed == seed]
        gru = s[(s.policy == "gru") & (s.block >= 0)].sort_values("block")
        w = gru.W.astype(int).to_numpy()
        ws, we = int(gru.window_start.iloc[0]), int(gru.window_end.iloc[0])
        for i, (col, label) in enumerate(metrics):
            ax = axes[i, j]
            for b in np.flatnonzero(w):
                ax.axvspan(b - 0.5, b + 0.5, color=POLICY_COLOR["gru"], alpha=0.15, linewidth=0)
            for x in (ws - 0.5, we + 0.5):
                ax.axvline(x, color=INK2, linestyle=(0, (3, 2)), linewidth=1.0, zorder=1)
            for policy in [p for p in DRAW_ORDER if p in present]:
                p = s[s.policy == policy].sort_values("block")
                ax.plot(p.block, p[col], color=POLICY_COLOR[policy], linewidth=2.0 if policy == "gru" else 1.3,
                        zorder=3 if policy == "gru" else 2)
            if xlim:
                ax.set_xlim(ws + xlim[0], ws + xlim[1])
            if i == 0:
                ax.set_title(f"seed {seed}: attack in blocks {ws}-{we}\nGRU defended {w.sum()} / {len(w)} blocks",
                             color=INK)
            if j == 0:
                ax.set_ylabel(label)
            if i == len(metrics) - 1:
                ax.set_xlabel("block (2 SGD steps each)")
            if col == "y_clean_acc":
                ax.set_ylim(max(0.0, blocks[blocks.block >= 10][col].min() - 0.02), min(1.0, blocks[col].max() + 0.01))
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    top = 1 - 0.8 / fig.get_figheight()
    fig.legend(handles=legend_handles(present), loc="upper left", bbox_to_anchor=(0.01, top + 0.045), ncol=7,
               fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, top - 0.02))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_decisions(blocks: pd.DataFrame) -> None:
    """Around the attack onset: what the GRU predicted for its two main plans, and what it chose."""
    g = blocks[(blocks.policy == "gru") & (blocks.block >= 0)]
    seeds = sorted(g.seed.unique())
    fig, axes = plt.subplots(2, len(seeds), figsize=(3.4 * len(seeds), 5.6), sharey="row", squeeze=False)
    for j, seed in enumerate(seeds):
        s = g[g.seed == seed].sort_values("block")
        ws, we = int(s.window_start.iloc[0]), int(s.window_end.iloc[0])
        lo, hi = ws - 15, ws + 30
        z = s[(s.block >= lo) & (s.block <= hi)]
        ax = axes[0, j]
        ax.plot(z.block, z.gru_pred_atk_h_off, color=POLICY_COLOR["off"], label=f"predicted if no defense ({HORIZON} blocks)")
        ax.plot(z.block, z.gru_pred_atk_h_on, color=POLICY_COLOR["on"], label=f"predicted if defended ({HORIZON} blocks)")
        ax.plot(z.block, z.y_atk, color=POLICY_COLOR["gru"], linewidth=2.0, label="actual (GRU policy)")
        ax.axvline(ws - 0.5, color=INK2, linestyle=(0, (3, 2)), linewidth=1.0)
        ax.set_title(f"seed {seed}: attack starts at block {ws}", color=INK)
        ax.set_ylim(-0.03, 1.03)
        ax2 = axes[1, j]
        ax2.step(z.block, z.W, where="mid", color=POLICY_COLOR["gru"], linewidth=2.0)
        ax2.axvline(ws - 0.5, color=INK2, linestyle=(0, (3, 2)), linewidth=1.0)
        ax2.set_yticks([0, 1], ["off", "on"])
        ax2.set_xlabel("block (2 SGD steps each)")
        first_on = z[(z.block >= ws) & (z.W == 1)].block
        if len(first_on):
            ax2.set_title(f"GRU defends from block {int(first_on.iloc[0])} "
                          f"(+{int(first_on.iloc[0]) - ws} blocks)", color=INK)
    axes[0, 0].set_ylabel("backdoor success")
    axes[1, 0].set_ylabel("GRU decision")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.96), ncol=3, fontsize=8)
    fig.suptitle("GRU decisions around the attack onset: predicted backdoor success under each plan, and the action",
                 fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "decisions_onset.png", dpi=130)
    plt.close(fig)


def fig_finals(per_run: pd.DataFrame) -> None:
    panels = [("final_clean_acc", "final clean accuracy"), ("final_backdoor_success", "final backdoor success"),
              ("defended_blocks", "defended blocks (of 250)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.4), squeeze=False)
    for ax, (col, label) in zip(axes[0], panels):
        for k, policy in enumerate(POLICIES):
            v = per_run[per_run.policy == policy][col].to_numpy(float)
            ax.scatter(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, s=22, color=POLICY_COLOR[policy],
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
            ax.plot([k - 0.25, k + 0.25], [v.mean()] * 2, color=INK, linewidth=2)
        ax.set_xticks(range(len(POLICIES)), [POLICY_LABEL[p] for p in POLICIES])
        ax.set_title(label, color=INK, loc="left")
    fig.suptitle(f"Windowed backdoor + Krum: final outcomes per policy (dots: seeds {SEEDS[0]}-{SEEDS[-1]}; "
                 f"bar: mean)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "finals_by_policy.png", dpi=130)
    plt.close(fig)


def report() -> None:
    files = sorted((OUT / "runs").glob("*/blocks.csv"))
    blocks = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    blocks.to_csv(OUT / "blocks_all.csv", index=False)
    per_run, by_policy = summarize(blocks)
    per_run.to_csv(OUT / "summary_runs.csv", index=False)
    by_policy.to_csv(OUT / "summary_by_policy.csv", index=False)
    title = (f"Windowed backdoor + Krum: GRU policy (horizon {HORIZON} block{'s' * (HORIZON > 1)}) vs threshold, oracle, always-on, "
             f"always-off (same seed = same data and noise)")
    policy_grid(blocks, [("y_clean_acc", "clean accuracy"), ("y_atk", "backdoor success")], title,
                OUT / "trajectory.png")
    policy_grid(blocks, [("y_atk", "attack success\n(backdoor success rate)")],
                f"Windowed backdoor + Krum: attack success under the GRU policy (horizon {HORIZON} "
                f"block{'s' * (HORIZON > 1)}) vs threshold, oracle, always-on, always-off", OUT / "attack_success.png")
    fig_decisions(blocks)
    fig_finals(per_run)
    pd.set_option("display.width", 220)
    cols = ["policy", "final_clean_acc_mean", "final_backdoor_success_mean", "backdoor_success_window_end_mean",
            "peak_backdoor_success_mean", "attack_killed_mean", "defended_blocks_mean", "defended_in_window_mean",
            "defended_outside_window_mean", "reaction_delay_blocks_mean", "switches_mean"]
    print(by_policy[cols].to_string(index=False))


def main() -> None:
    global OUT, HORIZON  # main process only (report + figures); workers receive both explicitly
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--horizon", type=int, default=HORIZON, help="GRU planning horizon in blocks")
    ap.add_argument("--out", default="deploy_xp2", help="output folder name under results/")
    ap.add_argument("--fixed-from", default="deploy_xp2",
                    help="results/ folder whose always-on/off runs are reused (they do not depend on the horizon)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT, HORIZON = ROOT / "results" / args.out, args.horizon
    fixed_from = ROOT / "results" / args.fixed_from
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()[:16]
        if sha != WEIGHTS_SHA:
            raise SystemExit(f"GRU weights changed: sha {sha}, expected the frozen {WEIGHTS_SHA}")
        train = pd.read_csv(TRAIN_RUNS)
        stats = Data(train[assign_splits(train) == "train"])  # the GRU's own input normalisation
        jobs = [(p, s) for p in POLICIES for s in SEEDS]
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
            futs = [pool.submit(run_one, p, s, stats.x_mean, stats.x_std, HORIZON, OUT,
                                fixed_from if fixed_from != OUT else None) for p, s in jobs]
            for k, fut in enumerate(as_completed(futs), 1):
                run_id, wall = fut.result()
                print(f"[{k}/{len(jobs)}] {run_id}  {wall:.0f}s  (elapsed {time.time() - started:.0f}s)", flush=True)
    report()


if __name__ == "__main__":
    main()
