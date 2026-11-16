"""Deploy the xp3 GRU as a probing defense controller, against the fixed evaluation protocol.

    python deploy_xp3.py --gru results/xp3/gru      # all policies x all environments -> results/deploy_xp3
    python deploy_xp3.py --report-only

Environments (seeds never seen in training: xp3 used 10000-10319). Every attack is the xp3 windowed backdoor with
Krum as the defense, decisions every 2 rounds (250 blocks):
    pf1/8, pf2/8, pf8/8   1, 2 or 8 of the 8 images in each poisoned micro-batch are patched (f = 3 of 16)
    none                  no attack (false alarms)
4 seeds per condition: 2 with an early window (start in blocks 2-20), 2 with a late one (21-150); length 30-80.

Policies (same seed = same data, same noise, same attack):
    off / on              never / always defend
    oracle                defends exactly the attack window (knows the attack; reference only)
    threshold_defender    defend when x_norm_std of the previous block > TAU; hold until K low blocks
    probe_threshold       threshold defender + a probe (one defended block) every P blocks
    gru                   probe every P blocks by default; before each block, the GRU predicts the outcomes at
                          block b+H-1 under two plans for the next H blocks: "keep probing" or "defend", and
                          applies the first action of the better plan (score = clean - LAMBDA * backdoor).

Why probes: a stealthy backdoor (1/8, 2/8) is invisible in X while undefended and obvious while defended, and Krum
still erases it if switched on within ~10 blocks (2/8) or ~20 blocks (1/8) (measure_erasure.py). A one-block probe
every P = 5 blocks looks often enough, at ~20% defended blocks.
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
TRAIN_RUNS = ROOT / "results/xp3/runs_xp3/all_blocks.csv"
OUT = ROOT / "results/deploy_xp3"

XP1_C, SIGMA, BLOCK_LEN, N_BLOCKS = 3.0552712798118593, 0.01, 2, 250
N_ROWS = N_BLOCKS + 1
DEFENSE = "Krum"
CONDITIONS = {"pf1/8": ("backdoor_win", 0.125), "pf2/8": ("backdoor_win", 0.25), "pf8/8": ("backdoor_win", 1.0),
              "none": ("none", 1.0)}
SEEDS_PER_CONDITION = 4  # k = 0, 1: early window; k = 2, 3: late window
POLICIES = ["off", "on", "oracle", "threshold_defender", "probe_threshold", "gru"]

SIGNAL, TAU, K_HOLD = "x_norm_std", 3.65, 12  # threshold defender, as in pilot_xp3 (TAU fitted on xp2 train runs)
PROBE_PERIOD = 5  # blocks between probes (one defended block each)
HORIZON = 20      # blocks (40 rounds): longer than the slowest erasure window (~20 rounds at 1/8)
LAMBDA = 1.0

POLICY_COLOR = {"off": "#1baf7a", "on": "#eb6834", "oracle": "#e87ba4", "threshold_defender": "#eda100",
                "probe_threshold": "#008300", "gru": "#2a78d6"}
POLICY_LABEL = {"off": "always off", "on": "always on", "oracle": "oracle (defends exactly the attack)",
                "threshold_defender": "threshold defender", "probe_threshold": f"threshold defender + probe every {PROBE_PERIOD}",
                "gru": "GRU (probes + decides when to hold)"}
DRAW_ORDER = ["off", "on", "oracle", "threshold_defender", "probe_threshold", "gru"]


def environment(cond: str, k: int) -> tuple[int, tuple[int, int]]:
    """Seed and attack window (blocks, inclusive) of environment k of a condition. No-attack runs get a
    window too (it is never used by the attack; it only defines the 'false alarm' blocks as all blocks)."""
    seed = 5000 + 100 * list(CONDITIONS).index(cond) + k
    rng = np.random.default_rng(60_000 + seed)
    start = int(rng.integers(2, 21)) if k < SEEDS_PER_CONDITION // 2 else int(rng.integers(21, 151))
    return seed, (start, start + int(rng.integers(30, 81)) - 1)


def is_probe(b: int) -> bool:
    return b % PROBE_PERIOD == 0


# ----------------------------------------------------------------------------- controllers
class ThresholdDefender:
    """Defend when the previous block's signal exceeds TAU; stay on until it has been low for K_HOLD blocks.
    With probe=True, also defend one block every PROBE_PERIOD blocks (so stealthy attacks become visible)."""

    def __init__(self, probe: bool = False):
        self.probe, self.on, self.low = probe, False, 0

    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        value = history[-1].get(SIGNAL)
        value = float("nan") if value is None else float(value)
        if value > TAU:
            self.on, self.low = True, 0
        elif self.on:
            self.low += 1
            if self.low >= K_HOLD:
                self.on, self.low = False, 0
        probe = self.probe and is_probe(b) and not self.on
        return int(self.on or probe), {"det_signal": value, "det_hold": int(self.on), "probe": int(probe)}


class GRUProbeController:
    """Default: keep probing. Before each block, compare 'keep probing' with 'defend' over the next H blocks."""

    def __init__(self, weights: Path, x_mean: np.ndarray, x_std: np.ndarray, device: str, horizon: int = HORIZON):
        self.x_mean, self.x_std, self.device, self.horizon = x_mean, x_std, torch.device(device), horizon
        probe = self._data([{"run_id": "p", "seed": 0, "defense": DEFENSE, "block": -1, "W": np.nan}])
        hidden = json.loads((weights.parent / "report.json").read_text()).get("hparams", {}).get("hidden", 64)
        self.model = OutcomeGRU(probe.enc.shape[-1], probe.dec_inputs(probe.W).shape[-1], hidden).to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
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
        H = min(self.horizon, N_BLOCKS - b)
        plans = {"probe": [int(is_probe(k)) for k in range(b, b + H)], "defend": [1] * H}
        W = np.zeros((2, N_ROWS))
        for k, p in enumerate(plans.values()):
            W[k, b + 1: b + 1 + H] = p  # row i = block i-1
        dec = data.dec_inputs(W)[:, b: b + H]
        pred = self.model.decode(h.expand(2, -1), dec.to(self.device)).cpu().numpy()[:, -1]  # (2, 2) at b+H-1
        score = pred[:, 0] - LAMBDA * pred[:, 1]
        choice = "defend" if score[1] > score[0] else "probe"
        return plans[choice][0], {
            "gru_choice": choice, "probe": int(choice == "probe" and plans["probe"][0] == 1),
            "gru_pred_clean_probe": float(pred[0, 0]), "gru_pred_atk_probe": float(pred[0, 1]),
            "gru_pred_clean_defend": float(pred[1, 0]), "gru_pred_atk_defend": float(pred[1, 1]),
            "gru_gain_defend": float(score[1] - score[0])}


# ----------------------------------------------------------------------------- runs
def config(cond: str, policy: str, k: int) -> RunConfig:
    attack, pf = CONDITIONS[cond]
    seed, window = environment(cond, k)
    if policy == "oracle":
        schedule = "react_d0_e0" if attack != "none" else "off"  # no attack: the oracle never defends
    else:
        schedule = policy if policy in ("on", "off") else "policy"
    return RunConfig(attack=attack, defense=DEFENSE, seed=seed, C=XP1_C, sigma=SIGMA, schedule=schedule,
                     block_len=BLOCK_LEN, n_blocks=N_BLOCKS, window=window, poison_frac=pf,
                     always_measure_backdoor=True)


def run_id_of(cond: str, policy: str, k: int) -> str:
    return f"{cond.replace('/', 'of')}__{policy}__s{environment(cond, k)[0]}"


def run_one(cond: str, policy: str, k: int, weights: str, x_mean: np.ndarray, x_std: np.ndarray,
            out_root: str) -> tuple[str, float]:
    cfg = config(cond, policy, k)
    run_id = run_id_of(cond, policy, k)
    out = Path(out_root) / "runs" / run_id
    if (out / "blocks.csv").exists():
        return run_id, json.loads((out / "config.json").read_text())["wall_clock_s"]
    controller = {"threshold_defender": lambda: ThresholdDefender(False),
                  "probe_threshold": lambda: ThresholdDefender(True),
                  "gru": lambda: GRUProbeController(Path(weights), x_mean, x_std, cfg.device)}.get(policy, lambda: None)()
    started = time.time()
    df = simulate(cfg, controller)
    wall = time.time() - started
    df.insert(0, "condition", cond)
    df.insert(1, "policy", policy)
    df["run_id"] = run_id
    df["window_start"], df["window_end"] = cfg.window  # evaluation labels (never seen by a controller)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "blocks.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "condition": cond, "policy": policy, "config": cfg.__dict__,
        "gru_weights": weights if policy == "gru" else None,
        "gru_weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest()[:16] if policy == "gru" else None,
        "lambda": LAMBDA, "horizon_blocks": HORIZON, "probe_period": PROBE_PERIOD, "tau": TAU, "k_hold": K_HOLD,
        "wall_clock_s": round(wall, 1),
        "device_name": torch.cuda.get_device_name(0) if cfg.device == "cuda" else cfg.device,
    }, indent=2, default=str))
    return run_id, wall


# ----------------------------------------------------------------------------- report
def summarize(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run_id, g in blocks.groupby("run_id"):
        tr = g[g.block >= 0].sort_values("block")
        w = tr.W.astype(int).to_numpy()
        attacked = tr.condition.iloc[0] != "none"
        active = tr.u_active.to_numpy().astype(bool)
        s = int(tr.window_start.iloc[0])
        sustained = np.flatnonzero(w[s:-1] & w[s + 1:]) if attacked else []  # two defended blocks in a row
        rows.append({
            "condition": tr.condition.iloc[0], "policy": tr.policy.iloc[0], "seed": int(tr.seed.iloc[0]),
            "early_window": s <= 20, "window_start": s, "window_end": int(tr.window_end.iloc[0]),
            # standard metrics first
            "final_clean_acc": tr.y_clean_acc.iloc[-1], "final_backdoor_success": tr.y_atk.iloc[-1],
            "attack_killed": bool(tr.y_atk.iloc[-1] < 0.1) if attacked else np.nan,
            "peak_backdoor_success": tr.y_atk.max(),
            # defense usage
            "defended_blocks": int(w.sum()), "defended_in_window": int(w[active].sum()) if attacked else 0,
            "false_alarm_blocks": int(w[~active].sum()),  # defended while no attack is active (incl. probes)
            "probe_blocks": int(tr.probe.sum()) if "probe" in tr else 0,
            "hold_delay_blocks": int(sustained[0]) if len(sustained) else np.nan,
        })
    per_run = pd.DataFrame(rows)
    on = per_run[per_run.policy == "on"].set_index("seed").final_clean_acc
    per_run["clean_vs_on_pts"] = 100 * (per_run.final_clean_acc - per_run.seed.map(on))
    per_run["score_true"] = per_run.final_clean_acc - LAMBDA * per_run.final_backdoor_success  # complementary
    per_run = per_run.sort_values(["condition", "policy", "seed"])
    metrics = ["final_clean_acc", "final_backdoor_success", "attack_killed", "clean_vs_on_pts", "defended_blocks",
               "false_alarm_blocks", "probe_blocks", "hold_delay_blocks", "score_true"]
    agg = per_run.groupby(["condition", "policy"])[metrics].mean().round(4).reset_index()
    agg["policy"] = pd.Categorical(agg.policy, POLICIES)
    return per_run, agg.sort_values(["condition", "policy"])


def fig_condition(blocks: pd.DataFrame, cond: str, out: Path) -> None:
    """Rows: backdoor success, clean accuracy, decisions per controller. Columns: the condition's seeds."""
    g = blocks[(blocks.condition == cond) & (blocks.block >= 0)]
    seeds = sorted(g.seed.unique())
    present = [p for p in DRAW_ORDER if p in set(g.policy)]
    deciders = [p for p in ["gru", "probe_threshold", "threshold_defender", "oracle"] if p in present]
    fig, axes = plt.subplots(3, len(seeds), figsize=(3.6 * len(seeds), 8.4), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1, 0.55]})
    for j, seed in enumerate(seeds):
        s = g[g.seed == seed]
        ref = s[s.policy == present[0]].sort_values("block")
        ws, we = int(ref.window_start.iloc[0]), int(ref.window_end.iloc[0])
        gw = s[s.policy == "gru"].sort_values("block").W.astype(int).to_numpy() if "gru" in present else []
        for i, (col, label) in enumerate([("y_atk", "backdoor success"), ("y_clean_acc", "clean accuracy")]):
            ax = axes[i, j]
            if cond != "none":
                ax.axvspan(ws - 0.5, we + 0.5, color=INK2, alpha=0.08, linewidth=0)
            for b in np.flatnonzero(gw):
                ax.axvspan(b - 0.5, b + 0.5, ymin=0, ymax=0.035, color=POLICY_COLOR["gru"], linewidth=0)
            for p in present:
                q = s[s.policy == p].sort_values("block")
                ax.plot(q.block, q[col], color=POLICY_COLOR[p], linewidth=2.0 if p == "gru" else 1.2,
                        zorder=3 if p == "gru" else 2)
            if col == "y_atk":
                ax.set_ylim(-0.03, 1.03)
                gru_run = s[s.policy == "gru"]
                ttl = f"seed {seed}: " + (f"attack in blocks {ws}-{we}" if cond != "none" else "no attack")
                if len(gru_run):
                    ttl += f"\nGRU defended {int(gru_run[gru_run.block >= 0].W.sum())} / {N_BLOCKS} blocks"
                ax.set_title(ttl, color=INK)
            else:
                ax.set_ylim(max(0.0, g[g.block >= 10][col].min() - 0.02), min(1.0, g[col].max() + 0.01))
            if j == 0:
                ax.set_ylabel(label)
        ax = axes[2, j]  # decision raster: one line per controller
        if cond != "none":
            ax.axvspan(ws - 0.5, we + 0.5, color=INK2, alpha=0.08, linewidth=0)
        for r, p in enumerate(deciders):
            q = s[s.policy == p].sort_values("block")
            on = q.block[q.W.astype(int) == 1].to_numpy()
            ax.broken_barh([(b - 0.5, 1) for b in on], (r - 0.35, 0.7), color=POLICY_COLOR[p], linewidth=0)
        ax.set_yticks(range(len(deciders)), [POLICY_LABEL[p].split(" (")[0] for p in deciders] if j == 0 else [])
        ax.set_ylim(-0.6, len(deciders) - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel("block (2 SGD steps each)")
        ax.grid(False)
    axes[2, 0].set_ylabel("defended blocks")
    handles = [Line2D([], [], color=POLICY_COLOR[p], linewidth=2.0 if p == "gru" else 1.3, label=POLICY_LABEL[p])
               for p in present]
    if cond != "none":
        handles.append(Patch(color=INK2, alpha=0.15, label="attack active"))
    desc = {"pf1/8": "1 of 8 images poisoned", "pf2/8": "2 of 8 images poisoned", "pf8/8": "8 of 8 images poisoned",
            "none": "no attack"}[cond]
    fig.suptitle(f"Deployment, windowed backdoor + Krum, {desc}: trajectories and defense decisions "
                 f"(same seed = same data and noise)", fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    top = 1 - 0.75 / fig.get_figheight()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, top + 0.04), ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, top - 0.03))
    fig.savefig(out / f"trajectory_{cond.replace('/', 'of')}.png", dpi=130)
    plt.close(fig)


def fig_summary(per_run: pd.DataFrame, out: Path) -> None:
    """Per condition: final backdoor success, clean accuracy vs always-on, defended blocks; one dot per seed."""
    panels = [("final_backdoor_success", "final backdoor success"), ("clean_vs_on_pts", "final clean acc. − always-on (pts)"),
              ("defended_blocks", f"defended blocks (of {N_BLOCKS})")]
    conds = list(CONDITIONS)
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 8.6), sharex=True)
    width = 0.8 / len(POLICIES)
    for ax, (col, label) in zip(axes, panels):
        for c, cond in enumerate(conds):
            for k, p in enumerate(POLICIES):
                v = per_run[(per_run.condition == cond) & (per_run.policy == p)][col].to_numpy(float)
                if not len(v):
                    continue
                x = c - 0.4 + width * (k + 0.5)
                ax.scatter(np.full(len(v), x), v, s=16, color=POLICY_COLOR[p], edgecolor=SURFACE, linewidth=0.6, zorder=3)
                ax.plot([x - width * 0.4, x + width * 0.4], [np.nanmean(v)] * 2, color=POLICY_COLOR[p], linewidth=2.2)
        ax.set_ylabel(label)
        if col == "clean_vs_on_pts":
            ax.axhline(0, color=INK2, linewidth=0.8)
    axes[-1].set_xticks(range(len(conds)), [c if c != "none" else "no attack" for c in conds])
    handles = [Line2D([], [], color=POLICY_COLOR[p], marker="o", linewidth=2, label=POLICY_LABEL[p]) for p in POLICIES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, fontsize=8)
    fig.suptitle("Deployment summary: dots = seeds (2 early + 2 late windows per condition), bar = mean",
                 fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "summary_by_policy.png", dpi=130)
    plt.close(fig)


def report(out: Path) -> None:
    blocks = pd.concat([pd.read_csv(f) for f in sorted((out / "runs").glob("*/blocks.csv"))], ignore_index=True)
    blocks.to_csv(out / "blocks_all.csv", index=False)
    per_run, agg = summarize(blocks)
    per_run.to_csv(out / "summary_runs.csv", index=False)
    agg.to_csv(out / "summary_by_policy.csv", index=False)
    for cond in CONDITIONS:
        if (blocks.condition == cond).any():
            fig_condition(blocks, cond, out)
    fig_summary(per_run, out)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(agg.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gru", type=Path, default=ROOT / "results/xp3/gru", help="folder with gru.pt + report.json")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--policies", nargs="*", default=POLICIES)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=SEEDS_PER_CONDITION, help="environments per condition (smoke tests)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        weights = (args.gru / "gru.pt").resolve()
        train = pd.read_csv(TRAIN_RUNS)
        stats = Data(train[assign_splits(train) == "train"])  # the GRU's own input normalisation
        jobs = [(c, p, k) for p in args.policies for c in args.conditions for k in range(args.seeds)]
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
            futs = [pool.submit(run_one, c, p, k, str(weights), stats.x_mean, stats.x_std, str(args.out))
                    for c, p, k in jobs]
            for n, fut in enumerate(as_completed(futs), 1):
                run_id, wall = fut.result()
                print(f"[{n}/{len(jobs)}] {run_id}  {wall:.0f}s  (elapsed {time.time() - started:.0f}s)", flush=True)
    report(args.out)


if __name__ == "__main__":
    main()
