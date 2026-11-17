"""Deploy controllers on the xp4 setup (signals v2) under the fixed evaluation protocol, and plot trajectories.

    python deploy_xp4.py                 # 7 policies x 4 conditions x 4 seeds -> results/deploy_xp4
    python deploy_xp4.py --report-only

Environments: windowed backdoor + Krum, decisions every 2 rounds (250 blocks), signals v2, seeds never seen in
training (xp4 used 20000-20319): 1/8, 2/8, 8/8 poisoned, or no attack; 4 seeds each (2 early, 2 late windows).

Policies (same seed = same data, same noise, same attack):
    off / on          never / always defend
    oracle            defends exactly the attack window (knows the attack; reference only)
    threshold_v1      the threshold defender of xp3 (x_norm_std > 3.65, hold until 12 low blocks)
    threshold_v2      alarm when x2_outlier_proj_max (block max) of the previous block exceeds the 99.5th percentile
                      of no-attack training blocks in the same defense state and 50-block time bin (fitted on xp4
                      training seeds); hold until K_HOLD_V2 blocks without alarm
    gru               detect with threshold_v2's alarm; then, while defending, the GRU decides each block whether
                      to keep defending: gain = score(defend H blocks) - score(release for H blocks) > DELTA_RELEASE
    gru_free          the GRU alone: defend block b iff score(defend H) - score(wait H) > DELTA_FREE
score = predicted clean accuracy - LAMBDA * predicted backdoor success at block b+H-1. The margins DELTA and the
models were chosen on xp4 validation seeds (check_probe.py: AUC, then the Youden point), never on test seeds or
on these deployment seeds.
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
from analyze import INK, INK2, SURFACE  # noqa: E402  (also applies the shared plot style)
from gru_experiment import Data, OutcomeGRU, assign_splits, x_columns  # noqa: E402
from poison_testbed.runner import RunConfig, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parent
TRAIN_RUNS = ROOT / "results/xp4/runs_xp4/all_blocks.csv"
OUT = ROOT / "results/deploy_xp4"
MARGINS = json.loads((ROOT / "results/xp4/gru_margins.json").read_text())
GRU_RELEASE = ROOT / f"results/xp4/gru_sweep/{MARGINS['release_defended']['model']}"
GRU_FREE = ROOT / f"results/xp4/gru_sweep/{MARGINS['release_all']['model']}"
DELTA_RELEASE, DELTA_FREE = MARGINS["release_defended"]["delta"], MARGINS["release_all"]["delta"]

XP1_C, SIGMA, BLOCK_LEN, N_BLOCKS = 3.0552712798118593, 0.01, 2, 250
N_ROWS = N_BLOCKS + 1
DEFENSE = "Krum"
CONDITIONS = {"pf1/8": ("backdoor_win", 0.125), "pf2/8": ("backdoor_win", 0.25), "pf8/8": ("backdoor_win", 1.0),
              "none": ("none", 1.0)}
SEEDS_PER_CONDITION = 4
POLICIES = ["off", "on", "oracle", "threshold_v1", "threshold_v2", "gru", "gru_free"]

V1_SIGNAL, V1_TAU, V1_HOLD = "x_norm_std", 3.65, 12
V2_SIGNAL, V2_Q, V2_BIN, K_HOLD_V2 = "x2_outlier_proj_max_max", 0.995, 50, 20
HORIZON, LAMBDA = 20, 1.0

POLICY_COLOR = {"off": "#1baf7a", "on": "#eb6834", "oracle": "#e87ba4", "threshold_v1": "#eda100",
                "threshold_v2": "#008300", "gru": "#2a78d6", "gru_free": "#104281"}
POLICY_LABEL = {"off": "always off", "on": "always on", "oracle": "oracle (defends exactly the attack)",
                "threshold_v1": "threshold defender, old signal", "threshold_v2": "threshold defender, new signal",
                "gru": "GRU (new-signal alarm, GRU decides release)", "gru_free": "GRU alone"}
SHORT = {"oracle": "oracle", "threshold_v1": "thr. old", "threshold_v2": "thr. new", "gru": "GRU", "gru_free": "GRU alone"}
DRAW_ORDER = ["off", "on", "oracle", "threshold_v1", "threshold_v2", "gru_free", "gru"]


def environment(cond: str, k: int) -> tuple[int, tuple[int, int]]:
    """k = 0, 1: early window (start 2-20); k = 2, 3: start 21-150; k = 4-7 ('late' set): start 60-150, added
    after the first deployment because a GRU that defends the first ~60 blocks by default covers early windows
    without detecting anything."""
    seed = 6000 + 100 * list(CONDITIONS).index(cond) + k
    rng = np.random.default_rng(60_000 + seed)
    lo, hi = (2, 21) if k < 2 else (21, 151) if k < 4 else (60, 151)
    start = int(rng.integers(lo, hi))
    return seed, (start, start + int(rng.integers(30, 81)) - 1)


def fit_v2_table(train: pd.DataFrame) -> dict:
    """Alarm level per (defense state, 50-block bin): 99.5th percentile on no-attack training blocks."""
    n = train[(train.attack == "none") & (train.block >= 0)]
    q = n.groupby([n.W.astype(int), n.block // V2_BIN])[V2_SIGNAL].quantile(V2_Q)
    return {f"{w}_{b}": float(v) for (w, b), v in q.items()}


# ----------------------------------------------------------------------------- controllers
class ThresholdV1:
    def __init__(self):
        self.on, self.low = False, 0

    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        value = float(history[-1].get(V1_SIGNAL) or np.nan)
        if value > V1_TAU:
            self.on, self.low = True, 0
        elif self.on:
            self.low += 1
            if self.low >= V1_HOLD:
                self.on, self.low = False, 0
        return int(self.on), {"alarm": int(value > V1_TAU)}


def v2_alarm(history: list[dict], table: dict) -> tuple[bool, float]:
    last = history[-1]
    if last["block"] < 0:
        return False, float("nan")
    value = float(last[V2_SIGNAL])
    return value > table[f"{int(last['W'])}_{int(last['block']) // V2_BIN}"], value


class ThresholdV2:
    def __init__(self, table: dict):
        self.table, self.on, self.low = table, False, 0

    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        alarm, value = v2_alarm(history, self.table)
        if alarm:
            self.on, self.low = True, 0
        elif self.on:
            self.low += 1
            if self.low >= K_HOLD_V2:
                self.on, self.low = False, 0
        return int(self.on), {"alarm": int(alarm), "signal": value}


class GRUScorer:
    """Scores 'defend for H blocks' against 'do not defend for H blocks' from the observable history."""

    def __init__(self, folder: Path, x_mean, x_std, x_cols, device: str):
        self.x_mean, self.x_std, self.x_cols, self.device = x_mean, x_std, x_cols, torch.device(device)
        hidden = json.loads((folder / "report.json").read_text())["hparams"]["hidden"]
        probe = self._data([{"run_id": "p", "seed": 0, "defense": DEFENSE, "block": -1, "W": np.nan}])
        self.model = OutcomeGRU(probe.enc.shape[-1], probe.dec_inputs(probe.W).shape[-1], hidden).to(self.device)
        self.model.load_state_dict(torch.load(folder / "gru.pt", map_location=self.device))
        self.model.eval()

    def _data(self, history: list[dict]) -> Data:
        df = pd.DataFrame(history)
        last = int(df.block.max())
        pad = pd.DataFrame([{"run_id": df.run_id.iloc[0], "seed": df.seed.iloc[0], "defense": df.defense.iloc[0],
                             "block": b} for b in range(last + 1, N_BLOCKS)])
        df = pd.concat([df, pad], ignore_index=True)
        df["attack"] = "unknown"  # the controller never knows the attack
        for c in ["W", "y_clean_acc", "y_atk", *self.x_cols]:
            if c not in df:
                df[c] = np.nan
        df["W"] = pd.to_numeric(df.W, errors="coerce")
        return Data(df, self.x_mean, self.x_std, self.x_cols)

    @torch.no_grad()
    def gain(self, history: list[dict], b: int) -> dict:
        data = self._data(history)
        h = self.model.encode(data.enc.to(self.device))[:, b]
        H = min(HORIZON, N_BLOCKS - b)
        W = np.zeros((2, N_ROWS))
        W[1, b + 1: b + 1 + H] = 1  # row 0: do not defend, row 1: defend
        dec = data.dec_inputs(W)[:, b: b + H]
        p = self.model.decode(h.expand(2, -1), dec.to(self.device)).cpu().numpy()[:, -1]
        s = p[:, 0] - LAMBDA * p[:, 1]
        return {"gru_gain": float(s[1] - s[0]), "gru_atk_wait": float(p[0, 1]), "gru_atk_defend": float(p[1, 1]),
                "gru_clean_wait": float(p[0, 0]), "gru_clean_defend": float(p[1, 0])}


class GRURelease:
    """Threshold_v2's alarm switches the defense on; the GRU decides each block whether to keep it on."""

    def __init__(self, scorer: GRUScorer, table: dict):
        self.scorer, self.table, self.on = scorer, table, False

    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        alarm, value = v2_alarm(history, self.table)
        extra = {"alarm": int(alarm), "signal": value}
        if alarm:
            self.on = True
        elif self.on:
            g = self.scorer.gain(history, b)
            extra.update(g)
            self.on = g["gru_gain"] > DELTA_RELEASE
        return int(self.on), extra


class GRUFree:
    def __init__(self, scorer: GRUScorer):
        self.scorer = scorer

    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        g = self.scorer.gain(history, b)
        return int(g["gru_gain"] > DELTA_FREE), g


# ----------------------------------------------------------------------------- runs
def config(cond: str, policy: str, k: int) -> RunConfig:
    attack, pf = CONDITIONS[cond]
    seed, window = environment(cond, k)
    if policy == "oracle":
        schedule = "react_d0_e0" if attack != "none" else "off"
    else:
        schedule = policy if policy in ("on", "off") else "policy"
    return RunConfig(attack=attack, defense=DEFENSE, seed=seed, C=XP1_C, sigma=SIGMA, schedule=schedule,
                     block_len=BLOCK_LEN, n_blocks=N_BLOCKS, window=window, poison_frac=pf,
                     always_measure_backdoor=True, signals="v2")


def run_id_of(cond: str, policy: str, k: int) -> str:
    return f"{cond.replace('/', 'of')}__{policy}__s{environment(cond, k)[0]}"


def run_one(cond: str, policy: str, k: int, stats: dict, out_root: str) -> tuple[str, float]:
    cfg = config(cond, policy, k)
    run_id = run_id_of(cond, policy, k)
    out = Path(out_root) / "runs" / run_id
    if (out / "blocks.csv").exists():
        return run_id, json.loads((out / "config.json").read_text())["wall_clock_s"]
    scorer = lambda folder: GRUScorer(folder, stats["x_mean"], stats["x_std"], stats["x_cols"], cfg.device)  # noqa: E731
    controller = {"threshold_v1": lambda: ThresholdV1(),
                  "threshold_v2": lambda: ThresholdV2(stats["table"]),
                  "gru": lambda: GRURelease(scorer(GRU_RELEASE), stats["table"]),
                  "gru_free": lambda: GRUFree(scorer(GRU_FREE))}.get(policy, lambda: None)()
    started = time.time()
    df = simulate(cfg, controller)
    wall = time.time() - started
    df.insert(0, "condition", cond)
    df.insert(1, "policy", policy)
    df["run_id"] = run_id
    df["window_start"], df["window_end"] = cfg.window  # evaluation labels (never seen by a controller)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "blocks.csv", index=False)
    folder = {"gru": GRU_RELEASE, "gru_free": GRU_FREE}.get(policy)
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "condition": cond, "policy": policy, "config": cfg.__dict__,
        "gru_model": str(folder) if folder else None,
        "gru_weights_sha256": hashlib.sha256((folder / "gru.pt").read_bytes()).hexdigest()[:16] if folder else None,
        "delta": {"gru": DELTA_RELEASE, "gru_free": DELTA_FREE}.get(policy), "lambda": LAMBDA, "horizon": HORIZON,
        "v2_table": stats["table"] if policy in ("threshold_v2", "gru") else None, "k_hold_v2": K_HOLD_V2,
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
        on_after = np.flatnonzero(w[s:]) if attacked else []
        rows.append({
            "condition": tr.condition.iloc[0], "policy": tr.policy.iloc[0], "seed": int(tr.seed.iloc[0]),
            "seed_set": "late" if int(tr.seed.iloc[0]) % 100 >= 4 else "main", "early_window": s <= 20, "window_start": s, "window_end": int(tr.window_end.iloc[0]),
            "final_clean_acc": tr.y_clean_acc.iloc[-1], "final_backdoor_success": tr.y_atk.iloc[-1],
            "attack_killed": bool(tr.y_atk.iloc[-1] < 0.1) if attacked else np.nan,
            "peak_backdoor_success": tr.y_atk.max(),
            "defended_blocks": int(w.sum()), "defended_in_window": int(w[active].sum()) if attacked else 0,
            "false_alarm_blocks": int(w[~active].sum()),
            "reaction_delay_blocks": int(on_after[0]) if len(on_after) else np.nan,
        })
    per_run = pd.DataFrame(rows)
    on = per_run[per_run.policy == "on"].set_index("seed").final_clean_acc
    per_run["clean_vs_on_pts"] = 100 * (per_run.final_clean_acc - per_run.seed.map(on))
    per_run["score_true"] = per_run.final_clean_acc - LAMBDA * per_run.final_backdoor_success  # complementary
    per_run = per_run.sort_values(["condition", "policy", "seed"])
    metrics = ["final_clean_acc", "final_backdoor_success", "attack_killed", "clean_vs_on_pts", "defended_blocks",
               "false_alarm_blocks", "reaction_delay_blocks", "score_true"]
    agg = per_run.groupby(["seed_set", "condition", "policy"])[metrics].mean().round(4).reset_index()
    agg["policy"] = pd.Categorical(agg.policy, POLICIES)
    return per_run, agg.sort_values(["seed_set", "condition", "policy"])


def fig_condition(blocks: pd.DataFrame, cond: str, out: Path, seed_set: str = "main") -> None:
    g = blocks[(blocks.condition == cond) & (blocks.block >= 0)]
    g = g[(g.seed % 100 >= 4) == (seed_set == "late")]
    if g.empty:
        return
    seeds = sorted(g.seed.unique())
    present = [p for p in DRAW_ORDER if p in set(g.policy)]
    deciders = [p for p in ["gru", "gru_free", "threshold_v2", "threshold_v1", "oracle"] if p in present]
    fig, axes = plt.subplots(3, len(seeds), figsize=(3.7 * len(seeds), 8.8), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1, 0.7]})
    for j, seed in enumerate(seeds):
        s = g[g.seed == seed]
        ref = s[s.policy == present[0]].sort_values("block")
        ws, we = int(ref.window_start.iloc[0]), int(ref.window_end.iloc[0])
        for i, (col, label) in enumerate([("y_atk", "backdoor success"), ("y_clean_acc", "clean accuracy")]):
            ax = axes[i, j]
            if cond != "none":
                ax.axvspan(ws - 0.5, we + 0.5, color=INK2, alpha=0.08, linewidth=0)
            for p in present:
                q = s[s.policy == p].sort_values("block")
                ax.plot(q.block, q[col], color=POLICY_COLOR[p], linewidth=2.0 if p == "gru" else 1.2,
                        zorder=3 if p == "gru" else 2)
            if col == "y_atk":
                ax.set_ylim(-0.03, 1.03)
                ttl = f"seed {seed}: " + (f"attack in blocks {ws}-{we}" if cond != "none" else "no attack")
                gr = s[(s.policy == "gru") & (s.block >= 0)]
                if len(gr):
                    ttl += f"\nGRU defended {int(gr.W.sum())} / {N_BLOCKS} blocks"
                ax.set_title(ttl, color=INK)
            else:
                ax.set_ylim(max(0.0, g[g.block >= 10][col].min() - 0.02), min(1.0, g[col].max() + 0.01))
            if j == 0:
                ax.set_ylabel(label)
        ax = axes[2, j]  # when each controller defends
        if cond != "none":
            ax.axvspan(ws - 0.5, we + 0.5, color=INK2, alpha=0.08, linewidth=0)
        for r, p in enumerate(deciders):
            q = s[s.policy == p].sort_values("block")
            on = q.block[q.W.astype(int) == 1].to_numpy()
            ax.broken_barh([(b - 0.5, 1) for b in on], (r - 0.35, 0.7), color=POLICY_COLOR[p], linewidth=0)
        ax.set_yticks(range(len(deciders)), [SHORT[p] for p in deciders] if j == 0 else [])
        ax.set_ylim(-0.6, len(deciders) - 0.4)
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_xlabel("block (2 SGD steps each)")
    axes[2, 0].set_ylabel("defended blocks")
    handles = [Line2D([], [], color=POLICY_COLOR[p], linewidth=2.0 if p == "gru" else 1.3, label=POLICY_LABEL[p])
               for p in present]
    if cond != "none":
        handles.append(Patch(color=INK2, alpha=0.15, label="attack active"))
    desc = {"pf1/8": "1 of 8 images poisoned", "pf2/8": "2 of 8 images poisoned", "pf8/8": "8 of 8 images poisoned",
            "none": "no attack"}[cond]
    which = " (late windows: start 60-150)" if seed_set == "late" else ""
    fig.suptitle(f"xp4 deployment{which}, windowed backdoor + Krum, {desc}: trajectories and defense decisions "
                 f"(same seed = same data and noise)", fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    top = 1 - 0.8 / fig.get_figheight()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, top + 0.045), ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, top - 0.035))
    fig.savefig(out / f"trajectory_{cond.replace('/', 'of')}{'_late' if seed_set == 'late' else ''}.png", dpi=130)
    plt.close(fig)


def fig_summary(per_run: pd.DataFrame, out: Path, seed_set: str = "main") -> None:
    per_run = per_run[per_run.seed_set == seed_set]
    if per_run.empty:
        return
    panels = [("final_backdoor_success", "final backdoor success"),
              ("clean_vs_on_pts", "final clean acc. − always-on (pts)"), ("defended_blocks", f"defended blocks (of {N_BLOCKS})")]
    conds = list(CONDITIONS)
    fig, axes = plt.subplots(len(panels), 1, figsize=(12.5, 8.8), sharex=True)
    width = 0.84 / len(POLICIES)
    for ax, (col, label) in zip(axes, panels):
        for c, cond in enumerate(conds):
            for k, p in enumerate(POLICIES):
                v = per_run[(per_run.condition == cond) & (per_run.policy == p)][col].to_numpy(float)
                if not len(v):
                    continue
                x = c - 0.42 + width * (k + 0.5)
                ax.scatter(np.full(len(v), x), v, s=15, color=POLICY_COLOR[p], edgecolor=SURFACE, linewidth=0.6, zorder=3)
                ax.plot([x - width * 0.4, x + width * 0.4], [np.nanmean(v)] * 2, color=POLICY_COLOR[p], linewidth=2.2)
        ax.set_ylabel(label)
        if col == "clean_vs_on_pts":
            ax.axhline(0, color=INK2, linewidth=0.8)
    axes[-1].set_xticks(range(len(conds)), [c if c != "none" else "no attack" for c in conds])
    handles = [Line2D([], [], color=POLICY_COLOR[p], marker="o", linewidth=2, label=POLICY_LABEL[p]) for p in POLICIES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=4, fontsize=8)
    fig.suptitle("xp4 deployment summary: dots = seeds, bar = mean; " + ("late windows only (start 60-150)" if seed_set == "late"
                 else "main set (2 early + 2 later windows per condition)"),
                 fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out / f"summary_by_policy{'_late' if seed_set == 'late' else ''}.png", dpi=130)
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
            fig_condition(blocks, cond, out, "late")
    fig_summary(per_run, out)
    fig_summary(per_run, out, "late")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(agg.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--policies", nargs="*", default=POLICIES)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=SEEDS_PER_CONDITION)
    ap.add_argument("--late", action="store_true", help="the late-window set (k = 4-7, start 60-150)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        df = pd.read_csv(TRAIN_RUNS)
        train_df = df[assign_splits(df) == "train"]
        train = Data(train_df)  # the GRU's own input normalisation and columns
        stats = {"x_mean": train.x_mean, "x_std": train.x_std, "x_cols": train.x_cols, "table": fit_v2_table(train_df)}
        (args.out / "threshold_v2_table.json").write_text(json.dumps(stats["table"], indent=2))
        ks = range(4, 8) if args.late else range(args.seeds)
        jobs = [(c, p, k) for p in args.policies for c in args.conditions for k in ks]
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
            futs = [pool.submit(run_one, c, p, k, stats, str(args.out)) for c, p, k in jobs]
            for n, fut in enumerate(as_completed(futs), 1):
                run_id, wall = fut.result()
                print(f"[{n}/{len(jobs)}] {run_id}  {wall:.0f}s  (elapsed {time.time() - started:.0f}s)", flush=True)
    report(args.out)


if __name__ == "__main__":
    main()
