"""xp5 deployment: the single mixed GRU as a defense controller, on seen and held-out configurations.

    python deploy_xp5.py --gru-summary results/xp5/gru/summary.json      -> results/deploy_xp5
    python deploy_xp5.py --report-only
    python deploy_xp5.py --smoke ...                                      (30 blocks, 1 seed, 2 environments)

Environments (Fashion-MNIST, decisions every 2 rounds, 250 blocks), 3 new seeds each: k = 0 early window
(start 2-20), k = 1, 2 later window (start 40-150); length 30-80 blocks.
    seen:      backdoor corner 2/8 x Krum, backdoor patch 8/8 x Median, burst corner 8/8 x TrimmedMean,
               flip 8/8 x Krum, ALIE x Krum, ALIE x Median, no attack x Krum
    held-out:  backdoor blend 2/8 x Krum, backdoor corner 3/8 x TrimmedMean, flip 8/8 x TrimmedMean
Policies (same seed = same data and noise):
    clean_ref     no attack, never defend: the no-attack reference of the seed (ALIE's goal is to reach it)
    off / on      never / always defend
    oracle        defends exactly the attack's active blocks (knows the attack; reference only)
    threshold     item-1 alarm: max of the z-scores of x_norm_std and x2_outlier_proj_max (z against no-attack
                  training blocks of the same defense, defense state and 25-block bin), threshold for ~1 alarm per
                  no-attack training run; hold until HOLD blocks without alarm
    gru           the same alarm switches the defense on; while defending, the GRU decides each block whether to
                  keep defending (gain of 'defend H blocks' over 'stop for H blocks' > margin)
    gru_free      the GRU alone: defend iff the gain of defending for H blocks over not defending > margin
Margins and the model come from gru_xp5.py (validation seeds only).
Harm: attack success (backdoor success, flip rate) for targeted attacks; for ALIE, the clean-accuracy gap to
clean_ref of the same seed.
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
from analyze import INK, INK2, SURFACE  # noqa: E402
from gru_experiment import Data, OutcomeGRU, assign_splits  # noqa: E402
from poison_testbed.runner import RunConfig, simulate  # noqa: E402
from run_experiment import xp5_params  # noqa: E402

ROOT = Path(__file__).resolve().parent
TRAIN_RUNS = ROOT / "results/xp5/runs_xp5/all_blocks.csv"
OUT = ROOT / "results/deploy_xp5"
N_BLOCKS = 250
ENVIRONMENTS = {  # name -> (RunConfig fields, defense, held out?)
    "backdoor corner 2/8 | Krum": (dict(attack="backdoor_win", poison_frac=0.25), "Krum", False),
    "backdoor patch 8/8 | Median": (dict(attack="backdoor_win", poison_frac=1.0, trigger="patch_r3_c5"), "Median", False),
    "burst corner 8/8 | TrimmedMean": (dict(attack="backdoor_burst", poison_frac=1.0, burst=(10, 5)), "TrimmedMean", False),
    "flip 8/8 | Krum": (dict(attack="flip_win", poison_frac=1.0), "Krum", False),
    "ALIE | Krum": (dict(attack="alie_win"), "Krum", False),
    "ALIE | Median": (dict(attack="alie_win"), "Median", False),
    "no attack | Krum": (dict(attack="none"), "Krum", False),
    "blend 2/8 | Krum (held-out)": (dict(attack="backdoor_win", poison_frac=0.25, trigger="blend"), "Krum", True),
    "corner 3/8 | TrimmedMean (held-out)": (dict(attack="backdoor_win", poison_frac=0.375), "TrimmedMean", True),
    "flip 8/8 | TrimmedMean (held-out)": (dict(attack="flip_win", poison_frac=1.0), "TrimmedMean", True),
}
SEEDS_PER_ENV = 3
POLICIES = ["clean_ref", "off", "on", "oracle", "threshold", "gru", "gru_free"]
V1, V2, BIN, HOLD, HORIZON, LAMBDA = "x_norm_std", "x2_outlier_proj_max_max", 25, 20, 20, 1.0

POLICY_COLOR = {"clean_ref": "#52514e", "off": "#1baf7a", "on": "#eb6834", "oracle": "#e87ba4",
                "threshold": "#008300", "gru": "#2a78d6", "gru_free": "#104281"}
POLICY_LABEL = {"clean_ref": "no-attack reference", "off": "always off", "on": "always on",
                "oracle": "oracle (defends exactly the attack)", "threshold": "threshold defender (old OR new signal)",
                "gru": "GRU (alarm on, GRU decides release)", "gru_free": "GRU alone"}
SHORT = {"oracle": "oracle", "threshold": "threshold", "gru": "GRU", "gru_free": "GRU alone"}
DRAW_ORDER = ["clean_ref", "off", "on", "oracle", "threshold", "gru_free", "gru"]


def environment(env: str, k: int, smoke: bool = False) -> tuple[int, tuple[int, int]]:
    seed = 8000 + 100 * list(ENVIRONMENTS).index(env) + k
    if smoke:
        return seed, (5, 20)
    rng = np.random.default_rng(90_000 + seed)
    start = int(rng.integers(2, 21)) if k == 0 else int(rng.integers(40, 151))
    return seed, (start, start + int(rng.integers(30, 81)) - 1)


# ----------------------------------------------------------------------------- alarm (item 1)
def fit_alarm(train: pd.DataFrame) -> dict:
    """z-score tables (defense, W, bin) from no-attack training blocks, and the threshold for ~1 alarm per run."""
    n = train[(train.attack == "none") & (train.block >= 0)].copy()
    n["W"] = n.W.astype(int)
    n["bin"] = n.block // BIN
    stats = n.groupby(["defense", "W", "bin"])[[V1, V2]].agg(["mean", "std"])
    table = {f"{d}|{w}|{b}": [float(stats.loc[(d, w, b), (c, s)]) for c in (V1, V2) for s in ("mean", "std")]
             for d, w, b in stats.index}
    n["z"] = [score_of(v1, v2, table.get(f"{d}|{w}|{b}"))
              for v1, v2, d, w, b in zip(n[V1], n[V2], n.defense, n.W, n.bin)]
    lo, hi = 0.0, 1.0
    for _ in range(40):  # lowest quantile with <= 1 alarm onset per no-attack training run
        q = (lo + hi) / 2
        thr = n.z.quantile(q)
        al = n.z > thr
        on = (al & ~al.groupby(n.run_id).shift(1, fill_value=False)).groupby(n.run_id).sum().mean()
        lo, hi = (q, hi) if on > 1.0 else (lo, q)
    return {"table": table, "threshold": float(n.z.quantile(hi))}


def score_of(v1: float, v2: float, row: list | None) -> float:
    if row is None or not np.isfinite(v1) or not np.isfinite(v2):
        return float("-inf")
    m1, s1, m2, s2 = row
    return max((v1 - m1) / max(s1, 1e-9), (v2 - m2) / max(s2, 1e-9))


def alarm_of(history: list[dict], alarm: dict) -> tuple[bool, float]:
    last = history[-1]
    if last["block"] < 0:
        return False, float("nan")
    key = f"{last['defense']}|{int(last['W'])}|{int(last['block']) // BIN}"
    z = score_of(float(last[V1]), float(last[V2]), alarm["table"].get(key))
    return z > alarm["threshold"], z


class Threshold:
    def __init__(self, alarm: dict):
        self.alarm, self.on, self.low = alarm, False, 0

    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        hit, z = alarm_of(history, self.alarm)
        if hit:
            self.on, self.low = True, 0
        elif self.on:
            self.low += 1
            if self.low >= HOLD:
                self.on, self.low = False, 0
        return int(self.on), {"alarm": int(hit), "alarm_z": z}


class GRUScorer:
    def __init__(self, folder: Path, stats: dict, device: str, n_blocks: int):
        self.stats, self.device, self.n_blocks = stats, torch.device(device), n_blocks
        hidden = json.loads((folder / "report.json").read_text())["hparams"]["hidden"]
        probe = self._data([{"run_id": "p", "seed": 0, "defense": "Krum", "block": -1, "W": np.nan}])
        self.model = OutcomeGRU(probe.enc.shape[-1], probe.dec_inputs(probe.W).shape[-1], hidden).to(self.device)
        self.model.load_state_dict(torch.load(folder / "gru.pt", map_location=self.device))
        self.model.eval()

    def _data(self, history: list[dict]) -> Data:
        df = pd.DataFrame(history)
        last = int(df.block.max())
        pad = pd.DataFrame([{"run_id": df.run_id.iloc[0], "seed": df.seed.iloc[0], "defense": df.defense.iloc[0],
                             "block": b} for b in range(last + 1, self.n_blocks)])
        df = pd.concat([df, pad], ignore_index=True)
        df["attack"] = "unknown"  # the controller never knows the attack
        for c in ["W", "y_clean_acc", "y_atk", *self.stats["x_cols"]]:
            if c not in df:
                df[c] = np.nan
        df["W"] = pd.to_numeric(df.W, errors="coerce")
        return Data(df, self.stats["x_mean"], self.stats["x_std"], self.stats["x_cols"])

    @torch.no_grad()
    def gain(self, history: list[dict], b: int) -> dict:
        data = self._data(history)
        h = self.model.encode(data.enc.to(self.device))[:, b]
        H = min(HORIZON, self.n_blocks - b)
        W = np.zeros((2, self.n_blocks + 1))
        W[1, b + 1: b + 1 + H] = 1
        p = self.model.decode(h.expand(2, -1), data.dec_inputs(W)[:, b: b + H].to(self.device)).cpu().numpy()[:, -1]
        s = p[:, 0] - LAMBDA * p[:, 1]
        return {"gru_gain": float(s[1] - s[0]), "gru_atk_wait": float(p[0, 1]), "gru_atk_defend": float(p[1, 1]),
                "gru_clean_wait": float(p[0, 0]), "gru_clean_defend": float(p[1, 0])}


class GRURelease:
    def __init__(self, scorer: GRUScorer, alarm: dict, delta: float):
        self.scorer, self.alarm, self.delta, self.on = scorer, alarm, delta, False

    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        hit, z = alarm_of(history, self.alarm)
        extra = {"alarm": int(hit), "alarm_z": z}
        if hit:
            self.on = True
        elif self.on:
            g = self.scorer.gain(history, b)
            extra.update(g)
            self.on = g["gru_gain"] > self.delta
        return int(self.on), extra


class GRUFree:
    def __init__(self, scorer: GRUScorer, delta: float):
        self.scorer, self.delta = scorer, delta

    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        g = self.scorer.gain(history, b)
        return int(g["gru_gain"] > self.delta), g


# ----------------------------------------------------------------------------- runs
def config(env: str, policy: str, k: int, smoke: bool) -> RunConfig:
    fields, defense, _ = ENVIRONMENTS[env]
    seed, window = environment(env, k, smoke)
    fields = dict(fields)
    if policy == "clean_ref":
        fields = {"attack": "none", **{f: v for f, v in fields.items() if f == "trigger"}}
        schedule = "off"
    elif policy == "oracle":
        schedule = "oracle" if fields["attack"] != "none" else "off"
    else:
        schedule = policy if policy in ("on", "off") else "policy"
    over = {"n_blocks": 30} if smoke else {}
    return RunConfig(defense=defense, seed=seed, schedule=schedule, window=window, **fields, **xp5_params(over))


def run_id_of(env: str, policy: str, k: int, smoke: bool) -> str:
    slug = env.split(" (")[0].replace(" | ", "__").replace("/", "of").replace(" ", "_")
    return f"{slug}__{policy}__s{environment(env, k, smoke)[0]}"


def run_one(env: str, policy: str, k: int, ctx: dict, out_root: str) -> tuple[str, float]:
    smoke = ctx["smoke"]
    cfg = config(env, policy, k, smoke)
    run_id = run_id_of(env, policy, k, smoke)
    out = Path(out_root) / "runs" / run_id
    if (out / "blocks.csv").exists():
        return run_id, json.loads((out / "config.json").read_text())["wall_clock_s"]
    folder = Path(ctx["gru_folder"])
    scorer = lambda: GRUScorer(folder, ctx["stats"], cfg.device, cfg.n_blocks)  # noqa: E731
    controller = {"threshold": lambda: Threshold(ctx["alarm"]),
                  "gru": lambda: GRURelease(scorer(), ctx["alarm"], ctx["margins"]["release_defended"]["delta"]),
                  "gru_free": lambda: GRUFree(scorer(), ctx["margins"]["release_all"]["delta"])}.get(policy, lambda: None)()
    started = time.time()
    df = simulate(cfg, controller).copy()
    wall = time.time() - started
    df.insert(0, "environment", env)
    df.insert(1, "policy", policy)
    df["held_out"] = ENVIRONMENTS[env][2]
    df["run_id"] = run_id
    df["window_start"], df["window_end"] = cfg.window
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "blocks.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "environment": env, "policy": policy, "config": cfg.__dict__,
        "gru_model": str(folder) if policy.startswith("gru") else None,
        "gru_weights_sha256": hashlib.sha256((folder / "gru.pt").read_bytes()).hexdigest()[:16]
        if policy.startswith("gru") else None, "margins": ctx["margins"], "alarm_threshold": ctx["alarm"]["threshold"],
        "horizon": HORIZON, "lambda": LAMBDA, "hold": HOLD, "wall_clock_s": round(wall, 1),
        "device_name": torch.cuda.get_device_name(0) if cfg.device == "cuda" else cfg.device,
    }, indent=2, default=str))
    return run_id, wall


# ----------------------------------------------------------------------------- report
def summarize(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = (blocks[(blocks.policy == "clean_ref") & (blocks.block >= 0)]
           .set_index(["seed", "block"]).y_clean_acc)
    rows = []
    for run_id, g in blocks[blocks.policy != "clean_ref"].groupby("run_id"):
        tr = g[g.block >= 0].sort_values("block")
        w = tr.W.astype(int).to_numpy()
        active = tr.u_active.to_numpy().astype(bool)
        attacked = tr.attack.iloc[0] != "none"
        s, e = int(tr.window_start.iloc[0]), int(tr.window_end.iloc[0])
        on_after = np.flatnonzero(w[s:]) if attacked else []
        r = ref.reindex(list(zip(tr.seed, tr.block))).to_numpy()
        targeted = tr.y_atk.notna().any() and tr.attack.iloc[0] in ("backdoor_win", "backdoor_burst", "flip_win")
        in_win = tr[(tr.block >= s) & (tr.block <= e)]
        rows.append({
            "environment": tr.environment.iloc[0], "policy": tr.policy.iloc[0], "seed": int(tr.seed.iloc[0]),
            "held_out": bool(tr.held_out.iloc[0]), "window_start": s, "window_end": e,
            "final_clean_acc": tr.y_clean_acc.iloc[-1],
            "clean_gap_to_no_attack_final": tr.y_clean_acc.iloc[-1] - r[-1],
            "clean_gap_to_no_attack_mean": float(np.nanmean(tr.y_clean_acc.to_numpy() - r)),
            "final_attack_success": tr.y_atk.iloc[-1] if targeted else np.nan,
            "mean_attack_success_in_window": in_win.y_atk.mean() if targeted else np.nan,
            "peak_attack_success": tr.y_atk.max() if targeted else np.nan,
            "neutralised": (bool(max(tr.y_atk.iloc[-1], in_win.y_atk.mean() if tr.attack.iloc[0] == "flip_win" else 0)
                                 < 0.1) if targeted else np.nan),
            "defended_blocks": int(w.sum()), "defended_while_active": int(w[active].sum()),
            "false_alarm_blocks": int(w[~active].sum()),
            "reaction_delay_blocks": int(on_after[0]) if len(on_after) else np.nan,
        })
    per_run = pd.DataFrame(rows)
    on = per_run[per_run.policy == "on"].set_index("seed").final_clean_acc
    per_run["clean_vs_on_pts"] = 100 * (per_run.final_clean_acc - per_run.seed.map(on))
    metrics = ["final_clean_acc", "clean_gap_to_no_attack_final", "final_attack_success",
               "mean_attack_success_in_window", "neutralised", "clean_vs_on_pts", "defended_blocks",
               "false_alarm_blocks", "reaction_delay_blocks"]
    agg = per_run.groupby(["environment", "policy"], sort=False)[metrics].mean().round(4).reset_index()
    agg["policy"] = pd.Categorical(agg.policy, POLICIES)
    agg["environment"] = pd.Categorical(agg.environment, list(ENVIRONMENTS))
    return per_run.sort_values(["environment", "policy", "seed"]), agg.sort_values(["environment", "policy"])


def fig_environment(blocks: pd.DataFrame, env: str, out: Path) -> None:
    g = blocks[(blocks.environment == env) & (blocks.block >= 0)].copy()
    if g.empty:
        return
    is_alie = g.attack.eq("alie_win").any()
    ref = g[g.policy == "clean_ref"].set_index(["seed", "block"]).y_clean_acc
    g["harm"] = (ref.reindex(list(zip(g.seed, g.block))).to_numpy() - g.y_clean_acc) if is_alie else g.y_atk
    seeds = sorted(g.seed.unique())
    present = [p for p in DRAW_ORDER if p in set(g.policy) and not (p == "clean_ref" and not is_alie)]
    deciders = [p for p in ["gru", "gru_free", "threshold", "oracle"] if p in present]
    fig, axes = plt.subplots(3, len(seeds), figsize=(3.9 * len(seeds) + 0.6, 8.6), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1, 0.6]})
    no_attack = g.attack.eq("none").all() and not is_alie
    for j, seed in enumerate(seeds):
        s = g[g.seed == seed]
        first = s[s.policy == present[0]].sort_values("block")
        ws, we = int(first.window_start.iloc[0]), int(first.window_end.iloc[0])
        active = first.u_active.to_numpy().astype(bool) if "u_active" in first else None
        for i, (col, label) in enumerate([("harm", "clean acc. lost vs no attack" if is_alie else "attack success"),
                                          ("y_clean_acc", "clean accuracy")]):
            ax = axes[i, j]
            if not no_attack and active is not None:
                for b in np.flatnonzero(active):
                    ax.axvspan(b - 0.5, b + 0.5, color=INK2, alpha=0.08, linewidth=0)
            for p in present:
                q = s[s.policy == p].sort_values("block")
                if col == "harm" and p == "clean_ref":
                    continue
                ax.plot(q.block, q[col], color=POLICY_COLOR[p], linewidth=2.0 if p == "gru" else 1.2,
                        zorder=3 if p == "gru" else 2)
            if i == 0:
                gr = s[s.policy == "gru"]
                ttl = f"seed {seed}: " + ("no attack" if no_attack else f"attack window {ws}-{we}")
                if len(gr):
                    ttl += f"\nGRU defended {int(gr.W.sum())} / {gr.block.max() + 1} blocks"
                ax.set_title(ttl, color=INK)
                if not is_alie:
                    ax.set_ylim(-0.03, 1.03)
            else:
                lo = g[g.block >= 10][col].min() if (g.block >= 10).any() else g[col].min()
                ax.set_ylim(max(0.0, lo - 0.02), min(1.0, g[col].max() + 0.01))
            if j == 0:
                ax.set_ylabel(label)
        ax = axes[2, j]
        for r_, p in enumerate(deciders):
            q = s[s.policy == p].sort_values("block")
            on = q.block[q.W.astype(int) == 1].to_numpy()
            ax.broken_barh([(b - 0.5, 1) for b in on], (r_ - 0.35, 0.7), color=POLICY_COLOR[p], linewidth=0)
        ax.set_yticks(range(len(deciders)), [SHORT[p] for p in deciders] if j == 0 else [])
        ax.set_ylim(-0.6, len(deciders) - 0.4)
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_xlabel("block (2 SGD steps each)")
    axes[2, 0].set_ylabel("defended blocks")
    handles = [Line2D([], [], color=POLICY_COLOR[p], linewidth=2.0 if p == "gru" else 1.3, label=POLICY_LABEL[p])
               for p in present]
    if not no_attack:
        handles.append(Patch(color=INK2, alpha=0.15, label="attack active"))
    fig.suptitle(f"xp5 deployment (Fashion-MNIST): {env}: trajectories and defense decisions (same seed = same data "
                 f"and noise)", fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    top = 1 - 0.8 / fig.get_figheight()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, top + 0.045), ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, top - 0.035))
    slug = env.replace(" (held-out)", "_heldout").replace(" | ", "__").replace("/", "of").replace(" ", "_")
    fig.savefig(out / f"trajectory_{slug}.png", dpi=120)
    plt.close(fig)


def fig_summary(per_run: pd.DataFrame, out: Path) -> None:
    envs = [e for e in ENVIRONMENTS if e in set(per_run.environment)]
    pols = [p for p in POLICIES if p != "clean_ref"]
    panels = [("harm", "harm: final attack success (backdoor), mean in window (flip),\nclean acc. lost vs no attack (ALIE)"),
              ("clean_gap_to_no_attack_final", "final clean acc. − no-attack reference"),
              ("defended_blocks", "defended blocks")]
    p = per_run.copy()
    p["harm"] = np.where(p.environment.str.startswith("flip"), p.mean_attack_success_in_window,
                         np.where(p.environment.str.startswith("ALIE"), -p.clean_gap_to_no_attack_final,
                                  p.final_attack_success))
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 10), sharex=True)
    width = 0.84 / len(pols)
    for ax, (col, label) in zip(axes, panels):
        for c, env in enumerate(envs):
            for k, pol in enumerate(pols):
                v = p[(p.environment == env) & (p.policy == pol)][col].to_numpy(float)
                if not len(v) or np.all(np.isnan(v)):
                    continue
                x = c - 0.42 + width * (k + 0.5)
                ax.scatter(np.full(len(v), x), v, s=14, color=POLICY_COLOR[pol], edgecolor=SURFACE, linewidth=0.6, zorder=3)
                ax.plot([x - width * 0.4, x + width * 0.4], [np.nanmean(v)] * 2, color=POLICY_COLOR[pol], linewidth=2.2)
        ax.set_ylabel(label, fontsize=8)
        held = [i for i, e in enumerate(envs) if ENVIRONMENTS[e][2]]
        if held:
            ax.axvspan(min(held) - 0.5, max(held) + 0.5, color=INK2, alpha=0.07, linewidth=0)
    axes[-1].set_xticks(range(len(envs)), [e.replace(" | ", "\n").replace(" (held-out)", "\n(held-out)") for e in envs],
                        fontsize=8)
    handles = [Line2D([], [], color=POLICY_COLOR[q], marker="o", linewidth=2, label=POLICY_LABEL[q]) for q in pols]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, fontsize=8)
    fig.suptitle("xp5 deployment summary (dots = seeds, bar = mean; shaded = held-out configurations)",
                 fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "summary_by_policy.png", dpi=120)
    plt.close(fig)


def report(out: Path) -> None:
    blocks = pd.concat([pd.read_csv(f) for f in sorted((out / "runs").glob("*/blocks.csv"))], ignore_index=True)
    blocks.to_csv(out / "blocks_all.csv", index=False)
    per_run, agg = summarize(blocks)
    per_run.to_csv(out / "summary_runs.csv", index=False)
    agg.to_csv(out / "summary_by_policy.csv", index=False)
    for env in ENVIRONMENTS:
        fig_environment(blocks, env, out)
    fig_summary(per_run, out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(agg.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gru-summary", type=Path, default=ROOT / "results/xp5/gru/summary.json")
    ap.add_argument("--runs", type=Path, default=TRAIN_RUNS)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="30 blocks, 1 seed, 2 environments")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        summary = json.loads(args.gru_summary.read_text())
        df = pd.read_csv(args.runs)
        from gru_xp5 import groups_of  # same seen / held-out labelling as the GRU training
        df = df.sort_values(["run_id", "block"]).reset_index(drop=True)
        seen = df[groups_of(df) == "seen"]
        train_df = seen[assign_splits(seen) == "train"]
        train = Data(train_df)
        ctx = {"smoke": args.smoke, "gru_folder": summary["selected_folder"], "margins": summary["margins"],
               "stats": {"x_mean": train.x_mean, "x_std": train.x_std, "x_cols": train.x_cols},
               "alarm": fit_alarm(train_df)}
        (args.out / "alarm.json").write_text(json.dumps(ctx["alarm"], indent=2))
        envs = list(ENVIRONMENTS)[:2] if args.smoke else list(ENVIRONMENTS)
        ks = range(1) if args.smoke else range(SEEDS_PER_ENV)
        jobs = [(e, p, k) for p in POLICIES for e in envs for k in ks]
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
            futs = [pool.submit(run_one, e, p, k, ctx, str(args.out)) for e, p, k in jobs]
            for n, fut in enumerate(as_completed(futs), 1):
                run_id, wall = fut.result()
                print(f"[{n}/{len(jobs)}] {run_id}  {wall:.0f}s  (elapsed {time.time() - started:.0f}s)", flush=True)
    report(args.out)


if __name__ == "__main__":
    main()
