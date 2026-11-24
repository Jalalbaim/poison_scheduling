"""xp7 deployment: the trained models as Krum controllers on new seeds (layout of GRU_mnist/deploy_xp5.py).

    ../.venv/Scripts/python deploy_xp7.py --models gru_g0=g0/gru cpc_g0=g0/A0 ...   # -> results/deploy_xp7/
    ../.venv/Scripts/python deploy_xp7.py --report-only
    ../.venv/Scripts/python deploy_xp7.py --smoke --models ...                       # 30 blocks, 2 environments

Environments (Fashion-MNIST, Krum, 250 blocks of 2 rounds; 3 new seeds each: k = 0 early window, k = 1, 2 later;
persistent: from block 1-10 to the end): the xp7 families (corner 2/8, patch 8/8, flip 8/8, ALIE, SignFlip 4.6,
no attack) and persistent corner 2/8, patch 8/8, ALIE.
Policies (same seed = same data and noise):
    clean_ref   no attack, never defend (ALIE / SignFlip only: their harm = clean accuracy lost against it)
    off / on    never / always defend
    oracle      defends exactly the attack's active blocks (knows the attack: reference only)
    alarm       the xp6 alarm rule alarm_t3.67_d0_h20 (the xp5 deployment threshold, hold 20 blocks)
    <model>     each block b: predicted gain of 'Krum on over [b, b + 20), then off' over 'off from b', whole-run
                utility (mean of clean - atk over blocks b..T); defend iff gain > margin (0 by default; --margin val: validation twins)
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
from cpc import _GRU  # noqa: E402,F401
from cpc.data import RunData  # noqa: E402
from cpc.evaluate import CPCPredictor, GRUPredictor  # noqa: E402
from cpc.train import load  # noqa: E402
from cpc.xp7data import signed_log  # noqa: E402
import poison_testbed  # noqa: E402,F401
from analyze import INK, INK2, SURFACE  # noqa: E402
from gru_experiment import OutcomeGRU  # noqa: E402
from poison_testbed.runner import RunConfig, simulate  # noqa: E402
from run_experiment import xp5_params  # noqa: E402
from xp7 import SIGNFLIP_SCALE  # noqa: E402

MODELS = HERE / "results/xp7_models"
SCORES = HERE / "results/xp7_scores/summary.csv"
OUT = HERE / "results/deploy_xp7"
N_BLOCKS, HORIZON, LAMBDA, SEED0 = 250, 20, 1.0, 60_000
ALARM = "alarm_t3.67_d0_h20"
ENVIRONMENTS = {  # name -> (RunConfig fields, persistent)
    "corner 2/8": (dict(attack="backdoor_win", trigger="corner", poison_frac=0.25), False),
    "patch 8/8": (dict(attack="backdoor_win", trigger="patch_r3_c5", poison_frac=1.0), False),
    "flip 8/8": (dict(attack="flip_win", poison_frac=1.0, flip_source=1), False),
    "ALIE": (dict(attack="alie_win"), False),
    "SignFlip": (dict(attack="sign_flip_win", sign_flip_scale=SIGNFLIP_SCALE), False),
    "no attack": (dict(attack="none"), False),
    "corner 2/8 persistent": (dict(attack="backdoor_win", trigger="corner", poison_frac=0.25), True),
    "patch 8/8 persistent": (dict(attack="backdoor_win", trigger="patch_r3_c5", poison_frac=1.0), True),
    "ALIE persistent": (dict(attack="alie_win"), True),
}
GRADIENT = ("alie_win", "sign_flip_win")
FIXED = ["clean_ref", "off", "on", "oracle", "alarm"]
COLOR = {"clean_ref": "#52514e", "off": "#1baf7a", "on": "#eb6834", "oracle": "#e87ba4", "alarm": "#008300"}
MODEL_COLORS = ["#2a78d6", "#104281", "#9ec2ee", "#7b4fc9", "#c94f9b", "#8a94a6"]
SEEDS = 3


def environment(env: str, k: int, smoke: bool) -> tuple[int, tuple[int, int], int]:
    seed = SEED0 + 100 * list(ENVIRONMENTS).index(env) + k
    rng = np.random.default_rng(91_000 + seed)
    target = int(rng.choice([0, 3, 7]))
    n = 30 if smoke else N_BLOCKS
    if smoke:
        return seed, (5, 20), target
    if ENVIRONMENTS[env][1]:
        return seed, (int(rng.integers(1, 11)), n - 1), target
    start = int(rng.integers(2, 21)) if k == 0 else int(rng.integers(40, 151))
    return seed, (start, start + int(rng.integers(30, 81)) - 1), target


def config(env: str, policy: str, k: int, smoke: bool) -> RunConfig:
    fields = dict(ENVIRONMENTS[env][0])
    seed, window, target = environment(env, k, smoke)
    if policy == "clean_ref":
        fields, schedule = {"attack": "none"}, "off"
    elif policy in ("off", "on"):
        schedule = policy
    elif policy == "oracle":
        schedule = "oracle" if fields["attack"] != "none" else "off"
    elif policy == "alarm":
        schedule = ALARM
    else:
        schedule = "policy"
    over = {"n_blocks": 30} if smoke else {}
    return RunConfig(defense="Krum", seed=seed, schedule=schedule, window=window, target=target, **fields,
                     **xp5_params(over))


# ----------------------------------------------------------------------------- model controller
class ModelPolicy:
    """Defend at block b iff the model's predicted whole-run gain of 'on for H blocks' over 'off' > margin."""

    def __init__(self, folder: Path, margin: float, device: str, n_blocks: int):
        rep = json.loads((folder / "report.json").read_text())
        norm = np.load(folder / "norm.npz")
        self.device, self.n_blocks, self.margin = device, n_blocks, margin
        if rep["model"] == "gru":
            x_cols = rep["x_cols"]
        else:
            x_cols = json.loads((folder / "config.json").read_text())["x_cols"]
        self.ref = SimpleNamespace(x_cols=x_cols, d=SimpleNamespace(x_mean=norm["x_mean"], x_std=norm["x_std"]))
        probe = self._data([{"run_id": "p", "seed": 0, "defense": "Krum", "block": -1, "W": np.nan}])
        if rep["model"] == "gru":
            m = OutcomeGRU(probe.d.enc.shape[-1], probe.d.dec_inputs(probe.d.W).shape[-1], rep["hparams"]["hidden"])
            m.load_state_dict(torch.load(folder / "gru.pt", map_location=device))
            self.pred = GRUPredictor(m.to(device).eval(), device)
        else:
            self.pred = CPCPredictor(load(folder, device)[0], device)

    def _data(self, history: list[dict]) -> RunData:
        df = pd.DataFrame(history)
        last = int(df.block.max())
        pad = pd.DataFrame([{"run_id": df.run_id.iloc[0], "seed": df.seed.iloc[0], "defense": "Krum", "block": b}
                            for b in range(last + 1, self.n_blocks)])
        df = pd.concat([df, pad], ignore_index=True)
        df["attack"] = "unknown"  # the controller never knows the attack
        for c in ["W", "y_clean_acc", "y_atk", *self.ref.x_cols]:
            if c not in df:
                df[c] = np.nan
        df[self.ref.x_cols] = signed_log(df[self.ref.x_cols].to_numpy(float))
        df["W"] = pd.to_numeric(df.W, errors="coerce")
        return RunData(df, ref=self.ref)

    def __call__(self, history: list[dict], b: int) -> tuple[int, dict]:
        rd = self._data(history)
        n = rd.n_rows
        h = min(HORIZON, self.n_blocks - b)
        plans = rd.W[0].clone().expand(2, -1).clone()  # rows 0..b = the realised past W (row b = block b-1)
        plans[:, b + 1:] = 0
        plans[1, b + 1:b + 1 + h] = 1
        y = self.pred.plans(rd, 0, b, plans).numpy()  # (2, n-1-b, 2): rows b+1 .. n-1
        u = (y[..., 0] - LAMBDA * y[..., 1]).mean(1)
        gain = float(u[1] - u[0])
        assert y.shape[1] == n - 1 - b
        return int(gain > self.margin), {"gain": gain}


# ----------------------------------------------------------------------------- runs
def run_one(env: str, policy: str, k: int, ctx: dict) -> pd.DataFrame:
    smoke = ctx["smoke"]
    cfg = config(env, policy, k, smoke)
    out = OUT / ("runs_smoke" if smoke else "runs") / f"{env.replace(' ', '_').replace('/', 'of')}__{policy}__s{cfg.seed}"
    meta = {"config": json.loads(json.dumps(cfg.__dict__, default=str)), "model": ctx["models"].get(policy)}
    if (out / "blocks.csv").exists() and json.loads((out / "meta.json").read_text()).get("key") == meta:
        return pd.read_csv(out / "blocks.csv", low_memory=False)
    controller = None
    if policy in ctx["models"]:
        spec = ctx["models"][policy]
        controller = ModelPolicy(MODELS / spec["folder"], spec["margin"], cfg.device, cfg.n_blocks)
    t0 = time.time()
    df = simulate(cfg, controller).copy()
    df.insert(0, "environment", env)
    df.insert(1, "policy", policy)
    df["k"] = k
    df["window_start"], df["window_end"] = cfg.window
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "blocks.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"key": meta, "wall_clock_s": round(time.time() - t0, 1)}, indent=1))
    return df


def summarize(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = blocks[(blocks.policy == "clean_ref") & (blocks.block >= 0)].set_index(["seed", "block"]).y_clean_acc
    rows = []
    for (env, pol, seed), g in blocks[blocks.policy != "clean_ref"].groupby(["environment", "policy", "seed"]):
        tr = g[g.block >= 0].sort_values("block")
        w = tr.W.astype(float).to_numpy()
        active = tr.u_active.to_numpy().astype(bool)
        attack = tr.attack.iloc[0]
        targeted = attack in ("backdoor_win", "flip_win")
        atk = tr.y_atk.to_numpy(float) if targeted else np.zeros(len(tr))
        s = int(tr.window_start.iloc[0])
        on_after = np.flatnonzero(w[s:]) if attack != "none" else []
        r = ref.reindex(list(zip(tr.seed, tr.block))).to_numpy() if attack in GRADIENT else np.full(len(tr), np.nan)
        rows.append({
            "environment": env, "policy": pol, "seed": seed,
            "utility": float(np.mean(tr.y_clean_acc.to_numpy() - LAMBDA * atk)),
            "final_clean_acc": float(tr.y_clean_acc.iloc[-1]), "mean_clean_acc": float(tr.y_clean_acc.mean()),
            "mean_attack_success_active": float(atk[active].mean()) if targeted and active.any() else np.nan,
            "final_attack_success": float(atk[-1]) if targeted else np.nan,
            "clean_lost_vs_no_attack": float(np.nanmean(r - tr.y_clean_acc.to_numpy())) if attack in GRADIENT else np.nan,
            "defended_blocks": int(w.sum()), "defended_while_active": int(w[active].sum()),
            "false_alarm_blocks": int(w[~active].sum()),
            "reaction_delay_blocks": int(on_after[0]) if len(on_after) else np.nan,
        })
    per_run = pd.DataFrame(rows)
    orc = per_run[per_run.policy == "oracle"].set_index(["environment", "seed"]).utility
    per_run["utility_vs_oracle"] = per_run.utility - orc.reindex(list(zip(per_run.environment, per_run.seed))).to_numpy()
    cols = ["utility", "utility_vs_oracle", "final_clean_acc", "mean_clean_acc", "mean_attack_success_active",
            "final_attack_success", "clean_lost_vs_no_attack", "defended_blocks", "false_alarm_blocks",
            "reaction_delay_blocks"]
    agg = per_run.groupby(["environment", "policy"], sort=False)[cols].mean().round(4).reset_index()
    return per_run, agg


# ----------------------------------------------------------------------------- figures
def colors(policies: list[str]) -> dict:
    c = dict(COLOR)
    for i, p in enumerate([p for p in policies if p not in COLOR]):
        c[p] = MODEL_COLORS[i % len(MODEL_COLORS)]
    return c


def fig_environment(blocks: pd.DataFrame, env: str, policies: list[str], out: Path) -> None:
    g = blocks[(blocks.environment == env) & (blocks.block >= 0)].copy()
    col = colors(policies)
    gradient = g.attack.isin(GRADIENT).any()
    ref = g[g.policy == "clean_ref"].set_index(["seed", "block"]).y_clean_acc
    g["harm"] = (ref.reindex(list(zip(g.seed, g.block))).to_numpy() - g.y_clean_acc) if gradient else g.y_atk
    seeds = sorted(g.seed.unique())
    deciders = [p for p in policies if p not in ("clean_ref", "off", "on")]
    fig, axes = plt.subplots(3, len(seeds), figsize=(3.9 * len(seeds) + 0.6, 8.8), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1, 0.75]})
    no_attack = env == "no attack"
    for j, seed in enumerate(seeds):
        s = g[g.seed == seed]
        first = s[s.policy == "off"].sort_values("block")
        active = first.u_active.to_numpy().astype(bool)
        for i, (c, label) in enumerate([("harm", "clean acc. lost vs no attack" if gradient else "attack success"),
                                        ("y_clean_acc", "clean accuracy")]):
            ax = axes[i, j]
            if not no_attack:
                for b in np.flatnonzero(active):
                    ax.axvspan(b - 0.5, b + 0.5, color=INK2, alpha=0.08, linewidth=0)
            for p in policies:
                if p == "clean_ref" and (c == "harm" or not gradient):
                    continue
                q = s[s.policy == p].sort_values("block")
                ax.plot(q.block, q[c], color=col[p], linewidth=1.8 if p not in FIXED else 1.1,
                        zorder=3 if p not in FIXED else 2)
            if i == 0:
                ws, we = int(first.window_start.iloc[0]), int(first.window_end.iloc[0])
                ttl = f"seed {seed}: " + ("no attack" if no_attack else f"attack {ws}-{we}")
                ax.set_title(ttl, color=INK, fontsize=9)
                if not gradient:
                    ax.set_ylim(-0.03, 1.03)
            else:
                ax.set_ylim(0.4 if not gradient else 0.0, 0.9)
            if j == 0:
                ax.set_ylabel(label)
        ax = axes[2, j]
        for r_, p in enumerate(deciders):
            q = s[s.policy == p].sort_values("block")
            on = q.block[q.W.astype(float) == 1].to_numpy()
            ax.broken_barh([(b - 0.5, 1) for b in on], (r_ - 0.35, 0.7), color=col[p], linewidth=0)
        ax.set_yticks(range(len(deciders)), deciders if j == 0 else [])
        ax.set_ylim(-0.6, len(deciders) - 0.4)
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_xlabel("block (2 SGD steps each)")
    axes[2, 0].set_ylabel("Krum on")
    handles = [Line2D([], [], color=col[p], linewidth=1.8 if p not in FIXED else 1.2, label=p) for p in policies
               if not (p == "clean_ref" and not gradient)]
    if not no_attack:
        handles.append(Patch(color=INK2, alpha=0.15, label="attack active"))
    fig.suptitle(f"xp7 deployment (Fashion-MNIST, Krum): {env}: trajectories and Krum decisions (same seed = same data "
                 f"and noise)", fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    top = 1 - 0.8 / fig.get_figheight()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, top + 0.045), ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, top - 0.035))
    fig.savefig(out / f"trajectory_{env.replace(' ', '_').replace('/', 'of')}.png", dpi=120)
    plt.close(fig)


def fig_summary(per_run: pd.DataFrame, policies: list[str], out: Path) -> None:
    envs = [e for e in ENVIRONMENTS if e in set(per_run.environment)]
    pols = [p for p in policies if p != "clean_ref"]
    col = colors(policies)
    panels = [("utility_vs_oracle", "utility − oracle's\n(mean over the run of clean − attack success)"),
              ("mean_clean_acc", "mean clean accuracy"),
              ("defended_blocks", "blocks with Krum on")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 10), sharex=True)
    width = 0.84 / len(pols)
    for ax, (c, label) in zip(axes, panels):
        for e_i, env in enumerate(envs):
            for k, pol in enumerate(pols):
                v = per_run[(per_run.environment == env) & (per_run.policy == pol)][c].to_numpy(float)
                if not len(v):
                    continue
                x = e_i - 0.42 + width * (k + 0.5)
                ax.scatter(np.full(len(v), x), v, s=14, color=col[pol], edgecolor=SURFACE, linewidth=0.6, zorder=3)
                ax.plot([x - width * 0.4, x + width * 0.4], [np.nanmean(v)] * 2, color=col[pol], linewidth=2.2)
        ax.set_ylabel(label, fontsize=8)
    axes[-1].set_xticks(range(len(envs)), [e.replace(" persistent", "\npersistent") for e in envs], fontsize=8)
    handles = [Line2D([], [], color=col[q], marker="o", linewidth=2, label=q) for q in pols]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=8, fontsize=8)
    fig.suptitle("xp7 deployment summary (dots = seeds, bar = mean)", fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "summary_by_policy.png", dpi=120)
    plt.close(fig)


def report(blocks: pd.DataFrame, policies: list[str], out: Path) -> None:
    per_run, agg = summarize(blocks)
    per_run.to_csv(out / "per_run.csv", index=False)
    agg.to_csv(out / "summary.csv", index=False)
    for env in blocks.environment.unique():
        fig_environment(blocks, env, policies, out)
    fig_summary(per_run, policies, out)
    pd.set_option("display.width", 220)
    print(agg.pivot_table(index="environment", columns="policy", values="utility_vs_oracle", sort=False).round(3)
          .to_string())
    print("\nmean over environments (utility - oracle):")
    print(per_run.groupby("policy").utility_vs_oracle.mean().round(4).sort_values(ascending=False).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[], metavar="NAME=LEVEL/MODEL")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="parallel runs (separate processes)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--margin", choices=["zero", "val"], default="zero",
                    help="zero: defend iff predicted gain > 0 (default). val: margin from the validation twins; those "
                         "sit at attack onsets, so it pushes towards always defending")
    args = ap.parse_args()
    scores = pd.read_csv(SCORES) if SCORES.exists() else pd.DataFrame(columns=["level", "model", "margin"])
    models = {}
    for spec in args.models:
        name, folder = spec.split("=")
        level, model = folder.split("/")
        row = scores[(scores.level == level) & (scores.model == model)]
        margin = float(row.margin.iloc[0]) if (len(row) and args.margin == "val") else 0.0
        models[name] = {"folder": folder, "margin": margin}
    policies = FIXED + list(models)
    ctx = {"smoke": args.smoke, "models": models}
    out = OUT / ("smoke" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    envs = ["corner 2/8", "ALIE"] if args.smoke else list(ENVIRONMENTS)
    seeds = 1 if args.smoke else SEEDS
    jobs = [(e, p, k) for e in envs for k in range(seeds) for p in policies
            if not (p == "clean_ref" and ENVIRONMENTS[e][0]["attack"] not in GRADIENT)]
    if args.report_only:
        frames = [pd.read_csv(f, low_memory=False) for f in (OUT / ("runs_smoke" if args.smoke else "runs")).glob("*/blocks.csv")]
    else:
        frames, t0 = [], time.time()
        if args.jobs <= 1:
            for i, (e, p, k) in enumerate(jobs):
                frames.append(run_one(e, p, k, ctx))
                print(f"[{i + 1}/{len(jobs)}] {e} | {p} | k={k}  (elapsed {time.time() - t0:.0f}s)", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
                futs = {pool.submit(run_one, e, p, k, ctx): (e, p, k) for e, p, k in jobs}
                for i, fut in enumerate(as_completed(futs)):
                    frames.append(fut.result())
                    e, p, k = futs[fut]
                    print(f"[{i + 1}/{len(jobs)}] {e} | {p} | k={k}  (elapsed {time.time() - t0:.0f}s)", flush=True)
    (out / "models.json").write_text(json.dumps(models, indent=1))
    report(pd.concat(frames, ignore_index=True), policies, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
