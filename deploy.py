"""Deploy the frozen GRU as a defense controller, against always-on and always-off.

    python deploy.py                 # 2 cases x 3 policies x 5 fresh seeds, then summary + figures
    python deploy.py --report-only   # rebuild summary + figures from finished runs

Cases: backdoor + Krum, ALIE + Bulyan (the least harmful defense under ALIE in Step 2).
Seeds 100-104: never seen by the GRU (it was trained on seeds 0-13 of xp1).
Everything else is the xp1 configuration the GRU was trained on.

GRU policy (receding horizon, decides from block 0). Before each block b, the GRU reads the
observable history (blocks -1..b-1: X signals, W, defense; never outcomes or attack labels)
and predicts the final outcomes under 4 candidate plans for blocks b..49:
    on      defend from now on          off     never defend from now on
    on1     defend this block, then stop        off1    skip this block, then defend
It applies the first action of the plan with the best score
    score = predicted final clean accuracy - LAMBDA * predicted final backdoor success.
The same score is used in both cases: the controller does not know which attack it faces.
The score is a complementary metric; results are reported on standard metrics first.
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

import poison_testbed  # noqa: E402,F401  (CUBLAS_WORKSPACE_CONFIG)
from analyze import INK, SURFACE, X_COLS  # noqa: E402  (also applies the shared plot style)
from gru_experiment import Data, OutcomeGRU, assign_splits  # noqa: E402
from poison_testbed.runner import RunConfig, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parent
XP1_RUNS = ROOT / "results/xp1/results_collect/runs/all_blocks.csv"
WEIGHTS = ROOT / "results/xp1/gru/gru.pt"
WEIGHTS_SHA = "467f5f9ea8669025"
XP1_C, XP1_SIGMA = 3.0552712798118593, 0.01
OUT = ROOT / "results/deploy"
N_ROWS = 51  # xp1-shaped runs: block -1 .. 49

CASES = {"backdoor_Krum": ("backdoor", "Krum"), "alie_Bulyan": ("alie", "Bulyan")}
# Clean reference per seed (no attack, no defense): the baseline for attacks without a trigger (ALIE).
REFERENCE = "reference"
ATTACK_SUCCESS_LABEL = {
    "backdoor_success": "attack success\n(backdoor success rate)",
    "accuracy_lost": "attack success\n(accuracy lost vs clean run)",
}
POLICIES = ["gru", "on", "off"]
SEEDS = [100, 101, 102, 103, 104]
PLANS = ["on", "off", "on1", "off1"]
LAMBDA = 1.0

# Validated categorical slots 1-3 of the reference palette (see analyze.py).
POLICY_COLOR = {"gru": "#2a78d6", "on": "#eb6834", "off": "#1baf7a"}
POLICY_LABEL = {"gru": "GRU policy", "on": "always on", "off": "always off"}


# ----------------------------------------------------------------------------- controller
class GRUController:
    """Stateless per call: re-encodes the observable history and scores the candidate plans."""

    def __init__(self, x_mean: np.ndarray, x_std: np.ndarray, device: str):
        self.x_mean, self.x_std, self.device = x_mean, x_std, torch.device(device)
        probe = self._data([{"run_id": "probe", "seed": 0, "defense": "Krum", "block": -1, "W": np.nan}])
        self.model = OutcomeGRU(probe.enc.shape[-1], probe.dec_inputs(probe.W).shape[-1]).to(self.device)
        self.model.load_state_dict(torch.load(WEIGHTS, map_location=self.device))
        self.model.eval()

    def _data(self, history: list[dict]) -> Data:
        """Pad the history to 51 rows (the encoder is causal, so padding never affects the state we read)."""
        df = pd.DataFrame(history)
        last = int(df.block.max())
        pad = pd.DataFrame([{"run_id": df.run_id.iloc[0], "seed": df.seed.iloc[0], "defense": df.defense.iloc[0],
                             "block": b} for b in range(last + 1, N_ROWS - 1)])
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
        rest = N_ROWS - 1 - b  # blocks b..49
        plans = {
            "on": [1] * rest, "off": [0] * rest,
            "on1": [1] + [0] * (rest - 1), "off1": [0] + [1] * (rest - 1),
        }
        W = np.zeros((len(PLANS), N_ROWS))
        for k, p in enumerate(PLANS):
            W[k, b + 1:] = plans[p]  # row i = block i-1
        dec = data.dec_inputs(W)[:, b:]  # decoder step for row i+1 is dec index i -> blocks b..49
        pred = self.model.decode(h.expand(len(PLANS), -1), dec.to(self.device)).cpu().numpy()  # (4, rest, 2)
        final = pred[:, -1]
        score = final[:, 0] - LAMBDA * final[:, 1]
        best = int(np.argmax(score))
        extra = {"gru_plan": PLANS[best]}
        for k, p in enumerate(PLANS):
            extra[f"gru_pred_clean_final_{p}"] = float(final[k, 0])
            extra[f"gru_pred_atk_final_{p}"] = float(final[k, 1])
            extra[f"gru_score_{p}"] = float(score[k])
        extra["gru_pred_clean_next"] = float(pred[best, 0, 0])
        extra["gru_pred_atk_next"] = float(pred[best, 0, 1])
        return plans[PLANS[best]][0], extra


# ----------------------------------------------------------------------------- runs
def config(case: str, policy: str, seed: int) -> RunConfig:
    if case == REFERENCE:
        return RunConfig(attack="none", defense="Average", seed=seed, C=XP1_C, sigma=XP1_SIGMA, schedule="off")
    attack, defense = CASES[case]
    schedule = "policy" if policy == "gru" else policy  # "on" / "off" are fixed schedules
    return RunConfig(attack=attack, defense=defense, seed=seed, C=XP1_C, sigma=XP1_SIGMA, schedule=schedule)


def run_one(case: str, policy: str, seed: int, x_mean: np.ndarray, x_std: np.ndarray) -> tuple[str, float]:
    cfg = config(case, policy, seed)
    run_id = f"{case}__{policy}__s{seed:03d}"
    out = OUT / "runs" / run_id
    if (out / "blocks.csv").exists():
        return run_id, json.loads((out / "config.json").read_text())["wall_clock_s"]
    controller = GRUController(x_mean, x_std, cfg.device) if policy == "gru" else None
    started = time.time()
    df = simulate(cfg, controller)
    wall = time.time() - started
    df.insert(0, "case", case)
    df.insert(1, "policy", policy)
    df["run_id"] = run_id
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "blocks.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "case": case, "policy": policy, "config": cfg.__dict__, "lambda": LAMBDA,
        "plans": PLANS, "gru_weights_sha256": WEIGHTS_SHA, "wall_clock_s": round(wall, 1),
        "device_name": torch.cuda.get_device_name(0) if cfg.device == "cuda" else cfg.device,
    }, indent=2, default=str))
    return run_id, wall


# ----------------------------------------------------------------------------- report
def summarize(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run_id, g in blocks.groupby("run_id"):
        g = g.sort_values("block")
        tr = g[g.block >= 0]
        w = tr.W.astype(int).to_numpy()
        final = tr.iloc[-1]
        atk = final.y_atk
        rows.append({
            "case": final.case, "attack": final.attack, "defense": final.defense, "policy": final.policy,
            "seed": int(final.seed), "run_id": run_id,
            # standard metrics first
            "final_clean_acc": final.y_clean_acc, "final_backdoor_success": atk,
            "last10_clean_acc": tr.y_clean_acc.iloc[-10:].mean(),  # robust to block-to-block swings (ALIE)
            "attack_success_def": final.attack_success_def,
            "final_attack_success": final.attack_success,
            "last10_attack_success": tr.attack_success.iloc[-10:].mean(),
            "mean_attack_success": tr.attack_success.mean(),
            "mean_clean_acc": tr.y_clean_acc.mean(),
            "mean_backdoor_success": tr.y_atk.mean() if tr.y_atk.notna().any() else np.nan,
            "defended_blocks": int(w.sum()), "switches": int(np.abs(np.diff(w)).sum()),
            "first_defended_block": int(np.argmax(w)) if w.any() else -1,
            # complementary: the controller's objective evaluated on the true outcomes
            "score_true": final.y_clean_acc - LAMBDA * (0.0 if pd.isna(atk) else atk),
        })
    per_run = pd.DataFrame(rows).sort_values(["case", "policy", "seed"])
    metrics = ["final_clean_acc", "last10_clean_acc", "final_backdoor_success", "final_attack_success",
               "last10_attack_success", "mean_attack_success", "mean_clean_acc", "defended_blocks", "switches",
               "score_true"]
    by_policy = per_run.groupby(["case", "policy"])[metrics].agg(["mean", "std"]).round(4)
    by_policy.columns = [f"{m}_{s}" for m, s in by_policy.columns]
    return per_run, by_policy.reset_index()


def shade_defended(ax, w: np.ndarray, color: str) -> None:
    for b in np.flatnonzero(w):
        ax.axvspan(b - 0.5, b + 0.5, color=color, alpha=0.13, linewidth=0)


def fig_trajectories(blocks: pd.DataFrame, case: str) -> None:
    g = blocks[blocks.case == case]
    has_atk = g.y_atk.notna().any()
    metrics = [("y_clean_acc", "clean accuracy")] + ([("y_atk", "backdoor success")] if has_atk else [])
    attack, defense = CASES[case]
    policy_grid(g, metrics, f"{attack} + {defense}: trajectories under the GRU policy vs always-on / always-off "
                f"(same seed = same data and noise)", OUT / f"trajectory_{case}.png")


def fig_attack_success(blocks: pd.DataFrame, case: str) -> None:
    """Same layout as the trajectory figure, with the attack success rate in place of clean accuracy."""
    g = blocks[blocks.case == case]
    attack, defense = CASES[case]
    label = ATTACK_SUCCESS_LABEL[g.attack_success_def.dropna().iloc[0]]
    note = ("" if g.y_atk.notna().any() else
            "  ALIE has no trigger: success = clean accuracy lost vs a no-attack, no-defense run on the same seed "
            "(includes the defense's own cost; 0 = attack neutralised)")
    policy_grid(g, [("attack_success", label)], f"{attack} + {defense}: attack success under the GRU policy vs "
                f"always-on / always-off.{note}", OUT / f"attack_success_{case}.png", zero_line=True)


def policy_grid(g: pd.DataFrame, metrics: list[tuple[str, str]], title: str, path: Path,
                zero_line: bool = False) -> None:
    """One column per seed, one row per metric: the 3 policies, GRU-defended blocks shaded, switches marked."""
    seeds = sorted(g.seed.unique())
    fig, axes = plt.subplots(len(metrics), len(seeds), figsize=(3.3 * len(seeds), 2.7 * len(metrics) + 0.9),
                             sharex=True, sharey="row", squeeze=False)
    for j, seed in enumerate(seeds):
        s = g[g.seed == seed]
        gru = s[(s.policy == "gru") & (s.block >= 0)].sort_values("block")
        w = gru.W.astype(int).to_numpy()
        for i, (col, label) in enumerate(metrics):
            ax = axes[i, j]
            shade_defended(ax, w, POLICY_COLOR["gru"])
            if zero_line:
                ax.axhline(0, color=INK, linewidth=0.8, zorder=1)
            for policy in ["off", "on", "gru"]:
                p = s[s.policy == policy].sort_values("block")
                ax.plot(p.block, p[col], color=POLICY_COLOR[policy], linewidth=2.0 if policy == "gru" else 1.4,
                        label=POLICY_LABEL[policy], zorder=3 if policy == "gru" else 2)
            switches = gru.block.to_numpy()[np.flatnonzero(np.diff(np.r_[0, w]))]
            for b in switches:
                y = gru.loc[gru.block == b, col].iloc[0]
                on = w[b] == 1
                ax.plot(b, y, marker="^" if on else "v", markersize=7, color=POLICY_COLOR["gru"],
                        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
            if i == 0:
                ax.set_title(f"seed {seed}: GRU defended {w.sum()} / {len(w)} blocks", color=INK)
            if j == 0:
                ax.set_ylabel(label)
            if i == len(metrics) - 1:
                ax.set_xlabel("block (10 SGD steps each)")
            if col == "y_clean_acc":
                ax.set_ylim(max(0.0, g[g.block >= 2][col].min() - 0.02), min(1.0, g[col].max() + 0.01))
    handles, _ = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles += [Patch(color=POLICY_COLOR["gru"], alpha=0.13, label="block defended by GRU"),
                Line2D([], [], marker="^", linestyle="", color=POLICY_COLOR["gru"], label="GRU switches defense on"),
                Line2D([], [], marker="v", linestyle="", color=POLICY_COLOR["gru"], label="GRU switches defense off")]
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    top = 1 - 0.75 / fig.get_figheight()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, top + 0.04), ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, top - 0.02))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_predicted_vs_actual(blocks: pd.DataFrame, case: str) -> None:
    g = blocks[(blocks.case == case) & (blocks.policy == "gru") & (blocks.block >= 0)]
    has_atk = g.y_atk.notna().any()
    cols = [("clean", "y_clean_acc", "final clean accuracy")] + ([("atk", "y_atk", "final backdoor success")]
                                                                  if has_atk else [])
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 4.2), squeeze=False)
    for ax, (key, col, label) in zip(axes[0], cols):
        for seed, s in g.groupby("seed"):
            s = s.sort_values("block")
            chosen = np.array([s[f"gru_pred_{key}_final_{p}"].iloc[k] for k, p in enumerate(s.gru_plan)])
            line, = ax.plot(s.block, chosen, linewidth=1.4, label=f"seed {seed}")
            ax.axhline(s[col].iloc[-1], color=line.get_color(), linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_xlabel("decision block")
        ax.set_ylabel(f"predicted {label}\n(chosen plan)")
        ax.set_title(f"{label}: GRU prediction at each decision (solid) vs actual (dashed)", color=INK, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / f"predicted_vs_actual_{case}.png", dpi=130)
    plt.close(fig)


def fig_finals(per_run: pd.DataFrame) -> None:
    cases = list(CASES)
    panels = [(c, "final_clean_acc", "final clean accuracy") for c in cases] + \
             [(c, "final_backdoor_success", "final backdoor success") for c in cases
              if per_run[per_run.case == c].final_backdoor_success.notna().any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.4), squeeze=False)
    for ax, (case, col, label) in zip(axes[0], panels):
        s = per_run[per_run.case == case]
        for k, policy in enumerate(POLICIES):
            v = s[s.policy == policy][col].to_numpy()
            ax.scatter(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, s=22, color=POLICY_COLOR[policy],
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
            ax.plot([k - 0.25, k + 0.25], [v.mean()] * 2, color=INK, linewidth=2)
        ax.set_xticks(range(len(POLICIES)), [POLICY_LABEL[p] for p in POLICIES])
        ax.set_title(f"{case}: {label}", color=INK, loc="left")
    fig.suptitle("Final outcomes per policy (dots: seeds 100-104; bar: mean)", fontsize=10, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "finals_by_policy.png", dpi=130)
    plt.close(fig)


def add_attack_success(blocks: pd.DataFrame) -> pd.DataFrame:
    """attack_success: backdoor success rate if the attack has a trigger, else clean accuracy lost vs the
    same seed's clean reference run (no attack, no defense; same data and noise)."""
    ref = blocks[blocks.case == REFERENCE][["seed", "block", "y_clean_acc"]].rename(columns={"y_clean_acc": "ref_acc"})
    out = blocks[blocks.case != REFERENCE].merge(ref, on=["seed", "block"], how="left")
    trigger = out.groupby("case").y_atk.transform(lambda s: s.notna().any())
    out["attack_success_def"] = np.where(trigger, "backdoor_success", "accuracy_lost")
    out["attack_success"] = np.where(trigger, out.y_atk, out.ref_acc - out.y_clean_acc)
    missing = out.loc[~trigger, "ref_acc"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} rows lack a clean reference run: run deploy.py without --report-only")
    return out


def report() -> None:
    files = sorted((OUT / "runs").glob("*/blocks.csv"))
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    blocks = add_attack_success(raw)
    pd.concat([blocks, raw[raw.case == REFERENCE]], ignore_index=True).to_csv(OUT / "blocks_all.csv", index=False)
    per_run, by_policy = summarize(blocks)
    per_run.to_csv(OUT / "summary_runs.csv", index=False)
    by_policy.to_csv(OUT / "summary_by_policy.csv", index=False)
    for case in CASES:
        if (blocks.case == case).any():
            fig_trajectories(blocks, case)
            fig_attack_success(blocks, case)
            fig_predicted_vs_actual(blocks, case)
    fig_finals(per_run)
    pd.set_option("display.width", 220)
    print(by_policy.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()[:16]
        if sha != WEIGHTS_SHA:
            raise SystemExit(f"GRU weights changed: sha {sha}, expected the frozen {WEIGHTS_SHA}")
        xp1 = pd.read_csv(XP1_RUNS)
        stats = Data(xp1[assign_splits(xp1) == "train"])  # the GRU's own input normalisation
        jobs = [(c, p, s) for c in CASES for p in POLICIES for s in SEEDS]
        jobs += [(REFERENCE, "off", s) for s in SEEDS]  # clean reference per seed (finished runs are skipped)
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
            futs = [pool.submit(run_one, c, p, s, stats.x_mean, stats.x_std) for c, p, s in jobs]
            for k, fut in enumerate(as_completed(futs), 1):
                run_id, wall = fut.result()
                print(f"[{k}/{len(jobs)}] {run_id}  {wall:.0f}s  (elapsed {time.time() - started:.0f}s)", flush=True)
    report()


if __name__ == "__main__":
    main()
