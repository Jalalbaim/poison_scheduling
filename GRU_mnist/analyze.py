"""Step 4: causal analysis of the randomized-defense logs.

    python analyze.py --runs results/xp1/results_collect/runs/all_blocks.csv --out results/xp1/analysis

Q1  When does defending matter?
    (i)  one-step effect: per block b and condition, regress Y[b] - Y[b-1] on W[b]
         (= difference in means, W is a single binary regressor); also pooled by phase.
    (ii) final effect: regress Y[49] on the fraction of defended blocks in 5 phases of 10.
    Bootstrap CIs resample runs (not rows) within each condition.
Q2  Does the one-step effect depend on history X, or only on time (and lr)?
    Ridge regression of Y[b] - Y[b-1] on W, time, W x time, X[b-1]  (M0)
    versus M0 + W x X[b-1]  (M1). Held-out MSE, 5-fold split by run.
Q3  Can the attack be detected from X alone? Logistic regression X[b] -> u_active[b],
    AUC per attack type (and per defense), 5-fold split by seed.
Outputs: figures (PNG), tables (CSV), summary.md.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ATTACKS = ["none", "label_flip", "backdoor", "backdoor_win", "alie", "sign_flip"]
DEFENSES = ["ClipNoise", "Krum", "MultiKrum", "Median", "TrimmedMean", "Bulyan"]
X_COLS = ["x_norm_mean", "x_norm_std", "x_pdist_mean", "x_maxdist_median", "x_cos_avg_median",
          "x_frac_clipped", "x_train_loss", "x_val_acc"]
OUTCOMES = {"y_clean_acc": "clean accuracy", "y_atk": "backdoor success"}
N_PHASES = 5

# Reference palette (validated: light surface, categorical slots 1-6; contrast WARN -> legends + tables).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = dict(zip(DEFENSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9,
    "lines.linewidth": 1.6, "legend.frameon": False,
})


# ----------------------------------------------------------------------------- data
def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values(["run_id", "block"]).reset_index(drop=True)
    sizes = df.groupby("run_id").size()
    if sizes.nunique() != 1:
        raise SystemExit(f"runs have different lengths: {sizes.value_counts().to_dict()}")
    return df


def panel(df: pd.DataFrame, attack: str, defense: str, y: str) -> dict[str, np.ndarray]:
    """Arrays for one condition: Y (R, B+1) incl. block -1, W (R, B), X (R, B, k) for blocks 0..B-1."""
    sub = df[(df.attack == attack) & (df.defense == defense)]
    runs = sorted(sub.run_id.unique())
    sub = sub.set_index(["run_id", "block"]).sort_index()
    Y = sub[y].unstack("block").loc[runs].to_numpy(float)
    W = sub["W"].unstack("block").loc[runs].drop(columns=-1).to_numpy(float)
    X = np.stack([sub[c].unstack("block").loc[runs].drop(columns=-1).to_numpy(float) for c in X_COLS], axis=2)
    seeds = sub["seed"].groupby("run_id").first().loc[runs].to_numpy()
    return {"Y": Y, "W": W, "X": X, "seeds": seeds, "runs": np.array(runs)}


def conditions(df: pd.DataFrame, y: str) -> list[tuple[str, str]]:
    have = df.groupby(["attack", "defense"])[y].apply(lambda s: s.notna().any())
    return [(a, d) for a in ATTACKS for d in DEFENSES if have.get((a, d), False)]


# ----------------------------------------------------------------------------- Q1
def diff_in_means(D: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Per-column E[D | W=1] - E[D | W=0]; works on (..., R, B) arrays."""
    on, off = W.sum(-2), (1 - W).sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (D * W).sum(-2) / on - (D * (1 - W)).sum(-2) / off


def q1(df: pd.DataFrame, y: str, n_boot: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step_rows, phase_rows, final_rows = [], [], []
    for attack, defense in conditions(df, y):
        p = panel(df, attack, defense, y)
        D, W, Y = np.diff(p["Y"], axis=1), p["W"], p["Y"]
        R, B = W.shape
        per_phase = B // N_PHASES
        idx = rng.integers(0, R, size=(n_boot, R))

        est = diff_in_means(D, W)
        boot = diff_in_means(D[idx], W[idx])  # (n_boot, B)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        for b in range(B):
            step_rows.append((attack, defense, b, est[b], lo[b], hi[b]))

        ph = est.reshape(N_PHASES, per_phase).mean(1)
        ph_boot = np.nanmean(boot.reshape(n_boot, N_PHASES, per_phase), axis=2)
        plo, phi = np.nanpercentile(ph_boot, [2.5, 97.5], axis=0)
        for k in range(N_PHASES):
            phase_rows.append((attack, defense, k + 1, ph[k], plo[k], phi[k]))

        F = W.reshape(R, N_PHASES, per_phase).mean(2)
        A = np.column_stack([np.ones(R), F])
        beta = np.linalg.lstsq(A, Y[:, -1], rcond=None)[0]
        bb = np.array([np.linalg.lstsq(A[i], Y[i, -1], rcond=None)[0] for i in idx])
        blo, bhi = np.percentile(bb, [2.5, 97.5], axis=0)
        for k in range(N_PHASES):
            final_rows.append((attack, defense, k + 1, beta[k + 1], blo[k + 1], bhi[k + 1]))

    cols = ["attack", "defense"]
    return (pd.DataFrame(step_rows, columns=[*cols, "block", "effect", "lo", "hi"]),
            pd.DataFrame(phase_rows, columns=[*cols, "phase", "effect", "lo", "hi"]),
            pd.DataFrame(final_rows, columns=[*cols, "phase", "effect", "lo", "hi"]))


# ----------------------------------------------------------------------------- Q2
def q2_design(p: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Rows (run, b) for b = 1..B-1: target Y[b]-Y[b-1], treatment W[b], history X[b-1]."""
    D = np.diff(p["Y"], axis=1)  # D[:, b] = Y[b] - Y[b-1], b = 0..B-1
    R, B = p["W"].shape
    b = np.arange(1, B)
    t = np.tile((b - B / 2) / (B / 2), R)
    T = np.column_stack([t, t**2, t**3])
    w = p["W"][:, 1:].reshape(-1, 1)
    X = p["X"][:, :-1, :].reshape(-1, len(X_COLS))
    X = (X - X.mean(0)) / X.std(0).clip(1e-12)
    designs = {
        "base (no W)": np.column_stack([T, X]),
        "M0: W x time": np.column_stack([w, T, w * T, X]),
        "M1: + W x X": np.column_stack([w, T, w * T, X, w * X]),
    }
    groups = np.repeat(np.arange(R), B - 1)
    return designs, D[:, 1:].reshape(-1), groups


def q2(df: pd.DataFrame, y: str) -> pd.DataFrame:
    rows = []
    for attack, defense in conditions(df, y):
        designs, target, groups = q2_design(panel(df, attack, defense, y))
        fold_mse = {name: [] for name in designs}
        for tr, te in GroupKFold(n_splits=5).split(target, groups=groups):
            for name, A in designs.items():
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(A[tr], target[tr])
                fold_mse[name].append(np.mean((model.predict(A[te]) - target[te]) ** 2))
        m = {k: np.array(v) for k, v in fold_mse.items()}
        gain = m["M0: W x time"] - m["M1: + W x X"]  # > 0: interactions help out of sample
        w_gain = m["base (no W)"] - m["M0: W x time"]
        rows.append({
            "attack": attack, "defense": defense,
            "mse_base": m["base (no W)"].mean(), "mse_M0": m["M0: W x time"].mean(), "mse_M1": m["M1: + W x X"].mean(),
            "rel_gain_M1_vs_M0": 1 - m["M1: + W x X"].mean() / m["M0: W x time"].mean(),
            "gain_se": gain.std(ddof=1) / np.sqrt(len(gain)),
            "gain_mean": gain.mean(),
            "rel_gain_W": 1 - m["M0: W x time"].mean() / m["base (no W)"].mean(),
            "w_gain_mean": w_gain.mean(), "w_gain_se": w_gain.std(ddof=1) / np.sqrt(len(w_gain)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Q3
def q3(df: pd.DataFrame) -> pd.DataFrame:
    trained = df[df.block >= 0]

    def auc(frame: pd.DataFrame) -> float:
        Xm, u, g = frame[X_COLS].to_numpy(float), frame.u_active.to_numpy(int), frame.seed.to_numpy()
        if u.min() == u.max():
            return np.nan
        scores = np.zeros(len(u))
        for tr, te in GroupKFold(n_splits=5).split(Xm, groups=g):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xm[tr], u[tr])
            scores[te] = model.predict_proba(Xm[te])[:, 1]
        return roc_auc_score(u, scores)

    rows = []
    for attack in ATTACKS[1:]:
        both = trained[trained.attack.isin(["none", attack])]
        rows.append({"attack": attack, "defense": "all", "auc": auc(both)})
        for defense in DEFENSES:
            rows.append({"attack": attack, "defense": defense, "auc": auc(both[both.defense == defense])})
    rows.append({"attack": "any", "defense": "all", "auc": auc(trained)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def fig_onestep(step: pd.DataFrame, phase: pd.DataFrame, y: str, out: Path) -> None:
    attacks = [a for a in ATTACKS if a in set(step.attack)]
    fig, axes = plt.subplots(len(attacks), len(DEFENSES), figsize=(15, 2.0 * len(attacks) + 0.8),
                             sharex=True, sharey="row", squeeze=False)
    for i, attack in enumerate(attacks):
        for j, defense in enumerate(DEFENSES):
            ax = axes[i, j]
            s = step[(step.attack == attack) & (step.defense == defense)]
            ph = phase[(phase.attack == attack) & (phase.defense == defense)]
            color = SERIES[defense]
            ax.axhline(0, color=INK2, linewidth=0.8)
            ax.fill_between(s.block, s.lo, s.hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(s.block, s.effect, color=color, linewidth=1.0, alpha=0.7)
            for _, r in ph.iterrows():  # phase-pooled estimate as a step
                x0, x1 = (r.phase - 1) * 10 - 0.5, r.phase * 10 - 0.5
                ax.plot([x0, x1], [r.effect, r.effect], color=INK, linewidth=2.0)
            if i == 0:
                ax.set_title(defense, color=INK, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{attack}\nΔ {OUTCOMES[y]}")
            if i == len(attacks) - 1:
                ax.set_xlabel("block")
    fig.suptitle(f"Q1(i) One-step effect of defending block b on {OUTCOMES[y]}:  "
                 f"E[Y_b − Y_(b−1) | W_b=1] − E[· | W_b=0].  Thin line + band: per block, 95% bootstrap CI "
                 f"(runs resampled).  Black: pooled per phase of 10 blocks.", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / f"q1_onestep_{y}.png", dpi=130)
    plt.close(fig)


def fig_final(final: pd.DataFrame, y: str, out: Path) -> None:
    attacks = [a for a in ATTACKS if a in set(final.attack)]
    cols = min(3, len(attacks))
    rows = int(np.ceil(len(attacks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.0 * rows + 0.6), squeeze=False)
    offsets = np.linspace(-0.25, 0.25, len(DEFENSES))
    for k, attack in enumerate(attacks):
        ax = axes[k // cols, k % cols]
        ax.axhline(0, color=INK2, linewidth=0.8)
        for off, defense in zip(offsets, DEFENSES):
            s = final[(final.attack == attack) & (final.defense == defense)]
            ax.errorbar(s.phase + off, s.effect, yerr=[s.effect - s.lo, s.hi - s.effect], color=SERIES[defense],
                        marker="o", markersize=4, linewidth=1.4, capsize=0, label=defense)
        ax.set_title(attack, color=INK, fontweight="bold")
        ax.set_xticks(range(1, N_PHASES + 1), [f"{10 * (p - 1)}–{10 * p - 1}" for p in range(1, N_PHASES + 1)])
        ax.set_xlabel("phase (blocks)")
        ax.set_ylabel(f"Δ final {OUTCOMES[y]}")
    for k in range(len(attacks), rows * cols):
        axes[k // cols, k % cols].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(DEFENSES), fontsize=8)
    fig.suptitle(f"Q1(ii) Effect on final {OUTCOMES[y]} of defending a whole phase (vs never), "
                 f"OLS on phase fractions; 95% bootstrap CI", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / f"q1_final_{y}.png", dpi=130)
    plt.close(fig)


def heatmap(ax, table: pd.DataFrame, cmap, vmin: float, vmax: float, fmt: str, title: str) -> None:
    im = ax.imshow(table.to_numpy(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=0)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(table.to_numpy(float)):
        if not np.isnan(v):
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8, color=INK)
    ax.set_title(title, color=INK, loc="left")
    return im


def fig_q2(q2s: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(len(q2s), 1, figsize=(9, 2.6 * len(q2s) + 0.4), squeeze=False)
    for ax, (y, t) in zip(axes[:, 0], q2s.items()):
        # Cell = gain of M1 over M0 in units of its fold standard error (|z| > 2 ~ significant).
        z = (t.gain_mean / t.gain_se).to_numpy()
        tab = pd.DataFrame({"attack": t.attack, "defense": t.defense, "z": z}).pivot(
            index="attack", columns="defense", values="z").reindex(
            index=[a for a in ATTACKS if a in set(t.attack)], columns=DEFENSES)
        im = heatmap(ax, tab, DIVERGING, -4, 4, "+.1f",
                     f"{OUTCOMES[y]}: held-out MSE gain of M1 (+ W×X) over M0 (W×time), in fold-SE units")
        fig.colorbar(im, ax=ax, shrink=0.8, label="z  (>0: history helps)")
    fig.tight_layout()
    fig.savefig(out / "q2_interactions.png", dpi=130)
    plt.close(fig)


def fig_q3(t: pd.DataFrame, out: Path) -> None:
    tab = t[t.attack != "any"].pivot(index="attack", columns="defense", values="auc").reindex(
        index=ATTACKS[1:], columns=["all", *DEFENSES])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = heatmap(ax, tab, SEQUENTIAL, 0.5, 1.0, ".2f",
                 "Q3  Detecting an active attack from X alone: held-out AUC (split by seed)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUC")
    fig.tight_layout()
    fig.savefig(out / "q3_detection_auc.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- summary
def fmt_ci(r: pd.Series, scale: float = 100) -> str:
    return f"{scale * r.effect:+.1f} [{scale * r.lo:+.1f}, {scale * r.hi:+.1f}]"


def summary(df: pd.DataFrame, res: dict, out: Path) -> str:
    n_runs = df.run_id.nunique()
    per_cond = df.groupby(["attack", "defense"]).run_id.nunique()
    shared = df[df.block >= 0].groupby("seed").apply(lambda g: g.groupby("run_id").W.apply(tuple).nunique()).max()
    L = [
        "# Step 4 summary: randomized defense schedules",
        "",
        f"Data: {n_runs} runs, {per_cond.min()}–{per_cond.max()} runs per (attack, defense), "
        f"{df.block.max() + 1} blocks per run. Effects are in **percentage points**; brackets are 95% bootstrap "
        "CIs resampling runs.",
        "",
        "Design note: the schedule seed is `1000 + seed`, so a given seed has the same W schedule in every "
        f"condition (max distinct schedules per seed: {shared}). Within a condition the 20 schedules are "
        "independent, which is what every estimate below relies on; conditions are not independent replicates "
        "of each other.",
        "",
    ]

    # Q1 ---------------------------------------------------------------------
    L += ["## Q1. When does defending matter?", ""]
    for y in res["q1"]:
        final = res["q1"][y][2]
        L += [f"### Final {OUTCOMES[y]}: effect of defending a whole phase (vs never)", "",
              "| attack | defense | " + " | ".join(f"blocks {10 * (p - 1)}–{10 * p - 1}" for p in range(1, 6)) + " |",
              "|---|---|" + "---|" * 5]
        for (a, d), g in final.groupby(["attack", "defense"], sort=False):
            cells = []
            for _, r in g.sort_values("phase").iterrows():
                sig = (r.lo > 0) or (r.hi < 0)
                cells.append(("**" + fmt_ci(r) + "**") if sig else fmt_ci(r))
            L.append(f"| {a} | {d} | " + " | ".join(cells) + " |")
        L += ["", "Bold: CI excludes 0.", ""]
        phase = res["q1"][y][1]
        top = phase.assign(sig=(phase.lo > 0) | (phase.hi < 0))
        n_sig = int(top.sig.sum())
        L += [f"One-step effect on {OUTCOMES[y]} (pooled per phase): {n_sig} of {len(top)} (condition, phase) "
              "cells have a CI excluding 0. Largest effects:", ""]
        for _, r in top.reindex(top.effect.abs().sort_values(ascending=False).index).head(6).iterrows():
            L.append(f"- {r.attack} / {r.defense}, phase {r.phase}: {fmt_ci(r, 100)} pts per defended block")
        L.append("")

    # Q2 ---------------------------------------------------------------------
    L += ["## Q2. Does the effect depend on history X?", "",
          "M0 lets the effect of W vary with time only (W × cubic in block; lr is constant in this "
          "dataset, so it cannot be separated from time). M1 adds W × X[b−1]. Held-out MSE, 5 folds split "
          "by run. z = mean fold gain / its standard error.", ""]
    for y, t in res["q2"].items():
        z = t.gain_mean / t.gain_se
        zw = t.w_gain_mean / t.w_gain_se
        L += [f"**{OUTCOMES[y]}**: W itself improves held-out error (z > 2) in {int((zw > 2).sum())}/{len(t)} "
              f"conditions. Adding W × X improves it (z > 2) in {int((z > 2).sum())}/{len(t)}, "
              f"hurts (z < −2) in {int((z < -2).sum())}/{len(t)}; median relative MSE change "
              f"{100 * t.rel_gain_M1_vs_M0.median():+.2f}%.", ""]
        best = t.assign(z=z).sort_values("z", ascending=False).head(5)
        L += ["| attack | defense | MSE gain from W (%) | MSE gain from W×X (%) | z (W×X) |", "|---|---|---|---|---|"]
        for _, r in best.iterrows():
            L.append(f"| {r.attack} | {r.defense} | {100 * r.rel_gain_W:+.2f} | {100 * r.rel_gain_M1_vs_M0:+.2f} | "
                     f"{r.z:+.1f} |")
        L.append("")

    # Q3 ---------------------------------------------------------------------
    t = res["q3"]
    L += ["## Q3. Can an attack be detected from X alone?", "",
          "Logistic regression X[b] → u_active[b] on the attacked runs plus the no-attack runs; "
          "held-out AUC with 5 folds split by seed (a seed's runs are never in train and test at once).", "",
          "| attack | all defenses | " + " | ".join(DEFENSES) + " |", "|---|---|" + "---|" * len(DEFENSES)]
    for a in ATTACKS[1:]:
        g = t[t.attack == a].set_index("defense").auc
        L.append(f"| {a} | **{g['all']:.3f}** | " + " | ".join(f"{g[d]:.3f}" for d in DEFENSES) + " |")
    L += ["", f"Any attack vs none, all conditions pooled: AUC {t[t.attack == 'any'].auc.iloc[0]:.3f}.", ""]

    L += ["## Files", "",
          "- `q1_onestep_<y>.png`, `q1_final_<y>.png`, `q2_interactions.png`, `q3_detection_auc.png`",
          "- tables: `q1_onestep_<y>.csv`, `q1_phase_<y>.csv`, `q1_final_<y>.csv`, `q2_<y>.csv`, `q3.csv`", ""]
    text = "\n".join(L)
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("results/xp1/results_collect/runs/all_blocks.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/xp1/analysis"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = load(args.runs)

    res: dict = {"q1": {}, "q2": {}}
    for y in OUTCOMES:
        step, phase, final = q1(df, y, args.boot, rng)
        res["q1"][y] = (step, phase, final)
        step.to_csv(args.out / f"q1_onestep_{y}.csv", index=False)
        phase.to_csv(args.out / f"q1_phase_{y}.csv", index=False)
        final.to_csv(args.out / f"q1_final_{y}.csv", index=False)
        fig_onestep(step, phase, y, args.out)
        fig_final(final, y, args.out)
        res["q2"][y] = q2(df, y)
        res["q2"][y].to_csv(args.out / f"q2_{y}.csv", index=False)
        print(f"done {y}", flush=True)
    fig_q2(res["q2"], args.out)
    res["q3"] = q3(df)
    res["q3"].to_csv(args.out / "q3.csv", index=False)
    fig_q3(res["q3"], args.out)
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(summary(df, res, args.out))


if __name__ == "__main__":
    main()
