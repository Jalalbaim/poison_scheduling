"""xp7 trajectory plots (layout of GRU_mnist/deploy_xp5.py): per attack family, 3 seeds (val/test setups, which have
an always-off twin), each run under the 4 policies (same seed = same data and noise).

    ../.venv/Scripts/python plot_xp7.py            # -> results/xp7_plots/trajectory_<family>.png, twins.png

Rows: attack success (backdoor / flip: y_atk; ALIE / SignFlip: no-attack reference clean accuracy at the same block
and W, minus clean accuracy), clean accuracy, defended blocks. Grey band = attack active.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "GRU_mnist"))
from analyze import INK, INK2  # noqa: E402  (also sets the GRU_mnist figure style)

RES = HERE / "results"
SRC = {"a": RES / "xp7_a/results/xp7_a", "b": RES / "xp7_b/results/xp7_b", "twins": RES / "xp7_twins/results/xp7_twins"}
OUT = RES / "xp7_plots"
COLS = ["run_id", "block", "W", "y_clean_acc", "y_atk", "u_active"]
POLICIES = ["off", "g0", "g2", "g8", "hard"]
COLOR = {"off": "#1baf7a", "g0": "#8a94a6", "g2": "#9ec2ee", "g8": "#2a78d6", "hard": "#eb6834"}
LABEL = {"off": "always off (twin)", "g0": "γ = 0 (coin flip)", "g2": "γ = 2", "g8": "γ = 8", "hard": "hard alarm rule"}
FAMILY = {"corner": "backdoor corner 2/8", "patch": "backdoor patch 8/8", "flip": "targeted flip 8/8", "alie": "ALIE",
          "signflip": "SignFlip (scale 4.6)", "none": "no attack"}
TWIN_COLOR = {"off": "#1baf7a", 0: "#104281", 5: "#2a78d6", 10: "#9ec2ee"}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for part in ("a", "b"):
        d = pd.read_csv(SRC[part] / "all_blocks.csv", usecols=COLS, low_memory=False)
        frames.append(d.merge(pd.read_csv(SRC[part] / "manifest.csv"), on="run_id"))
    main = pd.concat(frames)
    tw = pd.read_csv(SRC["twins"] / "all_blocks.csv", usecols=COLS, low_memory=False)
    tw = tw.merge(pd.read_csv(SRC["twins"] / "manifest.csv"), on="run_id")
    main, tw = main[main.block >= 0].copy(), tw[tw.block >= 0].copy()
    for d in (main, tw):
        d["W"] = d.W.astype(float)
    ref = main[main.family == "none"].groupby(["block", "W"]).y_clean_acc.mean()
    for d in (main, tw):
        r = ref.reindex(pd.MultiIndex.from_frame(d[["block", "W"]])).to_numpy()
        d["harm"] = np.where(d.family.isin(["alie", "signflip"]), r - d.y_clean_acc, d.y_atk)
    return main, tw


def pick_seeds(tw: pd.DataFrame, family: str) -> list[int]:
    """3 val/test setups of the family: a persistent one first if any, then early and late onsets."""
    s = tw[(tw.family == family) & (tw.plan == "off")].drop_duplicates("base").sort_values("onset")
    out = list(s[s.persistent].base[:1])
    rest = s[~s.base.isin(out)]
    for q in (0.2, 0.8, 0.5):
        if len(out) == 3 or rest.empty:
            break
        b = rest.iloc[int(q * (len(rest) - 1))].base
        out.append(b)
        rest = rest[rest.base != b]
    return out


def fig_family(main: pd.DataFrame, tw: pd.DataFrame, family: str) -> None:
    bases = pick_seeds(tw, family)
    fig, axes = plt.subplots(3, len(bases), figsize=(3.9 * len(bases) + 0.6, 8.6), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1, 0.6]})
    gradient = family in ("alie", "signflip")
    for j, base in enumerate(bases):
        runs = {lv: main[(main.base == base) & (main.level == lv)].sort_values("block") for lv in POLICIES[1:]}
        runs["off"] = tw[(tw.base == base) & (tw.plan == "off")].sort_values("block")
        first = runs["off"]
        active = first.u_active.to_numpy().astype(bool)
        for i, (col, label) in enumerate([("harm", "clean acc. lost vs no attack" if gradient else "attack success"),
                                          ("y_clean_acc", "clean accuracy")]):
            ax = axes[i, j]
            if family != "none" and not (i == 0 and family == "none"):
                for b in np.flatnonzero(active):
                    ax.axvspan(b - 0.5, b + 0.5, color=INK2, alpha=0.08, linewidth=0)
            for p in POLICIES:
                q = runs[p]
                ax.plot(q.block, q[col], color=COLOR[p], linewidth=1.8 if p in ("g8", "hard") else 1.2)
            if i == 0:
                row = first.iloc[0]
                win = "persistent" if row.persistent else "window"
                ttl = (f"seed {row.seed}: no attack" if family == "none"
                       else f"seed {row.seed}: {win} {int(row.onset)}-{int(first[first.u_active == 1].block.max())}")
                ttl += "\non: " + ", ".join(f"{p} {int(runs[p].W.sum())}" for p in POLICIES[1:])
                ax.set_title(ttl, color=INK, fontsize=9)
                if not gradient:
                    ax.set_ylim(-0.03, 1.03)
            else:
                ax.set_ylim(0.0 if gradient else 0.4, 0.9)
            if j == 0:
                ax.set_ylabel(label)
        ax = axes[2, j]
        for r_, p in enumerate(POLICIES[1:]):
            on = runs[p].block[runs[p].W == 1].to_numpy()
            ax.broken_barh([(b - 0.5, 1) for b in on], (r_ - 0.35, 0.7), color=COLOR[p], linewidth=0)
        ax.set_yticks(range(4), ["γ 0", "γ 2", "γ 8", "hard"] if j == 0 else [])
        ax.set_ylim(-0.6, 3.6)
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_xlabel("block (2 SGD steps each)")
    axes[2, 0].set_ylabel("Krum on")
    handles = [Line2D([], [], color=COLOR[p], linewidth=1.6, label=LABEL[p]) for p in POLICIES]
    if family != "none":
        handles.append(Patch(color=INK2, alpha=0.15, label="attack active"))
    fig.suptitle(f"xp7 (Fashion-MNIST, Krum): {FAMILY[family]}: trajectories under the 4 policies (same seed = same "
                 f"data and noise)", fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    top = 1 - 0.8 / fig.get_figheight()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, top + 0.045), ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, top - 0.035))
    fig.savefig(OUT / f"trajectory_{family}.png", dpi=120)
    plt.close(fig)


def fig_twins(tw: pd.DataFrame) -> None:
    """One attacked setup per family: always off vs Krum on for 20 blocks from onset + d (d = 0, 5, 10)."""
    fams = ["corner", "patch", "flip", "alie", "signflip", "none"]
    fig, axes = plt.subplots(2, len(fams), figsize=(3.3 * len(fams), 6.2), sharex=False, squeeze=False)
    for j, fam in enumerate(fams):
        s = tw[(tw.family == fam) & ~tw.persistent]
        base = s.drop_duplicates("base").sort_values("onset").iloc[len(s.base.unique()) // 2].base
        s = s[s.base == base]
        onset = int(s.onset.iloc[0])
        lo, hi = max(0, onset - 15), min(249, onset + 60)
        gradient = fam in ("alie", "signflip")
        for i, col in enumerate(["harm", "y_clean_acc"]):
            ax = axes[i, j]
            off = s[s.plan == "off"].sort_values("block")
            if fam != "none":
                for b in np.flatnonzero(off.u_active.to_numpy().astype(bool)):
                    if lo <= b <= hi:
                        ax.axvspan(b - 0.5, b + 0.5, color=INK2, alpha=0.08, linewidth=0)
            for key, q in [("off", off)] + [(int(dd), s[(s.plan == "on") & (s.d == dd)].sort_values("block"))
                                            for dd in (0, 5, 10)]:
                q = q[(q.block >= lo) & (q.block <= hi)]
                ax.plot(q.block, q[col], color=TWIN_COLOR[key], linewidth=1.5)
                if key != "off" and i == 0:
                    t = onset + key
                    ax.axvline(t - 0.5, color=TWIN_COLOR[key], linewidth=0.8, linestyle=":")
            if i == 0:
                ax.set_title(f"{FAMILY[fam]}\nseed {int(s.seed.iloc[0])}, onset {onset}", color=INK, fontsize=9)
                if not gradient:
                    ax.set_ylim(-0.03, 1.03)
                ax.set_ylabel("clean acc. lost" if gradient else ("attack success" if j == 0 else ""))
            else:
                ax.set_ylim(0.0 if gradient else 0.5, 0.9)
                ax.set_xlabel("block")
                if j == 0:
                    ax.set_ylabel("clean accuracy")
    handles = [Line2D([], [], color=TWIN_COLOR["off"], linewidth=1.6, label="always off")]
    handles += [Line2D([], [], color=TWIN_COLOR[dd], linewidth=1.6, label=f"Krum on 20 blocks from onset + {dd}")
                for dd in (0, 5, 10)]
    handles.append(Patch(color=INK2, alpha=0.15, label="attack active"))
    fig.suptitle("xp7 onset twins: the same run with Krum switched on 0 / 5 / 10 blocks after the attack starts "
                 "(dotted line = switch on)", fontsize=10, color=INK, x=0.01, y=0.995, ha="left")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "twins.png", dpi=120)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_df, tw = load()
    for fam in FAMILY:
        fig_family(main_df, tw, fam)
    fig_twins(tw)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
