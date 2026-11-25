"""xp8 live test: the chosen seed of model A and of model B decides Krum at every step of new runs (test setups).

    ../.venv/Scripts/python xp8/live8.py --jobs 4        # -> results/xp8/learn/{A,B}/live/, learn/live_compare.*
    ../.venv/Scripts/python xp8/live8.py --smoke         # 6 test setups x 40 steps (pipeline check)

Each test setup (seed 0 of each attack configuration + the no-attack test setups) is run again from scratch with
the model in charge. At step t: the 16 gradients arrive, the signals are computed (before aggregation), the model
gives G_hat_t, W_t = 1 iff G_hat_t > 0, then aggregation and update. Same seed as the xp8 trunks, so "Krum always
off" and "Krum always on" are the trunks themselves (no extra runs).
Check: until the model first defends, a live run must reproduce the off trunk exactly ('prefix_ok').
Utility of a run: mean over all steps of u = clean acc - LAMBDA * targeted attack success.

Output per model (learn/<A|B>/live/): runs/*.parquet, trajectories.parquet, summary.csv (one row per setup),
by_family.csv, overview.png, plots/<setup>.png (clean accuracy, attack success, W and G_hat vs off / on).
Top level (learn/): live_compare.csv, live_compare.png.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "xp8" else HERE
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import learn8 as L  # noqa: E402
import xp8 as X  # noqa: E402
from poison_testbed.runner import _outcomes, build  # noqa: E402

# fixed identity colours (validated default categorical order): off, on, A, B
COLOR = {"off": "#2a78d6", "on": "#eb6834", "A": "#1baf7a", "B": "#eda100"}
LABEL = {"off": "Krum always off", "on": "Krum always on", "A": "model A (direct gain)", "B": "model B (outcome CPC)"}
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"


# ============================================================================= one live run
def _live_once(setup: dict, model_dir: str, n_steps: int, f: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctrl = L.Controller(Path(model_dir), device, n_steps)
    cfg = X.config(setup, 0, n_steps)  # schedule placeholder: every W_t is set online below
    sim = build(cfg)
    prev = _outcomes(sim, cfg)["x_val_acc"]
    rows = []
    for s in range(n_steps):
        x = sim.gather()  # signals of step s, before aggregation
        g = ctrl.gain([x[c] for c in X.SIGNALS] + [prev])
        w = int(g > 0)
        sim.schedule[s] = w
        sim.apply()
        out = _outcomes(sim, cfg)
        rows.append({"step": s, "W": w, "g_hat": g, **{c: x[c] for c in X.SIGNALS}, "x_val_acc_pre": prev,
                     **{k: out.get(k, math.nan) for k in X.OUTCOMES}, "u_active": int(sim.attack_spec.active(s))})
        ctrl.update(w)
        prev = out["x_val_acc"]
    df = pd.DataFrame(rows)
    df.insert(0, "setup", int(setup["setup"]))
    df.insert(1, "model", ctrl.kind)
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(f)
    return {"status": "done", "error": ""}


def live_job(kind: str, setup: dict, model_dir: str, n_steps: int, out: str) -> dict:
    """Worker entry point (one process per run). Never raises. A CUDA out-of-memory gets two more tries."""
    f = Path(out) / f"s{int(setup['setup']):03d}.parquet"
    t0 = time.time()
    res = {}
    for attempt in range(3):
        try:
            res = _live_once(setup, model_dir, n_steps, f)
        except Exception:
            res = {"status": "failed", "error": traceback.format_exc(limit=5)}
        if res["status"] != "failed" or "out of memory" not in res["error"]:
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(60)
    return {"model": kind, "setup": int(setup["setup"]), **res, "wall_s": round(time.time() - t0, 1)}


# ============================================================================= summaries
def _u(df: pd.DataFrame, targeted: bool) -> np.ndarray:
    atk = df.y_atk.fillna(0.0).to_numpy() if targeted else np.zeros(len(df))
    return df.y_clean_acc.to_numpy() - L.LAMBDA * atk


def trunks(res: Path, setups: list[int], n_steps: int) -> pd.DataFrame:
    cols = ["setup", "trunk", "t_branch", "step", "W", *L.INPUTS, *X.OUTCOMES]
    parts = sorted(p for p in res.glob("part*") if (p / "all_blocks.parquet").exists() and "smoke" not in p.name)
    df = pd.concat([pd.read_parquet(p / "all_blocks.parquet", columns=cols, filters=[("t_branch", "==", -1)])
                    for p in parts], ignore_index=True)
    return df[df.setup.isin(setups) & (df.step >= 0) & (df.step < n_steps)].sort_values(["setup", "trunk", "step"])


def summarise(live: pd.DataFrame, tr: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mi = man.set_index("setup")
    for (kind, setup), g in live.groupby(["model", "setup"]):
        r = mi.loc[setup]
        targeted = r.attack in L.TARGETED
        g = g.sort_values("step")
        off = tr[(tr.setup == setup) & (tr.trunk == 0)]
        on = tr[(tr.setup == setup) & (tr.trunk == 1)]
        first = int(np.argmax(g.W.to_numpy() == 1)) if (g.W == 1).any() else len(g) - 1
        a = g[L.INPUTS].iloc[:first + 1].to_numpy(float)
        b = off[L.INPUTS].iloc[:first + 1].to_numpy(float)
        att = (g.step >= r.onset) & (g.step <= r.end) if r.family != "none" else pd.Series(False, index=g.index)
        row = {"model": kind, "setup": setup, "family": r.family, "config": r.config, "n_attackers": r.n_attackers,
               "onset": r.onset, "end": r.end,
               "U_model": _u(g, targeted).mean(), "U_off": _u(off, targeted).mean(), "U_on": _u(on, targeted).mean(),
               "final_acc_model": g.y_clean_acc.iloc[-1], "final_acc_off": off.y_clean_acc.iloc[-1],
               "final_acc_on": on.y_clean_acc.iloc[-1],
               "defend_rate": g.W.mean(), "defend_rate_in_attack": g.W[att].mean() if att.any() else np.nan,
               "defend_rate_outside": g.W[~att].mean() if (~att).any() else np.nan,
               "switches": int((g.W.diff().abs() > 0).sum()),
               "first_defense": int(g.step.iloc[first]) if (g.W == 1).any() else -1,
               "prefix_exact": bool(len(a) == len(b) and np.array_equal(a, b, equal_nan=True)),
               "prefix_max_rel_diff": float(np.nanmax(np.abs(a - b) / (np.abs(b) + 1e-12))) if len(a) == len(b)
               else np.inf}
        row["prefix_ok"] = row["prefix_max_rel_diff"] < 1e-4  # exact on the GPU model of the trunks (Kaggle T4)
        if targeted:
            for k, d in (("model", g), ("off", off), ("on", on)):
                row[f"atk_mean_{k}"] = d.y_atk.fillna(0).mean()
        row["best_trunk"] = max(row["U_off"], row["U_on"])
        row["model_minus_best_trunk"] = row["U_model"] - row["best_trunk"]
        rows.append(row)
    return pd.DataFrame(rows)


def by_family(s: pd.DataFrame) -> pd.DataFrame:
    f = s.groupby("family").agg(n=("setup", "size"), U_off=("U_off", "mean"), U_on=("U_on", "mean"),
                                U_model=("U_model", "mean"), defend_rate=("defend_rate", "mean"),
                                defend_in_attack=("defend_rate_in_attack", "mean"),
                                defend_outside=("defend_rate_outside", "mean"), switches=("switches", "mean"),
                                beats_both=("model_minus_best_trunk", lambda v: (v > 0).mean()))
    tot = pd.DataFrame([{"n": len(s), "U_off": s.U_off.mean(), "U_on": s.U_on.mean(), "U_model": s.U_model.mean(),
                         "defend_rate": s.defend_rate.mean(), "defend_in_attack": s.defend_rate_in_attack.mean(),
                         "defend_outside": s.defend_rate_outside.mean(), "switches": s.switches.mean(),
                         "beats_both": (s.model_minus_best_trunk > 0).mean()}], index=["all"])
    return pd.concat([f, tot])


# ============================================================================= plots
def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_setup(kind: str, g: pd.DataFrame, off: pd.DataFrame, on: pd.DataFrame, r, srow: dict, f: Path) -> None:
    targeted = r.attack in L.TARGETED
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.6]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        _style(ax)
        if r.family != "none" and r.onset <= g.step.max():
            ax.axvspan(r.onset, min(r.end, g.step.max()), color=SHADE, lw=0, zorder=0)
    series = (("off", off), ("on", on), (kind, g))
    for k, d in series:
        axes[0].plot(d.step, d.y_clean_acc, color=COLOR[k], lw=2 if k == kind else 1.4, label=LABEL[k])
    axes[0].set_ylabel("clean accuracy", color=INK2, fontsize=9)
    if targeted:
        for k, d in series:
            axes[1].plot(d.step, d.y_atk.fillna(0), color=COLOR[k], lw=2 if k == kind else 1.4)
        axes[1].set_ylabel("attack success", color=INK2, fontsize=9)
        axes[1].set_ylim(-0.03, 1.03)
    else:
        diff = g.y_clean_acc.to_numpy() - off.y_clean_acc.to_numpy()[:len(g)]
        axes[1].plot(g.step, diff, color=COLOR[kind], lw=2)
        axes[1].axhline(0, color=COLOR["off"], lw=1.4)
        axes[1].set_ylabel("clean acc\nminus always off", color=INK2, fontsize=9)
    ax = axes[2]
    w = np.r_[0, g.W.to_numpy(), 0]
    starts, ends = np.flatnonzero(np.diff(w) == 1), np.flatnonzero(np.diff(w) == -1)
    steps = g.step.to_numpy()
    for a, b in zip(starts, ends):  # Krum on (model's decision): coloured background over each defended stretch
        ax.axvspan(steps[a], steps[b - 1] + 1, color=COLOR[kind], alpha=0.3, lw=0, zorder=0.5)
    ax.plot(g.step, g.g_hat, color=INK, lw=1)
    ax.axhline(0, color=INK2, lw=0.8, ls=":")
    ax.set_ylabel("predicted gain\n(coloured: Krum on)", color=INK2, fontsize=9)
    ax.set_xlabel("SGD step" + ("   (shaded: attack active)" if r.family != "none" else ""), color=INK2, fontsize=9)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(f"{r.config}  ·  {r.n_attackers} attacker(s)  ·  setup {r.name}", color=INK, fontsize=11,
                 x=0.06, ha="left")
    fig.text(0.06, 0.935, f"utility  model {srow['U_model']:.3f}  ·  always off {srow['U_off']:.3f}  ·  "
             f"always on {srow['U_on']:.3f}  ·  defended {srow['defend_rate']:.0%} of steps, "
             f"{srow['switches']} switches", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f, dpi=110)
    plt.close(fig)


def plot_bars(tab: pd.DataFrame, keys: list[str], f: Path, title: str) -> None:
    """Grouped bars: mean run utility per attack family, one bar per policy (fixed colours)."""
    fams = [x for x in tab.index if x != "all"] + ["all"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _style(ax)
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        xs = np.arange(len(fams)) + (i - (len(keys) - 1) / 2) * w
        ax.bar(xs, tab.loc[fams, f"U_{k}"], width=w - 0.02, color=COLOR[k], label=LABEL[k])
    ax.set_xticks(np.arange(len(fams)), fams)
    ax.set_ylabel("mean utility of the run\n(clean acc - attack success)", color=INK2, fontsize=9)
    lo = float(np.nanmin(tab[[f"U_{k}" for k in keys]].to_numpy()))
    ax.set_ylim(min(0.0, lo - 0.05), 1.0)
    ax.legend(fontsize=8, frameon=False, ncol=len(keys), loc="upper left")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(f, dpi=110)
    plt.close(fig)


# ============================================================================= driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=Path, default=L.RES, help="folder with part1/, part2/")
    ap.add_argument("--learn", type=Path, default=None, help="learn8.py output (default res/learn[_smoke])")
    ap.add_argument("--models", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="1 test setup per family, 40 steps")
    ap.add_argument("--report-only", action="store_true", help="summaries and plots from existing runs")
    args = ap.parse_args()
    learn = args.learn or args.res / ("learn_smoke" if args.smoke else "learn")
    n_steps = 40 if args.smoke else X.T
    man = pd.concat([pd.read_csv(p / "manifest.csv") for p in sorted(args.res.glob("part[12]"))])
    man = man.drop_duplicates("setup")
    test = man[man.split == "test"]
    if args.smoke:
        test = test.groupby("family").head(1)
    test = test.sort_values("setup")
    t0 = time.time()
    dirs = {}
    for kind in args.models:
        best = json.loads((learn / kind / "best.json").read_text())
        dirs[kind] = learn / kind / best["folder"]
        (learn / kind / "live" / "runs").mkdir(parents=True, exist_ok=True)
    todo = [(k, r) for k in args.models for r in test.to_dict("records")
            if not (learn / k / "live" / "runs" / f"s{int(r['setup']):03d}.parquet").exists()]
    print(f"live: {len(test)} test setups x {args.models} = {len(test) * len(args.models)} runs of {n_steps} steps, "
          f"{len(todo)} to run, {args.jobs} parallel; models {({k: str(v.name) for k, v in dirs.items()})}",
          flush=True)
    status = []
    if todo and not args.report_only:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
            futs = {pool.submit(live_job, k, r, str(dirs[k]), n_steps, str(learn / k / "live" / "runs"))
                    for k, r in todo}
            done_n = 0
            while futs:
                fin, futs = wait(futs, return_when=FIRST_COMPLETED)
                for fu in fin:
                    try:
                        res = fu.result()
                    except Exception:
                        res = {"model": "?", "setup": -1, "status": "failed", "error": traceback.format_exc(limit=3),
                               "wall_s": 0.0}
                    status.append(res)
                    done_n += 1
                    err = res["error"].strip().splitlines()[-1] if res["status"] != "done" else ""
                    print(f"[{done_n}/{len(todo)}] {res['model']} s{res['setup']:03d} {res['status']} "
                          f"{res['wall_s']:.0f}s (elapsed {(time.time() - t0) / 60:.0f} min) {err}", flush=True)
    tr = trunks(args.res, test.setup.tolist(), n_steps)
    compare = {}
    for kind in args.models:
        live_dir = learn / kind / "live"
        files = sorted((live_dir / "runs").glob("s*.parquet"))
        if not files:
            print(f"== {kind}: no live run finished", flush=True)
            continue
        live = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        live.to_parquet(live_dir / "trajectories.parquet", index=False)
        s = summarise(live, tr, man)
        s.to_csv(live_dir / "summary.csv", index=False)
        fam = by_family(s)
        fam.round(4).to_csv(live_dir / "by_family.csv")
        compare[kind] = fam
        (live_dir / "plots").mkdir(exist_ok=True)
        mi = man.set_index("setup")
        for srow in s.to_dict("records"):
            st = srow["setup"]
            plot_setup(kind, live[live.setup == st].sort_values("step"), tr[(tr.setup == st) & (tr.trunk == 0)],
                       tr[(tr.setup == st) & (tr.trunk == 1)], mi.loc[st], srow,
                       live_dir / "plots" / f"s{st:03d}_{mi.loc[st].config.replace(' ', '_').replace('/', 'of')}.png")
        plot_bars(fam.rename(columns={"U_model": f"U_{kind}"}), ["off", "on", kind], live_dir / "overview.png",
                  f"Live test, model {kind}: mean utility per attack family")
        print(f"== {kind}: {len(s)} live runs; prefix vs off trunk: exact {int(s.prefix_exact.sum())}, "
              f"within 1e-4 {int(s.prefix_ok.sum())}, max rel diff {s.prefix_max_rel_diff.max():.1e}\n"
              f"{fam.round(3).to_string()}", flush=True)
    if compare:
        k0 = next(iter(compare))
        tab = compare[k0][["n", "U_off", "U_on"]].copy()
        for kind, fam in compare.items():
            tab[f"U_{kind}"] = fam.U_model
            tab[f"defend_rate_{kind}"] = fam.defend_rate
            tab[f"beats_both_{kind}"] = fam.beats_both
        tab.round(4).to_csv(learn / "live_compare.csv")
        plot_bars(tab, ["off", "on", *compare], learn / "live_compare.png",
                  "Live test on the test setups: mean utility per attack family")
        print(tab.round(3).to_string(), flush=True)
    if status:
        pd.DataFrame(status).to_csv(learn / "live_status.csv", index=False)
    print(f"== live done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
