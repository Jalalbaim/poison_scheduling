"""xp5: ONE GRU trained on the mix of all seen configurations, tested on seen and held-out configurations.

    python gru_xp5.py --runs results/xp5/runs_xp5/all_blocks.csv --out results/xp5/gru

Data: Fashion-MNIST runs with backdoor variants, bursts, targeted label flip, ALIE and no attack, under Krum, Median
and TrimmedMean (run_experiment.py xp5). Held-out groups (never used for training, validation or model selection):
the blend trigger, the 3/8 poison rate, and the flip x TrimmedMean pair. Seen runs are split by seed, stratified
by attack, poison rate and defense: 70% train, 10% validation, 20% test.

Inputs: v1 + v2 signals (block mean/max and per-round values), W, time, defense one-hot. Never the attack, the
trigger, the target or any outcome. Outputs: clean accuracy and attack success (backdoor success, or flip rate;
masked for ALIE, whose harm is in clean accuracy).

Steps:
  1. a small hyperparameter sweep (MODELS), one model each, early stopping on validation seeds; the model with the
     lowest validation loss is selected (validation seeds only);
  2. evaluation of the selected model: prediction error by horizon vs a ridge baseline and the defense-mean
     trajectory, on seen test runs (overall and per attack family) and on each held-out group; kill awareness;
  3. decision margins for deployment (deploy_xp5.py), calibrated on validation seeds (Youden point of check P):
     'release' (after a defended block: keep defending or not) and 'free' (from every block: defend or not).
  4. benefit check: at every block of the seen test runs and held-out groups, the predicted attack success of
     'defend the next H blocks' vs 'do not defend' (positive benefit = the GRU believes defending helps).

xp5a (--exclude-schedules react probehold): the 'react' and 'probehold' schedules switch the defense on at the
TRUE attack window (an unobservable), so in their runs a planned W = 1 announces an attack (P(attack in the next
20 blocks | W switches on) = 0.98 vs 0.31). The decoder, which receives the planned W, then learns 'W = 1 ->
high attack success', which cancels the real effect of the defense. With the option, training and validation use
only the runs whose schedule ignores the window (sticky, probe, on, off); the split itself (assign_splits on all
seen runs) and the test / held-out sets are unchanged, so every number is comparable with xp5. --compare scores
the xp5 model (its summary.json) on the same sets, with its own normalisation.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

import analyze  # noqa: E402
import check_probe  # noqa: E402
from analyze import INK, INK2, SERIES  # noqa: E402
from gru_experiment import (Data, OutcomeGRU, assign_splits, check1, check_kill, fit_gru, fit_ridge,  # noqa: E402
                            support)
from run_experiment import xp5_heldout_group  # noqa: E402

MODELS = {"a": dict(hidden=64, lr=5e-4, wd=0.0), "c": dict(hidden=64, lr=2e-4, wd=1e-2),
          "d": dict(hidden=128, lr=5e-4, wd=1e-2)}
FAMILY = {"none": "no attack", "backdoor_win": "backdoor", "backdoor_burst": "burst", "flip_win": "label flip",
          "alie_win": "ALIE"}


def groups_of(df: pd.DataFrame) -> pd.Series:
    first = df.groupby("run_id").first()
    g = pd.Series([xp5_heldout_group(a, t, pf, d) for a, t, pf, d in
                   zip(first.attack, first.u_trigger, first.poison_frac, first.defense)], index=first.index)
    return df.run_id.map(g)


SCHEDULE_FAMILIES = ("sticky", "react", "probehold", "probe", "on", "off", "alarm")


def schedule_family(run_id: pd.Series) -> pd.Series:
    """run_id = <attack>__<defense>__<schedule>__s<seed>; family = the schedule's leading word (twins: of the base)."""
    fam = run_id.str.split("__").str[2].str.extract(r"^([a-z]+)", expand=False)
    unknown = set(fam.dropna().unique()) - set(SCHEDULE_FAMILIES) | ({"<missing>"} if fam.isna().any() else set())
    if unknown:
        raise ValueError(f"unknown schedule families in run ids: {sorted(unknown)}")
    return fam


def seen_splits(df: pd.DataFrame, exclude: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                                pd.DataFrame]:
    """(seen, train, val, test). The split is drawn on all seen runs (as in xp5); `exclude` then removes schedule
    families from train and val only. The test set always keeps every schedule."""
    seen = df[groups_of(df) == "seen"]
    split = assign_splits(seen)
    for name in ("val", "test"):  # only for tiny (smoke-test) data: groups of 1-2 seeds all land in train
        if not (split == name).any():
            seeds = sorted(seen.seed[split == "train"].unique())[1::5]
            split[seen.seed.isin(seeds)] = name
            print(f"warning: no {name} runs (tiny data): moved seeds {seeds} from train to {name}", flush=True)
    keep = ~schedule_family(seen.run_id).isin(exclude)
    return seen, seen[(split == "train") & keep], seen[(split == "val") & keep], seen[split == "test"]


@torch.no_grad()
def benefit(model: OutcomeGRU, data: Data, device) -> pd.DataFrame:
    """At every block with a full horizon left: predicted attack success after H blocks if not defending minus
    if defending (> 0: the GRU believes defending lowers attack success), and the predicted change in clean
    accuracy (defend - not; < 0 = the cost). gain = clean_change + LAMBDA * benefit_atk."""
    check_probe.ALT, check_probe.CUTS = "release", "all"
    t = check_probe.gains(model, data, device)
    t = t[t.attack != "alie_win"].copy()  # no targeted outcome
    t["defense"] = t.run.map(dict(zip(data.run_ids, data.defense)))
    t["benefit_atk"] = t.atk_probe - t.atk_defend  # 'probe' column = the alternative plan (here: do not defend)
    t["clean_change"] = t.gain - check_probe.LAMBDA * t.benefit_atk
    return t


def benefit_table(t: pd.DataFrame) -> pd.DataFrame:
    t = t.assign(state=np.where(t.active, "attack active", "no attack active"))
    return t.groupby(["model", "set", "defense", "state"])[["benefit_atk", "clean_change", "gain"]].mean().reset_index()


def train_models(train: Data, val: Data, out: Path, device, epochs: int, models: list[str]) -> dict:
    res = {}
    for name in models:
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        hp = MODELS[name]
        if (folder / "gru.pt").exists() and (folder / "report.json").exists():
            res[name] = json.loads((folder / "report.json").read_text())
            print(f"model {name}: already trained (val {res[name]['best_val_mse']:.5f})", flush=True)
            continue
        model, history = fit_gru(train, val, device, epochs, hp["hidden"], 0, lr=hp["lr"], weight_decay=hp["wd"],
                                 patience_max=40)
        torch.save(model.state_dict(), folder / "gru.pt")
        rep = {"weights_sha256": hashlib.sha256((folder / "gru.pt").read_bytes()).hexdigest()[:16],
               "hparams": {**hp, "patience": 40, "seed": 0}, "epochs": len(history), "best_val_mse": min(history),
               "val_history": history, "x_cols": train.x_cols}
        (folder / "report.json").write_text(json.dumps(rep, indent=2))
        res[name] = rep
        print(f"model {name}: {len(history)} epochs, best val MSE {min(history):.5f}", flush=True)
    return res


def load_model(folder: Path, train: Data, device) -> OutcomeGRU:
    hidden = json.loads((folder / "report.json").read_text())["hparams"]["hidden"]
    model = OutcomeGRU(train.enc.shape[-1], train.dec_inputs(train.W).shape[-1], hidden).to(device)
    model.load_state_dict(torch.load(folder / "gru.pt", map_location=device))
    return model.eval()


def margins(model: OutcomeGRU, val: Data, device) -> dict:
    out = {}
    for tag, cuts in [("release_defended", "defended"), ("release_all", "all")]:
        check_probe.ALT, check_probe.CUTS = "release", cuts
        t = check_probe.gains(model, val, device)
        t = t[t.attack != "alie_win"]  # ALIE has no targeted outcome: 'active' is not a defend-or-not label there
        if len(t) == 0 or t.active.nunique() < 2:
            out[tag] = {"delta": 0.0, "note": "validation cuts have a single class; default margin 0"}
            continue
        fpr, tpr, thr = roc_curve(t.active, t.gain)
        k = int(np.argmax(tpr - fpr))
        out[tag] = {"delta": float(thr[k]) if np.isfinite(thr[k]) else 0.0, "val_tpr": float(tpr[k]),
                    "val_fpr": float(fpr[k]), "val_cuts": int(len(t))}
    return out


def fig_errors(table: pd.DataFrame, out: Path) -> None:
    sets = list(dict.fromkeys(table.set))
    models = ["GRU", "ridge (direct, W-aware)", "defense-mean trajectory"]
    colors = dict(zip(models, list(SERIES.values())[:3]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=False)
    width = 0.8 / len(models)
    for ax, y in zip(axes, ["y_atk", "y_clean_acc"]):
        s = table[(table.outcome == y) & (table.horizon == "all")]
        for k, m in enumerate(models):
            v = s[s.model == m].set_index("set").reindex(sets).rmse
            ax.bar(np.arange(len(sets)) - 0.4 + width * (k + 0.5), v, width * 0.9, color=colors[m], label=m)
        ax.set_xticks(range(len(sets)), sets, rotation=30, ha="right")
        ax.set_title({"y_atk": "attack success", "y_clean_acc": "clean accuracy"}[y] + ": prediction RMSE (all horizons)",
                     color=INK, loc="left")
        held = [i for i, n in enumerate(sets) if n.startswith("held-out")]
        if held:
            ax.axvspan(min(held) - 0.5, max(held) + 0.5, color=INK2, alpha=0.07, linewidth=0)
    axes[0].legend(fontsize=8)
    fig.suptitle("One GRU for all configurations: error on seen test runs (by family) and on held-out "
                 "configurations (shaded)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "prediction_error.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("results/xp5/runs_xp5/all_blocks.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/xp5/gru"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--exclude-schedules", nargs="*", default=[], choices=SCHEDULE_FAMILIES,
                    help="schedule families left out of training and validation (xp5a: react probehold)")
    ap.add_argument("--compare", type=Path, default=None,
                    help="summary.json of an earlier model (xp5), scored on the same sets for the benefit check")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = analyze.load(args.runs)
    df["group"] = groups_of(df)
    seen, train_df, val_df, test_df = seen_splits(df, args.exclude_schedules)
    train = Data(train_df)
    val = Data(val_df, train.x_mean, train.x_std, train.x_cols)
    fam = schedule_family(train_df.groupby("run_id").run_id.first())
    print(f"runs: train {len(train.seed)}, val {len(val.seed)}, test (seen) {test_df.run_id.nunique()}, held-out "
          f"{df[df.group != 'seen'].run_id.nunique()}; {len(train.x_cols)} input signals; device {device}", flush=True)
    print(f"training schedules: {fam.value_counts().to_dict()}; excluded: {args.exclude_schedules or 'none'}",
          flush=True)

    reports = train_models(train, val, args.out, device, args.epochs, args.models)
    best = min(reports, key=lambda k: reports[k]["best_val_mse"])
    model = load_model(args.out / best, train, device)
    print(f"selected model {best} (lowest validation loss)", flush=True)
    ridge = fit_ridge(train)

    sets = {"seen test: all": test_df}
    for a, name in FAMILY.items():
        sub = test_df[test_df.attack == a]
        if sub.run_id.nunique():
            sets[f"seen test: {name}"] = sub
    for g in sorted(df.group.unique()):
        if g != "seen":
            sets[f"held-out: {g}"] = df[df.group == g]
    # Models scored by the benefit check: this one, and optionally the earlier one with its own normalisation.
    scored = {"this model": (model, train)}
    if args.compare:
        prev = json.loads(args.compare.read_text())
        _, prev_train_df, _, _ = seen_splits(df, prev.get("exclude_schedules", []))
        prev_train = Data(prev_train_df)
        scored["compared model"] = (load_model(Path(prev["selected_folder"]), prev_train, device), prev_train)
        print(f"compared model: {prev['selected_folder']} (excluded schedules: "
              f"{prev.get('exclude_schedules') or 'none'})", flush=True)

    tables, kills, bens = [], [], []
    for name, sub in sets.items():
        data = Data(sub, train.x_mean, train.x_std, train.x_cols)
        tables.append(check1(model, ridge, train, data, device).assign(set=name, runs=len(data.seed)))
        for tag, (m, tr) in scored.items():
            d = data if tr is train else Data(sub, tr.x_mean, tr.x_std, tr.x_cols)
            k = check_kill(m, d, device, args.out)
            if len(k):
                kills.append(k.assign(set=name, model=tag))
            bens.append(benefit(m, d, device).assign(set=name, model=tag))
    (args.out / "checkK_kill.csv").unlink(missing_ok=True)  # check_kill writes per call; the combined file is below
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.out / "prediction_error.csv", index=False)
    kill = pd.concat(kills, ignore_index=True) if kills else pd.DataFrame()
    if len(kill):
        kill.to_csv(args.out / "kill_awareness.csv", index=False)
    ben = benefit_table(pd.concat(bens, ignore_index=True))
    ben.to_csv(args.out / "benefit.csv", index=False)
    fig_errors(table, args.out)
    marg = margins(model, val, device)
    summary = {"selected": best, "selected_folder": str(args.out / best), "models": {k: {kk: v[kk] for kk in
               ("hparams", "epochs", "best_val_mse", "weights_sha256")} for k, v in reports.items()},
               "margins": marg, "exclude_schedules": args.exclude_schedules,
               "train_schedules": {str(k): int(v) for k, v in fam.value_counts().items()},
               "support": support(seen)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 220)
    print("\n## Prediction error (RMSE, all horizons): GRU vs baselines, per evaluation set")
    piv = table[table.horizon == "all"].pivot_table(index="set", columns=["outcome", "model"], values="rmse", sort=False)
    print(piv.round(4).to_string())
    if len(kill):
        print("\n## Kill awareness: predicted attack success at window end if defending from the 2nd attack block "
              "vs never")
        print(kill.groupby(["model", "set"])[["gru_atk_window_end_if_defend", "gru_atk_window_end_if_never",
                                              "actual_atk_window_end"]].mean().round(3).to_string())
    print("\n## Benefit check (seen test + held-out, every block): predicted attack success removed by defending "
          "the next H blocks (> 0 = defending helps), predicted clean-accuracy change (< 0 = cost), gain = sum")
    b = ben[ben.state == "attack active"]
    print("attack active:")
    print(b.pivot_table(index=["set", "defense"], columns="model", values=["benefit_atk", "clean_change", "gain"],
                        sort=False).round(3).to_string())
    print("no attack active (benefit should be ~0):")
    b = ben[ben.state == "no attack active"]
    print(b.groupby(["model", "defense"])[["benefit_atk", "clean_change", "gain"]].mean().round(3).to_string())
    print("\n## Decision margins (validation seeds)")
    print(json.dumps(marg, indent=2))


if __name__ == "__main__":
    main()
