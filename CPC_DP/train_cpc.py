"""Train one Causal CPC on xp6 (seen runs; splits of the xp6 OutcomeGRU) and report its factual error.

    ../.venv/Scripts/python train_cpc.py --preset A2 --hidden 64 --out results/cpc/A2_h64
    ../.venv/Scripts/python train_cpc.py --preset A0 --smoke            # a few epochs: pipeline check

Presets (cpc/train.py PRESETS): A0 no pretraining (the OutcomeGRU recipe in the CPC architecture), A1 + CPC
pretraining, A2 + InfoMax, A3 + CLUB balancing, A4 + decoder feedback. --feedback / --no-feedback override.
Model selection (between hidden sizes / presets) is on the validation factual MSE (best_val_mse in report.json).
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = HERE.parent / "GRU_mnist/results/xp6_all/results/xp6/runs_xp6/all_blocks.csv"

from cpc.data import RunData, load_splits  # noqa: E402
from cpc.evaluate import CPCPredictor, factual_table  # noqa: E402
from cpc.train import PRESETS, Config, save, train_model  # noqa: E402
from gru_xp5 import groups_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="A2")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--feedback", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="3 + 3 epochs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = replace(Config(hidden=args.hidden, seed=args.seed), **PRESETS[args.preset])
    if args.feedback is not None:
        cfg = replace(cfg, feedback=args.feedback)
    if args.smoke:
        cfg = replace(cfg, pre_epochs=3, epochs=3)
    out = args.out or HERE / "results" / "cpc" / f"{args.preset}_h{args.hidden}{'_fb' if cfg.feedback else ''}{'_smoke' if args.smoke else ''}"

    sp = load_splits(str(args.runs))
    tr = RunData(sp.train)
    va, te, ho = RunData(sp.val, ref=tr), RunData(sp.test, ref=tr), RunData(sp.heldout, ref=tr)
    print(f"runs: train {len(tr)}, val {len(va)}, test {len(te)}, held-out {len(ho)}; inputs {tr.enc.shape[-1]}; "
          f"device {args.device}; {cfg}", flush=True)
    model, hist = train_model(cfg, tr, va, args.device)
    save(out, model, cfg, tr, hist)

    pred = CPCPredictor(model, args.device, name=out.name)
    parts = [factual_table(pred, te, "seen test")]
    for g, sub in sp.heldout.groupby(groups_of(sp.heldout)):
        parts.append(factual_table(pred, RunData(sub, ref=tr), f"held-out: {g}"))
    fact = pd.concat(parts, ignore_index=True)
    fact.to_csv(out / "factual.csv", index=False)
    rep = {"preset": args.preset, "config": asdict(cfg), "best_val_mse": hist["best_val_mse"],
           "epochs": {"stage1": len(hist["stage1"]), "stage2": len(hist["stage2"])}}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    t = fact[(fact.attack == "all") & (fact.horizon == "all")].pivot_table(index="set", columns="outcome", values="rmse")
    print(f"\nbest val MSE {hist['best_val_mse']:.5f}; factual RMSE (all horizons):\n{t.round(4).to_string()}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
