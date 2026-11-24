"""Train the GRU and Causal CPC models on xp7, one set per policy level (g0, g2, g8, hard), locally, one at a time.

    ../.venv/Scripts/python train_xp7.py                     # -> results/xp7_models/<level>/<model>/
    ../.venv/Scripts/python train_xp7.py --levels g0 --models gru A0 --smoke

Models: gru = the xp6 OutcomeGRU baseline (model d: hidden 128, lr 5e-4, wd 1e-2, patience 40);
A0 / A2 / A3 = Causal CPC presets (cpc/train.py) at hidden 64: no pretraining / CPC + InfoMax / + CLUB.
Early stopping on the validation runs' factual MSE (validation twins are used later, for decision margins only).
Finished models (report.json present) are skipped.
"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from cpc import xp7data as X  # noqa: E402
from cpc.train import PRESETS, Config, save, train_model  # noqa: E402
from gru_experiment import fit_gru  # noqa: E402

OUT = HERE / "results/xp7_models"
MODELS = ["gru", "A0", "A2", "A3"]
GRU_HP = dict(hidden=128, lr=5e-4, wd=1e-2)


def train_one(level: str, name: str, data, device: str, smoke: bool) -> None:
    folder = OUT / level / f"{name}{'_smoke' if smoke else ''}"
    if (folder / "report.json").exists():
        print(f"skip {level}/{folder.name}", flush=True)
        return
    tr, va, _ = data
    t0 = time.time()
    folder.mkdir(parents=True, exist_ok=True)
    if name == "gru":
        model, hist = fit_gru(tr.d, va.d, torch.device(device), 3 if smoke else 300, GRU_HP["hidden"], 0,
                              lr=GRU_HP["lr"], weight_decay=GRU_HP["wd"], patience_max=40)
        torch.save(model.state_dict(), folder / "gru.pt")
        np.savez(folder / "norm.npz", x_mean=tr.d.x_mean, x_std=tr.d.x_std)
        rep = {"model": "gru", "hparams": {**GRU_HP, "patience": 40, "seed": 0}, "best_val_mse": min(hist),
               "epochs": len(hist), "x_cols": tr.x_cols}
    else:
        cfg = replace(Config(hidden=64), **PRESETS[name])
        if smoke:
            cfg = replace(cfg, pre_epochs=3, epochs=3)
        model, hist = train_model(cfg, tr, va, device, log=lambda s: None)
        save(folder, model, cfg, tr, hist)
        rep = {"model": name, "config": asdict(cfg), "best_val_mse": hist["best_val_mse"],
               "epochs": {"stage1": len(hist["stage1"]), "stage2": len(hist["stage2"])}}
    rep.update(level=level, train_runs=len(tr), minutes=round((time.time() - t0) / 60, 1))
    (folder / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"done {level}/{folder.name}: best val MSE {rep['best_val_mse']:.5f} ({rep['minutes']} min)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="*", default=X.LEVELS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    for level in args.levels:
        data = X.level_data(level)
        for name in args.models:
            train_one(level, name, data, args.device, args.smoke)


if __name__ == "__main__":
    main()
