"""Score the xp7 models on the onset twins with the whole-run Delta (cpc/evaluate.score_twins_whole).

    ../.venv/Scripts/python xp7/score_xp7.py                 # -> results/xp7/scores/{twins.csv, summary.csv, by_family.csv}

Margins (defend iff predicted Delta > margin) are chosen on the VALIDATION twins; every reported number is on the
TEST twins. A model trained at level L is scored on the same twins as every other model (twins are open loop,
independent of the policy level).
"""

import argparse
import sys
import json
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name in ("xp6", "xp7") else HERE  # local: CPC_DP/xp7/; Kaggle: files flat in $WORK
sys.path.insert(0, str(ROOT))
from cpc import xp7data as X  # noqa: E402
from cpc.data import RunData  # noqa: E402
from cpc.evaluate import CPCPredictor, GRUPredictor, best_margin, decision_table, score_twins_whole  # noqa: E402
from cpc.train import load  # noqa: E402
from gru_experiment import OutcomeGRU  # noqa: E402

MODELS = ROOT / "results/xp7/models"
OUT = ROOT / "results/xp7/scores"


def predictor(folder: Path, tr: RunData, device: str):
    rep = json.loads((folder / "report.json").read_text())
    name = f"{rep['level']}/{folder.name}"
    if rep["model"] == "gru":
        m = OutcomeGRU(tr.d.enc.shape[-1], tr.d.dec_inputs(tr.d.W).shape[-1], rep["hparams"]["hidden"]).to(device)
        m.load_state_dict(torch.load(folder / "gru.pt", map_location=device))
        return GRUPredictor(m.eval(), device, name)
    return CPCPredictor(load(folder, device)[0], device, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()
    tw, man = X.twins()
    rows, summ = [], []
    for level in X.LEVELS:
        tr = RunData(X.level_splits(level)["train"])  # normalisation of this level's training runs
        ev = RunData(tw, ref=tr)
        for folder in sorted((MODELS / level).iterdir()):
            if not (folder / "report.json").exists() or folder.name.endswith("_smoke"):
                continue
            pred = predictor(folder, tr, args.device)
            s = score_twins_whole(pred, ev, man, args.lam).assign(level=level, model_type=folder.name)
            rows.append(s)
            m = best_margin(s[s.split == "val"])
            summ.append({"level": level, "model": folder.name, **decision_table(s[s.split == "test"], m)})
            print(f"{level}/{folder.name}: {summ[-1]['share_of_oracle']:.2f} of oracle, corr {summ[-1]['corr']:.2f}",
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    allr = pd.concat(rows, ignore_index=True)
    allr.to_csv(OUT / "twins.csv", index=False)
    summary = pd.DataFrame(summ)
    summary.to_csv(OUT / "summary.csv", index=False)
    test = allr[allr.split == "test"]
    fam = test.groupby(["level", "model_type", "family"]).agg(true=("true_delta", "mean"), pred=("pred_delta", "mean"),
                                                              n=("true_delta", "size")).reset_index()
    fam.to_csv(OUT / "by_family.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(fam.pivot_table(index=["level", "model_type"], columns="family", values="pred").round(3).to_string())
    print("true:", test.drop_duplicates(["base", "d"]).groupby("family").true_delta.mean().round(3).to_dict())
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
