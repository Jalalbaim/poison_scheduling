"""Score models on Delta_k against the xp6 onset twins (evaluation only), and their factual error on the same sets.

    ../.venv/Scripts/python xp6/score_delta.py --cpc results/xp6/models/A0_h64 results/xp6/models/A2_h64 --gru

--gru adds the xp6 OutcomeGRU (model d of xp6_all) as the baseline. Normalisation for every model = the xp6
training runs (identical to the GRU's). Output: delta_runs.csv (one row per on-plan), delta_summary.csv (by model x
group x attack, and by model x d x k), factual.csv; a printed summary.
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
XP6 = ROOT.parent / "GRU_mnist/results/xp6_all/results/xp6"

from cpc.data import RunData, load_splits  # noqa: E402
from cpc.evaluate import CPCPredictor, GRUPredictor, delta_metrics, factual_table, score_twins  # noqa: E402
from cpc.train import load  # noqa: E402
from gru_experiment import OutcomeGRU  # noqa: E402


def load_gru(folder: Path, tr: RunData, device) -> OutcomeGRU:
    hidden = json.loads((folder / "report.json").read_text())["hparams"]["hidden"]
    m = OutcomeGRU(tr.d.enc.shape[-1], tr.d.dec_inputs(tr.d.W).shape[-1], hidden).to(device)
    m.load_state_dict(torch.load(folder / "gru.pt", map_location=device))
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=XP6 / "runs_xp6/all_blocks.csv")
    ap.add_argument("--eval", type=Path, default=ROOT / "results/xp6/eval", help="folder with all_blocks.csv + manifest.csv")
    ap.add_argument("--cpc", type=Path, nargs="*", default=[])
    ap.add_argument("--gru", action="store_true")
    ap.add_argument("--gru-folder", type=Path, default=XP6 / "gru/d")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "results/xp6/delta")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sp = load_splits(str(args.runs))
    tr = RunData(sp.train)
    ev_df = pd.read_csv(args.eval / "all_blocks.csv", dtype={"round_start": "Int64", "W": "Int64"}, low_memory=False)
    man = pd.read_csv(args.eval / "manifest.csv")
    ev = RunData(ev_df, ref=tr)
    preds = [CPCPredictor(load(f, args.device)[0], args.device, name=f.name) for f in args.cpc]
    if args.gru:
        preds.insert(0, GRUPredictor(load_gru(args.gru_folder, tr, args.device), args.device, name="gru_xp6_d"))
    if not preds:
        raise SystemExit("nothing to score: pass --cpc folders and/or --gru")

    runs = pd.concat([score_twins(p, ev, man, args.lam, args.tau) for p in preds], ignore_index=True)
    offs = ev_df[ev_df.run_id.isin(man[man.plan == "off"].run_id)]
    fact = pd.concat([factual_table(p, RunData(offs, ref=tr), "onset twins: off runs") for p in preds], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out / "delta_runs.csv", index=False)
    fact.to_csv(args.out / "factual.csv", index=False)
    s1 = delta_metrics(runs, ["model", "group", "attack"])
    s2 = delta_metrics(runs[runs.group == "attacked"], ["model", "d", "k"])
    pd.concat([s1.assign(view="group x attack"), s2.assign(view="attacked: d x k")]).to_csv(
        args.out / "delta_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"Delta_k = U(on [t, t+k)) - U(off), U = y_clean(T) - {args.lam} * y_atk(t + {args.tau})\n")
    print(s1.round(3).to_string(index=False))
    print("\nattacked test seeds, by switch delay d and length k:")
    print(s2.round(3).to_string(index=False))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
