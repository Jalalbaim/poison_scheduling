"""Check P (probe decisions): does the GRU tell 'hold the defense' from 'keep probing' after a defended block?

    python check_probe.py results/xp3/gru_sweep/a results/xp3/gru_sweep/b ...   # score each model (val + test)
    python check_probe.py --select results/xp3/gru_sweep/*                     # pick the best on validation

For every xp3 run of a split and every block b whose previous block was defended (a probe or a hold), the
controller of deploy_xp3 compares two plans for blocks b..b+H-1: keep probing (one defended block every P) or
defend every block (--alt release: or not defend at all; --cuts all: from every block). Gain = score(defend) - score(alt), score = clean - LAMBDA * backdoor at block b+H-1.
Label (evaluation only, from u_active): the attack is active in block b-1. A good model has gain > 0 when an
attack is active and gain < 0 when none is: reported as AUC, and as TPR / FPR of the rule gain > 0.

--select scores every model on validation seeds only, picks the highest AUC (ties: lower FPR), then scores the
chosen one on test seeds, and writes results/xp3/gru_selected.json. Test seeds never influence the choice.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import analyze
from deploy_xp3 import HORIZON, LAMBDA, PROBE_PERIOD
from gru_experiment import Data, OutcomeGRU, assign_splits

RUNS = Path("results/xp3/runs_xp3/all_blocks.csv")
SELECTED = Path("results/xp3/gru_selected.json")
ALT = "probe"  # alternative plan to "defend": "probe" (xp3: keep probing) or "release" (xp4: defend no more)
CUTS = "defended"  # "defended": blocks after a defended block; "all": every block (xp4 gru_free)


@torch.no_grad()
def gains(model: OutcomeGRU, data: Data, device) -> pd.DataFrame:
    """One decoder call per run: every eligible cut x both plans (cuts with a full horizon left)."""
    rows, H, nb = [], HORIZON, data.n_blocks
    h_all = model.encode(data.enc.to(device))  # (R, rows, hidden): h_all[r, b] = state after blocks -1..b-1
    for r in range(len(data.seed)):
        cuts = np.array([b for b in range(1, nb - H + 1) if CUTS == "all" or data.W[r, b] == 1])
        if not len(cuts):
            continue
        W = np.zeros((2 * len(cuts), data.n_rows))
        for k, b in enumerate(cuts):
            if ALT == "probe":
                W[2 * k, b + 1: b + 1 + H] = [int(j % PROBE_PERIOD == 0) for j in range(b, b + H)]  # keep probing
            W[2 * k + 1, b + 1: b + 1 + H] = 1                                                # defend
        dec = data.dec_inputs(W, runs=slice(r, r + 1))  # (2C, n_rows-1, F); index i = row i+1 = block i
        idx = np.repeat(cuts, 2)[:, None] + np.arange(H)[None, :]  # blocks b..b+H-1
        seq = dec[np.arange(len(W))[:, None], idx].to(device)
        h0 = h_all[r, torch.as_tensor(np.repeat(cuts, 2), device=device)]
        p = model.decode(h0, seq)[:, -1].cpu().numpy().reshape(len(cuts), 2, 2)  # (cut, plan, outcome)
        s = p[..., 0] - LAMBDA * p[..., 1]
        rows.append(pd.DataFrame({"run": data.run_ids[r], "attack": data.attack[r], "block": cuts,
                                  "gain": s[:, 1] - s[:, 0], "active": data.U[r, cuts] > 0,
                                  "atk_probe": p[:, 0, 1], "atk_defend": p[:, 1, 1]}))
    if not rows:  # no eligible cut (e.g. very short runs): an empty table with the usual columns
        return pd.DataFrame(columns=["run", "attack", "block", "gain", "active", "atk_probe", "atk_defend"])
    return pd.concat(rows, ignore_index=True)


def score(t: pd.DataFrame) -> dict:
    y, g = t.active.to_numpy(bool), t.gain.to_numpy()
    return {"cuts": len(t), "active": int(y.sum()), "auc": round(float(roc_auc_score(y, g)), 3),
            "tpr": round(float((g[y] > 0).mean()), 3), "fpr": round(float((g[~y] > 0).mean()), 3),
            "fpr_no_attack_runs": round(float((t[t.attack == "none"].gain > 0).mean()), 3)}


def load_model(folder: Path, train: Data, device) -> tuple[OutcomeGRU, dict]:
    rep = json.loads((folder / "report.json").read_text())
    hidden = (rep.get("hparams") or {}).get("hidden", 64)
    model = OutcomeGRU(train.enc.shape[-1], train.dec_inputs(train.W).shape[-1], hidden).to(device)
    model.load_state_dict(torch.load(folder / "gru.pt", map_location=device))
    model.eval()
    return model, {"hparams": rep.get("hparams"), "epochs": rep["epochs"], "best_val_mse": round(rep["best_val_mse"], 5),
                   "weights_sha256": rep["weights_sha256"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--alt", choices=["probe", "release"], default="probe")
    ap.add_argument("--cuts", choices=["defended", "all"], default="defended")
    ap.add_argument("--out", type=Path, default=None, help="selection json (default results/xp3/gru_selected.json)")
    args = ap.parse_args()
    global ALT, CUTS
    ALT, CUTS = args.alt, args.cuts
    selected = args.out or SELECTED
    tag = f"{args.alt}_{args.cuts}"
    sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = analyze.load(args.runs)
    split = assign_splits(df)
    train = Data(df[split == "train"])
    sets = {name: Data(df[split == name], train.x_mean, train.x_std) for name in ("val", "test")}
    models = [m for m in args.models if (m / "gru.pt").exists() and (m / "report.json").exists()]
    out = {}
    for folder in models:
        model, res = load_model(folder, train, device)
        for name in ("val",) if args.select else ("val", "test"):
            t = gains(model, sets[name], device)
            t.to_csv(folder / f"checkP_{tag}_{name}.csv", index=False)
            res[name] = score(t)
        out[str(folder)] = res
        print(folder, json.dumps(res), flush=True)
    if args.select:
        best = max(out, key=lambda k: (out[k]["val"]["auc"], -out[k]["val"]["fpr"]))
        model, _ = load_model(Path(best), train, device)
        t = gains(model, sets["test"], device)
        t.to_csv(Path(best) / f"checkP_{tag}_test.csv", index=False)
        out[best]["test"] = score(t)
        verdict = "ok" if out[best]["val"]["auc"] >= 0.7 else "WEAK: validation AUC < 0.7, deployment is unlikely to help"
        selected.write_text(json.dumps({"selected": best, "verdict": verdict, "alt": ALT, "cuts": CUTS,
                                        "models": out}, indent=2))
        print(f"\nselected {best}: val {out[best]['val']}, test {out[best]['test']} -> {verdict}", flush=True)
    else:
        (args.out or Path("results/xp3/checkP_models.json")).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
