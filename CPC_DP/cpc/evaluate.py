"""Evaluation shared by CausalCPC and the xp6 OutcomeGRU: factual error by horizon, and Delta_k on onset twins.

Delta_k(H_t) = U(on over blocks [t, t + k), off elsewhere) - U(off), U = y_clean(T) - lam * y_atk(t + tau).
History H_t = the off twin's rows 0..t (blocks -1..t-1), identical in both twins. Row r holds the outcome after
block r - 1, so y_atk(t + tau) is row t + tau (tau blocks after the decision) and y_clean(T) is the last row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data import RunData
from . import _GRU  # noqa: F401
import gru_experiment as ge  # noqa: E402


class CPCPredictor:
    name = "cpc"

    def __init__(self, model, device, name: str = "cpc"):
        self.model, self.device, self.name = model.eval(), device, name

    @torch.no_grad()
    def cuts(self, rd: RunData, rows, cuts):
        y, _, valid, idx = self.model.cut_predictions(rd.enc[rows].to(self.device), rd.W[rows].to(self.device),
                                                      rd.t.to(self.device), cuts.to(self.device))
        return y.cpu(), valid.cpu(), idx.cpu()

    @torch.no_grad()
    def plans(self, rd: RunData, i: int, cut: int, plans: torch.Tensor) -> torch.Tensor:
        """(P, n - 1 - cut, 2): predictions for rows cut+1 .. n-1 from run i's history 0..cut, per plan (P, n)."""
        dev, n = self.device, rd.n_rows
        c = self.model.encode(rd.enc[i:i + 1].to(dev))[1]
        P = len(plans)
        w = plans.to(dev)
        y, _ = self.model.decode(c[:, cut].expand(P, -1), w[:, cut:n - 1], w[:, cut + 1:n],
                                 rd.t[cut + 1:n].to(dev).expand(P, -1))
        return y.cpu()


class GRUPredictor:
    """The xp6 OutcomeGRU (gru_experiment), on the same RunData (identical normalisation when built from the
    same training runs)."""

    def __init__(self, model, device, name: str = "gru"):
        self.model, self.device, self.name = model.eval(), device, name

    @torch.no_grad()
    def cuts(self, rd: RunData, rows, cuts):
        dec = rd.d.dec_inputs(rd.d.W)
        y, valid, idx = ge.cut_predictions(self.model, rd.d.enc[rows].to(self.device), dec[rows].to(self.device),
                                           cuts.to(self.device))
        return y.cpu(), valid.cpu(), idx.cpu()

    @torch.no_grad()
    def plans(self, rd: RunData, i: int, cut: int, plans: torch.Tensor) -> torch.Tensor:
        h = self.model.encode(rd.d.enc[i:i + 1].to(self.device))[:, cut]
        dec = rd.d.dec_inputs(plans.numpy(), runs=slice(i, i + 1))[:, cut:].to(self.device)
        return self.model.decode(h.expand(len(plans), -1), dec).cpu()


# ----------------------------------------------------------------------------- factual error
def factual_table(pred, rd: RunData, set_name: str, step: int = 5) -> pd.DataFrame:
    """RMSE by horizon (1, 5, n/5, n/2 blocks and all), per outcome and attack family, under each run's own W."""
    n_dec = rd.n_rows - 1
    cuts = torch.arange(0, n_dec, step)
    hz = ge.horizons(rd.d.n_blocks)
    acc: dict = {}
    for rows in torch.arange(len(rd)).split(16):
        y, valid, idx = pred.cuts(rd, rows, cuts)
        Y = rd.Y[rows][:, 1:][:, idx]
        M = rd.M[rows][:, 1:][:, idx] * valid[None, :, :, None]
        se = ((y - Y) ** 2 * M).numpy()
        m = M.numpy()
        for r, a in zip(range(len(rows)), rd.attack[rows.numpy()]):
            for o, oname in enumerate(ge.Y_COLS):
                for h in ["all", *hz]:
                    sl = slice(None) if h == "all" else slice(h - 1, h)
                    for fam in ("all", a):
                        k = (fam, oname, str(h))
                        s, c = acc.get(k, (0.0, 0.0))
                        acc[k] = (s + se[r, :, sl, o].sum(), c + m[r, :, sl, o].sum())
    out = [{"model": pred.name, "set": set_name, "attack": k[0], "outcome": k[1], "horizon": k[2],
            "rmse": np.sqrt(s / c) if c else np.nan, "n": int(c)} for k, (s, c) in acc.items()]
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- Delta on onset twins
def score_twins(pred, rd: RunData, manifest: pd.DataFrame, lam: float = 1.0, tau: int = 20) -> pd.DataFrame:
    pos = {r: i for i, r in enumerate(rd.run_ids)}
    n = rd.n_rows
    rows = []
    for seed, g in manifest.groupby("seed"):
        i_off = pos[g[g.plan == "off"].run_id.iloc[0]]
        for t, gt in g[g.plan == "on"].groupby("t"):
            t = int(t)
            plans = [torch.zeros(n)]
            for r in gt.itertuples():
                p = torch.zeros(n)
                p[t + 1:t + 1 + int(r.k)] = 1
                if not torch.equal(p, rd.W[pos[r.run_id]]):
                    raise ValueError(f"plan of {r.run_id} does not match its logged W")
                plans.append(p)
            yhat = pred.plans(rd, i_off, t, torch.stack(plans)).numpy()  # (P, n-1-t, 2)
            ra = min(t + tau, n - 1)
            u_hat = yhat[:, -1, 0] - lam * yhat[:, ra - (t + 1), 1]
            Y_off = rd.Y[i_off].numpy()
            for j, r in enumerate(gt.itertuples(), start=1):
                Y_on = rd.Y[pos[r.run_id]].numpy()
                rows.append({
                    "model": pred.name, "seed": seed, "group": r.group, "attack": r.attack,
                    "poison_frac": r.poison_frac, "trigger": r.trigger, "d": int(r.d), "k": int(r.k), "t": t,
                    "true_delta": (Y_on[-1, 0] - lam * Y_on[ra, 1]) - (Y_off[-1, 0] - lam * Y_off[ra, 1]),
                    "pred_delta": float(u_hat[j] - u_hat[0]),
                    "true_benefit_atk": Y_off[ra, 1] - Y_on[ra, 1], "pred_benefit_atk": float(yhat[0, ra - t - 1, 1] - yhat[j, ra - t - 1, 1]),
                    "true_clean_change": Y_on[-1, 0] - Y_off[-1, 0], "pred_clean_change": float(yhat[j, -1, 0] - yhat[0, -1, 0]),
                })
    return pd.DataFrame(rows)


def delta_metrics(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def one(g: pd.DataFrame) -> pd.Series:
        t, p = g.true_delta.to_numpy(), g.pred_delta.to_numpy()
        pos = t > 0
        both = pos.any() and (~pos).any()
        return pd.Series({
            "n": len(g), "true_mean": t.mean(), "pred_mean": p.mean(),
            "rmse": np.sqrt(np.mean((p - t) ** 2)),
            "corr": np.corrcoef(p, t)[0, 1] if len(g) > 2 and p.std() > 0 and t.std() > 0 else np.nan,
            "slope": np.polyfit(p, t, 1)[0] if len(g) > 2 and p.std() > 0 else np.nan,  # 1 = calibrated
            "sign_acc": np.mean(np.sign(p) == np.sign(t)),
            "auc_true_pos": roc_auc_score(pos, p) if both else np.nan,
            "share_true_pos": pos.mean(),
        })
    return df.groupby(by).apply(one, include_groups=False).reset_index()
