"""CausalCPC: no leakage (inputs, time, plan), contrastive masks, and recovery of a known effect on toy data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpc.data import RunData, input_columns  # noqa: E402
from cpc.evaluate import CPCPredictor  # noqa: E402
from cpc.model import CausalCPC, contrastive_mask  # noqa: E402
from cpc.train import Config, train_model  # noqa: E402
from analyze import X_COLS  # noqa: E402


def toy_runs(R: int = 96, n_blocks: int = 20, seed: int = 0) -> pd.DataFrame:
    """Hidden attack from block 5 in half the runs, visible in x_norm_std; W ~ Bernoulli(0.5) per block (randomised,
    ignorable). Attack success grows by 0.1 per attacked undefended block; each defended block costs 0.01 accuracy."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(R):
        attacked = r % 2 == 1
        atk, clean = 0.0, 0.85
        rows.append({"run_id": f"r{r:03d}", "seed": r, "attack": "backdoor_win" if attacked else "none",
                     "defense": "Krum", "poison_frac": 0.25, "block": -1, "W": np.nan,
                     **{c: np.nan for c in X_COLS}, "y_clean_acc": clean, "y_atk": atk, "u_active": 0})
        for b in range(n_blocks):
            w = int(rng.random() < 0.5)
            active = attacked and b >= 5
            atk = min(1.0, atk + 0.1) if active and not w else atk
            clean -= 0.01 * w
            x = {c: rng.normal() for c in X_COLS}
            x["x_norm_std"] += 3.0 * active
            rows.append({"run_id": f"r{r:03d}", "seed": r, "attack": "backdoor_win" if attacked else "none",
                         "defense": "Krum", "poison_frac": 0.25, "block": b, "W": w, **x,
                         "y_clean_acc": clean, "y_atk": atk, "u_active": int(active)})
    return pd.DataFrame(rows).astype({"W": "Int64"})


def test_inputs_are_signals_only():
    df = toy_runs(R=4)
    rd = RunData(df)
    assert input_columns(df) == list(X_COLS) and rd.enc.shape[-1] == len(X_COLS) + 3
    bad = df.assign(y_leak=0.0)
    assert "y_leak" not in input_columns(bad)  # x_columns only picks X_COLS / x2_*: outcomes cannot enter


def test_real_xp6_inputs_are_the_80_signals():
    csv = Path(__file__).resolve().parents[2] / "GRU_mnist/results/xp6_all/results/xp6/runs_xp6/all_blocks.csv"
    if not csv.exists():
        pytest.skip("xp6 data not available")
    cols = input_columns(pd.read_csv(csv, nrows=3))
    assert len(cols) == 80 and not any(c.startswith(("y_", "u_")) for c in cols)


def test_twins_are_never_negatives():
    m = contrastive_mask(torch.tensor([7, 7, 8, 9, 7]))
    assert m[0, 1] and m[0, 4] and m[1, 4] and not m[0, 0] and not m[0, 2] and not m[2, 3]


@pytest.mark.parametrize("feedback", [False, True])
def test_no_leak_from_the_future(feedback):
    torch.manual_seed(0)
    B, n, F_ = 3, 25, 11
    model = CausalCPC(F_, hidden=8, tau=5, feedback=feedback).eval()
    enc, W, t = torch.randn(B, n, F_), torch.randint(0, 2, (B, n)).float(), torch.linspace(0, 1, n)
    cut = torch.tensor([10])
    y, br, _, _ = model.cut_predictions(enc, W, t, cut)
    enc2 = enc.clone(); enc2[:, 11:] += 5.0  # future history rows
    assert torch.allclose(y, model.cut_predictions(enc2, W, t, cut)[0], atol=1e-6)
    s = 6  # step s targets row cut + 1 + s = 17: it may depend on W up to row 17 only
    W2 = W.clone(); W2[:, 18:] = 1 - W2[:, 18:]
    y2, br2, _, _ = model.cut_predictions(enc, W2, t, cut)
    assert torch.allclose(y[:, :, :s + 1], y2[:, :, :s + 1], atol=1e-6)
    W3 = W.clone(); W3[:, 17] = 1 - W3[:, 17]  # the planned W of step s itself
    y3, br3, _, _ = model.cut_predictions(enc, W3, t, cut)
    assert torch.allclose(br[:, :, :s + 1], br3[:, :, :s + 1], atol=1e-6)  # br_s never sees W_s (CLUB valid)
    assert not torch.allclose(y[:, :, s], y3[:, :, s])  # ... but the outcome head does


def test_recovers_a_known_effect():
    df = toy_runs()
    tr, va = RunData(df[df.seed < 80]), None
    va = RunData(df[df.seed >= 80], ref=tr)
    cfg = Config(hidden=16, tau=3, pretrain=True, infomax=True, pre_epochs=5, epochs=60, patience=60, lr=3e-3,
                 batch=16, cuts=8)
    model, _ = train_model(cfg, tr, va, "cpu", log=lambda *_: None)
    pred, cut, n = CPCPredictor(model, "cpu"), 10, tr.n_rows
    on = torch.zeros(n); on[cut + 1:] = 1
    plans = torch.stack([torch.zeros(n), on])
    ben = {a: [] for a in ("none", "backdoor_win")}
    for i, a in enumerate(va.attack):
        y = pred.plans(va, i, cut, plans).numpy()
        ben[a].append(y[0, -1, 1] - y[1, -1, 1])  # attack success removed by defending to the end
    assert np.mean(ben["backdoor_win"]) > np.mean(ben["none"]) + 0.2
