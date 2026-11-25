"""xp8 learning + live: data bank from real (short) simulations, gain labels, causality, live inference = offline
inference, live runs reproduce the trunk when the model never defends."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "xp8" if (HERE / "xp8").is_dir() else HERE))  # local: CPC_DP/xp8/; Kaggle: flat
import learn8 as L  # noqa: E402
import live8 as V  # noqa: E402
import xp8 as X  # noqa: E402

N, KK = 30, 4  # steps per run, branch length in these tests
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def data():
    """3 setups (train / val / test) x 2 trunks, 30 steps, branches at 5 and 12 (k = 4)."""
    s = X.setups()
    picks = []
    for fam, split in (("corner", "train"), ("flip", "val"), ("patch", "test")):
        r = s[(s.family == fam) & (s.split == split)].iloc[0].to_dict()
        r.update(onset=4, end=20, branch_points="5 12", n_attackers=3)
        picks.append(r)
    old_k = L.K
    rows = []
    for r in picks:
        for trunk in (0, 1):
            df, info = X.simulate_job(X.config(r, trunk, N), trunk, [5, 12], KK)
            assert info["restore_mismatch"] == [] and not info["errors"]
            df.insert(0, "setup", r["setup"])
            df.insert(1, "trunk", trunk)
            rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    man = pd.DataFrame(picks)
    return df, man, old_k


def test_bank(data):
    df, man, _ = data
    bank = L.build_bank(df, man)
    assert bank.x.shape == (3 * 2 * 3, N, len(L.INPUTS)) and bank.n == N
    assert not any(c.startswith(("y_", "u_")) for c in L.INPUTS[:-1]) and L.INPUTS[-1] == "x_val_acc_pre"
    assert len(bank.labels) == 3 * 2 * 2
    for r in bank.labels.itertuples():
        tr, br = r.traj, r.branch
        assert np.array_equal(bank.x[br, :r.t + 1], bank.x[tr, :r.t + 1])  # same history up to the signals of t
        assert np.array_equal(bank.W[br, :r.t], bank.W[tr, :r.t]) and bank.W[br, r.t] == 1 - r.trunk
        assert bank.W[br, r.t + KK] == r.trunk
        u = bank.y[..., 0] - L.LAMBDA * bank.y[..., 1]
        g = (u[br, r.t:].mean() - u[tr, r.t:].mean()) * (1 if r.trunk == 0 else -1)
        assert np.isclose(r.gain, g)
        # targeted attack: y[..., 1] is y_atk from the raw rows
        raw = df[(df.setup == r.setup) & (df.trunk == r.trunk) & (df.t_branch == -1) & (df.step >= 0)]
        assert np.allclose(bank.y[tr, :, 1], raw.y_atk.fillna(0).to_numpy())
    assert set(bank.meta.split) == {"train", "val", "test"}


def test_norm_row_equals_enc(data):
    df, man, _ = data
    bank = L.build_bank(df, man)
    norm = L.Norm.fit(bank.x[(bank.meta.t_branch == -1).to_numpy()])
    enc = norm.enc(bank.x[3], bank.W[3])
    for t in (0, 1, 7, N - 1):
        assert np.array_equal(enc[t], norm.row(bank.x[3, t], bank.W[3, t - 1] if t else 0.0, t))


@pytest.fixture(scope="module")
def trained(data, tmp_path_factory):
    df, man, _ = data
    bank = L.build_bank(df, man)
    norm = L.Norm.fit(bank.x[((bank.meta.split == "train") & (bank.meta.t_branch == -1)).to_numpy()])
    hp = L.HP(hidden=16, a_epochs=2, a_patience=5, pre_epochs=2, pre_patience=5, b_epochs=2, b_patience=5)
    out = tmp_path_factory.mktemp("learn")
    models = {}
    for kind in ("A", "B"):
        model, hist = (L.train_A if kind == "A" else L.train_B)(bank, norm, 0, hp, DEV, log=lambda s: None)
        L.save_model(out / kind, kind, model, norm, 0, hp, hist)
        models[kind] = model
    return bank, norm, models, out


@pytest.mark.parametrize("kind", ["A", "B"])
def test_live_inference_equals_offline(trained, kind):
    """Controller (one step at a time, as in the live runs) = predict() (whole history at once), at every step."""
    bank, norm, models, out = trained
    for traj in bank.meta.index[bank.meta.t_branch == -1][:2]:
        trunk = int(bank.meta.trunk[traj])
        labels = pd.DataFrame({"traj": traj, "t": np.arange(N), "trunk": trunk})
        off = L.predict(kind, models[kind], bank, norm, labels, DEV)
        raw_x = np.sign(bank.x[traj]) * np.expm1(np.abs(bank.x[traj]))  # undo the signed log: live gets raw values
        ctrl = L.Controller(out / kind, DEV, N)
        live = []
        for t in range(N):
            live.append(ctrl.gain(raw_x[t]))
            ctrl.update(int(bank.W[traj, t]))
        assert np.allclose(live, off, atol=1e-5), (kind, np.abs(np.array(live) - off).max())


@pytest.mark.parametrize("kind", ["A", "B"])
def test_causal(trained, kind):
    """The prediction at step t does not change when the signals after t change."""
    bank, norm, models, _ = trained
    traj = int(bank.meta.index[bank.meta.t_branch == -1][0])
    labels = pd.DataFrame({"traj": traj, "t": [10], "trunk": int(bank.meta.trunk[traj])})
    a = L.predict(kind, models[kind], bank, norm, labels, DEV)
    saved = bank.x[traj, 11:].copy()
    bank.x[traj, 11:] = 5.0
    try:
        b = L.predict(kind, models[kind], bank, norm, labels, DEV)
    finally:
        bank.x[traj, 11:] = saved
    assert np.allclose(a, b, atol=1e-6)


def test_offline_tables(trained):
    bank, norm, models, _ = trained
    Lb = bank.labels.copy()
    Lb["pred"] = L.predict("A", models["A"], bank, norm, Lb, DEV)
    d = L.decisions(Lb)
    assert np.allclose(d.regret, d.oracle - d.value) and (d.regret >= -1e-12).all()
    s = L.summarise(d, ["family"])
    assert set(s.index) == set(Lb.family) and (s.n > 0).all()


def test_live_never_defending_reproduces_off_trunk(data, trained, tmp_path):
    """A model whose predicted gain is always negative never defends: the live run must equal the off trunk."""
    df, man, _ = data
    bank, norm, models, out = trained
    folder = tmp_path / "never"
    m = L.GainGRU(L.N_IN, 16)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.head[-1].bias.fill_(-1.0)
    L.save_model(folder, "A", m, norm, 0, L.HP(hidden=16), {})
    setup = man.iloc[0].to_dict()
    res = V.live_job("A", setup, str(folder), N, str(tmp_path))
    assert res["status"] == "done", res["error"]
    lv = pd.read_parquet(tmp_path / f"s{setup['setup']:03d}.parquet")
    assert (lv.W == 0).all() and (lv.g_hat < 0).all() and list(lv.step) == list(range(N))
    tr = df[(df.setup == setup["setup"]) & (df.trunk == 0) & (df.t_branch == -1) & (df.step >= 0)]
    cols = X.SIGNALS + ["x_val_acc_pre"] + X.OUTCOMES
    assert np.array_equal(lv[cols].to_numpy(float), tr[cols].to_numpy(float), equal_nan=True)
    # the summary sees the same thing
    trk = tr.assign(t_branch=-1)
    s = V.summarise(lv, pd.concat([trk, df[(df.setup == setup["setup"]) & (df.trunk == 1) & (df.t_branch == -1)
                                          & (df.step >= 0)]]), man)
    assert bool(s.prefix_exact.iloc[0]) and s.defend_rate.iloc[0] == 0 and np.isclose(s.U_model.iloc[0], s.U_off.iloc[0])


def test_live_follows_its_rule(data, trained, tmp_path):
    df, man, _ = data
    _, _, _, out = trained
    setup = man.iloc[1].to_dict()
    res = V.live_job("B", setup, str(out / "B"), N, str(tmp_path))
    assert res["status"] == "done", res["error"]
    lv = pd.read_parquet(tmp_path / f"s{setup['setup']:03d}.parquet")
    assert ((lv.g_hat > 0).astype(int) == lv.W).all()
    assert np.isfinite(lv[X.SIGNALS].to_numpy(float)).all()
