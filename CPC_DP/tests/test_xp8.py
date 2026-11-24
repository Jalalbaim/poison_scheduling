"""xp8: plan invariants, the gather/apply split, save/resume (bit-exact), branch semantics, a full job end to end."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "xp8" if (HERE / "xp8").is_dir() else HERE))  # local: CPC_DP/xp8/; Kaggle: flat
import xp8 as X  # noqa: E402
from poison_testbed.runner import build  # noqa: E402

N = 30  # steps in the small runs below


def _setup(family: str, **over) -> dict:
    s = X.setups()
    r = s[s.family == family].iloc[0].to_dict()
    r.update(onset=4, end=20 if family != "none" else -1, **over)
    return r


# ----------------------------------------------------------------------------- plan
def test_plan():
    s = X.setups()
    assert len(s) == 264 and len(X.CONFIGS) == 34 and s.seed.is_unique
    assert s.seed.min() == X.SEED0 and s.seed.max() < X.SEED0 + 1000  # xp5-xp7 seeds are 30k-70k
    per = s.groupby(["config", "split"]).size().unstack(fill_value=0)
    assert (per.loc[per.index != "no attack", "test"] == 1).all() and (per.loc[per.index != "no attack", "val"] == 1).all()
    assert (per.loc[per.index != "no attack", "train"] == 4).all()
    assert s.groupby("config").part.nunique().min() == 2  # every configuration in both sessions
    assert s.groupby("part").size().to_dict() == {1: 132, 2: 132}
    assert not ((s.family == "signflip") & s.persistent).any()
    assert set(s[s.family == "alie"].n_attackers) <= {3, 5} and set(s[s.family == "signflip"].n_attackers) <= {1, 3}
    half = s[(s.family == "alie") & s.config.str.contains("half")].alie_z
    assert all(isinstance(z, float) and z > 0 for z in half)
    for r in s.itertuples():
        bps = [int(v) for v in r.branch_points.split()]
        assert bps == sorted(set(bps)) and 0 <= bps[0] and bps[-1] < X.T - 1 and len(bps) >= 3
        if r.family != "none":
            assert r.onset in bps and 0 < r.onset <= r.end <= X.T - 1
            if r.persistent:
                assert r.end == X.T - 1
    j = X.jobs_of(s)
    assert len(j) == 528 and j.job.is_unique


# ----------------------------------------------------------------------------- testbed
def test_gather_apply_equals_step():
    cfg = X.config(_setup("corner"), 0, N)
    a, b = build(cfg), build(cfg)
    for t in range(N):
        xa = a.step()
        xb = b.gather()
        b.apply()
        assert xa == xb
    assert torch.equal(a._model.parameters, b._model.parameters)
    with pytest.raises(RuntimeError):
        b.apply()


def test_n_attackers():
    for n_att in (1, 5):
        cfg = X.config(_setup("corner", n_attackers=n_att), 0, N)
        sim = build(cfg)
        assert sim.n_byz == n_att and sim.f == 3
        calls = []
        spec = sim.attack_spec
        sim.attack_spec = replace(spec, corrupt=lambda x, y, **kw: (calls.append(1), spec.corrupt(x, y, **kw))[1])
        for _ in range(5):  # onset 4: steps 0-3 clean, step 4 attacked
            sim.step()
        assert len(calls) == n_att


# ----------------------------------------------------------------------------- save / resume
@pytest.mark.parametrize("family", ["flip", "alie", "patch", "signflip"])
def test_resume_is_bit_exact(family):
    """k = 0: a branch is the trunk resumed from a snapshot; it must reproduce the trunk exactly."""
    r = _setup(family)
    df, info = X.simulate_job(X.config(r, 1, N), 1, [3, 11], 0)
    trunk = df[df.traj == "trunk"].set_index("step")
    for t in (3, 11):
        br = df[df.t_branch == t].set_index("step")
        assert list(br.index) == list(range(t, N))
        cols = X.SIGNALS + ["x_val_acc_pre", "W", "u_active"] + X.OUTCOMES
        assert np.array_equal(br[cols].to_numpy(float), trunk.loc[t:, cols].to_numpy(float), equal_nan=True)
    assert info["restore_mismatch"] == [] and info["branches_ok"] == 2 and not info["errors"]


def test_branch_semantics():
    """Off trunk, branch at 6 with k = 5: W = 1 on [6, 11), 0 after; step 6's signals equal the trunk's (they
    precede W_6) but its outcomes differ (W_6 changed the update); the branch differs from the trunk afterwards."""
    r = _setup("patch", n_attackers=5, poison_frac=1.0)
    df, info = X.simulate_job(X.config(r, 0, N), 0, [6], 5)
    trunk = df[df.traj == "trunk"].set_index("step")
    br = df[df.t_branch == 6].set_index("step")
    assert (trunk.loc[0:, "W"] == 0).all()
    assert list(br.W) == [1] * 5 + [0] * (N - 11)
    assert np.array_equal(br.loc[6, X.SIGNALS].to_numpy(float), trunk.loc[6, X.SIGNALS].to_numpy(float))
    assert br.loc[6, "x_val_acc_pre"] == trunk.loc[6, "x_val_acc_pre"]
    assert br.loc[6, "y_clean_loss"] != trunk.loc[6, "y_clean_loss"]
    assert not np.allclose(br.loc[7, X.SIGNALS].to_numpy(float), trunk.loc[7, X.SIGNALS].to_numpy(float))
    assert info["restore_mismatch"] == []
    # x_val_acc_pre(7) is measured after step 6, which differs between trunk and branch: it must differ too
    assert br.loc[7, "x_val_acc_pre"] != trunk.loc[7, "x_val_acc_pre"]


def test_run_job_end_to_end(tmp_path):
    s = X.smoke_setups()
    (tmp_path / "jobs").mkdir()
    for setup in s.to_dict("records")[:2]:
        bps = [int(v) for v in setup["branch_points"].split()]
        for trunk in (0, 1):
            job = {"job": f"s{setup['setup']:03d}_{trunk}", "setup": setup["setup"], "trunk": trunk,
                   "n_branches": len(bps)}
            res = X.run_job(job, setup, str(tmp_path), 40, 8)
            assert res["status"] == "done", res["error"]
            df = pd.read_parquet(tmp_path / "jobs" / f"{job['job']}.parquet")
            assert list(df.columns) == X.COLUMNS
            assert (df.traj == "trunk").sum() == 41
            for t in bps:
                assert (df.t_branch == t).sum() == 40 - t
            body = df[df.step >= 0]
            assert body[X.SIGNALS].notna().all().all() and body.x_val_acc_pre.notna().all()
            assert np.isfinite(body[X.SIGNALS].to_numpy(float)).all()
