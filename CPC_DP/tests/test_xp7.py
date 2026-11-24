"""xp7 plan: splits shared across gamma levels, unique runs, twins only on val/test setups and sharing the 'off'
prefix, the gamma policy's propensity."""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "xp7" if (HERE / "xp7").is_dir() else HERE))  # local: CPC_DP/xp7/; Kaggle: flat

import xp7  # noqa: E402
from poison_testbed.alarm import GammaSchedule  # noqa: E402
from poison_testbed.runner import make_schedule  # noqa: E402


def test_levels_share_setups_and_splits():
    a_cfg, a = xp7.main_configs(["g0", "g2"], {})
    b_cfg, b = xp7.main_configs(["g8", "hard"], {})
    import pandas as pd
    m = pd.concat([a, b])
    per_base = m.groupby("base").agg(levels=("level", "nunique"), splits=("split", "nunique"), seeds=("seed", "nunique"))
    assert (per_base.levels == 4).all() and (per_base.splits == 1).all() and (per_base.seeds == 1).all()
    t_cfg, t = xp7.twin_configs({})
    ids = [c.run_id for c in a_cfg + b_cfg + t_cfg]
    assert len(ids) == len(set(ids))


def test_twins_only_val_test_and_share_prefix():
    cfgs, man = xp7.twin_configs({})
    assert set(man.split) <= {"val", "test"}
    off = {c.seed: make_schedule(c) for c, r in zip(cfgs, man.itertuples()) if r.plan == "off"}
    for c, r in zip(cfgs, man.itertuples()):
        if r.plan == "on":
            s = make_schedule(c)
            assert s[: r.t] == off[c.seed][: r.t] and all(s[r.t: r.t + r.k]) and not any(s[r.t + r.k:])


def test_no_persistent_signflip():
    b = xp7.base_setups({})
    assert not (b.persistent & (b.family == "signflip")).any()
    assert (b[b.persistent].window.map(lambda w: w[1]) == 249).all()


def test_gamma_policy_propensity():
    row = {"block": 0, "W": 0, "defense": "Krum"}
    g0 = GammaSchedule(0.0, 3.0, 10, {}, seed=1)
    ws = [g0([row], b) for b in range(200)]
    assert all(e["p_on"] == 0.5 for _, e in ws)
    assert all(w == ws[b - b % 10][0] for b, (w, _) in enumerate(ws))  # held between decisions
    g8 = GammaSchedule(8.0, 3.0, 10, {}, seed=1)  # unscorable history: s = -inf -> p = 0
    assert g8([row], 0) == (0, {"p_on": 0.0, "decision": 1})
    assert not math.isnan(g8([row], 1)[1]["p_on"])
