"""Checks on the evaluation-only onset twins (no training runs needed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import onset_twins as O  # noqa: E402
import run_experiment as R  # noqa: E402
from poison_testbed.runner import make_schedule  # noqa: E402


def test_split_matches_gru_assign_splits():
    gru = pytest.importorskip("gru_experiment")  # local only (not bundled on Kaggle)
    main = R.xp6_configs({})[: R.XP6_MAIN]
    meta = pd.DataFrame([{"seed": c.seed, "attack": c.attack, "poison_frac": c.poison_frac, "defense": c.defense}
                         for c in main])
    assert list(gru.assign_splits(meta) == "test") == O.test_split(main)


def test_plans_share_the_off_prefix_and_are_on_exactly_k_blocks():
    configs, man = O.onset_configs()
    assert len(configs) == len(man) == 555 and man.run_id.is_unique
    ref = {r.seed: make_schedule(c) for c, r in zip(configs, man.itertuples()) if r.plan == "off"}
    assert all(sum(w) == 0 for w in ref.values())
    for c, r in zip(configs, man.itertuples()):
        if r.plan == "on":
            w = make_schedule(c)
            assert w[: r.t] == ref[r.seed][: r.t] and sum(w) == r.k and all(w[r.t: r.t + r.k])


def test_only_test_and_heldout_seeds_never_train_or_val():
    configs = R.xp6_configs({})
    main = configs[: R.XP6_MAIN]
    allowed = {c.seed for c, t in zip(main, O.test_split(main)) if t} | {c.seed for c in configs[R.XP6_MAIN + R.XP6_TWINS:]}
    _, man = O.onset_configs()
    assert set(man.seed) <= allowed
    assert man.groupby("group").seed.nunique().to_dict() == {"attacked": 53, "heldout": 40, "none": 16}
