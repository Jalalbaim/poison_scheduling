import math

import pandas as pd
import pytest
import torch

import poison_testbed  # noqa: F401  (sets CUBLAS_WORKSPACE_CONFIG)
from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.median import Median
from poison_testbed.aggregators import ClipNoise, ScheduledAggregator
from poison_testbed.runner import RunConfig, simulate

SHORT = dict(C=3.0, sigma=0.01, n_blocks=3, block_len=4)


def test_clipnoise_clips_and_scales_noise():
    g = torch.tensor([[3.0, 4.0], [0.3, 0.4]])  # norms 5 and 0.5
    out = ClipNoise.aggregate(g, C=1.0, sigma=0.0)
    assert torch.allclose(out, torch.tensor([(0.6 + 0.3) / 2, (0.8 + 0.4) / 2]))
    gen = torch.Generator().manual_seed(0)
    n, d, C, sigma = 16, 200_000, 2.0, 0.5
    out = ClipNoise.aggregate(torch.zeros(n, d), C=C, sigma=sigma, generator=gen)
    assert out.std().item() == pytest.approx(sigma * C / math.sqrt(n), rel=0.01)


def test_scheduled_aggregator_dispatches_by_block():
    g = torch.tensor([[0.0], [0.0], [9.0]])
    kw = dict(plain=Average, defense=Median, schedule=[0, 1], block_len=5)
    assert ScheduledAggregator.aggregate(g, round=4, **kw).item() == 3.0
    assert ScheduledAggregator.aggregate(g, round=5, **kw).item() == 0.0
    with pytest.raises(ValueError):
        ScheduledAggregator.aggregate(g, round=10, **kw)


def test_same_seed_same_schedule_is_bit_identical():
    cfg = RunConfig(attack="backdoor", defense="ClipNoise", seed=3, **SHORT)
    pd.testing.assert_frame_equal(simulate(cfg), simulate(cfg))


def test_schedule_does_not_shift_random_streams():
    """Common random numbers: with defense=Average, W changes nothing, so 'on' and 'off' runs must be identical.

    This fails if turning the defense on consumed data, permutation or noise draws differently.
    """
    cols = lambda df: df.drop(columns=["run_id", "W", "defense"])  # noqa: E731
    on = simulate(RunConfig(attack="label_flip", defense="Average", seed=1, schedule="on", **SHORT))
    off = simulate(RunConfig(attack="label_flip", defense="Average", seed=1, schedule="off", **SHORT))
    pd.testing.assert_frame_equal(cols(on), cols(off))
    # And with a real defense, the initial state is shared and the runs then diverge.
    on = simulate(RunConfig(attack="label_flip", defense="ClipNoise", seed=1, schedule="on", **SHORT))
    pd.testing.assert_frame_equal(cols(on).iloc[:1], cols(off).iloc[:1])
    assert on.y_clean_acc.iloc[-1] != off.y_clean_acc.iloc[-1]


def test_window_and_labels():
    cfg = RunConfig(attack="backdoor_win", defense="Krum", seed=0, window=(1, 1), **SHORT)
    df = simulate(cfg)
    assert df.u_active.tolist() == [0, 0, 1, 0]
    assert df.u_type.tolist() == ["none", "none", "backdoor_win", "none"]
    assert df.y_atk.notna().all()
    assert simulate(RunConfig(attack="alie", defense="Krum", seed=0, **SHORT)).y_atk.isna().all()


def test_no_attack_uses_all_workers_honestly():
    """attack=none with f=3: the 3 'Byzantine' workers are honest, so it equals an f=3 run whose window never opens."""
    a = simulate(RunConfig(attack="none", defense="Krum", seed=2, **SHORT))
    b = simulate(RunConfig(attack="backdoor_win", defense="Krum", seed=2, window=(99, 99), **SHORT))
    cols = [c for c in a.columns if c.startswith(("x_", "y_clean"))]
    pd.testing.assert_frame_equal(a[cols], b[cols])
