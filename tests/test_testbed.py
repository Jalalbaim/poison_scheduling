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


def test_counterfactual_schedules_share_the_defended_prefix():
    from poison_testbed.runner import counterfactual_schedule

    scheds = {n: counterfactual_schedule(n, 3, 50) for n in ["on", "off", "ontil20", "offtil20", "alt"]}
    assert all(len(s) == 50 and s[:3] == [1, 1, 1] for s in scheds.values())
    assert sum(scheds["off"]) == 3 and sum(scheds["on"]) == 50
    assert scheds["ontil20"][19] == 1 and scheds["ontil20"][20] == 0
    assert scheds["offtil20"][19] == 0 and scheds["offtil20"][20] == 1
    assert scheds["alt"][3:7] == [0, 1, 0, 1]


def test_sticky_and_react_schedules():
    from poison_testbed.runner import make_schedule

    base = dict(attack="none", defense="Krum", seed=5, C=1.0, sigma=0.0, n_blocks=250, block_len=2, window=(40, 99))
    sticky = make_schedule(RunConfig(schedule="sticky_p0.02", **base))
    assert sticky == make_schedule(RunConfig(schedule="sticky_p0.02", **base))  # seeded
    assert len(sticky) == 250 and set(sticky) <= {0, 1}
    switches = sum(a != b for a, b in zip(sticky, sticky[1:]))
    assert switches < 25  # sticky: long stretches
    react = make_schedule(RunConfig(schedule="react_d3_e5", **base))
    assert react.index(1) == 43 and max(i for i, w in enumerate(react) if w) == 104 and sum(react) == 62
    early = make_schedule(RunConfig(schedule="react_d-4_e0", **base))
    assert early.index(1) == 36


def test_xp3_schedules_and_backward_compatible_hash():
    from poison_testbed.runner import make_schedule

    base = dict(attack="none", defense="Krum", seed=5, C=1.0, sigma=0.0, n_blocks=60, block_len=2, window=(20, 39))
    probe = make_schedule(RunConfig(schedule="probe_P10_L2_o3", **base))
    assert [b for b, w in enumerate(probe) if w][:4] == [3, 4, 13, 14]
    hold = make_schedule(RunConfig(schedule="probehold_P10_L2_o3_d2_e3", **base))
    # probes before (block 3), hold over blocks 22..42, then probes resume (43-44, 53-54) with gaps between
    assert hold[3] == 1 and all(hold[22:43]) and hold[45:53] == [0] * 8 and hold[53:55] == [1, 1]
    twin = make_schedule(RunConfig(schedule="probe_P10_L2_o3+flip_s5_l2", **base))
    assert [b for b, (a, t) in enumerate(zip(probe, twin)) if a != t] == [5, 6]
    # New fields at their defaults leave the hash unchanged (xp1/xp2 configs stay valid).
    cfg = RunConfig(**base)
    assert cfg.hash() == RunConfig(**base, burst=None, always_measure_backdoor=False).hash()
    assert cfg.hash() != RunConfig(**base, always_measure_backdoor=True).hash()


def test_backdoor_measured_in_no_attack_runs_when_asked():
    on = simulate(RunConfig(attack="none", defense="Krum", seed=0, always_measure_backdoor=True, **SHORT))
    off = simulate(RunConfig(attack="none", defense="Krum", seed=0, **SHORT))
    assert on.y_atk.notna().all() and on.y_atk.max() < 0.5  # base rate, no backdoor
    assert off.y_atk.isna().all()
    assert (on.poison_frac == 1.0).all()


def test_no_attack_uses_all_workers_honestly():
    """attack=none with f=3: the 3 'Byzantine' workers are honest, so it equals an f=3 run whose window never opens."""
    a = simulate(RunConfig(attack="none", defense="Krum", seed=2, **SHORT))
    b = simulate(RunConfig(attack="backdoor_win", defense="Krum", seed=2, window=(99, 99), **SHORT))
    cols = [c for c in a.columns if c.startswith(("x_", "y_clean"))]
    pd.testing.assert_frame_equal(a[cols], b[cols])
