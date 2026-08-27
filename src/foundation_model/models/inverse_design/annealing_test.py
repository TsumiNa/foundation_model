# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the composition search's τ schedule.

None of this had a test before: the curve was three closures inside a 975-line method, reachable
only by running a full optimisation and inferring the temperature from its output. What it decides
— how fast a soft top-K commits to K elements — is a tuning knob someone will want to change, so
it is the part of that method most worth being able to read off directly.
"""

import pytest

from foundation_model.models.inverse_design.annealing import (
    TAU_END,
    TAU_FLOOR,
    AnnealingSchedule,
    interp_scalar,
    scale_to_tau,
)


@pytest.mark.parametrize("scale, expected", [(0.0, 1.0), (0.5, 5.0), (1.0, 25.0)])
def test_scale_maps_to_the_documented_temperatures(scale, expected):
    """The user-facing knob is normalised; the documented anchors are 0→1, 0.5→5, 1→25."""
    assert scale_to_tau(scale) == pytest.approx(expected)


@pytest.mark.parametrize("scale", [-1.0, 2.0])
def test_scale_is_clamped_rather_than_extrapolated(scale):
    assert scale_to_tau(scale) in (1.0, 25.0)


def test_default_schedule_falls_geometrically_to_the_final_tau():
    sched = AnnealingSchedule.build(scale=1.0, schedule=None, steps=11)

    taus = [sched.tau_for_step(i) for i in range(11)]

    assert taus[0] == pytest.approx(25.0)  # 25**1.0
    assert taus[-1] == pytest.approx(TAU_END)
    assert all(b < a for a, b in zip(taus, taus[1:])), "τ must fall monotonically"
    # Geometric means a constant ratio between consecutive steps.
    ratios = [b / a for a, b in zip(taus, taus[1:])]
    assert all(r == pytest.approx(ratios[0]) for r in ratios)


def test_a_disabled_schedule_reports_the_final_tau_at_every_step():
    """No max_elements means no soft mask to harden, so there is nothing to anneal."""
    sched = AnnealingSchedule.disabled()

    assert sched.tau_for_step(0) == sched.tau_for_step(999) == sched.final_tau
    assert sched.final_tau == max(TAU_END, TAU_FLOOR)


def test_a_single_step_search_starts_already_hard():
    """`steps <= 1` has no curve to walk — dividing by (steps - 1) would be a ZeroDivisionError."""
    assert AnnealingSchedule.build(scale=1.0, schedule=None, steps=1).tau_for_step(0) == TAU_END
    assert AnnealingSchedule.build(scale=1.0, schedule=None, steps=0).tau_for_step(0) == TAU_END


def test_dict_segments_override_the_front_and_the_tail_stays_geometric():
    """A dict ending before 1.0 hands over to the default geometric tail down to TAU_END."""
    sched = AnnealingSchedule.build(
        scale=1.0, schedule={"step": [0.5], "scale": [0.5], "annealing_func": ["constant"]}, steps=11
    )

    # "constant" holds the segment's start value (scale=1.0 → τ=25) for the whole first half.
    assert [sched.tau_for_step(i) for i in range(6)] == [pytest.approx(25.0)] * 6
    # Past step[-1] the tail runs from 25**scale[-1] = 5 down to TAU_END.
    assert sched.tau_for_step(6) < 5.0
    assert sched.tau_for_step(10) == pytest.approx(TAU_END)


def test_tau_never_goes_below_the_numerical_floor():
    """Below TAU_FLOOR, softmax(logits / τ) loses precision — the curve clamps rather than dives."""
    sched = AnnealingSchedule.build(
        scale=0.0, schedule={"step": [1.0], "scale": [0.0], "annealing_func": ["geometric"]}, steps=50
    )

    assert min(sched.tau_for_step(i) for i in range(50)) >= TAU_FLOOR


@pytest.mark.parametrize(
    "func, at_half",
    [("constant", 1.0), ("linear", 0.5), ("cosine", 0.5), ("geometric", 0.1)],
)
def test_every_interpolation_shape_is_reachable(func, at_half):
    assert interp_scalar(1.0, 0.0 if func != "geometric" else 0.01, 0.5, func) == pytest.approx(at_half)


def test_geometric_interpolation_falls_back_to_linear_on_a_zero_endpoint():
    """log-space interpolation is undefined at 0; the fallback keeps the curve finite."""
    assert interp_scalar(1.0, 0.0, 0.5, "geometric") == pytest.approx(0.5)


@pytest.mark.parametrize(
    "schedule, message",
    [
        ({"scale": [0.5], "annealing_func": ["linear"]}, "missing required keys"),
        ({"step": [0.5, 0.9], "scale": [0.5], "annealing_func": ["linear"]}, "same length"),
        ({"step": [], "scale": [], "annealing_func": []}, "non-empty"),
        ({"step": [0.0], "scale": [0.5], "annealing_func": ["linear"]}, r"must be in \(0, 1\]"),
        ({"step": [0.7, 0.3], "scale": [0.5, 0.5], "annealing_func": ["linear", "linear"]}, "strictly increasing"),
        ({"step": [0.5], "scale": [1.5], "annealing_func": ["linear"]}, r"must be in \[0, 1\]"),
        ({"step": [0.5], "scale": [0.5], "annealing_func": ["sigmoid"]}, "must be one of"),
    ],
)
def test_a_malformed_schedule_is_rejected_at_build_time(schedule, message):
    """Rejected when the search is configured, not part-way through it."""
    with pytest.raises((ValueError, TypeError), match=message):
        AnnealingSchedule.build(scale=0.5, schedule=schedule, steps=10)


def test_scale_out_of_range_is_rejected():
    with pytest.raises(ValueError, match=r"annealing_scale must be in \[0, 1\]"):
        AnnealingSchedule.build(scale=1.5, schedule=None, steps=10)
