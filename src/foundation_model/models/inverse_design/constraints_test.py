# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the composition search's constraint system.

The five knobs can contradict each other, and an uncaught contradiction does not crash — it
produces a plausible-looking recipe. That is the whole reason this validation exists, and it is
what these tests are mostly about. Until it was a module the only way to exercise any of it was to
call a twenty-keyword method and start a real optimisation.
"""

import pytest
import torch

from foundation_model.models.inverse_design.constraints import CompositionConstraints
from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS

N = len(DEFAULT_ELEMENTS)
SYMS = list(DEFAULT_ELEMENTS[:6])


def _build(**kwargs) -> CompositionConstraints:
    return CompositionConstraints.build(n_components=N, **kwargs)


def _seed(batch: int = 3, *, at: dict[int, float] | None = None) -> torch.Tensor:
    w = torch.zeros(batch, N)
    for idx, val in (at or {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}).items():
        w[:, idx] = val
    return w


# --- each knob on its own -------------------------------------------------------------------


def test_no_constraints_is_all_none():
    """The unconstrained search must not pay for machinery it does not use."""
    c = _build()

    assert c.elem_mask is None and c.step_scale is None and c.fixed_mask is None
    assert not c.schedule.active


def test_a_whitelist_becomes_a_boolean_mask_over_the_registry():
    c = _build(allowed_elements=SYMS)

    assert c.elem_mask is not None
    assert int(c.elem_mask.sum()) == len(SYMS)
    assert bool(c.elem_mask[DEFAULT_ELEMENTS.index(SYMS[0])])


def test_a_uniform_step_scale_of_one_stays_none():
    """1.0 means "no scaling"; materialising a vector of ones would cost a multiply per step."""
    assert _build(element_step_scale=1.0).step_scale is None
    assert _build(element_step_scale=0.5).step_scale is not None


@pytest.mark.parametrize(
    "kwargs, exc, message",
    [
        ({"allowed_elements": "some"}, ValueError, "must be 'all'"),
        ({"allowed_elements": []}, ValueError, "non-empty"),
        ({"allowed_elements": ["Xx"]}, ValueError, "Unknown element symbol"),
        ({"allowed_elements": 7}, TypeError, "must be 'all' or a non-empty list"),
        ({"element_step_scale": -1.0}, ValueError, "must be >= 0"),
        ({"element_step_scale": {"Xx": 0.5}}, ValueError, "Unknown element symbol"),
        ({"element_step_scale": "x"}, TypeError, "non-negative float or a mapping"),
        ({"fixed_amounts": {}}, ValueError, "non-empty"),
        ({"fixed_amounts": [1]}, TypeError, "must be a mapping"),
        ({"fixed_amounts": {SYMS[0]: 1.5}}, ValueError, "strictly between 0 and 1"),
        ({"min_nonzero_weight": 1.5}, ValueError, r"must be in \[0, 1\]"),
        ({"max_elements": 0}, ValueError, r"must be in \[1, n_components"),
        ({"max_elements": True}, TypeError, "must be an int or None"),
    ],
)
def test_a_malformed_knob_is_rejected(kwargs, exc, message):
    with pytest.raises(exc, match=message):
        _build(**kwargs)


# --- knobs contradicting each other ----------------------------------------------------------


def test_pinning_an_element_the_whitelist_excludes_is_rejected():
    with pytest.raises(ValueError, match="not in allowed_elements"):
        _build(allowed_elements=SYMS[:3], fixed_amounts={SYMS[5]: 0.3})


def test_pinning_the_same_element_twice_is_rejected():
    """fixed_amounts and element_step_scale=0 are both hard locks; one element, one mechanism."""
    with pytest.raises(ValueError, match="appear in both element_step_scale=0 and"):
        _build(fixed_amounts={SYMS[0]: 0.3}, element_step_scale={SYMS[0]: 0.0})


def test_pins_claiming_the_whole_simplex_are_rejected():
    with pytest.raises(ValueError, match="must be strictly less than 1.0"):
        _build(fixed_amounts={SYMS[0]: 0.6, SYMS[1]: 0.5})


def test_a_floor_above_one_over_k_is_infeasible():
    """K elements each ≥ floor can only sum to 1 if K · floor ≤ 1."""
    with pytest.raises(ValueError, match="exceeds 1 / max_elements"):
        _build(min_nonzero_weight=0.5, max_elements=4)


def test_a_floor_cannot_overrule_an_explicit_pin():
    with pytest.raises(ValueError, match="The floor cannot override an explicit pin"):
        _build(fixed_amounts={SYMS[0]: 0.01}, min_nonzero_weight=0.1)


def test_a_cardinality_limit_wider_than_the_whitelist_is_rejected():
    with pytest.raises(ValueError, match="exceeds the number of allowed elements"):
        _build(allowed_elements=SYMS[:3], max_elements=5)


def test_locking_every_slot_the_cardinality_limit_allows_is_rejected():
    """Equality, not just excess: the lock-paste needs one unlocked slot for the leftover mass.

    With K == n_locked the paste has nowhere to put 1 − Σ locked, and the rows come out summing to
    less than 1 — a broken simplex that nothing downstream checks for.
    """
    with pytest.raises(ValueError, match="must be > total locked elements"):
        _build(max_elements=1, element_step_scale={SYMS[0]: 0.0})


def test_annealing_knobs_are_unvalidated_without_a_cardinality_limit():
    """They describe how a constraint hardens; with no constraint they describe nothing.

    Rejecting them here would fail a search over an argument it never reads.
    """
    assert not _build(annealing_scale=1.5).schedule.active
    with pytest.raises(ValueError, match="annealing_scale must be in"):
        _build(annealing_scale=1.5, max_elements=4)


# --- locks, once there is a batch --------------------------------------------------------------


def test_no_locks_resolves_to_nothing():
    mask, values = _build().resolve_locks(w0_seed=None, batch_size=3, dtype=torch.float32)

    assert mask is None and values is None


def test_a_step_scale_lock_pins_at_the_seed_value():
    c = _build(element_step_scale={SYMS[0]: 0.0})

    mask, values = c.resolve_locks(w0_seed=_seed(), batch_size=3, dtype=torch.float32)

    idx = DEFAULT_ELEMENTS.index(SYMS[0])
    assert mask is not None and bool(mask[idx]) and int(mask.sum()) == 1
    assert values is not None and torch.allclose(values[:, idx], torch.full((3,), 0.25))
    assert float(values.sum()) == pytest.approx(3 * 0.25), "only locked positions carry value"


def test_a_step_scale_lock_without_a_seed_is_rejected():
    """There is no other source for the per-row value it is supposed to hold."""
    c = _build(element_step_scale={SYMS[0]: 0.0})

    with pytest.raises(ValueError, match="requires initial_weights"):
        c.resolve_locks(w0_seed=None, batch_size=3, dtype=torch.float32)


def test_locking_a_disallowed_element_is_rejected():
    c = _build(allowed_elements=SYMS[:3], element_step_scale={SYMS[5]: 0.0})

    with pytest.raises(ValueError, match="must also be in allowed_elements"):
        c.resolve_locks(w0_seed=_seed(), batch_size=3, dtype=torch.float32)


def test_a_fixed_amount_needs_no_seed_and_broadcasts_to_the_batch():
    c = _build(fixed_amounts={SYMS[1]: 0.4})

    mask, values = c.resolve_locks(w0_seed=None, batch_size=5, dtype=torch.float32)

    idx = DEFAULT_ELEMENTS.index(SYMS[1])
    assert mask is not None and int(mask.sum()) == 1
    assert values is not None and values.shape == (5, N)
    assert torch.allclose(values[:, idx], torch.full((5,), 0.4))


def test_both_lock_sources_combine_into_one_mask():
    """Validated disjoint at build time, so the masks OR and the values add."""
    c = _build(element_step_scale={SYMS[0]: 0.0}, fixed_amounts={SYMS[4]: 0.1})

    mask, values = c.resolve_locks(w0_seed=_seed(), batch_size=3, dtype=torch.float32)

    assert mask is not None and int(mask.sum()) == 2
    assert values is not None
    assert float(values[0, DEFAULT_ELEMENTS.index(SYMS[0])]) == pytest.approx(0.25)
    assert float(values[0, DEFAULT_ELEMENTS.index(SYMS[4])]) == pytest.approx(0.1)


def test_combined_locks_claiming_more_than_the_whole_simplex_are_rejected():
    """Each source is under 1 on its own; only their sum is impossible, and only per row.

    Neither could catch this at build time — the seed lock's values are per-row and arrive with the
    batch.
    """
    c = _build(element_step_scale={SYMS[0]: 0.0}, fixed_amounts={SYMS[4]: 0.65})

    with pytest.raises(ValueError, match="Combined locked mass exceeds 1.0"):
        c.resolve_locks(w0_seed=_seed(at={0: 0.5}), batch_size=3, dtype=torch.float32)


def test_a_seed_lock_below_the_floor_is_rejected():
    """fixed_amounts is checked at build time; a seed lock's value is only knowable here."""
    c = _build(element_step_scale={SYMS[0]: 0.0}, min_nonzero_weight=0.3)

    with pytest.raises(ValueError, match="falls below min_nonzero_weight"):
        c.resolve_locks(w0_seed=_seed(), batch_size=3, dtype=torch.float32)


# --- handoff ------------------------------------------------------------------------------------


def test_the_projector_carries_every_constraint_across():
    c = _build(allowed_elements=SYMS, min_nonzero_weight=0.05, max_elements=4, fixed_amounts={SYMS[1]: 0.3})
    mask, values = c.resolve_locks(w0_seed=None, batch_size=2, dtype=torch.float32)

    project = c.projector(mask, values)

    assert project.n_components == N
    assert project.elem_mask is c.elem_mask
    assert project.min_nonzero_weight == 0.05
    assert project.max_elements == 4
    assert project.locked_mask is mask and project.locked_w0 is values


def test_moving_to_a_device_preserves_every_field():
    c = _build(
        allowed_elements=SYMS,
        element_step_scale={SYMS[0]: 0.5},
        fixed_amounts={SYMS[1]: 0.3},
        min_nonzero_weight=0.05,
        max_elements=4,
    )

    moved = c.on(device=torch.device("cpu"), dtype=torch.float64)

    assert moved.step_scale is not None and moved.step_scale.dtype == torch.float64
    assert moved.fixed_w0 is not None and moved.fixed_w0.dtype == torch.float64
    assert moved.elem_mask is not None and moved.elem_mask.dtype == torch.bool  # masks stay bool
    assert moved.min_nonzero_weight == c.min_nonzero_weight
    assert moved.max_elements == c.max_elements
    assert moved.schedule == c.schedule
