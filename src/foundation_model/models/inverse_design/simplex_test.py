# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the logits → legal-composition projection.

This is the mathematical content of the composition search, and until it was a module the only way
to reach it was to run a three-hundred-step optimisation and inspect the recipe that fell out.
What it must guarantee — every row on the simplex, at most K non-zero, pinned elements pinned,
nothing below the floor, and gradient still flowing — is asserted here directly instead.
"""

import pytest
import torch

from foundation_model.models.inverse_design.simplex import (
    SimplexProjector,
    hard_topk_project,
    soft_topk_mask,
)

N = 8  # components in these fixtures
B = 4  # rows


def _logits(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, N) * 2.0


def _mask(*idx: int) -> torch.Tensor:
    m = torch.zeros(N, dtype=torch.bool)
    for i in idx:
        m[i] = True
    return m


# --- the selection maths ------------------------------------------------------------------------


@pytest.mark.parametrize("K", [1, 3, N])
@pytest.mark.parametrize("tau", [25.0, 1.0, 0.01])
def test_soft_topk_mask_sums_to_k(K, tau):
    """Plötz–Roth's defining property: K units of selection spread over n positions, at any τ.

    The mask accumulates K softmax vectors, each summing to 1, so the total is K by construction
    and independent of temperature. Per-*entry* boundedness is not implied and does not hold at
    high τ — see the next test.
    """
    m = soft_topk_mask(_logits(), K, tau=tau)

    assert torch.allclose(m.sum(dim=-1), torch.full((B,), float(K)), atol=1e-5)
    assert (m >= 0).all()


def test_a_mask_entry_can_exceed_one_while_the_selection_is_still_soft():
    """Only as τ → 0 is the mask a selection; at high τ it is K overlapping soft votes.

    Worth pinning because "mask" invites reading it as a per-entry gate in [0, 1], and downstream
    code must not assume that: at τ=1 a dominant position collects well over 1. What keeps the
    result on the simplex is the renormalisation after ``w_soft * m``, not a bound here.
    """
    lg = _logits()

    assert float(soft_topk_mask(lg, 3, tau=1.0).max()) > 1.0
    assert float(soft_topk_mask(lg, 3, tau=0.01).max()) == pytest.approx(1.0, abs=1e-4)


def test_soft_topk_mask_hardens_as_tau_falls():
    """High τ is a smooth mask that keeps gradient on every element; low τ is nearly K-hot."""
    lg = _logits()

    soft = soft_topk_mask(lg, 3, tau=25.0)
    hard = soft_topk_mask(lg, 3, tau=0.01)

    # "K-hot" = three entries at ~1 and the rest at ~0, so the sum of squares approaches K.
    assert float((hard**2).sum(dim=-1).mean()) > float((soft**2).sum(dim=-1).mean())
    assert float((hard**2).sum(dim=-1).mean()) == pytest.approx(3.0, abs=0.05)


def test_soft_topk_mask_never_picks_the_same_position_twice():
    """The log(1−p) shift is what stops the iteration re-selecting its own winner."""
    lg = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    m = soft_topk_mask(lg, 3, tau=0.01)

    assert m[0, 0] == pytest.approx(1.0, abs=1e-4), "the dominant logit is selected exactly once"
    assert m.sum() == pytest.approx(3.0, abs=1e-4)


def test_force_select_reserves_its_slots_without_consuming_all_of_them():
    """Locked positions are pre-seeded, not logit-boosted — boosting would pick them K times."""
    lg = _logits()
    locked = _mask(5, 6)

    m = soft_topk_mask(lg, 4, tau=0.01, force_select=locked)

    assert m[:, 5].min() == pytest.approx(1.0) and m[:, 6].min() == pytest.approx(1.0)
    assert torch.allclose(m.sum(dim=-1), torch.full((B,), 4.0), atol=1e-4)


def test_soft_topk_mask_keeps_gradient_on_every_element():
    """The point of the *soft* mask: the search can still change its mind about which K to keep.

    Scored against a fixed weight vector rather than summed — the mask's own sum is K by
    construction, so its gradient is zero for reasons that say nothing about the mask.
    """
    lg = _logits().requires_grad_(True)
    score = torch.linspace(1.0, 2.0, N)

    (soft_topk_mask(lg, 3, tau=5.0) * score).sum().backward()

    assert lg.grad is not None and (lg.grad.abs() > 0).all()


@pytest.mark.parametrize("K", [1, 3])
def test_hard_topk_project_keeps_exactly_k_and_renormalises(K):
    w = torch.softmax(_logits(), dim=-1)

    out = hard_topk_project(w, K)

    assert ((out > 0).sum(dim=-1) == K).all()
    assert torch.allclose(out.sum(dim=-1), torch.ones(B), atol=1e-6)


def test_hard_topk_project_always_keeps_the_locked_positions():
    """Locked positions must survive the projection — the lock-paste has to have a place to write."""
    w = torch.softmax(_logits(), dim=-1)
    locked = _mask(7)  # deliberately not among the largest

    out = hard_topk_project(w, 3, locked_mask=locked)

    assert (out[:, 7] > 0).all()
    assert ((out > 0).sum(dim=-1) == 3).all()


# --- the projector ------------------------------------------------------------------------------


def test_every_projected_row_is_on_the_simplex():
    """The invariant everything else is built on: non-negative and summing to one."""
    project = SimplexProjector(n_components=N)

    w = project.w_from_logits(_logits(), tau=1.0)

    assert (w >= 0).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-6)


def test_disallowed_elements_get_no_weight():
    project = SimplexProjector(n_components=N, elem_mask=_mask(0, 1, 2, 3))

    w = project.w_from_logits(_logits(), tau=1.0)

    assert (w[:, 4:] == 0).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-6)


def test_pinned_elements_hold_their_value_and_the_row_still_sums_to_one():
    """Zeroing a logit's gradient does NOT pin its weight — softmax renormalises around it.

    That is why the lock is a paste over the softmax output rather than a gradient mask, and why
    the unlocked positions are rescaled to fill exactly the leftover 1 − Σ locked.
    """
    locked, locked_w0 = _mask(1, 3), torch.zeros(B, N)
    locked_w0[:, 1], locked_w0[:, 3] = 0.25, 0.15
    project = SimplexProjector(n_components=N, locked_mask=locked, locked_w0=locked_w0)

    w = project.w_from_logits(_logits(), tau=1.0)

    assert torch.allclose(w[:, 1], torch.full((B,), 0.25), atol=1e-6)
    assert torch.allclose(w[:, 3], torch.full((B,), 0.15), atol=1e-6)
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-6)


def test_the_floor_drops_trace_amounts_and_redistributes_their_mass():
    project = SimplexProjector(n_components=N, min_nonzero_weight=0.10)

    w = project.w_from_logits(_logits(), tau=1.0)

    assert not ((w > 0) & (w < 0.10)).any(), "nothing may survive between zero and the floor"
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-6)


def test_the_floor_is_skipped_for_a_row_it_would_empty():
    """A floor that would zero every unlocked position leaves that row alone.

    Dropping them all would break the simplex invariant, which every downstream step assumes; the
    "at most K" guarantee survives either way.
    """
    project = SimplexProjector(n_components=N, min_nonzero_weight=0.99)

    w = project.w_from_logits(torch.zeros(B, N), tau=1.0)  # uniform row: every entry is 1/8

    assert torch.allclose(w, torch.full((B, N), 1.0 / N), atol=1e-6)


def test_a_pinned_element_is_exempt_from_the_floor():
    """Pins are user-set values; the floor must not silently overrule one."""
    locked, locked_w0 = _mask(2), torch.zeros(B, N)
    locked_w0[:, 2] = 0.01  # below the floor on purpose
    project = SimplexProjector(n_components=N, locked_mask=locked, locked_w0=locked_w0, min_nonzero_weight=0.10)

    w = project.w_from_logits(_logits(), tau=1.0)

    assert torch.allclose(w[:, 2], torch.full((B,), 0.01), atol=1e-6)


def test_the_cardinality_limit_only_bites_when_k_is_below_n():
    assert not SimplexProjector(n_components=N, max_elements=N).topk_active
    assert SimplexProjector(n_components=N, max_elements=N - 1).topk_active
    assert not SimplexProjector(n_components=N).topk_active


def test_the_final_projection_yields_at_most_k_elements_summing_to_one():
    """What the search reports: a genuine K-element recipe, not a near-K-hot blur."""
    project = SimplexProjector(n_components=N, max_elements=3)

    w = project.hard_project(project.w_from_logits(_logits(), tau=0.01))

    assert ((w > 0).sum(dim=-1) <= 3).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-6)


def test_the_final_projection_re_pastes_locks_and_re_applies_the_floor():
    """The hard top-K redistributes mass, so both have to run again after it."""
    locked, locked_w0 = _mask(6), torch.zeros(B, N)
    locked_w0[:, 6] = 0.30
    project = SimplexProjector(
        n_components=N, locked_mask=locked, locked_w0=locked_w0, min_nonzero_weight=0.05, max_elements=3
    )

    w = project.hard_project(project.w_from_logits(_logits(), tau=0.01))

    assert torch.allclose(w[:, 6], torch.full((B,), 0.30), atol=1e-6)
    assert not ((w > 0) & (w < 0.05)).any()
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-6)


def test_the_whole_projection_stays_differentiable():
    """The search descends on logits *through* every constraint — a break here trains nothing."""
    locked, locked_w0 = _mask(0), torch.zeros(B, N)
    locked_w0[:, 0] = 0.2
    project = SimplexProjector(
        n_components=N,
        elem_mask=_mask(0, 1, 2, 3, 4, 5),
        locked_mask=locked,
        locked_w0=locked_w0,
        min_nonzero_weight=0.02,
        max_elements=3,
    )
    lg = _logits().requires_grad_(True)
    # Scored against a fixed weight vector: every row of w sums to exactly 1, so w.sum() is a
    # constant and would report zero gradient no matter how the projection were wired.
    score = torch.linspace(1.0, 2.0, N)

    (project.w_from_logits(lg, tau=5.0) * score).sum().backward()

    assert lg.grad is not None
    assert torch.isfinite(lg.grad).all()
    assert (lg.grad.abs() > 0).any()
