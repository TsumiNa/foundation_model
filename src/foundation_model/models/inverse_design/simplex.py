# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""How a vector of logits becomes a legal composition.

The composition search optimises unconstrained logits, but what it has to report is a *recipe*: a
row on the simplex that respects every constraint the caller set. This module is that mapping, and
it is the mathematical content of the search — a soft top-K for the cardinality limit, a paste for
pinned elements, a floor for trace amounts, all differentiable so the optimiser can descend
through it.

Two free functions carry the selection maths (they depend on nothing but their arguments), and
:class:`SimplexProjector` binds one search's constraint tensors once so its loop can call
``w_from_logits(logits, tau)`` and get back something it can hand to the encoder.

τ arrives as an argument rather than from enclosing state. As closures these read a one-element
list — ``current_tau = [...]``, whose own comment explained that it existed "so the loop can mutate
it each step without rebuilding the closure that reads it". A parameter says the same thing
without the box.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def soft_topk_mask(lg: torch.Tensor, K: int, tau: float, *, force_select: torch.Tensor | None = None) -> torch.Tensor:
    """Plötz–Roth iterative softmax. Returns m ∈ [0,1]^(B, n) with Σm = K.

    ``force_select`` (n_components,) bool marks positions that must be in the K selection (e.g.
    hard-locked elements). Instead of boosting those logits — which would make the iterative
    softmax pick them K times in a row, never moving on — we **pre-seed** the mask with 1.0 at
    those positions and run only ``K - n_locked`` iterations on the *unlocked* positions (their
    logits are masked to ``-inf`` inside the iteration so they never compete).
    """
    if force_select is None:
        alpha = lg
        m = torch.zeros_like(lg)
        n_iter = K
    else:
        # Pre-mark locked positions as fully selected; iterate only on the rest.
        n_locked = int(force_select.sum().item())
        n_iter = K - n_locked
        locked_row = force_select.to(lg.dtype).unsqueeze(0).expand_as(lg)
        m = locked_row.clone()
        alpha = lg.masked_fill(force_select, float("-inf"))
    for _ in range(n_iter):
        p = torch.softmax(alpha / tau, dim=-1)
        m = m + p
        # The shift in scaled-logit space at the selected position is ``log(1−p)/τ`` — at small τ
        # this is enormously negative, so the next iteration cannot re-pick the same position.
        # (We must NOT multiply by τ here.)
        alpha = alpha + torch.log((1.0 - p).clamp(min=1e-12))
    return m


def hard_topk_project(w: torch.Tensor, K: int, *, locked_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Hard top-K projection: keep K largest per row, zero rest, renormalise.

    If ``locked_mask`` is set, every locked position is forced into the kept set (so the lock-paste
    still has a place to write its seed values); the remaining ``K − n_locked`` slots are filled by
    the largest unlocked weights.
    """
    if locked_mask is None:
        _, idx = w.topk(K, dim=-1)
        keep = torch.zeros_like(w).scatter_(-1, idx, 1.0)
    else:
        n_locked = int(locked_mask.sum().item())
        n_free = K - n_locked
        locked_row = locked_mask.to(w.dtype).unsqueeze(0).expand_as(w)
        if n_free > 0:
            # Exclude locked positions from the unlocked competition by sending them to ``-inf``
            # before topk; locked positions are added back via ``locked_row``.
            w_for_free = w.masked_fill(locked_mask.unsqueeze(0), float("-inf"))
            _, idx = w_for_free.topk(n_free, dim=-1)
            free_keep = torch.zeros_like(w).scatter_(-1, idx, 1.0)
            keep = (locked_row + free_keep).clamp(max=1.0)
        else:
            keep = locked_row
    w = w * keep
    return w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)


@dataclass(frozen=True)
class SimplexProjector:
    """One search's constraint tensors, bound so the loop can project logits in one call.

    Every field is settled before the search starts (see :class:`~.constraints.CompositionConstraints`),
    which is what makes this immutable and lets the same projector serve the initial scoring, every
    optimisation step and the final readout.
    """

    n_components: int
    elem_mask: torch.Tensor | None = None  # (n,) bool — allowed elements; None = all
    locked_mask: torch.Tensor | None = None  # (n,) bool — pinned positions
    locked_w0: torch.Tensor | None = None  # (B, n) — pinned values, 0 elsewhere
    min_nonzero_weight: float = 0.0
    max_elements: int | None = None

    @property
    def topk_active(self) -> bool:
        """Whether the cardinality limit actually bites (K < n is what makes it a constraint)."""
        return self.max_elements is not None and self.max_elements < self.n_components

    def apply_lock_paste(self, w: torch.Tensor) -> torch.Tensor:
        """Paste locked seed values onto ``w`` and renormalise unlocked positions."""
        if self.locked_mask is None:
            return w
        assert self.locked_w0 is not None  # set alongside locked_mask
        free_mask_f = (~self.locked_mask).to(w.dtype)
        w_unlocked = w * free_mask_f
        free_mass = (1.0 - self.locked_w0.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        w_unlocked = w_unlocked / w_unlocked.sum(dim=-1, keepdim=True).clamp(min=1e-12) * free_mass
        return w_unlocked + self.locked_w0

    def apply_min_floor(self, w: torch.Tensor) -> torch.Tensor:
        """Drop unlocked positions below ``min_nonzero_weight`` and re-fill free mass.

        Locked positions are exempt (their values are user-set). If dropping below-floor positions
        would leave a row with zero unlocked mass, the floor is skipped for that row — preserving
        the simplex invariant. The "at most K" guarantee still holds; some rows may end up with
        fewer than K non-zero positions.
        """
        if self.min_nonzero_weight <= 0.0:
            return w
        if self.locked_mask is not None:
            assert self.locked_w0 is not None  # set alongside locked_mask
            unlocked_f = (~self.locked_mask).to(w.dtype)
            free_mass = (1.0 - self.locked_w0.sum(dim=-1, keepdim=True)).clamp(min=0.0)
            unlocked_bool = (~self.locked_mask).unsqueeze(0).expand_as(w)
        else:
            unlocked_f = torch.ones_like(w[0])
            free_mass = torch.ones(w.shape[0], 1, dtype=w.dtype, device=w.device)
            unlocked_bool = torch.ones_like(w, dtype=torch.bool)
        below = (w > 0) & (w < self.min_nonzero_weight) & unlocked_bool
        if not below.any():
            return w
        w_drop = w.masked_fill(below, 0.0)
        # Per-row unlocked sum after the tentative drop.
        unlocked_after = w_drop * unlocked_f
        unlocked_sum = unlocked_after.sum(dim=-1, keepdim=True)
        # Rows where the drop is safe — at least one unlocked position survives.
        can_drop = unlocked_sum > 1e-12
        # Renormalise unlocked portion to fit the free mass; locked stays as-is.
        safe_sum = unlocked_sum.clamp(min=1e-12)
        if self.locked_mask is not None:
            locked_part = w_drop * self.locked_mask.to(w.dtype)
            w_renorm = locked_part + unlocked_after * (free_mass / safe_sum)
        else:
            w_renorm = w_drop / safe_sum
        return torch.where(can_drop.expand_as(w), w_renorm, w)

    def w_from_logits(self, lg: torch.Tensor, tau: float) -> torch.Tensor:
        """Softmax → optional soft top-K → optional hard-lock paste → optional min-floor."""
        if self.elem_mask is not None:
            lg = lg.masked_fill(~self.elem_mask, float("-inf"))
        w_soft = torch.softmax(lg, dim=-1)
        if self.topk_active:
            assert self.max_elements is not None  # implied by topk_active
            # Force locked positions to always sit in the K-hot mask so the lock-paste has
            # somewhere to write. ``w_soft`` itself is computed from the *unboosted* logits, so the
            # within-K ratios reflect the optimisation state.
            m_topk = soft_topk_mask(lg, self.max_elements, tau, force_select=self.locked_mask)
            w = w_soft * m_topk
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        else:
            w = w_soft
        return self.apply_min_floor(self.apply_lock_paste(w))

    def hard_project(self, w: torch.Tensor) -> torch.Tensor:
        """The final readout's clean-up: hard top-K, then re-paste and re-floor.

        The projection may redistribute mass across unlocked positions, so lock-paste's free-mass
        renormalisation has to be re-run for the row to still sum to exactly 1; and it may promote
        a previously-zeroed below-floor position back in, so the floor is re-applied after it.
        """
        if not self.topk_active:
            return w
        assert self.max_elements is not None  # implied by topk_active
        w = hard_topk_project(w, self.max_elements, locked_mask=self.locked_mask)
        return self.apply_min_floor(self.apply_lock_paste(w))
