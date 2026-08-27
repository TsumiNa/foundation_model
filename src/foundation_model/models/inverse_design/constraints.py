# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""What a composition search is allowed to produce.

Five knobs, and every one of them can contradict another:

* ``allowed_elements`` — a hard whitelist of element symbols;
* ``element_step_scale`` — a per-element learning-rate multiplier, where **0 means pinned** at the
  seed's value;
* ``fixed_amounts`` — elements pinned at explicit absolute amounts;
* ``min_nonzero_weight`` — a floor below which a weight is dropped rather than reported as a trace;
* ``max_elements`` — a cardinality limit, with :mod:`.annealing` deciding how fast it commits.

Pinning an element the whitelist excludes, flooring above ``1 / max_elements``, pinning the same
element by both mechanisms, locking more elements than the cardinality limit allows — each is a
contradiction that produces a plausible-looking recipe rather than an error if it is not caught.
So they are all caught here, once, before the search touches the model, and the result is frozen.

Two things cannot be settled at that point: an ``element_step_scale = 0`` lock pins an element at
its *seed* value, and the seed's rows only exist once the batch does. :meth:`resolve_locks` is the
second half, and it re-runs the two checks whose inputs were per-row.

This was ~340 lines at the head of a 975-line method, and the search's first real statement came
after all of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .annealing import AnnealingSchedule
from .simplex import SimplexProjector


@dataclass(frozen=True)
class CompositionConstraints:
    """Every constraint on one search, validated and resolved into tensors.

    ``elem_mask`` / ``step_scale`` / ``fixed_*`` are ``(n_components,)`` and start on the CPU;
    :meth:`on` returns the same constraints with them on the search's device.
    """

    n_components: int
    elem_mask: torch.Tensor | None = None  # bool — allowed elements; None = no whitelist
    step_scale: torch.Tensor | None = None  # float — per-element grad multiplier; None = all 1.0
    fixed_w0: torch.Tensor | None = None  # float — pinned amounts, 0 elsewhere
    fixed_mask: torch.Tensor | None = None  # bool — which elements are pinned by fixed_amounts
    min_nonzero_weight: float = 0.0
    max_elements: int | None = None
    schedule: AnnealingSchedule = field(default_factory=AnnealingSchedule.disabled)

    # -- construction ----------------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        n_components: int,
        allowed_elements: str | list[str] | Sequence[str] = "all",
        element_step_scale: float | Mapping[str, float] = 1.0,
        fixed_amounts: Mapping[str, float] | None = None,
        min_nonzero_weight: float = 0.0,
        max_elements: int | None = None,
        annealing_scale: float = 0.5,
        annealing_schedule: Mapping[str, Any] | None = None,
        steps: int = 300,
    ) -> CompositionConstraints:
        """Validate the caller's constraint keywords and freeze them.

        Symbol-based inputs are resolved against the bundled ``DEFAULT_ELEMENTS`` registry, which
        is why they require the kernel to align with it.
        """
        from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS  # local import; small list

        elem_mask = cls._build_allowed_mask(allowed_elements, n_components, DEFAULT_ELEMENTS)
        step_scale = cls._build_step_scale(element_step_scale, n_components, DEFAULT_ELEMENTS)
        fixed_w0, fixed_mask = cls._build_fixed_amounts(
            fixed_amounts, n_components, DEFAULT_ELEMENTS, elem_mask, step_scale
        )
        cls._check_floor(min_nonzero_weight, max_elements, fixed_amounts)
        schedule = cls._check_cardinality(
            max_elements=max_elements,
            n_components=n_components,
            elem_mask=elem_mask,
            step_scale=step_scale,
            fixed_mask=fixed_mask,
            annealing_scale=annealing_scale,
            annealing_schedule=annealing_schedule,
            steps=steps,
        )
        return cls(
            n_components=n_components,
            elem_mask=elem_mask,
            step_scale=step_scale,
            fixed_w0=fixed_w0,
            fixed_mask=fixed_mask,
            min_nonzero_weight=min_nonzero_weight,
            max_elements=max_elements,
            schedule=schedule,
        )

    @staticmethod
    def _build_allowed_mask(
        allowed_elements: str | list[str] | Sequence[str], n_components: int, registry: Sequence[str]
    ) -> torch.Tensor | None:
        if isinstance(allowed_elements, str):
            if allowed_elements != "all":
                raise ValueError(f"allowed_elements as a string must be 'all'; got {allowed_elements!r}.")
            return None  # "all": no constraint
        if isinstance(allowed_elements, (list, tuple)):
            if len(allowed_elements) == 0:
                raise ValueError("allowed_elements list must be non-empty.")
            sym_to_idx = {s: i for i, s in enumerate(registry)}
            bad = [s for s in allowed_elements if s not in sym_to_idx]
            if bad:
                raise ValueError(f"Unknown element symbol(s) in allowed_elements: {bad}.")
            if n_components != len(registry):
                raise ValueError(
                    f"allowed_elements as element symbols requires the kernel to align with "
                    f"DEFAULT_ELEMENTS (n_components={n_components}, expected {len(registry)})."
                )
            mask = torch.zeros(n_components, dtype=torch.bool)
            for sym in allowed_elements:
                mask[sym_to_idx[sym]] = True
            return mask
        raise TypeError(
            f"allowed_elements must be 'all' or a non-empty list of element symbols; "
            f"got {type(allowed_elements).__name__}."
        )

    @staticmethod
    def _build_step_scale(
        element_step_scale: float | Mapping[str, float], n_components: int, registry: Sequence[str]
    ) -> torch.Tensor | None:
        if isinstance(element_step_scale, (int, float)) and not isinstance(element_step_scale, bool):
            if element_step_scale < 0:
                raise ValueError(f"element_step_scale must be >= 0; got {element_step_scale}.")
            if float(element_step_scale) == 1.0:
                return None  # 1.0 means "no scaling" — keep the fast path
            return torch.full((n_components,), float(element_step_scale))
        if isinstance(element_step_scale, Mapping):
            sym_to_idx = {s: i for i, s in enumerate(registry)}
            bad = [s for s in element_step_scale if s not in sym_to_idx]
            if bad:
                raise ValueError(f"Unknown element symbol(s) in element_step_scale: {bad}.")
            if any(float(v) < 0 for v in element_step_scale.values()):
                raise ValueError("element_step_scale values must be >= 0.")
            if n_components != len(registry):
                raise ValueError(
                    f"element_step_scale as a symbol dict requires the kernel to align with "
                    f"DEFAULT_ELEMENTS (n_components={n_components}, expected {len(registry)})."
                )
            scale = torch.ones(n_components)
            for sym, val in element_step_scale.items():
                scale[sym_to_idx[sym]] = float(val)
            return scale
        raise TypeError(
            f"element_step_scale must be a non-negative float or a mapping of "
            f"element_symbol → float; got {type(element_step_scale).__name__}."
        )

    @staticmethod
    def _build_fixed_amounts(
        fixed_amounts: Mapping[str, float] | None,
        n_components: int,
        registry: Sequence[str],
        elem_mask: torch.Tensor | None,
        step_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if fixed_amounts is None:
            return None, None
        if not isinstance(fixed_amounts, Mapping):
            raise TypeError(
                f"fixed_amounts must be a mapping of element_symbol → float or None; "
                f"got {type(fixed_amounts).__name__}."
            )
        if len(fixed_amounts) == 0:
            raise ValueError("fixed_amounts must be non-empty when provided.")
        sym_to_idx = {s: i for i, s in enumerate(registry)}
        bad_syms = [s for s in fixed_amounts if s not in sym_to_idx]
        if bad_syms:
            raise ValueError(f"Unknown element symbol(s) in fixed_amounts: {bad_syms}.")
        if n_components != len(registry):
            raise ValueError(
                f"fixed_amounts requires the kernel to align with DEFAULT_ELEMENTS "
                f"(n_components={n_components}, expected {len(registry)})."
            )
        for sym, amt in fixed_amounts.items():
            if not 0.0 < float(amt) < 1.0:
                raise ValueError(f"fixed_amounts['{sym}']={amt} must be strictly between 0 and 1.")
        total = float(sum(fixed_amounts.values()))
        if total >= 1.0:
            raise ValueError(
                f"sum(fixed_amounts.values())={total:.4f} must be strictly less than 1.0 "
                "(the optimiser needs unfixed mass to allocate)."
            )
        # Allowed-list compatibility — pinning outside the whitelist is contradictory.
        if elem_mask is not None:
            bad_against_allowed = [s for s in fixed_amounts if not elem_mask[sym_to_idx[s]]]
            if bad_against_allowed:
                raise ValueError(
                    f"fixed_amounts symbols {bad_against_allowed} are not in allowed_elements — "
                    "pinning a disallowed element is contradictory."
                )
        # Mutual exclusion with element_step_scale = 0 (the other hard-lock path).
        if step_scale is not None:
            overlap = [s for s in fixed_amounts if float(step_scale[sym_to_idx[s]]) == 0.0]
            if overlap:
                raise ValueError(
                    f"Symbols {overlap} appear in both element_step_scale=0 and "
                    "fixed_amounts. Use one mechanism per element."
                )
        fixed_w0 = torch.zeros(n_components)
        fixed_mask = torch.zeros(n_components, dtype=torch.bool)
        for sym, amt in fixed_amounts.items():
            idx = sym_to_idx[sym]
            fixed_w0[idx] = float(amt)
            fixed_mask[idx] = True
        return fixed_w0, fixed_mask

    @staticmethod
    def _check_floor(
        min_nonzero_weight: float, max_elements: int | None, fixed_amounts: Mapping[str, float] | None
    ) -> None:
        if not 0.0 <= min_nonzero_weight <= 1.0:
            raise ValueError(f"min_nonzero_weight must be in [0, 1]; got {min_nonzero_weight}.")
        if min_nonzero_weight <= 0.0:
            return
        # If max_elements is set, the floor must be feasible: K elements ≥ floor summing to 1
        # implies K * floor ≤ 1.
        if max_elements is not None and min_nonzero_weight > 1.0 / max_elements:
            raise ValueError(
                f"min_nonzero_weight={min_nonzero_weight} exceeds 1 / max_elements="
                f"{1.0 / max_elements:.4f}. With at most {max_elements} non-zero positions, "
                "no row can have every weight ≥ floor and still sum to 1."
            )
        # Fixed amounts must themselves be ≥ the floor (else contradiction).
        if fixed_amounts is not None:
            bad_pins = sorted((s, float(v)) for s, v in fixed_amounts.items() if float(v) < min_nonzero_weight)
            if bad_pins:
                raise ValueError(
                    f"fixed_amounts entries {bad_pins} are below min_nonzero_weight="
                    f"{min_nonzero_weight}. The floor cannot override an explicit pin."
                )

    @staticmethod
    def _check_cardinality(
        *,
        max_elements: int | None,
        n_components: int,
        elem_mask: torch.Tensor | None,
        step_scale: torch.Tensor | None,
        fixed_mask: torch.Tensor | None,
        annealing_scale: float,
        annealing_schedule: Mapping[str, Any] | None,
        steps: int,
    ) -> AnnealingSchedule:
        if max_elements is None:
            return AnnealingSchedule.disabled()
        if not isinstance(max_elements, int) or isinstance(max_elements, bool):
            raise TypeError(f"max_elements must be an int or None; got {type(max_elements).__name__}.")
        if not 1 <= max_elements <= n_components:
            raise ValueError(f"max_elements must be in [1, n_components={n_components}]; got {max_elements}.")
        if elem_mask is not None:
            n_allowed = int(elem_mask.sum().item())
            if max_elements > n_allowed:
                raise ValueError(
                    f"max_elements={max_elements} exceeds the number of allowed elements "
                    f"({n_allowed}). Widen ``allowed_elements`` or lower ``max_elements``."
                )
        # Lock-vs-K check: locked positions (element_step_scale=0 ∪ fixed_amounts) all count toward
        # K. We require *strict* ``max_elements > n_locked`` for both lock paths: equality leaves
        # the lock-paste with no unlocked slot to absorb the leftover mass (1 − Σ locked) and
        # produces rows that sum to < 1 — silently breaking the simplex. For ``fixed_amounts`` this
        # is definite (``Σ < 1`` enforced at kwarg time); for ``element_step_scale=0`` the seed
        # values *could* sum to exactly 1, but K-constrained all-locked recipes have no degrees of
        # freedom anyway, so rejecting equality is both safe and clearer.
        n_locked_pre = 0
        if step_scale is not None:
            n_locked_pre += int((step_scale == 0).sum().item())
        if fixed_mask is not None:
            n_locked_pre += int(fixed_mask.sum().item())
        if n_locked_pre >= max_elements:
            raise ValueError(
                f"max_elements={max_elements} must be > total locked elements ({n_locked_pre}, "
                "counting element_step_scale=0 ∪ fixed_amounts) — the lock-paste needs at "
                "least one unlocked slot to absorb the leftover mass (1 − Σ locked); equality "
                "would silently produce row sums < 1. Raise max_elements or unlock some."
            )
        return AnnealingSchedule.build(scale=annealing_scale, schedule=annealing_schedule, steps=steps)

    # -- use -------------------------------------------------------------------------------------

    def on(self, *, device: torch.device, dtype: torch.dtype) -> CompositionConstraints:
        """The same constraints with every tensor on the search's device and precision."""
        return CompositionConstraints(
            n_components=self.n_components,
            elem_mask=self.elem_mask.to(device=device) if self.elem_mask is not None else None,
            step_scale=self.step_scale.to(device=device, dtype=dtype) if self.step_scale is not None else None,
            fixed_w0=self.fixed_w0.to(device=device, dtype=dtype) if self.fixed_w0 is not None else None,
            fixed_mask=self.fixed_mask.to(device=device) if self.fixed_mask is not None else None,
            min_nonzero_weight=self.min_nonzero_weight,
            max_elements=self.max_elements,
            schedule=self.schedule,
        )

    def resolve_locks(
        self, *, w0_seed: torch.Tensor | None, batch_size: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Both hard-lock sources as one ``(mask, values)`` pair, so the projector sees only one.

        Why a paste rather than a gradient mask: zeroing ``logit_i.grad`` keeps that logit constant
        but does NOT keep ``w_i`` constant — softmax renormalises across all logits, so when other
        logits move, the denominator changes and so does the "locked" weight. The fix is to capture
        the per-row target weights and paste them back over the softmax output, renormalising the
        unlocked positions into the remaining ``1 − Σ locked`` mass. Gradient through a locked index
        is then automatically zero, so ``step_scale`` zeroing is no longer needed for them — it
        stays active for the genuinely soft case ``0 < step_scale < 1``.

        The two runtime checks here could not run at build time: a ``element_step_scale = 0`` lock
        pins at *seed* values, which only exist once there is a batch.
        """
        locked_mask: torch.Tensor | None = None
        locked_w0: torch.Tensor | None = None

        # 1. element_step_scale = 0 — pins the listed elements at their (un-blended) seed values.
        if self.step_scale is not None:
            locked_idx_mask = self.step_scale == 0
            if locked_idx_mask.any():
                if w0_seed is None:
                    raise ValueError(
                        "element_step_scale = 0 (hard lock) requires initial_weights — there's no "
                        "per-row seed to lock to when initial_weights=None."
                    )
                if self.elem_mask is not None and (~self.elem_mask[locked_idx_mask]).any():
                    raise ValueError(
                        "Locked elements (element_step_scale = 0) must also be in allowed_elements; "
                        "locking a disallowed element is contradictory."
                    )
                locked_mask = locked_idx_mask
                # (B, n_components): seed values at locked positions, 0 elsewhere — constant.
                locked_w0 = (w0_seed * locked_mask.to(dtype)).detach()

        # 2. fixed_amounts — pins at user-given absolute amounts, no seed required. Validated
        #    disjoint from path 1 at build time, so the masks can simply be OR'd and the values
        #    added.
        if self.fixed_mask is not None:
            assert self.fixed_w0 is not None  # built together with fixed_mask
            fixed_w0_batch = self.fixed_w0.unsqueeze(0).expand(batch_size, -1).detach()
            if locked_mask is None:
                locked_mask = self.fixed_mask
                locked_w0 = fixed_w0_batch
            else:
                assert locked_w0 is not None  # set alongside locked_mask above
                locked_mask = locked_mask | self.fixed_mask
                locked_w0 = locked_w0 + fixed_w0_batch

        # Combined lock sum must leave room (or fit exactly) for the simplex. fixed_amounts
        # enforces Σ < 1 at build time and element_step_scale=0 locks at seed values summing to
        # ≤ 1 per row — but the *combined* total could exceed 1 (e.g. seed-lock Mg=0.50 plus fix
        # Au=0.65). Tiny tolerance for float noise.
        if locked_w0 is not None:
            lock_sums = locked_w0.sum(dim=-1)
            if (lock_sums > 1.0 + 1e-5).any():
                raise ValueError(
                    f"Combined locked mass exceeds 1.0 on at least one row "
                    f"(max row-sum = {float(lock_sums.max()):.4f}). Likely cause: "
                    "``element_step_scale=0`` locks plus ``fixed_amounts`` together claim more "
                    "than 100% of the simplex. Lower one set of values or drop a lock."
                )

        # Floored elements must not contradict the lock-paste targets. fixed_amounts was checked at
        # build time; element_step_scale=0 locks have per-row seed values we could not see then.
        if self.min_nonzero_weight > 0.0 and locked_mask is not None and locked_w0 is not None:
            locked_below_floor = (locked_w0 > 0) & (locked_w0 < self.min_nonzero_weight)
            if locked_below_floor.any():
                raise ValueError(
                    f"At least one locked element's value falls below min_nonzero_weight="
                    f"{self.min_nonzero_weight}. Likely cause: an element_step_scale=0 lock points "
                    "at a seed value below the floor (raise the seed, lower the floor, or "
                    "drop the lock)."
                )
        return locked_mask, locked_w0

    def projector(self, locked_mask: torch.Tensor | None, locked_w0: torch.Tensor | None) -> SimplexProjector:
        """The logits → recipe mapping these constraints imply."""
        return SimplexProjector(
            n_components=self.n_components,
            elem_mask=self.elem_mask,
            locked_mask=locked_mask,
            locked_w0=locked_w0,
            min_nonzero_weight=self.min_nonzero_weight,
            max_elements=self.max_elements,
        )
