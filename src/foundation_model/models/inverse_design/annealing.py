# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""How the cardinality constraint hardens over the course of a composition search.

``max_elements`` is enforced by a *soft* top-K mask (see :mod:`.simplex`), whose sharpness is a
temperature τ. Early in the search τ is high and the mask is smooth, so gradient reaches every
element and the optimiser can still change its mind about which K to keep; by the end τ is small
and the mask is nearly K-hot, so what comes out is a real K-element recipe rather than a blur.

This module owns the τ curve and nothing else. It is separate from :mod:`.simplex` because the
two answer different questions and are tuned by different people: *how* a logit vector becomes a
legal recipe is the mechanism, and lives there; *how fast* the selection commits is a policy the
caller sets with ``annealing_scale`` / ``annealing_schedule``, and lives here. Someone adjusting
the schedule should not have to read the Plötz–Roth iteration to do it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Numerical lower bound; below this softmax(lg/τ) loses precision.
TAU_FLOOR = 1e-3
# Fixed final hardness for the default schedule's tail, and the τ used for the final readout.
TAU_END = 0.01
# τ = _SCALE_TAU_BASE**scale → scale 0 → τ=1, 0.5 → τ=5, 1 → τ=25.
_SCALE_TAU_BASE = 25.0

_ALLOWED_FUNCS = ("geometric", "linear", "cosine", "constant")


def scale_to_tau(scale: float) -> float:
    """The user-facing normalised hardness knob ∈ [0, 1] as a raw temperature."""
    return float(_SCALE_TAU_BASE ** max(0.0, min(1.0, scale)))


def interp_scalar(a: float, b: float, t: float, func: str) -> float:
    """Interpolate from ``a`` to ``b`` at local-time ``t`` ∈ [0, 1]."""
    if func == "constant":
        return a
    if func == "linear":
        return a + (b - a) * t
    if func == "cosine":
        return b + 0.5 * (a - b) * (1.0 + math.cos(math.pi * t))
    # geometric — guard against zero/sign issues by working in log space when both >0.
    if a > 0.0 and b > 0.0:
        return a * (b / a) ** t
    # Fall back to linear for degenerate cases (shouldn't trigger in normal use).
    return a + (b - a) * t


@dataclass(frozen=True)
class AnnealingSchedule:
    """The τ curve for one search, validated once at construction.

    Shape:

    * Normalised scale ∈ [0, 1] is the user-facing knob; raw τ is ``25**scale`` (0 → 1, 0.5 → 5,
      1 → 25).
    * With no ``annealing_schedule`` dict: geometric from ``τ_start = 25**scale`` at fractional
      step 0 down to :data:`TAU_END` at fractional step 1.
    * With a dict: its segments override the front of the curve, and the segment from
      ``step[-1]`` to 1.0 (when the dict does not already end at 1.0) falls back to the geometric
      tail from ``25**scale[-1]`` down to :data:`TAU_END`.

    ``active=False`` is the no-cardinality-constraint case: there is no soft mask to harden, so
    every step reports the final τ.
    """

    steps: int
    scale: float = 0.5
    seg_steps: tuple[float, ...] = ()
    seg_scales: tuple[float, ...] = ()
    seg_funcs: tuple[str, ...] = ()
    active: bool = True

    @classmethod
    def disabled(cls) -> AnnealingSchedule:
        """The schedule for a search with no ``max_elements`` — a constant final τ.

        ``annealing_scale`` / ``annealing_schedule`` are deliberately NOT validated in this case,
        matching the long-standing behaviour: they describe how a constraint hardens, and with no
        constraint present rejecting them would fail a search over an argument it never reads.
        """
        return cls(steps=0, active=False)

    @classmethod
    def build(cls, *, scale: float, schedule: Mapping[str, Any] | None, steps: int) -> AnnealingSchedule:
        """Validate the caller's knobs and freeze them into a schedule."""
        if not 0.0 <= scale <= 1.0:
            raise ValueError(f"annealing_scale must be in [0, 1]; got {scale}.")
        if schedule is None:
            return cls(steps=steps, scale=scale)

        if not isinstance(schedule, Mapping):
            raise TypeError(f"annealing_schedule must be a mapping; got {type(schedule).__name__}.")
        missing = {"step", "scale", "annealing_func"} - set(schedule)
        if missing:
            raise ValueError(
                f"annealing_schedule missing required keys {sorted(missing)}. "
                "Required: step, scale, annealing_func — all parallel lists."
            )
        sched_steps = list(schedule["step"])
        sched_scales = list(schedule["scale"])
        sched_funcs = list(schedule["annealing_func"])
        if not (len(sched_steps) == len(sched_scales) == len(sched_funcs)):
            raise ValueError(
                f"annealing_schedule lists must be the same length; got "
                f"step={len(sched_steps)}, scale={len(sched_scales)}, "
                f"annealing_func={len(sched_funcs)}."
            )
        if len(sched_steps) == 0:
            raise ValueError("annealing_schedule lists must be non-empty.")
        prev_s = 0.0
        for s in sched_steps:
            if not 0.0 < float(s) <= 1.0:
                raise ValueError(f"annealing_schedule['step'] entries must be in (0, 1]; got {s}.")
            if float(s) <= prev_s:
                raise ValueError(f"annealing_schedule['step'] must be strictly increasing; got {sched_steps}.")
            prev_s = float(s)
        for t in sched_scales:
            if not 0.0 <= float(t) <= 1.0:
                raise ValueError(f"annealing_schedule['scale'] entries must be in [0, 1]; got {t}.")
        for f in sched_funcs:
            if f not in _ALLOWED_FUNCS:
                raise ValueError(
                    f"annealing_schedule['annealing_func'] entries must be one of {_ALLOWED_FUNCS}; got {f!r}."
                )

        return cls(
            steps=steps,
            scale=scale,
            seg_steps=tuple(float(s) for s in sched_steps),
            seg_scales=tuple(float(t) for t in sched_scales),
            seg_funcs=tuple(sched_funcs),
        )

    @property
    def final_tau(self) -> float:
        """The hardest τ — used for the final readout, and for every step when inactive."""
        return float(max(TAU_END, TAU_FLOOR))

    def tau_for_step(self, step: int) -> float:
        """The raw τ for integer optimisation step ``step``."""
        if not self.active or self.steps <= 1:
            return self.final_tau
        # Fractional progress in [0, 1].
        s = step / (self.steps - 1)
        # Default schedule (used directly when no dict, or for the tail when the dict ends < 1.0).
        default_tau_start = scale_to_tau(self.scale)
        default_tau_end = TAU_END

        if self.seg_steps:
            # Walk through dict segments to find the one containing ``s``.
            prev_step = 0.0
            prev_scale = self.scale  # segment 0 starts at the simple knob's value
            for i, seg_end in enumerate(self.seg_steps):
                if s <= seg_end:
                    local_t = (s - prev_step) / max(seg_end - prev_step, 1e-12)
                    scale_now = interp_scalar(prev_scale, self.seg_scales[i], local_t, self.seg_funcs[i])
                    return float(max(scale_to_tau(scale_now), TAU_FLOOR))
                prev_step = seg_end
                prev_scale = self.seg_scales[i]
            # ``s`` is past the dict's last step → use the geometric tail from
            # ``25**scale[-1]`` at ``step[-1]`` down to ``TAU_END`` at 1.0.
            tail_start_tau = scale_to_tau(self.seg_scales[-1])
            tail_end_step = 1.0
            tail_local_t = (s - self.seg_steps[-1]) / max(tail_end_step - self.seg_steps[-1], 1e-12)
            val = tail_start_tau * (default_tau_end / tail_start_tau) ** tail_local_t
            return float(max(val, TAU_FLOOR))

        # No dict — default geometric schedule from τ_start(scale) to TAU_END.
        val = default_tau_start * (default_tau_end / default_tau_start) ** s
        return float(max(val, TAU_FLOOR))
