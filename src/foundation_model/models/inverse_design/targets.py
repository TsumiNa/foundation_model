# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""What an inverse-design search optimises *toward*, and what it hands back.

The vocabulary both search spaces share: one target term, its validated tensor form, and the two
result containers. It is the package's leaf — nothing here knows about a model, a simplex or an
optimiser — which is also what lets ``latent`` and ``composition`` depend on it without either
depending on the mixin that hosts them.
"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

# Named tuple for optimization results. ``input_trajectory`` is None unless the caller passes
# ``record_input_trajectory=True`` to :meth:`optimize_latent` (gated because storing it costs
# O(B·R·steps·input_dim) memory and per-step latent-→-input decodes); when present it has shape
# ``(B, R, steps, input_dim)`` — used by the inverse-design trajectory animations to decode the
# per-step composition without rerunning the optimisation.
OptimizationResult = namedtuple(
    "OptimizationResult",
    ["optimized_input", "optimized_target", "initial_score", "trajectory", "input_trajectory"],
    defaults=[None],
)

# Composition-space optimization (gradient descent over element weights w ∈ simplex). The optimised
# w *is* the recipe (no AE-decode round-trip), so it is reported alongside the descriptor x = w @ K.
# ``weights_trajectory`` is None unless the caller passes ``record_weights_trajectory=True`` to
# :meth:`optimize_composition`; when present it has shape ``(steps, B, n_components)``.
CompositionOptimizationResult = namedtuple(
    "CompositionOptimizationResult",
    [
        "optimized_weights",
        "optimized_descriptor",
        "optimized_target",
        "initial_score",
        "trajectory",
        "weights_trajectory",
    ],
    defaults=[None],
)


@dataclass(frozen=True, kw_only=True)
class OptimizationTarget:
    """One user-specified inverse-design objective term.

    The target kind is derived from the task's head type (never guessed from the fields):

    - **regression** — exactly one of ``value`` (minimise ``(ŷ − value)²``) or ``direction``
      (``"high"`` maximises ``ŷ``, ``"low"`` minimises it; unbounded — there is no stationary
      point, so the achieved magnitude scales with ``steps × lr``).
    - **kernel_regression** — ``points`` = a target curve ``[[t, y], ...]``; minimises the MSE of
      the head evaluated at the given ``t`` values against the given ``y`` values.
    - **classification** — ``classes`` = label indices whose combined probability is pushed
      ``"high"`` (default) or ``"low"``. Must be a strict subset of the head's classes ("low" on
      the full set has an empty complement; "high" on the full set is a constant).

    ``weight`` scales this term relative to the others (all kinds; must be > 0).
    """

    task: str
    value: float | None = None
    direction: str | None = None  # "high" | "low"
    points: Sequence[Sequence[float]] | None = None  # [[t, y], ...]
    classes: Sequence[int] | None = None
    weight: float = 1.0


@dataclass(kw_only=True)
class _PreparedTarget:
    """Validated, tensor-ready form of one :class:`OptimizationTarget` (internal)."""

    task: str
    kind: str  # "value" | "direction" | "curve" | "class"
    weight: float
    value: torch.Tensor | None = None  # 0-d, value kind
    sign: float = 0.0  # direction kind: -1.0 high (maximize), +1.0 low
    t: torch.Tensor | None = None  # (K,), curve kind
    y: torch.Tensor | None = None  # (K,), curve kind
    classes: torch.Tensor | None = None  # (C_sel,) long, class kind
    complement: torch.Tensor | None = None  # (C_rest,) long, class kind ("low" objective)
    class_high: bool = True
    class_indices: list[int] = field(default_factory=list)


def _reduce_pred(pred: torch.Tensor) -> torch.Tensor:
    """Reduce a head prediction to one scalar per batch row: mean over all non-batch dims."""
    if pred.ndim == 1:
        return pred
    return pred.mean(dim=tuple(range(1, pred.ndim)))


def targets_from_mappings(
    task_targets: Mapping[str, torch.Tensor | float] | None,
    class_targets: Mapping[str, int | Sequence[int]] | None,
) -> list[OptimizationTarget]:
    """The ``task_targets`` / ``class_targets`` keyword styles as one target list.

    Both searches accept the same two mappings and turned them into targets with the same twenty
    lines. What they do *around* this differs legitimately — ``optimize_latent`` also carries the
    legacy ``task_name`` + ``mode`` / ``target_value`` path, so its mutual-exclusion rule has a
    third kwarg in it — so the shared part is what moves here and each caller keeps its own policy.
    """
    if task_targets is not None and (not isinstance(task_targets, Mapping) or len(task_targets) == 0):
        raise ValueError("task_targets must be a non-empty mapping of task_name -> target_value")
    if class_targets is not None and (not isinstance(class_targets, Mapping) or len(class_targets) == 0):
        raise ValueError("class_targets must be a non-empty mapping of task_name -> class index/indices")
    resolved = [
        OptimizationTarget(task=name, value=float(torch.as_tensor(val).reshape(-1)[0]))
        for name, val in (task_targets or {}).items()
    ]
    resolved += [
        OptimizationTarget(task=name, classes=[int(cls)] if isinstance(cls, int) else [int(c) for c in cls])
        for name, cls in (class_targets or {}).items()
    ]
    return resolved
