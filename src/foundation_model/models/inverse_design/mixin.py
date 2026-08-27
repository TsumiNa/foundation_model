# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""The model-facing surface of inverse design.

:class:`InverseDesignMixin` declares what the search needs from the host model and nothing else:
three attributes and one method. Everything the search does with them lives in the sibling
modules — see this package's ``__init__`` for the map.

WHY A MIXIN RATHER THAN FREE FUNCTIONS

The coupling to the model is genuinely thin — four members (``encoder``, ``task_heads``,
``task_configs_map``, ``_head``) plus ``nn.Module``'s own ``train`` / ``eval`` / ``training`` /
``parameters``.

Free functions taking the model explicitly express that better, and both searches ARE free
functions now: :func:`.latent.optimize_latent` and :func:`.composition.optimize_composition` take
the model as their first argument, and are bound here as class attributes. Python binds a plain
function assigned in a class body as a method, so ``model.optimize_composition(...)`` still works
for every caller and every keyword keeps its type — no wrapper, no duplicated signature.

What the class itself keeps is what genuinely needs the model: the declared contract, and the two
steps that turn a target into a loss term against real heads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.foundation_encoder import FoundationEncoder
from ..model_config import TaskConfigType, TaskType
from ..task_head.autoencoder import AutoEncoderHead
from ..task_head.base import BaseTaskHead
from ..task_head.kernel_regression import expand_for_kernel_regression
from .composition import optimize_composition
from .latent import optimize_latent
from .targets import (
    OptimizationTarget,
    PreparedTarget,
    reduce_pred,
)


class InverseDesignMixin(nn.Module):
    """Target-driven input search over a trained :class:`FlexibleMultiTaskModel`.

    THE INTERFACE THIS NEEDS FROM THE MODEL is the three attributes and one method declared
    below, plus ``train`` / ``eval`` / ``training`` / ``parameters`` — which is why the base is
    ``nn.Module`` rather than ``object``. The search toggles the host in and out of eval mode and
    freezes its parameters while it descends on the *input*, so it genuinely requires a module,
    not a duck-typed bag of tensors.

    Declaring the contract here rather than assuming it means mypy checks it: the first draft of
    this class inherited ``object`` and mypy immediately named all four module members it was
    silently borrowing.
    """

    # Supplied by FlexibleMultiTaskModel.
    encoder: FoundationEncoder
    task_heads: nn.ModuleDict
    task_configs_map: dict[str, TaskConfigType]

    def _head(self, name: str) -> BaseTaskHead:  # pragma: no cover - provided by the host class
        raise NotImplementedError

    def _resolve_target_and_mask(
        self,
        *,
        name: str,
        head: nn.Module,
        x: torch.Tensor,
        y_dict_batch: dict[str, Any],
        task_masks_batch: dict[str, Any],
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor] | None] | None:
        """This head's target and raw mask, or ``None`` when it does not participate in the batch.

        The autoencoder reconstructs the input itself, so it never appears in ``y_dict_batch`` and
        is always active.
        """
        if isinstance(head, AutoEncoderHead):
            return x, torch.ones_like(x, dtype=torch.bool, device=x.device)
        if name not in y_dict_batch or not self.task_configs_map[name].enabled:
            return None
        return y_dict_batch[name], task_masks_batch.get(name)

    def _prepare_optimization_targets(
        self, targets: Sequence[OptimizationTarget], *, device: torch.device, dtype: torch.dtype
    ) -> list[PreparedTarget]:
        """Validate a target list against the model's heads and build the per-term tensors.

        The target *kind* comes from the head type in ``task_configs_map`` — the caller never
        declares it. Raises ``ValueError`` on any field/kind mismatch so config errors surface
        before the optimisation loop starts.
        """
        if not targets:
            raise ValueError("targets must be a non-empty sequence of OptimizationTarget.")
        prepared: list[PreparedTarget] = []
        seen: set[str] = set()
        for spec in targets:
            name = spec.task
            if name in seen:
                raise ValueError(f"Duplicate optimization target for task '{name}'.")
            seen.add(name)
            if name not in self.task_heads:
                raise ValueError(f"Task '{name}' not found in model. Available tasks: {list(self.task_heads.keys())}")
            if spec.weight is None or float(spec.weight) <= 0:
                raise ValueError(f"targets['{name}'].weight must be > 0, got {spec.weight}.")
            weight = float(spec.weight)
            cfg = self.task_configs_map[name]
            if cfg.type == TaskType.REGRESSION:
                if spec.points is not None or spec.classes is not None:
                    raise ValueError(f"Regression target '{name}' accepts value/direction, not points/classes.")
                if (spec.value is None) == (spec.direction is None):
                    raise ValueError(f"Regression target '{name}' needs exactly one of value or direction.")
                if spec.value is not None:
                    prepared.append(
                        PreparedTarget(
                            task=name,
                            kind="value",
                            weight=weight,
                            value=torch.as_tensor(float(spec.value), device=device, dtype=dtype),
                        )
                    )
                else:
                    if spec.direction not in {"high", "low"}:
                        raise ValueError(
                            f"targets['{name}'].direction must be 'high' or 'low', got {spec.direction!r}."
                        )
                    prepared.append(
                        PreparedTarget(
                            task=name,
                            kind="direction",
                            weight=weight,
                            sign=-1.0 if spec.direction == "high" else 1.0,
                        )
                    )
            elif cfg.type == TaskType.KERNEL_REGRESSION:
                if spec.value is not None or spec.direction is not None or spec.classes is not None:
                    raise ValueError(f"Kernel-regression target '{name}' accepts points only.")
                if not spec.points:
                    raise ValueError(
                        f"Kernel-regression target '{name}' needs a non-empty points list of [t, y] pairs."
                    )
                pairs = []
                for p in spec.points:
                    pair = list(p)
                    if len(pair) != 2:
                        raise ValueError(f"targets['{name}'].points entries must be [t, y] pairs, got {p!r}.")
                    pairs.append((float(pair[0]), float(pair[1])))
                prepared.append(
                    PreparedTarget(
                        task=name,
                        kind="curve",
                        weight=weight,
                        t=torch.as_tensor([p[0] for p in pairs], device=device, dtype=dtype),
                        y=torch.as_tensor([p[1] for p in pairs], device=device, dtype=dtype),
                    )
                )
            elif cfg.type == TaskType.CLASSIFICATION:
                if spec.value is not None or spec.points is not None:
                    raise ValueError(f"Classification target '{name}' accepts classes (+ direction) only.")
                if not spec.classes:
                    raise ValueError(f"Classification target '{name}' needs a non-empty classes list.")
                direction = spec.direction if spec.direction is not None else "high"
                if direction not in {"high", "low"}:
                    raise ValueError(f"targets['{name}'].direction must be 'high' or 'low', got {spec.direction!r}.")
                idxs = sorted({int(c) for c in spec.classes})
                num_classes = getattr(cfg, "num_classes", None)
                if num_classes is None:
                    raise ValueError(f"Classification task '{name}' has no num_classes; cannot build a class target.")
                if any(not 0 <= i < num_classes for i in idxs):
                    raise ValueError(
                        f"targets['{name}'].classes {idxs} out of range for a {num_classes}-class head; "
                        f"valid indices are [0, {num_classes})."
                    )
                if len(idxs) >= num_classes:
                    raise ValueError(
                        f"targets['{name}'].classes {idxs} covers every class of a {num_classes}-class head; "
                        "the objective would be constant ('high') or undefined ('low'). Use a strict subset."
                    )
                complement = [i for i in range(num_classes) if i not in set(idxs)]
                prepared.append(
                    PreparedTarget(
                        task=name,
                        kind="class",
                        weight=weight,
                        classes=torch.as_tensor(idxs, device=device, dtype=torch.long),
                        complement=torch.as_tensor(complement, device=device, dtype=torch.long),
                        class_high=direction == "high",
                        class_indices=idxs,
                    )
                )
            else:
                raise ValueError(f"Task '{name}' has unsupported head type {cfg.type} for optimization targets.")
        return prepared

    def _optimization_objective(
        self, h_task: torch.Tensor, prepared: Sequence[PreparedTarget]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate every target term at ``h_task``.

        Returns ``(channels, losses)``, both ``(B, T)`` with one column per target in declaration
        order. ``channels`` is the human-readable per-target scalar (regression: ŷ; curve:
        RMSE-to-curve; class: P(classes)); ``losses`` is the weighted per-sample loss whose sum is
        the optimisation objective (lower = better; direction terms make it sign-indefinite).
        """
        channel_cols: list[torch.Tensor] = []
        loss_cols: list[torch.Tensor] = []
        for tgt in prepared:
            head = self._head(tgt.task)
            if tgt.kind == "curve":
                assert tgt.t is not None and tgt.y is not None
                k = tgt.t.shape[0]
                t_batch = tgt.t.unsqueeze(0).expand(h_task.shape[0], k)
                h_rep, t_rep = expand_for_kernel_regression(h_task, t_batch)
                pred = head(h_rep, t=t_rep).view(h_task.shape[0], k)
                sq = (pred - tgt.y.unsqueeze(0)) ** 2  # (B, K)
                per_sample = sq.mean(dim=1)
                channel_cols.append(per_sample.sqrt())  # RMSE-to-curve; 0 = perfect fit
                loss_cols.append(tgt.weight * per_sample)
            elif tgt.kind == "class":
                assert tgt.classes is not None and tgt.complement is not None
                log_probs = F.log_softmax(head(h_task), dim=-1)
                lp_sel = torch.logsumexp(log_probs.index_select(-1, tgt.classes), dim=-1)  # (B,)
                channel_cols.append(lp_sel.exp())  # P(classes) regardless of direction
                if tgt.class_high:
                    loss_cols.append(tgt.weight * (-lp_sel))
                else:
                    # "low" = maximize the complement's probability: numerically clean near
                    # P(classes) → 1 (a direct -log(1 - P) would blow up) and reuses the same
                    # logsumexp machinery.
                    lp_rest = torch.logsumexp(log_probs.index_select(-1, tgt.complement), dim=-1)
                    loss_cols.append(tgt.weight * (-lp_rest))
            else:
                pred = head(h_task)
                reduced = reduce_pred(pred)
                channel_cols.append(reduced)
                if tgt.kind == "value":
                    assert tgt.value is not None
                    expanded = tgt.value.reshape([1] * pred.ndim).expand(pred.shape)
                    per_sample = (pred - expanded) ** 2
                    per_sample = reduce_pred(per_sample)
                    loss_cols.append(tgt.weight * per_sample)
                else:  # direction
                    loss_cols.append(tgt.weight * tgt.sign * reduced)
        return torch.stack(channel_cols, dim=-1), torch.stack(loss_cols, dim=-1)

    @torch.no_grad()
    def evaluate_targets(
        self, x: torch.Tensor, targets: Sequence[OptimizationTarget]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score descriptors against a target list without optimising.

        Parameters
        ----------
        x : torch.Tensor
            Descriptors, shape ``(B, input_dim)`` (a 1-d tensor is treated as a single row).
        targets : Sequence[OptimizationTarget]
            The objective terms (same specs :meth:`optimize_latent` / :meth:`optimize_composition`
            accept via ``targets=``).

        Returns
        -------
        (channels, objective)
            ``channels`` — ``(B, T)`` per-target scalars (see :meth:`_optimization_objective`);
            ``objective`` — ``(B,)`` summed weighted loss, lower = better (sign-indefinite when
            direction targets are present). This is the exact quantity the optimisers minimise
            (minus the space-specific extras), so seed ranking and optimisation cannot drift.
        """
        ref = next(self.parameters())
        device, dtype = ref.device, ref.dtype
        prepared = self._prepare_optimization_targets(targets, device=device, dtype=dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(device=device, dtype=dtype)
        was_training = self.training
        self.eval()
        try:
            h_task = torch.tanh(self.encoder(x))
            channels, losses = self._optimization_objective(h_task, prepared)
        finally:
            self.train(was_training)
        return channels, losses.sum(dim=1)

    # The two searches, bound as methods. They are written in .latent / .composition as functions
    # taking the model explicitly — which states the dependency rather than assuming it, and is
    # what let them move out of this class — and a plain function assigned in a class body binds
    # as a method, so ``model.optimize_composition(...)`` is unchanged for every caller and every
    # keyword keeps its type. A def that forwarded *args/**kwargs would have erased both.
    optimize_latent = optimize_latent
    optimize_composition = optimize_composition
