# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Inverse design: search for inputs whose predicted properties hit a set of targets.

This is the *use* of a trained model, not part of training it. It arrived after the training
code and grew inside :class:`FlexibleMultiTaskModel` until it was half that class's body — 1536
lines against roughly 500 for the LightningModule surface itself. Nothing in the training path
calls any of it: the only consumers are ``workflows/inverse.py`` and
``workflows/inverse_trajectory.py``.

Two search spaces, and the difference between them is the point:

* :meth:`optimize_latent` descends in the encoder's latent space and decodes back through the
  autoencoder, so what it returns is a *descriptor* that still has to be inverted to a recipe.
* :meth:`optimize_composition` descends directly over element weights on the simplex, so the
  optimised variable IS the recipe and no decode round-trip is involved. This is the KMD
  differentiable path.

WHY A MIXIN RATHER THAN FREE FUNCTIONS
--------------------------------------
The coupling to the model is genuinely thin — 1536 lines reach for exactly six of its
attributes, listed below, plus ``nn.Module``'s own ``train`` / ``eval`` / ``training``. Free
functions taking the model as a first argument would express that better still.

A mixin was chosen for the move itself because it keeps ``model.optimize_composition(...)``
working unchanged across two workflow modules and 186 lines of tests, which makes this a pure
relocation that can be reviewed as one. Going from here to free functions is then mechanical,
and worth doing separately rather than folding a behaviour-visible API change into a
1500-line move.
"""

from __future__ import annotations

import math
from collections import namedtuple
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .components.foundation_encoder import FoundationEncoder
from .model_config import TaskConfigType, TaskType
from .task_head.autoencoder import AutoEncoderHead
from .task_head.base import BaseTaskHead
from .task_head.kernel_regression import expand_for_kernel_regression


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


class InverseDesignMixin(nn.Module):
    """Target-driven input search over a trained :class:`FlexibleMultiTaskModel`.

    THE INTERFACE THIS NEEDS FROM THE MODEL is the three attributes and two methods declared
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
    ) -> list[_PreparedTarget]:
        """Validate a target list against the model's heads and build the per-term tensors.

        The target *kind* comes from the head type in ``task_configs_map`` — the caller never
        declares it. Raises ``ValueError`` on any field/kind mismatch so config errors surface
        before the optimisation loop starts.
        """
        if not targets:
            raise ValueError("targets must be a non-empty sequence of OptimizationTarget.")
        prepared: list[_PreparedTarget] = []
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
                        _PreparedTarget(
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
                        _PreparedTarget(
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
                    _PreparedTarget(
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
                    _PreparedTarget(
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
        self, h_task: torch.Tensor, prepared: Sequence[_PreparedTarget]
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
                reduced = _reduce_pred(pred)
                channel_cols.append(reduced)
                if tgt.kind == "value":
                    assert tgt.value is not None
                    expanded = tgt.value.reshape([1] * pred.ndim).expand(pred.shape)
                    per_sample = (pred - expanded) ** 2
                    per_sample = _reduce_pred(per_sample)
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

    def optimize_latent(
        self,
        task_name: str | None = None,
        initial_input: torch.Tensor | None = None,
        mode: str = "max",
        steps: int = 200,
        lr: float = 0.1,
        num_restarts: int = 1,
        perturbation_std: float = 0.0,
        target_value: torch.Tensor | float | None = None,
        targets: Sequence[OptimizationTarget] | None = None,
        task_targets: Mapping[str, torch.Tensor | float] | None = None,
        class_targets: Mapping[str, int | Sequence[int]] | None = None,
        ae_align_scale: float = 0.5,
        optimize_space: str = "input",
        record_input_trajectory: bool = False,
    ) -> OptimizationResult:
        """
        Optimize inputs to drive one or multiple regression heads toward targets or extremes.

        Two strategies are available via ``optimize_space``:

        - ``"input"`` (default): gradient-descend directly on the input tensor X.
        - ``"latent"``: encode X to the latent space, optimise there, then reconstruct X
          via the built-in reconstruction head (requires ``enable_autoencoder=True`` at
          model construction time).

        Parameters
        ----------
        task_name : str | None
            Regression task to optimise (legacy single-task path). Optional — and ignored — when
            ``task_targets`` or ``class_targets`` is provided; required otherwise.
        initial_input : torch.Tensor
            Seed inputs, shape (B, input_dim). Always required (raises ``ValueError`` if ``None``).
        mode : str, optional
            ``"max"`` or ``"min"``. Ignored when ``target_value`` / ``task_targets`` is set.
        steps : int, optional
            Optimisation steps per restart. Default 200.
        lr : float, optional
            Adam learning rate. Default 0.1.
        num_restarts : int, optional
            Independent restarts (with optional perturbation). Default 1.
        perturbation_std : float, optional
            Gaussian noise std added to the starting point of each restart. Default 0.0.
        target_value : float | Tensor | None, optional
            Minimise MSE to this scalar target (single task). Overrides ``mode``.
        targets : Sequence[OptimizationTarget] | None, optional
            The primary multi-objective interface: one :class:`OptimizationTarget` per term —
            regression value/direction, kernel-regression target curves, classification
            probability high/low, each with its own ``weight``. Mutually exclusive with
            ``task_targets`` / ``class_targets`` / ``target_value``.
        task_targets : Mapping[str, float | Tensor] | None, optional
            Sugar for value-mode regression targets (each entry becomes an
            ``OptimizationTarget(task=..., value=..., weight=1.0)``). When provided, ``mode`` and
            ``target_value`` are ignored.
        class_targets : Mapping[str, int | Sequence[int]] | None, optional
            Sugar for "high"-direction classification objectives with ``weight=1.0`` (use
            ``targets=`` for "low" direction or a non-default weight). May be combined with
            ``task_targets``.
        ae_align_scale : float, optional
            Latent-space optimization only. How hard to pull the optimised latent ``h`` toward the
            AE's decode/encode fixed set, on a [0, 1] scale.

            * ``0.0``: **no alignment penalty** — pure unconstrained latent optimisation. This was
              shown in PR #18 to fail badly (QC drops from ~0.97 to ~0.35 after the decode/encode
              round-trip); recorded for completeness as a failure-mode baseline.
            * ``1.0``: **strong alignment penalty** — keeps ``h`` close to ``encode(decode(h))``,
              i.e. on the AE's stable manifold. Over-constraining tends to reduce target achievement.
            * ``0.5`` (default): the empirical sweet spot from PR #18 experiments.

            Implementation detail (skip if not curious): the loss gets a
            ``ae_align_scale · ‖tanh(encoder(AE.decode(h))) − h‖²`` term added. Operates in
            **latent space**; orthogonal to :meth:`optimize_composition`'s ``diversity_scale``
            which lives in composition space.
        optimize_space : str, optional
            ``"input"`` or ``"latent"``. Default ``"input"``.

        Returns
        -------
        OptimizationResult
            namedtuple with fields (``T`` = number of targets, one channel per target in
            declaration order — regression: ŷ, curve: RMSE-to-curve, classification: P(classes)):
            - optimized_input  : (B, R, input_dim)
            - optimized_target : (B, R, T)
            - initial_score    : (B, R, T)
            - trajectory       : (B, R, steps, T)

        Raises
        ------
        ValueError
            If ``optimize_space="latent"`` but the model was built without
            ``enable_autoencoder=True``, or if task/mode validation fails.
        """
        _AE_TASK = "__reconstruction__"

        # Validate optimization space
        if optimize_space not in {"input", "latent"}:
            raise ValueError(f"optimize_space must be 'input' or 'latent', got '{optimize_space}'")

        # Resolve every input style into one list of OptimizationTargets.
        resolved_targets: list[OptimizationTarget]
        if targets is not None:
            if task_targets is not None or class_targets is not None or target_value is not None:
                raise ValueError("targets is mutually exclusive with task_targets/class_targets/target_value.")
            resolved_targets = list(targets)
        elif task_targets is not None or class_targets is not None:
            if target_value is not None:
                raise ValueError("Use either task_targets (multi-task) or target_value (single task), not both.")
            if task_targets is not None and (not isinstance(task_targets, Mapping) or len(task_targets) == 0):
                raise ValueError("task_targets must be a non-empty mapping of task_name -> target_value")
            if class_targets is not None and (not isinstance(class_targets, Mapping) or len(class_targets) == 0):
                raise ValueError("class_targets must be a non-empty mapping of task_name -> class index/indices")
            resolved_targets = [
                OptimizationTarget(task=name, value=float(torch.as_tensor(val).reshape(-1)[0]))
                for name, val in (task_targets or {}).items()
            ]
            resolved_targets += [
                OptimizationTarget(task=name, classes=[int(cls)] if isinstance(cls, int) else [int(c) for c in cls])
                for name, cls in (class_targets or {}).items()
            ]
        else:
            # Legacy single-task path (task_name + mode / target_value).
            if task_name is None or task_name not in self.task_heads:
                raise ValueError(
                    f"Task '{task_name}' not found in model. Available tasks: {list(self.task_heads.keys())}"
                )
            if target_value is None and mode not in {"max", "min"}:
                raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")
            if target_value is not None:
                resolved_targets = [
                    OptimizationTarget(task=task_name, value=float(torch.as_tensor(target_value).reshape(-1)[0]))
                ]
            else:
                resolved_targets = [OptimizationTarget(task=task_name, direction="high" if mode == "max" else "low")]

        if not 0.0 <= ae_align_scale <= 1.0:
            raise ValueError(f"ae_align_scale must be in [0, 1], got {ae_align_scale}.")

        # Validate autoencoder availability for latent-space mode
        if optimize_space == "latent":
            if _AE_TASK not in self.task_heads:
                raise ValueError("optimize_space='latent' requires the model to be built with enable_autoencoder=True.")
            if not isinstance(self.task_heads[_AE_TASK], AutoEncoderHead):
                raise ValueError(
                    f"Task '{_AE_TASK}' exists but is not an AutoEncoderHead; "
                    "latent-space optimization requires the built-in reconstruction head."
                )

        if num_restarts < 1:
            raise ValueError(f"num_restarts must be >= 1, got {num_restarts}")

        device = next(self.parameters()).device
        if initial_input is None:
            raise ValueError("initial_input is required and represents the inputs to optimize")

        input_tensor = initial_input
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)
        expected_dim = getattr(self.encoder, "input_dim", None)
        if expected_dim is not None and input_tensor.shape[1] != expected_dim:
            raise ValueError(
                f"initial_input feature dimension mismatch: expected {expected_dim}, got {input_tensor.shape[1]}"
            )

        # Validate the targets against the heads and build the per-term tensors once — BEFORE the
        # requires_grad freeze below, so a validation error cannot leave the model frozen.
        prepared = self._prepare_optimization_targets(resolved_targets, device=device, dtype=input_tensor.dtype)

        # Store original training state. We also snapshot every parameter's ``requires_grad``
        # because the optimisation only differentiates through ``optim_input`` / ``optim_latent``
        # — leaving ``requires_grad=True`` on the model parameters would let ``loss.backward()``
        # populate stale ``.grad`` tensors on the encoder / heads. Mirrors the same pattern used
        # by :meth:`optimize_composition` so a later ``model.fit(...)`` works as expected.
        was_training = self.training
        saved_req_grad: list[tuple[torch.nn.Parameter, bool]] = [(p, p.requires_grad) for p in self.parameters()]
        self.eval()
        for p, _ in saved_req_grad:
            p.requires_grad_(False)

        optimized_inputs: list[torch.Tensor] = []
        optimized_targets: list[torch.Tensor] = []
        trajectories: list[torch.Tensor] = []
        # When ``record_input_trajectory=True`` we snapshot the per-step input every iteration
        # (input-space: ``optim_input`` directly; latent-space: ``AE.decode(tanh(h))``). Stored on
        # CPU to keep GPU memory flat on long trajectories. One per restart, stacked at the end.
        input_trajectories: list[torch.Tensor] = []
        initial_scores_list: list[torch.Tensor] = []

        for restart_idx in range(num_restarts):
            if optimize_space == "input":
                # Input space optimization: optimize X directly
                start_input = input_tensor.clone()
                if perturbation_std > 0:
                    start_input = start_input + torch.randn_like(start_input) * perturbation_std

                # Record initial score(s)
                with torch.no_grad():
                    h_task = torch.tanh(self.encoder(start_input))
                    channels, _ = self._optimization_objective(h_task, prepared)
                    initial_scores_list.append(channels.detach())  # (B, T)

                # Create optimizable input
                optim_input = start_input.detach().clone().requires_grad_(True)

                # Setup optimizer
                optimizer = optim.Adam([optim_input], lr=lr)

                # Optimization loop
                step_traj: list[torch.Tensor] = []
                step_input_traj: list[torch.Tensor] = []

                for step in range(steps):
                    optimizer.zero_grad()

                    # Forward through encoder and apply Tanh
                    h_task = torch.tanh(self.encoder(optim_input))
                    channels, per_sample_losses = self._optimization_objective(h_task, prepared)
                    # Σ over targets per sample, then batch mean — the same objective
                    # evaluate_targets() reports (and the documented Σ wᵢ·termᵢ form).
                    loss = per_sample_losses.sum(dim=1).mean()

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()

                    # Record history
                    step_traj.append(channels.detach())
                    if record_input_trajectory:
                        # Input-space optim variable IS the input — just snapshot it.
                        step_input_traj.append(optim_input.detach().cpu())

                # Get final optimized values
                with torch.no_grad():
                    h_task = torch.tanh(self.encoder(optim_input))
                    per_task_final_tensor, _ = self._optimization_objective(h_task, prepared)
                    per_task_final_tensor = per_task_final_tensor.detach()  # (B, T)
                    optimized_input = optim_input.detach()

                optimized_inputs.append(optimized_input.detach())  # (B, D)
                optimized_targets.append(per_task_final_tensor)  # (B, T)
                traj_tensor = torch.stack(step_traj, dim=0)  # (steps, B, T)
                trajectories.append(traj_tensor)
                if record_input_trajectory:
                    input_trajectories.append(torch.stack(step_input_traj, dim=0))  # (steps, B, D)

            else:  # optimize_space == "latent"
                # Latent space optimization: encode X -> optimize latent -> decode via AE
                with torch.no_grad():
                    initial_latent = self.encoder(input_tensor)

                start_latent = initial_latent.clone()
                if perturbation_std > 0:
                    start_latent = start_latent + torch.randn_like(start_latent) * perturbation_std

                # Record initial score(s)
                # Apply Tanh to get task representation (consistent with forward())
                with torch.no_grad():
                    h_task = torch.tanh(start_latent)
                    channels, _ = self._optimization_objective(h_task, prepared)
                    initial_scores_list.append(channels.detach())  # (B, T)

                # Create optimizable latent
                optim_latent = start_latent.detach().clone().requires_grad_(True)

                # Setup optimizer
                optimizer = optim.Adam([optim_latent], lr=lr)

                # Optimization loop (names already annotated in the input-space branch)
                step_traj = []
                step_input_traj = []

                for step in range(steps):
                    optimizer.zero_grad()

                    # Apply Tanh to get task representation (consistent with forward())
                    # This ensures architectural consistency on every optimization step
                    h_task = torch.tanh(optim_latent)
                    channels, per_sample_losses = self._optimization_objective(h_task, prepared)
                    # Σ over targets per sample, then batch mean (matches evaluate_targets); the
                    # AE term is added on top so its scale is independent of the target count.
                    loss = per_sample_losses.sum(dim=1).mean()
                    if ae_align_scale > 0:
                        # Pull the optimised latent toward what the AE faithfully reconstructs:
                        # decode it to a descriptor, re-encode, and penalise the drift in h_task.
                        # The user-facing knob is [0, 1] with 0 = no penalty / 1 = strong penalty.
                        re_h_task = torch.tanh(self.encoder(self.task_heads[_AE_TASK](h_task)))
                        loss = loss + ae_align_scale * F.mse_loss(re_h_task, h_task)

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()

                    # Record history
                    step_traj.append(channels.detach())
                    if record_input_trajectory:
                        # Latent-space optim: decode the current h via the AE head to recover the
                        # per-step input. ``no_grad`` keeps this from polluting the optim graph.
                        with torch.no_grad():
                            step_input = self.task_heads[_AE_TASK](torch.tanh(optim_latent))
                        step_input_traj.append(step_input.detach().cpu())

                # Get final optimized values and reconstruct via AE
                with torch.no_grad():
                    # Apply Tanh to get final task representation (consistent with forward())
                    final_h_task = torch.tanh(optim_latent)
                    per_task_final_tensor, _ = self._optimization_objective(final_h_task, prepared)
                    per_task_final_tensor = per_task_final_tensor.detach()  # (B, T)

                    # Reconstruct input via the built-in reconstruction head
                    reconstructed_input = self.task_heads[_AE_TASK](final_h_task)

                optimized_inputs.append(reconstructed_input.detach())  # (B, D)
                optimized_targets.append(per_task_final_tensor)  # (B, T)
                traj_tensor = torch.stack(step_traj, dim=0)  # (steps, B, T)
                trajectories.append(traj_tensor)
                if record_input_trajectory:
                    input_trajectories.append(torch.stack(step_input_traj, dim=0))  # (steps, B, D)

        # Restore training state + per-parameter ``requires_grad``. Without the latter, every
        # encoder / head parameter would be left frozen for any later ``.fit()`` in the same
        # Python session — the symptom is "training silently stops moving the encoder" which
        # is annoying to bisect.
        self.train(was_training)
        for p, prev in saved_req_grad:
            p.requires_grad_(prev)

        # Stack outputs
        opt_input_tensor = torch.stack(optimized_inputs, dim=1)  # (B, R, D)
        opt_target_tensor = torch.stack(optimized_targets, dim=1)  # (B, R, T)
        traj_tensor = torch.stack(trajectories, dim=0)  # (R, steps, B, T)
        traj_tensor = traj_tensor.permute(2, 0, 1, 3)  # (B, R, steps, T)
        initial_score_tensor = torch.stack(initial_scores_list, dim=0)  # (R, B, T)
        initial_score_tensor = initial_score_tensor.permute(1, 0, 2)  # (B, R, T)

        input_traj_tensor: torch.Tensor | None = None
        if record_input_trajectory and input_trajectories:
            input_traj_tensor = torch.stack(input_trajectories, dim=0)  # (R, steps, B, D)
            input_traj_tensor = input_traj_tensor.permute(2, 0, 1, 3)  # (B, R, steps, D)

        return OptimizationResult(
            optimized_input=opt_input_tensor,
            optimized_target=opt_target_tensor,
            initial_score=initial_score_tensor,
            trajectory=traj_tensor,
            input_trajectory=input_traj_tensor,
        )

    def optimize_composition(
        self,
        kmd_kernel: torch.Tensor,
        *,
        initial_weights: torch.Tensor | None = None,
        n_starts: int = 16,
        targets: Sequence[OptimizationTarget] | None = None,
        task_targets: Mapping[str, torch.Tensor | float] | None = None,
        class_targets: Mapping[str, int | Sequence[int]] | None = None,
        diversity_scale: float = 1.0,
        allowed_elements: str | list[str] = "all",
        element_step_scale: float | Mapping[str, float] = 1.0,
        fixed_amounts: Mapping[str, float] | None = None,
        min_nonzero_weight: float = 0.0,
        seed_blend: float = 0.95,
        max_elements: int | None = None,
        annealing_scale: float = 0.5,
        annealing_schedule: Mapping[str, Any] | None = None,
        steps: int = 300,
        lr: float = 0.05,
        record_weights_trajectory: bool = False,
    ) -> CompositionOptimizationResult:
        """Gradient-based inverse design in **composition space**.

        Optimises a simplex-constrained element-weight vector ``w`` directly through the
        differentiable KMD transform ``x = w @ K`` and the supervised heads:

        ``logits → w = softmax(logits) → x = w @ kmd_kernel → encoder → tanh → heads → loss``

        Because the optimisation variable *is* the recipe, there is **no AE-decode round-trip**:
        the optimised ``w`` is the composition you would report. Compared to :meth:`optimize_latent`
        in ``"latent"`` mode, this method (a) eliminates the round-trip fidelity drop, (b) keeps the
        solution on the legitimate composition simplex by construction, and (c) makes ``w`` itself
        the output.

        Parameters
        ----------
        kmd_kernel : torch.Tensor
            The precomputed KMD kernel matrix, shape ``(n_components, x_dim)`` — typically obtained
            from :meth:`foundation_model.utils.kmd_plus.KMD.kernel_torch`. ``x_dim`` must match the
            encoder's input dim.
        initial_weights : torch.Tensor | None
            Seed weights, shape ``(B, n_components)``. If ``None``, ``n_starts`` random starts are
            sampled from a Gaussian over the logits (mildly diverse simplex starting points).
        n_starts : int
            Batch size when ``initial_weights is None``. Default 16.
        targets, task_targets, class_targets :
            Same semantics as :meth:`optimize_latent`: ``targets`` is the primary
            :class:`OptimizationTarget` interface (regression value/direction, kernel-regression
            curves, classification high/low, per-target ``weight``); ``task_targets`` /
            ``class_targets`` are sugar for value-mode regression / "high" classification terms.
        diversity_scale : float, optional
            How spread-out the per-output element mixture is allowed to be, on a [0, 1] scale.
            Bigger = more diverse / multi-element per output.

            * ``1.0`` (default): **no penalty** on having many elements — the optimiser is free
              to land on a many-element recipe if the main objective likes it.
            * ``0.0``: **strong penalty** on having many elements — the optimiser is pushed
              toward peaky few-element recipes (e.g. binary alloys).
            * ``0.5`` etc.: linearly interpolates between the two.

            The point is to give users a simple [0, 1] knob without needing to know the underlying
            math. **Implementation detail** (skip if not curious): the loss gets a
            ``(1 − diversity_scale) · H(w)`` term added, where ``H(w) = −Σ w_i log w_i`` is the
            Shannon entropy of the per-row weight vector. ``diversity_scale = 1`` zeros that
            coefficient (no penalty); ``diversity_scale = 0`` applies the full entropy penalty.

            Important: this is a **per-output complexity** knob, not a diversity-*between*-outputs
            knob. Increasing it lets each of the ``B`` outputs individually use more elements;
            whether the ``B`` outputs are different from each other (pairwise L1) depends on the
            optimisation landscape, not on this knob.
        allowed_elements : str | list[str], optional
            Element whitelist for the optimisation. ``"all"`` (default) imposes no constraint.
            A non-empty list of element symbols (e.g. ``["Mg", "Al", "Cu", "Ni"]``) restricts the
            optimisation to those elements only — disallowed elements are forced to ``w = 0`` at
            every step (their logits are masked to ``-inf`` inside the softmax), so no gradient
            ever lifts them. Symbols are resolved against
            :data:`~foundation_model.utils.kmd_plus.DEFAULT_ELEMENTS`; the kernel must therefore
            have ``n_components == len(DEFAULT_ELEMENTS)`` when symbols are used.
        element_step_scale : float | Mapping[str, float], optional
            Per-element constraint on how fast each element's weight can move during optimisation.
            A scalar applies uniformly to every element (default ``1.0`` = no constraint). A
            symbol→float mapping overrides specific elements while leaving the rest at ``1.0``.

            Two regimes with different mechanics:

            * **Hard lock (value = 0):** ``{"Mg": 0.0, "Al": 0.0}`` pins those elements' weights
              at their un-blended ``initial_weights`` values for the entire optimisation. The
              implementation rewrites the softmax output to paste seed values back at locked
              positions and renormalises the unlocked positions over the remaining
              ``1 − Σ_locked seed`` mass — so the locked weights truly do not drift, even when
              other (unlocked) logits move. Requires ``initial_weights`` (no seed → nothing to
              lock to) and the locked elements must be in ``allowed_elements`` if a whitelist
              is set.
            * **Soft constraint (0 < value < 1):** the element's logit gradient is multiplied by
              the scale before each Adam step, slowing (but not freezing) its drift. ``0.1`` lets
              an element move at 10 % of the normal speed. The softmax denominator still couples
              it to the rest of the row, so this is a soft preference, not a hard guarantee.

            Symbols are resolved against ``DEFAULT_ELEMENTS`` (kernel alignment required, as above).
        fixed_amounts : Mapping[str, float] | None, optional
            Pin specific elements at user-specified weights for the entire optimisation; the
            optimiser distributes the remaining mass ``1 − Σ fixed_amounts.values()`` across
            the unfixed elements freely.

            Example: ``{"Au": 0.65, "Ga": 0.20}`` produces recipes with Au exactly 65 % and
            Ga exactly 20 %; the remaining 15 % is split among other allowed elements as the
            objective prefers.

            Implementation reuses the same lock-paste machinery as ``element_step_scale = 0``:
            a per-row tensor ``locked_w0`` is built with the user's amounts at the named
            positions; ``_w_from_logits`` overwrites those positions every step and
            renormalises the unlocked positions over ``1 − Σ locked``.

            Constraints:
              * Each symbol must be in :data:`DEFAULT_ELEMENTS` (kernel alignment required).
              * Each amount must be in ``(0, 1)``; ``Σ values < 1.0`` (need free mass).
              * If ``allowed_elements`` is set, every fixed element must also be in the
                whitelist (locking outside the whitelist is contradictory).
              * If ``element_step_scale = 0`` is also used, the two sets of locked symbols
                **must not overlap** — use one mechanism per element.
              * If ``max_elements`` is also set, fixed elements count toward K (they're
                always in the selection); strict inequality ``max_elements > n_locked_total``
                is enforced.

            Unlike ``element_step_scale = 0``'s hard lock, ``fixed_amounts`` does **not**
            require ``initial_weights`` — the lock values come straight from this kwarg.
        min_nonzero_weight : float, optional
            Lower bound on every unlocked element's final weight: positions with
            ``0 < w < min_nonzero_weight`` are zeroed out and their mass is redistributed across
            the remaining unlocked positions. Default ``0.0`` (no floor).

            Use case: avoid trace-amount appearances (e.g. ``Pt = 0.5%``) that are not
            synthesisable — "if you use it, use ≥ 10%".

            Implementation: applied as the *last* step in ``_w_from_logits`` (after soft top-K
            and lock-paste) and again after the final hard top-K projection. Locked elements
            (from ``element_step_scale = 0`` or ``fixed_amounts``) are **not** subject to the
            floor — their values are set explicitly by the user.

            Constraints:
              * ``0 ≤ min_nonzero_weight ≤ 1``.
              * If ``max_elements`` is set: ``min_nonzero_weight ≤ 1 / max_elements`` (otherwise
                ``K`` elements each ≥ floor can't sum to ≤ 1).
              * If ``fixed_amounts`` is set: every fixed value must be ≥ floor (else
                contradiction).
              * If ``element_step_scale = 0`` locks with ``initial_weights`` are present: every
                locked seed value must be ≥ floor (checked at runtime once the seed is
                normalised).

            Edge case: if dropping every below-floor position would leave a row with zero
            unlocked mass (no element survives), the floor is skipped *for that row only* —
            preserving the simplex (rows always sum to 1). When this happens, the row will
            contain unlocked positions below ``min_nonzero_weight``; if you see this in
            practice your floor is too aggressive for the model's preferred subset.

            Practical note: when ``max_elements`` is not set, no upper bound on the floor is
            enforced beyond ``floor ≤ 1``. A very large floor (e.g. 0.5 with 94 components) will
            silently trigger the per-row fallback on almost every row — the result is a valid
            simplex but the floor is effectively ignored. Pair the floor with ``max_elements``
            (which enforces ``floor ≤ 1 / max_elements``) when you want a hard guarantee.

            "At most K" implication: when combined with ``max_elements``, the floor can drop
            below-floor positions in the K-subset, so the final non-zero count can be **less
            than K** (still ≤ K — the user-facing promise is unchanged).
        seed_blend : float, optional
            How much of the (per-row) seed prior to keep when ``initial_weights`` is given;
            ``w0 ← seed_blend · seed + (1 − seed_blend) · uniform_over_allowed``. Default ``0.95``
            (5 % uniform mass spread over the allowed elements). The blend lifts non-seed-element
            logits from ``log(1e-12) ≈ −27.6`` (effectively unreachable by Adam in a few hundred
            steps) to ``log(0.05 / |allowed|) ≈ −7.6``, so the optimiser can introduce new elements
            when they help the objective. Set to ``1.0`` to reproduce the strict seed-only behaviour
            (no new elements can enter the support set); ``0.0`` makes the seed irrelevant and
            starts from uniform. Ignored when ``initial_weights is None``.
        max_elements : int | None, optional
            If set, restricts the final composition to at most this many non-zero elements.
            Unlike a naive post-hoc top-K projection, the constraint **participates in
            optimisation throughout** via a differentiable iterative-softmax K-hot mask
            (Plötz–Roth, NeurIPS 2018) coupled with a temperature-annealing schedule.

            How it works in one paragraph: at each step we compute a soft K-hot mask
            ``m ∈ [0,1]^n`` with ``Σm = K`` from the same logits the softmax uses, then form
            ``w = (softmax(lg) · m) / Σ(softmax(lg) · m)``. Temperature ``τ`` controls how
            "K-hot" ``m`` is: large τ → uniform-ish (the constraint is soft, gradient can flow
            between candidate subsets), small τ → near one-hot per iteration (constraint is hard).
            τ is driven by the ``annealing_scale`` / ``annealing_schedule`` kwargs below — by
            default a geometric schedule from ``25**annealing_scale`` down to a fixed
            ``τ_end = 0.01``. The annealing doubles as a continuation method that helps escape
            local optima.

            After the loop, a final hard top-K projection is applied so the returned
            ``optimized_weights`` has **at most** ``max_elements`` non-zero positions (subject
            to any locked elements, which are always counted toward K — see below). The
            count saturates at K when the optimiser left at least K positions with positive
            ``w_soft`` mass; if it drove some logits all the way to zero, the row can land
            below K — this is by design, not a bug ("at most K" is the user-facing promise).

            Constraints:
              * ``1 ≤ max_elements ≤ n_components``.
              * If any element is hard-locked via ``element_step_scale=0``, the lock counts
                toward K; require ``max_elements ≥ n_locked``.
              * If ``allowed_elements`` restricts the support, require ``max_elements ≤ |allowed|``.

            ``None`` (default) or ``max_elements == n_components`` disables the constraint.
        annealing_scale : float, optional
            Single-knob "softness" of the annealing schedule, normalised to ``[0, 1]``.
            Default ``0.5``. Maps internally to raw temperature via ``τ_start = 25**scale``:

              * ``0.0`` → ``τ_start = 1.0``    (no exploration; constraint hard from the start)
              * ``0.5`` → ``τ_start = 5.0``    (default; safe choice — QC stable, decent targets)
              * ``1.0`` → ``τ_start = 25.0``   (max exploration; longer soft phase)

            The full schedule is geometric from ``τ_start(scale)`` down to ``τ_end = 0.01``.
            Ignored when ``max_elements`` is None.

            **Calibration**: the 0.5 default was picked from a sweep on the inverse-design
            fine-tuned model (300 steps, K∈{3, 5}; see ``logs/sweep_tau_schedule.png``). Across
            the 3 paper scenarios it keeps QC within ±0.02 of the unconstrained baseline while
            hitting K=3/5 cardinality. For aggressive target chasing, raise toward 0.8-1.0
            (and consider an advanced schedule with ``annealing_func="linear"`` to hold the
            soft phase longer). For QC priority, leave at 0.5.
        annealing_schedule : dict | None, optional
            Advanced piecewise schedule. **Overrides the front of the simple schedule.**
            When supplied, this dict takes precedence over ``annealing_scale``'s implicit
            schedule for the steps it covers. The format is three parallel lists of length N:

            .. code-block:: python

                {
                    "step":           [0.2, 0.5, 1.0],         # fractional step boundaries (0,1]
                    "scale":          [0.8, 0.5, 0.5],         # normalised scale [0,1] at each boundary
                    "annealing_func": ["geometric", "geometric", "geometric"],   # interpolation in each segment
                }

            **Reading the dict**: the schedule starts at step=0 from the value given by
            ``annealing_scale``. Segment ``i`` covers ``(step[i-1], step[i]]`` (with
            ``step[-1] := 0``); within that segment, the normalised scale interpolates from the
            previous segment's endpoint (or ``annealing_scale`` for segment 0) to ``scale[i]``
            using ``annealing_func[i]``. The interpolated scale is then mapped to raw τ via the
            same ``25**scale`` formula used by ``annealing_scale``.

            **If ``step[-1] < 1.0``**, the remaining ``(step[-1], 1.0]`` portion continues with
            a default geometric tail: from the raw τ value at ``step[-1]`` (i.e.
            ``25**scale[-1]``) down to ``τ_end = 0.01``. This guarantees the schedule always
            reaches the hard end inside the loop (the final hard-projection cleans up K-hot
            either way).

            **Allowed annealing_func values**: ``"geometric"``, ``"linear"``, ``"cosine"``,
            ``"constant"``. ``"constant"`` holds the segment's starting value (``scale[i]`` is
            ignored — useful for warm-up phases).
        steps : int
            Adam optimisation steps. Default 300.
        lr : float
            Adam learning rate over the logits. Default 0.05.

        Returns
        -------
        CompositionOptimizationResult
            with fields (``T`` = number of targets, one channel per target in declaration order —
            regression: ŷ, curve: RMSE-to-curve, classification: P(classes)):
            - ``optimized_weights``    : (B, n_components), each row a simplex point — the recipe.
            - ``optimized_descriptor`` : (B, x_dim), equals ``optimized_weights @ kmd_kernel``.
            - ``optimized_target``     : (B, T), final per-target channel values.
            - ``initial_score``        : (B, T), same shape, evaluated at step 0.
            - ``trajectory``           : (steps, B, T), per-target channels across optimisation.
        """
        # --- Validate the kernel ----------------------------------------------------------------
        if not isinstance(kmd_kernel, torch.Tensor) or kmd_kernel.ndim != 2:
            raise ValueError("kmd_kernel must be a 2D torch.Tensor of shape (n_components, x_dim).")
        n_components, x_dim = kmd_kernel.shape
        expected_dim = getattr(self.encoder, "input_dim", None)
        if expected_dim is not None and x_dim != expected_dim:
            raise ValueError(f"kmd_kernel.shape[1]={x_dim} does not match encoder.input_dim={expected_dim}.")

        # --- Resolve the objective into one list of OptimizationTargets (mirrors optimize_latent)
        resolved_targets: list[OptimizationTarget]
        if targets is not None:
            if task_targets is not None or class_targets is not None:
                raise ValueError("targets is mutually exclusive with task_targets/class_targets.")
            resolved_targets = list(targets)
        else:
            if task_targets is not None and (not isinstance(task_targets, Mapping) or len(task_targets) == 0):
                raise ValueError("task_targets must be a non-empty mapping of task_name -> target_value")
            if class_targets is not None and (not isinstance(class_targets, Mapping) or len(class_targets) == 0):
                raise ValueError("class_targets must be a non-empty mapping of task_name -> class index/indices")
            resolved_targets = [
                OptimizationTarget(task=name, value=float(torch.as_tensor(val).reshape(-1)[0]))
                for name, val in (task_targets or {}).items()
            ]
            resolved_targets += [
                OptimizationTarget(task=name, classes=[int(cls)] if isinstance(cls, int) else [int(c) for c in cls])
                for name, cls in (class_targets or {}).items()
            ]
            if not resolved_targets:
                raise ValueError("Provide at least one of targets / task_targets / class_targets.")
        if not 0.0 <= diversity_scale <= 1.0:
            raise ValueError(f"diversity_scale must be in [0, 1], got {diversity_scale}.")
        if not 0.0 <= seed_blend <= 1.0:
            raise ValueError(f"seed_blend must be in [0, 1], got {seed_blend}")

        # --- Per-element constraints (symbol-based) -----------------------------------------------
        # ``allowed_elements`` is a hard whitelist; ``element_step_scale`` is a soft per-element
        # learning-rate multiplier (0 = frozen). Symbol-based inputs are resolved against the
        # bundled :data:`DEFAULT_ELEMENTS` registry — see argument docs above.
        from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS  # local import; small list

        elem_mask_arg: torch.Tensor | None = None
        if isinstance(allowed_elements, str):
            if allowed_elements != "all":
                raise ValueError(f"allowed_elements as a string must be 'all'; got {allowed_elements!r}.")
            # "all": no constraint, leave elem_mask_arg as None.
        elif isinstance(allowed_elements, (list, tuple)):
            if len(allowed_elements) == 0:
                raise ValueError("allowed_elements list must be non-empty.")
            sym_to_idx = {s: i for i, s in enumerate(DEFAULT_ELEMENTS)}
            bad = [s for s in allowed_elements if s not in sym_to_idx]
            if bad:
                raise ValueError(f"Unknown element symbol(s) in allowed_elements: {bad}.")
            if n_components != len(DEFAULT_ELEMENTS):
                raise ValueError(
                    f"allowed_elements as element symbols requires the kernel to align with "
                    f"DEFAULT_ELEMENTS (n_components={n_components}, expected {len(DEFAULT_ELEMENTS)})."
                )
            elem_mask_arg = torch.zeros(n_components, dtype=torch.bool)
            for sym in allowed_elements:
                elem_mask_arg[sym_to_idx[sym]] = True
        else:
            raise TypeError(
                f"allowed_elements must be 'all' or a non-empty list of element symbols; got {type(allowed_elements).__name__}."
            )

        step_scale_arg: torch.Tensor | None = None
        if isinstance(element_step_scale, (int, float)) and not isinstance(element_step_scale, bool):
            if element_step_scale < 0:
                raise ValueError(f"element_step_scale must be >= 0; got {element_step_scale}.")
            if float(element_step_scale) != 1.0:
                step_scale_arg = torch.full((n_components,), float(element_step_scale))
            # else: 1.0 means "no scaling"; keep step_scale_arg = None for the fast path.
        elif isinstance(element_step_scale, Mapping):
            sym_to_idx = {s: i for i, s in enumerate(DEFAULT_ELEMENTS)}
            bad = [s for s in element_step_scale if s not in sym_to_idx]
            if bad:
                raise ValueError(f"Unknown element symbol(s) in element_step_scale: {bad}.")
            if any(float(v) < 0 for v in element_step_scale.values()):
                raise ValueError("element_step_scale values must be >= 0.")
            if n_components != len(DEFAULT_ELEMENTS):
                raise ValueError(
                    f"element_step_scale as a symbol dict requires the kernel to align with "
                    f"DEFAULT_ELEMENTS (n_components={n_components}, expected {len(DEFAULT_ELEMENTS)})."
                )
            step_scale_arg = torch.ones(n_components)
            for sym, val in element_step_scale.items():
                step_scale_arg[sym_to_idx[sym]] = float(val)
        else:
            raise TypeError(
                f"element_step_scale must be a non-negative float or a mapping of "
                f"element_symbol → float; got {type(element_step_scale).__name__}."
            )

        # --- Validate fixed_amounts (per-element explicit pinning) -------------------------------
        # Build the (n_components,) tensors lazily: ``fixed_w0_vec`` (per-element pinned value,
        # zero elsewhere) and ``fixed_mask_vec`` (bool: True at pinned positions). The actual
        # batch-shaped ``locked_w0`` is materialised later (alongside step_scale=0 locks) once we
        # know the batch size.
        fixed_w0_vec: torch.Tensor | None = None
        fixed_mask_vec: torch.Tensor | None = None
        if fixed_amounts is not None:
            if not isinstance(fixed_amounts, Mapping):
                raise TypeError(
                    f"fixed_amounts must be a mapping of element_symbol → float or None; "
                    f"got {type(fixed_amounts).__name__}."
                )
            if len(fixed_amounts) == 0:
                raise ValueError("fixed_amounts must be non-empty when provided.")
            sym_to_idx = {s: i for i, s in enumerate(DEFAULT_ELEMENTS)}
            bad_syms = [s for s in fixed_amounts if s not in sym_to_idx]
            if bad_syms:
                raise ValueError(f"Unknown element symbol(s) in fixed_amounts: {bad_syms}.")
            if n_components != len(DEFAULT_ELEMENTS):
                raise ValueError(
                    f"fixed_amounts requires the kernel to align with DEFAULT_ELEMENTS "
                    f"(n_components={n_components}, expected {len(DEFAULT_ELEMENTS)})."
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
            if elem_mask_arg is not None:
                bad_against_allowed = [s for s in fixed_amounts if not elem_mask_arg[sym_to_idx[s]]]
                if bad_against_allowed:
                    raise ValueError(
                        f"fixed_amounts symbols {bad_against_allowed} are not in allowed_elements — "
                        "pinning a disallowed element is contradictory."
                    )
            # Mutual exclusion with element_step_scale = 0 (the other hard-lock path).
            if step_scale_arg is not None:
                overlap = [s for s in fixed_amounts if float(step_scale_arg[sym_to_idx[s]]) == 0.0]
                if overlap:
                    raise ValueError(
                        f"Symbols {overlap} appear in both element_step_scale=0 and "
                        "fixed_amounts. Use one mechanism per element."
                    )
            fixed_w0_vec = torch.zeros(n_components)
            fixed_mask_vec = torch.zeros(n_components, dtype=torch.bool)
            for sym, amt in fixed_amounts.items():
                idx = sym_to_idx[sym]
                fixed_w0_vec[idx] = float(amt)
                fixed_mask_vec[idx] = True

        # --- Validate min_nonzero_weight (per-element floor) -------------------------------------
        if not 0.0 <= min_nonzero_weight <= 1.0:
            raise ValueError(f"min_nonzero_weight must be in [0, 1]; got {min_nonzero_weight}.")
        if min_nonzero_weight > 0.0:
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

        # --- Validate cardinality constraint (max_elements + annealing knobs) -----------------------
        if max_elements is not None:
            if not isinstance(max_elements, int) or isinstance(max_elements, bool):
                raise TypeError(f"max_elements must be an int or None; got {type(max_elements).__name__}.")
            if not 1 <= max_elements <= n_components:
                raise ValueError(f"max_elements must be in [1, n_components={n_components}]; got {max_elements}.")
            if elem_mask_arg is not None:
                n_allowed = int(elem_mask_arg.sum().item())
                if max_elements > n_allowed:
                    raise ValueError(
                        f"max_elements={max_elements} exceeds the number of allowed elements "
                        f"({n_allowed}). Widen ``allowed_elements`` or lower ``max_elements``."
                    )
            # Lock-vs-K check: locked positions (element_step_scale=0 ∪ fixed_amounts) all count
            # toward K. We require *strict* ``max_elements > n_locked`` for both lock paths:
            # equality leaves the lock-paste with no unlocked slot to absorb the leftover mass
            # (1 − Σ locked) and produces rows that sum to < 1 — silently breaking the simplex.
            # For ``fixed_amounts`` this is definite (``Σ < 1`` enforced at kwarg time); for
            # ``element_step_scale=0`` the seed values *could* sum to exactly 1, but K-constrained
            # all-locked recipes have no degrees of freedom anyway, so rejecting equality is
            # both safe and clearer.
            n_locked_pre = 0
            if step_scale_arg is not None:
                n_locked_pre += int((step_scale_arg == 0).sum().item())
            if fixed_mask_vec is not None:
                n_locked_pre += int(fixed_mask_vec.sum().item())
            if n_locked_pre >= max_elements:
                raise ValueError(
                    f"max_elements={max_elements} must be > total locked elements ({n_locked_pre}, "
                    "counting element_step_scale=0 ∪ fixed_amounts) — the lock-paste needs at "
                    "least one unlocked slot to absorb the leftover mass (1 − Σ locked); equality "
                    "would silently produce row sums < 1. Raise max_elements or unlock some."
                )
            if not 0.0 <= annealing_scale <= 1.0:
                raise ValueError(f"annealing_scale must be in [0, 1]; got {annealing_scale}.")
            if annealing_schedule is not None:
                if not isinstance(annealing_schedule, Mapping):
                    raise TypeError(f"annealing_schedule must be a mapping; got {type(annealing_schedule).__name__}.")
                missing = {"step", "scale", "annealing_func"} - set(annealing_schedule)
                if missing:
                    raise ValueError(
                        f"annealing_schedule missing required keys {sorted(missing)}. "
                        "Required: step, scale, annealing_func — all parallel lists."
                    )
                sched_steps = list(annealing_schedule["step"])
                sched_scales = list(annealing_schedule["scale"])
                sched_funcs = list(annealing_schedule["annealing_func"])
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
                allowed_funcs = ("geometric", "linear", "cosine", "constant")
                for f in sched_funcs:
                    if f not in allowed_funcs:
                        raise ValueError(
                            f"annealing_schedule['annealing_func'] entries must be one of {allowed_funcs}; got {f!r}."
                        )

        # --- Validate the seed (BEFORE touching model state, so a bad input doesn't leave the
        #     model in eval() / with params switched off). ---------------------------------------
        if initial_weights is None:
            if n_starts < 1:
                raise ValueError("n_starts must be >= 1 when initial_weights is None.")
        else:
            if initial_weights.ndim != 2 or initial_weights.shape[1] != n_components:
                raise ValueError(
                    f"initial_weights must have shape (B, {n_components}); got {tuple(initial_weights.shape)}."
                )
            if (initial_weights < 0).any():
                raise ValueError("initial_weights must be non-negative (no silent clamping).")
            if (initial_weights.sum(dim=-1) <= 0).any():
                raise ValueError("initial_weights rows must have a positive sum.")

        # --- Save / restore model state ------------------------------------------------------------
        # Wrap the optimisation in try/finally so a later raise (e.g. a head failure) still
        # restores training mode and parameter requires_grad flags. During the call we also turn
        # off requires_grad on every parameter — only ``logits`` is being optimised, so
        # ``loss.backward()`` would otherwise populate stale ``.grad`` on every encoder/head
        # parameter for no benefit.
        was_training = self.training
        saved_req_grad: list[tuple[torch.nn.Parameter, bool]] = [(p, p.requires_grad) for p in self.parameters()]
        self.eval()
        for p, _ in saved_req_grad:
            p.requires_grad_(False)
        try:
            ref_param = next(self.parameters())
            device, dtype = ref_param.device, ref_param.dtype  # match the model's precision
            kmd_kernel = kmd_kernel.to(device=device, dtype=dtype)

            # --- Build logits over n_components ---------------------------------------------------
            # We additionally capture the *un-blended* normalised seed (``w0_seed``) — the
            # locked-element hard-lock below uses these values, not the post-blend ones, so a
            # user who writes ``element_step_scale={"Mg": 0.0}`` with ``initial_weights`` placing
            # Mg at 0.30 sees Mg held at exactly 0.30 (not the slightly blended 0.286).
            w0_seed: torch.Tensor | None = None
            if initial_weights is None:
                # Use the caller's existing global RNG state — don't reseed here (would defeat
                # the intended diversity across repeated calls and would leak state outward).
                logits = torch.randn(n_starts, n_components, device=device, dtype=dtype) * 0.5
                if elem_mask_arg is not None:
                    # Push disallowed elements to a deep negative logit so softmax mask works
                    # consistently for both the random and seeded branches (the per-step mask
                    # below also enforces this; we mirror it here for the t=0 score).
                    logits = logits.masked_fill(~elem_mask_arg.to(device=device), -1e9)
            else:
                w0 = initial_weights.to(device=device, dtype=dtype)
                w0 = w0 / w0.sum(dim=-1, keepdim=True)
                w0_seed = w0.detach().clone()  # un-blended; used as the lock reference below
                # Blend in a uniform prior so non-seed-element logits are reachable by Adam.
                # Without this, log(0) → −∞ (clamped to log(1e-12) ≈ −27.6); the softmax Jacobian
                # is proportional to w_i, so the per-step gradient on those logits is ≈ 1e-12 and
                # Adam cannot lift them within a few hundred steps — the support set is frozen to
                # the seed's nonzero elements. ``seed_blend < 1`` spreads a small uniform mass
                # over the allowed elements so every reachable element starts at a workable logit.
                if seed_blend < 1.0:
                    if elem_mask_arg is not None:
                        uniform_row = elem_mask_arg.to(device=device, dtype=dtype)
                        uniform_row = uniform_row / uniform_row.sum()
                    else:
                        uniform_row = torch.full((n_components,), 1.0 / n_components, device=device, dtype=dtype)
                    w0 = seed_blend * w0 + (1.0 - seed_blend) * uniform_row
                    w0 = w0 / w0.sum(dim=-1, keepdim=True)
                # Tiny floor only to avoid log(0) when an element is both disallowed AND not in
                # the uniform support (i.e. seed_blend == 1.0 with sparse seeds).
                logits = torch.log(w0.clamp(min=1e-12)).detach().clone()
            logits = logits.requires_grad_(True)
            optimizer = optim.Adam([logits], lr=lr)

            # Validate the targets against the heads and build the per-term tensors once.
            prepared = self._prepare_optimization_targets(resolved_targets, device=device, dtype=dtype)

            # Move the element-constraint tensors onto the right device (validated above).
            elem_mask = elem_mask_arg.to(device=device) if elem_mask_arg is not None else None
            step_scale = step_scale_arg.to(device=device, dtype=dtype) if step_scale_arg is not None else None
            fixed_w0_dev = fixed_w0_vec.to(device=device, dtype=dtype) if fixed_w0_vec is not None else None
            fixed_mask_dev = fixed_mask_vec.to(device=device) if fixed_mask_vec is not None else None

            # --- Hard-lock setup ----------------------------------------------------------------------
            # Two hard-lock sources both end up in the same ``(locked_mask, locked_w0)`` pair so the
            # downstream ``_w_from_logits`` / ``_apply_lock_paste`` logic is unchanged:
            #
            #   1. ``element_step_scale = 0``: pins the listed elements at their (un-blended)
            #      ``initial_weights`` values. Requires ``initial_weights`` because there's no other
            #      source for per-row seed values.
            #   2. ``fixed_amounts``: pins the listed elements at user-given absolute amounts. No
            #      ``initial_weights`` required — the lock values come straight from the kwarg.
            #
            # The two paths must not overlap (validated above). When both are present, we just
            # OR the masks and add the value tensors (disjoint by construction).
            #
            # Why this matters: zeroing ``logit_i.grad`` keeps that logit constant but does NOT keep
            # ``w_i`` constant — softmax renormalises across all logits, so when other (unlocked)
            # logits move, the softmax denominator changes and so does the locked weight. The fix
            # is to (a) detect locked indices, (b) capture their per-row target weights, and (c)
            # inside ``_w_from_logits`` paste those values back over the softmax output and
            # renormalise the unlocked positions to fill the remaining ``1 − Σ locked_w`` mass per
            # row. The gradient through the locked indices is automatically zero (the lock branch
            # uses a constant), so we no longer need the ``step_scale.mul_`` zeroing for them —
            # but we leave that path active for the genuinely soft case ``0 < step_scale < 1``.
            locked_mask: torch.Tensor | None = None
            locked_w0: torch.Tensor | None = None
            if step_scale is not None:
                locked_idx_mask = step_scale == 0
                if locked_idx_mask.any():
                    if w0_seed is None:
                        raise ValueError(
                            "element_step_scale = 0 (hard lock) requires initial_weights — there's no "
                            "per-row seed to lock to when initial_weights=None."
                        )
                    if elem_mask is not None and (~elem_mask[locked_idx_mask]).any():
                        raise ValueError(
                            "Locked elements (element_step_scale = 0) must also be in allowed_elements; "
                            "locking a disallowed element is contradictory."
                        )
                    locked_mask = locked_idx_mask  # (n_components,) bool, on device
                    # (B, n_components): seed values at locked positions, 0 elsewhere — constant.
                    locked_w0 = (w0_seed * locked_mask.to(dtype)).detach()
            if fixed_mask_dev is not None:
                assert fixed_w0_dev is not None  # built together with fixed_mask_dev
                # Broadcast the per-element fixed values to every row in the batch.
                B = logits.shape[0]
                fixed_w0_batch = fixed_w0_dev.unsqueeze(0).expand(B, -1).detach()
                if locked_mask is None:
                    locked_mask = fixed_mask_dev
                    locked_w0 = fixed_w0_batch
                else:
                    assert locked_w0 is not None  # set alongside locked_mask above
                    locked_mask = locked_mask | fixed_mask_dev  # validated disjoint
                    locked_w0 = locked_w0 + fixed_w0_batch

            # Runtime sanity: combined lock sum must leave room (or fit exactly) for the simplex.
            # ``fixed_amounts`` enforces ``Σ < 1`` at kwarg time, and ``element_step_scale=0``
            # locks at seed values which sum to ≤ 1 per row — but the *combined* total could
            # exceed 1 (e.g. seed-lock Mg=0.50 + fix Au=0.65). Check here, with a tiny tolerance
            # for float noise.
            if locked_w0 is not None:
                lock_sums = locked_w0.sum(dim=-1)
                if (lock_sums > 1.0 + 1e-5).any():
                    raise ValueError(
                        f"Combined locked mass exceeds 1.0 on at least one row "
                        f"(max row-sum = {float(lock_sums.max()):.4f}). Likely cause: "
                        "``element_step_scale=0`` locks plus ``fixed_amounts`` together claim more "
                        "than 100% of the simplex. Lower one set of values or drop a lock."
                    )

            # Runtime sanity: floored elements must not contradict the lock-paste targets.
            # ``fixed_amounts`` was checked at kwarg time; ``element_step_scale=0`` locks have
            # per-row seed values we couldn't see earlier — verify them now.
            if min_nonzero_weight > 0.0 and locked_mask is not None and locked_w0 is not None:
                locked_below_floor = (locked_w0 > 0) & (locked_w0 < min_nonzero_weight)
                if locked_below_floor.any():
                    raise ValueError(
                        f"At least one locked element's value falls below min_nonzero_weight="
                        f"{min_nonzero_weight}. Likely cause: an element_step_scale=0 lock points "
                        "at a seed value below the floor (raise the seed, lower the floor, or "
                        "drop the lock)."
                    )

            # --- Soft top-K (cardinality constraint) helpers ----------------------------------------
            # Schedule shape (controlled by ``annealing_scale`` and optionally ``annealing_schedule``):
            #
            #   * Normalised scale ∈ [0, 1] is the user-facing knob; raw τ is derived via
            #     ``τ = 25**scale``  (so scale=0 → τ=1, scale=0.5 → τ=5, scale=1 → τ=25).
            #   * Default schedule when no dict is given: geometric from ``τ_start=25**annealing_scale``
            #     at fractional step 0 down to ``_TAU_END=0.01`` at fractional step 1.
            #   * When ``annealing_schedule`` dict is provided, its segments override the front of
            #     the schedule; the segment from ``step[-1]`` to 1.0 (if not already at 1.0) falls
            #     back to the geometric tail from ``25**scale[-1]`` down to ``_TAU_END``.
            #
            # ``current_tau`` lives in a list so the optimisation loop can mutate it each step
            # without rebuilding the ``_w_from_logits`` closure that reads it.
            _TAU_FLOOR = 1e-3  # numerical lower bound; below this softmax(lg/τ) loses precision
            _TAU_END = 0.01  # fixed final hardness for the default schedule's tail
            _SCALE_TAU_BASE = 25.0  # τ = _SCALE_TAU_BASE**scale → 0→1, 0.5→5, 1→25

            def _scale_to_tau(scale: float) -> float:
                return float(_SCALE_TAU_BASE ** max(0.0, min(1.0, scale)))

            def _interp_scalar(a: float, b: float, t: float, func: str) -> float:
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

            # Materialise schedule arrays once (validated above), so the per-step lookup is light.
            _sched_steps: list[float] = (
                [float(s) for s in annealing_schedule["step"]] if annealing_schedule is not None else []
            )
            _sched_scales: list[float] = (
                [float(t) for t in annealing_schedule["scale"]] if annealing_schedule is not None else []
            )
            _sched_funcs: list[str] = (
                list(annealing_schedule["annealing_func"]) if annealing_schedule is not None else []
            )

            def _tau_for_step(step: int) -> float:
                """Return the raw τ for integer optimisation step ``step``."""
                if max_elements is None or steps <= 1:
                    return float(max(_TAU_END, _TAU_FLOOR))
                # Fractional progress in [0, 1].
                s = step / (steps - 1)
                # Default schedule (used directly when no dict, or for the tail when dict ends < 1.0).
                default_tau_start = _scale_to_tau(annealing_scale)
                default_tau_end = _TAU_END

                if _sched_steps:
                    # Walk through dict segments to find the one containing ``s``.
                    prev_step = 0.0
                    prev_scale = annealing_scale  # segment 0 starts at the simple knob's value
                    for i, seg_end in enumerate(_sched_steps):
                        if s <= seg_end:
                            local_t = (s - prev_step) / max(seg_end - prev_step, 1e-12)
                            scale_now = _interp_scalar(prev_scale, _sched_scales[i], local_t, _sched_funcs[i])
                            return float(max(_scale_to_tau(scale_now), _TAU_FLOOR))
                        prev_step = seg_end
                        prev_scale = _sched_scales[i]
                    # ``s`` is past the dict's last step → use the geometric tail from
                    # ``25**scale[-1]`` at ``step[-1]`` down to ``_TAU_END`` at 1.0.
                    tail_start_tau = _scale_to_tau(_sched_scales[-1])
                    tail_end_step = 1.0
                    tail_local_t = (s - _sched_steps[-1]) / max(tail_end_step - _sched_steps[-1], 1e-12)
                    val = tail_start_tau * (default_tau_end / tail_start_tau) ** tail_local_t
                    return float(max(val, _TAU_FLOOR))

                # No dict — default geometric schedule from τ_start(annealing_scale) to _TAU_END.
                val = default_tau_start * (default_tau_end / default_tau_start) ** s
                return float(max(val, _TAU_FLOOR))

            current_tau = [_tau_for_step(0)]

            def _soft_topk_mask(
                lg: torch.Tensor, K: int, tau: float, *, force_select: torch.Tensor | None = None
            ) -> torch.Tensor:
                """Plötz–Roth iterative softmax. Returns m ∈ [0,1]^(B, n) with Σm = K.

                ``force_select`` (n_components,) bool marks positions that must be in the K
                selection (e.g. hard-locked elements). Instead of boosting those logits — which
                would make the iterative softmax pick them K times in a row, never moving on —
                we **pre-seed** the mask with 1.0 at those positions and run only ``K - n_locked``
                iterations on the *unlocked* positions (their logits are masked to ``-inf``
                inside the iteration so they never compete).
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
                    # The shift in scaled-logit space at the selected position is
                    # ``log(1−p)/τ`` — at small τ this is enormously negative, so the next
                    # iteration cannot re-pick the same position. (We must NOT multiply by τ here.)
                    alpha = alpha + torch.log((1.0 - p).clamp(min=1e-12))
                return m

            def _hard_topk_project(w: torch.Tensor, K: int) -> torch.Tensor:
                """Hard top-K projection: keep K largest per row, zero rest, renormalise.

                If ``locked_mask`` is set, every locked position is forced into the kept set
                (so the lock-paste below still has a place to write its seed values); the
                remaining ``K − n_locked`` slots are filled by the largest unlocked weights.
                """
                if locked_mask is None:
                    _, idx = w.topk(K, dim=-1)
                    keep = torch.zeros_like(w).scatter_(-1, idx, 1.0)
                else:
                    n_locked = int(locked_mask.sum().item())
                    n_free = K - n_locked
                    locked_row = locked_mask.to(w.dtype).unsqueeze(0).expand_as(w)
                    if n_free > 0:
                        # Exclude locked positions from the unlocked competition by sending them
                        # to ``-inf`` before topk; locked positions are added back via ``locked_row``.
                        w_for_free = w.masked_fill(locked_mask.unsqueeze(0), float("-inf"))
                        _, idx = w_for_free.topk(n_free, dim=-1)
                        free_keep = torch.zeros_like(w).scatter_(-1, idx, 1.0)
                        keep = (locked_row + free_keep).clamp(max=1.0)
                    else:
                        keep = locked_row
                w = w * keep
                return w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)

            def _apply_lock_paste(w: torch.Tensor) -> torch.Tensor:
                """Paste locked seed values onto ``w`` and renormalise unlocked positions."""
                if locked_mask is None:
                    return w
                assert locked_w0 is not None  # set alongside locked_mask
                free_mask_f = (~locked_mask).to(w.dtype)
                w_unlocked = w * free_mask_f
                free_mass = (1.0 - locked_w0.sum(dim=-1, keepdim=True)).clamp(min=0.0)
                w_unlocked = w_unlocked / w_unlocked.sum(dim=-1, keepdim=True).clamp(min=1e-12) * free_mass
                return w_unlocked + locked_w0

            def _apply_min_floor(w: torch.Tensor) -> torch.Tensor:
                """Drop unlocked positions below ``min_nonzero_weight`` and re-fill free mass.

                Locked positions are exempt (their values are user-set). If dropping below-floor
                positions would leave a row with zero unlocked mass, the floor is skipped for
                that row — preserving the simplex invariant. The "at most K" guarantee still
                holds; some rows may end up with fewer than K non-zero positions.
                """
                if min_nonzero_weight <= 0.0:
                    return w
                if locked_mask is not None:
                    assert locked_w0 is not None  # set alongside locked_mask
                    unlocked_f = (~locked_mask).to(w.dtype)
                    free_mass = (1.0 - locked_w0.sum(dim=-1, keepdim=True)).clamp(min=0.0)
                    unlocked_bool = (~locked_mask).unsqueeze(0).expand_as(w)
                else:
                    unlocked_f = torch.ones_like(w[0])
                    free_mass = torch.ones(w.shape[0], 1, dtype=w.dtype, device=w.device)
                    unlocked_bool = torch.ones_like(w, dtype=torch.bool)
                below = (w > 0) & (w < min_nonzero_weight) & unlocked_bool
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
                if locked_mask is not None:
                    locked_part = w_drop * locked_mask.to(w.dtype)
                    w_renorm = locked_part + unlocked_after * (free_mass / safe_sum)
                else:
                    w_renorm = w_drop / safe_sum
                return torch.where(can_drop.expand_as(w), w_renorm, w)

            def _w_from_logits(lg: torch.Tensor) -> torch.Tensor:
                """Softmax → optional soft top-K → optional hard-lock paste → optional min-floor.

                Reads ``current_tau[0]`` (set by the outer loop) for the soft top-K temperature.
                """
                if elem_mask is not None:
                    lg = lg.masked_fill(~elem_mask, float("-inf"))
                w_soft = torch.softmax(lg, dim=-1)
                if max_elements is not None and max_elements < n_components:
                    # Force locked positions to always sit in the K-hot mask so the lock-paste
                    # below has somewhere to write. ``w_soft`` itself is computed from the
                    # *unboosted* logits, so the within-K ratios reflect the optimisation state.
                    m_topk = _soft_topk_mask(lg, max_elements, current_tau[0], force_select=locked_mask)
                    w = w_soft * m_topk
                    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                else:
                    w = w_soft
                return _apply_min_floor(_apply_lock_paste(w))

            def _heads_forward(h_task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                """Evaluate the target terms; return (per-target channels (B, T), scalar objective).

                The objective is Σ over targets per sample, then batch mean — the same quantity
                :meth:`evaluate_targets` reports (and the documented Σ wᵢ·termᵢ form).
                """
                channels, per_sample_losses = self._optimization_objective(h_task, prepared)
                return channels, per_sample_losses.sum(dim=1).mean()

            n_channels = len(prepared)

            # --- Record initial scores --------------------------------------------------------------
            # Initial scoring uses τ at step 0 of the annealing schedule — i.e. the softest end
            # of the (annealing_scale + annealing_schedule)-derived τ curve, where the optimisation
            # actually begins.
            current_tau[0] = _tau_for_step(0)
            with torch.no_grad():
                w0_tensor = _w_from_logits(logits)
                h0 = torch.tanh(self.encoder(w0_tensor @ kmd_kernel))
                initial_channels, _ = _heads_forward(h0)
                initial_score = initial_channels.detach()

            # --- Optimisation loop ------------------------------------------------------------------
            # With every model parameter at ``requires_grad=False``, ``loss.backward()`` populates
            # gradient only on ``logits`` — no stale grads accumulate on encoder/heads.
            trajectory: list[torch.Tensor] = []
            weights_trajectory: list[torch.Tensor] = [] if record_weights_trajectory else []
            for step in range(steps):
                current_tau[0] = _tau_for_step(step)
                optimizer.zero_grad()
                w = _w_from_logits(logits)
                x = w @ kmd_kernel
                h_task = torch.tanh(self.encoder(x))
                channels, loss = _heads_forward(h_task)
                if diversity_scale < 1.0:
                    # The penalty strength is (1 − diversity_scale): user sees a [0, 1] knob
                    # where 1 means "no penalty / most diverse" and 0 means "max penalty / most
                    # peaky". The internal term is `(1 − diversity_scale) · H(w)` added to loss —
                    # additively, so its scale is independent of the target count.
                    entropy = -(w * w.clamp(min=1e-12).log()).sum(dim=-1).mean()
                    loss = loss + (1.0 - diversity_scale) * entropy
                loss.backward()
                if step_scale is not None and logits.grad is not None:
                    # Soft per-element constraint: scale each element's logit gradient (0 = frozen).
                    logits.grad.mul_(step_scale)
                optimizer.step()
                trajectory.append(channels.detach())
                if record_weights_trajectory:
                    # Snapshot the post-step weights at the *current* (still-soft) τ — the
                    # trajectory thus reflects the annealing schedule, not the hard projection.
                    # Stored on CPU to keep GPU memory flat for long trajectories on large B.
                    with torch.no_grad():
                        weights_trajectory.append(_w_from_logits(logits).detach().cpu())

            # --- Final state ------------------------------------------------------------------------
            # Use the hardest τ for the final readout, then (if ``max_elements`` is active) apply
            # a hard top-K projection so the returned ``optimized_weights`` has **at most** K
            # non-zero positions (the floor below may reduce that further) — at τ_end ≈ 0.01 the
            # soft mask is already near-K-hot, so the projection just cleans up residual
            # sub-threshold weights.
            current_tau[0] = float(max(_TAU_END, _TAU_FLOOR))
            with torch.no_grad():
                w_final = _w_from_logits(logits)
                if max_elements is not None and max_elements < n_components:
                    w_final = _hard_topk_project(w_final, max_elements)
                    # Re-apply lock-paste — the projection may have re-distributed mass across
                    # unlocked positions, and lock-paste's "free mass" renormalisation needs to
                    # be re-run so the row still sums to exactly 1. Then re-floor: the projection
                    # may have promoted a previously-zeroed below-floor position back in.
                    w_final = _apply_lock_paste(w_final)
                    w_final = _apply_min_floor(w_final)
                x_final = w_final @ kmd_kernel
                h_final = torch.tanh(self.encoder(x_final))
                final_channels, _ = _heads_forward(h_final)
                final_target = final_channels.detach()

            weights_traj_tensor: torch.Tensor | None = None
            if record_weights_trajectory:
                # (steps, B, n_components). Same empty-steps fallback as ``trajectory`` so the
                # downstream code can rely on the shape contract without a None branch.
                weights_traj_tensor = (
                    torch.stack(weights_trajectory, dim=0)
                    if weights_trajectory
                    else torch.empty((0, logits.shape[0], n_components), dtype=torch.float32)
                )

            return CompositionOptimizationResult(
                optimized_weights=w_final.detach(),
                optimized_descriptor=x_final.detach(),
                optimized_target=final_target,
                initial_score=initial_score,
                # Preserve the (steps, B, T) shape contract even when steps == 0.
                trajectory=torch.stack(trajectory, dim=0)
                if trajectory
                else torch.empty((0, logits.shape[0], n_channels), device=device, dtype=dtype),
                weights_trajectory=weights_traj_tensor,
            )
        finally:
            if was_training:
                self.train()
            for p, prev in saved_req_grad:
                p.requires_grad_(prev)
