# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Inverse design in the encoder's latent space.

Descends on a descriptor — either the input directly, or a latent code decoded back through the
autoencoder — until the heads predict what the caller asked for. What comes back is therefore a
*descriptor*, which still has to be inverted to a recipe; :mod:`.composition` is the path that
optimises the recipe itself and skips that round trip.

The model arrives as the first argument rather than as ``self``. It is a mixin method one line
away (see :mod:`.mixin`), so ``model.optimize_latent(...)`` is unchanged for every caller, but the
dependency is stated here instead of assumed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..task_head.autoencoder import AutoEncoderHead
from .targets import OptimizationResult, OptimizationTarget, targets_from_mappings

if TYPE_CHECKING:  # the host class, for typing only — importing it at runtime would be a cycle
    from .mixin import InverseDesignMixin


def optimize_latent(
    model: InverseDesignMixin,
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
        resolved_targets = targets_from_mappings(task_targets, class_targets)
    else:
        # Legacy single-task path (task_name + mode / target_value).
        if task_name is None or task_name not in model.task_heads:
            raise ValueError(f"Task '{task_name}' not found in model. Available tasks: {list(model.task_heads.keys())}")
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
        if _AE_TASK not in model.task_heads:
            raise ValueError("optimize_space='latent' requires the model to be built with enable_autoencoder=True.")
        if not isinstance(model.task_heads[_AE_TASK], AutoEncoderHead):
            raise ValueError(
                f"Task '{_AE_TASK}' exists but is not an AutoEncoderHead; "
                "latent-space optimization requires the built-in reconstruction head."
            )

    if num_restarts < 1:
        raise ValueError(f"num_restarts must be >= 1, got {num_restarts}")

    device = next(model.parameters()).device
    if initial_input is None:
        raise ValueError("initial_input is required and represents the inputs to optimize")

    input_tensor = initial_input
    if input_tensor.ndim == 1:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)
    expected_dim = getattr(model.encoder, "input_dim", None)
    if expected_dim is not None and input_tensor.shape[1] != expected_dim:
        raise ValueError(
            f"initial_input feature dimension mismatch: expected {expected_dim}, got {input_tensor.shape[1]}"
        )

    # Validate the targets against the heads and build the per-term tensors once — BEFORE the
    # requires_grad freeze below, so a validation error cannot leave the model frozen.
    prepared = model._prepare_optimization_targets(resolved_targets, device=device, dtype=input_tensor.dtype)

    # Store original training state. We also snapshot every parameter's ``requires_grad``
    # because the optimisation only differentiates through ``optim_input`` / ``optim_latent``
    # — leaving ``requires_grad=True`` on the model parameters would let ``loss.backward()``
    # populate stale ``.grad`` tensors on the encoder / heads. Mirrors the same pattern used
    # by :meth:`optimize_composition` so a later ``model.fit(...)`` works as expected.
    was_training = model.training
    saved_req_grad: list[tuple[torch.nn.Parameter, bool]] = [(p, p.requires_grad) for p in model.parameters()]
    model.eval()
    for p, _ in saved_req_grad:
        p.requires_grad_(False)

    # try/finally, matching optimize_composition: an exception anywhere in the search
    # must not leave the model frozen and in eval() for the rest of the session. The
    # symptom of that leak is 'training silently stops moving the encoder', which is
    # expensive to bisect and invisible in logs.
    try:
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
                    h_task = torch.tanh(model.encoder(start_input))
                    channels, _ = model._optimization_objective(h_task, prepared)
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
                    h_task = torch.tanh(model.encoder(optim_input))
                    channels, per_sample_losses = model._optimization_objective(h_task, prepared)
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
                    h_task = torch.tanh(model.encoder(optim_input))
                    per_task_final_tensor, _ = model._optimization_objective(h_task, prepared)
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
                    initial_latent = model.encoder(input_tensor)

                start_latent = initial_latent.clone()
                if perturbation_std > 0:
                    start_latent = start_latent + torch.randn_like(start_latent) * perturbation_std

                # Record initial score(s)
                # Apply Tanh to get task representation (consistent with forward())
                with torch.no_grad():
                    h_task = torch.tanh(start_latent)
                    channels, _ = model._optimization_objective(h_task, prepared)
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
                    channels, per_sample_losses = model._optimization_objective(h_task, prepared)
                    # Σ over targets per sample, then batch mean (matches evaluate_targets); the
                    # AE term is added on top so its scale is independent of the target count.
                    loss = per_sample_losses.sum(dim=1).mean()
                    if ae_align_scale > 0:
                        # Pull the optimised latent toward what the AE faithfully reconstructs:
                        # decode it to a descriptor, re-encode, and penalise the drift in h_task.
                        # The user-facing knob is [0, 1] with 0 = no penalty / 1 = strong penalty.
                        re_h_task = torch.tanh(model.encoder(model.task_heads[_AE_TASK](h_task)))
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
                            step_input = model.task_heads[_AE_TASK](torch.tanh(optim_latent))
                        step_input_traj.append(step_input.detach().cpu())

                # Get final optimized values and reconstruct via AE
                with torch.no_grad():
                    # Apply Tanh to get final task representation (consistent with forward())
                    final_h_task = torch.tanh(optim_latent)
                    per_task_final_tensor, _ = model._optimization_objective(final_h_task, prepared)
                    per_task_final_tensor = per_task_final_tensor.detach()  # (B, T)

                    # Reconstruct input via the built-in reconstruction head
                    reconstructed_input = model.task_heads[_AE_TASK](final_h_task)

                optimized_inputs.append(reconstructed_input.detach())  # (B, D)
                optimized_targets.append(per_task_final_tensor)  # (B, T)
                traj_tensor = torch.stack(step_traj, dim=0)  # (steps, B, T)
                trajectories.append(traj_tensor)
                if record_input_trajectory:
                    input_trajectories.append(torch.stack(step_input_traj, dim=0))  # (steps, B, D)

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
    finally:
        # Restore training state + per-parameter ``requires_grad``. Without the latter, every
        # encoder / head parameter would be left frozen for any later ``.fit()`` in the same
        # Python session — the symptom is "training silently stops moving the encoder" which
        # is annoying to bisect.
        model.train(was_training)
        for p, prev in saved_req_grad:
            p.requires_grad_(prev)
