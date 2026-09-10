# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Inverse design directly on the composition simplex — the KMD differentiable path.

Optimises element weights ``w`` through ``x = w @ K`` and the supervised heads, so the optimisation
variable *is* the recipe: no autoencoder round trip, no fidelity drop on the way back, and the
result is already on the legitimate simplex.

What this module holds is the loop and the bookkeeping around it. The three things that used to
make it a 975-line method live next door: what the recipe is allowed to be
(:mod:`.constraints`), how a logit vector becomes one (:mod:`.simplex`), and how fast the
cardinality limit commits (:mod:`.annealing`).

The model arrives as the first argument rather than as ``self`` — see :mod:`.latent` for why.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.optim as optim

from .constraints import CompositionConstraints
from .targets import CompositionOptimizationResult, OptimizationTarget, targets_from_mappings

if TYPE_CHECKING:  # the host class, for typing only — importing it at runtime would be a cycle
    from .mixin import InverseDesignMixin


def optimize_composition(
    model: InverseDesignMixin,
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
    expected_dim = getattr(model.encoder, "input_dim", None)
    if expected_dim is not None and x_dim != expected_dim:
        raise ValueError(f"kmd_kernel.shape[1]={x_dim} does not match encoder.input_dim={expected_dim}.")

    # --- Resolve the objective into one list of OptimizationTargets (mirrors optimize_latent)
    resolved_targets: list[OptimizationTarget]
    if targets is not None:
        if task_targets is not None or class_targets is not None:
            raise ValueError("targets is mutually exclusive with task_targets/class_targets.")
        resolved_targets = list(targets)
    else:
        resolved_targets = targets_from_mappings(task_targets, class_targets)
        if not resolved_targets:
            raise ValueError("Provide at least one of targets / task_targets / class_targets.")
    if not 0.0 <= diversity_scale <= 1.0:
        raise ValueError(f"diversity_scale must be in [0, 1], got {diversity_scale}.")
    if not 0.0 <= seed_blend <= 1.0:
        raise ValueError(f"seed_blend must be in [0, 1], got {seed_blend}")

    # --- Every constraint on the recipe, validated once -------------------------------------
    # allowed_elements / element_step_scale / fixed_amounts / min_nonzero_weight /
    # max_elements can each contradict another, and a contradiction that is not caught
    # produces a plausible-looking recipe rather than an error. .constraints catches them
    # all here, before the model is touched, and freezes the result.
    constraints = CompositionConstraints.build(
        n_components=n_components,
        allowed_elements=allowed_elements,
        element_step_scale=element_step_scale,
        fixed_amounts=fixed_amounts,
        min_nonzero_weight=min_nonzero_weight,
        max_elements=max_elements,
        annealing_scale=annealing_scale,
        annealing_schedule=annealing_schedule,
        steps=steps,
    )
    schedule = constraints.schedule

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
    was_training = model.training
    saved_req_grad: list[tuple[torch.nn.Parameter, bool]] = [(p, p.requires_grad) for p in model.parameters()]
    model.eval()
    for p, _ in saved_req_grad:
        p.requires_grad_(False)
    try:
        ref_param = next(model.parameters())
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
            if constraints.elem_mask is not None:
                # Push disallowed elements to a deep negative logit so softmax mask works
                # consistently for both the random and seeded branches (the per-step mask
                # below also enforces this; we mirror it here for the t=0 score).
                logits = logits.masked_fill(~constraints.elem_mask.to(device=device), -1e9)
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
                if constraints.elem_mask is not None:
                    uniform_row = constraints.elem_mask.to(device=device, dtype=dtype)
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
        prepared = model._prepare_optimization_targets(resolved_targets, device=device, dtype=dtype)

        constraints = constraints.on(device=device, dtype=dtype)

        # --- Hard locks, now that there is a batch to lock ------------------------------------
        # element_step_scale=0 pins at the seed's per-row values, so this half could not run
        # at validation time. Both lock sources come back as one (mask, values) pair.
        locked_mask, locked_w0 = constraints.resolve_locks(w0_seed=w0_seed, batch_size=logits.shape[0], dtype=dtype)

        # --- Logits → a legal recipe -------------------------------------------------------------
        # Every constraint settled above, bound once. The loop calls
        # ``project.w_from_logits(logits, tau)`` and gets back a row on the simplex; the
        # maths lives in .simplex, which is where to read about the soft top-K, the
        # lock-paste and the floor.
        project = constraints.projector(locked_mask, locked_w0)

        def _heads_forward(h_task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Evaluate the target terms; return (per-target channels (B, T), scalar objective).

            The objective is Σ over targets per sample, then batch mean — the same quantity
            :meth:`evaluate_targets` reports (and the documented Σ wᵢ·termᵢ form).
            """
            channels, per_sample_losses = model._optimization_objective(h_task, prepared)
            return channels, per_sample_losses.sum(dim=1).mean()

        n_channels = len(prepared)

        # --- Record initial scores --------------------------------------------------------------
        # Initial scoring uses τ at step 0 of the annealing schedule — i.e. the softest end
        # of the (annealing_scale + annealing_schedule)-derived τ curve, where the optimisation
        # actually begins.
        with torch.no_grad():
            w0_tensor = project.w_from_logits(logits, schedule.tau_for_step(0))
            h0 = torch.tanh(model.encoder(w0_tensor @ kmd_kernel))
            initial_channels, _ = _heads_forward(h0)
            initial_score = initial_channels.detach()

        # --- Optimisation loop ------------------------------------------------------------------
        # With every model parameter at ``requires_grad=False``, ``loss.backward()`` populates
        # gradient only on ``logits`` — no stale grads accumulate on encoder/heads.
        trajectory: list[torch.Tensor] = []
        weights_trajectory: list[torch.Tensor] = [] if record_weights_trajectory else []
        for step in range(steps):
            tau = schedule.tau_for_step(step)
            optimizer.zero_grad()
            w = project.w_from_logits(logits, tau)
            x = w @ kmd_kernel
            h_task = torch.tanh(model.encoder(x))
            channels, loss = _heads_forward(h_task)
            if diversity_scale < 1.0:
                # The penalty strength is (1 − diversity_scale): user sees a [0, 1] knob
                # where 1 means "no penalty / most diverse" and 0 means "max penalty / most
                # peaky". The internal term is `(1 − diversity_scale) · H(w)` added to loss —
                # additively, so its scale is independent of the target count.
                entropy = -(w * w.clamp(min=1e-12).log()).sum(dim=-1).mean()
                loss = loss + (1.0 - diversity_scale) * entropy
            loss.backward()
            if constraints.step_scale is not None and logits.grad is not None:
                # Soft per-element constraint: scale each element's logit gradient (0 = frozen).
                logits.grad.mul_(constraints.step_scale)
            optimizer.step()
            trajectory.append(channels.detach())
            if record_weights_trajectory:
                # Snapshot the post-step weights at the *current* (still-soft) τ — the
                # trajectory thus reflects the annealing schedule, not the hard projection.
                # Stored on CPU to keep GPU memory flat for long trajectories on large B.
                with torch.no_grad():
                    weights_trajectory.append(project.w_from_logits(logits, tau).detach().cpu())

        # --- Final state ------------------------------------------------------------------------
        # Use the hardest τ for the final readout, then (if ``max_elements`` is active) apply
        # a hard top-K projection so the returned ``optimized_weights`` has **at most** K
        # non-zero positions (the floor below may reduce that further) — at τ_end ≈ 0.01 the
        # soft mask is already near-K-hot, so the projection just cleans up residual
        # sub-threshold weights.
        with torch.no_grad():
            # Hardest τ for the final readout, then the hard top-K clean-up: at τ_end ≈ 0.01
            # the soft mask is already near-K-hot, so the projection only removes residual
            # sub-threshold weights.
            w_final = project.hard_project(project.w_from_logits(logits, schedule.final_tau))
            x_final = w_final @ kmd_kernel
            h_final = torch.tanh(model.encoder(x_final))
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
            model.train()
        for p, prev in saved_req_grad:
            p.requires_grad_(prev)
