# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``finetune_inverse_heads.freeze_except`` — the per-parameter freeze contract.

The full ``finetune`` entry point needs a real checkpoint + data parquets, so it's exercised by
the smoke runs under ``artifacts/inverse_design_run/finetune/``. The unit-testable piece is the
freeze logic, which is the most refactor-fragile part: a future change that accidentally
un-freezes the encoder (or forgets the per-task loss-balancer scalars) would silently break
the "apples-to-apples" comparison the script exists to enable.
"""

from __future__ import annotations

import pytest
import torch

from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import (
    ClassificationTaskConfig,
    MLPEncoderConfig,
    RegressionTaskConfig,
)
from foundation_model.scripts.finetune_inverse_heads import freeze_except


INPUT_DIM = 16
LATENT_DIM = 8


def _make_model(enable_balancer: bool = False) -> FlexibleMultiTaskModel:
    """Three-head model mirroring the inverse-design tail (formation_energy / klat / material_type).

    ``enable_autoencoder=False`` keeps the test fast — the freeze contract doesn't depend on the
    AE head; the smoke run covers that path.
    """
    enc = MLPEncoderConfig(hidden_dims=[INPUT_DIM, LATENT_DIM])
    tasks = [
        RegressionTaskConfig(name="formation_energy", data_column="formation_energy", dims=[LATENT_DIM, 4, 1]),
        RegressionTaskConfig(name="klat", data_column="klat", dims=[LATENT_DIM, 4, 1]),
        ClassificationTaskConfig(name="material_type", data_column="material_type", num_classes=3, dims=[LATENT_DIM, 4, 3]),
        # An extra head that should be frozen (simulates ``density`` / ``tc`` / etc. in the real tail).
        RegressionTaskConfig(name="density", data_column="density", dims=[LATENT_DIM, 4, 1]),
    ]
    return FlexibleMultiTaskModel(
        task_configs=tasks,
        encoder_config=enc,
        enable_learnable_loss_balancer=enable_balancer,
    )


def _grad_state(model) -> dict[str, bool]:
    return {name: p.requires_grad for name, p in model.named_parameters()}


def test_freeze_except_freezes_encoder_and_unkept_heads():
    """Encoder + every head NOT in ``keep`` is frozen; kept heads remain trainable."""
    model = _make_model()
    inverse_heads = ("formation_energy", "klat", "material_type")
    freeze_except(model, inverse_heads)

    # Encoder: every param frozen.
    assert all(not p.requires_grad for p in model.encoder.parameters())
    # Kept heads: every param trainable.
    for head in inverse_heads:
        assert all(p.requires_grad for p in model.task_heads[head].parameters()), f"{head!r} should be trainable"
    # Non-kept head (``density``): every param frozen.
    assert all(not p.requires_grad for p in model.task_heads["density"].parameters())


def test_freeze_except_freezes_task_log_sigmas_when_balancer_enabled():
    """The learnable per-task loss-balancer scalars MUST be frozen, otherwise the optimiser
    silently shifts the inverse heads' relative weights during the head-only fine-tune and
    the downstream comparison stops being apples-to-apples."""
    model = _make_model(enable_balancer=True)
    # Sanity check: balancer is on so task_log_sigmas has at least one parameter. ``any()``
    # would unwrap to the scalar's bool (0.0 is falsy) — we want a count check instead.
    assert len(list(model.task_log_sigmas.parameters())) > 0, "fixture must register balancer scalars"
    freeze_except(model, ("formation_energy", "klat", "material_type"))
    assert all(not p.requires_grad for p in model.task_log_sigmas.parameters())


def test_freeze_except_returns_pre_freeze_requires_grad_state():
    """The ``saved`` dict captures the pre-call ``requires_grad`` for every named parameter —
    used by ``_restore_requires_grad`` if a caller wants to roll back. The contract is that the
    returned dict has one entry per ``named_parameters()`` key."""
    model = _make_model()
    pre = _grad_state(model)
    saved = freeze_except(model, ("formation_energy",))
    assert set(saved.keys()) == set(pre.keys())
    # All params were trainable before freezing → saved should reflect that.
    assert all(v is True for v in saved.values())


def test_freeze_except_handles_unknown_keep_head_silently():
    """An unknown ``keep_heads`` entry is *not* an error in this helper — it simply means
    no head matches, and every head ends up frozen. This is the right contract for a low-level
    freeze; the caller (``finetune``) is responsible for validating head names against the
    loaded checkpoint upstream (see ``finetune`` raising on ``missing`` heads)."""
    model = _make_model()
    freeze_except(model, ("not_a_head",))
    for head in model.task_heads.values():
        assert all(not p.requires_grad for p in head.parameters())
