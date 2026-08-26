# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared ``[model]`` / ``[training]`` config sections."""

from __future__ import annotations

import pytest

from foundation_model.models.model_config import OptimizerConfig
from foundation_model.workflows._sections import (
    OptimizerSectionConfig,
    SchedulerSectionConfig,
    TrainingSectionConfig,
    build_training_section,
)


def test_defaults_preserve_the_previously_hardcoded_weight_decays():
    """The four groups' weight decays used to live at call sites; the defaults must not shift."""
    training = build_training_section({})
    assert training.encoder_weight_decay == 1e-2
    assert training.ae_weight_decay == 1e-3
    assert training.head_weight_decay == 1e-5
    assert training.kr_weight_decay == 5e-5


def test_optimizer_config_carries_shared_numerics_to_every_group():
    training = build_training_section(
        {
            "encoder_lr": 1e-3,
            "encoder_weight_decay": 0.02,
            "optimizer": {"betas": [0.8, 0.95], "eps": 1e-8},
            "scheduler": {"factor": 0.3, "patience": 2, "min_lr": 1e-6},
        }
    )
    cfg = training.optimizer_config(lr=training.encoder_lr, weight_decay=training.encoder_weight_decay)
    assert (cfg.lr, cfg.weight_decay) == (1e-3, 0.02)
    assert cfg.betas == (0.8, 0.95)
    assert cfg.eps == 1e-8
    assert (cfg.factor, cfg.patience, cfg.min_lr) == (0.3, 2, 1e-6)
    assert cfg.scheduler_enabled is True


def test_scheduler_can_be_disabled_for_a_flat_learning_rate():
    training = build_training_section({"scheduler": {"enabled": False}})
    cfg = training.optimizer_config(lr=1e-3, weight_decay=0.0)
    assert cfg.scheduler_enabled is False


def test_min_lr_at_or_above_lr_is_rejected():
    """A floor at or above the LR leaves ReduceLROnPlateau unable to reduce — a silent no-op."""
    training = build_training_section({"scheduler": {"min_lr": 1e-3}})
    with pytest.raises(ValueError, match="must be below lr"):
        training.optimizer_config(lr=1e-3, weight_decay=0.0)
    # ...but it is allowed when the scheduler is off, where a constant LR is the intent.
    off = build_training_section({"scheduler": {"enabled": False, "min_lr": 1e-3}})
    assert off.optimizer_config(lr=1e-3, weight_decay=0.0).min_lr == 1e-3


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"encoder_lr": 0.0}, "training.encoder_lr must be > 0"),
        ({"head_lr": -1.0}, "training.head_lr must be > 0"),
        ({"kr_weight_decay": -1e-5}, "training.kr_weight_decay must be >= 0"),
        ({"ae_weight_decay": -1.0}, "training.ae_weight_decay must be >= 0"),
    ],
)
def test_rejects_nonsensical_group_settings(raw, message):
    with pytest.raises(ValueError, match=message):
        build_training_section(raw)


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"betas": [0.9]}, "two numbers"),
        ({"betas": [0.9, 1.0]}, "two numbers"),
        ({"eps": 0.0}, "eps must be > 0"),
    ],
)
def test_optimizer_subsection_validation(raw, message):
    with pytest.raises(ValueError, match=message):
        OptimizerSectionConfig(**raw)


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"mode": "lowest"}, "must be 'min' or 'max'"),
        ({"factor": 1.0}, r"factor must be in \(0, 1\)"),
        ({"factor": 0.0}, r"factor must be in \(0, 1\)"),
        ({"patience": -1}, "patience must be >= 0"),
        ({"min_lr": -1e-6}, "min_lr must be >= 0"),
    ],
)
def test_scheduler_subsection_validation(raw, message):
    with pytest.raises(ValueError, match=message):
        SchedulerSectionConfig(**raw)


def test_inert_lightning_scheduler_keys_are_not_exposed():
    """interval / frequency cannot take effect, so they must not be accepted as config.

    The model steps its schedulers itself in on_train_epoch_end — once per epoch by construction —
    so Lightning's cadence settings never apply. `monitor` IS honoured (looked up in
    trainer.callback_metrics) and is exposed; these two are not.
    """
    for key, value in (("interval", "step"), ("frequency", 2)):
        with pytest.raises(ValueError, match=rf"training\.scheduler.*{key}"):
            build_training_section({"scheduler": {key: value}})


def test_scheduler_monitor_is_exposed_and_reaches_the_optimizer_config():
    training = build_training_section({"scheduler": {"monitor": "val_final_loss"}})
    assert training.optimizer_config(lr=1e-3, weight_decay=0.0).monitor == "val_final_loss"


def test_scheduler_monitor_must_be_non_empty():
    with pytest.raises(ValueError, match="monitor must be a non-empty"):
        build_training_section({"scheduler": {"monitor": ""}})


def test_unknown_keys_are_named_in_the_error():
    with pytest.raises(ValueError, match=r"training.optimizer.*momentum"):
        build_training_section({"optimizer": {"momentum": 0.9}})
    with pytest.raises(ValueError, match=r"training.scheduler.*cooldown"):
        build_training_section({"scheduler": {"cooldown": 3}})


def test_optimizer_config_is_a_plain_optimizer_config():
    cfg = TrainingSectionConfig().optimizer_config(lr=5e-3, weight_decay=1e-3)
    assert isinstance(cfg, OptimizerConfig)
