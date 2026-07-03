# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Shared config sections used by more than one workflow (``[model]`` / ``[training]``)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def reject_unknown(section: str, raw: Mapping[str, Any], known: set[str]) -> None:
    """Raise ``ValueError`` naming any key in ``raw`` outside ``known``."""
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"[{section}]: unknown key(s) {unknown}; allowed keys are {sorted(known)}.")


@dataclass(kw_only=True)
class ModelSectionConfig:
    """``[model]`` — architecture dims shared by pretrain and finetune."""

    latent_dim: int = 128
    encoder_hidden: int = 256
    head_hidden_dim: int = 64
    n_kernel: int = 15

    def __post_init__(self) -> None:
        for name in ("latent_dim", "encoder_hidden", "head_hidden_dim", "n_kernel"):
            if getattr(self, name) < 1:
                raise ValueError(f"model.{name} must be >= 1, got {getattr(self, name)}.")


@dataclass(kw_only=True)
class TrainingSectionConfig:
    """``[training]`` — epochs, early-stop, learning rates, accelerator (pretrain + finetune)."""

    max_epochs: int = 100
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    encoder_lr: float = 5e-3
    head_lr: float = 5e-3
    kr_lr: float = 5e-4
    kr_weight_decay: float = 5e-5
    ae_lr: float = 5e-3
    accelerator: str = "auto"
    devices: int = 1
    seed: int = 2025

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError(f"training.max_epochs must be >= 1, got {self.max_epochs}.")
        if self.early_stop_patience < 1:
            raise ValueError(f"training.early_stop_patience must be >= 1, got {self.early_stop_patience}.")


def build_model_section(raw: Mapping[str, Any]) -> ModelSectionConfig:
    data = dict(raw)
    reject_unknown("model", data, set(ModelSectionConfig.__dataclass_fields__))
    return ModelSectionConfig(**data)


def build_training_section(raw: Mapping[str, Any]) -> TrainingSectionConfig:
    data = dict(raw)
    reject_unknown("training", data, set(TrainingSectionConfig.__dataclass_fields__))
    return TrainingSectionConfig(**data)
