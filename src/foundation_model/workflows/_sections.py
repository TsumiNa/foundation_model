# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Shared config sections used by more than one workflow (``[model]`` / ``[training]``)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from foundation_model.models.model_config import OptimizerConfig

_MODES = {"min", "max"}


def reject_unknown(section: str, raw: Mapping[str, Any], known: set[str]) -> None:
    """Raise ``ValueError`` naming any key in ``raw`` outside ``known``."""
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"[{section}]: unknown key(s) {unknown}; allowed keys are {sorted(known)}.")


def validate_positive_int(where: str, value: Any) -> None:
    """Require a positive ``int`` (``bool`` rejected — it is an ``int`` subclass).

    Guards config fields fed to int-only APIs (e.g. ``np.linspace`` kernel counts, layer dims) so a
    TOML float like ``128.5`` / ``10.0`` fails at config time instead of being silently coerced.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be a positive int, got {value!r}.")


def validate_devices(value: Any) -> None:
    """Lightning-compatible ``Trainer(devices=...)``: an int count (``-1`` = all, else ``>= 1``), a
    non-empty list of non-negative device indices (``[1, 3]``), or a non-empty string (``"auto"`` /
    ``"1,3"`` / ``"0-3"``). Lightning validates that the indices/string map to real devices at fit.
    """
    if isinstance(value, bool):  # bool is an int subclass — reject it explicitly
        raise ValueError(f"training.devices must be an int, list of ints, or str, got bool {value!r}.")
    if isinstance(value, int):
        if value == -1 or value >= 1:
            return
        raise ValueError(f"training.devices int must be -1 (all) or >= 1, got {value}.")
    if isinstance(value, list):
        if not value or any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in value):
            raise ValueError(f"training.devices list must be non-empty non-negative int indices, got {value!r}.")
        return
    if isinstance(value, str):
        if not value.strip():
            raise ValueError('training.devices string must be non-empty (e.g. "auto", "1,3", "0-3").')
        return
    raise ValueError(f"training.devices must be an int, list of ints, or str, got {value!r}.")


def validate_hidden_dims(where: str, dims: Any, *, allow_empty: bool = False) -> None:
    """Validate a hidden-layer width list: a (possibly empty) list of positive ints.

    Shared by ``[model]`` (``ModelSectionConfig``) and the per-task ``[[tasks]]`` overrides so both
    reject e.g. ``[0]``, ``[1.5]``, ``[true]`` or a bare int with the same message.
    """
    if not isinstance(dims, list):
        raise ValueError(f"{where} must be a list of positive ints, got {dims!r}.")
    if not dims and not allow_empty:
        raise ValueError(f"{where} must have at least one hidden layer.")
    for d in dims:
        if isinstance(d, bool) or not isinstance(d, int) or d < 1:
            raise ValueError(f"{where} must be a list of positive ints, got {dims!r}.")


@dataclass(kw_only=True)
class ModelSectionConfig:
    """``[model]`` — architecture defaults shared by every subcommand.

    The ``*_hidden_dims`` lists are the *hidden* layer widths; the input dim (descriptor width for
    the encoder, ``latent_dim`` for the heads) is prepended automatically, and the output dim (1 for
    regression, ``num_classes`` for classification, the kernel projection for KR) is appended. So
    ``encoder_hidden_dims = [256]`` builds ``descriptor_dim → 256 → latent_dim``. Each ``[[tasks]]``
    entry may override its own head's dims (``hidden_dims`` for reg/clf; ``x_hidden_dims`` /
    ``t_hidden_dims`` / ``n_kernel`` for KR); unset tasks fall back to these defaults.
    """

    latent_dim: int = 128
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [256])
    head_hidden_dims: list[int] = field(default_factory=lambda: [64])
    kr_x_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    kr_t_hidden_dims: list[int] = field(default_factory=lambda: [16, 8])
    n_kernel: int = 15

    def __post_init__(self) -> None:
        validate_positive_int("model.latent_dim", self.latent_dim)
        validate_positive_int("model.n_kernel", self.n_kernel)
        # encoder_hidden_dims may be empty (a shallow descriptor_dim → latent_dim encoder); the head
        # branches need at least one hidden layer.
        validate_hidden_dims("model.encoder_hidden_dims", self.encoder_hidden_dims, allow_empty=True)
        validate_hidden_dims("model.head_hidden_dims", self.head_hidden_dims)
        validate_hidden_dims("model.kr_x_hidden_dims", self.kr_x_hidden_dims)
        validate_hidden_dims("model.kr_t_hidden_dims", self.kr_t_hidden_dims)


@dataclass(kw_only=True)
class EarlyStoppingConfig:
    """``[training.early_stopping]`` — a subset of Lightning's ``EarlyStopping``."""

    enabled: bool = True
    monitor: str = "val_final_loss"
    mode: str = "min"
    patience: int = 8
    min_delta: float = 1e-4

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"training.early_stopping.mode must be 'min' or 'max', got {self.mode!r}.")
        if self.patience < 1:
            raise ValueError(f"training.early_stopping.patience must be >= 1, got {self.patience}.")


@dataclass(kw_only=True)
class CheckpointConfig:
    """``[training.checkpoint]`` — a subset of Lightning's ``ModelCheckpoint``.

    Off by default: the RunRecorder already writes replay-schema checkpoints that the finetune/
    inverse/predict flows consume. Enable to also emit Lightning ``.ckpt`` files (best/last).
    """

    enabled: bool = False
    monitor: str = "val_final_loss"
    mode: str = "min"
    save_top_k: int = 1
    save_last: bool = False
    filename: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"training.checkpoint.mode must be 'min' or 'max', got {self.mode!r}.")


@dataclass(kw_only=True)
class OptimizerSectionConfig:
    """``[training.optimizer]`` — AdamW numerics shared by every parameter group.

    Per-group learning rates and weight decays stay on ``[training]`` next to each other
    (``encoder_lr`` / ``encoder_weight_decay`` …); only the terms that are genuinely global live
    here.
    """

    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if len(self.betas) != 2 or any(not isinstance(b, (int, float)) or not 0.0 <= b < 1.0 for b in self.betas):
            raise ValueError(f"training.optimizer.betas must be two numbers in [0, 1), got {self.betas!r}.")
        if self.eps <= 0:
            raise ValueError(f"training.optimizer.eps must be > 0, got {self.eps}.")


@dataclass(kw_only=True)
class SchedulerSectionConfig:
    """``[training.scheduler]`` — ``ReduceLROnPlateau``, applied to every parameter group.

    ``patience`` counts **epochs**. Under manual optimization Lightning does not drive schedulers,
    so the model steps them itself in ``on_train_epoch_end`` — once per epoch, on the
    epoch-aggregated ``monitor`` metric. (Before 0.3.1 it stepped inside ``training_step``, i.e.
    once per *batch*, which made ``patience`` count batches and drove the LR to ``min_lr`` inside
    the first epoch on a 24k-row task.)

    ``interval`` / ``frequency`` are not exposed: stepping is per-epoch by construction, and
    Lightning's own scheduler cadence does not apply under manual optimization.

    ``min_lr`` is a FLOOR on the reduced learning rate. A low configured LR plus this floor leaves
    the scheduler almost no room to anneal — at the default ``1e-4``, an ``lr`` of ``2e-4`` can only
    be halved once. ``OptimizerConfig`` rejects the degenerate case (``min_lr >= lr``) outright.
    """

    enabled: bool = True
    mode: Literal["min", "max"] = "min"
    factor: float = 0.5
    patience: int = 5  # in epochs
    min_lr: float = 1e-4
    monitor: str = "train_final_loss_epoch"  # must be logged with on_epoch=True during training

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"training.scheduler.mode must be 'min' or 'max', got {self.mode!r}.")
        if not 0.0 < self.factor < 1.0:
            raise ValueError(f"training.scheduler.factor must be in (0, 1), got {self.factor}.")
        if self.patience < 0:
            raise ValueError(f"training.scheduler.patience must be >= 0, got {self.patience}.")
        if self.min_lr < 0:
            raise ValueError(f"training.scheduler.min_lr must be >= 0, got {self.min_lr}.")
        if not self.monitor:
            raise ValueError("training.scheduler.monitor must be a non-empty metric name.")


@dataclass(kw_only=True)
class LoggingConfig:
    """``[training.logging]`` — enable Lightning's ``CSVLogger`` / ``TensorBoardLogger``."""

    csv: bool = False
    tensorboard: bool = False


@dataclass(kw_only=True)
class TrainingSectionConfig:
    """``[training]`` — epochs, per-group optimizer settings, accelerator + Lightning callbacks.

    There are four parameter groups (shared encoder, regression/classification heads,
    kernel-regression heads, the always-on autoencoder head) and each gets its own AdamW instance.
    Every group's learning rate AND weight decay is configurable here; before this section grew
    them, three of the four weight decays were hard-coded at call sites spanning three orders of
    magnitude (1e-2 encoder / 1e-3 AE / 1e-5 reg+clf), which made them invisible to tuning.
    """

    max_epochs: int = 100
    encoder_lr: float = 5e-3
    encoder_weight_decay: float = 1e-2
    head_lr: float = 5e-3
    head_weight_decay: float = 1e-5
    kr_lr: float = 5e-4
    kr_weight_decay: float = 5e-5
    ae_lr: float = 5e-3
    ae_weight_decay: float = 1e-3
    # Kendall/Gal/Cipolla (CVPR 2018) uncertainty weighting: learn one log sigma per supervised
    # task and combine losses as sum_i [ 0.5 * exp(-2 log sigma_i) * L_i + log sigma_i ]. It exists
    # to stop multi-task training from collapsing onto whichever tasks descend fastest, which is a
    # live risk here — the 24-task sequence spans three orders of magnitude in label count.
    #
    # The model has implemented this since before this section existed, but nothing ever routed a
    # value to it, so it has never been switched on in any run. Exposing it does not turn it on;
    # it makes the comparison runnable.
    learnable_loss_balancer: bool = False
    accelerator: str = "auto"
    # Passed straight to Lightning's Trainer(devices=...): "auto" (all devices for the accelerator),
    # an int count (-1 = all), a list of device indices ([1, 3]), or a string ("1,3" / "0-3").
    devices: int | list[int] | str = "auto"
    seed: int = 2025
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    optimizer: OptimizerSectionConfig = field(default_factory=OptimizerSectionConfig)
    scheduler: SchedulerSectionConfig = field(default_factory=SchedulerSectionConfig)

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError(f"training.max_epochs must be >= 1, got {self.max_epochs}.")
        validate_devices(self.devices)
        for group in ("encoder", "head", "kr", "ae"):
            lr = getattr(self, f"{group}_lr")
            if lr <= 0:
                raise ValueError(f"training.{group}_lr must be > 0, got {lr}.")
            if getattr(self, f"{group}_weight_decay") < 0:
                raise ValueError(
                    f"training.{group}_weight_decay must be >= 0, got {getattr(self, f'{group}_weight_decay')}."
                )

    def optimizer_config(self, *, lr: float, weight_decay: float) -> "OptimizerConfig":
        """One parameter group's :class:`OptimizerConfig`, built from this section.

        Single place where ``[training]`` becomes an optimizer, so the four groups cannot drift
        apart on anything except the lr/weight-decay pair that distinguishes them.
        """
        from foundation_model.models.model_config import OptimizerConfig

        return OptimizerConfig(
            lr=lr,
            weight_decay=weight_decay,
            eps=self.optimizer.eps,
            betas=(self.optimizer.betas[0], self.optimizer.betas[1]),
            scheduler_enabled=self.scheduler.enabled,
            mode=self.scheduler.mode,
            factor=self.scheduler.factor,
            patience=self.scheduler.patience,
            min_lr=self.scheduler.min_lr,
            monitor=self.scheduler.monitor,
        )


def build_model_section(raw: Mapping[str, Any]) -> ModelSectionConfig:
    data = dict(raw)
    reject_unknown("model", data, set(ModelSectionConfig.__dataclass_fields__))
    return ModelSectionConfig(**data)


def build_training_section(raw: Mapping[str, Any]) -> TrainingSectionConfig:
    data = dict(raw)
    reject_unknown("training", data, set(TrainingSectionConfig.__dataclass_fields__))

    es_raw = dict(data.pop("early_stopping", {}))
    reject_unknown("training.early_stopping", es_raw, set(EarlyStoppingConfig.__dataclass_fields__))
    ckpt_raw = dict(data.pop("checkpoint", {}))
    reject_unknown("training.checkpoint", ckpt_raw, set(CheckpointConfig.__dataclass_fields__))
    log_raw = dict(data.pop("logging", {}))
    reject_unknown("training.logging", log_raw, set(LoggingConfig.__dataclass_fields__))
    opt_raw = dict(data.pop("optimizer", {}))
    reject_unknown("training.optimizer", opt_raw, set(OptimizerSectionConfig.__dataclass_fields__))
    sched_raw = dict(data.pop("scheduler", {}))
    reject_unknown("training.scheduler", sched_raw, set(SchedulerSectionConfig.__dataclass_fields__))

    return TrainingSectionConfig(
        **data,
        early_stopping=EarlyStoppingConfig(**es_raw),
        checkpoint=CheckpointConfig(**ckpt_raw),
        logging=LoggingConfig(**log_raw),
        optimizer=OptimizerSectionConfig(**opt_raw),
        scheduler=SchedulerSectionConfig(**sched_raw),
    )
