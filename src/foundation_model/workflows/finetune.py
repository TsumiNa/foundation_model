# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``fm finetune`` — frozen-encoder fine-tuning of selected task heads.

Rebuilds the checkpoint's model from the :class:`TaskCatalog`, loads its weights with
``strict=False``, optionally adds heads absent from the checkpoint, then fine-tunes only the
requested heads. Freeze policy (rewrite of ``finetune_inverse_heads.freeze_except`` with one
intentional change): the encoder and every non-target head are frozen, **except the built-in
autoencoder head which always stays trainable** (at ``ae_lr``); ``task_log_sigmas`` are frozen.
Non-target heads are disabled for the fit and restored before saving so the checkpoint keeps
every head.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from loguru import logger

from ._engine import (
    AE_NAME,
    DropLastTrainCompoundDataModule,
    build_empty_model,
    build_head_config,
    build_trainer_extras,
    evaluate_task,
)
from ._sections import (
    ModelSectionConfig,
    TrainingSectionConfig,
    build_model_section,
    build_training_section,
    reject_unknown,
)
from .recording import RunRecorder, load_checkpoint_state
from .task_catalog import TaskCatalog, TaskCatalogConfig, build_task_catalog_config

_FINETUNE_ROOT_KEYS = {"data", "descriptor", "datasets", "tasks", "model", "training", "finetune", "output"}
_CATALOG_KEYS = {"data", "descriptor", "datasets", "tasks"}


@dataclass(kw_only=True)
class FinetuneConfig:
    catalog: TaskCatalogConfig
    model: ModelSectionConfig
    training: TrainingSectionConfig
    checkpoint: Path
    tasks: list[str]
    output_dir: Path
    epochs: int = 20
    freeze_encoder: bool = True
    add_new_tasks: bool = True

    def __post_init__(self) -> None:
        self.checkpoint = Path(self.checkpoint)
        self.output_dir = Path(self.output_dir)
        if not self.tasks:
            raise ValueError("finetune.tasks must be non-empty.")
        if self.epochs < 1:
            raise ValueError(f"finetune.epochs must be >= 1, got {self.epochs}.")
        catalog_tasks = {t.name for t in self.catalog.tasks}
        unknown = [t for t in self.tasks if t not in catalog_tasks]
        if unknown:
            raise ValueError(f"finetune.tasks references unknown task(s): {unknown}.")


def build_finetune_config(
    raw: Mapping[str, Any], *, output_dir: str | Path | None = None, checkpoint: str | Path | None = None
) -> FinetuneConfig:
    """Normalize a parsed-TOML tree into a :class:`FinetuneConfig`."""

    reject_unknown("<root>", raw, _FINETUNE_ROOT_KEYS)
    catalog = build_task_catalog_config({k: raw[k] for k in _CATALOG_KEYS if k in raw})
    model = build_model_section(raw.get("model", {}))
    training = build_training_section(raw.get("training", {}))

    ft_raw = dict(raw.get("finetune", {}))
    reject_unknown("finetune", ft_raw, {"checkpoint", "tasks", "epochs", "freeze_encoder", "add_new_tasks"})

    resolved_checkpoint = checkpoint if checkpoint is not None else ft_raw.get("checkpoint")
    if resolved_checkpoint is None:
        raise ValueError("checkpoint must be given via --checkpoint or [finetune].checkpoint.")
    resolved_output = output_dir if output_dir is not None else raw.get("output", {}).get("dir")
    if resolved_output is None:
        raise ValueError("output directory must be given via --output-dir or [output].dir.")

    return FinetuneConfig(
        catalog=catalog,
        model=model,
        training=training,
        checkpoint=Path(resolved_checkpoint),
        tasks=list(ft_raw.get("tasks", [])),
        output_dir=Path(resolved_output),
        epochs=int(ft_raw.get("epochs", 20)),
        freeze_encoder=bool(ft_raw.get("freeze_encoder", True)),
        add_new_tasks=bool(ft_raw.get("add_new_tasks", True)),
    )


def apply_freeze_policy(model: Any, target_tasks: Sequence[str], *, freeze_encoder: bool) -> dict[str, bool]:
    """Freeze the encoder (optional) + non-target heads (except the AE) + task_log_sigmas.

    Returns the resulting ``{param_name: requires_grad}`` snapshot (taken AFTER applying the
    policy) for provenance. The autoencoder head (``"__reconstruction__"``) always stays
    trainable — the intentional behaviour change vs. the legacy ``freeze_except`` (which froze it).
    """
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    keep = set(target_tasks) | {AE_NAME}  # AE always trainable
    for head_name, head in model.task_heads.items():
        trainable = head_name in keep
        for p in head.parameters():
            p.requires_grad_(trainable)
    for p in model.task_log_sigmas.parameters():
        p.requires_grad_(False)
    return {name: p.requires_grad for name, p in model.named_parameters()}


class _FrozenEncoderEval(Callback):
    """Keep a frozen encoder in eval mode during fit.

    Freezing ``requires_grad`` stops gradients but Lightning still puts the module in train mode,
    so the encoder's ``BatchNorm1d`` ``running_mean``/``running_var`` would keep updating and the
    saved encoder would differ. Forcing eval mode each train epoch freezes those buffers too.
    """

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        pl_module.encoder.eval()


def _task_names_from_state(state_dict: Mapping[str, Any]) -> list[str]:
    """Head task names present in a checkpoint state_dict (excludes the AE head)."""
    names: list[str] = []
    for key in state_dict:
        if key.startswith("task_heads."):
            name = key.split(".", 2)[1]
            if name != AE_NAME and name not in names:
                names.append(name)
    return names


def run(cfg: FinetuneConfig, recorder: RunRecorder | None = None) -> dict[str, Any]:
    """Fine-tune ``cfg.tasks`` on top of ``cfg.checkpoint``; return the finetune summary dict."""

    catalog = TaskCatalog(cfg.catalog)
    owns_recorder = recorder is None
    rec = recorder or RunRecorder(cfg.output_dir)
    seed_everything(cfg.training.seed, workers=True)

    try:
        state = load_checkpoint_state(cfg.checkpoint)
        ckpt_tasks = list(state.get("task_sequence") or _task_names_from_state(state["model"]))
        catalog_tasks = {t.name for t in cfg.catalog.tasks}
        missing_from_catalog = [t for t in ckpt_tasks if t not in catalog_tasks]
        if missing_from_catalog:
            raise ValueError(
                f"Checkpoint tasks {missing_from_catalog} are not defined in the catalog "
                f"(available: {sorted(catalog_tasks)})."
            )

        model = build_empty_model(catalog, cfg.model, cfg.training)
        for name in ckpt_tasks:
            model.add_task(build_head_config(catalog, cfg.model, cfg.training, name, masking_ratio=1.0))
        incompatible = model.load_state_dict(state["model"], strict=False)
        if incompatible.missing_keys:
            logger.info(
                f"load_state_dict missing keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys[:8]}"
            )
        if incompatible.unexpected_keys:
            logger.info(
                f"load_state_dict unexpected keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys[:8]}"
            )

        added_tasks = [t for t in cfg.tasks if t not in model.task_heads]
        if added_tasks and not cfg.add_new_tasks:
            raise ValueError(
                f"Heads {added_tasks} are absent from the checkpoint and add_new_tasks=False "
                f"(checkpoint heads: {sorted(model.task_heads)})."
            )
        for name in added_tasks:
            logger.info(f"Adding new head '{name}' (absent from checkpoint).")
            model.add_task(build_head_config(catalog, cfg.model, cfg.training, name, masking_ratio=1.0))

        apply_freeze_policy(model, cfg.tasks, freeze_encoder=cfg.freeze_encoder)
        frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
        trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)

        # Disable non-target supervised heads for the fit (keep the AE active + trainable); restore
        # before saving so the checkpoint retains every head.
        disabled = [n for n in list(model.task_heads) if n not in cfg.tasks and n != AE_NAME]
        if disabled:
            model.disable_task(*disabled)

        datamodule = catalog.build_datamodule(
            cfg.tasks,
            masking_ratios={t: 1.0 for t in cfg.tasks},
            datamodule_cls=DropLastTrainCompoundDataModule,  # drop size-1 tail batch (BatchNorm)
        )
        datamodule.setup("fit")
        test_keys = _test_keys(datamodule)

        before = {
            name: evaluate_task(
                model, catalog, name, rec, rec.paths.training / "before_finetune", is_new=False, test_keys=test_keys
            )
            for name in cfg.tasks
        }

        callbacks, loggers, enable_ckpt = build_trainer_extras(
            cfg.training,
            log_dir=rec.paths.root / "logs",
            ckpt_dir=rec.paths.training / "finetune" / "lightning",
            run_name="finetune",
        )
        if cfg.freeze_encoder:
            callbacks.append(_FrozenEncoderEval())  # keep frozen encoder's BatchNorm buffers fixed
        trainer = Trainer(
            max_epochs=cfg.epochs,
            accelerator=cfg.training.accelerator,
            devices=cfg.training.devices,
            logger=loggers,
            enable_checkpointing=enable_ckpt,
            enable_progress_bar=False,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule=datamodule)

        if disabled:
            model.enable_task(*disabled)

        eval_dir = rec.paths.training / "finetune"
        after: dict[str, dict[str, float]] = {}
        for name in cfg.tasks:
            after[name] = evaluate_task(model, catalog, name, rec, eval_dir, is_new=True, test_keys=test_keys)

        rec.save_final_model(model, ckpt_tasks + added_tasks, _task_spec_dump(catalog, ckpt_tasks + added_tasks))
        summary = {
            "from_checkpoint": str(cfg.checkpoint),
            "finetuned_heads": list(cfg.tasks),
            "added_tasks": added_tasks,
            "checkpoint_tasks": ckpt_tasks,
            "epochs": cfg.epochs,
            "epochs_run": trainer.current_epoch,
            "freeze_encoder": cfg.freeze_encoder,
            "frozen_params": frozen,
            "trainable_params": trainable,
            "metrics_before": before,
            "metrics_after": after,
        }
        (rec.paths.training / "finetune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(
            f"Fine-tuned {cfg.tasks} for {trainer.current_epoch} epochs → {rec.paths.training / 'final_model.pt'}"
        )
        return summary
    finally:
        if owns_recorder:
            rec.close()


def _test_keys(datamodule: Any) -> set[str] | None:
    if datamodule.split_series is None:
        return None
    resolved = datamodule.split_series
    return set(resolved.index[resolved == "test"].astype(str))


def _task_spec_dump(catalog: TaskCatalog, task_names: Sequence[str]) -> dict[str, Any]:
    dump: dict[str, Any] = {}
    for name in task_names:
        spec = catalog.task_spec(name)
        dump[name] = {"kind": spec.kind.value, "column": spec.column, "source": spec.dataset}
    return dump
