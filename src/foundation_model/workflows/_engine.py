# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Shared engine internals for the training workflows (pretrain / finetune).

Model construction (``build_empty_model`` / ``build_head_config``), the ``drop_last`` train
datamodule, and per-head evaluation (``evaluate_task`` + dumps/plots) live here so both engines
use one implementation. Only :class:`RunRecorder` writes files.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import MLPEncoderConfig, OptimizerConfig

from . import plots
from ._sections import ModelSectionConfig, TrainingSectionConfig
from .recording import RunRecorder
from .task_catalog import TaskCatalog, TaskConfig, TaskKind

# Reserved built-in autoencoder head name (see FlexibleMultiTaskModel).
AE_NAME = "__reconstruction__"
# Legacy weight decay for regression / classification heads (kernel heads use kr_weight_decay).
_HEAD_WEIGHT_DECAY = 1e-5


def task_names_from_state(state_dict: Mapping[str, Any]) -> list[str]:
    """Task-head names in a state dict (``task_heads.<name>.*``), in first-seen order, minus the AE.

    Used when a checkpoint carries no ``task_sequence`` list (e.g. a bare/Lightning state dict).
    """
    names: list[str] = []
    for key in state_dict:
        if key.startswith("task_heads."):
            name = key.split(".", 2)[1]
            if name != AE_NAME and name not in names:
                names.append(name)
    return names


def checkpoint_task_order(state: Mapping[str, Any]) -> list[str]:
    """Every task head present in a loaded checkpoint, ordered by its ``task_sequence`` when given.

    The state dict is the ground truth for which heads *exist*; ``task_sequence`` only records order
    — and a pretrain **step** checkpoint stores just that step's active subset — so heads that are in
    the weights but missing from ``task_sequence`` are appended rather than silently dropped.
    """
    in_state = task_names_from_state(state["model"])
    ordered = [t for t in (state.get("task_sequence") or []) if t in in_state]
    return ordered + [h for h in in_state if h not in ordered]


def build_encoder_config(model: ModelSectionConfig, descriptor_dim: int) -> MLPEncoderConfig:
    """MLP encoder ``descriptor_dim → *encoder_hidden_dims → latent_dim`` from ``[model]``."""
    return MLPEncoderConfig(hidden_dims=[descriptor_dim, *model.encoder_hidden_dims, model.latent_dim])


def build_empty_model(
    catalog: TaskCatalog, model: ModelSectionConfig, training: TrainingSectionConfig
) -> FlexibleMultiTaskModel:
    """Bare model (AE head only). The AE always trains; only ``ae_lr`` is configurable."""
    built = FlexibleMultiTaskModel(
        task_configs=[],
        encoder_config=build_encoder_config(model, catalog.descriptor_dim),
        enable_autoencoder=True,
        shared_block_optimizer=OptimizerConfig(lr=training.encoder_lr, weight_decay=1e-2),
    )
    if AE_NAME in built.task_configs_map:
        built.task_configs_map[AE_NAME].optimizer = OptimizerConfig(lr=training.ae_lr)
    return built


def build_model_for_checkpoint(
    catalog: TaskCatalog, model: ModelSectionConfig, task_names: list[str], *, lr: float = 5e-3
) -> FlexibleMultiTaskModel:
    """AE-enabled model with one head per checkpoint task, ready for ``load_state_dict``.

    Shared by ``fm predict`` / ``fm inverse``, which only need the architecture to match the
    checkpoint (weights are then loaded); ``lr`` is a placeholder as no optimization runs.
    """
    built = FlexibleMultiTaskModel(
        task_configs=[],
        encoder_config=build_encoder_config(model, catalog.descriptor_dim),
        enable_autoencoder=True,
        shared_block_optimizer=OptimizerConfig(lr=lr, weight_decay=1e-2),
    )
    for name in task_names:
        built.add_task(
            catalog.build_task_config(
                name,
                latent_dim=model.latent_dim,
                head_hidden_dims=model.head_hidden_dims,
                kr_x_hidden_dims=model.kr_x_hidden_dims,
                kr_t_hidden_dims=model.kr_t_hidden_dims,
                n_kernel=model.n_kernel,
                lr=lr,
            )
        )
    return built


def build_trainer_extras(
    training: TrainingSectionConfig, *, log_dir: Path, ckpt_dir: Path, run_name: str
) -> tuple[list[Callback], list[Logger] | bool, bool]:
    """Build Lightning callbacks + loggers from ``[training]`` config.

    Returns ``(callbacks, logger_arg, enable_checkpointing)`` for ``Trainer(...)``. EarlyStopping is
    on by default; ModelCheckpoint and the CSV/TensorBoard loggers are opt-in (config-driven) so the
    default flow keeps the RunRecorder as the sole checkpoint/log writer.
    """
    callbacks: list[Callback] = []
    es = training.early_stopping
    if es.enabled:
        callbacks.append(EarlyStopping(monitor=es.monitor, mode=es.mode, patience=es.patience, min_delta=es.min_delta))
    ckpt = training.checkpoint
    if ckpt.enabled:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor=ckpt.monitor,
                mode=ckpt.mode,
                save_top_k=ckpt.save_top_k,
                save_last=ckpt.save_last,
                filename=ckpt.filename,
            )
        )
    loggers: list[Logger] = []
    if training.logging.csv:
        loggers.append(CSVLogger(save_dir=str(log_dir), name=run_name))
    if training.logging.tensorboard:
        loggers.append(TensorBoardLogger(save_dir=str(log_dir), name=run_name))
    return callbacks, (loggers or False), ckpt.enabled


def build_head_config(
    catalog: TaskCatalog,
    model: ModelSectionConfig,
    training: TrainingSectionConfig,
    name: str,
    *,
    masking_ratio: float = 1.0,
    init_from_data: bool = True,
) -> TaskConfig:
    """Build a task's head config with the right LR/weight-decay for its kind."""
    spec = catalog.task_spec(name)
    if spec.kind is TaskKind.KERNEL_REGRESSION:
        lr, weight_decay = training.kr_lr, training.kr_weight_decay
    else:
        lr, weight_decay = training.head_lr, _HEAD_WEIGHT_DECAY
    return catalog.build_task_config(
        name,
        latent_dim=model.latent_dim,
        head_hidden_dims=model.head_hidden_dims,
        kr_x_hidden_dims=model.kr_x_hidden_dims,
        kr_t_hidden_dims=model.kr_t_hidden_dims,
        n_kernel=model.n_kernel,
        lr=lr,
        weight_decay=weight_decay,
        masking_ratio=masking_ratio,
        init_from_data=init_from_data,
    )


class DropLastTrainCompoundDataModule(CompoundDataModule):
    """``CompoundDataModule`` whose train loader drops the final partial batch.

    Guards against BatchNorm1d crashing on a size-1 tail batch. Only the train loader is
    affected; val/test/predict keep every held-out row. A non-default sampler (e.g. a
    ``DistributedSampler``) is preserved rather than replaced with ``shuffle=True``.
    """

    def train_dataloader(self) -> DataLoader | None:  # type: ignore[override]
        base = super().train_dataloader()
        if base is None:
            return None
        kwargs: dict[str, Any] = dict(
            batch_size=base.batch_size,
            num_workers=base.num_workers,
            pin_memory=base.pin_memory,
            collate_fn=base.collate_fn,
            drop_last=True,
        )
        sampler = base.sampler
        if isinstance(sampler, (RandomSampler, SequentialSampler)):
            return DataLoader(base.dataset, shuffle=True, **kwargs)
        return DataLoader(base.dataset, sampler=sampler, **kwargs)


def as_float_array(cell: Any) -> np.ndarray:
    if isinstance(cell, str):
        cell = ast.literal_eval(cell)
    return np.asarray(cell, dtype=float).ravel()


def test_rows(catalog: TaskCatalog, name: str, test_keys: set[str] | None) -> list[str]:
    """Compositions with a non-NaN target on the resolved test split."""
    spec = catalog.task_spec(name)
    frame = catalog.task_frames([name])[name]
    mask = frame[spec.column].notna()
    if test_keys is not None:
        mask &= frame.index.isin(test_keys)
    elif "split" in frame.columns:
        mask &= frame["split"] == "test"
    return list(frame.index[mask])


def descriptor_tensor(catalog: TaskCatalog, comps: list[str], device: torch.device) -> tuple[torch.Tensor, list[str]]:
    desc = catalog.descriptor_fn()(comps)
    kept = [c for c in comps if c in desc.index]
    tensor = torch.tensor(desc.loc[kept].values, dtype=torch.float32, device=device)
    return tensor, kept


def _kr_long_frame(
    comps: list[str], t_list: list[np.ndarray], true_parts: list[np.ndarray], pred: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offset = 0
    for comp, t_arr, y_true in zip(comps, t_list, true_parts):
        n = int(y_true.size)
        for k in range(n):
            rows.append(
                {"composition": comp, "t": float(t_arr[k]), "true": float(y_true[k]), "pred": float(pred[offset + k])}
            )
        offset += n
    return pd.DataFrame(rows)


def fig_rel(recorder: RunRecorder, step_dir: Path, filename: str) -> str:
    return str((step_dir / filename).relative_to(recorder.paths.root))


def evaluate_task(
    model: FlexibleMultiTaskModel,
    catalog: TaskCatalog,
    name: str,
    recorder: RunRecorder,
    step_dir: Path,
    *,
    is_new: bool,
    test_keys: set[str] | None,
) -> dict[str, float]:
    """Evaluate one head on the fixed test split; dump parquet + metrics (+ plot if new)."""

    spec = catalog.task_spec(name)
    model.eval()
    device = next(model.parameters()).device
    comps = test_rows(catalog, name, test_keys)
    if not comps:
        return {"primary": float("nan"), "samples": 0}
    frame = catalog.task_frames([name])[name]
    head = model.task_heads[name]

    with torch.no_grad():
        if spec.kind in (TaskKind.REGRESSION, TaskKind.CLASSIFICATION):
            x, comps = descriptor_tensor(catalog, comps, device)
            if not comps:
                return {"primary": float("nan"), "samples": 0}
            h = torch.tanh(model.encoder(x))
            if spec.kind is TaskKind.REGRESSION:
                pred = catalog.inverse_transform(name, head(h).squeeze(-1).cpu().numpy())
                true = catalog.inverse_transform(name, frame.loc[comps, spec.column].astype(float).to_numpy())
                r2 = float(r2_score(true, pred))
                metric = {"r2": r2, "mae": float(mean_absolute_error(true, pred)), "samples": len(comps), "primary": r2}
                recorder.dump_predictions(
                    step_dir, name, pd.DataFrame({"composition": comps, "true": true, "pred": pred})
                )
                recorder.dump_metrics(step_dir, name, metric)
                if is_new:
                    fig = plots.plot_parity(true, pred, r2=r2, title=name)
                    recorder.save_figure(fig_rel(recorder, step_dir, f"{name}_parity.png"), fig)
                    plots.plt.close(fig)
                return metric
            logits = head(h)
            pred = logits.argmax(dim=-1).cpu().numpy()
            true = frame.loc[comps, spec.column].astype(int).to_numpy()
            acc = float(accuracy_score(true, pred))
            metric = {
                "accuracy": acc,
                "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
                "samples": len(comps),
                "primary": acc,
            }
            recorder.dump_predictions(step_dir, name, pd.DataFrame({"composition": comps, "true": true, "pred": pred}))
            recorder.dump_metrics(step_dir, name, metric)
            if is_new:
                assert spec.num_classes is not None
                fig = plots.plot_confusion(
                    true,
                    pred,
                    num_classes=spec.num_classes,
                    acc=acc,
                    title=name,
                    special_material_type=(name == "material_type"),
                )
                recorder.save_figure(fig_rel(recorder, step_dir, f"{name}_confusion.png"), fig)
                plots.plt.close(fig)
            return metric

        # kernel regression
        assert spec.t_column is not None
        available = set(catalog.descriptor_fn()(comps).index)
        keep: list[str] = []
        t_list: list[np.ndarray] = []
        true_parts: list[np.ndarray] = []
        for comp in comps:
            if comp not in available:
                continue
            y_arr = as_float_array(frame.at[comp, spec.column])
            t_arr = as_float_array(frame.at[comp, spec.t_column])
            if y_arr.size == 0 or y_arr.size != t_arr.size:
                continue
            keep.append(comp)
            t_list.append(t_arr)
            true_parts.append(y_arr)
        if not keep:
            return {"primary": float("nan"), "samples": 0}
        xk, _ = descriptor_tensor(catalog, keep, device)
        h_k = torch.tanh(model.encoder(xk))
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=device) for t in t_list]
        expanded_h, expanded_t = model._expand_for_kernel_regression(h_k, t_tensors)
        pred = catalog.inverse_transform(name, head(expanded_h, t=expanded_t).squeeze(-1).cpu().numpy())
        true = catalog.inverse_transform(name, np.concatenate(true_parts))
        r2 = float(r2_score(true, pred))
        metric = {
            "r2": r2,
            "mae": float(mean_absolute_error(true, pred)),
            "samples": len(keep),
            "points": int(true.size),
            "primary": r2,
        }
        recorder.dump_predictions(step_dir, name, _kr_long_frame(keep, t_list, true_parts, pred))
        recorder.dump_metrics(step_dir, name, metric)
        if is_new:
            kr_fig = plots.plot_kr_sequences(keep, t_list, true_parts, pred, title=name)
            if kr_fig is not None:
                recorder.save_figure(fig_rel(recorder, step_dir, f"{name}_sequences.png"), kr_fig)
                plots.plt.close(kr_fig)
        return metric
