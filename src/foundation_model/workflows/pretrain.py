# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``fm pretrain`` — continual-rehearsal pre-training engine with an n-runs sweep.

Migrated from ``scripts/continual_rehearsal_full.py`` (``ContinualRehearsalFullRunner.run`` /
``_build_empty_model`` / ``_evaluate_task``) with three deliberate changes vs. the legacy runner:

1. **Rehearsal interval** (new): on step ``s`` (1-based) already-learned tasks join training only
   when ``s % interval == 0``; other steps train the new task + AE alone. ``interval == 1``
   reproduces the legacy "every step replays every old task" behaviour. Evaluation still covers
   **all** learned heads at every step (the forgetting trajectory is interval-independent).
2. **Replay amount** replaces the legacy ``fixed_tail`` / ``replay_ratio_high`` two-tier design:
   a global ``default_replay`` plus per-task overrides, each a fraction (< 1) or an absolute
   label count (>= 1, converted to a ratio against the task's valid-label count).
3. **n-runs sweep**: ``n_runs > 1`` writes ``runs/runNN/`` subdirectories with a per-run seed and
   (optionally) a reshuffled task order, aggregated into a top-level ``experiment_records.json``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import MLPEncoderConfig, OptimizerConfig

from . import plots
from .recording import RunRecorder
from .task_catalog import TaskCatalog, TaskCatalogConfig, TaskKind, build_task_catalog_config

# Reserved built-in autoencoder head name (see FlexibleMultiTaskModel).
_AE_NAME = "__reconstruction__"
# Legacy head weight decay for regression / classification heads (kernel heads use kr_weight_decay).
_HEAD_WEIGHT_DECAY = 1e-5
# Stable qualitative palette so each task keeps one colour across the forgetting figure.
_TASK_PALETTE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
    "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#E377C2", "#17BECF",
)  # fmt: skip

# Root TOML sections a pretrain config may contain.
_PRETRAIN_ROOT_KEYS = {"data", "descriptor", "datasets", "tasks", "model", "training", "pretrain", "output"}
_CATALOG_KEYS = {"data", "descriptor", "datasets", "tasks"}


class TaskOrder(str, Enum):
    FIXED = "fixed"
    RANDOM = "random"


# --- config dataclasses -------------------------------------------------------------------


@dataclass(kw_only=True)
class ModelSectionConfig:
    """[model] — architecture dims shared with finetune."""

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
    """[training] — epochs, early-stop, learning rates, accelerator (shared with finetune)."""

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


def _validate_replay_amount(value: float | int, *, where: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{where}: replay must be a number, got bool {value!r}.")
    if isinstance(value, float):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{where}: replay float must be in (0, 1), got {value}.")
    elif isinstance(value, int):
        if value < 1:
            raise ValueError(f"{where}: replay int must be >= 1, got {value}.")
    else:
        raise ValueError(f"{where}: replay must be a float in (0, 1) or int >= 1, got {value!r}.")


@dataclass(kw_only=True)
class RehearsalConfig:
    """[pretrain.rehearsal] — interval schedule + replay amounts."""

    interval: int = 1
    default_replay: float | int = 0.05
    per_task: dict[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError(f"rehearsal.interval must be >= 1, got {self.interval}.")
        _validate_replay_amount(self.default_replay, where="rehearsal.default_replay")
        for task, value in self.per_task.items():
            _validate_replay_amount(value, where=f"rehearsal.per_task.{task}")


@dataclass(kw_only=True)
class PretrainConfig:
    catalog: TaskCatalogConfig
    model: ModelSectionConfig
    training: TrainingSectionConfig
    rehearsal: RehearsalConfig
    output_dir: Path
    task_sequence: list[str] = field(default_factory=list)
    n_runs: int = 1
    task_order: TaskOrder = TaskOrder.FIXED

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if not isinstance(self.task_order, TaskOrder):
            self.task_order = TaskOrder(str(self.task_order))
        if self.n_runs < 1:
            raise ValueError(f"pretrain.n_runs must be >= 1, got {self.n_runs}.")
        catalog_tasks = {t.name for t in self.catalog.tasks}
        if not self.task_sequence:
            self.task_sequence = [t.name for t in self.catalog.tasks]
        unknown = [t for t in self.task_sequence if t not in catalog_tasks]
        if unknown:
            raise ValueError(f"pretrain.task_sequence references unknown task(s): {unknown}.")
        bad_replay = [t for t in self.rehearsal.per_task if t not in catalog_tasks]
        if bad_replay:
            raise ValueError(f"rehearsal.per_task references unknown task(s): {bad_replay}.")


# --- builder ------------------------------------------------------------------------------


def _reject_unknown(section: str, raw: Mapping[str, Any], known: set[str]) -> None:
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"[{section}]: unknown key(s) {unknown}; allowed keys are {sorted(known)}.")


def build_pretrain_config(raw: Mapping[str, Any], *, output_dir: str | Path | None = None) -> PretrainConfig:
    """Normalize a parsed-TOML tree into a :class:`PretrainConfig`."""

    _reject_unknown("<root>", raw, _PRETRAIN_ROOT_KEYS)
    catalog = build_task_catalog_config({k: raw[k] for k in _CATALOG_KEYS if k in raw})

    model_raw = dict(raw.get("model", {}))
    _reject_unknown("model", model_raw, set(ModelSectionConfig.__dataclass_fields__))
    model = ModelSectionConfig(**model_raw)

    training_raw = dict(raw.get("training", {}))
    _reject_unknown("training", training_raw, set(TrainingSectionConfig.__dataclass_fields__))
    training = TrainingSectionConfig(**training_raw)

    pretrain_raw = dict(raw.get("pretrain", {}))
    _reject_unknown("pretrain", pretrain_raw, {"task_sequence", "n_runs", "task_order", "rehearsal"})
    rehearsal_raw = dict(pretrain_raw.get("rehearsal", {}))
    _reject_unknown("pretrain.rehearsal", rehearsal_raw, {"interval", "default_replay", "per_task"})
    rehearsal = RehearsalConfig(
        interval=rehearsal_raw.get("interval", 1),
        default_replay=rehearsal_raw.get("default_replay", 0.05),
        per_task=dict(rehearsal_raw.get("per_task", {})),
    )

    resolved_output = output_dir if output_dir is not None else raw.get("output", {}).get("dir")
    if resolved_output is None:
        raise ValueError("output directory must be given via --output-dir or [output].dir.")

    return PretrainConfig(
        catalog=catalog,
        model=model,
        training=training,
        rehearsal=rehearsal,
        output_dir=Path(resolved_output),
        task_sequence=list(pretrain_raw.get("task_sequence", [])),
        n_runs=int(pretrain_raw.get("n_runs", 1)),
        task_order=TaskOrder(str(pretrain_raw.get("task_order", "fixed"))),
    )


# --- pure helpers -------------------------------------------------------------------------


def active_old_tasks(step: int, learned: Sequence[str], interval: int) -> list[str]:
    """Old tasks that participate in training at 1-based ``step`` (empty on non-interval steps)."""
    if interval <= 1 or step % interval == 0:
        return list(learned)
    return []


def replay_to_ratio(replay: float | int, n_valid: int) -> float:
    """Convert a replay amount (fraction or absolute count) to a keep-ratio in [0, 1]."""
    if isinstance(replay, float):
        return replay
    if n_valid <= 0:
        return 1.0
    return min(1.0, replay / n_valid)


class _DropLastTrainCompoundDataModule(CompoundDataModule):
    """``CompoundDataModule`` whose train loader drops the final partial batch.

    Guards against BatchNorm1d crashing on a size-1 tail batch (``Expected more than 1 value per
    channel``). Only the train loader is affected; val/test/predict keep every held-out row.
    """

    def train_dataloader(self) -> DataLoader | None:  # type: ignore[override]
        base = super().train_dataloader()
        if base is None:
            return None
        return DataLoader(
            base.dataset,
            batch_size=base.batch_size,
            shuffle=True,
            num_workers=base.num_workers,
            pin_memory=base.pin_memory,
            collate_fn=base.collate_fn,
            drop_last=True,
        )


def _as_float_array(cell: Any) -> np.ndarray:
    import ast

    if isinstance(cell, str):
        cell = ast.literal_eval(cell)
    return np.asarray(cell, dtype=float).ravel()


# --- engine -------------------------------------------------------------------------------


def run(cfg: PretrainConfig, recorder: RunRecorder | None = None) -> dict[str, Any]:
    """Execute the pretraining sweep. Returns the aggregated cross-run records dict.

    ``recorder`` is the root recorder (already carrying provenance); per-run recorders are created
    under it. When ``n_runs == 1`` the single run writes directly under ``output_dir``.
    """

    catalog = TaskCatalog(cfg.catalog)
    root_recorder = recorder or RunRecorder(cfg.output_dir)
    aggregate: list[dict[str, Any]] = []

    for run_idx in range(cfg.n_runs):
        run_seed = cfg.training.seed + run_idx
        order = _run_task_order(cfg, run_seed)
        if cfg.n_runs == 1:
            run_recorder = root_recorder
        else:
            run_recorder = RunRecorder(cfg.output_dir / "runs" / f"run{run_idx:02d}")
        logger.info(f"=== pretrain run {run_idx + 1}/{cfg.n_runs} (seed={run_seed}) order={order} ===")
        records = _run_single(cfg, catalog, run_recorder, task_order=order, seed=run_seed)
        for rec in records:
            aggregate.append({"run": run_idx, **rec})
        if run_recorder is not root_recorder:
            run_recorder.close()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "experiment_records.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return {"records": aggregate}


def _run_task_order(cfg: PretrainConfig, seed: int) -> list[str]:
    if cfg.task_order is TaskOrder.RANDOM:
        rng = np.random.default_rng(seed)
        order = list(cfg.task_sequence)
        rng.shuffle(order)
        return order
    return list(cfg.task_sequence)


def _run_single(
    cfg: PretrainConfig,
    catalog: TaskCatalog,
    recorder: RunRecorder,
    *,
    task_order: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    seed_everything(seed, workers=True)
    model = _build_empty_model(cfg, catalog)

    task_colors = {name: _TASK_PALETTE[i % len(_TASK_PALETTE)] for i, name in enumerate(task_order)}
    clf_tasks = frozenset(n for n in task_order if catalog.task_spec(n).kind is TaskKind.CLASSIFICATION)
    metric_history: dict[str, list[tuple[int, float]]] = {name: [] for name in task_order}
    records: list[dict[str, Any]] = []
    built: dict[str, Any] = {}

    for step, task_name in enumerate(task_order, start=1):
        logger.info(f"--- step {step}/{len(task_order)}: add '{task_name}' ---")
        built[task_name] = _build_task_config(cfg, catalog, task_name, masking_ratio=1.0)
        model.add_task(built[task_name])

        learned = list(task_order[: step - 1])
        participating = active_old_tasks(step, learned, cfg.rehearsal.interval)
        active = [task_name, *participating]
        for name in participating:
            ratio = _replay_ratio_for(cfg, catalog, name)
            built[name].task_masking_ratio = ratio
        built[task_name].task_masking_ratio = 1.0

        datamodule = _DropLastTrainCompoundDataModule(
            task_configs=[built[name] for name in active],
            descriptor_fn=catalog.descriptor_fn(),
            task_frames={name: catalog.task_frames([name])[name] for name in active},
            composition_column=cfg.catalog.data.composition_column,
            random_seed=cfg.catalog.data.split_random_seed,
            val_split=cfg.catalog.data.val_split,
            test_split=cfg.catalog.data.test_split,
            batch_size=cfg.catalog.data.batch_size,
            num_workers=cfg.catalog.data.num_workers,
        )
        callbacks: list[Callback] = [
            EarlyStopping(
                monitor="val_final_loss",
                mode="min",
                patience=cfg.training.early_stop_patience,
                min_delta=cfg.training.early_stop_min_delta,
            )
        ]
        trainer = Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator=cfg.training.accelerator,
            devices=cfg.training.devices,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule=datamodule)

        test_keys: set[str] | None = None
        if datamodule.split_series is not None:
            resolved = datamodule.split_series
            test_keys = set(resolved.index[resolved == "test"].astype(str))

        step_dir = recorder.paths.step_dir(step, task_name)
        step_metrics: dict[str, dict[str, float]] = {}
        for name in task_order[:step]:  # evaluate ALL learned heads, regardless of interval
            metric = _evaluate_task(
                model, catalog, name, recorder, step_dir, is_new=(name == task_name), test_keys=test_keys
            )
            step_metrics[name] = metric
            metric_history[name].append((step, metric["primary"]))

        recorder.save_step_checkpoint(step, task_name, model, list(active))
        record = {"step": step, "new_task": task_name, "epochs_run": trainer.current_epoch, "metrics": step_metrics}
        records.append(record)
        recorder.append_record(record)

    fig = plots.plot_forgetting(metric_history, task_colors=task_colors, clf_tasks=clf_tasks)
    recorder.save_figure("training/forgetting_trajectory.png", fig)
    plots.plt.close(fig)
    recorder.write_records()
    recorder.write_metrics_table()
    recorder.save_final_model(model, list(task_order), _task_spec_dump(cfg, catalog, task_order))
    return records


def _build_empty_model(cfg: PretrainConfig, catalog: TaskCatalog) -> FlexibleMultiTaskModel:
    encoder_config = MLPEncoderConfig(
        hidden_dims=[catalog.descriptor_dim, cfg.model.encoder_hidden, cfg.model.latent_dim]
    )
    model = FlexibleMultiTaskModel(
        task_configs=[],
        encoder_config=encoder_config,
        enable_autoencoder=True,
        shared_block_optimizer=OptimizerConfig(lr=cfg.training.encoder_lr, weight_decay=1e-2),
    )
    # AE head always trains; only its LR is configurable (ae_lr).
    if _AE_NAME in model.task_configs_map:
        model.task_configs_map[_AE_NAME].optimizer = OptimizerConfig(lr=cfg.training.ae_lr)
    return model


def _build_task_config(cfg: PretrainConfig, catalog: TaskCatalog, name: str, *, masking_ratio: float) -> Any:
    spec = catalog.task_spec(name)
    if spec.kind is TaskKind.KERNEL_REGRESSION:
        lr, weight_decay = cfg.training.kr_lr, cfg.training.kr_weight_decay
    else:
        lr, weight_decay = cfg.training.head_lr, _HEAD_WEIGHT_DECAY
    return catalog.build_task_config(
        name,
        latent_dim=cfg.model.latent_dim,
        head_hidden_dim=cfg.model.head_hidden_dim,
        n_kernel=cfg.model.n_kernel,
        lr=lr,
        weight_decay=weight_decay,
        masking_ratio=masking_ratio,
    )


def _replay_ratio_for(cfg: PretrainConfig, catalog: TaskCatalog, name: str) -> float:
    replay = cfg.rehearsal.per_task.get(name, cfg.rehearsal.default_replay)
    if isinstance(replay, float):
        return replay
    return replay_to_ratio(replay, _n_valid_train_labels(catalog, name))


def _n_valid_train_labels(catalog: TaskCatalog, name: str) -> int:
    spec = catalog.task_spec(name)
    frame = catalog.task_frames([name])[name]
    mask = frame[spec.column].notna()
    if "split" in frame.columns:
        mask &= frame["split"] == "train"
    return int(mask.sum())


def _task_spec_dump(cfg: PretrainConfig, catalog: TaskCatalog, task_order: Sequence[str]) -> dict[str, Any]:
    dump: dict[str, Any] = {}
    for name in task_order:
        spec = catalog.task_spec(name)
        dump[name] = {"kind": spec.kind.value, "column": spec.column, "source": spec.dataset}
    return dump


def _test_rows(catalog: TaskCatalog, name: str, test_keys: set[str] | None) -> list[str]:
    spec = catalog.task_spec(name)
    frame = catalog.task_frames([name])[name]
    mask = frame[spec.column].notna()
    if test_keys is not None:
        mask &= frame.index.isin(test_keys)
    elif "split" in frame.columns:
        mask &= frame["split"] == "test"
    return list(frame.index[mask])


def _descriptor_tensor(catalog: TaskCatalog, comps: list[str], device: torch.device) -> tuple[torch.Tensor, list[str]]:
    desc = catalog.descriptor_fn()(comps)
    kept = [c for c in comps if c in desc.index]
    tensor = torch.tensor(desc.loc[kept].values, dtype=torch.float32, device=device)
    return tensor, kept


def _evaluate_task(
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
    comps = _test_rows(catalog, name, test_keys)
    if not comps:
        return {"primary": float("nan"), "samples": 0}
    frame = catalog.task_frames([name])[name]
    head = model.task_heads[name]

    with torch.no_grad():
        if spec.kind in (TaskKind.REGRESSION, TaskKind.CLASSIFICATION):
            x, comps = _descriptor_tensor(catalog, comps, device)
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
                    recorder.save_figure(_fig_rel(recorder, step_dir, f"{name}_parity.png"), fig)
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
                recorder.save_figure(_fig_rel(recorder, step_dir, f"{name}_confusion.png"), fig)
                plots.plt.close(fig)
            return metric

        # kernel regression
        assert spec.t_column is not None
        keep: list[str] = []
        t_list: list[np.ndarray] = []
        true_parts: list[np.ndarray] = []
        for comp in comps:
            if catalog.descriptor_fn()([comp]).empty:
                continue
            y_arr = _as_float_array(frame.at[comp, spec.column])
            t_arr = _as_float_array(frame.at[comp, spec.t_column])
            if y_arr.size == 0 or y_arr.size != t_arr.size:
                continue
            keep.append(comp)
            t_list.append(t_arr)
            true_parts.append(y_arr)
        if not keep:
            return {"primary": float("nan"), "samples": 0}
        xk, _ = _descriptor_tensor(catalog, keep, device)
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
                recorder.save_figure(_fig_rel(recorder, step_dir, f"{name}_sequences.png"), kr_fig)
                plots.plt.close(kr_fig)
        return metric


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


def _fig_rel(recorder: RunRecorder, step_dir: Path, filename: str) -> str:
    return str((step_dir / filename).relative_to(recorder.paths.root))
