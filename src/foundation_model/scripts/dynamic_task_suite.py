#!/usr/bin/env python
# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:  # pragma: no cover - optional distributed import
    import torch.distributed as dist
except Exception:  # noqa: BLE001 - broad to cover ImportError/RuntimeError
    dist = None
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from loguru import logger as fm_logger

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import (
    BaseEncoderConfig,
    ClassificationTaskConfig,
    EncoderType,
    KernelRegressionTaskConfig,
    MLPEncoderConfig,
    OptimizerConfig,
    RegressionTaskConfig,
    TaskType,
    TransformerEncoderConfig,
    build_encoder_config,
)

torch.serialization.add_safe_globals(
    [
        RegressionTaskConfig,
        ClassificationTaskConfig,
        KernelRegressionTaskConfig,
        OptimizerConfig,
        TransformerEncoderConfig,
        MLPEncoderConfig,
        TaskType,
        EncoderType,
    ]
)
# Default task configuration mirrors the notebook for discoverability.
DEFAULT_PRETRAIN_TASKS: list[str] = [
    "density",
    "Rg",
    "r2",
    "self-diffusion",
    "Cp",
    "Cv",
    "bulk_modulus",
    "volume_expansion",
    "linear_expansion",
    "static_dielectric_const",
    "dielectric_const_dc",
    "refractive_index",
    "tg",
    "thermal_conductivity",
    "thermal_diffusivity",
]

DEFAULT_FINETUNE_TASKS: list[str] = [
    "density",
    "Rg",
    "r2",
    "self-diffusion",
    "Cp",
    "Cv",
    "linear_expansion",
    "refractive_index",
    "tg",
]


@dataclass
class SuiteConfig:
    descriptor_path: Path
    pretrain_data_path: Path
    finetune_data_path: Path
    output_dir: Path
    scaler_path: Path | None = None
    use_normalized_targets: bool = True
    keep_normalized_targets: bool = False
    freeze_shared_encoder: bool = True
    enable_learnable_loss: bool = False
    quiet_model_logging: bool = True
    shared_block_dims: Sequence[int] = (190, 256, 128)
    encoder_config: Mapping[str, Any] | BaseEncoderConfig | None = None
    head_hidden_dim: int = 64
    head_lr: float | None = None
    num_pretrain_runs: int = 10
    pretrain_max_epochs: int = 200
    finetune_max_epochs: int = 200
    batch_size: int = 256
    num_workers: int = 0
    log_every_n_steps: int = 20
    random_seed_base: int = 1729
    datamodule_random_seed: int | None = 42
    pretrain_sample: int | None = None
    finetune_sample: int | None = None
    pretrain_tasks: list[str] = field(default_factory=lambda: list(DEFAULT_PRETRAIN_TASKS))
    finetune_tasks: list[str] = field(default_factory=lambda: list(DEFAULT_FINETUNE_TASKS))
    task_sequence: list[str] | None = None
    task_masking_ratios: float | Dict[str, float] | None = None
    swap_train_val_split: float = 0.0
    val_split: float = 0.1
    test_split: float = 0.1
    test_all: bool = False
    accelerator: str = "auto"
    devices: str | int | list[int] | None = "auto"
    early_stopping_patience: int = 10

    def __post_init__(self) -> None:
        if self.num_pretrain_runs <= 0:
            raise ValueError("num_pretrain_runs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.swap_train_val_split < 0.0 or self.swap_train_val_split > 1.0:
            raise ValueError("swap_train_val_split must be within [0, 1]")
        if self.val_split < 0.0 or self.test_split < 0.0:
            raise ValueError("val_split and test_split must be non-negative")
        if self.task_sequence:
            invalid = sorted(set(self.task_sequence) - set(self.pretrain_tasks))
            if invalid:
                raise ValueError(f"task_sequence contains tasks {invalid} not present in the pretrain task list")
        if len(self.shared_block_dims) < 2:
            raise ValueError("shared_block_dims must include input dimension and at least one latent dimension.")
        if self.head_lr is not None and self.head_lr <= 0:
            raise ValueError("head_lr must be positive when provided.")


class DynamicTaskSuiteRunner:
    def __init__(self, config: SuiteConfig):
        self.config = config
        self.property_scalers: dict[str, Any] = {}
        self.pretrain_target_columns: dict[str, str] = {}
        self.finetune_target_columns: dict[str, str] = {}
        self.pretrain_features: pd.DataFrame | None = None
        self.pretrain_targets: pd.DataFrame | None = None
        self.finetune_features: pd.DataFrame | None = None
        self.finetune_targets: pd.DataFrame | None = None
        self.encoder_config = self._build_encoder_config()
        self.encoder_latent_dim = self.encoder_config.latent_dim
        if config.quiet_model_logging:
            fm_logger.disable("foundation_model")
        else:
            fm_logger.enable("foundation_model")

    def run(self) -> Path:
        self._seed_everything(self.config.random_seed_base)
        self._load_datasets()

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        experiment_records: list[dict[str, Any]] = []

        for run_idx in range(1, self.config.num_pretrain_runs + 1):
            run_label = f"run{run_idx:02d}"
            run_root = output_dir / run_label
            run_root.mkdir(parents=True, exist_ok=True)

            task_sequence = self._resolve_task_sequence(run_idx)
            self._log_run_header(run_label, task_sequence)

            previous_checkpoint: str | None = None
            pretrain_stage_records: list[dict[str, Any]] = []
            finetune_records: list[dict[str, Any]] = []

            for stage_index, task_name in enumerate(task_sequence, start=1):
                stage_tasks = task_sequence[:stage_index]
                model, datamodule = self._prepare_pretrain_stage(stage_tasks, previous_checkpoint)
                stage_dir = run_root / f"pretrain_stage{stage_index:02d}_{self._safe_slug(task_name)}"
                stage_dir.mkdir(parents=True, exist_ok=True)

                best_model_path = self._fit_stage(
                    model=model,
                    datamodule=datamodule,
                    stage_dir=stage_dir,
                    stage_name=task_name,
                    max_epochs=self.config.pretrain_max_epochs,
                )
                if best_model_path:
                    previous_checkpoint = best_model_path
                else:
                    fm_logger.warning(
                        "No checkpoint produced for %s stage %s; continuing with in-memory weights.",
                        run_label,
                        stage_index,
                    )

                prediction_dir = stage_dir / "prediction"
                self._plot_predictions(
                    model=model,
                    datamodule=datamodule,
                    phase="pretrain",
                    run_id=run_idx,
                    stage_num=stage_index,
                    stage_tasks=stage_tasks,
                    new_task_name=task_name,
                    output_dir=prediction_dir,
                )

                stage_record: dict[str, Any] = {
                    "stage": stage_index,
                    "task_name": task_name,
                    "task_sequence": list(stage_tasks),
                    "checkpoint": previous_checkpoint,
                    "stage_dir": str(stage_dir),
                }

                finetune_stage_records: list[dict[str, Any]] = []
                if previous_checkpoint is not None:
                    finetune_results = self._run_finetune_stages(
                        checkpoint_path=previous_checkpoint,
                        stage_dir=stage_dir,
                        run_idx=run_idx,
                        stage_idx=stage_index,
                        stage_tasks=stage_tasks,
                    )
                    finetune_stage_records.extend(finetune_results["records"])
                    finetune_records.extend(finetune_results["records"])
                else:
                    fm_logger.warning(
                        "Skipping finetune for run %s stage %s because no checkpoint was available.",
                        run_label,
                        stage_index,
                    )

                stage_record["finetune"] = finetune_stage_records
                pretrain_stage_records.append(stage_record)

            if previous_checkpoint is None:
                raise RuntimeError(f"{run_label} did not produce a checkpoint; aborting.")

            experiment_records.append(
                {
                    "run": run_label,
                    "task_sequence": task_sequence,
                    "pretrain": pretrain_stage_records,
                    "pretrain_checkpoint": previous_checkpoint,
                    "finetune": finetune_records,
                }
            )

        summary_path = output_dir / "experiment_records.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(experiment_records, handle, indent=2)
        return summary_path

    def _resolve_task_sequence(self, run_idx: int) -> list[str]:
        if self.config.task_sequence:
            return list(self.config.task_sequence)
        rng = random.Random(self.config.random_seed_base + run_idx)
        return rng.sample(self.config.pretrain_tasks, k=len(self.config.pretrain_tasks))

    def _build_encoder_config(self) -> BaseEncoderConfig:
        source_config = self.config.encoder_config
        input_dim = self.config.shared_block_dims[0]

        if isinstance(source_config, BaseEncoderConfig):
            updated_config = copy.deepcopy(source_config)
            if isinstance(updated_config, MLPEncoderConfig):
                hidden_dims = list(updated_config.hidden_dims)
                if not hidden_dims or hidden_dims[0] != input_dim:
                    hidden_dims = [input_dim, *hidden_dims]
                updated_config.hidden_dims = tuple(hidden_dims)
            if isinstance(updated_config, TransformerEncoderConfig):
                updated_config.input_dim = input_dim
            if isinstance(updated_config, TransformerEncoderConfig):
                updated_config.d_model = updated_config.d_model or self.config.shared_block_dims[-1]
            config_payload: BaseEncoderConfig | Mapping[str, Any] = updated_config
        else:
            mapping_payload: dict[str, Any] = {}
            if isinstance(source_config, Mapping):
                mapping_payload.update(source_config)
            mapping_type = mapping_payload.get("type", EncoderType.MLP)
            if not isinstance(mapping_type, EncoderType):
                mapping_type = EncoderType(str(mapping_type).lower())
            if mapping_type is EncoderType.MLP:
                default_hidden = list(self.config.shared_block_dims)
                if len(default_hidden) < 2:
                    raise ValueError("shared_block_dims must include input_dim and at least one latent dimension.")
                mapping_payload.setdefault("hidden_dims", default_hidden)
            else:
                mapping_payload.setdefault("input_dim", input_dim)
                mapping_payload.setdefault("d_model", self.config.shared_block_dims[-1])
            config_payload = mapping_payload

        return build_encoder_config(config_payload)

    def _encoder_config_instance(self) -> BaseEncoderConfig:
        return copy.deepcopy(self.encoder_config)

    def _prepare_pretrain_stage(
        self,
        stage_tasks: Sequence[str],
        previous_checkpoint: str | None,
    ) -> tuple[FlexibleMultiTaskModel, CompoundDataModule]:
        datamodule = self._build_pretrain_datamodule(stage_tasks)
        task_configs = [self._build_regression_task(name, self.pretrain_target_columns[name]) for name in stage_tasks]

        # Scale learning rate with world_size (Linear Scaling Rule)
        base_lr = 5e-2
        is_distributed = dist is not None and dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        scaled_lr = base_lr * world_size

        if previous_checkpoint is None:
            model = FlexibleMultiTaskModel(
                task_configs=task_configs,
                encoder_config=self._encoder_config_instance(),
                enable_learnable_loss_balancer=self.config.enable_learnable_loss,
                shared_block_optimizer=OptimizerConfig(lr=scaled_lr),
            )
        else:
            model = FlexibleMultiTaskModel.load_from_checkpoint(
                checkpoint_path=previous_checkpoint,
                strict=False,
                enable_learnable_loss_balancer=self.config.enable_learnable_loss,
                encoder_config=self._encoder_config_instance(),
            )
            existing = set(model.task_heads.keys())
            new_configs = [cfg for cfg in task_configs if cfg.name not in existing]
            if new_configs:
                model.add_task(*new_configs)
        return model, datamodule

    def _run_finetune_stages(
        self,
        *,
        checkpoint_path: str,
        stage_dir: Path,
        run_idx: int,
        stage_idx: int,
        stage_tasks: Sequence[str],
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []

        # Scale learning rate with world_size (Linear Scaling Rule)
        base_lr = 5e-2
        is_distributed = dist is not None and dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        scaled_lr = base_lr * world_size

        for finetune_name in self.config.finetune_tasks:
            finetune_model = FlexibleMultiTaskModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                strict=False,
                enable_learnable_loss_balancer=self.config.enable_learnable_loss,
                freeze_shared_encoder=self.config.freeze_shared_encoder,
                encoder_config=self._encoder_config_instance(),
                shared_block_optimizer=OptimizerConfig(lr=scaled_lr),
            )
            active = list(finetune_model.task_heads.keys())
            if active:
                finetune_model.remove_tasks(*active)
            finetune_task = self._build_regression_task(
                finetune_name,
                self.finetune_target_columns[finetune_name],
            )
            finetune_model.add_task(finetune_task)

            datamodule = self._build_finetune_datamodule(finetune_name)
            finetune_root = stage_dir / "finetune"
            finetune_stage_dir = finetune_root / self._safe_slug(finetune_name)
            finetune_stage_dir.mkdir(parents=True, exist_ok=True)

            best_model_path = self._fit_stage(
                model=finetune_model,
                datamodule=datamodule,
                stage_dir=finetune_stage_dir,
                stage_name=finetune_name,
                max_epochs=self.config.finetune_max_epochs,
            )
            if best_model_path:
                state = torch.load(best_model_path, map_location="cpu", weights_only=True)
                state_dict = state.get("state_dict", state)
                finetune_model.load_state_dict(state_dict)

            prediction_dir = finetune_stage_dir / "prediction"
            self._plot_predictions(
                model=finetune_model,
                datamodule=datamodule,
                phase="finetune",
                run_id=run_idx,
                stage_num=stage_idx,
                stage_tasks=[finetune_name],
                new_task_name=finetune_name,
                output_dir=prediction_dir,
            )

            records.append(
                {
                    "stage": stage_idx,
                    "task_name": finetune_name,
                    "pretrain_task_sequence": list(stage_tasks),
                    "checkpoint": best_model_path,
                    "stage_dir": str(finetune_stage_dir),
                }
            )

        return {"records": records}

    def _fit_stage(
        self,
        *,
        model: FlexibleMultiTaskModel,
        datamodule: CompoundDataModule,
        stage_dir: Path,
        stage_name: str,
        max_epochs: int,
    ) -> str | None:
        checkpoint_dir = stage_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cb = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{self._safe_slug(stage_name)}-{{epoch:02d}}-{{val_final_loss:.4f}}",
            monitor="val_final_loss",
            mode="min",
            save_top_k=1,
        )
        early_stopping = EarlyStopping(
            monitor="val_final_loss",
            mode="min",
            patience=self.config.early_stopping_patience,
        )
        csv_logger = CSVLogger(save_dir=stage_dir / "logs", name="csv")
        tensorboard_logger = TensorBoardLogger(save_dir=stage_dir / "logs", name="tensorboard")

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            sync_batchnorm=True,  # Synchronize BatchNorm statistics across GPUs
            callbacks=[checkpoint_cb, early_stopping],
            logger=[csv_logger, tensorboard_logger],
            log_every_n_steps=self.config.log_every_n_steps,
        )

        trainer.fit(model, datamodule=datamodule)
        best_path = checkpoint_cb.best_model_path
        fm_logger.info(f"Stage {stage_name} finished; best checkpoint: {best_path or '<none>'}")
        if best_path:
            state = torch.load(best_path, map_location="cpu", weights_only=True)
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict)
        return best_path

    def _build_pretrain_datamodule(self, task_names: Sequence[str]) -> CompoundDataModule:
        if self.pretrain_features is None or self.pretrain_targets is None:
            raise RuntimeError("Pretrain data is not loaded.")
        target_columns = [self.pretrain_target_columns[name] for name in task_names]
        stage_targets = self.pretrain_targets.loc[:, target_columns]
        return CompoundDataModule(
            formula_desc_source=self.pretrain_features,
            attributes_source=stage_targets,
            task_configs=[self._build_regression_task(name, self.pretrain_target_columns[name]) for name in task_names],
            task_masking_ratios=self.config.task_masking_ratios,
            random_seed=self.config.datamodule_random_seed,
            val_split=self.config.val_split,
            test_split=self.config.test_split,
            test_all=self.config.test_all,
            swap_train_val_split=self.config.swap_train_val_split,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def _build_finetune_datamodule(self, task_name: str) -> CompoundDataModule:
        if self.finetune_features is None or self.finetune_targets is None:
            raise RuntimeError("Finetune data is not loaded.")
        column = self.finetune_target_columns[task_name]
        target_frame = self.finetune_targets.loc[:, [column]]
        return CompoundDataModule(
            formula_desc_source=self.finetune_features,
            attributes_source=target_frame,
            task_configs=[self._build_regression_task(task_name, column)],
            task_masking_ratios=self.config.task_masking_ratios,
            random_seed=self.config.datamodule_random_seed,
            val_split=self.config.val_split,
            test_split=self.config.test_split,
            test_all=self.config.test_all,
            swap_train_val_split=self.config.swap_train_val_split,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def _build_regression_task(self, name: str, column: str) -> RegressionTaskConfig:
        optimizer = None
        if self.config.head_lr is not None:
            optimizer = OptimizerConfig(
                lr=self.config.head_lr,
                weight_decay=1e-5,
            )
        return RegressionTaskConfig(
            name=name,
            data_column=column,
            dims=[self.encoder_latent_dim, self.config.head_hidden_dim, 1],
            norm=True,
            residual=False,
            optimizer=optimizer,
        )

    def _plot_predictions(
        self,
        *,
        model: FlexibleMultiTaskModel,
        datamodule: CompoundDataModule,
        phase: str,
        run_id: int,
        stage_num: int | None,
        stage_tasks: Sequence[str],
        new_task_name: str,
        output_dir: Path | str,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        predictions_path = output_dir / "predictions.parquet"
        task_order_path = output_dir / "tasks.txt"

        is_distributed = dist is not None and dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1

        if rank == 0:
            task_order_path.write_text(" -> ".join(stage_tasks), encoding="utf-8")

        device = self._resolve_device()
        datamodule.setup(stage="test")
        test_loader = datamodule.test_dataloader()
        if test_loader is None:
            raise RuntimeError(f"{phase} stage {stage_num} has no test dataloader")

        # Get sample indices for this rank (for proper ordering in distributed mode)
        sampler = test_loader.sampler
        if sampler is not None and hasattr(sampler, "dataset"):
            # Mirror DistributedSampler padding to maintain original dataset ordering
            dataset_size = len(sampler.dataset)
            num_samples = len(sampler)
            total_size = num_samples * world_size
            base_indices = list(range(dataset_size))
            if len(base_indices) < total_size:
                base_indices.extend(base_indices[: total_size - len(base_indices)])
            indices_for_rank = base_indices[rank:total_size:world_size]
        else:
            # Non-distributed case: process all samples in order
            indices_for_rank = list(range(len(test_loader.dataset)))

        original_device = next(model.parameters()).device
        was_training = model.training
        model.to(device)
        model.eval()

        prediction_rows: list[dict[str, float | int | str | None]] = []
        global_sample_idx = 0  # Track position in indices_for_rank

        with torch.no_grad():
            for batch in test_loader:
                x, y_dict, mask_dict, t_sequences = batch
                x = x.to(device)
                preds = model(x, t_sequences)

                # Determine batch size for index mapping
                batch_size = x.shape[0]
                batch_indices = indices_for_rank[global_sample_idx : global_sample_idx + batch_size]

                for name, pred_tensor in preds.items():
                    if name not in y_dict:
                        continue
                    target_tensor = y_dict[name]
                    mask_tensor = mask_dict.get(name)

                    target_flat = self._flatten_tensor(target_tensor)
                    pred_flat = self._flatten_tensor(pred_tensor)

                    if mask_tensor is not None:
                        mask_flat = self._flatten_tensor(mask_tensor).bool()
                        target_flat = target_flat[mask_flat]
                        pred_flat = pred_flat[mask_flat]

                    if target_flat.numel() == 0:
                        continue

                    target_np = target_flat.cpu().numpy()
                    pred_np = pred_flat.cpu().numpy()
                    target_np = self._maybe_inverse_transform(name, target_np)
                    pred_np = self._maybe_inverse_transform(name, pred_np)

                    entry = aggregated.setdefault(name, {"preds": [], "targets": []})
                    entry["preds"].extend(pred_np.tolist())
                    entry["targets"].extend(target_np.tolist())

                    # Assign original dataset indices to predictions
                    for in_batch_idx, (actual_val, pred_val) in enumerate(zip(target_np.tolist(), pred_np.tolist())):
                        # Map to original dataset index
                        if in_batch_idx < len(batch_indices):
                            dataset_idx = batch_indices[in_batch_idx]
                        else:
                            # Fallback for edge cases
                            dataset_idx = batch_indices[-1] if batch_indices else global_sample_idx + in_batch_idx

                        prediction_rows.append(
                            {
                                "run": run_id,
                                "phase": phase,
                                "stage": stage_num,
                                "task": name,
                                "dataset_index": dataset_idx,  # Original dataset position
                                "actual": actual_val,
                                "predicted": pred_val,
                            }
                        )

                global_sample_idx += batch_size

        if is_distributed:
            rank_prediction_path = output_dir / f"predictions_rank{rank:03d}.parquet"
            if prediction_rows:
                pd.DataFrame(prediction_rows).to_parquet(rank_prediction_path, index=False)
            elif rank_prediction_path.exists():
                rank_prediction_path.unlink()
            dist.barrier()
            if rank != 0:
                model.to(original_device)
                model.train(was_training)
                return
            prediction_rows = self._merge_rank_prediction_rows(output_dir, world_size)

        aggregated: dict[str, dict[str, list[float]]] = {}

        if prediction_rows:
            # Deduplicate by (task, dataset_index) - keep first occurrence
            # This handles DistributedSampler padding where samples may be duplicated
            seen_keys: set[tuple[str, int]] = set()
            deduplicated_rows: list[dict[str, float | int | str | None]] = []

            for row in prediction_rows:
                task_name = str(row.get("task"))
                dataset_idx = int(row.get("dataset_index", -1))
                key = (task_name, dataset_idx)

                if key not in seen_keys:
                    seen_keys.add(key)
                    deduplicated_rows.append(row)

            # Sort by task name, then by dataset_index to restore original order
            deduplicated_rows.sort(key=lambda r: (str(r.get("task")), int(r.get("dataset_index", 0))))

            # Assign sequential sample_index per task (for backward compatibility)
            sample_counters: dict[str, int] = {}
            for row in deduplicated_rows:
                task_name = str(row.get("task"))
                current_index = sample_counters.get(task_name, 0)
                row["sample_index"] = current_index
                sample_counters[task_name] = current_index + 1

            prediction_rows = deduplicated_rows
            aggregated = {}
            for row in deduplicated_rows:
                task_name = str(row.get("task"))
                entry = aggregated.setdefault(task_name, {"preds": [], "targets": []})
                actual_val = row.get("actual")
                pred_val = row.get("predicted")
                if actual_val is None or pred_val is None:
                    continue
                entry["targets"].append(float(actual_val))
                entry["preds"].append(float(pred_val))

        metrics: dict[str, dict[str, float | int | None]] = {}
        for name in stage_tasks:
            if name not in aggregated:
                continue
            preds = np.asarray(aggregated[name]["preds"], dtype=np.float64)
            targets = np.asarray(aggregated[name]["targets"], dtype=np.float64)
            if preds.size == 0 or targets.size == 0:
                continue
            diff = preds - targets
            mae = float(np.mean(np.abs(diff)))
            mse = float(np.mean(diff**2))
            rmse = float(np.sqrt(np.mean(diff**2)))
            ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
            ss_res = float(np.sum(diff**2))
            r2_value = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
            metrics[name] = {
                "samples": int(targets.size),
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2_value,
            }

            lo = float(np.min([preds.min(), targets.min()]))
            hi = float(np.max([preds.max(), targets.max()]))
            if not np.isfinite(lo) or not np.isfinite(hi):
                fm_logger.warning(f"Skipping plot for {name}: non-finite range.")
                continue
            buffer = 0.05 * (hi - lo) if hi > lo else 0.1
            lo -= buffer
            hi += buffer

            fig, ax = plt.subplots(figsize=(9, 9))
            ax.scatter(targets, preds, s=14, alpha=0.6, edgecolors="none")
            ax.plot([lo, hi], [lo, hi], "--", color="tab:red", linewidth=1.5)
            annotation_lines = [
                f"MAE: {mae:.3f}",
                rf"$R^2$: {r2_value:.3f}" if r2_value is not None else r"$R^2$: N/A",
                f"Samples: {int(targets.size):,}",
            ]
            ax.text(
                0.05,
                0.95,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                fontsize=13,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7),
            )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            if phase == "pretrain" and stage_num is not None:
                title_prefix = f"Pretrain Stage {stage_num}"
            else:
                title_prefix = "Finetune"
            ax.set_title(f"{title_prefix}: {name}")
            ax.grid(alpha=0.25)
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()
            fig.savefig(output_dir / f"{self._safe_slug(name)}_pred.png", dpi=100)
            plt.close(fig)

        metrics_payload = {
            "run_id": run_id,
            "phase": phase,
            "stage": stage_num,
            "new_task": new_task_name,
            "task_sequence": list(stage_tasks),
            "metrics": metrics,
        }

        if prediction_rows:
            pd.DataFrame(prediction_rows).to_parquet(predictions_path, index=False)
            fm_logger.info(f"Saved predictions to {predictions_path}")

        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, indent=2)
        fm_logger.info(f"Saved metrics to {metrics_path}")

        model.to(original_device)
        model.train(was_training)

    @staticmethod
    def _merge_rank_prediction_rows(output_dir: Path, world_size: int) -> list[dict[str, float | int | str | None]]:
        """Load per-rank prediction files, concatenate, and clean up temporary artifacts."""
        combined_rows: list[dict[str, float | int | str | None]] = []
        for rank in range(world_size):
            rank_path = output_dir / f"predictions_rank{rank:03d}.parquet"
            if not rank_path.exists():
                continue
            df = pd.read_parquet(rank_path)
            combined_rows.extend(df.to_dict(orient="records"))
            rank_path.unlink(missing_ok=True)
        return combined_rows

    def _load_datasets(self) -> None:
        config = self.config
        descriptor_df = pd.read_parquet(config.descriptor_path)
        pretrain_df = pd.read_parquet(config.pretrain_data_path)
        finetune_df = pd.read_parquet(config.finetune_data_path)

        property_names = sorted(set(config.pretrain_tasks) | set(config.finetune_tasks))
        self.property_scalers = self._load_scalers(property_names)

        self.pretrain_target_columns = {
            name: self._target_column(name, config.use_normalized_targets) for name in config.pretrain_tasks
        }
        finetune_target_columns = {
            name: self._target_column(name, config.use_normalized_targets) for name in config.finetune_tasks
        }

        missing_pretrain = [
            column for column in self.pretrain_target_columns.values() if column not in pretrain_df.columns
        ]
        if missing_pretrain:
            raise KeyError(f"Pretrain table missing target columns: {missing_pretrain}")

        available_finetune = []
        missing_finetune = []
        for name, column in finetune_target_columns.items():
            if column in finetune_df.columns:
                available_finetune.append(name)
            else:
                missing_finetune.append(name)

        if missing_finetune:
            fm_logger.warning(
                "Finetune targets missing for tasks %s; they will be skipped.",
                ", ".join(missing_finetune),
            )
        if not available_finetune:
            raise ValueError("No finetune tasks remain after filtering missing columns.")

        self.finetune_target_columns = {name: finetune_target_columns[name] for name in available_finetune}
        self.config.finetune_tasks = available_finetune

        common_pretrain_index = descriptor_df.index.intersection(pretrain_df.index)
        pretrain_features = descriptor_df.loc[common_pretrain_index]
        pretrain_targets = pretrain_df.loc[common_pretrain_index, list(self.pretrain_target_columns.values())]
        if config.pretrain_sample and config.pretrain_sample < len(pretrain_features):
            pretrain_features = pretrain_features.sample(n=config.pretrain_sample, random_state=42)
            pretrain_targets = pretrain_targets.loc[pretrain_features.index]

        common_finetune_index = descriptor_df.index.intersection(finetune_df.index)
        finetune_features = descriptor_df.loc[common_finetune_index]
        finetune_targets = finetune_df.loc[common_finetune_index, list(self.finetune_target_columns.values())]
        if config.finetune_sample and config.finetune_sample < len(finetune_features):
            finetune_features = finetune_features.sample(n=config.finetune_sample, random_state=13)
            finetune_targets = finetune_targets.loc[finetune_features.index]

        self.pretrain_features = pretrain_features
        self.pretrain_targets = pretrain_targets
        self.finetune_features = finetune_features
        self.finetune_targets = finetune_targets

        fm_logger.info(f"Pretrain feature matrix: {pretrain_features.shape}")
        fm_logger.info(f"Pretrain target matrix: {pretrain_targets.shape}")
        fm_logger.info(f"Finetune feature matrix: {finetune_features.shape}")
        fm_logger.info(f"Finetune target matrix: {finetune_targets.shape}")

    def _load_scalers(self, property_names: Iterable[str]) -> dict[str, Any]:
        if not self.config.use_normalized_targets:
            return {}
        if self.config.scaler_path is None:
            raise ValueError("scaler_path must be provided when use_normalized_targets is True.")
        if not self.config.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {self.config.scaler_path}")
        scalers = joblib.load(self.config.scaler_path)
        missing = [name for name in property_names if name not in scalers]
        if missing:
            raise KeyError(f"Scaler file missing entries for: {missing}")
        return scalers

    def _maybe_inverse_transform(self, property_name: str, values: np.ndarray) -> np.ndarray:
        if not self.config.use_normalized_targets or self.config.keep_normalized_targets:
            return values
        scaler = self.property_scalers.get(property_name)
        if scaler is None:
            raise KeyError(f"Scaler not found for property '{property_name}'")
        restored = scaler.inverse_transform(values.reshape(-1, 1))
        return np.asarray(restored).reshape(-1)

    @staticmethod
    def _flatten_tensor(tensor_like: Any) -> torch.Tensor:
        if isinstance(tensor_like, list):
            return torch.cat([t.detach().reshape(-1) for t in tensor_like])
        return tensor_like.detach().reshape(-1)

    @staticmethod
    def _target_column(property_name: str, use_normalized: bool) -> str:
        return f"{property_name} (normalized)" if use_normalized else property_name

    @staticmethod
    def _safe_slug(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug or "task"

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _log_run_header(run_label: str, task_sequence: Sequence[str]) -> None:
        header = f"""
====================
Starting {run_label}
Task order: {list(task_sequence)}
===================="""
        fm_logger.info(header)


def parse_arguments(args: Sequence[str] | None = None) -> SuiteConfig:
    parser = argparse.ArgumentParser(
        description="Run the dynamic pretrain + finetune suite originally prototyped in the notebook.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="YAML or TOML file containing argument defaults.",
    )
    parser.add_argument(
        "--descriptor-path",
        type=Path,
        default=None,
        help="Path to the shared descriptor parquet.",
    )
    parser.add_argument(
        "--pretrain-data-path",
        type=Path,
        default=None,
        help="Path to the pretraining targets parquet.",
    )
    parser.add_argument(
        "--finetune-data-path",
        type=Path,
        default=None,
        help="Path to the finetuning targets parquet.",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        help="Path to the joblib scaler used for normalized targets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where artifacts will be written.",
    )

    parser.add_argument(
        "--pretrain-task",
        dest="pretrain_tasks",
        action="append",
        default=None,
        help="Pretrain task name. Provide multiple times to override the default list.",
    )
    parser.add_argument(
        "--finetune-task",
        dest="finetune_tasks",
        action="append",
        default=None,
        help="Finetune task name. Provide multiple times to override the default list.",
    )
    parser.add_argument(
        "--task-sequence",
        nargs="+",
        help="Explicit pretrain task ordering. Defaults to a run-specific random permutation.",
    )
    parser.add_argument(
        "--task-masking-ratios",
        nargs="+",
        default=None,
        metavar="VALUE or TASK=VALUE",
        help="Either a single float applied to all tasks or key=value pairs (repeat for multiple tasks).",
    )
    parser.add_argument(
        "--swap-train-val-split",
        type=float,
        default=0.0,
        help="Fraction of samples to swap between train/val splits after initial split (0-1).",
    )
    parser.add_argument("--num-pretrain-runs", type=int, default=10, help="Number of random pretrain runs to execute.")
    parser.add_argument("--pretrain-max-epochs", type=int, default=200, help="Maximum epochs for each pretrain stage.")
    parser.add_argument("--finetune-max-epochs", type=int, default=200, help="Maximum epochs for each finetune stage.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for all dataloaders.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--random-seed-base", type=int, default=1729, help="Base seed for run/task shuffling.")
    parser.add_argument(
        "--datamodule-random-seed",
        type=int,
        default=42,
        help="Random seed provided to CompoundDataModule for all stochastic operations.",
    )
    parser.add_argument("--pretrain-sample", type=int, help="Optional sample size cap for the pretrain table.")
    parser.add_argument("--finetune-sample", type=int, help="Optional sample size cap for the finetune table.")
    parser.add_argument("--log-every-n-steps", type=int, default=20, help="Trainer logging frequency.")
    parser.add_argument(
        "--shared-block-dims",
        nargs="+",
        type=int,
        default=[190, 256, 128],
        help="Shared encoder dims.",
    )
    parser.add_argument(
        "--encoder-config",
        help=(
            "JSON string or path to JSON file describing the foundation encoder "
            '(e.g. \'{"type": "transformer", "d_model": 256, "nhead": 8}\'). '
            "Defaults to the legacy MLP when omitted."
        ),
    )
    parser.add_argument("--head-hidden-dim", type=int, default=64, help="Hidden dimension for task heads.")
    parser.add_argument(
        "--head-lr",
        type=float,
        help="Override learning rate for dynamically constructed regression heads (defaults to model-config behavior).",
    )
    parser.add_argument(
        "--disable-normalized-targets",
        action="store_true",
        help="Train on raw targets (scaler not required).",
    )
    parser.add_argument(
        "--keep-normalized-targets",
        action="store_true",
        help="Skip inverse-transform when writing predictions even if scalers are available.",
    )
    parser.add_argument(
        "--no-freeze-shared-encoder",
        dest="freeze_shared_encoder",
        action="store_false",
        help="Allow shared encoder to update during finetune stages.",
    )
    parser.add_argument(
        "--enable-learnable-loss",
        action="store_true",
        help="Enable learnable loss balancing across tasks.",
    )
    parser.add_argument(
        "--no-quiet-logging",
        dest="quiet_logging",
        action="store_false",
        help="Enable verbose model logging.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help=(
            "Validation split size passed to CompoundDataModule "
            "(only applied when the attributes table lacks a 'split' column)."
        ),
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help=(
            "Test split size passed to CompoundDataModule "
            "(only applied when the attributes table lacks a 'split' column)."
        ),
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="If set, use the entire dataset for testing (train/val empty).",
    )
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator argument.")
    parser.add_argument(
        "--devices",
        default="auto",
        help="Device specification forwarded to Lightning Trainer (e.g. 'auto', 1, '1,2').",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Patience parameter for EarlyStopping on validation loss.",
    )

    preliminary, _ = parser.parse_known_args(args=args)
    config_extras: dict[str, Any] = {}
    if preliminary.config_file is not None:
        config_mapping = _load_config_file(preliminary.config_file)
        config_extras = _apply_config_defaults(parser, config_mapping)

    parsed = parser.parse_args(args=args)

    required_fields = ["descriptor_path", "pretrain_data_path", "finetune_data_path", "output_dir"]
    missing = [field for field in required_fields if getattr(parsed, field) is None]
    if missing:
        missing_flags = ", ".join(f"--{field.replace('_', '-')}" for field in missing)
        raise SystemExit(f"Missing required arguments: {missing_flags}. Provide them via CLI or config file.")

    pretrain_tasks = parsed.pretrain_tasks or list(DEFAULT_PRETRAIN_TASKS)
    finetune_tasks = parsed.finetune_tasks or list(DEFAULT_FINETUNE_TASKS)
    task_masking = _normalize_task_masking(parsed.task_masking_ratios)
    encoder_config_value: Mapping[str, Any] | None = None
    if isinstance(parsed.encoder_config, Mapping):
        encoder_config_value = parsed.encoder_config
    elif parsed.encoder_config:
        candidate_path = Path(parsed.encoder_config)
        if candidate_path.exists():
            try:
                encoder_config_value = json.loads(candidate_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse encoder config JSON from file {candidate_path}: {exc}") from exc
        else:
            try:
                encoder_config_value = json.loads(parsed.encoder_config)
            except json.JSONDecodeError:
                lowered = parsed.encoder_config.strip().lower()
                if lowered in {"mlp", "transformer"}:
                    encoder_config_value = {"type": lowered}
                else:
                    raise SystemExit(
                        "Failed to parse --encoder-config. Provide a JSON object or path to a JSON file; "
                        f"received: {parsed.encoder_config!r}"
                    ) from None

    return SuiteConfig(
        descriptor_path=parsed.descriptor_path,
        pretrain_data_path=parsed.pretrain_data_path,
        finetune_data_path=parsed.finetune_data_path,
        scaler_path=parsed.scaler_path,
        output_dir=parsed.output_dir,
        use_normalized_targets=not parsed.disable_normalized_targets,
        keep_normalized_targets=parsed.keep_normalized_targets,
        freeze_shared_encoder=parsed.freeze_shared_encoder,
        enable_learnable_loss=parsed.enable_learnable_loss,
        quiet_model_logging=parsed.quiet_logging,
        shared_block_dims=tuple(parsed.shared_block_dims),
        encoder_config=encoder_config_value,
        head_hidden_dim=parsed.head_hidden_dim,
        head_lr=parsed.head_lr,
        num_pretrain_runs=parsed.num_pretrain_runs,
        pretrain_max_epochs=parsed.pretrain_max_epochs,
        finetune_max_epochs=parsed.finetune_max_epochs,
        batch_size=parsed.batch_size,
        num_workers=parsed.num_workers,
        log_every_n_steps=parsed.log_every_n_steps,
        random_seed_base=parsed.random_seed_base,
        datamodule_random_seed=parsed.datamodule_random_seed,
        pretrain_sample=parsed.pretrain_sample,
        finetune_sample=parsed.finetune_sample,
        pretrain_tasks=pretrain_tasks,
        finetune_tasks=finetune_tasks,
        task_sequence=parsed.task_sequence,
        task_masking_ratios=task_masking,
        swap_train_val_split=parsed.swap_train_val_split,
        val_split=parsed.val_split,
        test_split=parsed.test_split,
        test_all=parsed.test_all,
        accelerator=parsed.accelerator,
        devices=_infer_devices(parsed.devices),
        early_stopping_patience=parsed.early_stopping_patience,
    )


def _load_config_file(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise SystemExit(
                "Reading YAML config files requires the 'PyYAML' package. Install it or use TOML instead."
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".toml":
        try:
            import tomllib  # type: ignore[attr-defined]
        except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
            try:
                import tomli as tomllib  # type: ignore
            except ModuleNotFoundError as exc:
                raise SystemExit("Reading TOML config files requires Python 3.11+ or the 'tomli' package.") from exc
        data = tomllib.loads(text)
    else:
        raise SystemExit(f"Unsupported config file extension '{suffix}'. Use a .yaml or .toml file.")

    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise SystemExit(f"Top-level structure in {config_path} must be a mapping/object.")
    return data


def _apply_config_defaults(parser: argparse.ArgumentParser, config_mapping: Mapping[str, Any]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    action_by_dest = {
        action.dest: action
        for action in parser._actions
        if action.dest and action.dest != argparse.SUPPRESS  # type: ignore[attr-defined]
    }
    defaults: dict[str, Any] = {}
    for key, value in config_mapping.items():
        action = action_by_dest.get(key)
        if action is None:
            extras[key] = value
            continue
        defaults[key] = _coerce_config_value(action, value)
    if defaults:
        parser.set_defaults(**defaults)
    return extras


def _coerce_config_value(action: argparse.Action, value: Any) -> Any:
    if _is_boolean_flag(action):
        return _to_bool(value)

    if isinstance(action, argparse._AppendAction):
        items = value if isinstance(value, (list, tuple)) else [value]
        return [_coerce_scalar_value(action, item) for item in items]

    if action.nargs in ("+", "*"):
        items = value if isinstance(value, (list, tuple)) else [value]
        return [_coerce_scalar_value(action, item) for item in items]

    return _coerce_scalar_value(action, value)


def _coerce_scalar_value(action: argparse.Action, value: Any) -> Any:
    converter = getattr(action, "type", None)
    if converter is None:
        return value
    if converter is Path:
        if isinstance(value, Path):
            return value
        return Path(value)
    try:
        if isinstance(value, converter):  # type: ignore[arg-type]
            return value
    except TypeError:
        pass
    return converter(value)


def _is_boolean_flag(action: argparse.Action) -> bool:
    return (
        action.nargs == 0
        and isinstance(getattr(action, "const", None), bool)
        and isinstance(getattr(action, "default", None), bool)
    )


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_task_masking(values: Any) -> float | Dict[str, float] | None:
    if values is None:
        return None
    if isinstance(values, (float, int)):
        return float(values)
    if isinstance(values, Mapping):
        return {str(k): float(v) for k, v in values.items()}
    if isinstance(values, str):
        values = [values]
    flattened: list[str] = []
    if isinstance(values, Sequence):
        for entry in values:
            flattened.extend(part for part in str(entry).split(",") if part)
    else:
        raise ValueError(
            f"Unsupported task_masking_ratios value: {values!r}. Expected float, mapping, or list of strings."
        )
    if not flattened:
        return None
    if len(flattened) == 1 and "=" not in flattened[0]:
        return float(flattened[0])
    ratios: dict[str, float] = {}
    for item in flattened:
        if "=" not in item:
            raise ValueError(f"Invalid task_masking_ratios entry '{item}'. Expected float or task=ratio format.")
        task, value = item.split("=", 1)
        ratios[task.strip()] = float(value)
    return ratios


def _infer_devices(devices_arg: str | int | None) -> str | int | list[int] | None:
    if devices_arg is None:
        return None
    if isinstance(devices_arg, int):
        return devices_arg
    text = str(devices_arg).strip()
    if not text or text.lower() == "none":
        return None
    if text.lower() == "auto":
        return "auto"
    if text.isdigit():
        return int(text)
    # Support comma-separated GPU indices.
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if all(part.isdigit() for part in parts):
        return [int(part) for part in parts]
    return text


def main() -> None:
    # Configure logging for distributed training - only rank 0 outputs to console
    is_distributed = dist is not None and dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    if rank != 0:
        # Non-rank 0 processes: silence loguru output to avoid duplication
        fm_logger.remove()
        # Add a no-op handler to prevent errors if fm_logger is called
        fm_logger.add(lambda _: None, level="INFO")

    config = parse_arguments()
    runner = DynamicTaskSuiteRunner(config)
    summary_path = runner.run()

    # Only rank 0 prints final summary
    if rank == 0:
        fm_logger.info(f"Experiment summary written to {summary_path}")


if __name__ == "__main__":
    main()
