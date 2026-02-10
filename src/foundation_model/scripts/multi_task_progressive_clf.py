#!/usr/bin/env python
# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Progressive multi-task pretrain → classification finetune CLI.

Source tasks: 9 Regression + 7 Kernel Regression (configurable via --source-task)
Target task:  1 Classification (material_type)

Usage:
    python -m foundation_model.scripts.multi_task_progressive_clf \
        --data-path       ../data/qc_ac_te_mp_dos_reformat_20250615_enforce_quaternary_test.pd.parquet \
        --descriptor-path ../data/qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet \
        --preprocessing-path ../data/preprocessing_objects_20250615.pkl.z \
        --output-dir      logs/multi_task_suite/my_run \
        --n-runs 10 --pretrain-max-epochs 200 --finetune-max-epochs 200

Or via the entry-point (after `pip install -e .`):
    fm-progressive-clf --config-file samples/progressive_clf_config.toml
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from loguru import logger as fm_logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import (
    ClassificationTaskConfig,
    EncoderType,
    KernelRegressionTaskConfig,
    MLPEncoderConfig,
    OptimizerConfig,
    RegressionTaskConfig,
    TaskType,
    TransformerEncoderConfig,
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

# ---------------------------------------------------------------------------
# Kernel initialization helper (adapted from notebooks/KRFD_utils.py)
# ---------------------------------------------------------------------------


def _init_centers_and_sigmas(
    t_train: np.ndarray,
    n_kernels: int,
    sigma_alpha: float = 0.5,
    inverse_density: bool = False,
    density_smoothing: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantile-based kernel center/sigma initialisation."""
    t = np.asarray(t_train).flatten()
    bins = 64
    hist, edges = np.histogram(t, bins=bins, density=True)

    if inverse_density:
        pdf = 1.0 / (hist + 1e-8)
    else:
        pdf = hist + 1e-8

    if density_smoothing > 0:
        pdf = pdf ** (1.0 - density_smoothing)
    pdf = pdf / pdf.sum()

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    cum_pdf = np.cumsum(pdf)
    cum_pdf = cum_pdf / cum_pdf[-1]

    qs = np.linspace(0.0, 1.0, num=n_kernels)
    centers = np.interp(qs, cum_pdf, bin_centers)
    centers = torch.tensor(np.sort(centers), dtype=torch.float32)

    c = centers.numpy()
    M = len(c)
    if M == 1:
        return centers, torch.tensor([max(1e-3, sigma_alpha)], dtype=torch.float32)
    d = np.zeros(M)
    d[0] = c[1] - c[0]
    d[-1] = c[-1] - c[-2]
    for i in range(1, M - 1):
        d[i] = 0.5 * ((c[i] - c[i - 1]) + (c[i + 1] - c[i]))
    sigmas = torch.tensor(np.maximum(1e-3, sigma_alpha * d), dtype=torch.float32)
    return centers, sigmas


# ---------------------------------------------------------------------------
# Source task registry
# ---------------------------------------------------------------------------

# Regression tasks: name → data_column
REGRESSION_TASK_REGISTRY: dict[str, str] = {
    "density": "Density (normalized)",
    "efermi": "Efermi (normalized)",
    "final_energy": "Final energy per atom (normalized)",
    "formation_energy": "Formation energy per atom (normalized)",
    "total_magnetization": "Total magnetization (normalized)",
    "volume": "Volume (normalized)",
    "dielectric_total": "Dielectric total (normalized)",
    "dielectric_ionic": "Dielectric ionic (normalized)",
    "dielectric_electronic": "Dielectric electronic (normalized)",
}

# Kernel regression tasks: name → (data_column, t_column)
KERNEL_REGRESSION_TASK_REGISTRY: dict[str, tuple[str, str]] = {
    "electrical_resistivity": ("Electrical resistivity (normalized)", "Electrical resistivity (T/K)"),
    "power_factor": ("Power factor (normalized)", "Power factor (T/K)"),
    "seebeck_coefficient": ("Seebeck coefficient (normalized)", "Seebeck coefficient (T/K)"),
    "thermal_conductivity": ("Thermal conductivity (normalized)", "Thermal conductivity (T/K)"),
    "zt": ("ZT (normalized)", "ZT (T/K)"),
    "magnetic_susceptibility": ("Magnetic susceptibility (normalized)", "Magnetic susceptibility (T/K)"),
    "dos_density": ("DOS density (normalized)", "DOS energy"),
}

ALL_SOURCE_TASK_NAMES: list[str] = list(REGRESSION_TASK_REGISTRY) + list(KERNEL_REGRESSION_TASK_REGISTRY)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProgressiveClfConfig:
    """Configuration for the progressive pretrain → classification finetune pipeline."""

    data_path: Path = Path("data/qc_ac_te_mp_dos_reformat_20250615_enforce_quaternary_test.pd.parquet")
    descriptor_path: Path = Path("data/qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet")
    preprocessing_path: Path = Path("data/preprocessing_objects_20250615.pkl.z")
    output_dir: Path | None = None

    n_runs: int = 10
    pretrain_max_epochs: int = 200
    finetune_max_epochs: int = 200
    batch_size: int = 128
    num_workers: int = 0
    random_seed_base: int = 1729
    datamodule_random_seed: int = 42
    early_stopping_patience: int = 50
    log_every_n_steps: int = 50

    latent_dim: int = 128
    head_hidden_dim: int = 64
    head_lr: float = 0.005
    n_kernel: int = 15
    kr_lr: float = 5e-4
    kr_decay: float = 5e-5

    source_tasks: list[str] = field(default_factory=lambda: list(ALL_SOURCE_TASK_NAMES))

    accelerator: str = "auto"
    devices: str | int | list[int] | None = "auto"

    quiet_model_logging: bool = True

    def __post_init__(self) -> None:
        if self.n_runs <= 0:
            raise ValueError("n_runs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        all_known = set(REGRESSION_TASK_REGISTRY) | set(KERNEL_REGRESSION_TASK_REGISTRY)
        unknown = sorted(set(self.source_tasks) - all_known)
        if unknown:
            raise ValueError(f"Unknown source tasks: {unknown}. Known: {sorted(all_known)}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ProgressiveClfRunner:
    """Progressive multi-task pretrain → classification finetune runner."""

    def __init__(self, config: ProgressiveClfConfig) -> None:
        self.config = config

        if config.quiet_model_logging:
            fm_logger.disable("foundation_model")
        else:
            fm_logger.enable("foundation_model")

        # Load data and preprocessing objects
        self.preprocessing_objects = joblib.load(config.preprocessing_path)
        self.all_data = self._load_data()
        self.desc_trans = pd.read_parquet(config.descriptor_path)
        self.x_dim = self.desc_trans.shape[1]

        # Label encoder
        le = self.preprocessing_objects.get("material_type_label_encoder")
        if le is not None:
            self.num_classes = len(le.classes_)
            self.class_names = list(le.classes_)
        else:
            labels = self.all_data["Material type (label)"].dropna().astype(int)
            self.num_classes = int(labels.max()) + 1
            self.class_names = [str(i) for i in range(self.num_classes)]

        # Encoder config
        self.encoder_config = MLPEncoderConfig(
            hidden_dims=[self.x_dim, 256, config.latent_dim],
            norm=True,
            residual=False,
        )

        # Build task configs
        self.source_task_configs = self._build_source_task_configs()
        self.finetune_task_config = self._build_finetune_task_config()
        self.source_task_names = config.source_tasks

        # Pre-compute element count for ge4 filtering
        self.n_elements_series = self.all_data["elements"].apply(len)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        all_data = pd.read_parquet(self.config.data_path)
        dropped_idx = self.preprocessing_objects.get("dropped_idx", [])
        all_data = all_data.loc[~all_data.index.isin(dropped_idx)]
        return all_data

    # ------------------------------------------------------------------
    # Task configs
    # ------------------------------------------------------------------

    def _build_source_task_configs(self) -> dict[str, RegressionTaskConfig | KernelRegressionTaskConfig]:
        cfg = self.config
        ld = cfg.latent_dim
        hd = cfg.head_hidden_dim
        nk = cfg.n_kernel

        configs: dict[str, RegressionTaskConfig | KernelRegressionTaskConfig] = {}

        for name in cfg.source_tasks:
            if name in REGRESSION_TASK_REGISTRY:
                data_col = REGRESSION_TASK_REGISTRY[name]
                configs[name] = RegressionTaskConfig(
                    name=name,
                    data_column=data_col,
                    dims=[ld, hd, 1],
                    norm=True,
                    residual=False,
                    optimizer=OptimizerConfig(lr=cfg.head_lr, weight_decay=1e-5),
                )
            elif name in KERNEL_REGRESSION_TASK_REGISTRY:
                data_col, t_col = KERNEL_REGRESSION_TASK_REGISTRY[name]
                # Initialize kernel centers/sigmas from data
                t_data = self.all_data[t_col].dropna()
                if len(t_data) == 0:
                    raise ValueError(f"No data found for t_column '{t_col}' of task '{name}'.")
                t_all = np.concatenate(t_data.values)
                centers, sigmas = _init_centers_and_sigmas(t_all, nk, inverse_density=True)
                configs[name] = KernelRegressionTaskConfig(
                    name=name,
                    data_column=data_col,
                    t_column=t_col,
                    x_dim=[ld, 128, 64],
                    t_dim=[16, 8],
                    norm=True,
                    residual=False,
                    kernel_num_centers=nk,
                    kernel_learnable_centers=True,
                    kernel_learnable_sigmas=True,
                    kernel_centers_init=centers.tolist(),
                    kernel_sigmas_init=sigmas.tolist(),
                    enable_mu3=False,
                    optimizer=OptimizerConfig(lr=cfg.kr_lr, weight_decay=cfg.kr_decay),
                )
            else:
                known = sorted(set(REGRESSION_TASK_REGISTRY) | set(KERNEL_REGRESSION_TASK_REGISTRY))
                raise ValueError(f"Unknown source task '{name}'. Known tasks: {known}")

        return configs

    def _build_finetune_task_config(self) -> ClassificationTaskConfig:
        cfg = self.config
        return ClassificationTaskConfig(
            name="material_type",
            data_column="Material type (label)",
            dims=[cfg.latent_dim, cfg.head_hidden_dim, 32],
            num_classes=self.num_classes,
            norm=True,
            residual=False,
            optimizer=OptimizerConfig(lr=cfg.head_lr, weight_decay=1e-5),
        )

    # ------------------------------------------------------------------
    # DataModule builders
    # ------------------------------------------------------------------

    def _get_task_columns(self, task_cfgs: list) -> set[str]:
        columns: set[str] = set()
        for tc in task_cfgs:
            columns.add(tc.data_column)
            if isinstance(tc, KernelRegressionTaskConfig):
                columns.add(tc.t_column)
        return columns

    def _build_pretrain_datamodule(self, task_cfgs: list) -> CompoundDataModule:
        columns = list(self._get_task_columns(task_cfgs))
        if "split" in self.all_data.columns:
            columns.append("split")
        columns = list(dict.fromkeys(columns))
        attributes = self.all_data[columns].copy()

        return CompoundDataModule(
            formula_desc_source=self.desc_trans,
            attributes_source=attributes,
            task_configs=list(task_cfgs),
            random_seed=self.config.datamodule_random_seed,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def _build_finetune_datamodule(self) -> CompoundDataModule:
        data_col = self.finetune_task_config.data_column
        columns = [data_col]
        if "split" in self.all_data.columns:
            columns.append("split")
        attributes = self.all_data[columns].dropna(subset=[data_col]).copy()

        return CompoundDataModule(
            formula_desc_source=self.desc_trans,
            attributes_source=attributes,
            task_configs=[self.finetune_task_config],
            random_seed=self.config.datamodule_random_seed,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_clf_predictions(batch_preds: list) -> tuple[np.ndarray, np.ndarray | None]:
        labels, probas = [], []
        for batch in batch_preds:
            if "material_type_label" in batch:
                labels.append(batch["material_type_label"])
            if "material_type_proba" in batch:
                probas.append(batch["material_type_proba"])
        pred_labels = np.concatenate(labels) if labels else np.array([])
        pred_probas = np.concatenate(probas) if probas else None
        return pred_labels, pred_probas

    @staticmethod
    def _collect_clf_true(dataset) -> np.ndarray:
        y_list = []
        for i in range(len(dataset)):
            sample = dataset[i]
            y = sample[1]["material_type"]
            y_list.append(y.numpy())
        return np.concatenate(y_list)

    @staticmethod
    def _safe_slug(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug or "task"

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> Path:
        cfg = self.config
        _ = seed_everything(cfg.random_seed_base)

        output_dir = cfg.output_dir or Path(f"logs/multi_task_suite/{datetime.now().strftime('%m%d_%H%M')}")
        output_dir.mkdir(parents=True, exist_ok=True)

        experiment_records: list[dict[str, Any]] = []
        accuracy_matrix: list[list[float | None]] = []

        print(f"Output directory: {output_dir}")
        print(f"Source tasks: {self.source_task_names}")
        print(f"Classification classes: {self.class_names} ({self.num_classes})")
        print(
            f"N_RUNS={cfg.n_runs}, pretrain_epochs={cfg.pretrain_max_epochs}, finetune_epochs={cfg.finetune_max_epochs}"
        )

        for run_idx in range(1, cfg.n_runs + 1):
            run_label = f"run{run_idx:02d}"
            run_root = output_dir / run_label
            run_root.mkdir(parents=True, exist_ok=True)

            rng = random.Random(cfg.random_seed_base + run_idx)
            task_sequence = rng.sample(self.source_task_names, k=len(self.source_task_names))

            print(f"\n{'=' * 70}")
            print(f"Run {run_idx}/{cfg.n_runs}: task order = {task_sequence}")
            print(f"{'=' * 70}")

            previous_checkpoint: str | None = None
            run_records: dict[str, Any] = {
                "run": run_label,
                "task_sequence": task_sequence,
                "pretrain": [],
                "finetune": [],
            }
            run_accuracies: list[float | None] = []

            for stage_idx, new_task_name in enumerate(task_sequence, start=1):
                stage_tasks = task_sequence[:stage_idx]
                stage_task_cfgs = [copy.deepcopy(self.source_task_configs[n]) for n in stage_tasks]

                print(f"\n--- Pretrain Stage {stage_idx}/{len(task_sequence)}: {stage_tasks} ---")

                # ====== 1. Pretrain ======
                previous_checkpoint = self._run_pretrain_stage(
                    stage_task_cfgs=stage_task_cfgs,
                    previous_checkpoint=previous_checkpoint,
                    stage_dir=run_root / f"pretrain_stage{stage_idx:02d}_{self._safe_slug(new_task_name)}",
                    new_task_name=new_task_name,
                )

                run_records["pretrain"].append(
                    {
                        "stage": stage_idx,
                        "new_task": new_task_name,
                        "accumulated_tasks": list(stage_tasks),
                        "checkpoint": previous_checkpoint,
                    }
                )

                # ====== 2. Finetune to Classification ======
                if previous_checkpoint is None:
                    print("  Skipping finetune: no checkpoint available.")
                    run_accuracies.append(None)
                    continue

                stage_dir = run_root / f"pretrain_stage{stage_idx:02d}_{self._safe_slug(new_task_name)}"
                ft_results = self._run_finetune_eval(
                    previous_checkpoint=previous_checkpoint,
                    stage_dir=stage_dir,
                    stage_idx=stage_idx,
                    stage_tasks=stage_tasks,
                    run_idx=run_idx,
                )

                run_accuracies.append(ft_results["test_accuracy"])
                run_records["finetune"].append(ft_results)

            accuracy_matrix.append(run_accuracies)
            experiment_records.append(run_records)

        # Save experiment summary
        summary_path = output_dir / "experiment_records.json"
        with open(summary_path, "w") as f:
            json.dump(experiment_records, f, indent=2, default=str)

        print(f"\nExperiment records saved to {summary_path}")
        print("Per-stage clf reports saved as clf_report_all.json / clf_report_ge4.json under each finetune dir.")
        return summary_path

    # ------------------------------------------------------------------
    # Pretrain stage
    # ------------------------------------------------------------------

    def _run_pretrain_stage(
        self,
        *,
        stage_task_cfgs: list,
        previous_checkpoint: str | None,
        stage_dir: Path,
        new_task_name: str,
    ) -> str | None:
        cfg = self.config
        stage_dir.mkdir(parents=True, exist_ok=True)

        dm = self._build_pretrain_datamodule(stage_task_cfgs)

        if previous_checkpoint is None:
            model = FlexibleMultiTaskModel(
                task_configs=stage_task_cfgs,
                encoder_config=copy.deepcopy(self.encoder_config),
                enable_learnable_loss_balancer=True,
                shared_block_optimizer=OptimizerConfig(lr=5e-2),
            )
        else:
            model = FlexibleMultiTaskModel.load_from_checkpoint(
                checkpoint_path=previous_checkpoint,
                strict=False,
                enable_learnable_loss_balancer=True,
                encoder_config=copy.deepcopy(self.encoder_config),
            )
            existing = set(model.task_heads.keys())
            new_cfgs = [c for c in stage_task_cfgs if c.name not in existing]
            if new_cfgs:
                model.add_task(*new_cfgs)

        ckpt_cb = ModelCheckpoint(
            dirpath=stage_dir / "checkpoints",
            filename=f"{self._safe_slug(new_task_name)}-{{epoch:02d}}-{{val_final_loss:.4f}}",
            monitor="val_final_loss",
            mode="min",
            save_top_k=1,
        )
        early_stop = EarlyStopping(
            monitor="val_final_loss",
            mode="min",
            patience=cfg.early_stopping_patience,
        )
        loggers = [
            CSVLogger(stage_dir / "logs", name="csv"),
            TensorBoardLogger(stage_dir / "logs", name="tensorboard"),
        ]
        trainer = Trainer(
            max_epochs=cfg.pretrain_max_epochs,
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            callbacks=[ckpt_cb, early_stop],
            logger=loggers,
            log_every_n_steps=cfg.log_every_n_steps,
        )
        trainer.fit(model, datamodule=dm)

        best_path = ckpt_cb.best_model_path
        if best_path:
            state = torch.load(best_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state.get("state_dict", state))
        print(f"  Pretrain done. Best checkpoint: {best_path or '<none>'}")
        return best_path or previous_checkpoint

    # ------------------------------------------------------------------
    # Finetune + evaluate
    # ------------------------------------------------------------------

    def _run_finetune_eval(
        self,
        *,
        previous_checkpoint: str,
        stage_dir: Path,
        stage_idx: int,
        stage_tasks: list[str],
        run_idx: int,
    ) -> dict[str, Any]:
        cfg = self.config

        print("  Finetuning to classification...")
        ft_model = FlexibleMultiTaskModel.load_from_checkpoint(
            checkpoint_path=previous_checkpoint,
            strict=False,
            enable_learnable_loss_balancer=True,
            freeze_shared_encoder=True,
            encoder_config=copy.deepcopy(self.encoder_config),
            shared_block_optimizer=OptimizerConfig(lr=5e-2),
        )
        active_heads = list(ft_model.task_heads.keys())
        if active_heads:
            ft_model.remove_tasks(*active_heads)
        ft_model.add_task(copy.deepcopy(self.finetune_task_config))

        ft_dm = self._build_finetune_datamodule()
        ft_dir = stage_dir / "finetune" / "material_type"
        ft_dir.mkdir(parents=True, exist_ok=True)

        ft_ckpt = ModelCheckpoint(
            dirpath=ft_dir / "checkpoints",
            filename="material_type-{epoch:02d}-{val_final_loss:.4f}",
            monitor="val_final_loss",
            mode="min",
            save_top_k=1,
        )
        ft_early = EarlyStopping(
            monitor="val_final_loss",
            mode="min",
            patience=cfg.early_stopping_patience,
        )
        ft_loggers = [
            CSVLogger(ft_dir / "logs", name="csv"),
            TensorBoardLogger(ft_dir / "logs", name="tensorboard"),
        ]
        ft_trainer = Trainer(
            max_epochs=cfg.finetune_max_epochs,
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            callbacks=[ft_ckpt, ft_early],
            logger=ft_loggers,
            log_every_n_steps=cfg.log_every_n_steps,
        )
        ft_trainer.fit(ft_model, datamodule=ft_dm)

        ft_best = ft_ckpt.best_model_path
        if ft_best:
            state = torch.load(ft_best, map_location="cpu", weights_only=True)
            ft_model.load_state_dict(state.get("state_dict", state))

        # === Evaluate ===
        ft_dm.setup(stage="fit")
        ft_dm.setup(stage="test")

        test_preds = ft_trainer.predict(ft_model, dataloaders=ft_dm.test_dataloader())
        train_preds = ft_trainer.predict(ft_model, dataloaders=ft_dm.train_dataloader())

        test_labels, test_probas = self._collect_clf_predictions(test_preds)
        test_true = self._collect_clf_true(ft_dm.test_dataset)
        train_labels, _ = self._collect_clf_predictions(train_preds)
        train_true = self._collect_clf_true(ft_dm.train_dataset)

        test_mask = test_true.flatten() != -100
        train_mask = train_true.flatten() != -100

        # n_elements >= 4 mask
        test_n_elem = self.n_elements_series.loc[ft_dm.test_idx].values
        ge4_mask = test_n_elem >= 4
        test_ge4_mask = test_mask & ge4_mask

        test_acc = float(accuracy_score(test_true.flatten()[test_mask], test_labels.flatten()[test_mask]))
        train_acc = float(accuracy_score(train_true.flatten()[train_mask], train_labels.flatten()[train_mask]))

        # Classification reports
        clf_report_all = classification_report(
            test_true.flatten()[test_mask],
            test_labels.flatten()[test_mask],
            target_names=self.class_names,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        n_ge4 = int(test_ge4_mask.sum())
        if n_ge4 > 0:
            clf_report_ge4 = classification_report(
                test_true.flatten()[test_ge4_mask],
                test_labels.flatten()[test_ge4_mask],
                target_names=self.class_names,
                digits=4,
                output_dict=True,
                zero_division=0,
            )
        else:
            clf_report_ge4 = {}

        print(f"  Stage {stage_idx}: test_acc={test_acc:.4f}, train_acc={train_acc:.4f}")
        print(f"  [ALL test] ({test_mask.sum()} samples)")
        print(
            classification_report(
                test_true.flatten()[test_mask],
                test_labels.flatten()[test_mask],
                target_names=self.class_names,
                digits=4,
                zero_division=0,
            )
        )
        if n_ge4 > 0:
            test_acc_ge4 = accuracy_score(
                test_true.flatten()[test_ge4_mask],
                test_labels.flatten()[test_ge4_mask],
            )
            print(f"  [n_elements >= 4] ({n_ge4} samples, acc={test_acc_ge4:.4f})")
            print(
                classification_report(
                    test_true.flatten()[test_ge4_mask],
                    test_labels.flatten()[test_ge4_mask],
                    target_names=self.class_names,
                    digits=4,
                    zero_division=0,
                )
            )

        # Confusion matrix
        cm = confusion_matrix(test_true.flatten()[test_mask], test_labels.flatten()[test_mask])
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Run {run_idx} Stage {stage_idx} (tasks: {stage_tasks})")
        fig.savefig(ft_dir / "confusion_matrix.png", bbox_inches="tight")
        plt.close(fig)

        # Save to disk
        ft_results: dict[str, Any] = {
            "stage": stage_idx,
            "pretrain_tasks": list(stage_tasks),
            "test_accuracy": test_acc,
            "train_accuracy": train_acc,
            "checkpoint": ft_best or None,
        }
        with open(ft_dir / "clf_report_all.json", "w") as f:
            json.dump(clf_report_all, f, indent=2, default=str)
        with open(ft_dir / "clf_report_ge4.json", "w") as f:
            json.dump(clf_report_ge4, f, indent=2, default=str)
        with open(ft_dir / "metrics.json", "w") as f:
            json.dump(ft_results, f, indent=2, default=str)

        print(f"  Saved: clf_report_all.json, clf_report_ge4.json, metrics.json → {ft_dir}")
        return ft_results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_arguments(args: Sequence[str] | None = None) -> ProgressiveClfConfig:
    parser = argparse.ArgumentParser(
        description="Progressive multi-task pretrain → classification finetune.",
    )
    parser.add_argument("--config-file", type=Path, help="TOML/YAML config file with argument defaults.")
    parser.add_argument("--data-path", type=Path, help="Path to the main data parquet.")
    parser.add_argument("--descriptor-path", type=Path, help="Path to the descriptor parquet.")
    parser.add_argument("--preprocessing-path", type=Path, help="Path to the preprocessing pickle (.pkl.z).")
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (default: logs/multi_task_suite/<timestamp>)."
    )
    parser.add_argument("--n-runs", type=int, default=10, help="Number of random task-order repeats.")
    parser.add_argument("--pretrain-max-epochs", type=int, default=200, help="Max pretrain epochs per stage.")
    parser.add_argument("--finetune-max-epochs", type=int, default=200, help="Max finetune epochs per stage.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--random-seed-base", type=int, default=1729, help="Base random seed.")
    parser.add_argument("--datamodule-random-seed", type=int, default=42, help="DataModule random seed.")
    parser.add_argument("--early-stopping-patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--log-every-n-steps", type=int, default=50, help="Trainer logging frequency.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Shared encoder latent dim.")
    parser.add_argument("--head-hidden-dim", type=int, default=64, help="Task head hidden dim.")
    parser.add_argument("--head-lr", type=float, default=0.005, help="Task head learning rate.")
    parser.add_argument("--n-kernel", type=int, default=15, help="Number of kernel centres.")
    parser.add_argument("--kr-lr", type=float, default=5e-4, help="Kernel regression head learning rate.")
    parser.add_argument("--kr-decay", type=float, default=5e-5, help="Kernel regression weight decay.")
    parser.add_argument(
        "--source-task",
        dest="source_tasks",
        action="append",
        default=None,
        help=(
            f"Source task name (repeat for multiple). Default: all tasks. Available: {', '.join(ALL_SOURCE_TASK_NAMES)}"
        ),
    )
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator.")
    parser.add_argument("--devices", default="auto", help="Lightning devices.")
    parser.add_argument(
        "--no-quiet-logging", dest="quiet_logging", action="store_false", help="Enable verbose logging."
    )

    # Parse config file first for defaults
    preliminary, _ = parser.parse_known_args(args=args)
    config_mapping: Mapping[str, Any] = {}
    if preliminary.config_file is not None:
        config_mapping = _load_config_file(preliminary.config_file)
        _apply_config_defaults(parser, config_mapping)

    parsed = parser.parse_args(args=args)

    # Required paths check
    for field_name in ["data_path", "descriptor_path", "preprocessing_path"]:
        if getattr(parsed, field_name) is None:
            parser.error(f"--{field_name.replace('_', '-')} is required.")

    # source_tasks priority: CLI --source-task > TOML source_tasks > ALL
    if parsed.source_tasks is not None:
        source_tasks = parsed.source_tasks
    elif "source_tasks" in config_mapping:
        source_tasks = list(config_mapping["source_tasks"])
    else:
        source_tasks = list(ALL_SOURCE_TASK_NAMES)
    devices = _infer_devices(parsed.devices)

    return ProgressiveClfConfig(
        data_path=parsed.data_path,
        descriptor_path=parsed.descriptor_path,
        preprocessing_path=parsed.preprocessing_path,
        output_dir=parsed.output_dir,
        n_runs=parsed.n_runs,
        pretrain_max_epochs=parsed.pretrain_max_epochs,
        finetune_max_epochs=parsed.finetune_max_epochs,
        batch_size=parsed.batch_size,
        num_workers=parsed.num_workers,
        random_seed_base=parsed.random_seed_base,
        datamodule_random_seed=parsed.datamodule_random_seed,
        early_stopping_patience=parsed.early_stopping_patience,
        log_every_n_steps=parsed.log_every_n_steps,
        latent_dim=parsed.latent_dim,
        head_hidden_dim=parsed.head_hidden_dim,
        head_lr=parsed.head_lr,
        n_kernel=parsed.n_kernel,
        kr_lr=parsed.kr_lr,
        kr_decay=parsed.kr_decay,
        source_tasks=source_tasks,
        accelerator=parsed.accelerator,
        devices=devices,
        quiet_model_logging=parsed.quiet_logging,
    )


# ---------------------------------------------------------------------------
# Config file loading (same pattern as dynamic_task_suite.py)
# ---------------------------------------------------------------------------


def _load_config_file(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML config files.") from exc
        data = yaml.safe_load(text)
    elif suffix == ".toml":
        import tomllib

        data = tomllib.loads(text)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file must be a mapping, got {type(data).__name__}")
    return data


def _apply_config_defaults(parser: argparse.ArgumentParser, config_mapping: Mapping[str, Any]) -> None:
    action_by_dest = {
        action.dest: action for action in parser._actions if action.dest and action.dest != argparse.SUPPRESS
    }
    defaults: dict[str, Any] = {}
    for key, value in config_mapping.items():
        dest = key.replace("-", "_")
        if dest == "source_tasks":
            continue  # handled separately (append action conflicts with set_defaults)
        if dest in action_by_dest:
            defaults[dest] = value
    if defaults:
        parser.set_defaults(**defaults)


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
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if all(part.isdigit() for part in parts):
        return [int(p) for p in parts]
    return text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    torch.set_float32_matmul_precision("medium")
    config = parse_arguments()
    runner = ProgressiveClfRunner(config)
    summary_path = runner.run()
    print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
