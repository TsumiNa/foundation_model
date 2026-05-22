# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Continual multi-task learning demo with rehearsal (5% replay).

Comprehensive end-to-end exercise of the composition-keyed data stack + FlexibleMultiTaskModel
across all task types. Tasks are added one at a time (mimicking dynamic finetuning); the
built-in autoencoder head stays on the whole time, and each time a new task head is appended,
previously learned tasks are *not* fully retrained — instead each keeps only ~``replay_ratio``
of its valid training targets active per epoch (via per-task ``task_masking_ratio``), striking a
balance between catastrophic forgetting and training cost.

After every step the new head's performance and the degradation of all previously learned heads
are evaluated on the fixed test split and plotted.

Run via the wrapper:
    ./run_continual_rehearsal_demo.sh samples/continual_rehearsal_demo_config_smoke.toml
or directly:
    python -m foundation_model.scripts.continual_rehearsal_demo --config-file <toml>
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless

import joblib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score  # type: ignore[import-untyped]

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import (
    ClassificationTaskConfig,
    KernelRegressionTaskConfig,
    MLPEncoderConfig,
    OptimizerConfig,
    RegressionTaskConfig,
)

# --- Task catalogue (columns in the qc_ac_te_mp DOS/material dataset) ---------
# Each task owns its target column; kernel-regression tasks add a t (x-axis) column.
TASK_SPECS: dict[str, dict[str, Any]] = {
    "density": {"kind": "reg", "column": "Density (normalized)"},
    "formation_energy": {"kind": "reg", "column": "Formation energy per atom (normalized)"},
    "dos_density": {"kind": "kr", "column": "DOS density (normalized)", "t_column": "DOS energy"},
    "power_factor": {"kind": "kr", "column": "Power factor (normalized)", "t_column": "Power factor (T/K)"},
    "material_type": {"kind": "clf", "column": "Material type (label)", "num_classes": 5},
}
DEFAULT_SEQUENCE = ["density", "formation_energy", "dos_density", "power_factor", "material_type"]


@dataclass
class ContinualRehearsalConfig:
    """Configuration for the continual rehearsal demo."""

    data_path: Path = Path("data/qc_ac_te_mp_dos_reformat_20250615_enforce_quaternary_test.pd.parquet")
    descriptor_path: Path = Path("data/qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet")
    preprocessing_path: Path | None = Path("data/preprocessing_objects_20250615.pkl.z")
    output_dir: Path = Path("artifacts/continual_rehearsal")

    task_sequence: list[str] = field(default_factory=lambda: list(DEFAULT_SEQUENCE))
    replay_ratio: float = 0.05  # fraction of a learned task's valid train targets kept per step
    sample: int | None = None  # optional cap on total compositions (for fast runs)

    max_epochs_per_step: int = 10
    batch_size: int = 256
    num_workers: int = 0
    early_stopping_patience: int = 0  # 0 disables early stopping (fixed epochs)

    latent_dim: int = 128
    encoder_hidden: int = 256
    head_hidden_dim: int = 64
    head_lr: float = 5e-3
    encoder_lr: float = 5e-3
    n_kernel: int = 15
    kr_lr: float = 5e-4
    kr_decay: float = 5e-5

    random_seed: int = 2025
    datamodule_random_seed: int = 42
    accelerator: str = "cpu"
    devices: int = 1

    def __post_init__(self) -> None:
        unknown = [t for t in self.task_sequence if t not in TASK_SPECS]
        if unknown:
            raise ValueError(f"Unknown task(s) {unknown}. Available: {sorted(TASK_SPECS)}")
        if not 0.0 < self.replay_ratio <= 1.0:
            raise ValueError("replay_ratio must be in (0, 1].")


def _as_float_array(cell: Any) -> np.ndarray:
    """Coerce a sequence cell (ndarray/list/str) to a 1D float array."""
    if isinstance(cell, str):
        cell = ast.literal_eval(cell)
    return np.asarray(cell, dtype=float).ravel()


def _init_kernels(t_values: np.ndarray, n_kernel: int) -> tuple[list[float], list[float]]:
    """Quantile-spaced kernel centres + uniform sigmas from the training t distribution."""
    t = np.asarray(t_values, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return [], []
    centers = np.quantile(t, np.linspace(0.0, 1.0, n_kernel))
    span = max((float(t.max()) - float(t.min())) / max(n_kernel - 1, 1), 1e-3)
    sigmas = np.full(n_kernel, span, dtype=float)
    return centers.tolist(), sigmas.tolist()


class ContinualRehearsalRunner:
    def __init__(self, config: ContinualRehearsalConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_data()

    # ------------------------------------------------------------------ data

    def _load_data(self) -> None:
        cfg = self.config
        logger.info("Loading descriptors and properties...")
        descriptors = pd.read_parquet(cfg.descriptor_path)
        properties = pd.read_parquet(cfg.data_path)

        if cfg.preprocessing_path is not None and Path(cfg.preprocessing_path).exists():
            dropped = joblib.load(cfg.preprocessing_path).get("dropped_idx", [])
            properties = properties.loc[~properties.index.isin(dropped)]
            logger.info(f"Dropped {len(dropped)} preprocessing-flagged rows.")

        common = descriptors.index.intersection(properties.index)
        if cfg.sample is not None and cfg.sample < len(common):
            common = pd.Index(
                pd.Series(common).sample(n=cfg.sample, random_state=cfg.datamodule_random_seed).to_numpy()
            )
        descriptors = descriptors.loc[common]
        properties = properties.loc[common]
        descriptors.index = descriptors.index.astype(str)
        properties.index = properties.index.astype(str)

        self.descriptors = descriptors
        self.properties = properties
        self.x_dim = descriptors.shape[1]
        self.composition_column: str = str(descriptors.index.name) if descriptors.index.name is not None else "id"
        logger.info(f"Aligned {len(common)} compositions; descriptor dim = {self.x_dim}.")
        if "split" in properties.columns:
            logger.info(f"Split counts: {properties['split'].value_counts().to_dict()}")

        # Precompute a label -> original-index map for the descriptor_fn (no matrix copy).
        self._label_by_str = {str(idx): idx for idx in self.descriptors.index}

    def descriptor_fn(self, compositions: list[str]) -> pd.DataFrame:
        labels = [self._label_by_str[c] for c in compositions if c in self._label_by_str]
        return self.descriptors.loc[labels]

    def _task_frame(self, task_name: str) -> pd.DataFrame:
        """Composition-indexed frame holding this task's column(s) + split."""
        spec = TASK_SPECS[task_name]
        cols = [spec["column"]]
        if spec["kind"] == "kr":
            cols.append(spec["t_column"])
        if "split" in self.properties.columns:
            cols.append("split")
        return self.properties[cols].copy()

    # ------------------------------------------------------------------ configs

    def _build_task_config(self, task_name: str):
        cfg = self.config
        spec = TASK_SPECS[task_name]
        ld, hd = cfg.latent_dim, cfg.head_hidden_dim
        if spec["kind"] == "reg":
            return RegressionTaskConfig(
                name=task_name,
                data_column=spec["column"],
                dims=[ld, hd, 1],
                optimizer=OptimizerConfig(lr=cfg.head_lr, weight_decay=1e-5),
            )
        if spec["kind"] == "clf":
            return ClassificationTaskConfig(
                name=task_name,
                data_column=spec["column"],
                dims=[ld, hd, 32],
                num_classes=spec["num_classes"],
                optimizer=OptimizerConfig(lr=cfg.head_lr, weight_decay=1e-5),
            )
        # kernel regression
        train_t = self._collect_train_t(task_name)
        centers, sigmas = _init_kernels(train_t, cfg.n_kernel)
        return KernelRegressionTaskConfig(
            name=task_name,
            data_column=spec["column"],
            t_column=spec["t_column"],
            x_dim=[ld, 128, 64],
            t_dim=[16, 8],
            kernel_num_centers=cfg.n_kernel,
            kernel_centers_init=centers or None,
            kernel_sigmas_init=sigmas or None,
            kernel_learnable_centers=True,
            kernel_learnable_sigmas=True,
            enable_mu3=False,
            optimizer=OptimizerConfig(lr=cfg.kr_lr, weight_decay=cfg.kr_decay),
        )

    def _collect_train_t(self, task_name: str) -> np.ndarray:
        spec = TASK_SPECS[task_name]
        frame = self.properties
        mask = frame[spec["column"]].notna()
        if "split" in frame.columns:
            mask &= frame["split"] == "train"
        t_cells = frame.loc[mask, spec["t_column"]].dropna()
        if t_cells.empty:
            return np.array([])
        return np.concatenate([_as_float_array(c) for c in t_cells])

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        cfg = self.config
        seed_everything(cfg.random_seed, workers=True)

        encoder_config = MLPEncoderConfig(hidden_dims=[self.x_dim, cfg.encoder_hidden, cfg.latent_dim])
        model = FlexibleMultiTaskModel(
            task_configs=[],
            encoder_config=encoder_config,
            enable_autoencoder=True,
            shared_block_optimizer=OptimizerConfig(lr=cfg.encoder_lr, weight_decay=1e-2),
        )

        task_configs: dict[str, Any] = {}
        # metric_history[task] = list of (step_index, metric_value)
        metric_history: dict[str, list[tuple[int, float]]] = {name: [] for name in cfg.task_sequence}
        records: list[dict[str, Any]] = []

        for step, task_name in enumerate(cfg.task_sequence):
            logger.info(f"=== Step {step + 1}/{len(cfg.task_sequence)}: add task '{task_name}' ===")
            new_cfg = self._build_task_config(task_name)
            task_configs[task_name] = new_cfg
            model.add_task(new_cfg)

            # Rehearsal: new task full, previously learned tasks keep replay_ratio of train targets.
            active = cfg.task_sequence[: step + 1]
            for name in active:
                task_configs[name].task_masking_ratio = 1.0 if name == task_name else cfg.replay_ratio

            datamodule = CompoundDataModule(
                task_configs=[task_configs[name] for name in active],
                descriptor_fn=self.descriptor_fn,
                task_frames={name: self._task_frame(name) for name in active},
                composition_column=self.composition_column,
                random_seed=cfg.datamodule_random_seed,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )

            callbacks: list[Any] = []
            if cfg.early_stopping_patience > 0:
                callbacks.append(EarlyStopping(monitor="val_final_loss", patience=cfg.early_stopping_patience))
            trainer = Trainer(
                max_epochs=cfg.max_epochs_per_step,
                accelerator=cfg.accelerator,
                devices=cfg.devices,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=callbacks,
            )
            trainer.fit(model, datamodule=datamodule)

            step_dir = self.output_dir / f"step{step + 1:02d}_{task_name}"
            step_dir.mkdir(parents=True, exist_ok=True)
            step_metrics: dict[str, dict[str, float]] = {}
            for name in active:
                metric, primary = self._evaluate_task(model, name, step_dir, is_new=(name == task_name))
                step_metrics[name] = metric
                metric_history[name].append((step + 1, primary))

            records.append(
                {"step": step + 1, "new_task": task_name, "active_tasks": list(active), "metrics": step_metrics}
            )
            logger.info(f"Step {step + 1} metrics: { {k: round(v['primary'], 4) for k, v in step_metrics.items()} }")

        self._plot_forgetting(metric_history)
        summary_path = self.output_dir / "experiment_records.json"
        summary_path.write_text(json.dumps(records, indent=2))
        logger.info(f"Done. Summary: {summary_path}")

    # ------------------------------------------------------------------ eval

    def _test_rows(self, task_name: str) -> pd.Index:
        spec = TASK_SPECS[task_name]
        frame = self.properties
        mask = frame[spec["column"]].notna()
        if "split" in frame.columns:
            mask &= frame["split"] == "test"
        return frame.index[mask]

    def _evaluate_task(
        self, model: FlexibleMultiTaskModel, task_name: str, step_dir: Path, *, is_new: bool
    ) -> tuple[dict[str, float], float]:
        spec = TASK_SPECS[task_name]
        kind = spec["kind"]
        comps = list(self._test_rows(task_name))
        model.eval()
        device = next(model.parameters()).device
        if not comps:
            return {"primary": float("nan"), "samples": 0}, float("nan")

        x = torch.tensor(self.descriptors.loc[comps].values, dtype=torch.float32, device=device)
        head = model.task_heads[task_name]

        with torch.no_grad():
            # Evaluate the target head directly off the shared latent so other (e.g. kernel)
            # heads — which require their own t_sequences — are not invoked.
            h_task = torch.tanh(model.encoder(x))

            if kind == "reg":
                pred = head(h_task).squeeze(-1).cpu().numpy()
                true = self.properties.loc[comps, spec["column"]].astype(float).to_numpy()
                metric = {
                    "r2": float(r2_score(true, pred)),
                    "mae": float(mean_absolute_error(true, pred)),
                    "samples": len(comps),
                }
                metric["primary"] = metric["r2"]
                if is_new:
                    self._plot_parity(true, pred, task_name, metric["r2"], step_dir)
                return metric, metric["r2"]

            if kind == "clf":
                logits = head(h_task)
                pred = logits.argmax(dim=-1).cpu().numpy()
                true = self.properties.loc[comps, spec["column"]].astype(int).to_numpy()
                acc = float(accuracy_score(true, pred))
                metric = {
                    "accuracy": acc,
                    "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
                    "samples": len(comps),
                    "primary": acc,
                }
                if is_new:
                    self._plot_confusion(true, pred, task_name, acc, step_dir, spec["num_classes"])
                return metric, acc

            # kernel regression: build per-sample t lists, evaluate concatenated points
            t_col = spec["t_column"]
            keep, t_list, true_parts = [], [], []
            for comp in comps:
                y_cell = self.properties.at[comp, spec["column"]]
                t_cell = self.properties.at[comp, t_col]
                if y_cell is None or t_cell is None:
                    continue
                y_arr, t_arr = _as_float_array(y_cell), _as_float_array(t_cell)
                if y_arr.size == 0 or y_arr.size != t_arr.size:
                    continue
                keep.append(comp)
                t_list.append(torch.tensor(t_arr, dtype=torch.float32, device=device))
                true_parts.append(y_arr)
            if not keep:
                return {"primary": float("nan"), "samples": 0}, float("nan")
            xk = torch.tensor(self.descriptors.loc[keep].values, dtype=torch.float32, device=device)
            h_k = torch.tanh(model.encoder(xk))
            expanded_h, expanded_t = model._expand_for_kernel_regression(h_k, t_list)
            pred = head(expanded_h, t=expanded_t).squeeze(-1).cpu().numpy()
            true = np.concatenate(true_parts)
            metric = {
                "r2": float(r2_score(true, pred)),
                "mae": float(mean_absolute_error(true, pred)),
                "samples": len(keep),
                "points": int(true.size),
                "primary": float(r2_score(true, pred)),
            }
            if is_new:
                self._plot_kr_sequences(keep, t_list, true_parts, pred, task_name, metric["r2"], step_dir)
            return metric, metric["primary"]

    # ------------------------------------------------------------------ plots

    def _plot_parity(self, true, pred, task_name, r2, step_dir):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
        ax.scatter(true, pred, s=8, alpha=0.4, edgecolor="none")
        lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.set_title(f"{task_name} (new) — R²={r2:.3f}, n={len(true)}")
        fig.tight_layout()
        fig.savefig(step_dir / f"{task_name}_parity.png")
        plt.close(fig)

    def _plot_confusion(self, true, pred, task_name, acc, step_dir, num_classes):
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(true, pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
        fig, ax = plt.subplots(figsize=(5, 4.5), dpi=130)
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ax.set_title(f"{task_name} (new) — acc={acc:.3f}, n={int(cm.sum())}")
        fig.tight_layout()
        fig.savefig(step_dir / f"{task_name}_confusion.png")
        plt.close(fig)

    def _plot_kr_sequences(self, comps, t_list, true_parts, pred, task_name, r2, step_dir):
        # Split the concatenated prediction back into per-sample chunks.
        fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
        offset = 0
        for i in range(min(3, len(comps))):
            n = true_parts[i].size
            t = t_list[i].cpu().numpy()
            ax.plot(t, true_parts[i], lw=1.2, alpha=0.8, label=f"true #{i}")
            ax.plot(t, pred[offset : offset + n], lw=1.0, ls="--", alpha=0.8, label=f"pred #{i}")
            offset += n
        ax.set_xlabel("t")
        ax.set_ylabel("value (normalized)")
        ax.set_title(f"{task_name} (new) — R²={r2:.3f} over all points")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(step_dir / f"{task_name}_sequences.png")
        plt.close(fig)

    def _plot_forgetting(self, metric_history: dict[str, list[tuple[int, float]]]):
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
        for task_name, points in metric_history.items():
            if not points:
                continue
            steps = [s for s, _ in points]
            vals = [v for _, v in points]
            ax.plot(steps, vals, marker="o", label=task_name)
        ax.set_xlabel("finetuning step")
        ax.set_ylabel("primary metric (R² / accuracy)")
        ax.set_title("Per-task performance vs continual finetuning step (forgetting)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output_dir / "forgetting_trajectory.png")
        plt.close(fig)
        logger.info(f"Saved forgetting trajectory to {self.output_dir / 'forgetting_trajectory.png'}")


# --- CLI ---------------------------------------------------------------------


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _parse_args(argv: list[str] | None = None) -> ContinualRehearsalConfig:
    parser = argparse.ArgumentParser(description="Continual multi-task rehearsal demo.")
    parser.add_argument("--config-file", type=Path, default=None, help="TOML config path.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--max-epochs-per-step", type=int, default=None)
    parser.add_argument("--replay-ratio", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--task-sequence", nargs="+", default=None)
    args = parser.parse_args(argv)

    data = _load_toml(args.config_file) if args.config_file else {}
    # CLI overrides take precedence over the config file.
    for key in (
        "output_dir",
        "sample",
        "max_epochs_per_step",
        "replay_ratio",
        "batch_size",
        "accelerator",
        "task_sequence",
    ):
        val = getattr(args, key)
        if val is not None:
            data[key] = val

    field_names = {f for f in ContinualRehearsalConfig.__dataclass_fields__}
    path_fields = {"data_path", "descriptor_path", "preprocessing_path", "output_dir"}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_names:
            logger.warning(f"Ignoring unknown config key '{key}'.")
            continue
        kwargs[key] = Path(value) if key in path_fields and value is not None else value
    return ContinualRehearsalConfig(**kwargs)


def main(argv: list[str] | None = None) -> None:
    config = _parse_args(argv)
    ContinualRehearsalRunner(config).run()


if __name__ == "__main__":
    main()
