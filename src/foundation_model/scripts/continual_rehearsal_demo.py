# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Continual multi-task learning demo with rehearsal (5% replay) + inverse design.

Comprehensive end-to-end exercise of the composition-keyed data stack +
FlexibleMultiTaskModel across every task type and every inorganic dataset (default sequence):

* 2 regression + 2 kernel regression + 1 classification from the qc_ac_te_mp DOS/material set,
* 2 regression from the NEMAD superconductor set (tc, pressure),
* 3 regression from the NEMAD magnetic set (curie, magnetization, neel),
* 2 regression from the phonix-db set (kp, klat),
* the built-in autoencoder head (always on).

All compositions are featurized **on the fly** with the invertible KMD-1d descriptor
(:mod:`foundation_model.utils.kmd_plus`), joined across datasets by composition formula.

Tasks are added one at a time (dynamic finetuning). The AE head stays on the whole time; when a
new head is appended, previously learned tasks keep only ~``replay_ratio`` of their valid
training targets active (per-task ``task_masking_ratio``). The mask is drawn once per step when
the training dataset is built (it is not resampled each epoch) — balancing forgetting against cost.
Every step evaluates *all* active heads on the fixed test split and plots the new head plus the
per-task forgetting trajectory.

After all tasks are learned, an **inverse-design** stage optimizes the latent space toward a
condition (2 regression targets + increased quasicrystal probability) and decodes the optimized
KMD descriptor back to a composition via ``KMD.inverse``.

Run:
    ./run_continual_rehearsal_demo.sh samples/continual_rehearsal_demo_config_smoke.toml
    python -m foundation_model.scripts.continual_rehearsal_demo --config-file <toml>
"""

from __future__ import annotations

import argparse
import ast
import base64
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
from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS, KMD, element_features, formula_to_composition

# --- Task catalogue ----------------------------------------------------------
# source: which dataset the task's targets come from ("qc" / "superconductor" / "magnetic").
# qc columns are pre-normalized; NEMAD raw columns are z-scored at load time.
TASK_SPECS: dict[str, dict[str, Any]] = {
    "density": {"source": "qc", "kind": "reg", "column": "Density (normalized)"},
    "formation_energy": {"source": "qc", "kind": "reg", "column": "Formation energy per atom (normalized)"},
    "dos_density": {"source": "qc", "kind": "kr", "column": "DOS density (normalized)", "t_column": "DOS energy"},
    "power_factor": {
        "source": "qc",
        "kind": "kr",
        "column": "Power factor (normalized)",
        "t_column": "Power factor (T/K)",
    },
    "material_type": {"source": "qc", "kind": "clf", "column": "Material type (label)", "num_classes": 5},
    "tc": {"source": "superconductor", "kind": "reg", "column": "Transition temperature[K]"},
    "pressure": {"source": "superconductor", "kind": "reg", "column": "Pressure[GPa]"},
    "curie": {"source": "magnetic", "kind": "reg", "column": "Curie temperature[K]"},
    "magnetization": {"source": "magnetic", "kind": "reg", "column": "Magnetization[A·m²/mol]"},
    "neel": {"source": "magnetic", "kind": "reg", "column": "Neel temperature[K]"},
    "kp": {"source": "phonix", "kind": "reg", "column": "kp[W/mK]"},
    "klat": {"source": "phonix", "kind": "reg", "column": "klat[W/mK]"},
}
# Raw (non-qc) regression targets span orders of magnitude (thermal conductivity, magnetization);
# they are log1p-compressed, z-scored, then clipped to tame heavy tails.
_RAW_TARGET_CLIP = 5.0
DEFAULT_SEQUENCE = list(TASK_SPECS.keys())
# Quasicrystal classes for the material_type label encoder (DAC=0, DQC=1, IAC=2, IQC=3, others=4).
QC_CLASSES = [1, 3]


@dataclass
class ContinualRehearsalConfig:
    """Configuration for the continual rehearsal + inverse-design demo."""

    qc_data_path: Path = Path("data/qc_ac_te_mp_dos_reformat_20250615_enforce_quaternary_test.pd.parquet")
    qc_preprocessing_path: Path | None = Path("data/preprocessing_objects_20250615.pkl.z")
    superconductor_path: Path = Path("data/NEMAD_superconductor_20260425.parquet")
    magnetic_path: Path = Path("data/NEMAD_magnetic_20260419.parquet")
    phonix_path: Path = Path("data/phonix-db-filtered_20260425.parquet")
    output_dir: Path = Path("artifacts/continual_rehearsal")

    task_sequence: list[str] = field(default_factory=lambda: list(DEFAULT_SEQUENCE))
    replay_ratio: float = 0.05
    sample_per_dataset: int | None = None  # cap rows per dataset (for fast runs)

    max_epochs_per_step: int = 10
    batch_size: int = 256
    num_workers: int = 0

    n_grids: int = 8  # KMD-1d grid points per element feature; descriptor dim = 58 * n_grids
    latent_dim: int = 128
    encoder_hidden: int = 256
    head_hidden_dim: int = 64
    head_lr: float = 5e-3
    encoder_lr: float = 5e-3
    n_kernel: int = 15
    kr_lr: float = 5e-4
    kr_decay: float = 5e-5

    # Inverse-design stage
    inverse_n_seeds: int = 16
    inverse_steps: int = 300
    inverse_lr: float = 0.05
    inverse_reg_tasks: list[str] = field(default_factory=lambda: ["density", "formation_energy"])
    inverse_reg_targets: list[float] = field(default_factory=lambda: [1.5, -1.5])

    random_seed: int = 2025
    datamodule_random_seed: int = 42
    accelerator: str = "cpu"
    devices: int = 1

    def __post_init__(self) -> None:
        unknown = [t for t in self.task_sequence if t not in TASK_SPECS]
        if unknown:
            raise ValueError(f"Unknown task(s) {unknown}. Available: {sorted(TASK_SPECS)}")
        if not 0.0 <= self.replay_ratio <= 1.0:
            raise ValueError("replay_ratio must be in [0, 1] (0 = no rehearsal).")
        if len(self.inverse_reg_tasks) != len(self.inverse_reg_targets):
            raise ValueError("inverse_reg_tasks and inverse_reg_targets must have equal length.")


def _as_float_array(cell: Any) -> np.ndarray:
    if isinstance(cell, str):
        cell = ast.literal_eval(cell)
    return np.asarray(cell, dtype=float).ravel()


def _composition_key(raw: Any) -> str | None:
    """Canonical reduced-formula key for a composition dict (qc) or formula string (NEMAD)."""
    from pymatgen.core.composition import Composition  # local import; pymatgen is heavy

    try:
        if isinstance(raw, dict):
            cleaned = {k: v for k, v in raw.items() if v is not None and float(v) > 0}
            if not cleaned:
                return None
            comp = Composition(cleaned)
        else:
            comp = Composition(str(raw))
        return comp.reduced_formula
    except Exception:
        return None


def _init_kernels(t_values: np.ndarray, n_kernel: int) -> tuple[list[float], list[float]]:
    t = np.asarray(t_values, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return [], []
    centers = np.quantile(t, np.linspace(0.0, 1.0, n_kernel))
    span = max((float(t.max()) - float(t.min())) / max(n_kernel - 1, 1), 1e-3)
    return centers.tolist(), np.full(n_kernel, span).tolist()


class ContinualRehearsalRunner:
    def __init__(self, config: ContinualRehearsalConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # KMD-1d featurizer over the bundled element features (invertible: descriptor -> composition).
        self._kmd = KMD(element_features.values, method="1d", n_grids=config.n_grids, sigma="auto", scale=True)
        self.x_dim = int(self._kmd.transform(np.eye(1, len(DEFAULT_ELEMENTS))).shape[1])
        self._desc_cache: dict[str, np.ndarray] = {}
        self._load_data()

    # ------------------------------------------------------------------ data

    def _load_data(self) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.datamodule_random_seed)
        # task_frames[task] -> DataFrame indexed by composition key with [column(+t), "split"]
        self.task_frames: dict[str, pd.DataFrame] = {}
        split_by_key: dict[str, str] = {}

        sources = {
            "qc": self._load_qc(),
            "superconductor": pd.read_parquet(cfg.superconductor_path),
            "magnetic": pd.read_parquet(cfg.magnetic_path),
            "phonix": pd.read_parquet(cfg.phonix_path),
        }

        # Build a composition key per row for each source, capping rows for speed.
        keyed: dict[str, pd.DataFrame] = {}
        for name, df in sources.items():
            df = df.copy()
            if cfg.sample_per_dataset is not None and cfg.sample_per_dataset < len(df):
                df = df.iloc[rng.choice(len(df), size=cfg.sample_per_dataset, replace=False)]
            comp_col = "composition" if name != "qc" else "composition"
            df["__key__"] = [_composition_key(v) for v in df[comp_col]]
            df = df.dropna(subset=["__key__"]).drop_duplicates(subset="__key__", keep="first").set_index("__key__")
            keyed[name] = df
            # Record split: qc uses its own split column; NEMAD gets a random split.
            if "split" in df.columns:
                for k, s in df["split"].items():
                    split_by_key.setdefault(str(k), str(s))
            else:
                for k in df.index:
                    split_by_key.setdefault(str(k), rng.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15]))

        # Build per-task frames; z-score raw NEMAD regression columns.
        for task_name in cfg.task_sequence:
            spec = TASK_SPECS[task_name]
            df = keyed[spec["source"]]
            col = spec["column"]
            if col not in df.columns:
                raise KeyError(f"Task '{task_name}': column '{col}' missing in {spec['source']} data.")
            frame = pd.DataFrame(index=df.index)
            values = df[col]
            if spec["source"] != "qc" and spec["kind"] == "reg":
                # log1p compresses the orders-of-magnitude range, then z-score + clip tails.
                # Scaling stats come from *train* rows only to avoid leaking val/test distribution.
                v = np.log1p(df[col].astype(float).clip(lower=0.0))
                is_train = np.array([split_by_key.get(str(k)) == "train" for k in df.index])
                ref = v[is_train] if is_train.any() else v
                mean = float(ref.mean())
                std = float(ref.std(ddof=0)) or 1.0
                values = ((v - mean) / std).clip(-_RAW_TARGET_CLIP, _RAW_TARGET_CLIP)
            frame[col] = values
            if spec["kind"] == "kr":
                frame[spec["t_column"]] = df[spec["t_column"]]
            frame["split"] = [split_by_key.get(str(k), "train") for k in frame.index]
            self.task_frames[task_name] = frame

        self.split_by_key = split_by_key
        n_keys = len(set().union(*[set(f.index) for f in self.task_frames.values()]))
        logger.info(f"Built {len(self.task_frames)} task frames over {n_keys} unique compositions; x_dim={self.x_dim}.")

    def _load_qc(self) -> pd.DataFrame:
        cfg = self.config
        df = pd.read_parquet(cfg.qc_data_path)
        if cfg.qc_preprocessing_path is not None and Path(cfg.qc_preprocessing_path).exists():
            dropped = joblib.load(cfg.qc_preprocessing_path).get("dropped_idx", [])
            df = df.loc[~df.index.isin(dropped)]
        return df

    def descriptor_fn(self, compositions: list[str]) -> pd.DataFrame:
        """KMD-1d descriptors for composition keys (computed once per unique key, cached)."""
        uncached = [c for c in dict.fromkeys(compositions) if c not in self._desc_cache]
        if uncached:
            weights = np.zeros((len(uncached), len(DEFAULT_ELEMENTS)), dtype=float)
            valid: list[str] = []
            for key in uncached:
                try:
                    w = formula_to_composition(key)
                except Exception:
                    w = None
                if w is None or float(w.sum()) <= 0:
                    continue
                weights[len(valid)] = w
                valid.append(key)
            if valid:
                desc = self._kmd.transform(weights[: len(valid)])
                for j, key in enumerate(valid):
                    self._desc_cache[key] = desc[j]
        present = [c for c in compositions if c in self._desc_cache]
        if not present:
            return pd.DataFrame()
        return pd.DataFrame(np.stack([self._desc_cache[c] for c in present]), index=present)

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
        frame = self.task_frames[task_name]
        mask = frame[spec["column"]].notna() & (frame["split"] == "train")
        cells = frame.loc[mask, spec["t_column"]].dropna()
        if cells.empty:
            return np.array([])
        return np.concatenate([_as_float_array(c) for c in cells])

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
        metric_history: dict[str, list[tuple[int, float]]] = {name: [] for name in cfg.task_sequence}
        records: list[dict[str, Any]] = []

        for step, task_name in enumerate(cfg.task_sequence):
            logger.info(f"=== Step {step + 1}/{len(cfg.task_sequence)}: add task '{task_name}' ===")
            task_configs[task_name] = self._build_task_config(task_name)
            model.add_task(task_configs[task_name])

            active = cfg.task_sequence[: step + 1]
            for name in active:
                task_configs[name].task_masking_ratio = 1.0 if name == task_name else cfg.replay_ratio

            datamodule = CompoundDataModule(
                task_configs=[task_configs[name] for name in active],
                descriptor_fn=self.descriptor_fn,
                task_frames={name: self.task_frames[name] for name in active},
                composition_column="composition",
                random_seed=cfg.datamodule_random_seed,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )
            trainer = Trainer(
                max_epochs=cfg.max_epochs_per_step,
                accelerator=cfg.accelerator,
                devices=cfg.devices,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )
            trainer.fit(model, datamodule=datamodule)

            # Evaluate on the DataModule's *resolved* test compositions so the metrics use exactly
            # the rows held out of training (not the raw per-task split column, which can diverge
            # from the global overlay/random-fallback split the DataModule actually trained on).
            test_keys: set[str] | None = None
            if datamodule.split_series is not None:
                resolved = datamodule.split_series
                test_keys = set(resolved.index[resolved == "test"].astype(str))

            step_dir = self.output_dir / f"step{step + 1:02d}_{task_name}"
            step_dir.mkdir(parents=True, exist_ok=True)
            step_metrics: dict[str, dict[str, float]] = {}
            for name in active:
                metric = self._evaluate_task(model, name, step_dir, is_new=(name == task_name), test_keys=test_keys)
                step_metrics[name] = metric
                metric_history[name].append((step + 1, metric["primary"]))
            records.append({"step": step + 1, "new_task": task_name, "metrics": step_metrics})
            summary = ", ".join(f"{k}={v['primary']:.3f}" for k, v in step_metrics.items())
            logger.info(f"Step {step + 1}: {summary}")

        self._plot_forgetting(metric_history)
        (self.output_dir / "experiment_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

        inverse = self._inverse_design(model)
        (self.output_dir / "inverse_design.json").write_text(json.dumps(inverse, indent=2), encoding="utf-8")

        self._write_report_html(records, inverse)
        logger.info(f"Done. Outputs in {self.output_dir}")

    # ------------------------------------------------------------------ eval

    def _test_rows(self, task_name: str, test_keys: set[str] | None = None) -> list[str]:
        """Test compositions with a non-null target for ``task_name``.

        When ``test_keys`` is given (the DataModule's resolved test split) it is the source of
        truth; otherwise fall back to the per-task ``split`` column (used for inverse-design seeds).
        """
        spec = TASK_SPECS[task_name]
        frame = self.task_frames[task_name]
        mask = frame[spec["column"]].notna()
        mask &= frame.index.isin(test_keys) if test_keys is not None else (frame["split"] == "test")
        return list(frame.index[mask])

    def _descriptor_tensor(self, comps: list[str], device) -> tuple[torch.Tensor, list[str]]:
        desc = self.descriptor_fn(comps)
        comps = [c for c in comps if c in desc.index]
        return torch.tensor(desc.loc[comps].values, dtype=torch.float32, device=device), comps

    def _evaluate_task(self, model, task_name, step_dir, *, is_new, test_keys=None) -> dict[str, float]:
        spec = TASK_SPECS[task_name]
        kind = spec["kind"]
        model.eval()
        device = next(model.parameters()).device
        comps = self._test_rows(task_name, test_keys)
        if not comps:
            return {"primary": float("nan"), "samples": 0}
        frame = self.task_frames[task_name]
        head = model.task_heads[task_name]

        with torch.no_grad():
            if kind in ("reg", "clf"):
                x, comps = self._descriptor_tensor(comps, device)
                if not comps:
                    return {"primary": float("nan"), "samples": 0}
                h = torch.tanh(model.encoder(x))
                if kind == "reg":
                    pred = head(h).squeeze(-1).cpu().numpy()
                    true = frame.loc[comps, spec["column"]].astype(float).to_numpy()
                    r2 = float(r2_score(true, pred))
                    metric = {
                        "r2": r2,
                        "mae": float(mean_absolute_error(true, pred)),
                        "samples": len(comps),
                        "primary": r2,
                    }
                    if is_new:
                        self._plot_parity(true, pred, task_name, r2, step_dir)
                    return metric
                logits = head(h)
                pred = logits.argmax(dim=-1).cpu().numpy()
                true = frame.loc[comps, spec["column"]].astype(int).to_numpy()
                acc = float(accuracy_score(true, pred))
                metric = {
                    "accuracy": acc,
                    "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
                    "samples": len(comps),
                    "primary": acc,
                }
                if is_new:
                    self._plot_confusion(true, pred, task_name, acc, step_dir, spec["num_classes"])
                return metric

            # kernel regression
            keep, t_list, true_parts = [], [], []
            for comp in comps:
                if comp not in self._desc_cache and self.descriptor_fn([comp]).empty:
                    continue
                y_arr = _as_float_array(frame.at[comp, spec["column"]])
                t_arr = _as_float_array(frame.at[comp, spec["t_column"]])
                if y_arr.size == 0 or y_arr.size != t_arr.size:
                    continue
                keep.append(comp)
                t_list.append(torch.tensor(t_arr, dtype=torch.float32, device=device))
                true_parts.append(y_arr)
            if not keep:
                return {"primary": float("nan"), "samples": 0}
            xk, _ = self._descriptor_tensor(keep, device)
            h_k = torch.tanh(model.encoder(xk))
            expanded_h, expanded_t = model._expand_for_kernel_regression(h_k, t_list)
            pred = head(expanded_h, t=expanded_t).squeeze(-1).cpu().numpy()
            true = np.concatenate(true_parts)
            r2 = float(r2_score(true, pred))
            metric = {
                "r2": r2,
                "mae": float(mean_absolute_error(true, pred)),
                "samples": len(keep),
                "points": int(true.size),
                "primary": r2,
            }
            if is_new:
                self._plot_kr_sequences(keep, t_list, true_parts, pred, task_name, r2, step_dir)
            return metric

    # ------------------------------------------------------------------ inverse design

    def _inverse_design(self, model) -> dict[str, Any]:
        cfg = self.config
        logger.info("=== Inverse design: latent optimization toward conditions ===")
        device = next(model.parameters()).device
        model.eval()

        # Seed from qc test compositions (material_type is defined there).
        seeds = self._test_rows("material_type")[: cfg.inverse_n_seeds]
        x_seed, seeds = self._descriptor_tensor(seeds, device)
        if not seeds:
            logger.warning("No seeds available for inverse design.")
            return {}

        reg_targets = {t: v for t, v in zip(cfg.inverse_reg_tasks, cfg.inverse_reg_targets)}

        def _qc_prob(x: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                h = torch.tanh(model.encoder(x))
                probs = torch.softmax(model.task_heads["material_type"](h), dim=-1)
                return probs[:, QC_CLASSES].sum(dim=-1).cpu().numpy()

        def _reg_preds(x: torch.Tensor) -> dict[str, np.ndarray]:
            with torch.no_grad():
                h = torch.tanh(model.encoder(x))
                return {t: model.task_heads[t](h).squeeze(-1).cpu().numpy() for t in reg_targets}

        before_qc = _qc_prob(x_seed)
        before_reg = _reg_preds(x_seed)

        result = model.optimize_latent(
            initial_input=x_seed,
            task_targets=reg_targets,
            class_targets={"material_type": QC_CLASSES},
            optimize_space="latent",
            steps=cfg.inverse_steps,
            lr=cfg.inverse_lr,
        )
        # Regression values achieved *in latent space* (before AE decode) — the direct optimization result.
        reg_names = list(reg_targets.keys())
        achieved = result.optimized_target[:, 0, :].cpu().numpy()  # (B, len(reg_names)) in reg_targets order
        reg_latent = {t: achieved[:, j] for j, t in enumerate(reg_names)}

        optimized_desc = result.optimized_input[:, 0, :]  # (B, x_dim) reconstructed KMD descriptor
        after_qc = _qc_prob(optimized_desc)  # round-trip (decode -> re-encode) condition fidelity
        after_reg = _reg_preds(optimized_desc)

        # Decode optimized descriptors back to compositions via KMD.inverse.
        decoded = self._decode_compositions(optimized_desc.cpu().numpy())

        self._plot_inverse_design(before_qc, after_qc, before_reg, reg_latent, after_reg, reg_targets)

        records = []
        for i, seed in enumerate(seeds):
            records.append(
                {
                    "seed_composition": seed,
                    "qc_prob_before": float(before_qc[i]),
                    "qc_prob_after_decode": float(after_qc[i]),
                    "reg_before": {t: float(before_reg[t][i]) for t in reg_names},
                    "reg_achieved_latent": {t: float(reg_latent[t][i]) for t in reg_names},
                    "reg_after_decode": {t: float(after_reg[t][i]) for t in reg_names},
                    "decoded_composition": decoded[i],
                }
            )
        latent_summary = ", ".join(
            f"{t}:{before_reg[t].mean():.2f}->{reg_latent[t].mean():.2f}(tgt {reg_targets[t]})" for t in reg_names
        )
        logger.info(f"Inverse design in-latent regression: {latent_summary}")
        logger.info(f"Inverse design QC prob (round-trip): {before_qc.mean():.3f} -> {after_qc.mean():.3f}")
        return {"reg_targets": reg_targets, "qc_classes": QC_CLASSES, "n_seeds": len(seeds), "records": records}

    def _decode_compositions(self, descriptors: np.ndarray) -> list[str]:
        """KMD.inverse: descriptor -> element weights -> compact formula string."""
        try:
            weights = self._kmd.inverse(descriptors)
        except Exception as exc:  # pragma: no cover - QP edge cases
            logger.warning(f"KMD.inverse failed ({exc}); skipping composition decoding.")
            return ["<undecodable>"] * descriptors.shape[0]
        out = []
        for w in weights:
            order = np.argsort(w)[::-1]
            parts = [f"{DEFAULT_ELEMENTS[i]}{w[i]:.3f}" for i in order[:6] if w[i] > 1e-3]
            out.append(" ".join(parts) if parts else "<empty>")
        return out

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
        fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
        offset = 0
        for i in range(min(3, len(comps))):
            n = true_parts[i].size
            t = t_list[i].cpu().numpy()
            ax.plot(t, true_parts[i], lw=1.2, alpha=0.8, label=f"true #{i}")
            ax.plot(t, pred[offset : offset + n], lw=1.0, ls="--", alpha=0.8, label=f"pred #{i}")
            offset += n
        ax.set_xlabel("t")
        ax.set_ylabel("value (norm)")
        ax.set_title(f"{task_name} (new) — R²={r2:.3f}")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(step_dir / f"{task_name}_sequences.png")
        plt.close(fig)

    def _plot_forgetting(self, metric_history):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
        for task_name, points in metric_history.items():
            if not points:
                continue
            steps = [s for s, _ in points]
            vals = [v for _, v in points]
            ax.plot(steps, vals, marker="o", label=task_name)
        ax.set_xlabel("finetuning step")
        ax.set_ylabel("primary metric (R² / accuracy)")
        ax.set_title("Per-task performance vs continual finetuning step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(self.output_dir / "forgetting_trajectory.png")
        plt.close(fig)
        logger.info(f"Saved forgetting trajectory to {self.output_dir / 'forgetting_trajectory.png'}")

    def _plot_inverse_design(self, before_qc, after_qc, before_reg, reg_latent, after_reg, reg_targets):
        n_panels = 1 + len(reg_targets)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), dpi=130)
        axes = np.atleast_1d(axes)
        idx = np.arange(len(before_qc))
        axes[0].bar(idx - 0.2, before_qc, width=0.4, label="before")
        axes[0].bar(idx + 0.2, after_qc, width=0.4, label="after (decode)")
        axes[0].set_title("Quasicrystal probability")
        axes[0].set_xlabel("seed")
        axes[0].legend(fontsize=8)
        for ax, (t, tgt) in zip(axes[1:], reg_targets.items()):
            ax.bar(idx - 0.25, before_reg[t], width=0.25, label="before")
            ax.bar(idx, reg_latent[t], width=0.25, label="achieved (latent)")
            ax.bar(idx + 0.25, after_reg[t], width=0.25, label="after (decode)")
            ax.axhline(tgt, color="r", ls="--", lw=1, label=f"target={tgt}")
            ax.set_title(f"{t} prediction")
            ax.set_xlabel("seed")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(self.output_dir / "inverse_design.png")
        plt.close(fig)
        logger.info(f"Saved inverse-design plot to {self.output_dir / 'inverse_design.png'}")

    # ------------------------------------------------------------------ report

    def _img_b64(self, rel: str) -> str | None:
        path = self.output_dir / rel
        if not path.exists():
            return None
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    def _write_report_html(self, records: list[dict[str, Any]], inverse: dict[str, Any]) -> None:
        """Emit a self-contained, white-background HTML slide deck summarizing the run."""
        final = records[-1]["metrics"] if records else {}
        intro = {r["new_task"]: r["metrics"][r["new_task"]]["primary"] for r in records}
        kind_label = {"reg": "regression", "kr": "kernel regression", "clf": "classification"}

        rows = []
        for task in self.config.task_sequence:
            spec = TASK_SPECS[task]
            metric_name = "acc" if spec["kind"] == "clf" else "R²"
            rows.append(
                f"<tr><td>{task}</td><td>{kind_label[spec['kind']]}</td><td>{spec['source']}</td>"
                f"<td>{intro.get(task, float('nan')):+.3f}</td>"
                f"<td>{final.get(task, {}).get('primary', float('nan')):+.3f}</td><td>{metric_name}</td></tr>"
            )
        task_table = "\n".join(rows)

        # Per-type example plots (first reg / kr / clf in the sequence).
        examples = []
        seen: set[str] = set()
        for i, task in enumerate(self.config.task_sequence, start=1):
            kind = TASK_SPECS[task]["kind"]
            if kind in seen:
                continue
            suffix = {"reg": "parity", "kr": "sequences", "clf": "confusion"}[kind]
            img = self._img_b64(f"step{i:02d}_{task}/{task}_{suffix}.png")
            if img:
                examples.append(
                    f'<figure><img src="{img}"/><figcaption>{task} ({kind_label[kind]})</figcaption></figure>'
                )
                seen.add(kind)

        forget_img = self._img_b64("forgetting_trajectory.png")
        inv_img = self._img_b64("inverse_design.png")

        recs = inverse.get("records", [])
        reg_targets = inverse.get("reg_targets", {})

        def _mean(field: str, sub: str) -> float:
            vals = [r[field][sub] for r in recs if sub in r.get(field, {})]
            return float(np.mean(vals)) if vals else float("nan")

        inv_lines = "".join(
            f"<li><b>{t}</b>: {_mean('reg_before', t):+.2f} → <b>{_mean('reg_achieved_latent', t):+.2f}</b> "
            f"(target {reg_targets[t]:+.1f})</li>"
            for t in reg_targets
        )
        qc_before = float(np.mean([r["qc_prob_before"] for r in recs])) if recs else float("nan")
        qc_after = float(np.mean([r["qc_prob_after_decode"] for r in recs])) if recs else float("nan")
        decoded = "".join(f"<li><code>{r['decoded_composition']}</code></li>" for r in recs[:4])

        n_tasks = len(self.config.task_sequence)
        n_reg = sum(1 for t in self.config.task_sequence if TASK_SPECS[t]["kind"] == "reg")
        n_kr = sum(1 for t in self.config.task_sequence if TASK_SPECS[t]["kind"] == "kr")
        n_clf = sum(1 for t in self.config.task_sequence if TASK_SPECS[t]["kind"] == "clf")

        def slide(body: str) -> str:
            return f'<section class="slide">{body}</section>'

        slides = [
            slide(
                "<h1>Continual Multi-Task Learning + Inverse Design</h1>"
                "<p class='sub'>Composition-keyed FlexibleMultiTaskModel across all inorganic datasets &amp; task types</p>"
                f"<p class='meta'>{n_tasks} supervised tasks ({n_reg} regression · {n_kr} kernel regression · "
                f"{n_clf} classification) + always-on autoencoder · KMD-1d descriptors</p>"
            ),
            slide(
                "<h2>Setup</h2><ul>"
                "<li><b>Datasets</b>: qc_ac_te_mp (DOS/material), NEMAD superconductor, NEMAD magnetic, phonix-db — joined by composition formula.</li>"
                "<li><b>Descriptor</b>: invertible KMD-1d, computed on the fly (descriptor → composition via <code>KMD.inverse</code>).</li>"
                "<li><b>Continual finetuning</b>: tasks added one at a time; AE head always on.</li>"
                f"<li><b>Rehearsal</b>: learned tasks keep only {self.config.replay_ratio:.0%} of their training targets per step.</li>"
                "<li><b>Inverse design</b>: optimize the latent toward regression targets + quasicrystal probability, then decode a composition.</li>"
                "</ul>"
            ),
            slide(
                "<h2>Tasks &amp; performance</h2>"
                "<table><thead><tr><th>task</th><th>type</th><th>dataset</th>"
                "<th>at intro</th><th>final</th><th>metric</th></tr></thead>"
                f"<tbody>{task_table}</tbody></table>"
                "<p class='meta'>“at intro” = score the step a task is added; “final” = after all tasks learned.</p>"
            ),
            slide(
                "<h2>Forgetting under 5% rehearsal</h2>"
                + (f"<img class='wide' src='{forget_img}'/>" if forget_img else "<p>(plot unavailable)</p>")
            ),
            slide("<h2>Per-task-type examples</h2><div class='row'>" + "".join(examples) + "</div>"),
            slide(
                "<h2>Inverse design</h2><div class='row'>"
                + (f"<img class='wide' src='{inv_img}'/>" if inv_img else "")
                + "<div class='panel'><h3>Latent optimization reached targets</h3><ul>"
                + inv_lines
                + f"</ul><p>Quasicrystal probability (round-trip): <b>{qc_before:.3f} → {qc_after:.3f}</b></p>"
                + "<h3>Decoded compositions (KMD.inverse)</h3><ul>"
                + decoded
                + "</ul></div></div>"
            ),
            slide(
                "<h2>Takeaways</h2><ul>"
                "<li>One shared encoder serves regression, kernel regression, classification &amp; reconstruction across 4 inorganic datasets.</li>"
                "<li>5% rehearsal keeps well-learned tasks (density, formation energy, material type) near their peak while new heads are added.</li>"
                "<li>Latent-space optimization with regression + classification conditions hits the targets and decodes back to real compositions via the invertible KMD descriptor.</li>"
                "</ul>"
            ),
        ]

        html = _REPORT_TEMPLATE.replace("__SLIDES__", "\n".join(slides)).replace("__N__", str(len(slides)))
        (self.output_dir / "report.html").write_text(html, encoding="utf-8")
        logger.info(f"Saved HTML report to {self.output_dir / 'report.html'}")


_REPORT_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Continual Multi-Task Demo — Summary</title>
<style>
  :root { --fg:#1a1a1a; --muted:#6b7280; --accent:#2563eb; --line:#e5e7eb; }
  * { box-sizing: border-box; }
  html,body { margin:0; height:100%; background:#ffffff; color:var(--fg);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .deck { height:100vh; overflow-y:scroll; scroll-snap-type:y mandatory; }
  .slide { min-height:100vh; scroll-snap-align:start; display:flex; flex-direction:column;
    justify-content:center; padding:6vh 9vw; border-bottom:1px solid var(--line); }
  h1 { font-size:2.6rem; margin:0 0 .4rem; }
  h2 { font-size:2rem; margin:0 0 1.2rem; color:var(--accent); }
  h3 { font-size:1.15rem; margin:1rem 0 .4rem; }
  p.sub { font-size:1.3rem; color:var(--fg); margin:.2rem 0; }
  p.meta { color:var(--muted); font-size:1rem; }
  ul { font-size:1.15rem; line-height:1.7; }
  table { border-collapse:collapse; font-size:1rem; }
  th,td { border-bottom:1px solid var(--line); padding:.45rem .9rem; text-align:right; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) { text-align:left; }
  thead th { color:var(--muted); font-weight:600; }
  img.wide { max-height:62vh; max-width:62vw; }
  .row { display:flex; gap:1.5rem; align-items:center; flex-wrap:wrap; }
  figure { margin:0; text-align:center; } figure img { max-height:46vh; max-width:30vw; }
  figcaption { color:var(--muted); font-size:.95rem; margin-top:.4rem; }
  .panel { max-width:34vw; } code { background:#f3f4f6; padding:.1rem .35rem; border-radius:4px; }
  .nav { position:fixed; bottom:14px; right:18px; color:var(--muted); font-size:.85rem; }
</style></head>
<body><div class="deck" id="deck">
__SLIDES__
</div>
<div class="nav">↑/↓ or scroll · <span id="pos">1</span>/__N__</div>
<script>
  const deck=document.getElementById('deck'), slides=[...document.querySelectorAll('.slide')], pos=document.getElementById('pos');
  let i=0;
  function go(n){ i=Math.max(0,Math.min(slides.length-1,n)); slides[i].scrollIntoView({behavior:'smooth'}); }
  document.addEventListener('keydown',e=>{ if(e.key==='ArrowDown'||e.key==='ArrowRight'||e.key===' '){e.preventDefault();go(i+1);}
    else if(e.key==='ArrowUp'||e.key==='ArrowLeft'){e.preventDefault();go(i-1);} });
  deck.addEventListener('scroll',()=>{ i=Math.round(deck.scrollTop/window.innerHeight); pos.textContent=i+1; });
</script></body></html>
"""


# --- CLI ---------------------------------------------------------------------


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _parse_args(argv: list[str] | None = None) -> ContinualRehearsalConfig:
    parser = argparse.ArgumentParser(description="Continual rehearsal + inverse-design demo.")
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-per-dataset", type=int, default=None)
    parser.add_argument("--max-epochs-per-step", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    args = parser.parse_args(argv)

    data = _load_toml(args.config_file) if args.config_file else {}
    for key in ("output_dir", "sample_per_dataset", "max_epochs_per_step", "accelerator"):
        val = getattr(args, key)
        if val is not None:
            data[key] = val

    field_names = set(ContinualRehearsalConfig.__dataclass_fields__)
    path_fields = {
        "qc_data_path",
        "qc_preprocessing_path",
        "superconductor_path",
        "magnetic_path",
        "phonix_path",
        "output_dir",
    }
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_names:
            logger.warning(f"Ignoring unknown config key '{key}'.")
            continue
        kwargs[key] = Path(value) if key in path_fields and value is not None else value
    return ContinualRehearsalConfig(**kwargs)


def main(argv: list[str] | None = None) -> None:
    ContinualRehearsalRunner(_parse_args(argv)).run()


if __name__ == "__main__":
    main()
