"""Configuration, paired head initialization and composition-level metrics for this experiment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
import re
import tomllib
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning import seed_everything

from foundation_model.workflows._engine import build_empty_model, build_head_config, checkpoint_task_order
from foundation_model.workflows._sections import build_model_section, build_training_section
from foundation_model.workflows.recording import RunRecorder, load_checkpoint_state
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

DATE = "20261002"
SOURCE_TASKS = ("material_type", "volume", "formation_energy", "efermi", "band_gap", "seebeck", "zt")
TARGET_TASKS = ("dielectric_total", "power_factor")


@dataclass(kw_only=True)
class BenchmarkConfig:
    seeds: list[int]
    source_counts: list[int]
    fractions: list[float]
    max_epochs: int
    patience: int
    batch_size: int
    package_version: str
    image_revision: str
    image_uri: str
    sif_sha256: str
    data_manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if not self.source_counts or any(
            isinstance(k, bool) or not isinstance(k, int) or k not in range(1, 8) for k in self.source_counts
        ):
            raise ValueError("source_counts must be in 1..7")
        if len(set(self.source_counts)) != len(self.source_counts):
            raise ValueError("source_counts must be unique")
        if not self.fractions or any(
            isinstance(f, bool) or not isinstance(f, (int, float)) or not np.isfinite(f) or not 0 < f <= 1
            for f in self.fractions
        ):
            raise ValueError("fractions must be in (0, 1]")
        if len({round(f * 100) for f in self.fractions}) != len(self.fractions):
            raise ValueError("fractions must have distinct integer-percent identifiers")
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1
            for n in (self.max_epochs, self.patience, self.batch_size)
        ):
            raise ValueError("epochs, patience and batch_size must be positive integers")
        if not re.fullmatch(r"[0-9a-f]{40}", self.image_revision) or not self.image_uri.endswith(self.image_revision):
            raise ValueError("An immutable image reference matching its source revision is required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sif_sha256):
            raise ValueError("The verified deployed SIF checksum is required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.data_manifest_sha256):
            raise ValueError("The registered scientific data manifest checksum is required")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> tuple[BenchmarkConfig, dict[str, Any], dict[str, dict[str, Any]]]:
    raw = tomllib.loads(path.read_text())
    if set(raw) != {"benchmark", "model", "encoders"}:
        raise ValueError("Protocol must contain benchmark, model and encoders tables")
    config = BenchmarkConfig(**raw["benchmark"])
    if not raw["encoders"]:
        raise ValueError("At least one encoder is required")
    for name, arm in raw["encoders"].items():
        if not np.isfinite(arm["encoder_lr"]) or arm["encoder_lr"] <= 0:
            raise ValueError(f"{name}: encoder_lr must be positive and finite")
        build_model_section({**raw["model"], **{k: v for k, v in arm.items() if k != "encoder_lr"}})
    return config, raw["model"], raw["encoders"]


def workflow_config(
    protocol_path: Path,
    data_dir: Path,
    arm: str,
    seed: int,
    *,
    target: str | None = None,
    fraction: float = 1,
    mode: str = "source",
) -> dict[str, Any]:
    config, model, arms = load_protocol(protocol_path)
    if seed not in config.seeds or arm not in arms:
        raise ValueError("Unknown arm or unregistered seed")
    if mode not in {"source", "scratch", "frozen", "full"}:
        raise ValueError("Unknown training mode")
    entry = dict(arms[arm])
    encoder_lr = entry.pop("encoder_lr")
    manifest = json.loads((data_dir / f"manifest_{DATE}.json").read_text())
    raw: dict[str, Any] = {
        "data": {
            "batch_size": min(config.batch_size, manifest.get("smoke_batch_size", config.batch_size))
            if manifest.get("functional_smoke")
            else config.batch_size,
            "num_workers": 0,
            "split_random_seed": 20261002,
        },
        "descriptor": {"kind": "kmd", "n_grids": 8},
        "datasets": {"source": {"path": str(data_dir / manifest["source"]["file"])}},
        "tasks": [],
        "model": {**model, **entry},
        "training": {
            "seed": 20261002 + seed,
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": config.max_epochs,
            "encoder_lr": encoder_lr * (0.1 if mode in {"frozen", "full"} else 1),
            "head_lr": 0.002,
            "kr_lr": 0.0005,
            "ae_lr": 0.001,
            "early_stopping": {"patience": config.patience, "min_delta": 1e-4},
            "scheduler": {"min_lr": 1e-6},
            "logging": {"csv": True},
        },
    }
    for name in SOURCE_TASKS:
        task: dict[str, Any] = {
            "name": name,
            "dataset": "source",
            "column": name,
            "kind": "kernel_regression" if name in {"seebeck", "zt"} else "regression",
        }
        if name == "material_type":
            task.update(kind="classification", num_classes=5)
        elif name in {"seebeck", "zt"}:
            task["t_column"] = f"{name}_t"
        if name != "material_type":
            task["scaler"] = {"path": str(data_dir / f"source_scalers_{DATE}.joblib"), "key": name}
        raw["tasks"].append(task)
    if target is not None:
        selected = next(
            (t for t in manifest["targets"] if t["task"] == target and t["seed"] == seed and t["fraction"] == fraction),
            None,
        )
        if selected is None:
            raise ValueError("Target subset is not in the preprocessing manifest")
        raw["datasets"]["target"] = {"path": str(data_dir / selected["file"])}
        target_task = {
            "name": target,
            "dataset": "target",
            "column": target,
            "kind": "kernel_regression" if target == "power_factor" else "regression",
            "scaler": {"path": str(data_dir / selected["scaler"])},
        }
        if target == "power_factor":
            target_task["t_column"] = f"{target}_t"
        raw["tasks"].append(target_task)
    return raw


def write_toml(raw: dict[str, Any], path: Path) -> None:
    """Write these primitive workflow tables and task array tables, checking the round trip."""
    lines: list[str] = []

    def table(values: dict[str, Any], prefix: str, *, array: bool = False) -> None:
        lines.append(f"\n[[{prefix}]]" if array else f"\n[{prefix}]")
        for key, value in values.items():
            if not isinstance(value, dict) and not (isinstance(value, list) and value and isinstance(value[0], dict)):
                lines.append(f"{key} = {json.dumps(value, ensure_ascii=False, allow_nan=False)}")
        for key, value in values.items():
            if isinstance(value, dict):
                table(value, f"{prefix}.{key}")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                for item in value:
                    table(item, f"{prefix}.{key}", array=True)

    for key, value in raw.items():
        if key == "tasks":
            for task in value:
                table(task, key, array=True)
        else:
            table(value, key)
    content = "\n".join(lines) + "\n"
    if tomllib.loads(content) != raw:
        raise ValueError("Generated TOML did not round-trip")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def initialize_target(raw: dict[str, Any], target: str, path: Path, *, source: Path | None) -> dict[str, Any]:
    """Load the source representation and initialize an identically seeded target head.

    Scratch uses a fresh trainable encoder. The temporary checkpoint is a workflow initializer;
    its added target head has never seen labels and does not count as a pretrained task.
    """
    catalog = TaskCatalog(build_task_catalog_config({k: raw[k] for k in ("data", "descriptor", "datasets", "tasks")}))
    model_cfg, training_cfg = build_model_section(raw["model"]), build_training_section(raw["training"])
    seed_everything(training_cfg.seed, workers=True)
    model = build_empty_model(catalog, model_cfg, training_cfg)
    source_tasks: list[str] = []
    if source is not None:
        state = load_checkpoint_state(source)
        source_tasks = checkpoint_task_order(state)
        if target in source_tasks:
            raise ValueError("Held-out target was already pretrained")
        for name in source_tasks:
            model.add_task(build_head_config(catalog, model_cfg, training_cfg, name, masking_ratio=1))
        model.load_state_dict(state["model"], strict=True)
    torch.manual_seed(training_cfg.seed + (10000 if target == "dielectric_total" else 20000))
    model.add_task(build_head_config(catalog, model_cfg, training_cfg, target, masking_ratio=1))
    with_recorder = RunRecorder(path.parent)
    try:
        saved = with_recorder.save_final_model(model, [*source_tasks, target], {})
        path.parent.mkdir(parents=True, exist_ok=True)
        if saved != path:
            saved.replace(path)
    finally:
        with_recorder.close()
    return {
        "source_task_count": len(source_tasks),
        "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "target_head_parameters": sum(p.numel() for p in model.task_heads[target].parameters()),
    }


def composition_metrics(frame: pd.DataFrame, scale: float) -> dict[str, float | int]:
    """RMSE weights compositions equally, averaging each curve's squared error first."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization scale must be finite and positive")
    if frame.empty or not np.isfinite(frame[["true", "pred"]].to_numpy(dtype=float)).all():
        raise ValueError("Predictions must be nonempty and finite")
    errors = frame["pred"].to_numpy(dtype=float) - frame["true"].to_numpy(dtype=float)
    if frame["composition"].isna().any() or not np.isfinite(errors**2).all():
        raise ValueError("Composition keys and squared errors must be finite")
    grouped = frame.assign(squared_error=errors**2, absolute_error=np.abs(errors)).groupby("composition")
    rmse = float(np.sqrt(grouped["squared_error"].mean().mean()))
    return {
        "rmse": rmse,
        "standardized_rmse": rmse / scale,
        "mae": float(grouped["absolute_error"].mean().mean()),
        "compositions": grouped.ngroups,
        "points": len(frame),
    }


def inference_raw(raw: dict[str, Any], target: str, checkpoint: Path, output: Path, split: str) -> dict[str, Any]:
    pred = {key: deepcopy(raw[key]) for key in ("data", "descriptor", "datasets", "tasks", "model")}
    pred["predict"] = {
        "checkpoint": str(checkpoint),
        "tasks": [target],
        "split": split,
        "accelerator": raw["training"]["accelerator"],
        "seed": raw["training"]["seed"],
    }
    pred["output"] = {"dir": str(output)}
    return pred
