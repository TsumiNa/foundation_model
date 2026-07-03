# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``fm predict`` — evaluate / predict with an arbitrary checkpoint.

Rebuilds the checkpoint's model from the :class:`TaskCatalog`, loads its weights with
``strict=False`` (preserving the fm-trainer ``strict=False`` semantics), resolves a prediction
set (a split or an explicit composition list), and writes per-head prediction parquets — plus
metrics when true targets are available. Single-device only; the legacy distributed prediction
gather is not migrated.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning import seed_everything
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score  # type: ignore[import-untyped]

from foundation_model.data.composition_sources import canonical_key, normalize_composition

from ._engine import AE_NAME, as_float_array, build_model_for_checkpoint
from ._sections import ModelSectionConfig, build_model_section, reject_unknown
from .recording import RunRecorder, load_checkpoint_state
from .task_catalog import TaskCatalog, TaskCatalogConfig, TaskKind, build_task_catalog_config

_PREDICT_ROOT_KEYS = {"data", "descriptor", "datasets", "tasks", "model", "predict", "output"}
_CATALOG_KEYS = {"data", "descriptor", "datasets", "tasks"}
_SPLITS = {"train", "val", "test", "all"}


@dataclass(kw_only=True)
class PredictConfig:
    catalog: TaskCatalogConfig
    model: ModelSectionConfig
    checkpoint: Path
    output_dir: Path
    tasks: list[str] = field(default_factory=list)  # empty → every head in the checkpoint
    split: str = "test"
    compositions: list[str] = field(default_factory=list)  # overrides split when given
    with_metrics: bool = True
    seed: int = 2025  # seeds torch/numpy for reproducible split resolution
    accelerator: str = "auto"  # "auto" (cuda if available, else cpu) | "cpu"

    def __post_init__(self) -> None:
        self.checkpoint = Path(self.checkpoint)
        self.output_dir = Path(self.output_dir)
        if self.split not in _SPLITS:
            raise ValueError(f"predict.split must be one of {sorted(_SPLITS)}, got {self.split!r}.")
        catalog_tasks = {t.name for t in self.catalog.tasks}
        unknown = [t for t in self.tasks if t not in catalog_tasks]
        if unknown:
            raise ValueError(f"predict.tasks references unknown task(s): {unknown}.")


def build_predict_config(
    raw: Mapping[str, Any], *, output_dir: str | Path | None = None, checkpoint: str | Path | None = None
) -> PredictConfig:
    """Normalize a parsed-TOML tree into a :class:`PredictConfig`."""

    reject_unknown("<root>", raw, _PREDICT_ROOT_KEYS)
    catalog = build_task_catalog_config({k: raw[k] for k in _CATALOG_KEYS if k in raw})
    model = build_model_section(raw.get("model", {}))

    pred_raw = dict(raw.get("predict", {}))
    reject_unknown(
        "predict", pred_raw, {"checkpoint", "tasks", "split", "compositions", "with_metrics", "seed", "accelerator"}
    )
    resolved_checkpoint = checkpoint if checkpoint is not None else pred_raw.get("checkpoint")
    if resolved_checkpoint is None:
        raise ValueError("checkpoint must be given via --checkpoint or [predict].checkpoint.")
    resolved_output = output_dir if output_dir is not None else raw.get("output", {}).get("dir")
    if resolved_output is None:
        raise ValueError("output directory must be given via --output-dir or [output].dir.")

    return PredictConfig(
        catalog=catalog,
        model=model,
        checkpoint=Path(resolved_checkpoint),
        output_dir=Path(resolved_output),
        tasks=list(pred_raw.get("tasks", [])),
        split=str(pred_raw.get("split", "test")),
        compositions=list(pred_raw.get("compositions", [])),
        with_metrics=bool(pred_raw.get("with_metrics", True)),
        seed=int(pred_raw.get("seed", 2025)),
        accelerator=str(pred_raw.get("accelerator", "auto")),
    )


def _task_names_from_state(state_dict: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for key in state_dict:
        if key.startswith("task_heads."):
            name = key.split(".", 2)[1]
            if name != AE_NAME and name not in names:
                names.append(name)
    return names


def _resolve_device(accelerator: str) -> torch.device:
    """``"cpu"`` forces CPU; ``"auto"`` uses CUDA when available, else CPU."""
    if accelerator != "cpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run(cfg: PredictConfig, recorder: RunRecorder | None = None) -> dict[str, Any]:
    """Predict ``cfg.tasks`` (or every checkpoint head) on the resolved set. Returns metrics dict."""

    catalog = TaskCatalog(cfg.catalog)
    owns_recorder = recorder is None
    rec = recorder or RunRecorder(cfg.output_dir)
    seed_everything(cfg.seed, workers=True)

    try:
        model, ckpt_tasks = _rebuild_model(cfg, catalog)
        model = model.to(_resolve_device(cfg.accelerator))
        heads = set(model.task_heads)
        requested = cfg.tasks or [t for t in ckpt_tasks if t in heads]
        missing = [t for t in requested if t not in heads]
        if missing:
            raise ValueError(f"requested task(s) {missing} not in the checkpoint (available: {sorted(heads)}).")

        comps = _resolve_compositions(cfg, catalog, requested)
        if not comps:
            raise RuntimeError("no compositions resolved for prediction.")
        logger.info(f"Predicting {requested} on {len(comps)} compositions.")

        device = next(model.parameters()).device
        pred_dir = rec.paths.root / "predict"
        pred_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, dict[str, float]] = {}
        for task in requested:
            metric = _predict_task(model, catalog, task, comps, device, pred_dir, with_metrics=cfg.with_metrics)
            if metric:
                metrics[task] = metric

        if metrics:
            rec.append_record({"split": cfg.split, "metrics": metrics})
            (pred_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            rows = [{"task": t, **m} for t, m in metrics.items()]
            pd.DataFrame(rows).to_csv(pred_dir / "metrics_table.csv", index=False)
        return {"metrics": metrics, "n_compositions": len(comps)}
    finally:
        if owns_recorder:
            rec.close()


def _rebuild_model(cfg: PredictConfig, catalog: TaskCatalog) -> tuple[Any, list[str]]:
    state = load_checkpoint_state(cfg.checkpoint)
    ckpt_tasks = list(state.get("task_sequence") or _task_names_from_state(state["model"]))
    catalog_tasks = {t.name for t in cfg.catalog.tasks}
    missing = [t for t in ckpt_tasks if t not in catalog_tasks]
    if missing:
        raise ValueError(f"checkpoint tasks {missing} are not in the catalog (have {sorted(catalog_tasks)}).")

    model = build_model_for_checkpoint(catalog, cfg.model, ckpt_tasks)
    incompatible = model.load_state_dict(state["model"], strict=False)
    if incompatible.missing_keys:
        logger.info(f"load_state_dict missing keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys[:8]}")
    if incompatible.unexpected_keys:
        logger.info(
            f"load_state_dict unexpected keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys[:8]}"
        )
    model.eval()
    return model, ckpt_tasks


def _resolve_compositions(cfg: PredictConfig, catalog: TaskCatalog, tasks: Sequence[str]) -> list[str]:
    """Explicit compositions win (request order); otherwise resolve the split via the datamodule."""

    if cfg.compositions:
        # Normalize to canonical keys (matching the catalog frames), preserving order + dropping dups.
        seen: dict[str, None] = {}
        for comp in cfg.compositions:
            seen.setdefault(canonical_key(comp, normalize_composition), None)
        return list(seen)

    # Split-based: go through CompoundDataModule so the composition-level split overlay applies —
    # including the [data].val_split/test_split random fallback when a task frame has no explicit
    # 'split' column (otherwise every row would leak into the requested split).
    datamodule = catalog.build_datamodule(tasks, predict_idx=cfg.split)
    datamodule.setup("predict")
    return list(datamodule.predict_compositions)


def _predict_task(
    model: Any,
    catalog: TaskCatalog,
    name: str,
    comps: list[str],
    device: torch.device,
    pred_dir: Path,
    *,
    with_metrics: bool,
) -> dict[str, float]:
    """Predict one head on ``comps``; write parquet (+ metrics when enabled and targets exist)."""

    spec = catalog.task_spec(name)
    frame = catalog.task_frames([name])[name]
    head = model.task_heads[name]

    if spec.kind is TaskKind.KERNEL_REGRESSION:
        return _predict_kr(model, catalog, name, comps, device, pred_dir, with_metrics=with_metrics)

    desc = catalog.descriptor_fn()(comps)
    kept = [c for c in comps if c in desc.index]
    if not kept:
        logger.warning(f"task '{name}': no descriptors available for the prediction set.")
        return {}
    x = torch.tensor(desc.loc[kept].values, dtype=torch.float32, device=device)
    with torch.no_grad():
        h = torch.tanh(model.encoder(x))
        if spec.kind is TaskKind.REGRESSION:
            pred = catalog.inverse_transform(name, head(h).squeeze(-1).cpu().numpy())
        else:  # classification
            pred = head(h).argmax(dim=-1).cpu().numpy()

    true = _true_values(frame, spec.column, kept, kind=spec.kind, catalog=catalog, name=name)
    pd.DataFrame({"composition": kept, "pred": pred, "true": true}).to_parquet(pred_dir / f"{name}_pred.parquet")
    return _metrics(spec.kind, true, pred) if with_metrics else {}


def _predict_kr(
    model: Any,
    catalog: TaskCatalog,
    name: str,
    comps: list[str],
    device: torch.device,
    pred_dir: Path,
    *,
    with_metrics: bool,
) -> dict[str, float]:
    spec = catalog.task_spec(name)
    assert spec.t_column is not None
    frame = catalog.task_frames([name])[name]
    head = model.task_heads[name]
    desc = catalog.descriptor_fn()(comps)  # compute descriptors once and reuse for filtering + tensor
    available = set(desc.index)
    keep: list[str] = []
    t_list: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    for comp in comps:
        if comp not in available or comp not in frame.index:
            continue
        y = as_float_array(frame.at[comp, spec.column]) if pd.notna(frame.at[comp, spec.column]) else np.array([])
        t = as_float_array(frame.at[comp, spec.t_column]) if pd.notna(frame.at[comp, spec.t_column]) else np.array([])
        if t.size == 0:
            continue
        keep.append(comp)
        t_list.append(t)
        # Inverse-transform the true sequence too, so it matches the (inverse-transformed) preds.
        true_parts.append(catalog.inverse_transform(name, y) if y.size == t.size else np.full(t.size, np.nan))
    if not keep:
        logger.warning(f"task '{name}': no compositions with a t-grid to predict.")
        return {}
    x = torch.tensor(desc.loc[keep].values, dtype=torch.float32, device=device)
    with torch.no_grad():
        h = torch.tanh(model.encoder(x))
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=device) for t in t_list]
        expanded_h, expanded_t = model._expand_for_kernel_regression(h, t_tensors)
        pred = catalog.inverse_transform(name, head(expanded_h, t=expanded_t).squeeze(-1).cpu().numpy())
    rows: list[dict[str, Any]] = []
    offset = 0
    for comp, t_arr, y_true in zip(keep, t_list, true_parts):
        n = int(t_arr.size)
        for k in range(n):
            rows.append(
                {"composition": comp, "t": float(t_arr[k]), "pred": float(pred[offset + k]), "true": float(y_true[k])}
            )
        offset += n
    long = pd.DataFrame(rows)
    long.to_parquet(pred_dir / f"{name}_pred.parquet")
    mask = ~long["true"].isna()
    if with_metrics and mask.sum() >= 2:
        return {
            "r2": float(r2_score(long.loc[mask, "true"], long.loc[mask, "pred"])),
            "mae": float(mean_absolute_error(long.loc[mask, "true"], long.loc[mask, "pred"])),
            "points": int(mask.sum()),
        }
    return {}


def _true_values(
    frame: pd.DataFrame, column: str, comps: list[str], *, kind: TaskKind, catalog: TaskCatalog, name: str
) -> np.ndarray:
    raw = np.array([frame.at[c, column] if c in frame.index else np.nan for c in comps], dtype=float)
    if kind is TaskKind.REGRESSION:
        return catalog.inverse_transform(name, raw)
    return raw


def _metrics(kind: TaskKind, true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = ~np.isnan(np.asarray(true, dtype=float))
    if mask.sum() < 1:
        return {}
    t, p = np.asarray(true)[mask], np.asarray(pred)[mask]
    if kind is TaskKind.REGRESSION:
        if mask.sum() < 2:
            return {"samples": int(mask.sum())}
        return {"r2": float(r2_score(t, p)), "mae": float(mean_absolute_error(t, p)), "samples": int(mask.sum())}
    return {
        "accuracy": float(accuracy_score(t.astype(int), p.astype(int))),
        "macro_f1": float(f1_score(t.astype(int), p.astype(int), average="macro", zero_division=0)),
        "samples": int(mask.sum()),
    }
