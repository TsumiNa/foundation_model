# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""The single artifact writer for the training / predict workflows.

:class:`RunRecorder` centralises the output layout, provenance, per-step checkpoints, metrics,
prediction parquets and figures that used to be scattered across the legacy rehearsal scripts.
Every ``fm`` subcommand instantiates one recorder at startup and calls :meth:`write_provenance`
before doing any work. :func:`load_checkpoint_state` normalizes both the rehearsal checkpoint
schema and a bare ``state_dict`` (fm-trainer era) so downstream flows can consume either.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger

# Package names whose versions we record in provenance (distribution name → key label).
_PROVENANCE_PACKAGES = {
    "torch": "torch",
    "lightning": "lightning",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "foundation-model": "foundation-model",
}


def _json_default(obj: Any) -> Any:
    """Coerce values ``json`` cannot serialize: Enum → .value, Path/other → str."""
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


def _git_info() -> dict[str, Any]:
    """``{commit, dirty}`` from git, or nulls when unavailable (never raises)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


@dataclass(kw_only=True)
class RunPaths:
    """Resolved output layout for one run."""

    root: Path
    training: Path

    def step_dir(self, step: int, task: str) -> Path:
        return self.training / f"step{step:02d}_{task}"


class RunRecorder:
    """The only artifact writer for training / predict flows."""

    def __init__(self, root: Path) -> None:
        self.paths = RunPaths(root=Path(root), training=Path(root) / "training")
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.training.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._log_sink_id: int | None = logger.add(self.paths.root / "run.log", level="INFO", enqueue=False)

    # -- provenance

    def write_provenance(self, *, config: Any, argv: list[str], seeds: Mapping[str, int]) -> Path:
        """Write ``<root>/run_provenance.json`` (see module docstring)."""

        resolved = (
            dataclasses.asdict(config) if dataclasses.is_dataclass(config) and not isinstance(config, type) else config
        )
        packages: dict[str, str] = {"python": sys.version.split()[0]}
        for dist, label in _PROVENANCE_PACKAGES.items():
            try:
                packages[label] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                packages[label] = "unknown"
        now_utc = datetime.now(timezone.utc)
        provenance = {
            "resolved_config": resolved,
            "packages": packages,
            "datetime_utc": now_utc.isoformat(),
            "datetime_local": now_utc.astimezone().isoformat(),
            "git": _git_info(),
            "argv": list(argv),
            "seeds": dict(seeds),
        }
        path = self.paths.root / "run_provenance.json"
        path.write_text(json.dumps(provenance, indent=2, default=_json_default), encoding="utf-8")
        logger.info(f"Wrote provenance to {path}")
        return path

    # -- checkpoints

    def save_step_checkpoint(self, step: int, task: str, model: Any, active_tasks: list[str]) -> Path:
        """``training/stepNN_<task>/checkpoint.pt`` with the rehearsal schema."""

        step_dir = self.paths.step_dir(step, task)
        step_dir.mkdir(parents=True, exist_ok=True)
        path = step_dir / "checkpoint.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "task_sequence": list(active_tasks),
                "step": step,
                "new_task": task,
                "active_tasks": list(active_tasks),
            },
            path,
        )
        return path

    def save_final_model(self, model: Any, task_sequence: list[str], task_spec_dump: dict[str, Any]) -> Path:
        """``training/final_model.pt`` + ``training/final_model_taskconfigs.json`` (legacy schema)."""

        path = self.paths.training / "final_model.pt"
        torch.save({"model": model.state_dict(), "task_sequence": list(task_sequence)}, path)
        (self.paths.training / "final_model_taskconfigs.json").write_text(
            json.dumps(task_spec_dump, indent=2, default=_json_default), encoding="utf-8"
        )
        return path

    # -- per-task artifacts

    @staticmethod
    def dump_predictions(step_dir: Path, task: str, frame: pd.DataFrame) -> Path:
        """``<task>_pred.parquet`` — caller supplies the (composition, true, pred[, t]) frame."""

        step_dir.mkdir(parents=True, exist_ok=True)
        path = step_dir / f"{task}_pred.parquet"
        frame.to_parquet(path)
        return path

    @staticmethod
    def dump_metrics(step_dir: Path, task: str, metrics: dict[str, Any]) -> Path:
        """``<task>_metrics.json``."""

        step_dir.mkdir(parents=True, exist_ok=True)
        path = step_dir / f"{task}_metrics.json"
        path.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
        return path

    def save_figure(self, path_rel: str, fig: Any) -> Path:
        """Save any matplotlib figure at ``<root>/<path_rel>``."""

        path = self.paths.root / path_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        return path

    # -- records

    def append_record(self, record: dict[str, Any]) -> None:
        self._records.append(dict(record))

    def write_records(self) -> Path:
        """``training/experiment_records.json``."""

        path = self.paths.training / "experiment_records.json"
        path.write_text(json.dumps(self._records, indent=2, default=_json_default), encoding="utf-8")
        return path

    def write_metrics_table(self) -> Path:
        """``training/metrics_table.csv`` — a flat per-task row for every recorded metric dict."""

        rows: list[dict[str, Any]] = []
        for record in self._records:
            context = {k: v for k, v in record.items() if k != "metrics" and not isinstance(v, (dict, list))}
            metrics = record.get("metrics", {})
            if isinstance(metrics, Mapping) and metrics:
                for task, metric in metrics.items():
                    row = dict(context)
                    row["task"] = task
                    if isinstance(metric, Mapping):
                        row.update(metric)
                    else:
                        row["value"] = metric
                    rows.append(row)
            else:
                rows.append(dict(context))
        path = self.paths.training / "metrics_table.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def close(self) -> None:
        """Detach the loguru file sink (safe to call more than once)."""
        if self._log_sink_id is not None:
            try:
                logger.remove(self._log_sink_id)
            except ValueError:
                pass
            self._log_sink_id = None


def load_checkpoint_state(path: Path) -> dict[str, Any]:
    """Load a checkpoint and normalize to ``{"model": state_dict, "task_sequence": list|None, ...}``.

    Accepts both the rehearsal schema (``{"model": ..., "task_sequence": ...}``) and a bare
    ``state_dict`` (fm-trainer era).
    """

    obj = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(obj, Mapping) and "model" in obj and isinstance(obj["model"], Mapping):
        normalized = dict(obj)
        normalized.setdefault("task_sequence", None)
        return normalized
    # Bare state_dict (OrderedDict of param tensors).
    return {"model": obj, "task_sequence": None}
