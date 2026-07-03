# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`foundation_model.workflows.recording`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from foundation_model.workflows.recording import RunRecorder, load_checkpoint_state


@dataclass
class _Cfg:
    output_dir: Path
    max_epochs: int


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)


def test_write_provenance(tmp_path) -> None:
    rec = RunRecorder(tmp_path)
    cfg = _Cfg(output_dir=tmp_path / "run", max_epochs=5)
    path = rec.write_provenance(config=cfg, argv=["fm", "pretrain"], seeds={"seed": 2025})
    rec.close()

    assert path.exists()
    data = json.loads(path.read_text())
    # resolved config values (Path coerced to str by _json_default)
    assert data["resolved_config"]["max_epochs"] == 5
    assert str(tmp_path / "run") in data["resolved_config"]["output_dir"]
    # every package-version key present
    for key in ("python", "torch", "lightning", "numpy", "pandas", "scikit-learn", "foundation-model"):
        assert key in data["packages"]
    # ISO datetime parseable
    datetime.fromisoformat(data["datetime_utc"])
    datetime.fromisoformat(data["datetime_local"])
    assert data["argv"] == ["fm", "pretrain"]
    assert data["seeds"] == {"seed": 2025}
    assert set(data["git"]) == {"commit", "dirty"}  # present (values may be null)
    assert (tmp_path / "run.log").exists()


def test_save_step_checkpoint_schema_and_reload(tmp_path) -> None:
    rec = RunRecorder(tmp_path)
    model = _TinyModel()
    path = rec.save_step_checkpoint(1, "density", model, ["density"])
    rec.close()

    assert path == tmp_path / "training" / "step01_density" / "checkpoint.pt"
    raw = torch.load(path, weights_only=False)
    assert set(raw) == {"model", "task_sequence", "step", "new_task", "active_tasks"}
    assert raw["step"] == 1 and raw["new_task"] == "density" and raw["active_tasks"] == ["density"]

    state = load_checkpoint_state(path)
    assert state["task_sequence"] == ["density"]
    assert "linear.weight" in state["model"]


def test_save_final_model(tmp_path) -> None:
    rec = RunRecorder(tmp_path)
    model = _TinyModel()
    spec_dump = {"density": {"kind": "regression", "column": "d", "source": "qc"}}
    path = rec.save_final_model(model, ["density"], spec_dump)
    rec.close()

    assert path == tmp_path / "training" / "final_model.pt"
    raw = torch.load(path, weights_only=False)
    assert set(raw) == {"model", "task_sequence"}
    dumped = json.loads((tmp_path / "training" / "final_model_taskconfigs.json").read_text())
    assert dumped == spec_dump


def test_load_checkpoint_state_normalizes_bare_state_dict(tmp_path) -> None:
    model = _TinyModel()
    bare = tmp_path / "bare.pt"
    torch.save(model.state_dict(), bare)
    state = load_checkpoint_state(bare)
    assert state["task_sequence"] is None
    assert "linear.weight" in state["model"]


def test_dump_predictions_and_metrics_roundtrip(tmp_path) -> None:
    rec = RunRecorder(tmp_path)
    step_dir = rec.paths.step_dir(1, "density")
    frame = pd.DataFrame({"composition": ["Fe2 O3", "Al2 O3"], "true": [1.0, 2.0], "pred": [1.1, 1.9]})
    pred_path = rec.dump_predictions(step_dir, "density", frame)
    metric_path = rec.dump_metrics(step_dir, "density", {"r2": 0.95, "mae": 0.1, "primary": 0.95})
    rec.close()

    assert pred_path.exists() and metric_path.exists()
    roundtrip = pd.read_parquet(pred_path)
    pd.testing.assert_frame_equal(roundtrip, frame)
    assert json.loads(metric_path.read_text())["r2"] == 0.95


def test_write_records_and_metrics_table(tmp_path) -> None:
    rec = RunRecorder(tmp_path)
    rec.append_record(
        {"step": 1, "new_task": "density", "epochs_run": 3, "metrics": {"density": {"r2": 0.9, "mae": 0.2}}}
    )
    rec.append_record(
        {
            "step": 2,
            "new_task": "mat",
            "epochs_run": 4,
            "metrics": {"density": {"r2": 0.85, "mae": 0.3}, "mat": {"accuracy": 0.7}},
        }
    )
    records_path = rec.write_records()
    table_path = rec.write_metrics_table()
    rec.close()

    records = json.loads(records_path.read_text())
    assert len(records) == 2 and records[0]["new_task"] == "density"
    table = pd.read_csv(table_path)
    # one row per (record, task): 1 + 2 = 3 rows
    assert len(table) == 3
    assert set(table["task"]) == {"density", "mat"}
    assert "r2" in table.columns and "accuracy" in table.columns
