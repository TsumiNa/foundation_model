# Copyright 2026 TsumiNa.
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


def test_load_checkpoint_state_folds_disabled_heads(tmp_path) -> None:
    # A checkpoint saved while head 'b' was disabled stores it under disabled_task_heads.*;
    # load_checkpoint_state must fold it back onto task_heads.* so downstream loads every head.
    sd = {
        "encoder.shared.0.weight": torch.zeros(2),
        "task_heads.a.net.weight": torch.zeros(2),
        "disabled_task_heads.b.net.weight": torch.ones(2),
    }
    ckpt = tmp_path / "disabled.pt"
    torch.save({"model": sd, "task_sequence": None}, ckpt)
    keys = set(load_checkpoint_state(ckpt)["model"])
    assert "task_heads.b.net.weight" in keys  # folded onto task_heads
    assert "disabled_task_heads.b.net.weight" not in keys
    assert {"task_heads.a.net.weight", "encoder.shared.0.weight"} <= keys  # others untouched


def test_load_checkpoint_state_unwraps_lightning_checkpoint(tmp_path) -> None:
    model = _TinyModel()
    ckpt = tmp_path / "epoch=2.ckpt"
    # Mimic a Lightning ModelCheckpoint payload: params nested under "state_dict".
    torch.save({"state_dict": model.state_dict(), "epoch": 2, "global_step": 10}, ckpt)
    state = load_checkpoint_state(ckpt)
    assert "linear.weight" in state["model"]  # unwrapped, not the whole checkpoint dict
    assert state["task_sequence"] is None


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


# --- the load side, on a real model -----------------------------------------------------------
#
# Every checkpoint test above runs on _TinyModel, a single nn.Linear. The mechanism that used to
# key optimizer/scheduler state by parameter group was removed because nothing reads it back: the
# CLIs load weights only, via load_checkpoint_state -> load_state_dict(strict=False). That claim
# is what these two pin, on the real model and its real key names.


def _real_model(*, disabled: tuple[str, ...] = ()):
    """A FlexibleMultiTaskModel with one head of each kind plus the AE head, weights perturbed.

    Perturbed rather than freshly initialised: two models built from the same config with the same
    seed would compare equal even if the load did nothing at all.
    """
    from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
    from foundation_model.models.model_config import (
        ClassificationTaskConfig,
        KernelRegressionTaskConfig,
        MLPEncoderConfig,
        OptimizerConfig,
        RegressionTaskConfig,
        TaskType,
    )

    latent = 16
    model = FlexibleMultiTaskModel(
        task_configs=[
            RegressionTaskConfig(
                name="density",
                type=TaskType.REGRESSION,
                data_column="density",
                dims=[latent, 8, 1],
                optimizer=OptimizerConfig(lr=1e-3, min_lr=1e-6),
            ),
            ClassificationTaskConfig(
                name="is_metal",
                type=TaskType.CLASSIFICATION,
                data_column="is_metal",
                dims=[latent, 8],
                num_classes=2,
                optimizer=OptimizerConfig(lr=1e-3, min_lr=1e-6),
            ),
            KernelRegressionTaskConfig(
                name="dos",
                data_column="dos",
                t_column="energy",
                x_dim=[latent, 8],
                t_dim=[latent, 8],
                kernel_num_centers=4,
                optimizer=OptimizerConfig(lr=1e-3, min_lr=1e-6),
            ),
        ],
        encoder_config=MLPEncoderConfig(hidden_dims=[12, 16, latent]),
        enable_autoencoder=True,
        shared_block_optimizer=OptimizerConfig(lr=5e-3, min_lr=1e-6),
    )
    if disabled:
        model.disable_task(*disabled)
    return model


def test_real_model_checkpoint_round_trips_through_the_loader(tmp_path) -> None:
    """save_step_checkpoint -> load_checkpoint_state -> load_state_dict must restore every weight.

    This is the whole of what the CLIs ask of a checkpoint — ``fm finetune``, ``fm predict`` and
    ``fm pretrain --resume`` all load weights and nothing else — so it is the guarantee that has to
    hold, and it had never been asserted against a real model.
    """
    torch.manual_seed(0)
    saved_model = _real_model()
    with torch.no_grad():
        for parameter in saved_model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)

    rec = RunRecorder(tmp_path)
    path = rec.save_step_checkpoint(1, "density", saved_model, ["density", "is_metal", "dos"])
    rec.close()

    torch.manual_seed(1)  # a differently-initialised model, so equality can only come from the load
    fresh = _real_model()
    assert not torch.equal(
        fresh.state_dict()["encoder.shared.layers.0.layer.weight"],
        saved_model.state_dict()["encoder.shared.layers.0.layer.weight"],
    ), "the two models must start apart, or the comparison below proves nothing"
    state = load_checkpoint_state(path)
    incompatible = fresh.load_state_dict(state["model"], strict=False)

    assert not incompatible.missing_keys, (
        f"weights the fresh model wanted but the checkpoint lacks: {incompatible.missing_keys}"
    )
    assert not incompatible.unexpected_keys, (
        f"weights the checkpoint carries but the model has no slot for: {incompatible.unexpected_keys}"
    )
    expected = saved_model.state_dict()
    assert set(fresh.state_dict()) == set(expected)
    for key, value in fresh.state_dict().items():
        assert torch.equal(value, expected[key]), f"{key} did not survive the round trip"


def test_disabled_head_survives_the_round_trip_onto_a_full_model(tmp_path) -> None:
    """A head disabled at save time must come back onto ``task_heads`` at load time.

    ``fm finetune`` disables the non-target heads for the fit, so a mid-fit checkpoint stores them
    under ``disabled_task_heads.*``; load_checkpoint_state folds them back. The synthetic test
    above pins the folding on hand-written keys — this pins it on the ones the model really emits.
    """
    torch.manual_seed(0)
    saved_model = _real_model(disabled=("is_metal",))
    with torch.no_grad():
        for parameter in saved_model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    raw_keys = set(saved_model.state_dict())
    assert any(k.startswith("disabled_task_heads.is_metal.") for k in raw_keys), (
        "fixture must actually park the head under disabled_task_heads"
    )

    rec = RunRecorder(tmp_path)
    path = rec.save_step_checkpoint(1, "density", saved_model, ["density", "dos"])
    rec.close()

    torch.manual_seed(1)
    fresh = _real_model()  # every head enabled, as predict/inverse rebuild it
    incompatible = fresh.load_state_dict(load_checkpoint_state(path)["model"], strict=False)

    assert not incompatible.missing_keys, incompatible.missing_keys
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys
    disabled_weights = {
        k.split("disabled_task_heads.")[1]: v
        for k, v in saved_model.state_dict().items()
        if k.startswith("disabled_task_heads.")
    }
    for suffix, value in disabled_weights.items():
        assert torch.equal(fresh.state_dict()[f"task_heads.{suffix}"], value), (
            f"the disabled head's {suffix} did not land on task_heads"
        )
