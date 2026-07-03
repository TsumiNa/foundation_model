# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`foundation_model.workflows.pretrain`."""

from __future__ import annotations

import json
import tomllib

import numpy as np
import pandas as pd
import pytest

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from foundation_model.workflows._engine import build_trainer_extras
from foundation_model.workflows._sections import (
    CheckpointConfig,
    EarlyStoppingConfig,
    LoggingConfig,
    TrainingSectionConfig,
    build_training_section,
)
from foundation_model.workflows.pretrain import (
    active_old_tasks,
    build_pretrain_config,
    replay_to_ratio,
)
from foundation_model.workflows.pretrain import run as pretrain_run
from foundation_model.workflows.recording import RunRecorder

# 24 distinct real-element binary/ternary formulas so KMD descriptors are computable.
_ELEMENTS = ["Fe", "Al", "Cu", "Ni", "Ti", "Zn", "Mg", "Ca", "Na", "Cl", "O", "Si"]
_FORMULAS = [f"{a}2 {b}3" for i, a in enumerate(_ELEMENTS) for b in _ELEMENTS[i + 1 :]][:24]


def _config_toml(*, pretrain_extra: str = "", rehearsal: str = "") -> str:
    return f"""
[descriptor]
kind = "kmd"
n_grids = 4

[datasets.d1]
path = "data/x.parquet"

[[tasks]]
name = "a"
kind = "regression"
dataset = "d1"
column = "a"

[[tasks]]
name = "b"
kind = "regression"
dataset = "d1"
column = "b"

[pretrain]
{pretrain_extra}
{rehearsal}
"""


def _build(toml_str: str):
    return build_pretrain_config(tomllib.loads(toml_str), output_dir="out")


# --- config validation -------------------------------------------------------------------


def test_build_happy_path_defaults_task_sequence() -> None:
    cfg = _build(_config_toml())
    assert cfg.task_sequence == ["a", "b"]  # defaults to [[tasks]] order
    assert cfg.n_runs == 1 and cfg.rehearsal.interval == 1


def test_interval_below_one_raises() -> None:
    with pytest.raises(ValueError, match="interval must be >= 1"):
        _build(_config_toml(rehearsal="[pretrain.rehearsal]\ninterval = 0"))


def test_unknown_task_in_sequence_raises() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        _build(_config_toml(pretrain_extra='task_sequence = ["a", "nope"]'))


def test_invalid_task_order_raises() -> None:
    with pytest.raises(ValueError):
        _build(_config_toml(pretrain_extra='task_order = "weird"'))


def test_per_task_replay_unknown_task_raises() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        _build(_config_toml(rehearsal="[pretrain.rehearsal.per_task]\nnope = 0.1"))


def test_per_task_replay_bad_value_raises() -> None:
    with pytest.raises(ValueError, match="replay"):
        _build(_config_toml(rehearsal="[pretrain.rehearsal.per_task]\na = 0.0"))


# --- pure helpers ------------------------------------------------------------------------


def test_active_old_tasks_interval_schedule() -> None:
    order = ["t1", "t2", "t3", "t4"]
    participating_sets = []
    for step, task in enumerate(order, start=1):
        learned = order[: step - 1]
        active = [task, *active_old_tasks(step, learned, 2)]
        participating_sets.append(active)
    assert participating_sets == [["t1"], ["t2", "t1"], ["t3"], ["t4", "t1", "t2", "t3"]]


def test_active_old_tasks_interval_one_always_replays() -> None:
    assert active_old_tasks(3, ["t1", "t2"], 1) == ["t1", "t2"]


@pytest.mark.parametrize(
    ("replay", "n_valid", "expected"),
    [(0.1, 100, 0.1), (500, 100, 1.0), (50, 100, 0.5), (5, 0, 1.0)],
)
def test_replay_to_ratio(replay: float, n_valid: int, expected: float) -> None:
    assert replay_to_ratio(replay, n_valid) == expected


def test_random_order_same_seed_same_permutation() -> None:
    from foundation_model.workflows.pretrain import _run_task_order

    cfg = _build(_config_toml(pretrain_extra='task_sequence = ["a", "b"]\ntask_order = "random"'))
    assert _run_task_order(cfg, 7) == _run_task_order(cfg, 7)


# --- end-to-end smoke --------------------------------------------------------------------


@pytest.fixture
def smoke_dir(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "composition": _FORMULAS,
            "a": rng.normal(size=len(_FORMULAS)),
            "b": rng.normal(size=len(_FORMULAS)),
        }
    )
    df.to_parquet(tmp_path / "x.parquet")
    return tmp_path


def _smoke_config(smoke_dir, output_dir):
    toml = f"""
[data]
batch_size = 8

[descriptor]
kind = "kmd"
n_grids = 4

[datasets.d1]
path = "{smoke_dir / "x.parquet"}"

[[tasks]]
name = "a"
kind = "regression"
dataset = "d1"
column = "a"

[[tasks]]
name = "b"
kind = "regression"
dataset = "d1"
column = "b"

[model]
latent_dim = 8
encoder_hidden = 16
head_hidden_dim = 8
n_kernel = 4

[training]
max_epochs = 1
accelerator = "cpu"
seed = 1

[pretrain]
task_sequence = ["a", "b"]

[pretrain.rehearsal]
interval = 1
default_replay = 0.5
"""
    return build_pretrain_config(tomllib.loads(toml), output_dir=str(output_dir))


# --- Lightning callbacks / loggers config ------------------------------------------------


def test_training_subtables_parse() -> None:
    cfg = build_training_section(
        tomllib.loads(
            "max_epochs = 5\n"
            '[early_stopping]\npatience = 4\nmode = "max"\n'
            "[checkpoint]\nenabled = true\nsave_top_k = 2\n"
            "[logging]\ncsv = true\ntensorboard = true\n"
        )
    )
    assert cfg.early_stopping.patience == 4 and cfg.early_stopping.mode == "max"
    assert cfg.checkpoint.enabled and cfg.checkpoint.save_top_k == 2
    assert cfg.logging.csv and cfg.logging.tensorboard


def test_early_stopping_bad_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        build_training_section(tomllib.loads('[early_stopping]\nmode = "up"'))


def test_training_subtable_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="save_topk"):
        build_training_section(tomllib.loads("[checkpoint]\nsave_topk = 2"))


def test_build_trainer_extras_default(tmp_path) -> None:
    callbacks, loggers, enable = build_trainer_extras(
        TrainingSectionConfig(), log_dir=tmp_path, ckpt_dir=tmp_path, run_name="r"
    )
    assert any(isinstance(c, EarlyStopping) for c in callbacks)  # on by default
    assert not any(isinstance(c, ModelCheckpoint) for c in callbacks)  # opt-in
    assert loggers is False and enable is False


def test_build_trainer_extras_all_enabled(tmp_path) -> None:
    training = TrainingSectionConfig(
        early_stopping=EarlyStoppingConfig(enabled=False),
        checkpoint=CheckpointConfig(enabled=True),
        logging=LoggingConfig(csv=True, tensorboard=True),
    )
    callbacks, loggers, enable = build_trainer_extras(
        training, log_dir=tmp_path, ckpt_dir=tmp_path / "ck", run_name="r"
    )
    assert not any(isinstance(c, EarlyStopping) for c in callbacks)
    assert any(isinstance(c, ModelCheckpoint) for c in callbacks)
    assert enable is True
    assert isinstance(loggers, list) and len(loggers) == 2
    assert any(isinstance(lg, CSVLogger) for lg in loggers)
    assert any(isinstance(lg, TensorBoardLogger) for lg in loggers)


def test_pretrain_lightning_checkpoint_and_csv_logger(smoke_dir, tmp_path) -> None:
    out = tmp_path / "run"
    toml = f"""
[data]
batch_size = 4

[descriptor]
kind = "kmd"
n_grids = 4

[datasets.d1]
path = "{smoke_dir / "x.parquet"}"

[[tasks]]
name = "a"
kind = "regression"
dataset = "d1"
column = "a"

[model]
latent_dim = 8
encoder_hidden = 16
head_hidden_dim = 8

[training]
max_epochs = 1
accelerator = "cpu"
seed = 1

[training.checkpoint]
enabled = true

[training.logging]
csv = true

[pretrain]
task_sequence = ["a"]
"""
    cfg = build_pretrain_config(tomllib.loads(toml), output_dir=str(out))
    rec = RunRecorder(out)
    rec.write_provenance(config=cfg, argv=["fm", "pretrain"], seeds={"seed": 1})
    pretrain_run(cfg, rec)
    rec.close()
    assert (out / "logs").exists()  # CSVLogger wrote metric curves
    assert list(out.glob("training/step*/lightning/*.ckpt"))  # ModelCheckpoint wrote a .ckpt


def test_pretrain_smoke_end_to_end(smoke_dir, tmp_path) -> None:
    out = tmp_path / "run"
    cfg = _smoke_config(smoke_dir, out)
    recorder = RunRecorder(out)
    recorder.write_provenance(config=cfg, argv=["fm", "pretrain"], seeds={"seed": 1})
    result = pretrain_run(cfg, recorder)
    recorder.close()

    # n_runs == 1 → outputs directly under the root.
    assert (out / "run_provenance.json").exists()
    assert (out / "run.log").exists()
    training = out / "training"
    assert (training / "final_model.pt").exists()
    assert (training / "final_model_taskconfigs.json").exists()
    assert (training / "experiment_records.json").exists()
    assert (training / "metrics_table.csv").exists()
    assert (training / "forgetting_trajectory.png").exists()
    # step dirs + per-task artifacts (2 heads evaluated at step 2).
    step2 = training / "step02_b"
    assert (step2 / "checkpoint.pt").exists()
    for task in ("a", "b"):
        assert (step2 / f"{task}_pred.parquet").exists()
        assert (step2 / f"{task}_metrics.json").exists()
    assert (step2 / "b_parity.png").exists()  # newest head plotted
    assert len(result["records"]) == 2  # one record per step


def test_pretrain_nruns_sweep_layout(smoke_dir, tmp_path) -> None:
    out = tmp_path / "sweep"
    toml = f"""
[descriptor]
kind = "kmd"
n_grids = 4

[datasets.d1]
path = "{smoke_dir / "x.parquet"}"

[[tasks]]
name = "a"
kind = "regression"
dataset = "d1"
column = "a"

[[tasks]]
name = "b"
kind = "regression"
dataset = "d1"
column = "b"

[model]
latent_dim = 8
encoder_hidden = 16
head_hidden_dim = 8

[training]
max_epochs = 1
accelerator = "cpu"
seed = 1

[pretrain]
n_runs = 2
"""
    cfg = build_pretrain_config(tomllib.loads(toml), output_dir=str(out))
    pretrain_run(cfg)

    assert (out / "runs" / "run00" / "training" / "final_model.pt").exists()
    assert (out / "runs" / "run01" / "training" / "final_model.pt").exists()
    aggregate = json.loads((out / "experiment_records.json").read_text())
    assert sorted({r["run"] for r in aggregate}) == [0, 1]
    assert not (out / "training").exists()  # root training/ stays absent for n_runs > 1
