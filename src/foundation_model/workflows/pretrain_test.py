# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`foundation_model.workflows.pretrain`."""

from __future__ import annotations

import json
import tomllib

import numpy as np
import pandas as pd
import pytest
import torch

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from foundation_model.workflows._engine import build_trainer_extras
from foundation_model.workflows._sections import (
    CheckpointConfig,
    EarlyStoppingConfig,
    LoggingConfig,
    TrainingSectionConfig,
    build_model_section,
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
encoder_hidden_dims = [16]
head_hidden_dims = [8]
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


# --- warm-start from a checkpoint --------------------------------------------------------


def test_checkpoint_task_order_includes_all_heads_in_state() -> None:
    from foundation_model.workflows._engine import checkpoint_task_order

    # A step checkpoint stores task_sequence as the active subset, but the weights carry every
    # learned head; checkpoint_task_order must return all of them (AE excluded), ordered by the
    # sequence first, then any heads the sequence omitted.
    state = {
        "model": {
            "task_heads.a.w": 0,
            "task_heads.b.w": 0,
            "task_heads.c.w": 0,
            "task_heads.__reconstruction__.w": 0,
        },
        "task_sequence": ["c"],
    }
    assert checkpoint_task_order(state) == ["c", "a", "b"]
    # no task_sequence → fall back to state order
    assert checkpoint_task_order({"model": state["model"], "task_sequence": None}) == ["a", "b", "c"]


def _ws_toml(smoke_dir, sequence: list[str]) -> str:
    return f"""
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

[[tasks]]
name = "b"
kind = "regression"
dataset = "d1"
column = "b"

[model]
latent_dim = 8
encoder_hidden_dims = [16]
head_hidden_dims = [8]

[training]
max_epochs = 1
accelerator = "cpu"
seed = 1

[pretrain]
task_sequence = {sequence!r}
[pretrain.rehearsal]
interval = 1
default_replay = 0.5
"""


def _run_pretrain(cfg, out) -> None:
    rec = RunRecorder(out)
    rec.write_provenance(config=cfg, argv=["fm", "pretrain"], seeds={"seed": 1})
    pretrain_run(cfg, rec)
    rec.close()


def test_warm_start_continues_sequence(smoke_dir, tmp_path) -> None:
    # 1. pre-train task 'a' only → checkpoint.
    out1 = tmp_path / "run1"
    _run_pretrain(build_pretrain_config(tomllib.loads(_ws_toml(smoke_dir, ["a"])), output_dir=str(out1)), out1)
    ckpt = out1 / "training" / "final_model.pt"

    # 2. warm-start with sequence [a, b] → only 'b' is a new step; the model keeps both heads.
    out2 = tmp_path / "run2"
    cfg2 = build_pretrain_config(
        tomllib.loads(_ws_toml(smoke_dir, ["a", "b"])), output_dir=str(out2), checkpoint=str(ckpt)
    )
    assert cfg2.checkpoint is not None
    _run_pretrain(cfg2, out2)

    final = torch.load(out2 / "training" / "final_model.pt", weights_only=True)
    assert final["task_sequence"] == ["a", "b"]
    heads = {k.split(".", 2)[1] for k in final["model"] if k.startswith("task_heads.")}
    assert {"a", "b"} <= heads
    # 'a' was not re-introduced as a training step; global step numbering continues past it, so the
    # new task 'b' is step 2 (not step 1) and no step for 'a' is written this run.
    assert (out2 / "training" / "step02_b").exists()
    assert not (out2 / "training" / "step01_b").exists()
    assert not list((out2 / "training").glob("step*_a"))
    # 'a' head is still evaluated at the new step (already learned)
    assert (out2 / "training" / "step02_b" / "a_metrics.json").exists()


def test_warm_start_checkpoint_task_not_in_catalog_raises(smoke_dir, tmp_path) -> None:
    ckpt = tmp_path / "bad.pt"
    torch.save({"model": {"task_heads.zzz.net.weight": torch.zeros(2)}, "task_sequence": ["zzz"]}, ckpt)
    cfg = build_pretrain_config(
        tomllib.loads(_ws_toml(smoke_dir, ["a", "b"])), output_dir=str(tmp_path / "o"), checkpoint=str(ckpt)
    )
    with pytest.raises(ValueError, match="not in the catalog"):
        _run_pretrain(cfg, tmp_path / "o")


def test_warm_start_nothing_to_train_raises(smoke_dir, tmp_path) -> None:
    out1 = tmp_path / "run1"
    _run_pretrain(build_pretrain_config(tomllib.loads(_ws_toml(smoke_dir, ["a", "b"])), output_dir=str(out1)), out1)
    ckpt = out1 / "training" / "final_model.pt"
    cfg = build_pretrain_config(
        tomllib.loads(_ws_toml(smoke_dir, ["a", "b"])), output_dir=str(tmp_path / "o"), checkpoint=str(ckpt)
    )
    with pytest.raises(ValueError, match="nothing to train"):
        _run_pretrain(cfg, tmp_path / "o")


def test_checkpoint_from_toml_and_cli_precedence(smoke_dir) -> None:
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
[pretrain]
task_sequence = ["a"]
checkpoint = "from_toml.pt"
[output]
dir = "o"
"""
    raw = tomllib.loads(toml)
    assert str(build_pretrain_config(raw).checkpoint) == "from_toml.pt"
    assert str(build_pretrain_config(raw, checkpoint="from_cli.pt").checkpoint) == "from_cli.pt"  # CLI wins


def test_resume_config_from_flag_and_toml(smoke_dir) -> None:
    base = tomllib.loads(_ws_toml(smoke_dir, ["a"]))
    assert build_pretrain_config(base, output_dir="o").resume is False
    assert build_pretrain_config(base, output_dir="o", resume=True).resume is True  # --resume flag
    base["pretrain"]["resume"] = True
    assert build_pretrain_config(base, output_dir="o").resume is True  # [pretrain].resume


def test_resume_continues_after_simulated_kill(smoke_dir, tmp_path) -> None:
    out = tmp_path / "run"
    # 1. train [a] fully, then delete final_model.pt to simulate a kill right after step 1.
    _run_pretrain(build_pretrain_config(tomllib.loads(_ws_toml(smoke_dir, ["a"])), output_dir=str(out)), out)
    assert (out / "training" / "step01_a" / "checkpoint.pt").exists()
    (out / "training" / "final_model.pt").unlink()

    # 2. re-run the full sequence [a, b] with --resume → picks up from the step-1 checkpoint.
    cfg = build_pretrain_config(tomllib.loads(_ws_toml(smoke_dir, ["a", "b"])), output_dir=str(out), resume=True)
    _run_pretrain(cfg, out)
    final = torch.load(out / "training" / "final_model.pt", weights_only=True)
    assert final["task_sequence"] == ["a", "b"]
    assert (out / "training" / "step02_b").exists()  # continued at the next step, in place


def test_resume_skips_completed_run(smoke_dir, tmp_path) -> None:
    out = tmp_path / "run"
    _run_pretrain(build_pretrain_config(tomllib.loads(_ws_toml(smoke_dir, ["a", "b"])), output_dir=str(out)), out)
    steps_before = sorted(p.name for p in (out / "training").glob("step*"))
    cfg = build_pretrain_config(tomllib.loads(_ws_toml(smoke_dir, ["a", "b"])), output_dir=str(out), resume=True)
    _run_pretrain(cfg, out)  # final_model.pt present → run skipped, nothing re-trained
    assert sorted(p.name for p in (out / "training").glob("step*")) == steps_before


# --- Lightning callbacks / loggers config ------------------------------------------------


def test_model_section_rejects_non_int_dims() -> None:
    # TOML floats (128.5 / 10.0) must fail at config time, not silently coerce downstream.
    with pytest.raises(ValueError, match="positive int"):
        build_model_section({"latent_dim": 128.5})
    with pytest.raises(ValueError, match="positive int"):
        build_model_section({"n_kernel": 10.0})
    # hidden-dims lists reject non-positive / non-int entries too.
    with pytest.raises(ValueError, match="positive ints"):
        build_model_section({"encoder_hidden_dims": [16, 0]})


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


@pytest.mark.parametrize(
    ("toml_value", "expected"),
    [("2", 2), ("-1", -1), ("[1, 3]", [1, 3]), ("[0]", [0]), ('"1,3"', "1,3"), ('"auto"', "auto")],
)
def test_devices_accepts_lightning_forms(toml_value: str, expected) -> None:
    cfg = build_training_section(tomllib.loads(f"devices = {toml_value}"))
    assert cfg.devices == expected


@pytest.mark.parametrize(
    "toml_value",
    ["[]", '[1, "x"]', '""', "true", "0", "-2", "[-1]", "[0, -2]"],  # incl. bad ints + negative indices
)
def test_devices_rejects_bad_values(toml_value: str) -> None:
    with pytest.raises(ValueError, match="training.devices"):
        build_training_section(tomllib.loads(f"devices = {toml_value}"))


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
encoder_hidden_dims = [16]
head_hidden_dims = [8]

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
encoder_hidden_dims = [16]
head_hidden_dims = [8]

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
