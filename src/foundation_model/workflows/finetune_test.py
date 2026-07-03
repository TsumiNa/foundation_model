# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`foundation_model.workflows.finetune`."""

from __future__ import annotations

import tomllib

import numpy as np
import pandas as pd
import pytest
import torch

from foundation_model.workflows._engine import AE_NAME, build_empty_model, build_head_config
from foundation_model.workflows._sections import EarlyStoppingConfig, ModelSectionConfig, TrainingSectionConfig
from foundation_model.workflows.finetune import apply_freeze_policy, build_finetune_config
from foundation_model.workflows.finetune import run as finetune_run
from foundation_model.workflows.recording import RunRecorder
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

_ELEMENTS = ["Fe", "Al", "Cu", "Ni", "Ti", "Zn", "Mg", "Ca", "Na", "Cl", "O", "Si"]
_FORMULAS = [f"{a}2 {b}3" for i, a in enumerate(_ELEMENTS) for b in _ELEMENTS[i + 1 :]][:24]

_MODEL = ModelSectionConfig(latent_dim=8, encoder_hidden=16, head_hidden_dim=8, n_kernel=4)
_TRAIN = TrainingSectionConfig(max_epochs=1, early_stopping=EarlyStoppingConfig(patience=2), accelerator="cpu", seed=1)


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "composition": _FORMULAS,
            "a": rng.normal(size=len(_FORMULAS)),
            "b": rng.normal(size=len(_FORMULAS)),
            "c": rng.integers(0, 3, size=len(_FORMULAS)),
        }
    )
    df.to_parquet(tmp_path / "x.parquet")
    return tmp_path


def _catalog_toml(data_dir) -> str:
    return f"""
[descriptor]
kind = "kmd"
n_grids = 4

[datasets.d1]
path = "{data_dir / "x.parquet"}"

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

[[tasks]]
name = "c"
kind = "classification"
dataset = "d1"
column = "c"
num_classes = 3
"""


def _catalog(data_dir) -> TaskCatalog:
    return TaskCatalog(build_task_catalog_config(tomllib.loads(_catalog_toml(data_dir))))


def _model_with_heads(data_dir, heads):
    cat = _catalog(data_dir)
    model = build_empty_model(cat, _MODEL, _TRAIN)
    for name in heads:
        model.add_task(build_head_config(cat, _MODEL, _TRAIN, name))
    return cat, model


def _save_checkpoint(model, tasks, path) -> None:
    torch.save({"model": model.state_dict(), "task_sequence": list(tasks)}, path)


def _finetune_cfg(data_dir, out, *, checkpoint, tasks, add_new_tasks=True, epochs=1):
    toml = (
        _catalog_toml(data_dir)
        + f"""
[model]
latent_dim = 8
encoder_hidden = 16
head_hidden_dim = 8
n_kernel = 4

[training]
max_epochs = 1
accelerator = "cpu"
seed = 1

[finetune]
tasks = {tasks!r}
epochs = {epochs}
add_new_tasks = {"true" if add_new_tasks else "false"}
"""
    )
    return build_finetune_config(tomllib.loads(toml), output_dir=str(out), checkpoint=str(checkpoint))


# --- freeze policy -----------------------------------------------------------------------


def test_freeze_policy(data_dir) -> None:
    _, model = _model_with_heads(data_dir, ["a", "b"])
    apply_freeze_policy(model, ["a"], freeze_encoder=True)
    assert all(p.requires_grad for p in model.task_heads["a"].parameters())  # target trainable
    assert not any(p.requires_grad for p in model.task_heads["b"].parameters())  # non-target frozen
    assert all(p.requires_grad for p in model.task_heads[AE_NAME].parameters())  # AE always trainable
    assert not any(p.requires_grad for p in model.encoder.parameters())  # encoder frozen
    assert not any(p.requires_grad for p in model.task_log_sigmas.parameters())  # balancer frozen


def test_freeze_encoder_false_leaves_encoder_trainable(data_dir) -> None:
    _, model = _model_with_heads(data_dir, ["a", "b"])
    apply_freeze_policy(model, ["a"], freeze_encoder=False)
    assert any(p.requires_grad for p in model.encoder.parameters())


# --- config validation -------------------------------------------------------------------


def test_empty_tasks_raises(data_dir) -> None:
    toml = _catalog_toml(data_dir) + '\n[finetune]\ntasks = []\ncheckpoint = "ck.pt"\n\n[output]\ndir = "o"\n'
    with pytest.raises(ValueError, match="tasks must be non-empty"):
        build_finetune_config(tomllib.loads(toml))


def test_task_not_in_catalog_raises(data_dir) -> None:
    toml = _catalog_toml(data_dir) + '\n[finetune]\ntasks = ["nope"]\ncheckpoint = "ck.pt"\n\n[output]\ndir = "o"\n'
    with pytest.raises(ValueError, match="unknown task"):
        build_finetune_config(tomllib.loads(toml))


def test_missing_checkpoint_raises(data_dir) -> None:
    toml = _catalog_toml(data_dir) + '\n[finetune]\ntasks = ["a"]\n\n[output]\ndir = "o"\n'
    with pytest.raises(ValueError, match="checkpoint must be given"):
        build_finetune_config(tomllib.loads(toml))


# --- engine ------------------------------------------------------------------------------


def test_finetune_smoke_and_all_heads_saved(data_dir, tmp_path) -> None:
    _, model = _model_with_heads(data_dir, ["a", "b"])
    ckpt = tmp_path / "ck.pt"
    _save_checkpoint(model, ["a", "b"], ckpt)

    out = tmp_path / "ft"
    cfg = _finetune_cfg(data_dir, out, checkpoint=ckpt, tasks=["a"], epochs=1)
    rec = RunRecorder(out)  # mimic the CLI: provenance is written by the subcommand, not run()
    rec.write_provenance(config=cfg, argv=["fm", "finetune"], seeds={"seed": cfg.training.seed})
    summary = finetune_run(cfg, rec)
    rec.close()

    assert (out / "training" / "final_model.pt").exists()
    assert (out / "training" / "finetune_summary.json").exists()
    assert (out / "run_provenance.json").exists()
    assert summary["finetuned_heads"] == ["a"]
    assert "a" in summary["metrics_before"] and "a" in summary["metrics_after"]

    # regression guard: the saved state_dict keeps ALL heads (disable/re-enable round-trip).
    saved = torch.load(out / "training" / "final_model.pt", weights_only=True)["model"]
    head_names = {k.split(".", 2)[1] for k in saved if k.startswith("task_heads.")}
    assert {"a", "b", AE_NAME} <= head_names


def test_add_new_task_head_added_and_trained(data_dir, tmp_path) -> None:
    _, model = _model_with_heads(data_dir, ["a", "b"])
    ckpt = tmp_path / "ck.pt"
    _save_checkpoint(model, ["a", "b"], ckpt)

    cfg = _finetune_cfg(data_dir, tmp_path / "add", checkpoint=ckpt, tasks=["c"], add_new_tasks=True)
    summary = finetune_run(cfg)
    assert summary["added_tasks"] == ["c"]
    saved = torch.load(tmp_path / "add" / "training" / "final_model.pt", weights_only=True)["model"]
    assert any(k.startswith("task_heads.c.") for k in saved)


def test_frozen_encoder_unchanged_after_finetune(data_dir, tmp_path) -> None:
    # freeze_encoder=True must keep encoder params AND BatchNorm buffers fixed across the fit.
    _, model = _model_with_heads(data_dir, ["a", "b"])
    ckpt = tmp_path / "ck.pt"
    _save_checkpoint(model, ["a", "b"], ckpt)
    before = {k: v.clone() for k, v in torch.load(ckpt, weights_only=True)["model"].items() if k.startswith("encoder.")}

    cfg = _finetune_cfg(data_dir, tmp_path / "frz", checkpoint=ckpt, tasks=["a"], epochs=2)
    finetune_run(cfg)

    after = torch.load(tmp_path / "frz" / "training" / "final_model.pt", weights_only=True)["model"]
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"encoder tensor changed: {key}"


def test_add_new_task_disabled_raises(data_dir, tmp_path) -> None:
    _, model = _model_with_heads(data_dir, ["a", "b"])
    ckpt = tmp_path / "ck.pt"
    _save_checkpoint(model, ["a", "b"], ckpt)

    cfg = _finetune_cfg(data_dir, tmp_path / "noadd", checkpoint=ckpt, tasks=["c"], add_new_tasks=False)
    with pytest.raises(ValueError, match="add_new_tasks"):
        finetune_run(cfg)
