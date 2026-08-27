# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and builders for the ``fm inverse`` tests.

A synthetic four-task catalog (two regressions, one classification, one kernel regression) and the
TOML fragments that drive it. Every test module here builds on these rather than on data files, so
the suite runs without the parquet corpus.
"""

from __future__ import annotations

import tomllib

import numpy as np
import pandas as pd
import pytest
import torch

from foundation_model.workflows._engine import build_empty_model, build_head_config
from foundation_model.workflows._sections import ModelSectionConfig, TrainingSectionConfig
from foundation_model.workflows.inverse import TargetSpec, build_inverse_config
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config


_ELEMENTS = ["Fe", "Al", "Cu", "Ni", "Ti", "Zn", "Mg", "Ca", "Na", "Cl", "O", "Si", "K", "Mn"]


_FORMULAS = [f"{a}2 {b}3" for i, a in enumerate(_ELEMENTS) for b in _ELEMENTS[i + 1 :]][:30]


_MODEL = ModelSectionConfig(latent_dim=8, encoder_hidden_dims=[16], head_hidden_dims=[8], n_kernel=4)


_TRAIN = TrainingSectionConfig(max_epochs=1, accelerator="cpu", seed=1)


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "composition": _FORMULAS,
            "a": rng.normal(size=len(_FORMULAS)),
            "b": rng.normal(size=len(_FORMULAS)),
            "mat": rng.integers(0, 3, size=len(_FORMULAS)),
            # kernel-regression sequences: list cells as literal strings (see AGENTS.md data notes)
            "k": ["[0.2, 0.8, 0.5]"] * len(_FORMULAS),
            "k_t": ["[0.1, 0.5, 0.9]"] * len(_FORMULAS),
            "split": (["train", "val", "test"] * len(_FORMULAS))[: len(_FORMULAS)],
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
name = "mat"
kind = "classification"
dataset = "d1"
column = "mat"
num_classes = 3

[[tasks]]
name = "k"
kind = "kernel_regression"
dataset = "d1"
column = "k"
t_column = "k_t"
"""


def _catalog(data_dir) -> TaskCatalog:
    return TaskCatalog(build_task_catalog_config(tomllib.loads(_catalog_toml(data_dir))))


def _checkpoint(data_dir, path) -> None:
    cat = _catalog(data_dir)
    model = build_empty_model(cat, _MODEL, _TRAIN)
    for name in ["a", "b", "mat", "k"]:
        model.add_task(build_head_config(cat, _MODEL, _TRAIN, name))
    torch.save({"model": model.state_dict(), "task_sequence": ["a", "b", "mat", "k"]}, path)


def _spec(cat: TaskCatalog, **kwargs) -> TargetSpec:
    """Build + kind-resolve a TargetSpec against the test catalog (mirrors the config builder)."""
    t = TargetSpec(**kwargs)
    t.resolve_kind(cat.task_spec(t.task))
    return t


# One scenario covering every target kind: value, direction-only, classification, curve.
_SCENARIO_ALL_KINDS = """
[[inverse.scenarios]]
name = "sc1"

[[inverse.scenarios.targets]]
task = "a"
value = -1.0

[[inverse.scenarios.targets]]
task = "b"
direction = "high"

[[inverse.scenarios.targets]]
task = "mat"
classes = [1]
weight = 2.0

[[inverse.scenarios.targets]]
task = "k"
points = [[0.1, 0.4], [0.9, 0.6]]
"""


def _inverse_cfg(data_dir, out, checkpoint, *, animation: str = "[]"):
    toml = (
        _catalog_toml(data_dir)
        + f"""
[model]
latent_dim = 8
encoder_hidden_dims = [16]
head_hidden_dims = [8]
n_kernel = 4

[inverse]
steps = 2
lr = 0.05
record_trajectory = true
animation_formats = {animation}

[inverse.seeds]
strategy = "top_objective"
n = 3
split = "all"
{_SCENARIO_ALL_KINDS}
[[inverse.paths]]
name = "latent_align1"
method = "latent"
ae_align_scale = 1.0

[[inverse.paths]]
name = "comp_seed_blend95"
method = "composition"
init = "seed"
seed_blend = 0.95

[output]
dir = "o"
"""
    )
    return build_inverse_config(tomllib.loads(toml), output_dir=str(out), checkpoint=str(checkpoint))


def _cfg_with_scenario(data_dir, scenario_toml: str, *, inverse_extra: str = ""):
    toml = (
        _catalog_toml(data_dir)
        + f"""
[inverse]
steps = 2
{inverse_extra}
[inverse.seeds]
n = 2
{scenario_toml}
[[inverse.paths]]
name = "latent1"
method = "latent"

[output]
dir = "o"
"""
    )
    return build_inverse_config(tomllib.loads(toml), checkpoint="ck.pt")


def _model_with_heads(data_dir):
    cat = _catalog(data_dir)
    model = build_empty_model(cat, _MODEL, _TRAIN)
    for name in ["a", "b", "mat", "k"]:
        model.add_task(build_head_config(cat, _MODEL, _TRAIN, name))
    return cat, model
