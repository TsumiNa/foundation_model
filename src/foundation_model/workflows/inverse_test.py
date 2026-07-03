# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`foundation_model.workflows.inverse`."""

from __future__ import annotations

import json
import tomllib

import numpy as np
import pandas as pd
import pytest
import torch

from foundation_model.workflows._engine import build_empty_model, build_head_config
from foundation_model.workflows._sections import ModelSectionConfig, TrainingSectionConfig
from foundation_model.workflows.inverse import (
    DEFAULT_ALLOY_PALETTE,
    InverseConfig,
    InverseMethod,
    PathConfig,
    ScenarioConfig,
    SeedConfig,
    SeedStrategy,
    _default_paths,
    build_inverse_config,
    select_seeds,
)
from foundation_model.workflows.inverse import run as inverse_run
from foundation_model.workflows.recording import RunRecorder
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

_ELEMENTS = ["Fe", "Al", "Cu", "Ni", "Ti", "Zn", "Mg", "Ca", "Na", "Cl", "O", "Si", "K", "Mn"]
_FORMULAS = [f"{a}2 {b}3" for i, a in enumerate(_ELEMENTS) for b in _ELEMENTS[i + 1 :]][:30]

_MODEL = ModelSectionConfig(latent_dim=8, encoder_hidden=16, head_hidden_dim=8, n_kernel=4)
_TRAIN = TrainingSectionConfig(max_epochs=1, accelerator="cpu", seed=1)


# --- config validation (no model needed) -------------------------------------------------


def test_scenario_reg_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ScenarioConfig(name="s", reg_tasks=["a"], reg_targets=[1.0, 2.0])


def test_latent_path_rejects_composition_key() -> None:
    with pytest.raises(ValueError, match="composition-only"):
        PathConfig(name="p", method=InverseMethod.LATENT, seed_blend=0.9)


def test_composition_path_rejects_ae_align_scale() -> None:
    with pytest.raises(ValueError, match="ae_align_scale"):
        PathConfig(name="p", method=InverseMethod.COMPOSITION, ae_align_scale=0.9)


def test_default_paths_count_and_kwargs() -> None:
    paths = _default_paths()
    assert len(paths) == 11
    latent = [p for p in paths if p.method is InverseMethod.LATENT]
    comp = [p for p in paths if p.method is InverseMethod.COMPOSITION]
    assert len(latent) == 3 and len(comp) == 8
    assert sorted(p.ae_align_scale for p in latent) == [0.0, 0.25, 1.0]
    k5_linear = next(p for p in comp if p.name.endswith("k5_linear"))
    assert k5_linear.max_elements == 5
    assert k5_linear.annealing_scale == 0.715
    assert k5_linear.annealing_schedule == {"step": [1.0], "scale": [0.0], "annealing_func": ["linear"]}
    assert len(DEFAULT_ALLOY_PALETTE) == 48
    # composition paths with an element list use the palette
    assert any(p.allowed_elements == DEFAULT_ALLOY_PALETTE for p in comp)


# --- fixtures ----------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "composition": _FORMULAS,
            "a": rng.normal(size=len(_FORMULAS)),
            "b": rng.normal(size=len(_FORMULAS)),
            "mat": rng.integers(0, 3, size=len(_FORMULAS)),
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
"""


def _catalog(data_dir) -> TaskCatalog:
    return TaskCatalog(build_task_catalog_config(tomllib.loads(_catalog_toml(data_dir))))


def _checkpoint(data_dir, path) -> None:
    cat = _catalog(data_dir)
    model = build_empty_model(cat, _MODEL, _TRAIN)
    for name in ["a", "b", "mat"]:
        model.add_task(build_head_config(cat, _MODEL, _TRAIN, name))
    torch.save({"model": model.state_dict(), "task_sequence": ["a", "b", "mat"]}, path)


def _inverse_cfg(data_dir, out, checkpoint, *, animation: str = "[]") -> InverseConfig:
    toml = (
        _catalog_toml(data_dir)
        + f"""
[model]
latent_dim = 8
encoder_hidden = 16
head_hidden_dim = 8
n_kernel = 4

[inverse]
steps = 2
lr = 0.05
record_trajectory = true
animation_formats = {animation}

[inverse.seeds]
strategy = "top_qc"
n = 3
split = "all"

[[inverse.scenarios]]
name = "sc1"
reg_tasks = ["a", "b"]
reg_targets = [-1.0, 1.0]
class_task = "mat"

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


# --- InverseConfig-level validation ------------------------------------------------------


def test_empty_scenarios_raises(data_dir) -> None:
    toml = _catalog_toml(data_dir) + '\n[inverse]\nsteps = 2\n[inverse.seeds]\nn = 2\n[output]\ndir = "o"\n'
    with pytest.raises(ValueError, match="scenario"):
        build_inverse_config(tomllib.loads(toml), checkpoint="ck.pt")


def test_bad_animation_format_raises(data_dir, tmp_path) -> None:
    with pytest.raises(ValueError, match="animation_formats"):
        _inverse_cfg(data_dir, tmp_path / "o", "ck.pt", animation='["mp4"]')


def test_precomputed_descriptor_plus_composition_path_raises(tmp_path) -> None:
    desc = pd.DataFrame(np.arange(6.0).reshape(3, 2), columns=["f0", "f1"])
    desc["composition"] = ["Fe2 O3", "Al2 O3", "Na1 Cl1"]
    desc.to_parquet(tmp_path / "desc.parquet")
    df = pd.DataFrame({"composition": ["Fe2 O3", "Al2 O3", "Na1 Cl1"], "a": [1.0, 2, 3], "mat": [0, 1, 2]})
    df.to_parquet(tmp_path / "x.parquet")
    toml = f"""
[descriptor]
kind = "precomputed"
path = "{tmp_path / "desc.parquet"}"

[datasets.d1]
path = "{tmp_path / "x.parquet"}"

[[tasks]]
name = "a"
kind = "regression"
dataset = "d1"
column = "a"

[[tasks]]
name = "mat"
kind = "classification"
dataset = "d1"
column = "mat"
num_classes = 3

[[inverse.scenarios]]
name = "sc1"
reg_tasks = ["a"]
reg_targets = [1.0]
class_task = "mat"

[[inverse.paths]]
name = "comp"
method = "composition"
"""
    with pytest.raises(ValueError, match="composition paths require descriptor.kind == 'kmd'"):
        build_inverse_config(tomllib.loads(toml), output_dir="o", checkpoint="ck.pt")


# --- seed selection ----------------------------------------------------------------------


def test_seed_selection_explicit_verbatim(data_dir) -> None:
    cat = _catalog(data_dir)
    model = build_empty_model(cat, _MODEL, _TRAIN)
    for name in ["a", "b", "mat"]:
        model.add_task(build_head_config(cat, _MODEL, _TRAIN, name))
    seed_cfg = SeedConfig(strategy=SeedStrategy.EXPLICIT, n=2, explicit=["Fe2 O3", "Al2 O3"])
    seeds = select_seeds(cat, model, seed_cfg, class_task="mat", class_indices=[1], device=torch.device("cpu"))
    assert set(seeds) <= {"Fe2 O3", "Al2 O3"}


def test_seed_selection_explicit_append_reduces_budget(data_dir) -> None:
    cat = _catalog(data_dir)
    model = build_empty_model(cat, _MODEL, _TRAIN)
    for name in ["a", "b", "mat"]:
        model.add_task(build_head_config(cat, _MODEL, _TRAIN, name))
    seed_cfg = SeedConfig(strategy=SeedStrategy.TOP_QC, n=4, split="all", explicit_append=["Fe2 O3"])
    seeds = select_seeds(cat, model, seed_cfg, class_task="mat", class_indices=[1], device=torch.device("cpu"))
    assert "Fe2 O3" in seeds  # appended seed always survives
    assert len(seeds) <= 4


# --- end-to-end smoke --------------------------------------------------------------------


def test_inverse_smoke_end_to_end(data_dir, tmp_path) -> None:
    ckpt = tmp_path / "ck.pt"
    _checkpoint(data_dir, ckpt)
    out = tmp_path / "inv"
    cfg = _inverse_cfg(data_dir, out, ckpt)
    rec = RunRecorder(out)
    rec.write_provenance(config=cfg, argv=["fm", "inverse"], seeds={"seed": 2025})
    summary = inverse_run(cfg, rec)
    rec.close()

    assert (out / "seeds.json").exists()
    assert (out / "inverse_design.json").exists()
    assert (out / "run_provenance.json").exists()
    sc = out / "sc1"
    for name in (
        "scenario.json",
        "results.json",
        "summary.json",
        "targets.json",
        "comparison.png",
        "qc_vs_secondary_scatter.png",
        "element_frequency_heatmap.png",
    ):
        assert (sc / name).exists(), name
    assert (sc / "seed_to_optimized__latent_align1.png").exists()
    results = json.loads((sc / "results.json").read_text())["results"]
    assert [r["path"] for r in results] == ["latent_align1", "comp_seed_blend95"]
    assert "qc_after_decode" in results[0] and "decoded_composition" in results[0]
    assert "sc1" in summary and len(summary["sc1"]) == 2
