# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``fm inverse`` TOML schema — targets, scenarios, paths, seeds.

Almost all of this is rejection: the schema's job is to turn a malformed run into an error at parse
time rather than a plausible-looking result three hours later.
"""

from __future__ import annotations

import tomllib

import numpy as np
import pandas as pd
import pytest

from foundation_model.workflows.inverse import (
    DEFAULT_ALLOY_PALETTE,
    InverseMethod,
    PathConfig,
    ScenarioConfig,
    SeedConfig,
    TargetKind,
    build_inverse_config,
    target_label,
)
from foundation_model.workflows.inverse.config import _default_paths

from .conftest import _catalog, _catalog_toml, _cfg_with_scenario, _inverse_cfg, _spec, _SCENARIO_ALL_KINDS


def test_old_schema_keys_fail_loudly(data_dir) -> None:
    with pytest.raises(ValueError, match="reg_tasks"):
        _cfg_with_scenario(data_dir, '[[inverse.scenarios]]\nname = "s"\nreg_tasks = ["a"]\nreg_targets = [1.0]\n')


def test_class_weight_key_rejected(data_dir) -> None:
    with pytest.raises(ValueError, match="class_weight"):
        _cfg_with_scenario(data_dir, _SCENARIO_ALL_KINDS, inverse_extra="class_weight = 5.0\n")


def test_top_qc_strategy_removed() -> None:
    with pytest.raises(ValueError, match="top_qc"):
        SeedConfig(strategy="top_qc")


def test_weighted_random_requires_weight_task() -> None:
    with pytest.raises(ValueError, match="requires seeds.weight_task"):
        SeedConfig(strategy="weighted_random")
    with pytest.raises(ValueError, match="only apply to"):
        SeedConfig(strategy="random", weight_task="a")
    with pytest.raises(ValueError, match="only apply to"):
        SeedConfig(strategy="random", weight_direction="low")
    with pytest.raises(ValueError, match="mutually exclusive"):
        SeedConfig(strategy="weighted_random", weight_task="a", weight_direction="high", weight_value=1.0)
    with pytest.raises(ValueError, match="'high' or 'low'"):
        SeedConfig(strategy="weighted_random", weight_task="a", weight_direction="up")
    assert SeedConfig(strategy="weighted_random", weight_task="a").weight_direction == "high"
    assert SeedConfig(strategy="weighted_random", weight_task="a", weight_value=0.5).weight_direction is None


def _weighted_seed_cfg_toml(data_dir, weight_task: str) -> str:
    return (
        _catalog_toml(data_dir)
        + f"""
[inverse]
steps = 2

[inverse.seeds]
strategy = "weighted_random"
weight_task = "{weight_task}"
n = 2
{_SCENARIO_ALL_KINDS}
[[inverse.paths]]
name = "latent1"
method = "latent"

[output]
dir = "o"
"""
    )


def test_weighted_random_weight_task_validated_against_catalog(data_dir) -> None:
    with pytest.raises(ValueError, match="not a catalog task"):
        build_inverse_config(tomllib.loads(_weighted_seed_cfg_toml(data_dir, "nope")), checkpoint="ck.pt")
    with pytest.raises(ValueError, match="must be a regression task"):
        build_inverse_config(tomllib.loads(_weighted_seed_cfg_toml(data_dir, "k")), checkpoint="ck.pt")


@pytest.mark.parametrize(
    ("target_toml", "match"),
    [
        ('task = "a"\nvalue = 1.0\ndirection = "high"', "exactly one of value or direction"),
        ('task = "a"', "exactly one of value or direction"),
        ('task = "a"\npoints = [[0.0, 1.0]]', "value/direction, not points/classes"),
        ('task = "a"\nclasses = [1]', "value/direction, not points/classes"),
        ('task = "a"\nvalue = 1.0\nweight = 0.0', "weight must be > 0"),
        ('task = "a"\nvalue = 1.0\ndirections = "high"', "unknown key"),
        ('task = "nope"\nvalue = 1.0', "unknown task"),
        ('task = "k"', "non-empty points"),
        ('task = "k"\npoints = [[0.0, 1.0, 2.0]]', r"\[t, y\] pairs"),
        ('task = "k"\nvalue = 1.0', "points only"),
        ('task = "mat"', "non-empty classes"),
        ('task = "mat"\nclasses = [3]', "out of range"),
        ('task = "mat"\nclasses = [0, 1, 2]', "strict subset"),
        ('task = "mat"\nclasses = [1]\ndirection = "down"', "'high' or 'low'"),
    ],
)
def test_target_validation_matrix(data_dir, target_toml: str, match: str) -> None:
    scenario = f'[[inverse.scenarios]]\nname = "s"\n[[inverse.scenarios.targets]]\n{target_toml}\n'
    with pytest.raises(ValueError, match=match):
        _cfg_with_scenario(data_dir, scenario)


def test_scenario_needs_targets(data_dir) -> None:
    with pytest.raises(ValueError, match="at least one"):
        _cfg_with_scenario(data_dir, '[[inverse.scenarios]]\nname = "s"\n')


def test_scenario_duplicate_target_task(data_dir) -> None:
    scenario = (
        '[[inverse.scenarios]]\nname = "s"\n'
        '[[inverse.scenarios.targets]]\ntask = "a"\nvalue = 1.0\n'
        '[[inverse.scenarios.targets]]\ntask = "a"\ndirection = "low"\n'
    )
    with pytest.raises(ValueError, match="duplicate target task"):
        _cfg_with_scenario(data_dir, scenario)


def test_classification_direction_defaults_to_high(data_dir) -> None:
    cfg = _cfg_with_scenario(
        data_dir, '[[inverse.scenarios]]\nname = "s"\n[[inverse.scenarios.targets]]\ntask = "mat"\nclasses = [1]\n'
    )
    (target,) = cfg.scenarios[0].targets
    assert target.kind is TargetKind.CLASS and target.direction == "high"


def test_target_labels(data_dir) -> None:
    cat = _catalog(data_dir)
    assert target_label(_spec(cat, task="a", value=-1.0)) == "a→-1"
    assert target_label(_spec(cat, task="b", direction="low")) == "b↓"
    assert target_label(_spec(cat, task="k", points=[[0.0, 1.0], [1.0, 2.0]])) == "k~curve(2pts)"
    assert target_label(_spec(cat, task="mat", classes=[1, 2], direction="low")) == "P(mat∈{1,2})↓"


def test_scenario_config_direct_construction_requires_targets() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ScenarioConfig(name="s", targets=[])


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

[[inverse.scenarios]]
name = "sc1"

[[inverse.scenarios.targets]]
task = "a"
value = 1.0

[[inverse.paths]]
name = "comp"
method = "composition"
"""
    with pytest.raises(ValueError, match="composition paths require descriptor.kind == 'kmd'"):
        build_inverse_config(tomllib.loads(toml), output_dir="o", checkpoint="ck.pt")


def test_seed_and_accelerator_config(data_dir, tmp_path) -> None:
    # --seed/--accelerator route into [inverse] and must not be rejected as unknown root keys.
    cfg = _cfg_with_scenario(data_dir, _SCENARIO_ALL_KINDS, inverse_extra='seed = 7\naccelerator = "cpu"\n')
    assert cfg.seed == 7 and cfg.accelerator == "cpu"


def test_invalid_accelerator_raises(data_dir, tmp_path) -> None:
    with pytest.raises(ValueError, match="accelerator must be one of"):
        _cfg_with_scenario(data_dir, _SCENARIO_ALL_KINDS, inverse_extra='accelerator = "cpuu"\n')
