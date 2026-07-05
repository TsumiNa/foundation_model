# Copyright 2027 TsumiNa.
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
    InverseMethod,
    PathConfig,
    ScenarioConfig,
    SeedConfig,
    SeedStrategy,
    TargetKind,
    TargetSpec,
    _default_paths,
    build_inverse_config,
    select_seeds,
    target_label,
)
from foundation_model.workflows.inverse import run as inverse_run
from foundation_model.workflows.recording import RunRecorder
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

_ELEMENTS = ["Fe", "Al", "Cu", "Ni", "Ti", "Zn", "Mg", "Ca", "Na", "Cl", "O", "Si", "K", "Mn"]
_FORMULAS = [f"{a}2 {b}3" for i, a in enumerate(_ELEMENTS) for b in _ELEMENTS[i + 1 :]][:30]

_MODEL = ModelSectionConfig(latent_dim=8, encoder_hidden_dims=[16], head_hidden_dims=[8], n_kernel=4)
_TRAIN = TrainingSectionConfig(max_epochs=1, accelerator="cpu", seed=1)


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


# --- target/scenario validation ------------------------------------------------------------


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
    with pytest.raises(ValueError, match="only applies to"):
        SeedConfig(strategy="random", weight_task="a")


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


def test_weighted_random_selection_deterministic_and_from_pool(data_dir) -> None:
    cat, model = _model_with_heads(data_dir)
    seed_cfg = SeedConfig(strategy="weighted_random", weight_task="a", n=4, split="all")
    specs = [_spec(cat, task="a", value=1.0)]
    s1 = select_seeds(cat, model, seed_cfg, targets=specs, device=torch.device("cpu"))
    s2 = select_seeds(cat, model, seed_cfg, targets=specs, device=torch.device("cpu"))
    assert s1 == s2 and len(s1) == 4
    assert set(s1) <= set(str(c) for c in cat.task_frames(["a"])["a"].index)


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


# --- path validation (unchanged behavior) ---------------------------------------------------


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


# --- seed selection ----------------------------------------------------------------------


def _model_with_heads(data_dir):
    cat = _catalog(data_dir)
    model = build_empty_model(cat, _MODEL, _TRAIN)
    for name in ["a", "b", "mat", "k"]:
        model.add_task(build_head_config(cat, _MODEL, _TRAIN, name))
    return cat, model


def test_seed_selection_explicit_verbatim(data_dir) -> None:
    cat, model = _model_with_heads(data_dir)
    seed_cfg = SeedConfig(strategy=SeedStrategy.EXPLICIT, n=2, explicit=["Fe2 O3", "Al2 O3"])
    seeds = select_seeds(cat, model, seed_cfg, targets=[_spec(cat, task="a", value=1.0)], device=torch.device("cpu"))
    assert set(seeds) <= {"Fe2 O3", "Al2 O3"}


def test_seed_selection_explicit_append_reduces_budget(data_dir) -> None:
    cat, model = _model_with_heads(data_dir)
    seed_cfg = SeedConfig(strategy=SeedStrategy.TOP_OBJECTIVE, n=4, split="all", explicit_append=["Fe2 O3"])
    seeds = select_seeds(cat, model, seed_cfg, targets=[_spec(cat, task="a", value=1.0)], device=torch.device("cpu"))
    assert "Fe2 O3" in seeds  # appended seed always survives
    assert len(seeds) <= 4


def test_seed_selection_works_without_classification_target(data_dir) -> None:
    # Regression guard for the QC de-hardcoding: the candidate pool must not depend on any
    # classification head/frame.
    cat, model = _model_with_heads(data_dir)
    seed_cfg = SeedConfig(strategy=SeedStrategy.RANDOM, n=3, split="all")
    seeds = select_seeds(
        cat, model, seed_cfg, targets=[_spec(cat, task="a", direction="low")], device=torch.device("cpu")
    )
    assert len(seeds) == 3


def test_top_objective_ranking_matches_evaluate_targets(data_dir) -> None:
    cat, model = _model_with_heads(data_dir)
    specs = [_spec(cat, task="a", value=5.0)]
    seed_cfg = SeedConfig(strategy=SeedStrategy.TOP_OBJECTIVE, n=3, split="all", dedup_by_element_system=False)
    seeds = select_seeds(cat, model, seed_cfg, targets=specs, device=torch.device("cpu"))

    # Hand-compute the expected ranking with the exact same pool order + scoring call.
    frame = cat.task_frames(["a"])["a"]
    descriptor_fn = cat.descriptor_fn()
    pool = [c for c in frame.index if not descriptor_fn([c]).empty]
    desc = descriptor_fn(pool)
    kept = [c for c in pool if c in desc.index]
    x = torch.tensor(desc.loc[kept].values, dtype=torch.float32)
    _, objective = model.evaluate_targets(x, [s.to_model_target() for s in specs])
    expected = [kept[i] for i in np.argsort(objective.numpy(), kind="stable")][:3]
    assert seeds == expected


# --- config routing ------------------------------------------------------------------------


def test_seed_and_accelerator_config(data_dir, tmp_path) -> None:
    # --seed/--accelerator route into [inverse] and must not be rejected as unknown root keys.
    cfg = _cfg_with_scenario(data_dir, _SCENARIO_ALL_KINDS, inverse_extra='seed = 7\naccelerator = "cpu"\n')
    assert cfg.seed == 7 and cfg.accelerator == "cpu"


def test_invalid_accelerator_raises(data_dir, tmp_path) -> None:
    with pytest.raises(ValueError, match="accelerator must be one of"):
        _cfg_with_scenario(data_dir, _SCENARIO_ALL_KINDS, inverse_extra='accelerator = "cpuu"\n')


# --- end-to-end smoke --------------------------------------------------------------------


def test_trajectory_static_and_svg_animation_emitted(data_dir, tmp_path) -> None:
    ckpt = tmp_path / "ck.pt"
    _checkpoint(data_dir, ckpt)
    out = tmp_path / "inv"
    cfg = _inverse_cfg(data_dir, out, ckpt, animation='["svg"]')  # svg avoids slow FuncAnimation writers
    rec = RunRecorder(out)
    rec.write_provenance(config=cfg, argv=["fm", "inverse"], seeds={"seed": 2025})
    inverse_run(cfg, rec)
    rec.close()
    traj = out / "sc1" / "trajectories"
    assert (traj / "latent_align1_trajectory.png").exists()  # static plot always
    assert (traj / "latent_align1_trajectory.svg").exists()  # requested animation format
    npz = np.load(traj / "latent_align1.npz", allow_pickle=False)
    assert npz["targets"].shape[2] == 4  # one channel per target
    assert [str(v) for v in npz["labels"]] == ["a→-1", "b↑", "P(mat∈{1})↑", "k~curve(2pts)"]


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
        "objective_vs_targets_scatter.png",
        "element_frequency_heatmap.png",
    ):
        assert (sc / name).exists(), name
    assert (sc / "seed_to_optimized__latent_align1.png").exists()

    scenario = json.loads((sc / "scenario.json").read_text())
    assert [t["task"] for t in scenario["targets"]] == ["a", "b", "mat", "k"]
    assert [t["kind"] for t in scenario["targets"]] == ["value", "direction", "class", "curve"]

    payload = json.loads((sc / "results.json").read_text())
    assert set(payload["seed_predictions"]["channels"]) == {"a", "b", "mat", "k"}
    results = payload["results"]
    assert [r["path"] for r in results] == ["latent_align1", "comp_seed_blend95"]
    assert "objective_after_decode" in results[0] and "decoded_composition" in results[0]
    assert set(results[0]["channels_after_decode"]) == {"a", "b", "mat", "k"}
    assert "sc1" in summary and len(summary["sc1"]) == 2
    assert all("objective_mean" in row for row in summary["sc1"])
