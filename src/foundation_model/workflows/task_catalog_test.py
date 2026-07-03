# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`foundation_model.workflows.task_catalog`."""

from __future__ import annotations

import tomllib

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.model_config import (
    ClassificationTaskConfig,
    KernelRegressionTaskConfig,
    RegressionTaskConfig,
)
from foundation_model.workflows.task_catalog import (
    ScalerSpec,
    TaskCatalog,
    TaskKind,
    build_task_catalog_config,
    init_kernel_centers_sigmas,
)

# Element counts: Fe2 O3=2, Al2 O3=2, Na1 Cl1=2, Ca1 Ti1 O3=3, Li2 O1=2, Fe1=1
_COMPS = ["Fe2O3", "Al2O3", "NaCl", "Ca Ti O3", "Li2 O", "Fe"]


def _base_toml(qc_path: str) -> str:
    return f"""
[data]
composition_column = "composition"
batch_size = 8

[descriptor]
kind = "kmd"
n_grids = 4

[datasets.qc]
path = "{qc_path}"

[[tasks]]
name = "density"
kind = "regression"
dataset = "qc"
column = "density"

[[tasks]]
name = "mat"
kind = "classification"
dataset = "qc"
column = "mtype"
num_classes = 3
"""


def _build(toml_str: str):
    return build_task_catalog_config(tomllib.loads(toml_str))


# --- build_task_catalog_config -----------------------------------------------------------


def test_build_happy_path() -> None:
    cfg = _build(_base_toml("data/qc.parquet"))
    assert [t.name for t in cfg.tasks] == ["density", "mat"]
    assert cfg.tasks[0].kind is TaskKind.REGRESSION
    assert cfg.tasks[1].kind is TaskKind.CLASSIFICATION
    assert cfg.descriptor.kind == "kmd" and cfg.descriptor.n_grids == 4
    assert "qc" in cfg.datasets


def test_legacy_kind_aliases_normalize() -> None:
    toml = """
[descriptor]
kind = "kmd"

[datasets.qc]
path = "data/qc.parquet"

[[tasks]]
name = "dos"
kind = "kr"
dataset = "qc"
column = "dos"
t_column = "energy"
"""
    cfg = _build(toml)
    assert cfg.tasks[0].kind is TaskKind.KERNEL_REGRESSION


def test_unknown_key_raises() -> None:
    toml = """
[data]
composition_column = "composition"
not_a_field = 1

[datasets.qc]
path = "data/qc.parquet"

[[tasks]]
name = "density"
kind = "regression"
dataset = "qc"
column = "density"
"""
    with pytest.raises(ValueError, match="not_a_field"):
        _build(toml)


def test_unknown_task_key_raises() -> None:
    toml = (
        _base_toml("data/qc.parquet")
        + """
[[tasks]]
name = "bogus"
kind = "regression"
dataset = "qc"
column = "x"
not_a_field = 1
"""
    )
    with pytest.raises(ValueError, match="not_a_field"):
        _build(toml)


def test_kr_without_t_column_raises() -> None:
    toml = """
[datasets.qc]
path = "data/qc.parquet"

[[tasks]]
name = "dos"
kind = "kernel_regression"
dataset = "qc"
column = "dos"
"""
    with pytest.raises(ValueError, match="t_column"):
        _build(toml)


def test_clf_without_num_classes_raises() -> None:
    toml = """
[datasets.qc]
path = "data/qc.parquet"

[[tasks]]
name = "mat"
kind = "classification"
dataset = "qc"
column = "mtype"
"""
    with pytest.raises(ValueError, match="num_classes"):
        _build(toml)


def test_duplicate_task_names_raise() -> None:
    toml = (
        _base_toml("data/qc.parquet")
        + """
[[tasks]]
name = "density"
kind = "regression"
dataset = "qc"
column = "density"
"""
    )
    with pytest.raises(ValueError, match="Duplicate task names"):
        _build(toml)


def test_task_dataset_not_defined_raises() -> None:
    toml = """
[datasets.qc]
path = "data/qc.parquet"

[[tasks]]
name = "density"
kind = "regression"
dataset = "missing"
column = "density"
"""
    with pytest.raises(ValueError, match="dataset 'missing'"):
        _build(toml)


@pytest.mark.parametrize(
    ("replay", "ok"),
    [(0.1, True), (500, True), (0.0, False), (-1, False), (1.0, False)],
)
def test_replay_validation(replay: float, ok: bool) -> None:
    toml = f"""
[datasets.qc]
path = "data/qc.parquet"

[[tasks]]
name = "density"
kind = "regression"
dataset = "qc"
column = "density"
replay = {replay!r}
"""
    if ok:
        assert _build(toml).tasks[0].replay == replay
    else:
        with pytest.raises(ValueError, match="replay"):
            _build(toml)


def test_unsupported_extension_raises() -> None:
    toml = """
[datasets.qc]
path = "data/qc.pickle"

[[tasks]]
name = "density"
kind = "regression"
dataset = "qc"
column = "density"
"""
    with pytest.raises(ValueError, match="unsupported data file"):
        _build(toml)


def test_multi_suffix_extension_accepted() -> None:
    toml = """
[datasets.qc]
path = "data/qc.pd.parquet"

[[tasks]]
name = "density"
kind = "regression"
dataset = "qc"
column = "density"
"""
    assert "qc" in _build(toml).datasets


def test_kmd_n_grids_below_two_raises() -> None:
    toml = """
[descriptor]
kind = "kmd"
n_grids = 1

[datasets.qc]
path = "data/qc.parquet"

[[tasks]]
name = "density"
kind = "regression"
dataset = "qc"
column = "density"
"""
    with pytest.raises(ValueError, match="n_grids must be >= 2"):
        _build(toml)


# --- TaskCatalog with synthetic fixtures -------------------------------------------------


@pytest.fixture
def catalog_dir(tmp_path):
    qc = pd.DataFrame(
        {
            "composition": _COMPS,
            "density": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "mtype": [0, 1, 2, 0, 1, 2],
            "split": ["train", "train", "test", "val", "train", "test"],
        }
    )
    qc.to_parquet(tmp_path / "qc.parquet")
    kr = pd.DataFrame(
        {
            "composition": _COMPS,
            "dos": ["[0.1, 0.2, 0.3]"] * 6,
            "energy": ["[1.0, 2.0, 3.0]"] * 6,
            "split": ["train"] * 4 + ["test"] * 2,
        }
    )
    kr.to_parquet(tmp_path / "kr.parquet")
    return tmp_path


def _catalog(catalog_dir, *, extra_tasks: str = "", data_extra: str = "") -> TaskCatalog:
    toml = f"""
[data]
composition_column = "composition"
batch_size = 4

[descriptor]
kind = "kmd"
n_grids = 4

[datasets.qc]
path = "{catalog_dir / "qc.parquet"}"
{data_extra}

[datasets.kr]
path = "{catalog_dir / "kr.parquet"}"

[[tasks]]
name = "density"
kind = "regression"
dataset = "qc"
column = "density"

[[tasks]]
name = "mat"
kind = "classification"
dataset = "qc"
column = "mtype"
num_classes = 3

[[tasks]]
name = "dos"
kind = "kernel_regression"
dataset = "kr"
column = "dos"
t_column = "energy"
{extra_tasks}
"""
    return TaskCatalog(_build(toml))


def test_task_frames_composition_keyed(catalog_dir) -> None:
    cat = _catalog(catalog_dir)
    frames = cat.task_frames(["density"])
    frame = frames["density"]
    assert list(frame.index) == ["Fe2 O3", "Al2 O3", "Na1 Cl1", "Ca1 Ti1 O3", "Li2 O1", "Fe1"]
    assert "density" in frame.columns and "split" in frame.columns


def test_task_frames_nan_preserved(catalog_dir) -> None:
    frame = _catalog(catalog_dir).task_frames(["density"])["density"]
    assert frame["density"].isna().sum() == 1  # NaCl row NaN kept, not dropped


def test_min_elements_drops_rows(catalog_dir) -> None:
    cat = _catalog(catalog_dir, data_extra="min_elements = 2")
    frame = cat.task_frames(["density"])["density"]
    assert "Fe1" not in frame.index  # single-element composition dropped
    assert len(frame) == 5


def test_sample_caps_rows(catalog_dir) -> None:
    cat = _catalog(catalog_dir, data_extra="sample = 3")
    frame = cat.task_frames(["density"])["density"]
    assert len(frame) == 3


def test_build_task_config_per_kind(catalog_dir) -> None:
    cat = _catalog(catalog_dir)
    reg = cat.build_task_config("density", latent_dim=8, head_hidden_dim=4, n_kernel=5, lr=1e-3)
    clf = cat.build_task_config("mat", latent_dim=8, head_hidden_dim=4, n_kernel=5, lr=1e-3)
    kr = cat.build_task_config("dos", latent_dim=8, head_hidden_dim=4, n_kernel=5, lr=1e-3)
    assert isinstance(reg, RegressionTaskConfig) and reg.dims == [8, 4, 1]
    assert isinstance(clf, ClassificationTaskConfig) and clf.num_classes == 3
    assert clf.class_weights is not None and len(clf.class_weights) == 3
    assert isinstance(kr, KernelRegressionTaskConfig) and kr.kernel_num_centers == 5
    assert reg.optimizer is not None and reg.optimizer.lr == 1e-3


def test_build_task_config_per_task_lr_override(catalog_dir) -> None:
    cat = _catalog(catalog_dir, extra_tasks="lr = 0.02")  # per-task override on the dos task
    kr = cat.build_task_config("dos", latent_dim=8, head_hidden_dim=4, n_kernel=5, lr=1e-3)
    assert kr.optimizer is not None and kr.optimizer.lr == 0.02


def test_kernel_init_finite(catalog_dir) -> None:
    kr = _catalog(catalog_dir).build_task_config("dos", latent_dim=8, head_hidden_dim=4, n_kernel=5, lr=1e-3)
    assert isinstance(kr, KernelRegressionTaskConfig)
    assert kr.kernel_centers_init is not None and len(kr.kernel_centers_init) == 5
    assert kr.kernel_sigmas_init is not None
    assert np.all(np.isfinite(kr.kernel_centers_init)) and np.all(np.isfinite(kr.kernel_sigmas_init))


def test_init_kernel_centers_sigmas_empty() -> None:
    centers, sigmas = init_kernel_centers_sigmas(np.array([]), 5)
    assert centers == [] and sigmas == []


def test_inverse_transform_identity_without_scaler(catalog_dir) -> None:
    cat = _catalog(catalog_dir)
    values = np.array([1.0, 2.0, 3.0])
    assert np.allclose(cat.inverse_transform("density", values), values)


def test_inverse_transform_roundtrip_with_scaler(catalog_dir) -> None:
    raw = np.array([[10.0], [20.0], [30.0], [40.0]])
    scaler = StandardScaler().fit(raw)
    scaler_path = catalog_dir / "scalers.z"
    joblib.dump({"density": scaler}, scaler_path)
    cat = _catalog(catalog_dir)
    cat.task_spec("density").scaler = ScalerSpec(path=scaler_path, key="density")
    normalized = scaler.transform(raw).reshape(-1)
    restored = cat.inverse_transform("density", normalized)
    assert np.allclose(restored, raw.reshape(-1))


def test_kmd_descriptor_deterministic_and_cached(catalog_dir) -> None:
    cat = _catalog(catalog_dir)
    fn = cat.descriptor_fn()
    keys = ["Fe2 O3", "Al2 O3"]
    first = fn(keys)
    second = fn(keys)
    assert list(first.index) == keys
    assert np.allclose(first.values, second.values)
    assert cat.kmd() is not None
    assert cat.descriptor_dim == first.shape[1]


def test_precomputed_non_formula_keys_preserved(tmp_path) -> None:
    # Non-formula IDs (e.g. Materials Project IDs) that normalize_composition can't parse must be
    # preserved via canonical_key, not dropped — otherwise a precomputed catalog is emptied.
    ids = ["mp-149", "mp-2534", "mp-66"]
    qc = pd.DataFrame({"composition": ids, "density": [1.0, 2.0, 3.0]})
    qc.to_parquet(tmp_path / "qc.parquet")
    desc = pd.DataFrame(np.arange(6, dtype=float).reshape(3, 2), columns=["f0", "f1"])
    desc["composition"] = ids
    desc.to_parquet(tmp_path / "desc.parquet")
    toml = f"""
[descriptor]
kind = "precomputed"
path = "{tmp_path / "desc.parquet"}"

[datasets.qc]
path = "{tmp_path / "qc.parquet"}"

[[tasks]]
name = "density"
kind = "regression"
dataset = "qc"
column = "density"
"""
    cat = TaskCatalog(_build(toml))
    frame = cat.task_frames(["density"])["density"]
    assert list(frame.index) == ids  # rows kept, not dropped
    out = cat.descriptor_fn()(ids)
    assert list(out.index) == ids


def test_precomputed_descriptor_source(tmp_path) -> None:
    qc = pd.DataFrame({"composition": _COMPS, "density": [1.0, 2, 3, 4, 5, 6]})
    qc.to_parquet(tmp_path / "qc.parquet")
    desc = pd.DataFrame(
        np.arange(12, dtype=float).reshape(6, 2),
        columns=["f0", "f1"],
    )
    desc["composition"] = _COMPS
    desc.to_parquet(tmp_path / "desc.parquet")
    toml = f"""
[descriptor]
kind = "precomputed"
path = "{tmp_path / "desc.parquet"}"

[datasets.qc]
path = "{tmp_path / "qc.parquet"}"

[[tasks]]
name = "density"
kind = "regression"
dataset = "qc"
column = "density"
"""
    cat = TaskCatalog(_build(toml))
    assert cat.kmd() is None
    fn = cat.descriptor_fn()
    out = fn(["Fe2 O3", "Al2 O3"])
    assert list(out.index) == ["Fe2 O3", "Al2 O3"]
    assert cat.descriptor_dim == 2


def test_build_datamodule(catalog_dir) -> None:
    cat = _catalog(catalog_dir)
    dm = cat.build_datamodule(["density", "mat"], masking_ratios={"density": 0.5}, predict_idx="test")
    assert isinstance(dm, CompoundDataModule)
    cfgs = {c.name: c for c in dm.task_configs}
    assert cfgs["density"].task_masking_ratio == 0.5
    assert cfgs["density"].predict_idx == "test"
