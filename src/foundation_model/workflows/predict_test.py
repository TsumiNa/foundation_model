# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`foundation_model.workflows.predict`."""

from __future__ import annotations

import tomllib

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from foundation_model.workflows._engine import build_empty_model, build_head_config
from foundation_model.workflows._sections import ModelSectionConfig, TrainingSectionConfig
from foundation_model.workflows.predict import PredictConfig, build_predict_config
from foundation_model.workflows.predict import run as predict_run
from foundation_model.workflows.recording import RunRecorder
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

_ELEMENTS = ["Fe", "Al", "Cu", "Ni", "Ti", "Zn", "Mg", "Ca", "Na", "Cl", "O", "Si"]
_FORMULAS = [f"{a}2 {b}3" for i, a in enumerate(_ELEMENTS) for b in _ELEMENTS[i + 1 :]][:24]

_MODEL = ModelSectionConfig(latent_dim=8, encoder_hidden_dims=[16], head_hidden_dims=[8], n_kernel=4)
_TRAIN = TrainingSectionConfig(max_epochs=1, accelerator="cpu", seed=1)


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=len(_FORMULAS))
    a[3] = np.nan  # one masked regression target
    df = pd.DataFrame(
        {
            "composition": _FORMULAS,
            "a": a,
            "mat": rng.integers(0, 3, size=len(_FORMULAS)),
            "split": (["train", "val", "test"] * len(_FORMULAS))[: len(_FORMULAS)],
        }
    )
    df.to_parquet(tmp_path / "x.parquet")
    return tmp_path


def _catalog_toml(data_dir, *, scaler: str = "") -> str:
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
{scaler}

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
    for name in ["a", "mat"]:
        model.add_task(build_head_config(cat, _MODEL, _TRAIN, name))
    torch.save({"model": model.state_dict(), "task_sequence": ["a", "mat"]}, path)


def _predict_cfg(data_dir, out, checkpoint, *, predict_block: str, scaler: str = "") -> PredictConfig:
    toml = (
        _catalog_toml(data_dir, scaler=scaler)
        + f"""
[model]
latent_dim = 8
encoder_hidden_dims = [16]
head_hidden_dims = [8]
n_kernel = 4

[predict]
{predict_block}

[output]
dir = "o"
"""
    )
    return build_predict_config(tomllib.loads(toml), output_dir=str(out), checkpoint=str(checkpoint))


def _run(cfg, out) -> dict:
    rec = RunRecorder(out)
    rec.write_provenance(config=cfg, argv=["fm", "predict"], seeds={"seed": 2025})
    result = predict_run(cfg, rec)
    rec.close()
    return result


# --- config validation -------------------------------------------------------------------


def test_invalid_split_raises(data_dir) -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        _predict_cfg(data_dir, "o", "ck.pt", predict_block='split = "bogus"')


def test_unknown_task_raises(data_dir) -> None:
    with pytest.raises(ValueError, match="unknown task"):
        _predict_cfg(data_dir, "o", "ck.pt", predict_block='tasks = ["nope"]')


def test_seed_and_accelerator_parsed(data_dir) -> None:
    cfg = _predict_cfg(data_dir, "o", "ck.pt", predict_block='seed = 7\naccelerator = "cpu"')
    assert cfg.seed == 7 and cfg.accelerator == "cpu"


def test_invalid_accelerator_raises(data_dir) -> None:
    # a typo must fail, not silently fall back to CUDA-if-available.
    with pytest.raises(ValueError, match="accelerator must be one of"):
        _predict_cfg(data_dir, "o", "ck.pt", predict_block='accelerator = "cpuu"')


def test_predict_unknown_key_rejected(data_dir) -> None:
    with pytest.raises(ValueError, match="unknown key"):
        _predict_cfg(data_dir, "o", "ck.pt", predict_block="bogus = 1")


def test_compositions_override_split(data_dir) -> None:
    cfg = _predict_cfg(data_dir, "o", "ck.pt", predict_block='split = "test"\ncompositions = ["Fe2 O3"]')
    assert cfg.compositions == ["Fe2 O3"]  # documented: compositions win over split


# --- engine ------------------------------------------------------------------------------


def test_predict_smoke_split_test(data_dir, tmp_path) -> None:
    ckpt = tmp_path / "ck.pt"
    _checkpoint(data_dir, ckpt)
    out = tmp_path / "pred"
    cfg = _predict_cfg(data_dir, out, ckpt, predict_block='split = "test"')
    result = _run(cfg, out)

    assert (out / "run_provenance.json").exists()
    for task in ("a", "mat"):
        assert (out / "predict" / f"{task}_pred.parquet").exists()
    reg = pd.read_parquet(out / "predict" / "a_pred.parquet")
    assert {"composition", "pred", "true"} <= set(reg.columns)
    assert np.isfinite(reg["pred"]).all()
    if "a" in result["metrics"]:
        assert np.isfinite(result["metrics"]["a"]["mae"])


def test_explicit_compositions_in_order(data_dir, tmp_path) -> None:
    ckpt = tmp_path / "ck.pt"
    _checkpoint(data_dir, ckpt)
    out = tmp_path / "pred"
    wanted = ["Fe2 O3", "Al2 O3", "Cu2 Ni3"]
    cfg = _predict_cfg(data_dir, out, ckpt, predict_block=f'tasks = ["a"]\ncompositions = {wanted!r}')
    _run(cfg, out)
    reg = pd.read_parquet(out / "predict" / "a_pred.parquet")
    assert list(reg["composition"]) == wanted  # exactly those rows, in order


def test_head_absent_from_checkpoint_errors(data_dir, tmp_path) -> None:
    ckpt = tmp_path / "ck.pt"
    _checkpoint(data_dir, ckpt)  # heads a, mat only
    out = tmp_path / "pred"
    # Catalog holds every checkpoint task (a, mat) plus an extra head 'b' that the checkpoint
    # lacks; requesting 'b' must error listing the available heads.
    toml = f"""
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
column = "a"
[[tasks]]
name = "mat"
kind = "classification"
dataset = "d1"
column = "mat"
num_classes = 3
[model]
latent_dim = 8
encoder_hidden_dims = [16]
head_hidden_dims = [8]
n_kernel = 4
[predict]
tasks = ["b"]
[output]
dir = "o"
"""
    cfg = build_predict_config(tomllib.loads(toml), output_dir=str(out), checkpoint=str(ckpt))
    with pytest.raises(ValueError, match="not in the checkpoint"):
        _run(cfg, out)


def test_masked_nan_target_in_parquet_excluded_from_metrics(data_dir, tmp_path) -> None:
    ckpt = tmp_path / "ck.pt"
    _checkpoint(data_dir, ckpt)
    out = tmp_path / "pred"
    cfg = _predict_cfg(data_dir, out, ckpt, predict_block='tasks = ["a"]\nsplit = "all"')
    result = _run(cfg, out)
    reg = pd.read_parquet(out / "predict" / "a_pred.parquet")
    assert reg["true"].isna().any()  # the masked row is present with true=NaN
    if "a" in result["metrics"]:
        # metric sample count excludes the NaN row
        assert result["metrics"]["a"]["samples"] == int(reg["true"].notna().sum())


def test_no_metrics_skips_metric_artifacts(data_dir, tmp_path) -> None:
    ckpt = tmp_path / "ck.pt"
    _checkpoint(data_dir, ckpt)
    out = tmp_path / "pred"
    cfg = _predict_cfg(data_dir, out, ckpt, predict_block='tasks = ["a"]\nsplit = "all"\nwith_metrics = false')
    result = _run(cfg, out)
    assert (out / "predict" / "a_pred.parquet").exists()  # predictions still written
    assert not (out / "predict" / "metrics.json").exists()  # ...but no metrics artifacts
    assert not (out / "predict" / "metrics_table.csv").exists()
    assert result["metrics"] == {}


def test_scaler_inverse_transform_applied(data_dir, tmp_path) -> None:
    ckpt = tmp_path / "ck.pt"
    _checkpoint(data_dir, ckpt)
    scaler = StandardScaler().fit(np.array([[100.0], [150.0], [200.0], [250.0]]))  # physical-scale scaler
    scaler_path = tmp_path / "scalers.z"
    joblib.dump({"a": scaler}, scaler_path)
    scaler_block = f'scaler = {{ path = "{scaler_path}", key = "a" }}'
    out = tmp_path / "pred"
    cfg = _predict_cfg(data_dir, out, ckpt, predict_block='tasks = ["a"]\nsplit = "all"', scaler=scaler_block)
    _run(cfg, out)
    reg = pd.read_parquet(out / "predict" / "a_pred.parquet")
    # inverse-transformed predictions land on the physical (~100-250) scale, not the model's ~[-1,1].
    assert np.nanmedian(np.abs(reg["pred"])) > 10.0
