# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""AGIS curve preparation, fold isolation and prediction-unit restoration."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.signal import savgol_filter

from foundation_model.models.model_config import KernelRegressionTaskConfig
from foundation_model.workflows._engine import build_empty_model, build_head_config
from foundation_model.workflows._sections import ModelSectionConfig, TrainingSectionConfig
from foundation_model.workflows.predict import PredictConfig, run as predict_run
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

from process_agis_data import AGISPreprocessingConfig, load_agis_curves, prepare_agis, standardize_fold

_FORMULAS = [
    "La3Ni2O7",
    "La2.9Sr0.1Ni2O7",
    "La2NdNi2O7",
    "La1.9NdSr0.1Ni2O7",
    "La1.8NdSr0.2Ni2O7",
    "La2EuNi2O7",
    "La1.9EuSr0.1Ni2O7",
    "La1.8EuSr0.2Ni2O7",
]
_CONFIG = AGISPreprocessingConfig(temperature_min_K=6, temperature_max_K=10, n_points=5, date_suffix="20261001")


def _write_curve(path: Path, formula: str, pressure: int, *, offset: float = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rho_column = "rho (muOhm cm)" if pressure == 0 else "rho_xx (muOhm cm)"
    # Unsorted temperatures and a duplicate at 7 K; the duplicate's mean is exactly 2+offset.
    pairs = [(12, 9), (5, -1), (6, 0), (7, 1), (7, 3), (9, 5), (10, 7)]
    text = (
        f"CAP-rho data file: {path.name}\n"
        f"sample material: MKR-TEST - {formula}\n"
        f"p = {pressure} GPa\n\n"
        f"T (K)\tB (T)\t{rho_column}\n"
    )
    text += "\n".join(f"{t}\t0\t{y + offset}" for t, y in pairs) + "\n"
    path.write_text(text)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    for i, formula in enumerate(_FORMULAS, 1):
        for pressure in (0, 10, 20):
            _write_curve(root / f"{i} {formula}" / f"rho_{pressure}.dat", formula, pressure, offset=i + pressure)
    _write_curve(root / f"1 {_FORMULAS[0]}" / "rho_15.dat", _FORMULAS[0], 15)
    return root


def test_read_sort_duplicates_units_and_pressure_selection(raw_dir: Path) -> None:
    curves = load_agis_curves(raw_dir, _CONFIG)
    assert len(curves) == 24
    assert set(curves.pressure_GPa) == {0, 10, 20}
    row = curves.iloc[0]
    np.testing.assert_allclose(row.temperature_K, [6, 7, 8, 9, 10])
    np.testing.assert_allclose(row.rho_uohm_cm, [1, 3, 4.5, 6, 8])
    assert row.n_raw == 7 and row.n_unique_temperature == 6
    assert len(row.source_sha256) == 64
    assert "MKR-TEST" in row.header_json


def test_one_training_compound_excludes_all_seven_tests_from_scalers(raw_dir: Path) -> None:
    curves = load_agis_curves(raw_dir, _CONFIG)
    heldout = curves.composition.drop_duplicates().to_list()[1:]
    fold, scalers = standardize_fold(curves, heldout)
    assert fold[fold.split == "train"].composition.nunique() == 1
    changed = curves.copy(deep=True)
    for idx in changed.index[changed.composition.isin(heldout)]:
        changed.at[idx, "rho_uohm_cm"] = (np.asarray(changed.at[idx, "rho_uohm_cm"]) * 1e6).tolist()
    _, other = standardize_fold(changed, heldout)
    for pressure in (0, 10, 20):
        key = f"agis_rho_{pressure}gpa_scaler"
        assert scalers[key]["prescale"].n_samples_seen_ == _CONFIG.n_points
        for step in ("prescale", "standardscaler"):
            np.testing.assert_array_equal(scalers[key][step].scale_, other[key][step].scale_)
        data = fold[fold.pressure_GPa == pressure]
        restored = scalers[key].inverse_transform(np.concatenate(data.rho_normalized.to_list()).reshape(-1, 1))
        np.testing.assert_allclose(restored.ravel(), np.concatenate(data.rho_uohm_cm.to_list()), rtol=1e-11)


@pytest.mark.parametrize("heldout", [[], ["unknown"], ["La3 Ni2 O7"] * 2])
def test_invalid_multiple_holdouts(raw_dir: Path, heldout: list[str]) -> None:
    with pytest.raises(ValueError, match="held-out"):
        standardize_fold(load_agis_curves(raw_dir, _CONFIG), heldout)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature_min_K": -1},
        {"temperature_min_K": float("nan")},
        {"temperature_max_K": float("inf")},
        {"temperature_min_K": 10, "temperature_max_K": 6},
        {"n_points": 1},
        {"n_points": 5.5},
    ],
)
def test_invalid_config(kwargs: dict[str, float | str]) -> None:
    with pytest.raises(ValueError):
        AGISPreprocessingConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("window", [0, -1, float("nan"), float("inf"), 1, 1000])
def test_invalid_smoothing_window(window: float) -> None:
    with pytest.raises(ValueError):
        AGISPreprocessingConfig(smooth_window_K=window)


def test_training_smoothing_preserves_raw_holdout_and_fits_only_smoothed_train(raw_dir: Path, tmp_path: Path) -> None:
    config = AGISPreprocessingConfig(
        temperature_min_K=6, temperature_max_K=10, n_points=5, date_suffix="20261001", smooth_window_K=5
    )
    curves = load_agis_curves(raw_dir, config)
    originals = [np.asarray(v).copy() for v in curves.rho_uohm_cm]
    heldout = curves.composition.iloc[0]
    fold, scalers = standardize_fold(curves, heldout, config)
    for row, original in zip(fold.itertuples(), originals, strict=True):
        np.testing.assert_array_equal(row.unsmoothed_rho_uohm_cm, original)
        expected = original if row.split == "test" else savgol_filter(original, 5, 2, mode="interp")
        np.testing.assert_array_equal(row.rho_uohm_cm, expected)
        assert row.smoothing_applied == (row.split == "train")
    for current, original in zip(curves.rho_uohm_cm, originals, strict=True):
        np.testing.assert_array_equal(current, original)
    modified = curves.copy(deep=True)
    for idx in modified.index[modified.composition == heldout]:
        modified.at[idx, "rho_uohm_cm"] = (np.asarray(modified.at[idx, "rho_uohm_cm"]) * 1000).tolist()
    _, other = standardize_fold(modified, heldout, config)
    for pressure in (0, 10, 20):
        scaler = scalers[f"agis_rho_{pressure}gpa_scaler"]
        values = np.concatenate(
            fold.loc[(fold.pressure_GPa == pressure) & (fold.split == "train"), "rho_uohm_cm"].to_list()
        )
        np.testing.assert_allclose(scaler["prescale"].scale_, [values.std()], rtol=1e-12)
        np.testing.assert_array_equal(
            scaler["standardscaler"].mean_, other[f"agis_rho_{pressure}gpa_scaler"]["standardscaler"].mean_
        )
        assert scaler["prescale"].n_samples_seen_ == 35
    manifest_path = prepare_agis(raw_dir, tmp_path, config)
    manifest = json.loads(manifest_path.read_text())
    assert "agis_preprocessing_smoothed_20261001" in str(manifest_path)
    assert manifest["smoothing"]["window_points"] == 5
    np.testing.assert_array_equal(pd.read_parquet(manifest["curves_path"]).rho_uohm_cm.iloc[0], originals[0])


def test_smoothing_rejects_wrong_grid_or_nonfinite_curves(raw_dir: Path) -> None:
    config = AGISPreprocessingConfig(temperature_min_K=6, temperature_max_K=10, n_points=5, smooth_window_K=5)
    curves = load_agis_curves(raw_dir, config)
    malformed = curves.copy(deep=True)
    malformed.at[0, "temperature_K"] = [6, 7, 8, 9, 11]
    with pytest.raises(ValueError, match="uniform"):
        standardize_fold(malformed, curves.composition.iloc[0], config)
    malformed = curves.copy(deep=True)
    malformed.at[0, "rho_uohm_cm"] = [1, 2, float("nan"), 4, 5]
    with pytest.raises(ValueError, match="finite resistivity"):
        standardize_fold(malformed, curves.composition.iloc[0], config)


@pytest.mark.parametrize("date", ["2026-10-01", "20260230", "../20261001", "00000000"])
def test_invalid_date_suffix_rejected(date: str) -> None:
    with pytest.raises(ValueError):
        AGISPreprocessingConfig(date_suffix=date)


def test_nonfinite_points_and_negative_values_are_handled(raw_dir: Path) -> None:
    path = raw_dir / "1 La3Ni2O7" / "rho_0.dat"
    _write_curve(path, "La3Ni2O7", 0, offset=-2)
    with path.open("a") as f:
        f.write("8\t0\tnan\nnan\t0\t1\n")
    curves = load_agis_curves(raw_dir, _CONFIG)
    row = curves.iloc[0]
    assert row.n_nonfinite_dropped == 2
    assert row.rho_uohm_cm[0] == -2
    assert np.isfinite(row.rho_uohm_cm).all()


@pytest.mark.parametrize("failure", ["missing_pressure", "duplicate", "header", "units", "coverage", "formula"])
def test_malformed_input_rejected(raw_dir: Path, failure: str) -> None:
    path = raw_dir / "1 La3Ni2O7" / "rho_0.dat"
    message = ""
    if failure == "missing_pressure":
        path.unlink()
        message = "Missing pressures"
    elif failure == "duplicate":
        (path.parent / "duplicate.dat").write_text(path.read_text())
        message = "Duplicate"
    elif failure == "header":
        path.write_text(path.read_text().replace("T (K)", "Temperature"))
        message = "T \\(K\\)"
    elif failure == "units":
        path.write_text(path.read_text().replace("muOhm cm", "Ohm m"))
        message = "resistivity column"
    elif failure == "coverage":
        message = "does not cover"
    else:
        path.write_text(path.read_text().replace("MKR-TEST - La3Ni2O7", "MKR-TEST - La2EuNi2O7"))
        message = "composition"
    config = AGISPreprocessingConfig(temperature_max_K=290) if failure == "coverage" else _CONFIG
    with pytest.raises(ValueError, match=message):
        load_agis_curves(raw_dir, config)


def test_empty_or_missing_input_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_agis_curves(tmp_path / "missing", _CONFIG)
    with pytest.raises(ValueError, match="No AGIS"):
        load_agis_curves(tmp_path, _CONFIG)


def test_entire_compound_missing_rejected(raw_dir: Path) -> None:
    for path in (raw_dir / "8 La1.8EuSr0.2Ni2O7").glob("*.dat"):
        path.unlink()
    with pytest.raises(ValueError, match="all eight AGIS compounds"):
        load_agis_curves(raw_dir, _CONFIG)


def test_fold_scalers_ignore_heldout_all_pressures_and_restore_values(raw_dir: Path) -> None:
    curves = load_agis_curves(raw_dir, _CONFIG)
    heldout = curves.composition.iloc[0]
    fold, scalers = standardize_fold(curves, heldout)
    modified = curves.copy(deep=True)
    for idx in modified.index[modified.composition == heldout]:
        modified.at[idx, "rho_uohm_cm"] = np.asarray(modified.at[idx, "rho_uohm_cm"]) * 1000
    other_fold, other_scalers = standardize_fold(modified, heldout)
    assert set(fold.loc[fold.split == "test", "composition"]) == {heldout}
    assert len(fold.loc[fold.split == "test"]) == 3
    for pressure in (0, 10, 20):
        key = f"agis_rho_{pressure}gpa_scaler"
        scaler, other_scaler = scalers[key], other_scalers[key]
        for step in ("prescale", "standardscaler"):
            np.testing.assert_array_equal(scaler[step].scale_, other_scaler[step].scale_)
            np.testing.assert_array_equal(scaler[step].mean_, other_scaler[step].mean_)
            assert scaler[step].n_samples_seen_ == 35
        group = fold[fold.pressure_GPa == pressure]
        train = np.concatenate(group.loc[group.split == "train", "rho_normalized"].to_list())
        np.testing.assert_allclose(train.mean(), 0, atol=1e-14)
        np.testing.assert_allclose(train.std(), 1, atol=1e-14)
        for row in group.itertuples():
            restored = scaler.inverse_transform(np.asarray(row.rho_normalized).reshape(-1, 1)).ravel()
            np.testing.assert_allclose(restored, np.asarray(row.rho_uohm_cm, dtype=float), rtol=1e-12, atol=1e-12)
        # Domain extrapolation must be invertible too: unlike a negative-lambda Yeo-Johnson fit.
        restored = scaler.inverse_transform(np.array([[-10.0], [10.0]]))
        assert np.isfinite(restored).all()
    np.testing.assert_array_equal(
        np.concatenate(fold.loc[fold.split == "train", "rho_normalized"].to_list()),
        np.concatenate(other_fold.loc[other_fold.split == "train", "rho_normalized"].to_list()),
    )


def test_constant_zero_training_values_and_unknown_fold(raw_dir: Path) -> None:
    curves = load_agis_curves(raw_dir, _CONFIG)
    for idx in curves.index:
        curves.at[idx, "rho_uohm_cm"] = np.zeros(_CONFIG.n_points)
    fold, scalers = standardize_fold(curves, curves.composition.iloc[0])
    assert not np.concatenate(fold.rho_normalized.to_list()).any()
    for scaler in scalers.values():
        np.testing.assert_allclose(scaler.inverse_transform([[0]]), [[0]])
    with pytest.raises(ValueError, match="held-out"):
        standardize_fold(curves, "LaNiO3")


def test_export_and_existing_prediction_inverse(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    manifest_path = prepare_agis(raw_dir, out, _CONFIG)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_curves"] == 24 and len(manifest["folds"]) == 8
    assert pd.read_parquet(out / "agis_resistivity_20261001.pd.parquet").shape[0] == 24
    fold_dir = out / "agis_preprocessing_20261001" / "fold_01"
    raw = tomllib.loads((fold_dir / "tasks_20261001.toml").read_text())
    raw["descriptor"] = {"kind": "kmd", "n_grids": 4}
    catalog_config = build_task_catalog_config(raw)
    catalog = TaskCatalog(catalog_config)
    model_config = ModelSectionConfig(latent_dim=8, encoder_hidden_dims=[16], n_kernel=4)
    training = TrainingSectionConfig(max_epochs=1, accelerator="cpu")
    model = build_empty_model(catalog, model_config, training)
    for name in catalog.task_names:
        config = build_head_config(catalog, model_config, training, name)
        assert isinstance(config, KernelRegressionTaskConfig)
        model.add_task(config)
        for param in model.task_heads[name].parameters():
            param.data.zero_()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "task_sequence": catalog.task_names}, checkpoint)
    predict_out = tmp_path / "predict"
    predict_run(
        PredictConfig(
            catalog=catalog_config,
            model=model_config,
            checkpoint=checkpoint,
            output_dir=predict_out,
            accelerator="cpu",
            with_metrics=False,
        )
    )
    scalers = joblib.load(fold_dir / "scalers_20261001.joblib")
    for pressure in (0, 10, 20):
        name = f"agis_rho_{pressure}gpa"
        data = pd.read_parquet(fold_dir / f"agis_resistivity_fold01_{pressure}gpa_20261001.pd.parquet")
        test_row = data[data.split == "test"].iloc[0]
        prediction = pd.read_parquet(predict_out / "predict" / f"{name}_pred.parquet")
        np.testing.assert_allclose(prediction.true, test_row.rho_uohm_cm, rtol=1e-12, atol=1e-12)
        expected = scalers[f"{name}_scaler"].inverse_transform([[0.0]])[0, 0]
        np.testing.assert_allclose(prediction.pred, expected, rtol=1e-6)
        np.testing.assert_array_equal(prediction.t, test_row.temperature_K)
    with pytest.raises(FileExistsError):
        prepare_agis(raw_dir, out, _CONFIG)
