"""Split leakage, reversible transforms and paired target-label budget checks."""

import numpy as np
import pandas as pd
import pytest

from process_transformer_transfer_data_20261002 import (
    PreparationConfig,
    TASK_COLUMNS,
    isolate_compositions,
    normalize_target,
    prepare,
    target_subset,
)


def test_atomic_fraction_aliases_cannot_leak_between_splits():
    raw = pd.DataFrame(
        {
            "composition": ["Fe2O3", "Fe4O6", "SiO2", "Si2O4", "AmO2", "invalid", "C"],
            "split": ["train", "test", "train", "val", "train", "val", "train"],
            "Volume": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    out, audit = isolate_compositions(raw)
    assert out["composition"].tolist() == ["Fe2O3", "SiO2", "C"]
    assert out["split"].tolist() == ["test", "val", "train"]
    assert out["Volume"].tolist() == [1, 3, 7]
    assert audit["invalid_compositions"] == 2
    assert audit["alias_records_removed"] == 2


def test_scalar_normalization_excludes_validation_and_test_and_inverts():
    raw = pd.DataFrame({"split": ["train", "train", "val", "test"], "Dielectric total": [2, 4, 1000, 2000]})
    out, scaler = normalize_target(raw, "dielectric_total")
    assert scaler.mean_[0] == 3 and scaler.scale_[0] == 1
    np.testing.assert_allclose(scaler.inverse_transform(out[["dielectric_total"]]), raw[["Dielectric total"]])


def test_curve_normalization_preserves_grid_and_rejects_malformed_curves():
    raw = pd.DataFrame(
        {
            "split": ["train", "train", "test", "val"],
            "Power factor": [np.array([1.0, 3.0]), np.array([3.0, 5.0]), np.array([999.0, 1001.0]), np.ones(3)],
            "Power factor (T/K)": [np.array([10.0, 20.0])] * 4,
        }
    )
    out, scaler = normalize_target(raw, "power_factor")
    assert scaler.mean_[0] == 3
    assert out["power_factor"].iloc[3] is None
    np.testing.assert_array_equal(out["power_factor_t"].iloc[2], [10, 20])
    np.testing.assert_allclose(scaler.inverse_transform(out["power_factor"].iloc[2][:, None]).ravel(), [999, 1001])


def test_target_budgets_are_nested_and_share_validation_test_compositions():
    raw = pd.DataFrame(
        {
            "composition": [f"id{i}" for i in range(25)],
            "split": ["train"] * 20 + ["val"] * 2 + ["test"] * 3,
            "Dielectric total": np.arange(25),
        }
    )
    low, high = target_subset(raw, "dielectric_total", 0.1, 1), target_subset(raw, "dielectric_total", 1, 1)
    assert len(low[low["split"].eq("train")]) == 2
    assert set(low["composition"]) < set(high["composition"])
    pd.testing.assert_frame_equal(low.loc[low["split"].ne("train")], high.loc[high["split"].ne("train")])
    assert target_subset(raw, "dielectric_total", 0.1, 1).equals(low)


def test_missing_training_targets_fail_before_scaler_fit():
    with pytest.raises(ValueError, match="no valid training"):
        normalize_target(pd.DataFrame({"split": ["test"], "Dielectric total": [2]}), "dielectric_total")


@pytest.mark.parametrize(
    "options", [{"seeds": (True,)}, {"seeds": (-1,)}, {"fractions": (True,)}, {"fractions": (0.101, 0.102)}]
)
def test_preparation_rejects_ambiguous_subset_ids(tmp_path, options):
    with pytest.raises(ValueError):
        PreparationConfig(input=tmp_path / "input", output=tmp_path / "output", **options)


def test_source_excludes_test_aliases_and_held_out_target_columns(tmp_path):
    raw = pd.DataFrame({"composition": ["FeO", "SiO2", "C", "Fe2O2"], "split": ["train", "train", "val", "test"]})
    # Another train composition keeps the source and target affine fit defined after alias exclusion.
    raw.loc[len(raw)] = ["NaCl", "train"]
    for name, (column, temperature) in TASK_COLUMNS.items():
        raw[column] = (
            [np.array([1.0, 2.0])] * len(raw)
            if temperature
            else ([4] * len(raw) if name == "material_type" else np.arange(len(raw), dtype=float))
        )
        if temperature:
            raw[temperature] = [np.array([10.0, 20.0])] * len(raw)
    path = tmp_path / "raw.parquet"
    raw.to_parquet(path)
    manifest = prepare(PreparationConfig(input=path, output=tmp_path / "prepared", seeds=(0,)))
    source = pd.read_parquet(tmp_path / "prepared" / manifest["source"]["file"])
    assert "FeO" not in set(source["composition"])
    assert "test" not in set(source["split"])
    assert "dielectric_total" not in source and "power_factor" not in source
    assert manifest["audit"]["alias_records_removed"] == 1
