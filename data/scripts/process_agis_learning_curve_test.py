# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
import hashlib
import json
import tomllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from process_agis_learning_curve import LearningCurveSettings, nested_splits, prepare_learning_curve


def test_nested_splits_are_balanced_disjoint_and_keep_reference_test_material() -> None:
    compositions = [f"material{i}" for i in range(8)]
    splits = nested_splits(compositions, 20261001)
    assert splits == nested_splits(compositions, 20261001)
    for n_train, folds in splits.items():
        assert len(folds) == 8
        assert Counter(c for f in folds for c in f["fit_compositions"]) == dict.fromkeys(compositions, n_train)
        assert Counter(c for f in folds for c in f["test_compositions"]) == dict.fromkeys(compositions, 8 - n_train)
        assert len({tuple(sorted(f["fit_compositions"])) for f in folds}) == 8
        for index, fold in enumerate(folds):
            train, test = set(fold["fit_compositions"]), set(fold["test_compositions"])
            assert not train & test and train | test == set(compositions)
            assert fold["heldout_composition"] == compositions[index] and compositions[index] in test
            if n_train < 7:
                assert train < set(splits[n_train + 1][index]["fit_compositions"])


@pytest.mark.parametrize("compositions", [[], ["a"] * 8, [str(i) for i in range(7)]])
def test_nested_splits_reject_invalid_compound_universe(compositions: list[str]) -> None:
    with pytest.raises(ValueError, match="eight distinct"):
        nested_splits(compositions, 42)


def test_prepare_writes_dated_pressure_datasets_and_training_only_scalers(tmp_path: Path) -> None:
    grid = np.linspace(6, 290, 300)
    compositions = [f"material{i}" for i in range(8)]
    rows = [
        {
            "composition": c,
            "formula": c,
            "pressure_GPa": p,
            "temperature_K": grid,
            "rho_uohm_cm": grid * (i + 1) + p,
            "raw_temperature_K": grid,
            "raw_rho_uohm_cm": grid,
        }
        for i, c in enumerate(compositions)
        for p in (0, 10, 20)
    ]
    curves = tmp_path / "curves.parquet"
    pd.DataFrame(rows).to_parquet(curves)
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "n_compounds": 8,
                "n_curves": 24,
                "curves_path": str(curves),
                "config": {"temperature_min_K": 6, "temperature_max_K": 290, "n_points": 300},
                "pressures_GPa": [0, 10, 20],
                "units": {"temperature": "K", "resistivity": "muOhm cm"},
                "transform": "StandardScaler(with_mean=False) -> asinh -> StandardScaler",
                "folds": [{"heldout_composition": c} for c in compositions],
            }
        )
    )
    out = tmp_path / "data"
    summary_path = prepare_learning_curve(reference, out, LearningCurveSettings())
    summary = json.loads(summary_path.read_text())
    for n in range(1, 7):
        manifest = json.loads(Path(summary["manifests"][str(n)]).read_text())
        assert manifest["n_train"] == n and manifest["n_test"] == 8 - n
        assert manifest["reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
        for fold in manifest["folds"]:
            tasks = tomllib.loads((Path(fold["directory"]) / "tasks_20261001.toml").read_text())
            assert len(tasks["tasks"]) == 3
            for task in tasks["tasks"]:
                path = Path(tasks["datasets"][task["dataset"]]["path"])
                assert "20261001" in path.name and path.suffix == ".parquet"
                data = pd.read_parquet(path)
                train, test = data[data.split == "train"], data[data.split == "test"]
                assert len(train) == n and len(test) == 8 - n
                assert set(train.composition) == set(fold["fit_compositions"])
                assert set(test.composition) == set(fold["test_compositions"])
                scalers = joblib.load(task["scaler"]["path"])
                scaler = scalers[task["scaler"]["key"]]
                assert scaler["prescale"].n_samples_seen_ == n * 300
                values = np.concatenate(train.rho_uohm_cm.to_list())
                np.testing.assert_allclose(scaler["prescale"].scale_[0], values.std(), rtol=1e-12)
                restored = scaler.inverse_transform(np.concatenate(data.rho_normalized.to_list()).reshape(-1, 1))
                np.testing.assert_allclose(restored.ravel(), np.concatenate(data.rho_uohm_cm.to_list()), rtol=1e-11)
    with pytest.raises(FileExistsError):
        prepare_learning_curve(reference, out, LearningCurveSettings())
