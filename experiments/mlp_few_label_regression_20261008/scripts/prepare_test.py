"""Exact label counts, nested subsets, and fixed holdout validation."""

import copy
import json
import random

import numpy as np
import pandas as pd
import pytest

import prepare
from prepare import COUNTS, TARGETS, mask_nested, validate_selection


def frame() -> pd.DataFrame:
    d = pd.DataFrame({"split": ["train"] * 120 + ["val"] * 5 + ["test"] * 5})
    d.index = pd.Index([f"compound_{i}" for i in range(len(d))])
    for col in TARGETS.values():
        d[col] = np.arange(len(d), dtype=float)
    return d


def test_exact_nested_counts_and_unchanged_holdout():
    original = frame()
    outputs, selected = mask_nested(original, 0)
    for task, col in TARGETS.items():
        for n in COUNTS:
            out = outputs[n]
            actual = set(out.index[(out.split == "train") & out[col].notna()])
            assert actual == set(selected[task][:n])
            assert len(actual) == n
            pd.testing.assert_series_equal(
                out.loc[out.split != "train", col], original.loc[original.split != "train", col]
            )
    again, _ = mask_nested(original, 0)
    pd.testing.assert_frame_equal(outputs[50], again[50])
    different, _ = mask_nested(original, 1)
    assert not outputs[10].equals(different[10])
    assert original.notna().all().all()


def test_invalid_inputs():
    d = frame()
    with pytest.raises(ValueError, match="Insufficient"):
        mask_nested(d.iloc[:60], 0)
    with pytest.raises(ValueError, match="Missing"):
        mask_nested(d.drop(columns=[next(iter(TARGETS.values()))]), 0)
    d.index = ["alias"] * len(d)
    with pytest.raises(ValueError, match="unique"):
        mask_nested(d, 0)


def test_frozen_draw_rejects_substitution_duplicates_and_population_edits(tmp_path, monkeypatch):
    models = [{"run": f"m{i:03d}", "sha256": str(i)} for i in range(240)]
    path = tmp_path / "population.json"
    path.write_text(json.dumps({"models": models, "n_models": 240, "library": prepare.LIBRARY}))
    monkeypatch.setattr(prepare, "POPULATION_SHA256", prepare.sha256(path))
    selection = {
        "sampling_seed": 20261001,
        "source_library": prepare.LIBRARY,
        "population": 240,
        "selection_method": "Uniform random sample without replacement; no AGIS evaluation used",
        "models": random.Random(20261001).sample(models, 10),
    }
    validate_selection(selection, path)
    for replacement in (models[0], selection["models"][0]):
        edited = copy.deepcopy(selection)
        edited["models"][1] = replacement
        with pytest.raises(ValueError, match="random draw"):
            validate_selection(edited, path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        validate_selection(selection, path)
