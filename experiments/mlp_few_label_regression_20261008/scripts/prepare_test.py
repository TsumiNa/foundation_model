"""Exact label counts, nested subsets, and fixed holdout validation."""

import numpy as np
import pandas as pd
import pytest

from prepare import COUNTS, TARGETS, mask_nested


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
