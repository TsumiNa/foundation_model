# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest

from process_agis_learning_curve import nested_splits


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
