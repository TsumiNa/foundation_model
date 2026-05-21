# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the composition-level MultiTaskSplitter (refactor PR4)."""

import pandas as pd
import pytest

from foundation_model.data.splitter import MultiTaskSplitter


def _availability(rows: dict[str, list[bool]]) -> pd.DataFrame:
    """Build a composition-indexed availability matrix from {task: per-composition flags}."""
    index = [f"c{i}" for i in range(len(next(iter(rows.values()))))]
    return pd.DataFrame(rows, index=pd.Index(index, name="composition"))


def test_rejects_negative_ratio():
    with pytest.raises(ValueError, match="non-negative"):
        MultiTaskSplitter(val_ratio=-0.1, test_ratio=0.1)


def test_rejects_ratio_sum_above_one():
    with pytest.raises(ValueError, match="must not exceed 1.0"):
        MultiTaskSplitter(val_ratio=0.6, test_ratio=0.6)


def test_partition_is_complete_and_disjoint():
    avail = _availability({"t1": [True] * 10})
    train, val, test = MultiTaskSplitter(val_ratio=0.2, test_ratio=0.2, random_state=0).split(avail)
    combined = sorted(train + val + test)
    assert combined == sorted(avail.index)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0


def test_proportional_counts_single_task():
    avail = _availability({"t1": [True] * 10})
    train, val, test = MultiTaskSplitter(val_ratio=0.2, test_ratio=0.2, random_state=42).split(avail)
    assert len(test) == 2
    assert len(val) == 2
    assert len(train) == 6


def test_deterministic_with_seed():
    avail = _availability({"t1": [True] * 12})
    a = MultiTaskSplitter(val_ratio=0.25, test_ratio=0.25, random_state=7).split(avail)
    b = MultiTaskSplitter(val_ratio=0.25, test_ratio=0.25, random_state=7).split(avail)
    assert a == b


def test_rare_task_is_represented_in_each_split():
    """A scarce task (allocated first) should land in val and test, not only train."""
    # t_rare has 6 compositions; t_common covers all 60. With 0.2/0.2 the rare task alone
    # must contribute to val and test.
    rare = [True] * 6 + [False] * 54
    common = [True] * 60
    avail = _availability({"t_common": common, "t_rare": rare})
    splitter = MultiTaskSplitter(val_ratio=0.2, test_ratio=0.2, random_state=1)
    train, val, test = splitter.split(avail)

    rare_comps = {c for c, flag in zip(avail.index, rare) if flag}
    assert rare_comps & set(val), "rare task should appear in validation"
    assert rare_comps & set(test), "rare task should appear in test"
    # Whole set still partitioned exactly once.
    assert sorted(train + val + test) == sorted(avail.index)


def test_many_size_one_chunks_preserve_global_holdouts():
    """Disjoint single-composition tasks must not collapse all holdouts to train (PR12 P1)."""
    # 20 tasks, each owning exactly one distinct composition -> 20 size-1 chunks.
    n = 20
    rows = {}
    for i in range(n):
        flags = [False] * n
        flags[i] = True
        rows[f"t{i}"] = flags
    avail = _availability(rows)
    train, val, test = MultiTaskSplitter(val_ratio=0.1, test_ratio=0.1, random_state=0).split(avail)
    # Per-chunk independent rounding would give 0 here; cumulative allocation preserves ~10%.
    assert len(test) == 2
    assert len(val) == 2
    assert len(train) == 16
    assert sorted(train + val + test) == sorted(avail.index)


def test_no_columns_splits_proportionally():
    avail = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(10)], name="composition"))
    train, val, test = MultiTaskSplitter(val_ratio=0.1, test_ratio=0.1, random_state=0).split(avail)
    assert len(test) == 1
    assert len(val) == 1
    assert len(train) == 8


def test_all_train_when_ratios_zero():
    avail = _availability({"t1": [True] * 5})
    train, val, test = MultiTaskSplitter(val_ratio=0.0, test_ratio=0.0, random_state=0).split(avail)
    assert sorted(train) == sorted(avail.index)
    assert val == []
    assert test == []
