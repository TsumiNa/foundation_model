# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Composition-level multi-task train/val/test splitter."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class MultiTaskSplitter:
    """Split compositions into train/val/test, prioritizing rare tasks for representation.

    Operates on a composition-indexed **availability matrix** (rows = compositions, columns =
    tasks, truthy where the task has data for that composition). Compositions are grouped into
    chunks rarest-task-first (then a leftover chunk for compositions available to no task), and
    the holdout (val/test) budget is allocated across chunks with a **cumulative** rounding
    scheme so global proportions are preserved — many tiny chunks no longer round individually
    to zero and collapse val/test. Processing rare tasks first gives them first claim on that
    budget, which *improves* their representation in val/test.

    Note: this does not *guarantee* every task appears in every split — a task with very few
    compositions, or small ratios, may still round to zero for that task (the global val/test
    proportions are preserved regardless). Used as the random fallback in
    :func:`foundation_model.data.composition_sources.resolve_splits` for compositions that
    carry no explicit ``split`` label.

    Parameters
    ----------
    val_ratio : float, default=0.1
        Overall fraction of compositions assigned to validation.
    test_ratio : float, default=0.1
        Overall fraction of compositions assigned to test.
    random_state : int | None, optional
        Seed for deterministic shuffling.
    """

    def __init__(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: Optional[int] = None,
    ):
        if val_ratio < 0.0 or test_ratio < 0.0:
            raise ValueError("val_ratio and test_ratio must be non-negative.")
        if val_ratio + test_ratio > 1.0:
            raise ValueError(f"val_ratio + test_ratio must not exceed 1.0 (got {val_ratio} + {test_ratio}).")
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def _chunks(self, availability: pd.DataFrame, index: List[str]) -> List[List[str]]:
        """Group compositions rarest-task-first, then a leftover chunk for the rest."""
        assigned: set[str] = set()
        chunks: List[List[str]] = []
        if availability.shape[1] > 0:
            mask = availability.astype(bool)
            mask.index = pd.Index(index)
            counts = mask.sum(axis=0).sort_values()  # rarest task first
            for task in counts.index:
                column = mask[task]
                avail = [comp for comp in index if column.loc[comp] and comp not in assigned]
                if avail:
                    chunks.append(avail)
                    assigned.update(avail)
        leftover = [comp for comp in index if comp not in assigned]
        if leftover:
            chunks.append(leftover)
        return chunks

    def split(self, availability: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """Split the availability matrix into train/val/test composition lists.

        Holdout counts are accumulated across chunks (cumulative rounding) so global val/test
        proportions are preserved even when chunks are tiny.

        Parameters
        ----------
        availability : pd.DataFrame
            Indexed by composition; each column is a task with truthy values where the task
            has data for that composition. May have zero columns (then every composition is
            split proportionally).

        Returns
        -------
        Tuple[List[str], List[str], List[str]]
            ``(train, val, test)`` composition keys (each composition appears exactly once).
        """
        rng = np.random.default_rng(self.random_state)
        index = [str(c) for c in availability.index]
        train: List[str] = []
        val: List[str] = []
        test: List[str] = []

        # Running float targets + already-allocated integer counts implement cumulative
        # (largest-remainder) rounding, preventing many small chunks from each rounding to 0.
        cum_test = 0.0
        cum_val = 0.0
        alloc_test = 0
        alloc_val = 0
        for chunk in self._chunks(availability, index):
            n = len(chunk)
            order = rng.permutation(n)
            shuffled = [chunk[i] for i in order]

            cum_test += n * self.test_ratio
            cum_val += n * self.val_ratio
            n_test = max(0, min(int(round(cum_test)) - alloc_test, n))
            n_val = max(0, min(int(round(cum_val)) - alloc_val, n - n_test))
            alloc_test += n_test
            alloc_val += n_val

            test.extend(shuffled[:n_test])
            val.extend(shuffled[n_test : n_test + n_val])
            train.extend(shuffled[n_test + n_val :])
        return train, val, test
