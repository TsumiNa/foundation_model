# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Composition-level multi-task train/val/test splitter."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class MultiTaskSplitter:
    """Split compositions into train/val/test while keeping every task represented.

    Operates on a composition-indexed **availability matrix** (rows = compositions, columns =
    tasks, truthy where the task has data for that composition). Tasks are allocated rarest
    first, so a scarce task secures representation in the validation and test splits before
    common tasks consume the shared pool. Compositions available to no task are split
    proportionally at the end.

    Used as the random fallback in
    :func:`foundation_model.data.composition_sources.resolve_splits` for compositions that
    carry no explicit ``split`` label.

    Parameters
    ----------
    val_ratio : float, default=0.1
        Fraction of each task's compositions assigned to validation.
    test_ratio : float, default=0.1
        Fraction of each task's compositions assigned to test.
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

    def _allocate(
        self,
        compositions: List[str],
        rng: np.random.Generator,
        train: List[str],
        val: List[str],
        test: List[str],
    ) -> None:
        """Shuffle ``compositions`` and append proportional shares to train/val/test."""
        n = len(compositions)
        if n == 0:
            return
        order = rng.permutation(n)
        shuffled = [compositions[i] for i in order]
        n_test = min(int(round(n * self.test_ratio)), n)
        n_val = min(int(round(n * self.val_ratio)), n - n_test)
        test.extend(shuffled[:n_test])
        val.extend(shuffled[n_test : n_test + n_val])
        train.extend(shuffled[n_test + n_val :])

    def split(self, availability: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """Split the availability matrix into train/val/test composition lists.

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
        assigned: set[str] = set()

        if availability.shape[1] > 0:
            mask = availability.astype(bool)
            mask.index = pd.Index(index)
            # Rarest tasks first so scarce tasks secure val/test representation.
            counts = mask.sum(axis=0).sort_values()
            for task in counts.index:
                column = mask[task]
                avail = [comp for comp in index if column.loc[comp] and comp not in assigned]
                self._allocate(avail, rng, train, val, test)
                assigned.update(avail)

        leftover = [comp for comp in index if comp not in assigned]
        self._allocate(leftover, rng, train, val, test)
        return train, val, test
