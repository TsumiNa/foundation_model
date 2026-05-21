# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Composition-keyed data-source helpers (refactor PR2).

Pure building blocks used by the composition-keyed ``CompoundDataModule`` (wired in PR3):

* :func:`read_data_file` / :func:`load_task_frame` — load a task's own file(s) and index
  them by a composition column.
* :func:`build_composition_universe` — the union of compositions across tasks (plus any
  predict-only compositions).
* :class:`DescriptorCache` — apply a user descriptor function once per unique composition.
* :func:`resolve_splits` — derive a single composition-level train/val/test split by
  overlaying per-task ``split`` columns and randomly assigning the rest.

These functions are intentionally side-effect free (apart from logging) so they can be unit
tested in isolation with in-memory frames and a fake descriptor function.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from loguru import logger

DescriptorFn = Callable[[list[str]], pd.DataFrame]

# Split labels, ordered by precedence (later wins on conflict).
_SPLIT_PRECEDENCE: dict[str, int] = {"train": 1, "val": 2, "test": 3}
VALID_SPLIT_LABELS = frozenset(_SPLIT_PRECEDENCE)


def read_data_file(path: str) -> pd.DataFrame:
    """Load a single data file into a DataFrame based on its extension.

    Supported: ``.csv``, ``.parquet`` / ``.pd.parquet``, pickled frames
    (``.pd`` / ``.pd.z`` / ``.pd.xz``), and ``.pkl`` (joblib).

    .. warning::
        The pickle-based formats (``.pd`` / ``.pd.z`` / ``.pd.xz`` via ``pd.read_pickle``
        and ``.pkl`` via ``joblib.load``) deserialize arbitrary Python objects and can
        execute arbitrary code. Only load these from sources you trust; prefer ``.csv`` or
        ``.parquet`` for untrusted inputs.

    Parameters
    ----------
    path : str
        Path to the data file.

    Returns
    -------
    pd.DataFrame
        The loaded frame (index untouched; caller sets the composition index).

    Raises
    ------
    ValueError
        If the file extension is not recognized or the loaded object is not a DataFrame.
    """
    lower = path.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(path)
    elif lower.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif lower.endswith((".pd", ".pd.z", ".pd.xz")):
        df = pd.read_pickle(path)
    elif lower.endswith(".pkl"):
        df = joblib.load(path)
    else:
        raise ValueError(
            f"Unsupported file type: {path}. Must be .csv, .parquet/.pd.parquet, .pd/.pd.z/.pd.xz, or .pkl."
        )
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"File {path} did not load into a pandas DataFrame (got {type(df).__name__}).")
    return df


def load_task_frame(
    data_files: Sequence[str],
    composition_column: str,
    *,
    task_name: str = "",
) -> pd.DataFrame:
    """Load and concatenate a task's data file(s), indexed by composition.

    Multiple files are concatenated by rows. Duplicate compositions are dropped keeping the
    first occurrence (with a warning). The composition column becomes a string index and is
    removed from the columns.

    Parameters
    ----------
    data_files : Sequence[str]
        One or more paths to the task's data file(s).
    composition_column : str
        Name of the column holding composition keys.
    task_name : str, optional
        Task name used only for log messages.

    Returns
    -------
    pd.DataFrame
        Frame indexed by composition (string), columns are the remaining data columns.

    Raises
    ------
    ValueError
        If ``data_files`` is empty or the composition column is missing from a file.
    """
    label = task_name or "<task>"
    if not data_files:
        raise ValueError(f"Task '{label}': data_files is empty; cannot load a task frame.")

    frames: list[pd.DataFrame] = []
    for path in data_files:
        df = read_data_file(path)
        if composition_column in df.columns:
            df = df.set_index(composition_column)
        elif df.index.name == composition_column:
            pass  # already indexed by the composition column
        else:
            raise ValueError(
                f"Task '{label}': composition column '{composition_column}' not found in '{path}' "
                f"as a column or index name (available columns: {list(df.columns)}, "
                f"index name: {df.index.name!r})."
            )
        frames.append(df)

    combined = pd.concat(frames, axis=0) if len(frames) > 1 else frames[0]
    combined.index = combined.index.astype(str)
    combined.index.name = composition_column

    duplicated = combined.index.duplicated(keep="first")
    if duplicated.any():
        logger.warning(
            f"Task '{label}': {int(duplicated.sum())} duplicate composition(s) across data_files; "
            "keeping the first occurrence of each."
        )
        combined = combined[~duplicated]

    return combined


def build_composition_universe(
    task_frames: Mapping[str, pd.DataFrame],
    extra_compositions: Sequence[str] | None = None,
) -> list[str]:
    """Return the order-preserving union of all task compositions plus any extras.

    Parameters
    ----------
    task_frames : Mapping[str, pd.DataFrame]
        Per-task frames indexed by composition (as produced by :func:`load_task_frame`).
    extra_compositions : Sequence[str] | None, optional
        Additional compositions to include (e.g. predict-only compositions) so descriptors
        are computed for them even when they carry no task target.

    Returns
    -------
    list[str]
        Unique compositions, preserving first-seen order across tasks then extras.
    """
    universe: dict[str, None] = {}
    for frame in task_frames.values():
        for comp in frame.index.astype(str):
            universe.setdefault(comp, None)
    if extra_compositions is not None:
        for comp in extra_compositions:
            universe.setdefault(str(comp), None)
    return list(universe)


class PrecomputedDescriptorSource:
    """A ``descriptor_fn`` backed by a precomputed, composition-indexed descriptor file.

    This is a YAML/CLI-friendly callable (it can be instantiated via ``class_path`` /
    ``init_args``) for the common case where descriptors are already computed and stored. The
    file is loaded once on first call and rows are looked up by composition; compositions absent
    from the file are simply omitted from the result (and reported as dropped by the DataModule).

    Parameters
    ----------
    path : str
        Path to a composition-indexed descriptor file (see :func:`read_data_file` for formats).
    composition_column : str | None, optional
        If given and present as a column, it is set as the index; otherwise the file's existing
        index is treated as the composition key.
    """

    def __init__(self, path: str, composition_column: str | None = None):
        self.path = path
        self.composition_column = composition_column
        self._frame: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._frame is None:
            frame = read_data_file(self.path)
            if self.composition_column is not None and self.composition_column in frame.columns:
                frame = frame.set_index(self.composition_column)
            frame = frame.copy()
            frame.index = frame.index.astype(str)
            self._frame = frame
        return self._frame

    def __call__(self, compositions: Sequence[str]) -> pd.DataFrame:
        frame = self._load()
        present = [str(c) for c in compositions if str(c) in frame.index]
        return frame.loc[present]


class DescriptorCache:
    """Apply a user descriptor function once per unique composition.

    The descriptor function maps a list of composition strings to a DataFrame indexed by
    composition. Results are cached; subsequent calls only compute compositions not seen
    before. Compositions the function fails to produce (or returns as all-NaN rows) are
    reported as "dropped" by :meth:`resolve`.

    Parameters
    ----------
    descriptor_fn : Callable[[list[str]], pd.DataFrame]
        Function returning a composition-indexed descriptor frame.
    """

    def __init__(self, descriptor_fn: DescriptorFn):
        self._descriptor_fn = descriptor_fn
        self._cache: pd.DataFrame | None = None
        self._attempted: set[str] = set()

    def compute(self, compositions: Sequence[str]) -> pd.DataFrame:
        """Return cached descriptors for ``compositions``, computing any not yet attempted.

        The returned frame contains only the requested compositions the function has
        produced, in the requested order.
        """
        requested = [str(c) for c in compositions]
        uncached = [c for c in dict.fromkeys(requested) if c not in self._attempted]
        if uncached:
            produced = self._descriptor_fn(uncached)
            if not isinstance(produced, pd.DataFrame):
                raise TypeError(f"descriptor_fn must return a pandas DataFrame, got {type(produced).__name__}.")
            produced = produced.copy()
            produced.index = produced.index.astype(str)
            self._attempted.update(uncached)
            if not produced.empty:
                self._cache = produced if self._cache is None else pd.concat([self._cache, produced], axis=0)
                # Keep the first descriptor row for any composition seen more than once.
                self._cache = self._cache[~self._cache.index.duplicated(keep="first")]

        if self._cache is None:
            return pd.DataFrame()
        present = [c for c in requested if c in self._cache.index]
        return self._cache.loc[present]

    def resolve(self, compositions: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
        """Compute descriptors and split the request into valid frame + dropped list.

        A composition is "dropped" when the descriptor function did not produce it or
        produced an all-NaN row.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            ``(valid_descriptors, dropped_compositions)`` where ``valid_descriptors`` is
            indexed by composition in the requested order.
        """
        requested = [str(c) for c in compositions]
        computed = self.compute(requested)
        if computed.empty:
            return computed, list(dict.fromkeys(requested))

        all_nan = computed.isna().all(axis=1)
        valid_index = computed.index[~all_nan]
        valid_set = set(valid_index)
        valid = [c for c in requested if c in valid_set]
        dropped = [c for c in dict.fromkeys(requested) if c not in valid_set]
        return computed.loc[valid], dropped


def _random_split(
    compositions: Sequence[str],
    *,
    val_split: float,
    test_split: float,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Assign train/val/test labels to ``compositions`` by proportional random split."""
    comps = list(compositions)
    n = len(comps)
    if n == 0:
        return {}
    order = rng.permutation(n)
    shuffled = [comps[i] for i in order]
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    n_test = min(n_test, n)
    n_val = min(n_val, n - n_test)
    labels: dict[str, str] = {}
    for comp in shuffled[:n_test]:
        labels[comp] = "test"
    for comp in shuffled[n_test : n_test + n_val]:
        labels[comp] = "val"
    for comp in shuffled[n_test + n_val :]:
        labels[comp] = "train"
    return labels


def resolve_splits(
    task_frames: Mapping[str, pd.DataFrame],
    master_index: Sequence[str],
    split_columns: Mapping[str, str],
    *,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int | None = 42,
    test_all: bool = False,
) -> pd.Series:
    """Derive a single composition-level train/val/test split.

    Per-task ``split`` columns are overlaid (precedence ``test > val > train``; conflicts are
    warned). Compositions with no label from any task fall back to a proportional random
    split. With ``test_all=True`` every composition is assigned to ``test``.

    Parameters
    ----------
    task_frames : Mapping[str, pd.DataFrame]
        Per-task frames indexed by composition.
    master_index : Sequence[str]
        The compositions to label (typically those with valid descriptors).
    split_columns : Mapping[str, str]
        Per-task name of the split column to read (tasks absent from the map, or whose
        column is missing from their frame, contribute no labels).
    val_split, test_split : float
        Random-fallback proportions for unlabeled compositions.
    random_seed : int | None
        Seed for the random fallback.
    test_all : bool
        If True, assign every composition to ``test``.

    Returns
    -------
    pd.Series
        Index = ``master_index``, values in ``{"train", "val", "test"}``.

    Raises
    ------
    ValueError
        If a split column contains a value outside ``{"train", "val", "test"}``.
    """
    master = [str(c) for c in master_index]
    if test_all:
        return pd.Series(["test"] * len(master), index=pd.Index(master, name="composition"), dtype=object)

    labels: dict[str, str] = {}
    for task_name, frame in task_frames.items():
        col = split_columns.get(task_name)
        if not col or col not in frame.columns:
            continue
        series = frame[col].dropna()
        for comp, raw in series.items():
            comp = str(comp)
            value = str(raw)
            if value not in VALID_SPLIT_LABELS:
                raise ValueError(
                    f"Task '{task_name}': invalid value '{value}' in split column '{col}'. "
                    f"Must be one of {sorted(VALID_SPLIT_LABELS)}."
                )
            existing = labels.get(comp)
            if existing is None:
                labels[comp] = value
            elif existing != value:
                logger.warning(
                    f"Composition '{comp}' has conflicting split labels ('{existing}' vs '{value}'); "
                    f"keeping higher precedence."
                )
                if _SPLIT_PRECEDENCE[value] > _SPLIT_PRECEDENCE[existing]:
                    labels[comp] = value

    unlabeled = [comp for comp in master if comp not in labels]
    if unlabeled:
        rng = np.random.default_rng(random_seed)
        labels.update(_random_split(unlabeled, val_split=val_split, test_split=test_split, rng=rng))

    resolved = [labels[comp] for comp in master]
    return pd.Series(resolved, index=pd.Index(master, name="composition"), dtype=object)
