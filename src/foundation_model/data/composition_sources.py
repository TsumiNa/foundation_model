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

from collections.abc import Mapping
from typing import Callable, Sequence

import joblib  # type: ignore[import-untyped]
import pandas as pd
from loguru import logger

from .splitter import MultiTaskSplitter

DescriptorFn = Callable[[list[str]], pd.DataFrame]

# Split labels, ordered by precedence (later wins on conflict).
_SPLIT_PRECEDENCE: dict[str, int] = {"train": 1, "val": 2, "test": 3}
VALID_SPLIT_LABELS = frozenset(_SPLIT_PRECEDENCE)

def normalize_composition(value: object) -> str | None:
    """Canonical composition key shared across every data source.

    Different files spell the same composition differently — a pymatgen ``Composition`` /
    element-amount ``dict`` (the qc dataset) versus a formula string (NEMAD / phonix), and
    ``"Fe3O2"`` versus ``"Fe3.0O2.0"``. Routing all of them through pymatgen and returning the
    (non-reduced) ``Composition.formula`` yields a single readable canonical string — pymatgen
    already normalizes element order and integer-vs-decimal amounts — so the composition-keyed
    DataModule joins heterogeneous sources by exact match. Compositions that pymatgen considers
    equal collapse to the same key, which is exactly the duplicate the DataModule then keeps once.

    The amounts are **not reduced** (``Fe2O3`` ≠ ``Fe4O6``) because some descriptors aggregate by
    sum rather than by mean, so the absolute stoichiometry must be preserved.

    Parameters
    ----------
    value : object
        A formula string, a pymatgen ``Composition``, or an element→amount mapping. Mapping
        entries that are ``None`` or non-positive are dropped (the qc ``composition`` column
        stores every element with mostly-``None`` amounts).

    Returns
    -------
    str | None
        e.g. ``"Fe2 O3"``; ``None`` if the input is empty or unparseable.
    """
    from pymatgen.core.composition import Composition  # local import; pymatgen is heavy

    try:
        if isinstance(value, dict):
            cleaned = {k: float(v) for k, v in value.items() if v is not None and float(v) > 0}
            if not cleaned:
                return None
            comp = Composition(cleaned)
        else:
            text = str(value).strip()
            # Fast-path: a real formula starts with an element symbol (uppercase) or an opening
            # bracket. This cheaply rejects non-formula join keys (``mp-1234`` IDs, synthetic
            # ``s0`` keys, SMILES like ``*CC*``) without a pymatgen parse, while still letting
            # charged/bracketed formulas through for pymatgen to decide.
            if not text or not (text[0].isupper() or text[0] in "(["):
                return None
            comp = Composition(text)
    except Exception:
        return None
    if len(comp) == 0:
        return None
    return comp.formula


CompositionNormalizer = Callable[[object], str | None]


def canonical_key(value: object, normalizer: CompositionNormalizer | None) -> str:
    """Apply ``normalizer`` to ``value``, falling back to ``str(value)`` when it can't parse.

    The fallback is what keeps normalization safe to enable by default: real formulas are
    canonicalized while non-formula keys (synthetic IDs, material IDs) pass through unchanged,
    and — crucially — both the task and descriptor sides use this same rule, so even unparseable
    keys still align.
    """
    if normalizer is None:
        return str(value)
    out = normalizer(value)
    return out if out is not None else str(value)


def lookup_descriptor_fn(
    features: pd.DataFrame,
    *,
    composition_normalizer: CompositionNormalizer | None = normalize_composition,
) -> DescriptorFn:
    """Build a ``descriptor_fn`` that looks up rows of an in-memory descriptor frame.

    The frame is re-indexed by :func:`canonical_key` so lookups match the canonical composition
    keys the DataModule passes (it normalizes task frames the same way). Rows whose keys collide
    after normalization — including pre-existing duplicate index labels — are collapsed keep-first
    (with a warning), guaranteeing a 1:1 key→row map so lookups never mismatch in length.

    Parameters
    ----------
    features : pd.DataFrame
        Descriptor matrix indexed by composition.
    composition_normalizer : Callable | None, optional
        Normalizer applied to the frame index. Defaults to :func:`normalize_composition`;
        pass ``None`` to look up by the raw (stringified) index instead.
    """
    frame = _reindex_by_canonical(features, composition_normalizer, source="lookup_descriptor_fn")

    index = frame.index

    def descriptor_fn(compositions: Sequence[str]) -> pd.DataFrame:
        # Keys are usually already canonical (the DataModule normalized them), so try a direct
        # hit first and only pay for canonicalization on a miss.
        present = [c if c in index else canonical_key(c, composition_normalizer) for c in compositions]
        return frame.loc[[k for k in present if k in index]]

    return descriptor_fn


def _reindex_by_canonical(
    frame: pd.DataFrame, normalizer: CompositionNormalizer | None, *, source: str
) -> pd.DataFrame:
    """Return ``frame`` re-indexed by canonical composition key, deduped keep-first (with warning).

    Uses a shallow copy (the descriptor matrix is shared, only the index is replaced) to avoid
    duplicating large descriptor frames.
    """
    canon = pd.Index([canonical_key(c, normalizer) for c in frame.index])
    out = frame.copy(deep=False)
    out.index = canon
    duplicated = canon.duplicated(keep="first")
    if duplicated.any():
        logger.warning(
            f"{source}: {int(duplicated.sum())} descriptor row(s) collapsed to a duplicate "
            "composition key after normalization; keeping the first occurrence of each."
        )
        out = out[~duplicated]
    return out


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
    composition_normalizer: CompositionNormalizer | None = None,
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
    composition_normalizer : Callable | None, optional
        When given, the composition index is mapped through :func:`canonical_key` (deduped after
        normalization). The DataModule passes its normalizer here; standalone callers default to
        ``None`` (raw string index).

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
    if composition_normalizer is not None:
        combined.index = pd.Index([canonical_key(c, composition_normalizer) for c in combined.index])
    else:
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
    composition_normalizer : Callable | None, optional
        Normalizer applied to the descriptor index via :func:`canonical_key` (deduped keep-first
        afterwards), so lookups match the canonical keys the DataModule passes. Defaults to
        :func:`normalize_composition`; pass ``None`` to look up by the raw string index.
    """

    def __init__(
        self,
        path: str,
        composition_column: str | None = None,
        *,
        composition_normalizer: CompositionNormalizer | None = normalize_composition,
    ):
        self.path = path
        self.composition_column = composition_column
        self._composition_normalizer = composition_normalizer
        self._frame: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._frame is None:
            frame = read_data_file(self.path)
            if self.composition_column is not None and self.composition_column in frame.columns:
                frame = frame.set_index(self.composition_column)
            frame = frame.copy()
            if self._composition_normalizer is not None:
                frame = _reindex_by_canonical(
                    frame, self._composition_normalizer, source=f"PrecomputedDescriptorSource('{self.path}')"
                )
            else:
                frame.index = frame.index.astype(str)
            self._frame = frame
        return self._frame

    def __call__(self, compositions: Sequence[str]) -> pd.DataFrame:
        frame = self._load()
        index = frame.index
        # Keys are usually already canonical (the DataModule normalized them); try a direct hit
        # first and only canonicalize on a miss, so we don't re-normalize every query at scale.
        present = [c if c in index else canonical_key(c, self._composition_normalizer) for c in compositions]
        return frame.loc[[k for k in present if k in index]]


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
    warned). Compositions with no label from any task fall back to a representation-aware
    random split (:class:`~foundation_model.data.splitter.MultiTaskSplitter`), which
    prioritizes rare tasks and preserves global val/test proportions. With ``test_all=True``
    every composition is assigned to ``test``.

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
        Random-fallback proportions for unlabeled compositions (overall val/test fractions,
        preserved globally by MultiTaskSplitter's cumulative allocation).
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
        # Build a composition-availability matrix and split it so each task keeps
        # representation across train/val/test (proportional for tasks/compositions with none).
        availability = pd.DataFrame(
            {task: [comp in frame.index for comp in unlabeled] for task, frame in task_frames.items()},
            index=pd.Index(unlabeled, name="composition"),
        )
        splitter = MultiTaskSplitter(val_ratio=val_split, test_ratio=test_split, random_state=random_seed)
        train_c, val_c, test_c = splitter.split(availability)
        for comp in train_c:
            labels[comp] = "train"
        for comp in val_c:
            labels[comp] = "val"
        for comp in test_c:
            labels[comp] = "test"

    resolved = [labels[comp] for comp in master]
    return pd.Series(resolved, index=pd.Index(master, name="composition"), dtype=object)
