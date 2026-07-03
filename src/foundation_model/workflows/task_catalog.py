# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""TOML-driven task catalog: datasets, tasks, descriptors, scalers and datamodules.

This module replaces every hardcoded task registry in the repository (the old
``TASK_SPECS`` dicts and ``SuiteConfig`` descriptor/scaler fields). A parsed-TOML tree with
``[data]`` / ``[descriptor]`` / ``[datasets.*]`` / ``[[tasks]]`` sections is normalized into a
:class:`TaskCatalogConfig` by :func:`build_task_catalog_config`; :class:`TaskCatalog` then loads
the composition-keyed frames and descriptors once and hands out task configs and datamodules.

Scope boundary (intentional): the catalog loads **model-ready** target columns. Dataset-specific
*forward* preprocessing that used to be inlined in the legacy scripts — ``log1p`` + z-scoring of
raw NEMAD regression columns, and the 5-to-3 material-type label merge — is **not** reproduced
here; data files are expected to carry the target column already in the form the model trains on.
A per-task ``scaler`` provides *inverse*-transform for human-readable reporting only.

Kernel-regression heads are initialized from data with :func:`init_kernel_centers_sigmas`
(quantile centers + constant span), which is the initializer the pre-training path exercises
today. A histogram-density variant existed in ``multi_task_progressive_clf`` but is intentionally
not the default.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from loguru import logger

from foundation_model.data.composition_sources import (
    PrecomputedDescriptorSource,
    canonical_key,
    normalize_composition,
    read_data_file,
)
from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.model_config import (
    ClassificationTaskConfig,
    KernelRegressionTaskConfig,
    OptimizerConfig,
    RegressionTaskConfig,
)
from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS, KMD, element_features, formula_to_composition

TaskConfig = RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig

# A composition key element token, e.g. "Fe" / "Cu" in "Fe0.5 Cu0.5".
_ELEMENT_TOKEN = re.compile(r"[A-Z][a-z]?")


class TaskKind(str, Enum):
    """Supported task kinds (TOML-facing spellings)."""

    REGRESSION = "regression"
    KERNEL_REGRESSION = "kernel_regression"
    CLASSIFICATION = "classification"


# Legacy short spellings (``TASK_SPECS`` used "reg"/"kr"/"clf") mapped to the enum.
_KIND_ALIASES = {
    "reg": TaskKind.REGRESSION,
    "regression": TaskKind.REGRESSION,
    "kr": TaskKind.KERNEL_REGRESSION,
    "kernel_regression": TaskKind.KERNEL_REGRESSION,
    "clf": TaskKind.CLASSIFICATION,
    "classification": TaskKind.CLASSIFICATION,
}

# Full-name suffix patterns accepted by ``data.composition_sources.read_data_file`` (kept in
# sync with it). ``.parquet`` also covers ``.pd.parquet``; multi-suffix pickle forms need the
# full ending, so we match on ``path.name`` rather than ``Path.suffix`` (which sees only ".z").
_SUPPORTED_DATA_PATTERNS = (".csv", ".parquet", ".pd", ".pd.z", ".pd.xz", ".pkl")


def _coerce_task_kind(value: Any) -> TaskKind:
    if isinstance(value, TaskKind):
        return value
    try:
        return _KIND_ALIASES[str(value).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown task kind {value!r}; expected one of {sorted({k.value for k in TaskKind})}."
        ) from exc


def _validate_replay(value: float | int | None, *, where: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):  # bool is an int subclass — reject it explicitly
        raise ValueError(f"{where}: replay must be a number, got bool {value!r}.")
    if isinstance(value, float):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{where}: replay float must be in (0, 1) (fraction of labels), got {value}.")
    elif isinstance(value, int):
        if value < 1:
            raise ValueError(f"{where}: replay int must be >= 1 (label count), got {value}.")
    else:
        raise ValueError(f"{where}: replay must be a float in (0, 1) or an int >= 1, got {value!r}.")


@dataclass(kw_only=True)
class ScalerSpec:
    """Points at a fitted scaler used to inverse-transform a task's predictions for reporting."""

    path: Path
    key: str | None = None  # key inside a dict-of-scalers pickle; None = the whole object

    def __post_init__(self) -> None:
        self.path = Path(self.path)


@dataclass(kw_only=True)
class DatasetSpec:
    """One composition-keyed data file plus optional filters."""

    name: str
    path: Path
    preprocessing_path: Path | None = None  # joblib dict with "dropped_idx" (qc dataset)
    min_elements: int | None = None  # keep only compositions with >= this many elements
    sample: int | None = None  # row cap for smoke runs

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.name.lower().endswith(_SUPPORTED_DATA_PATTERNS):
            raise ValueError(
                f"Dataset '{self.name}': unsupported data file '{self.path.name}'; "
                f"expected a name ending in one of {_SUPPORTED_DATA_PATTERNS} (see read_data_file)."
            )
        if self.preprocessing_path is not None:
            self.preprocessing_path = Path(self.preprocessing_path)
        if self.min_elements is not None and self.min_elements < 1:
            raise ValueError(f"Dataset '{self.name}': min_elements must be >= 1, got {self.min_elements}.")
        if self.sample is not None and self.sample < 1:
            raise ValueError(f"Dataset '{self.name}': sample must be >= 1, got {self.sample}.")


@dataclass(kw_only=True)
class TaskSpec:
    """One supervised task: a column in a dataset, plus kind-conditional metadata."""

    name: str
    kind: TaskKind  # accepts str/legacy alias; normalized in __post_init__
    dataset: str  # key into TaskCatalogConfig.datasets
    column: str
    t_column: str | None = None  # required iff kind == KERNEL_REGRESSION
    num_classes: int | None = None  # required iff kind == CLASSIFICATION
    lr: float | None = None  # per-task LR override
    replay: float | int | None = None  # per-task rehearsal override (fraction or count)
    scaler: ScalerSpec | None = None

    def __post_init__(self) -> None:
        self.kind = _coerce_task_kind(self.kind)
        if self.kind is TaskKind.KERNEL_REGRESSION and not self.t_column:
            raise ValueError(f"Task '{self.name}': kernel_regression requires 't_column'.")
        if self.kind is not TaskKind.KERNEL_REGRESSION and self.t_column:
            raise ValueError(f"Task '{self.name}': 't_column' is only valid for kernel_regression.")
        if self.kind is TaskKind.CLASSIFICATION:
            if self.num_classes is None:
                raise ValueError(f"Task '{self.name}': classification requires 'num_classes'.")
            if self.num_classes < 2:
                raise ValueError(f"Task '{self.name}': num_classes must be >= 2, got {self.num_classes}.")
        elif self.num_classes is not None:
            raise ValueError(f"Task '{self.name}': 'num_classes' is only valid for classification.")
        _validate_replay(self.replay, where=f"Task '{self.name}'")


@dataclass(kw_only=True)
class DescriptorConfig:
    """How composition descriptors are produced."""

    kind: str = "kmd"  # "kmd" (on-the-fly, invertible) | "precomputed"
    n_grids: int = 8  # kmd only
    path: Path | None = None  # precomputed only; required then

    def __post_init__(self) -> None:
        if self.kind not in {"kmd", "precomputed"}:
            raise ValueError(f"descriptor.kind must be 'kmd' or 'precomputed', got {self.kind!r}.")
        if self.kind == "kmd" and self.n_grids < 2:
            # KMD(method="1d") requires n_grids >= 2; surface it at config time, not at load.
            raise ValueError(f'descriptor.n_grids must be >= 2 when kind="kmd", got {self.n_grids}.')
        if self.kind == "precomputed":
            if self.path is None:
                raise ValueError("descriptor.kind == 'precomputed' requires 'path'.")
            self.path = Path(self.path)


@dataclass(kw_only=True)
class DataConfig:
    """Shared data-loading knobs (consumed by every subcommand)."""

    composition_column: str = "composition"
    val_split: float = 0.1
    test_split: float = 0.1
    split_random_seed: int = 42
    batch_size: int = 256
    num_workers: int = 0

    def __post_init__(self) -> None:
        for name in ("val_split", "test_split"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"data.{name} must be in [0, 1), got {value}.")
        if self.val_split + self.test_split >= 1.0:
            raise ValueError("data.val_split + data.test_split must be < 1.0.")
        if self.batch_size < 1:
            raise ValueError(f"data.batch_size must be >= 1, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"data.num_workers must be >= 0, got {self.num_workers}.")


@dataclass(kw_only=True)
class TaskCatalogConfig:
    """The fully-resolved catalog: data knobs, descriptor, datasets and tasks."""

    data: DataConfig
    descriptor: DescriptorConfig
    datasets: dict[str, DatasetSpec]
    tasks: list[TaskSpec]

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("At least one task is required ([[tasks]]).")
        names = [t.name for t in self.tasks]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"Duplicate task names: {dupes}.")
        for task in self.tasks:
            if task.dataset not in self.datasets:
                raise ValueError(
                    f"Task '{task.name}': dataset '{task.dataset}' is not defined in [datasets] "
                    f"(available: {sorted(self.datasets)})."
                )


# --- TOML → dataclass builder -------------------------------------------------------------


def _reject_unknown(section: str, raw: Mapping[str, Any], known: set[str]) -> None:
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"[{section}]: unknown key(s) {unknown}; allowed keys are {sorted(known)}.")


def _build_scaler_spec(raw: Mapping[str, Any], *, task_name: str) -> ScalerSpec:
    _reject_unknown(f"tasks.{task_name}.scaler", raw, {"path", "key"})
    if "path" not in raw:
        raise ValueError(f"Task '{task_name}': scaler requires 'path'.")
    return ScalerSpec(path=Path(raw["path"]), key=raw.get("key"))


def build_task_catalog_config(raw: Mapping[str, Any]) -> TaskCatalogConfig:
    """Normalize the parsed-TOML tree into a :class:`TaskCatalogConfig`.

    Unknown keys raise ``ValueError`` naming the offending key (an explicit per-section check —
    the dataclasses alone would surface a less friendly ``TypeError``).
    """

    _reject_unknown("<root>", raw, {"data", "descriptor", "datasets", "tasks"})

    data_raw = dict(raw.get("data", {}))
    _reject_unknown("data", data_raw, set(DataConfig.__dataclass_fields__))
    data = DataConfig(**data_raw)

    descriptor_raw = dict(raw.get("descriptor", {}))
    _reject_unknown("descriptor", descriptor_raw, set(DescriptorConfig.__dataclass_fields__))
    descriptor = DescriptorConfig(**descriptor_raw)

    datasets_raw = raw.get("datasets", {})
    if not isinstance(datasets_raw, Mapping) or not datasets_raw:
        raise ValueError("At least one [datasets.<name>] table is required.")
    datasets: dict[str, DatasetSpec] = {}
    for name, spec_raw in datasets_raw.items():
        spec_map = dict(spec_raw)
        _reject_unknown(f"datasets.{name}", spec_map, set(DatasetSpec.__dataclass_fields__) - {"name"})
        datasets[name] = DatasetSpec(name=name, **spec_map)

    tasks_raw = raw.get("tasks", [])
    if not tasks_raw:
        raise ValueError("At least one [[tasks]] entry is required.")
    tasks: list[TaskSpec] = []
    for task_raw in tasks_raw:
        task_map = dict(task_raw)
        _reject_unknown(
            f"tasks.{task_map.get('name', '?')}",
            task_map,
            set(TaskSpec.__dataclass_fields__),
        )
        scaler_raw = task_map.pop("scaler", None)
        scaler = _build_scaler_spec(scaler_raw, task_name=task_map.get("name", "?")) if scaler_raw is not None else None
        tasks.append(TaskSpec(scaler=scaler, **task_map))

    return TaskCatalogConfig(data=data, descriptor=descriptor, datasets=datasets, tasks=tasks)


# --- kernel init-from-data ----------------------------------------------------------------


def init_kernel_centers_sigmas(t_values: np.ndarray, n_kernel: int) -> tuple[list[float], list[float]]:
    """Quantile kernel centers + constant span from a task's observed ``t`` distribution.

    Migrated from ``continual_rehearsal_demo._init_kernels`` — the initializer the pre-training
    path exercises today. Returns empty lists when no finite ``t`` values are available (callers
    fall back to the head's equal-spaced default).
    """

    t = np.asarray(t_values, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return [], []
    centers = np.quantile(t, np.linspace(0.0, 1.0, n_kernel))
    span = max((float(t.max()) - float(t.min())) / max(n_kernel - 1, 1), 1e-3)
    return centers.tolist(), np.full(n_kernel, span).tolist()


def _count_elements(composition_key: str) -> int:
    return len(_ELEMENT_TOKEN.findall(composition_key))


def _as_float_array(cell: Any) -> np.ndarray:
    import ast

    if isinstance(cell, str):
        cell = ast.literal_eval(cell)
    return np.asarray(cell, dtype=float).ravel()


# --- catalog ------------------------------------------------------------------------------


class TaskCatalog:
    """Loads composition-keyed frames + descriptors once; hands out task configs and datamodules."""

    def __init__(self, config: TaskCatalogConfig) -> None:
        self.config = config
        self._task_specs = {t.name: t for t in config.tasks}

        self._kmd: KMD | None = None
        self._descriptor_source: PrecomputedDescriptorSource | None = None
        self._desc_cache: dict[str, np.ndarray] = {}
        if config.descriptor.kind == "kmd":
            self._kmd = KMD(
                element_features.values,
                method="1d",
                n_grids=config.descriptor.n_grids,
                sigma="auto",
                scale=True,
            )
            self._descriptor_dim = int(self._kmd.transform(np.eye(1, len(DEFAULT_ELEMENTS))).shape[1])
        else:
            assert config.descriptor.path is not None  # guaranteed by DescriptorConfig.__post_init__
            self._descriptor_source = PrecomputedDescriptorSource(
                str(config.descriptor.path), composition_column=config.data.composition_column
            )
            self._descriptor_dim = 0  # resolved lazily on first descriptor_fn call

        self._dataset_frames: dict[str, pd.DataFrame] = {}  # keyed, filtered source frames (cached)
        self._task_frames: dict[str, pd.DataFrame] = {}  # per-task composition-indexed frames (cached)
        self._scalers: dict[str, Any] = {}  # loaded lazily per task

    # -- introspection

    @property
    def task_names(self) -> list[str]:
        return list(self._task_specs)

    @property
    def descriptor_dim(self) -> int:
        """Descriptor feature dimension (encoder input dim)."""
        if self._descriptor_dim == 0 and self._descriptor_source is not None:
            # Force one lookup to learn the precomputed width. Use any task's first composition.
            frame = self._descriptor_source._load()  # noqa: SLF001 — read-only width probe
            self._descriptor_dim = int(frame.shape[1])
        return self._descriptor_dim

    def task_spec(self, name: str) -> TaskSpec:
        try:
            return self._task_specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown task '{name}'; known tasks: {self.task_names}.") from exc

    def kmd(self) -> KMD | None:
        """The invertible KMD object, or ``None`` when descriptors are precomputed."""
        return self._kmd

    # -- data loading

    def _load_dataset(self, name: str) -> pd.DataFrame:
        """Load one dataset file, apply preprocessing/min_elements/sample, key by composition."""

        if name in self._dataset_frames:
            return self._dataset_frames[name]
        spec = self.config.datasets[name]
        df = read_data_file(str(spec.path)).copy()

        # Drop rows recorded in a preprocessing cache (e.g. qc dropped_idx).
        if spec.preprocessing_path is not None and spec.preprocessing_path.exists():
            dropped = joblib.load(spec.preprocessing_path).get("dropped_idx", [])
            df = df.loc[~df.index.isin(dropped)]

        # Composition key → index, keeping first duplicate. Use canonical_key (the same rule the
        # DataModule / descriptor sources use): real formulas are canonicalized while non-formula
        # IDs (e.g. precomputed-descriptor material IDs) pass through unchanged instead of being
        # dropped. Only genuinely-missing (NaN/None) compositions are dropped.
        comp_col = self.config.data.composition_column
        raw = df[comp_col] if comp_col in df.columns else df.index.to_series()
        keys = [canonical_key(v, normalize_composition) if pd.notna(v) else None for v in raw]
        df["__key__"] = np.asarray(keys, dtype=object)
        df = df.dropna(subset=["__key__"]).drop_duplicates(subset="__key__", keep="first").set_index("__key__")

        if spec.min_elements is not None:
            df = df.loc[[_count_elements(str(k)) >= spec.min_elements for k in df.index]]

        if spec.sample is not None and spec.sample < len(df):
            rng = np.random.default_rng(self.config.data.split_random_seed)
            df = df.iloc[np.sort(rng.choice(len(df), size=spec.sample, replace=False))]

        self._dataset_frames[name] = df
        logger.info(f"Loaded dataset '{name}': {len(df)} rows keyed by composition.")
        return df

    def task_frames(self, names: Sequence[str]) -> dict[str, pd.DataFrame]:
        """Composition-keyed per-task frames with ``[column(+t_column)[, split]]``."""

        out: dict[str, pd.DataFrame] = {}
        for name in names:
            if name not in self._task_frames:
                spec = self.task_spec(name)
                source = self._load_dataset(spec.dataset)
                if spec.column not in source.columns:
                    raise KeyError(f"Task '{name}': column '{spec.column}' missing in dataset '{spec.dataset}'.")
                frame = pd.DataFrame(index=source.index)
                frame[spec.column] = source[spec.column]
                if spec.kind is TaskKind.KERNEL_REGRESSION:
                    assert spec.t_column is not None
                    if spec.t_column not in source.columns:
                        raise KeyError(
                            f"Task '{name}': t_column '{spec.t_column}' missing in dataset '{spec.dataset}'."
                        )
                    frame[spec.t_column] = source[spec.t_column]
                if "split" in source.columns:
                    frame["split"] = source["split"]
                self._task_frames[name] = frame
            out[name] = self._task_frames[name]
        return out

    def descriptor_fn(self) -> Callable[[list[str]], pd.DataFrame]:
        """Return a ``Callable[[list[str]], pd.DataFrame]`` producing composition descriptors."""

        if self._descriptor_source is not None:
            source = self._descriptor_source
            return lambda compositions: source(list(compositions))

        def _kmd_descriptor(compositions: list[str]) -> pd.DataFrame:
            uncached = [c for c in dict.fromkeys(compositions) if c not in self._desc_cache]
            if uncached:
                weights = np.zeros((len(uncached), len(DEFAULT_ELEMENTS)), dtype=float)
                valid: list[str] = []
                for key in uncached:
                    try:
                        w = formula_to_composition(key)
                    except Exception:
                        w = None
                    if w is None or float(np.asarray(w).sum()) <= 0:
                        continue
                    weights[len(valid)] = w
                    valid.append(key)
                if valid:
                    assert self._kmd is not None
                    desc = self._kmd.transform(weights[: len(valid)])
                    for j, key in enumerate(valid):
                        self._desc_cache[key] = desc[j]
            present = [c for c in compositions if c in self._desc_cache]
            if not present:
                return pd.DataFrame()
            return pd.DataFrame(np.stack([self._desc_cache[c] for c in present]), index=present)

        return _kmd_descriptor

    # -- task configs

    def _train_t_values(self, name: str) -> np.ndarray:
        spec = self.task_spec(name)
        assert spec.t_column is not None
        frame = self.task_frames([name])[name]
        mask = frame[spec.column].notna()
        if "split" in frame.columns:
            mask &= frame["split"] == "train"
        cells = frame.loc[mask, spec.t_column].dropna()
        if cells.empty:
            return np.array([])
        return np.concatenate([_as_float_array(c) for c in cells])

    def _class_weights(self, name: str) -> list[float]:
        spec = self.task_spec(name)
        assert spec.num_classes is not None
        frame = self.task_frames([name])[name]
        mask = frame[spec.column].notna()
        if "split" in frame.columns:
            mask &= frame["split"] == "train"
        labels = frame.loc[mask, spec.column].dropna().astype(int)
        counts = np.bincount(labels, minlength=spec.num_classes).astype(float)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (spec.num_classes * counts)  # sklearn "balanced" scheme
        return weights.tolist()

    def build_task_config(
        self,
        name: str,
        *,
        latent_dim: int,
        head_hidden_dim: int,
        n_kernel: int,
        lr: float,
        weight_decay: float = 0.0,
        masking_ratio: float = 1.0,
        init_from_data: bool = True,
    ) -> TaskConfig:
        """Build the model-side task config for ``name`` (see module docstring for field mapping)."""

        spec = self.task_spec(name)
        head_lr = spec.lr if spec.lr is not None else lr
        optimizer = OptimizerConfig(lr=head_lr, weight_decay=weight_decay)

        if spec.kind is TaskKind.REGRESSION:
            return RegressionTaskConfig(
                name=name,
                data_column=spec.column,
                dims=[latent_dim, head_hidden_dim, 1],
                optimizer=optimizer,
                task_masking_ratio=masking_ratio,
            )
        if spec.kind is TaskKind.CLASSIFICATION:
            assert spec.num_classes is not None
            return ClassificationTaskConfig(
                name=name,
                data_column=spec.column,
                dims=[latent_dim, head_hidden_dim, 32],
                num_classes=spec.num_classes,
                class_weights=self._class_weights(name),
                optimizer=optimizer,
                task_masking_ratio=masking_ratio,
            )

        assert spec.t_column is not None
        centers: list[float] = []
        sigmas: list[float] = []
        if init_from_data:
            centers, sigmas = init_kernel_centers_sigmas(self._train_t_values(name), n_kernel)
        return KernelRegressionTaskConfig(
            name=name,
            data_column=spec.column,
            t_column=spec.t_column,
            x_dim=[latent_dim, 128, 64],
            t_dim=[16, 8],
            kernel_num_centers=n_kernel,
            kernel_centers_init=centers or None,
            kernel_sigmas_init=sigmas or None,
            kernel_learnable_centers=True,
            kernel_learnable_sigmas=True,
            enable_mu3=False,
            optimizer=optimizer,
            task_masking_ratio=masking_ratio,
        )

    # -- scalers / inverse-transform

    def scaler(self, name: str) -> Any | None:
        """The fitted scaler for ``name`` (loaded lazily), or ``None`` if not configured."""

        spec = self.task_spec(name)
        if spec.scaler is None:
            return None
        if name not in self._scalers:
            obj = joblib.load(spec.scaler.path)
            if spec.scaler.key is not None:
                if spec.scaler.key not in obj:
                    raise KeyError(f"Task '{name}': scaler key '{spec.scaler.key}' not found in {spec.scaler.path}.")
                obj = obj[spec.scaler.key]
            self._scalers[name] = obj
        return self._scalers[name]

    def inverse_transform(self, name: str, values: np.ndarray) -> np.ndarray:
        """Apply the task's scaler inverse-transform if configured, else identity."""

        scaler = self.scaler(name)
        if scaler is None:
            return np.asarray(values)
        restored = scaler.inverse_transform(np.asarray(values, dtype=float).reshape(-1, 1))
        return np.asarray(restored).reshape(-1)

    # -- datamodule

    def _data_task_config(self, name: str, *, masking_ratio: float | None, predict_idx: Any) -> TaskConfig:
        """A lightweight task config carrying only the fields the datamodule reads."""

        spec = self.task_spec(name)
        kwargs: dict[str, Any] = {
            "name": name,
            "data_column": spec.column,
            "task_masking_ratio": masking_ratio,
            "predict_idx": predict_idx,
        }
        if spec.kind is TaskKind.REGRESSION:
            return RegressionTaskConfig(**kwargs)
        if spec.kind is TaskKind.CLASSIFICATION:
            assert spec.num_classes is not None
            return ClassificationTaskConfig(num_classes=spec.num_classes, **kwargs)
        assert spec.t_column is not None
        return KernelRegressionTaskConfig(t_column=spec.t_column, **kwargs)

    def build_datamodule(
        self,
        active_tasks: Sequence[str],
        *,
        masking_ratios: Mapping[str, float] | None = None,
        predict_idx: str | Sequence[str] | None = None,
    ) -> CompoundDataModule:
        """Assemble a :class:`CompoundDataModule` over ``active_tasks``.

        ``masking_ratios`` overrides each task's keep-ratio (default 1.0); ``predict_idx`` is set
        on every active task config (literal split name or explicit composition sequence).
        """

        masking_ratios = masking_ratios or {}
        frames = self.task_frames(active_tasks)
        task_configs = [
            self._data_task_config(name, masking_ratio=masking_ratios.get(name, 1.0), predict_idx=predict_idx)
            for name in active_tasks
        ]
        data = self.config.data
        return CompoundDataModule(
            task_configs=task_configs,
            descriptor_fn=self.descriptor_fn(),
            task_frames={name: frames[name] for name in active_tasks},
            composition_column=data.composition_column,
            random_seed=data.split_random_seed,
            val_split=data.val_split,
            test_split=data.test_split,
            batch_size=data.batch_size,
            num_workers=data.num_workers,
        )
