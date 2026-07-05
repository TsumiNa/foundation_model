# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Mapping, Sequence

import lightning as L
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from foundation_model.models.model_config import (
    ClassificationTaskConfig,
    KernelRegressionTaskConfig,
    RegressionTaskConfig,
    TaskType,
)

from .composition_sources import (
    CompositionNormalizer,
    DescriptorCache,
    DescriptorFn,
    PrecomputedDescriptorSource,
    build_composition_universe,
    canonical_key,
    load_task_frame,
    normalize_composition,
    resolve_splits,
)
from .dataset import CompoundDataset

TaskConfig = RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig


class CollateFnWithTaskInfo:
    """
    Collate function that handles KernelRegression tasks properly.

    Implemented as a class to be pickle-serializable for multiprocessing.

    KernelRegression tasks need List[Tensor] format for both targets and t-parameters
    to support variable-length sequences without padding waste.

    Parameters
    ----------
    task_configs : List
        List of task configuration objects
    """

    def __init__(self, task_configs):
        self.kernel_regression_tasks = {
            cfg.name for cfg in task_configs if cfg.type == TaskType.KERNEL_REGRESSION and cfg.enabled
        }

    def __call__(self, batch):
        """
        Custom collate function for batching data.

        Parameters
        ----------
        batch : List[Tuple]
            List of (model_input_x, sample_y_dict, sample_task_masks_dict, sample_t_sequences_dict)

        Returns
        -------
        Tuple
            (batched_input, batched_y_dict, batched_mask_dict, batched_t_sequences_dict)
        """
        model_inputs, y_dicts, mask_dicts, t_sequences_dicts = zip(*batch)

        # Handle model inputs (formula features only)
        batched_input = torch.stack(model_inputs)

        # Handle targets and masks based on task type
        batched_y_dict = {}
        batched_mask_dict = {}

        for key in y_dicts[0].keys():
            if key in self.kernel_regression_tasks:
                # KernelRegression: Keep List[Tensor] format for variable-length sequences
                batched_y_dict[key] = [d[key] for d in y_dicts]
                batched_mask_dict[key] = [d[key] for d in mask_dicts]
            else:
                # Other tasks: Normal stacking
                batched_y_dict[key] = torch.stack([d[key] for d in y_dicts])
                batched_mask_dict[key] = torch.stack([d[key] for d in mask_dicts])

        # Handle sequence data (t-parameters) - always List[Tensor] format
        batched_t_sequences_dict = {}
        for key in t_sequences_dicts[0].keys():
            batched_t_sequences_dict[key] = [d[key] for d in t_sequences_dicts]

        return batched_input, batched_y_dict, batched_mask_dict, batched_t_sequences_dict


def create_collate_fn_with_task_info(task_configs):
    """
    Creates a custom collate function that handles KernelRegression tasks properly.

    Parameters
    ----------
    task_configs : List
        List of task configuration objects

    Returns
    -------
    CollateFnWithTaskInfo
        Custom collate function for DataLoader
    """
    return CollateFnWithTaskInfo(task_configs)


class CompoundDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for composition-keyed, per-task data sources.

    Each task owns its own data file(s) (or an in-memory frame), joined to the others by a
    **composition** column. Descriptors are computed on demand from the union of compositions
    via a user-supplied ``descriptor_fn`` (results cached per unique composition). Adding a
    new property task means adding one more file + one more task config — no monolithic
    attributes file to rebuild.

    Data assembly (performed once, lazily, in :meth:`setup`)
    -------------------------------------------------------
    1. Load each enabled non-AUTOENCODER task's data (in-memory ``task_frames`` override, else
       ``cfg.data_files``), indexed by composition.
    2. Build the composition universe = union of all task compositions plus any compositions
       named by explicit per-task ``predict_idx`` sequences (so predict-only compositions get
       descriptors too).
    3. Compute descriptors via ``descriptor_fn`` (cached); drop compositions with no valid
       descriptor. The survivors form the master composition index.
    4. Resolve a single composition-level train/val/test split by overlaying each task's
       ``split`` column (precedence ``test > val > train``) with a random fallback.
    5. Build per-stage :class:`CompoundDataset` instances; each reindexes the per-task frames
       to its split's compositions.

    Prediction
    ----------
    Each task's ``predict_idx`` selects a composition subset (literal ``train``/``val``/``test``/
    ``all``, an explicit sequence, or ``None`` → the test split, falling back to all). The
    predict set is the union of these subsets, exposed in order as
    :attr:`predict_compositions` so a prediction writer can attach composition keys.

    Parameters
    ----------
    task_configs : Sequence[TaskConfig]
        Task configurations. Per-task ``data_files`` / ``composition_column`` / ``split_column``
        / ``task_masking_ratio`` / ``predict_idx`` drive data loading.
    descriptor_fn : Callable[[list[str]], pd.DataFrame]
        Maps composition keys to a composition-indexed descriptor frame.
    task_frames : Mapping[str, pd.DataFrame] | None, optional
        In-memory per-task frames (indexed by composition or carrying the composition column),
        used instead of ``cfg.data_files`` for the named tasks. Convenient for programmatic /
        test usage.
    default_data_files : str | Sequence[str] | None, optional
        Shared fallback data file(s) used for any enabled task that has neither an in-memory
        frame nor its own ``cfg.data_files``. Supports the legacy "one file holds all targets"
        case from a YAML/CLI config without repeating the path on every task.
    composition_column : str, optional
        Global default composition column name; overridable per task via
        ``cfg.composition_column``. Defaults to ``"composition"``.
    composition_normalizer : Callable[[object], str | None] | None, optional
        Canonicalizes every composition key the DataModule ingests (task frames, ``data_files``,
        explicit ``predict_idx`` sequences) so heterogeneously-spelled sources — a pymatgen
        ``Composition``/dict vs a formula string, ``Fe3O2`` vs ``Fe3.0O2.0`` — join by exact
        match. Keys that don't parse fall back to their raw string (so synthetic/non-formula IDs
        are untouched). Defaults to :func:`~foundation_model.data.composition_sources.normalize_composition`;
        pass ``None`` to disable. **The descriptor side must use the same rule**: the bundled
        :class:`~foundation_model.data.composition_sources.PrecomputedDescriptorSource` and
        :func:`~foundation_model.data.composition_sources.lookup_descriptor_fn` normalize by
        default; a custom ``descriptor_fn`` must accept these canonical keys.
    random_seed : int | None, optional
        Base seed for all stochastic operations (split, masking, swap). Defaults to 42.
    val_split, test_split : float, optional
        Random-fallback proportions for compositions lacking a ``split`` label. Defaults 0.1.
    test_all : bool, optional
        If True, every composition is assigned to the test set. Defaults to False.
    swap_train_val_split : float, optional
        Fraction of samples to exchange between train and validation after splitting. Defaults 0.
    batch_size, num_workers : int, optional
        DataLoader settings.

    See Also
    --------
    foundation_model.data.dataset.CompoundDataset
    foundation_model.data.composition_sources
    """

    def __init__(
        self,
        task_configs: Sequence[TaskConfig],
        descriptor_fn: DescriptorFn,
        *,
        task_frames: Mapping[str, pd.DataFrame] | None = None,
        default_data_files: str | Sequence[str] | None = None,
        composition_column: str = "composition",
        composition_normalizer: CompositionNormalizer | None = normalize_composition,
        random_seed: int | None = 42,
        val_split: float = 0.1,
        test_split: float = 0.1,
        test_all: bool = False,
        swap_train_val_split: float = 0.0,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        logger.info("Initializing CompoundDataModule...")

        if not task_configs:
            raise ValueError("task_configs cannot be empty.")
        if descriptor_fn is None:
            raise ValueError("descriptor_fn must be provided.")
        if not 0.0 <= swap_train_val_split <= 1.0:
            raise ValueError("swap_train_val_split must be between 0.0 and 1.0 inclusive.")
        for split_name, split_value in (("val_split", val_split), ("test_split", test_split)):
            if not 0.0 <= split_value <= 1.0:
                raise ValueError(f"{split_name} must be between 0.0 and 1.0 inclusive, got {split_value}.")
        if val_split + test_split > 1.0:
            raise ValueError(
                f"val_split + test_split must not exceed 1.0 (got {val_split} + {test_split} = {val_split + test_split})."
            )

        self.task_configs = list(task_configs)
        self.descriptor_fn = descriptor_fn
        self._input_task_frames = dict(task_frames) if task_frames is not None else {}
        if default_data_files is None:
            self.default_data_files: tuple[str, ...] = ()
        elif isinstance(default_data_files, str):
            self.default_data_files = (default_data_files,)
        else:
            self.default_data_files = tuple(str(p) for p in default_data_files)
        self.composition_column = composition_column
        self.composition_normalizer = composition_normalizer
        # The DataModule owns the normalization policy; keep a recognized descriptor source in
        # sync so the opt-out (composition_normalizer=None) only has to be set in one place.
        if isinstance(descriptor_fn, PrecomputedDescriptorSource):
            descriptor_fn._composition_normalizer = composition_normalizer
        self.random_seed = random_seed
        self.val_split = val_split
        self.test_split = test_split
        self.test_all = test_all
        self.swap_train_val_split = swap_train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._seed_offsets: Dict[str, int] = {
            "train_split": 0,
            "test_split": 10_000,
            "task_masking": 20_000,
            "swap": 30_000,
        }

        # Populated by _prepare_sources()
        self._sources_ready = False
        self._descriptor_cache = DescriptorCache(descriptor_fn)
        self.descriptors: pd.DataFrame | None = None
        self._task_frames: Dict[str, pd.DataFrame] = {}
        self.master_index: List[str] = []
        self.split_series: pd.Series | None = None
        self.train_idx: List[str] = []
        self.val_idx: List[str] = []
        self.test_idx: List[str] = []
        self.predict_compositions: List[str] = []
        self.train_dataset: CompoundDataset | None = None
        self.val_dataset: CompoundDataset | None = None
        self.test_dataset: CompoundDataset | None = None
        self.predict_dataset: CompoundDataset | None = None
        self._train_sampler: DistributedSampler | None = None

        self.save_hyperparameters(
            # Callables / large frames must not be pickled into checkpoints (a function reference
            # such as composition_normalizer also trips torch's weights_only=True loader).
            ignore=["task_configs", "descriptor_fn", "task_frames", "composition_normalizer"],
        )

        logger.info(f"DataModule initialized with {len(self.task_configs)} task configurations.")

    # ------------------------------------------------------------------ helpers

    def _seed_for(self, purpose: str) -> int | None:
        """Return a deterministic seed for the given purpose derived from the base random_seed."""
        if self.random_seed is None:
            return None
        offset = self._seed_offsets.get(purpose, 0)
        return (self.random_seed + offset) % (2**32)

    def _supervised_tasks(self) -> List[TaskConfig]:
        """Enabled tasks that load external data (everything except AUTOENCODER)."""
        return [cfg for cfg in self.task_configs if cfg.enabled and cfg.type != TaskType.AUTOENCODER]

    def _composition_column_for(self, cfg: TaskConfig) -> str:
        return cfg.composition_column or self.composition_column

    def _canon(self, value: object) -> str:
        """Canonical composition key (normalizer with raw-string fallback)."""
        return canonical_key(value, self.composition_normalizer)

    def _normalize_frame(self, frame: pd.DataFrame, composition_column: str, *, task_name: str = "") -> pd.DataFrame:
        """Return a copy indexed by canonical composition keys, deduped keep-first."""
        frame = frame.copy()
        if composition_column in frame.columns:
            frame = frame.set_index(composition_column)
        if self.composition_normalizer is None:
            frame.index = frame.index.astype(str)  # vectorized fast path when normalization is off
        else:
            frame.index = pd.Index([self._canon(c) for c in frame.index])
        duplicated = frame.index.duplicated(keep="first")
        if duplicated.any():
            logger.warning(
                f"Task '{task_name or '<task>'}': in-memory frame has {int(duplicated.sum())} duplicate "
                "composition(s); keeping the first occurrence of each."
            )
            frame = frame[~duplicated]
        return frame

    def _resolved_task_masking_ratios(self) -> Dict[str, float]:
        """Per-task keep ratios sourced from each config's task_masking_ratio."""
        ratios: Dict[str, float] = {}
        for cfg in self.task_configs:
            ratio = getattr(cfg, "task_masking_ratio", None)
            if getattr(cfg, "enabled", True) and ratio is not None:
                ratios[cfg.name] = float(ratio)
        return ratios

    # ------------------------------------------------------------------ sources

    def _prepare_sources(self) -> None:
        """Load task frames, compute descriptors, and resolve the global split (idempotent)."""
        if self._sources_ready:
            return

        logger.info("--- Preparing composition-keyed data sources ---")

        # 1. Load each supervised task's frame (in-memory override or data_files).
        self._task_frames = {}
        for cfg in self._supervised_tasks():
            comp_col = self._composition_column_for(cfg)
            if cfg.name in self._input_task_frames:
                frame = self._normalize_frame(self._input_task_frames[cfg.name], comp_col, task_name=cfg.name)
                logger.info(f"Task '{cfg.name}': using provided in-memory frame ({len(frame)} rows).")
            elif cfg.data_files:
                frame = load_task_frame(
                    cfg.data_files, comp_col, task_name=cfg.name, composition_normalizer=self.composition_normalizer
                )
                logger.info(f"Task '{cfg.name}': loaded {len(frame)} rows from {len(cfg.data_files)} file(s).")
            elif self.default_data_files:
                frame = load_task_frame(
                    self.default_data_files,
                    comp_col,
                    task_name=cfg.name,
                    composition_normalizer=self.composition_normalizer,
                )
                logger.info(
                    f"Task '{cfg.name}': loaded {len(frame)} rows from shared default_data_files "
                    f"({len(self.default_data_files)} file(s))."
                )
            else:
                logger.warning(
                    f"Task '{cfg.name}': no in-memory frame, data_files, or default_data_files; it will only "
                    "contribute placeholder (masked) targets. Provide a data source for supervised training."
                )
                continue
            self._task_frames[cfg.name] = frame

        # 2. Composition universe = task compositions plus explicit predict_idx compositions.
        extra: List[str] = []
        for cfg in self._supervised_tasks():
            pid = cfg.predict_idx
            if isinstance(pid, (list, tuple)):
                extra.extend(self._canon(c) for c in pid)
        universe = build_composition_universe(self._task_frames, extra_compositions=extra)
        if not universe:
            raise ValueError(
                "No compositions found across task data sources. Provide data_files / task_frames "
                "or explicit predict_idx composition sequences."
            )
        logger.info(f"Composition universe size: {len(universe)}")

        # 3. Descriptors (cached); drop compositions without a valid descriptor.
        self.descriptors, dropped = self._descriptor_cache.resolve(universe)
        if dropped:
            logger.warning(
                f"{len(dropped)} composition(s) dropped: descriptor_fn produced no valid descriptor "
                f"(e.g. {', '.join(dropped[:5])})."
            )
        if self.descriptors.empty:
            hint = ""
            if self.composition_normalizer is None:
                hint = (
                    " The DataModule has composition_normalizer=None but bundled descriptor sources "
                    "(PrecomputedDescriptorSource / lookup_descriptor_fn) normalize by default — pass "
                    "composition_normalizer=None to the descriptor source too so both sides use the same keys."
                )
            raise ValueError(f"descriptor_fn produced no valid descriptors for any composition.{hint}")
        self.master_index = [str(c) for c in self.descriptors.index]
        logger.info(f"Master composition index size (with valid descriptors): {len(self.master_index)}")

        # 4. Resolve a single composition-level split.
        split_columns = {
            cfg.name: cfg.split_column for cfg in self._supervised_tasks() if cfg.name in self._task_frames
        }
        self.split_series = resolve_splits(
            self._task_frames,
            self.master_index,
            split_columns,
            val_split=self.val_split,
            test_split=self.test_split,
            random_seed=self._seed_for("test_split"),
            test_all=self.test_all,
        )
        labels = self.split_series
        self.train_idx = [c for c in self.master_index if labels[c] == "train"]
        self.val_idx = [c for c in self.master_index if labels[c] == "val"]
        self.test_idx = [c for c in self.master_index if labels[c] == "test"]

        self._apply_train_val_swap(self.swap_train_val_split, self._seed_for("swap"))

        logger.info(f"Split sizes: Train={len(self.train_idx)}, Val={len(self.val_idx)}, Test={len(self.test_idx)}")
        self._sources_ready = True

    def _apply_train_val_swap(self, swap_ratio: float, random_seed: int | None) -> None:
        """Randomly exchange a fraction of samples between train and validation splits."""
        if swap_ratio <= 0.0:
            return
        train_count = len(self.train_idx)
        val_count = len(self.val_idx)
        if train_count == 0 or val_count == 0:
            logger.warning(
                f"swap_train_val_split skipped because one split is empty (train={train_count}, val={val_count})."
            )
            return
        n_swap = int(min(train_count, val_count) * swap_ratio)
        if n_swap == 0:
            logger.info(f"swap_train_val_split computed zero samples to swap (swap_ratio={swap_ratio}).")
            return

        rng = np.random.default_rng(random_seed)
        train_choice = rng.choice(np.array(self.train_idx), size=n_swap, replace=False).tolist()
        val_choice = rng.choice(np.array(self.val_idx), size=n_swap, replace=False).tolist()
        train_swap_set = set(train_choice)
        val_swap_set = set(val_choice)
        train_remaining = [c for c in self.train_idx if c not in train_swap_set]
        val_remaining = [c for c in self.val_idx if c not in val_swap_set]
        self.train_idx = train_remaining + val_choice
        self.val_idx = val_remaining + train_choice
        logger.info(f"Swapped {n_swap} samples between train and validation (swap_ratio={swap_ratio}).")

    def _build_dataset(
        self,
        compositions: Sequence[str],
        *,
        is_predict: bool,
        dataset_name: str,
        apply_masking: bool,
    ) -> CompoundDataset | None:
        if len(compositions) == 0:
            logger.warning(f"{dataset_name}: no compositions; dataset will be None.")
            return None
        assert self.descriptors is not None
        masking = self._resolved_task_masking_ratios() if apply_masking else None
        return CompoundDataset(
            compositions=list(compositions),
            descriptors=self.descriptors,
            task_frames=self._task_frames,
            task_configs=self.task_configs,
            task_masking_ratios=masking,
            task_masking_seed=self._seed_for("task_masking"),
            is_predict_set=is_predict,
            dataset_name=dataset_name,
        )

    def _resolve_predict_compositions(self) -> List[str]:
        """Union of each task's predict_idx subset, preserving master-index order."""
        master_set = set(self.master_index)
        default_pool = list(self.test_idx) if len(self.test_idx) > 0 else list(self.master_index)
        selected: set[str] = set()

        for cfg in self._supervised_tasks():
            pid = cfg.predict_idx
            if pid is None:
                subset = default_pool
            elif isinstance(pid, str):
                if pid == "all":
                    subset = list(self.master_index)
                elif pid == "train":
                    subset = list(self.train_idx)
                elif pid == "val":
                    subset = list(self.val_idx)
                else:  # "test"
                    subset = list(self.test_idx)
            else:  # explicit sequence of composition keys
                requested = [self._canon(c) for c in pid]
                subset = [c for c in requested if c in master_set]
                missing = [c for c in requested if c not in master_set]
                if missing:
                    logger.warning(
                        f"Task '{cfg.name}': {len(missing)} predict_idx composition(s) have no descriptor "
                        f"and are skipped (e.g. {', '.join(missing[:5])})."
                    )
            selected.update(subset)

        return [c for c in self.master_index if c in selected]

    # ------------------------------------------------------------------ lightning

    def on_train_epoch_start(self):
        """Update the DistributedSampler epoch so shuffling differs across epochs."""
        if getattr(self, "_train_sampler", None) is not None and hasattr(self._train_sampler, "set_epoch"):
            if getattr(self, "trainer", None) is not None:
                self._train_sampler.set_epoch(self.trainer.current_epoch)

    def setup(self, stage: str | None = None):
        """Prepare datasets for the requested stage (fit, test, predict)."""
        logger.info(f"--- Setting up DataModule for stage: {stage} ---")
        self._prepare_sources()

        if stage == "fit" or stage is None:
            logger.info("--- Creating 'fit' stage datasets (train/val) ---")
            self.train_dataset = self._build_dataset(
                self.train_idx, is_predict=False, dataset_name="train_dataset", apply_masking=True
            )
            self.val_dataset = self._build_dataset(
                self.val_idx, is_predict=False, dataset_name="val_dataset", apply_masking=False
            )

        if stage == "test" or stage is None:
            logger.info("--- Creating 'test' stage dataset ---")
            self.test_dataset = self._build_dataset(
                self.test_idx, is_predict=False, dataset_name="test_dataset", apply_masking=False
            )

        if stage == "predict":
            logger.info("--- Creating 'predict' stage dataset ---")
            self.predict_compositions = self._resolve_predict_compositions()
            self.predict_dataset = self._build_dataset(
                self.predict_compositions, is_predict=True, dataset_name="predict_dataset", apply_masking=False
            )
        logger.info(f"--- DataModule setup for stage '{stage}' complete ---")

    # ------------------------------------------------------------------ dataloaders

    def _make_loader(self, dataset, *, shuffle: bool, track_sampler: bool):
        collate_fn = create_collate_fn_with_task_info(self.task_configs)
        use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        sampler: DistributedSampler | None
        if use_ddp:
            sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
            loader_shuffle = False
        else:
            sampler = None
            loader_shuffle = shuffle
        if track_sampler:
            self._train_sampler = sampler
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=loader_shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        if self.train_dataset is None or len(self.train_dataset) == 0:
            logger.warning("train_dataloader: Train dataset is None or empty. Returning None.")
            return None
        return self._make_loader(self.train_dataset, shuffle=True, track_sampler=True)

    def val_dataloader(self):
        if self.val_dataset is None or len(self.val_dataset) == 0:
            logger.info("val_dataloader: Validation dataset is None or empty. Returning None.")
            return None
        return self._make_loader(self.val_dataset, shuffle=False, track_sampler=False)

    def test_dataloader(self):
        if self.test_dataset is None or len(self.test_dataset) == 0:
            logger.info("test_dataloader: Test dataset is None or empty. Returning None.")
            return None
        return self._make_loader(self.test_dataset, shuffle=False, track_sampler=False)

    def predict_dataloader(self):
        if self.predict_dataset is None or len(self.predict_dataset) == 0:
            logger.info("predict_dataloader: Predict dataset is None or empty. Returning None.")
            return None
        return self._make_loader(self.predict_dataset, shuffle=False, track_sampler=False)
