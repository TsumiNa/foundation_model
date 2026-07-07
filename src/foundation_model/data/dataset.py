# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import ast  # For safely evaluating string representations of lists/arrays
import zlib
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from foundation_model.models.model_config import TaskType


# Helper function to parse elements that might be scalars, lists, or string representations of lists
def _parse_structured_element(
    element, task_name: str, column_name: str, dataset_name: str, expected_shape_for_nan: tuple
):
    """
    Parses an element from a pandas Series.
    Handles scalars, lists, numpy arrays, string representations of lists/arrays, and NaNs.
    Returns a numpy array. For NaNs or parsing errors, returns np.full(expected_shape_for_nan, np.nan).
    """
    if isinstance(element, np.ndarray) and np.issubdtype(element.dtype, np.number):
        return element.astype(float)  # Ensure float type for consistency
    elif isinstance(element, (list, tuple)) or (isinstance(element, np.ndarray) and element.dtype == object):
        # Handle lists, tuples, or object arrays that might contain mixed types
        processed_list = []
        for item in element:
            if isinstance(item, str):
                try:
                    # Attempt to evaluate if it's a string representation of a number
                    eval_item = ast.literal_eval(item)
                    if isinstance(eval_item, (int, float, bool)):  # bool is subclass of int
                        processed_list.append(float(eval_item))
                    else:  # ast.literal_eval returned a list/tuple/dict, treat as unparsable for this context
                        logger.debug(
                            f"[{dataset_name}] Task '{task_name}', column '{column_name}': Item '{item}' in list evaluated to non-scalar {type(eval_item)}. Using NaN."
                        )
                        processed_list.append(np.nan)
                except (ValueError, SyntaxError, TypeError):
                    logger.warning(  # Changed from debug to warning
                        f"[{dataset_name}] Task '{task_name}', column '{column_name}': Could not parse string element '{item}' in list. Using NaN."  # Changed "item" to "element"
                    )
                    processed_list.append(np.nan)
            elif isinstance(item, (int, float, bool)):
                processed_list.append(float(item))
            elif pd.isna(item):
                processed_list.append(np.nan)
            else:  # Other non-numeric types within the list
                logger.debug(
                    f"[{dataset_name}] Task '{task_name}', column '{column_name}': Non-numeric item '{item}' (type: {type(item)}) in list. Using NaN."
                )
                processed_list.append(np.nan)
        return np.array(processed_list, dtype=float)
    elif isinstance(element, str):
        try:
            # For string elements that are supposed to be lists/arrays themselves
            parsed_element = ast.literal_eval(element)
            # Ensure it's an array, even if ast.literal_eval returns a scalar
            # This branch now primarily handles strings that represent entire lists/arrays like "[1, 2, 3]"
            if isinstance(parsed_element, (list, tuple)):
                # Recursively call to process items within the parsed list/tuple
                return _parse_structured_element(
                    parsed_element, task_name, column_name, dataset_name, expected_shape_for_nan
                )
            elif isinstance(parsed_element, (int, float, bool)):
                return np.array([float(parsed_element)])  # Single numeric value from string
            else:  # Parsed to something else unexpected
                logger.warning(
                    f"[{dataset_name}] Task '{task_name}', column '{column_name}': String element '{element}' evaluated to unexpected type {type(parsed_element)}. Using NaN placeholder."
                )
                return np.full(expected_shape_for_nan, np.nan)
        except (ValueError, SyntaxError, TypeError):
            logger.warning(
                f"[{dataset_name}] Task '{task_name}', column '{column_name}': Could not parse string element '{element}' as a list/scalar. Using NaN placeholder."
            )
            return np.full(expected_shape_for_nan, np.nan)
    elif pd.isna(element):
        return np.full(expected_shape_for_nan, np.nan)
    elif isinstance(element, (int, float, bool)):  # Scalar number
        return np.array([float(element)])
    else:  # Other unhandled type, treat as error / NaN
        logger.warning(
            f"[{dataset_name}] Task '{task_name}', column '{column_name}': Unhandled element type '{type(element)}' ('{element}'). Using NaN placeholder."
        )
        return np.full(expected_shape_for_nan, np.nan)


def subsample_keep_mask(base_mask: np.ndarray, keep_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly reduce a boolean row mask to ``round(n_valid * keep_ratio)`` kept rows.

    Pure function: returns a copy of ``base_mask`` with ``n_valid - n_keep`` of its True entries
    flipped to False, drawn without replacement from ``rng``. Rows that are False in ``base_mask``
    (no usable label) are never turned on.
    """
    kept = base_mask.copy()
    valid_indices = np.flatnonzero(base_mask)
    num_to_keep = int(np.round(len(valid_indices) * keep_ratio))
    num_to_drop = len(valid_indices) - num_to_keep
    if num_to_drop > 0:
        kept[rng.choice(valid_indices, size=num_to_drop, replace=False)] = False
    return kept


class CompoundDataset(Dataset):
    """Composition-keyed dataset for the multi-task model.

    Alignment happens at construction: ``descriptors`` and every per-task frame are reindexed
    to ``compositions`` (the sample order / dataset length for this split). Compositions absent
    from a task's frame become NaN rows and are masked out for that task. ``__getitem__`` then
    fetches positionally from these aligned structures, yielding the unchanged batch tuple
    ``(x_formula, y_dict, task_masks_dict, t_sequences_dict)``.

    Mask lifecycle
    --------------
    Two layers, kept separate so subsampling is re-drawable:

    1. **Base mask** (``_base_masks[task]``, immutable): one bool per row — "this row has a
       usable label for this task" (derived from NaN presence at construction).
    2. **Keep mask** (``task_masks_dict[task]``, what ``__getitem__`` serves): the base mask,
       optionally subsampled to ``task_masking_ratios[task]`` of its valid rows (training only).

    Each task draws its keep mask from its **own** RNG stream seeded with
    ``(task_masking_seed, crc32(task_name), epoch)`` — so the drawn subset depends only on the
    seed, the task, and the epoch, never on which other tasks share the dataset or the order
    they are processed in. :meth:`resample_task_masks` redraws layer 2 for a new epoch
    (construction uses ``epoch=0``, so a later ``resample_task_masks(epoch=0)`` is a no-op).

    Parameters
    ----------
    compositions : Sequence[str]
        Composition keys defining the sample order and dataset length for this split.
    descriptors : pd.DataFrame
        Descriptor features indexed by composition. Must cover every entry in ``compositions``.
    task_frames : Mapping[str, pd.DataFrame]
        Per-task data frames indexed by composition (each holding the task's ``data_column``,
        optional ``t_column``, etc.). Tasks absent here (or AUTOENCODER tasks) get placeholders.
    task_configs : Sequence
        Task configuration objects.
    task_masking_ratios : Mapping[str, float] | None
        Optional per-task keep ratios applied during training only.
    task_masking_seed : int | None
        Base seed for the per-task masking streams. ``None`` = non-reproducible draws.
    is_predict_set : bool
        When True, disables random task masking (masks come from NaN presence only).
    dataset_name : str
        Label used in log messages.
    """

    def __init__(
        self,
        compositions: Sequence[str],
        descriptors: pd.DataFrame,
        task_frames: Mapping[str, pd.DataFrame],
        task_configs: List,
        *,
        task_masking_ratios: Mapping[str, float] | None = None,
        task_masking_seed: int | None = None,
        is_predict_set: bool = False,
        dataset_name: str = "dataset",
    ):
        self.dataset_name = dataset_name
        logger.info(f"[{self.dataset_name}] Initializing CompoundDataset...")
        self.is_predict_set = is_predict_set
        self.task_masking_ratios = task_masking_ratios
        self.task_masking_seed = task_masking_seed

        # --- Input validation ---
        if not isinstance(descriptors, pd.DataFrame):
            logger.error(f"[{self.dataset_name}] descriptors type error: {type(descriptors)}")
            raise TypeError("descriptors must be a pd.DataFrame")
        if not task_configs:
            raise ValueError("task_configs list cannot be empty.")

        self.compositions: List[str] = [str(c) for c in compositions]
        if len(self.compositions) == 0:
            raise ValueError("compositions cannot be empty.")
        if descriptors.empty:
            raise ValueError("descriptors DataFrame cannot be empty.")

        self.x_formula = self._align_descriptors(descriptors)
        self.x_struct: torch.Tensor | None = None
        logger.info(f"[{self.dataset_name}] Final x_formula shape: {self.x_formula.shape}")

        self._aligned_frames = self._align_task_frames(task_frames)

        # Per-task aligned structures, filled by _build_task_targets below.
        self.y_dict: Dict[str, torch.Tensor | List[torch.Tensor]] = {}
        self.t_sequences_dict: Dict[str, torch.Tensor | List[torch.Tensor]] = {}
        self.task_masks_dict: Dict[str, torch.Tensor | List[torch.Tensor]] = {}
        self._base_masks: Dict[str, np.ndarray] = {}  # 1-D "row has a label" masks; never mutated
        self.enabled_task_names: List[str] = []

        for cfg in task_configs:
            if not hasattr(cfg, "enabled") or not cfg.enabled:
                continue
            self.enabled_task_names.append(cfg.name)
            logger.info(f"[{self.dataset_name}] Processing enabled task '{cfg.name}' (type: {cfg.type.name})")
            if cfg.type == TaskType.AUTOENCODER:
                # AutoEncoder tasks use the input features as target, so no external data loading is
                # needed. The model handles target generation (x) and masking (ones) internally.
                continue
            self._build_task_targets(cfg)
            self.task_masks_dict[cfg.name] = self._mask_to_storage(cfg.name, self._base_masks[cfg.name])

        self.resample_task_masks(epoch=0)

        logger.info(
            f"[{self.dataset_name}] CompoundDataset initialization complete. "
            f"Processed {len(self.enabled_task_names)} enabled tasks."
        )

    # ------------------------------------------------------------------ construction stages

    def _align_descriptors(self, descriptors: pd.DataFrame) -> torch.Tensor:
        """Reindex the descriptor frame to the composition order and return it as a tensor."""
        descriptors = descriptors.copy()
        descriptors.index = descriptors.index.astype(str)
        missing_desc = [c for c in self.compositions if c not in descriptors.index]
        if missing_desc:
            preview = ", ".join(missing_desc[:5])
            raise ValueError(
                f"[{self.dataset_name}] descriptors is missing {len(missing_desc)} composition(s) "
                f"required by this split (e.g. {preview})."
            )
        return torch.tensor(descriptors.loc[self.compositions].values, dtype=torch.float32)

    def _align_task_frames(self, task_frames: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Reindex each provided task frame to the composition order (absent rows become NaN)."""
        aligned: Dict[str, pd.DataFrame] = {}
        for task_name, frame in task_frames.items():
            if frame is None:
                continue
            frame = frame.copy()
            frame.index = frame.index.astype(str)
            aligned[task_name] = frame.reindex(self.compositions)
        return aligned

    def _task_column(self, task_name: str, column: str) -> pd.Series | None:
        """One aligned column for a task; None when the column is absent from its frame."""
        frame = self._aligned_frames.get(task_name)
        if frame is None or column not in frame.columns:
            return None
        logger.debug(f"[{self.dataset_name}] Task '{task_name}': Loading data from column '{column}'.")
        return frame[column]

    def _build_task_targets(self, cfg) -> None:
        """Build y (+t) tensors and the base mask for one supervised task."""
        task_name = cfg.name
        if not cfg.data_column:
            logger.error(f"[{self.dataset_name}] Task '{task_name}': data_column is not specified in config.")
            raise ValueError(f"data_column for task '{task_name}' must be specified.")
        raw = self._task_column(task_name, cfg.data_column)
        if raw is None:
            level = logger.debug if self.is_predict_set else logger.warning
            level(
                f"[{self.dataset_name}] Task '{task_name}': data_column '{cfg.data_column}' not found. "
                "Using placeholder values."
            )
            raw = pd.Series([np.nan] * len(self.compositions), index=self.compositions)

        if cfg.type == TaskType.KERNEL_REGRESSION:
            self._build_kernel_targets(cfg, raw)
        else:
            self._build_dense_targets(cfg, raw)

    def _build_kernel_targets(self, cfg, raw: pd.Series) -> None:
        """Variable-length sequence targets stored as List[Tensor]; a row is valid unless all-NaN."""
        task_name = cfg.name
        values = [
            _parse_structured_element(element, task_name, cfg.data_column, self.dataset_name, (1,)) for element in raw
        ]
        self.y_dict[task_name] = [torch.tensor(np.nan_to_num(seq, nan=0.0), dtype=torch.float32) for seq in values]
        self._base_masks[task_name] = np.array([not np.all(np.isnan(seq)) for seq in values], dtype=bool)
        logger.debug(
            f"[{self.dataset_name}] Task '{task_name}': y_dict stored as List[Tensor] with "
            f"{len(values)} sequences, base_mask valid count: {int(self._base_masks[task_name].sum())}"
        )

        if not cfg.t_column:
            logger.error(f"[{self.dataset_name}] Task '{task_name}': t_column is not specified in config.")
            raise ValueError(f"t_column for KernelRegression task '{task_name}' must be specified.")
        raw_t = self._task_column(task_name, cfg.t_column)
        if raw_t is None:
            logger.error(
                f"[{self.dataset_name}] Task '{task_name}': t_column '{cfg.t_column}' "
                f"is specified in config but not found in the task data frame."
            )
            raise ValueError(f"T-parameter column '{cfg.t_column}' for task '{task_name}' not found in task data.")
        t_values = [
            _parse_structured_element(element, task_name, cfg.t_column, self.dataset_name, (1,)) for element in raw_t
        ]
        self.t_sequences_dict[task_name] = [
            torch.tensor(np.nan_to_num(seq, nan=0.0), dtype=torch.float32) for seq in t_values
        ]
        logger.debug(
            f"[{self.dataset_name}] Task '{task_name}': t_sequences_dict stored as List[Tensor] "
            f"with {len(t_values)} sequences"
        )

    def _build_dense_targets(self, cfg, raw: pd.Series) -> None:
        """Scalar regression / classification targets stacked into one tensor."""
        task_name = cfg.name
        values = [
            _parse_structured_element(element, task_name, cfg.data_column, self.dataset_name, (1,)) for element in raw
        ]
        try:
            stacked = np.stack(values)
        except ValueError as e:  # Stacking failed, likely inconsistent shapes
            logger.error(
                f"[{self.dataset_name}] Task '{task_name}', column '{cfg.data_column}': "
                f"Error stacking parsed data - {e}. Check data consistency."
            )
            raise ValueError(f"Error processing data column '{cfg.data_column}' for task '{task_name}': {e}")

        self._base_masks[task_name] = np.array([not np.all(np.isnan(v)) for v in values], dtype=bool)

        if cfg.type == TaskType.CLASSIFICATION:
            # Use -100 as placeholder for missing data (avoids conflict with real class labels
            # 0,1,2,...). Actual missing data handling is done via the mask mechanism.
            y_tensor = torch.tensor(np.nan_to_num(stacked, nan=-100).astype(np.int64), dtype=torch.long)
        else:  # REGRESSION
            y_tensor = torch.tensor(np.nan_to_num(stacked, nan=0.0), dtype=torch.float32)
        self.y_dict[task_name] = y_tensor
        logger.debug(
            f"[{self.dataset_name}] Task '{task_name}': y_dict shape {y_tensor.shape}, "
            f"base_mask valid count: {int(self._base_masks[task_name].sum())}"
        )

    # ------------------------------------------------------------------ keep-ratio masking

    def _mask_rng(self, task_name: str, epoch: int) -> np.random.Generator:
        """Independent RNG stream per (seed, task, epoch); entropy-seeded when seed is None."""
        if self.task_masking_seed is None:
            return np.random.default_rng()
        return np.random.default_rng([self.task_masking_seed, zlib.crc32(task_name.encode()), epoch])

    def _mask_to_storage(self, task_name: str, mask_np: np.ndarray) -> torch.Tensor | List[torch.Tensor]:
        """Convert a 1-D row mask to the per-task storage format served by ``__getitem__``."""
        y = self.y_dict.get(task_name)
        if isinstance(y, list):  # kernel regression: one bool tensor per variable-length sequence
            return [
                torch.ones_like(seq, dtype=torch.bool) if mask_np[i] else torch.zeros_like(seq, dtype=torch.bool)
                for i, seq in enumerate(y)
            ]
        return torch.tensor(mask_np.reshape(-1, 1), dtype=torch.bool)

    def resample_task_masks(self, *, epoch: int) -> None:
        """(Re)draw the keep mask of every ratio-masked task for ``epoch``.

        Deterministic given ``(task_masking_seed, task_name, epoch)``; construction already draws
        with ``epoch=0``. Predict sets and tasks with ratio >= 1.0 are untouched. Used by the
        per-epoch replay resampling callback (``pretrain.replay.resample = "epoch"``).
        """
        if self.is_predict_set:
            return
        for task_name, keep_ratio in (self.task_masking_ratios or {}).items():
            if task_name not in self._base_masks or not 0.0 <= keep_ratio < 1.0:
                continue
            base = self._base_masks[task_name]
            kept = subsample_keep_mask(base, keep_ratio, self._mask_rng(task_name, epoch))
            self.task_masks_dict[task_name] = self._mask_to_storage(task_name, kept)
            logger.debug(
                f"[{self.dataset_name}] Task '{task_name}': keep mask drawn for epoch {epoch} "
                f"(keep_ratio={keep_ratio}, kept {int(kept.sum())}/{int(base.sum())} labelled rows)."
            )

    # ------------------------------------------------------------------ Dataset protocol

    def __len__(self):
        return len(self.x_formula)

    def __getitem__(self, idx):
        model_input_x = self.x_formula[idx]

        sample_y_dict = {name: self.y_dict[name][idx] for name in self.enabled_task_names if name in self.y_dict}
        sample_task_masks_dict = {
            name: self.task_masks_dict[name][idx] for name in self.enabled_task_names if name in self.task_masks_dict
        }
        sample_t_sequences_dict = {
            name: self.t_sequences_dict[name][idx] for name in self.enabled_task_names if name in self.t_sequences_dict
        }

        return model_input_x, sample_y_dict, sample_task_masks_dict, sample_t_sequences_dict

    @property
    def attribute_names(self) -> List[str]:
        return self.enabled_task_names[:]
