# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Union

import joblib
import lightning as L
import numpy as np
import pandas as pd
import torch
from jsonargparse.typing import Path_fr
from loguru import logger
from sklearn.model_selection import train_test_split  # For data splitting
from torch.utils.data import DataLoader

from foundation_model.models.model_config import (
    ClassificationTaskConfig,
    ExtendRegressionTaskConfig,
    RegressionTaskConfig,
    TaskType,
)

from .dataset import CompoundDataset


def create_collate_fn_with_task_info(task_configs):
    """
    Creates a custom collate function that handles ExtendRegression tasks properly.

    ExtendRegression tasks need List[Tensor] format for both targets and t-parameters
    to support variable-length sequences without padding waste.

    Parameters
    ----------
    task_configs : List
        List of task configuration objects

    Returns
    -------
    callable
        Custom collate function for DataLoader
    """
    extend_regression_tasks = {
        cfg.name for cfg in task_configs if cfg.type == TaskType.ExtendRegression and cfg.enabled
    }

    def custom_collate_fn(batch):
        """
        Custom collate function for batching data.

        Parameters
        ----------
        batch : List[Tuple]
            List of (model_input_x, sample_y_dict, sample_task_masks_dict, sample_sequence_data_dict)

        Returns
        -------
        Tuple
            (batched_input, batched_y_dict, batched_mask_dict, batched_sequence_data_dict)
        """
        model_inputs, y_dicts, mask_dicts, sequence_data_dicts = zip(*batch)

        # Handle model inputs (formula/structure features)
        if isinstance(model_inputs[0], tuple):
            # Structure data is present
            formulas, structures = zip(*model_inputs)
            batched_input = (torch.stack(formulas), torch.stack(structures))
        else:
            # Only formula data
            batched_input = torch.stack(model_inputs)

        # Handle targets and masks based on task type
        batched_y_dict = {}
        batched_mask_dict = {}

        for key in y_dicts[0].keys():
            if key in extend_regression_tasks:
                # ExtendRegression: Keep List[Tensor] format for variable-length sequences
                batched_y_dict[key] = [d[key] for d in y_dicts]
                batched_mask_dict[key] = [d[key] for d in mask_dicts]
            else:
                # Other tasks: Normal stacking
                batched_y_dict[key] = torch.stack([d[key] for d in y_dicts])
                batched_mask_dict[key] = torch.stack([d[key] for d in mask_dicts])

        # Handle sequence data (t-parameters) - always List[Tensor] format
        batched_sequence_data_dict = {}
        for key in sequence_data_dicts[0].keys():
            batched_sequence_data_dict[key] = [d[key] for d in sequence_data_dicts]

        return batched_input, batched_y_dict, batched_mask_dict, batched_sequence_data_dict

    return custom_collate_fn


class CompoundDataModule(L.LightningDataModule):
    def __init__(
        self,
        formula_desc_source: Union[pd.DataFrame, Path_fr],  # type: ignore
        task_configs: List[Union[RegressionTaskConfig, ClassificationTaskConfig, ExtendRegressionTaskConfig]],
        attributes_source: Optional[Union[pd.DataFrame, Path_fr]] = None,  # type: ignore
        structure_desc_source: Optional[Union[pd.DataFrame, Path_fr]] = None,  # type: ignore
        with_structure: bool = False,
        task_masking_ratios: Optional[Dict[str, float]] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        train_random_seed: int = 42,
        test_random_seed: int = 24,
        test_all: bool = False,
        predict_idx: Optional[Union[pd.Index, np.ndarray, List]] = None,  # Reintroduced
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        """
        Initialize the data module.
        The `formula_desc_source` and its index are treated as the primary reference for data alignment.

        Parameters
        ----------
        formula_desc_source : Union[pd.DataFrame, np.ndarray, str]
            Formula descriptors (DataFrame, NumPy array, or path to pickle/CSV/parquet). Its index is the master reference.
        task_configs : List
            List of task configurations.
        attributes_source : Optional[Union[pd.DataFrame, str]], optional
            Source for task target attributes, sequence data, temperature data, and 'split' column.
            Can be a DataFrame, NumPy array, or a path to a pickle/CSV/parquet file.
            Defaults to None. If None, the DataModule can only be used for prediction with non-sequence tasks,
            as sequence tasks typically require input columns (e.g., for series or temperatures) from this source.
            If provided, it will be aligned with `formula_desc_source`.
        structure_desc_source : Optional[Union[pd.DataFrame, np.ndarray, str]], optional
            Structure descriptors. If provided, it will be aligned with `formula_desc_source`. Defaults to None.
        with_structure : bool, optional
            If True, attempt to load and use structure descriptors. Defaults to False.
        task_masking_ratios : Optional[Dict[str, float]], optional
            Ratios for random masking per task in the training set. Defaults to None.
        val_split : float, optional
            Proportion of non-test data (derived from `attributes_source` if available, or `formula_desc_source` otherwise)
            to use for validation. Defaults to 0.1. Only applicable if `attributes_source` is provided and no 'split' column is used.
        test_split : float, optional
            Proportion of data (derived from `attributes_source` if available, or `formula_desc_source` otherwise)
            to use for testing. Defaults to 0.1. Only applicable if `attributes_source` is provided and no 'split' column is used.
        train_random_seed : int, optional
            Random seed for splitting train and validation sets. Defaults to 42.
        test_random_seed : int, optional
            Random seed for splitting the test set from the rest. Defaults to 24.
        test_all : bool, optional
            If True, all data (after alignment) is used for the test set. `train_idx` and `val_idx` will be empty.
            This overrides `test_split` and 'split' column logic if `attributes_source` is provided. Defaults to False.
        predict_idx : Optional[Union[pd.Index, np.ndarray, List]], optional
            Specific indices to use for the prediction set. These indices must be present in `formula_desc_source`.
            If provided, `setup(stage='predict')` will use these indices directly.
            If None (default), `predict_idx` will be derived during `setup(stage='predict')` from `test_idx` (if available and not empty)
            or otherwise from the full set of available indices from `formula_desc_source`.
        batch_size : int, optional
            Batch size for dataloaders. Defaults to 32.
        num_workers : int, optional
            Number of workers for dataloaders. Defaults to 0.
        """
        super().__init__()
        logger.info("Initializing CompoundDataModule...")

        # Explicitly assign parameters to self for clarity and linter compatibility
        self.task_configs = task_configs
        self.with_structure = with_structure
        self.task_masking_ratios = task_masking_ratios
        self.val_split = val_split
        self.test_split = test_split
        self.train_random_seed = train_random_seed
        self.test_random_seed = test_random_seed
        self.test_all = test_all
        self.init_predict_idx = (
            pd.Index(predict_idx) if predict_idx is not None else None
        )  # Store user-provided predict_idx
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.save_hyperparameters(
            ignore=[
                "formula_desc_source",
                "attributes_source",
                "structure_desc_source",
                "predict_idx",
                "task_configs",
            ]  # ADDED "task_configs"
        )

        # --- Load Data ---
        logger.info("--- Loading Data ---")
        source_to_load = str(formula_desc_source) if isinstance(formula_desc_source, Path_fr) else formula_desc_source  # type: ignore
        self.formula_df = self._load_data(source_to_load, "formula_desc")
        if self.formula_df is None or self.formula_df.empty:
            raise ValueError("formula_desc_source must be successfully loaded and cannot be empty.")
        logger.info(f"Initial loaded formula_df length: {len(self.formula_df)}")

        # Formula_df is the primary reference. Clean it first.
        self.formula_df.dropna(how="all", inplace=True)
        if self.formula_df.empty:  # Check again after dropna
            raise ValueError("formula_df became empty after removing rows with all NaNs.")
        master_index = self.formula_df.index
        logger.info(
            f"Formula_df length after initial dropna: {len(self.formula_df)}. This index is now the master reference."
        )

        # Handle optional attributes_source
        self.attributes_df = None
        if attributes_source is not None:
            source_to_load_attrs = (
                str(attributes_source) if isinstance(attributes_source, Path_fr) else attributes_source  # type: ignore
            )
            self.attributes_df = self._load_data(source_to_load_attrs, "attributes")
            if self.attributes_df is None or self.attributes_df.empty:
                # If source was provided but failed to load or was empty, it's an error.
                raise ValueError("attributes_source was provided but could not be loaded or is empty.")
            logger.info(f"Initial loaded attributes_df length: {len(self.attributes_df)}")

            # --- Align DataFrames ---
            logger.info(f"--- Aligning DataFrames by formula_df index (master_index length: {len(master_index)}) ---")

            # Align attributes_df to master_index (from formula_df)
            original_attributes_len = len(self.attributes_df)
            self.attributes_df = self.attributes_df.reindex(master_index)
            if len(self.attributes_df) != original_attributes_len:
                logger.warning(
                    f"Attributes_df reindexed. Original length: {original_attributes_len}, new length: {len(self.attributes_df)}."
                )
            if self.attributes_df.isnull().values.any():
                logger.warning(
                    "attributes_df contains NaN values after reindexing. This is expected if some properties are missing for certain samples."
                )
            self.attributes_df.dropna(how="all", inplace=True)
            if self.attributes_df.empty:
                raise ValueError(
                    "attributes_df became empty after reindexing and removing rows with all NaNs. Check index compatibility with formula_df."
                )

            # Update master_index based on intersection after attributes_df processing
            common_index_after_attrs = master_index.intersection(self.attributes_df.index)
            if common_index_after_attrs.empty:
                raise ValueError(
                    "No common_index found between formula_df and attributes_df after processing. Data is not alignable."
                )
            if len(common_index_after_attrs) < len(master_index):
                logger.warning(
                    f"Master index (from formula_df) reduced from {len(master_index)} to {len(common_index_after_attrs)} after aligning with attributes_df. "
                    "Some formula_df entries might not have corresponding attributes_df entries or vice-versa after NaN removal."
                )
            master_index = common_index_after_attrs

            self.formula_df = self.formula_df.loc[master_index]
            self.attributes_df = self.attributes_df.loc[master_index]  # attributes_df is now aligned
            logger.info(f"Length after aligning formula_df and attributes_df: {len(master_index)}")
        else:  # attributes_source is None
            logger.info(
                "attributes_source is None. Proceeding without attributes_df. This is only supported if no sequence tasks require it."
            )
            # Validate that no task requires attributes_df if it's None,
            # especially if an ExtendRegression task specifies a t_column.
            for cfg in self.task_configs:
                if cfg.enabled and cfg.type == TaskType.ExtendRegression:
                    # Type check and cast to access ExtendRegression-specific attributes
                    if isinstance(cfg, ExtendRegressionTaskConfig) and hasattr(cfg, "t_column") and cfg.t_column:
                        logger.error(
                            f"Task '{cfg.name}' is an ExtendRegression task that specifies a 't_column' ('{cfg.t_column}'), "
                            f"but attributes_source is None. attributes_source is required to load this t-parameter data."
                        )
                        raise ValueError(
                            f"attributes_source cannot be None when ExtendRegression task '{cfg.name}' requires a t_column ('{cfg.t_column}')."
                        )
            # If we reach here, attributes_source is None, and no enabled ExtendRegressionTaskConfig requires a t_column.
            # Other tasks (or ExtendRegression tasks without a t_column) will have their data_column handled by CompoundDataset
            # (likely resulting in placeholders if data_column was specified).
            # self.attributes_df remains None. self.formula_df uses the original master_index.
            logger.info(f"formula_df (master) length: {len(master_index)}")

        # Handle structure_df
        self.structure_df = None
        self.actual_with_structure = False
        if self.with_structure:
            logger.info("Attempting to load and align structure_desc as with_structure is True.")
            if structure_desc_source is not None:
                source_to_load_struct = (
                    str(structure_desc_source) if isinstance(structure_desc_source, Path_fr) else structure_desc_source  # type: ignore
                )
                loaded_structure_df = self._load_data(source_to_load_struct, "structure_desc")
                if loaded_structure_df is not None and not loaded_structure_df.empty:
                    logger.info(f"Initial loaded structure_df length: {len(loaded_structure_df)}")

                    # Attempt to align loaded_structure_df with the current master_index
                    aligned_structure_df = loaded_structure_df.reindex(master_index)
                    aligned_structure_df.dropna(how="all", inplace=True)  # Remove rows that became all NaN

                    # Check if the aligned_structure_df perfectly matches the master_index
                    if not aligned_structure_df.empty and aligned_structure_df.index.equals(master_index):
                        self.structure_df = aligned_structure_df
                        self.actual_with_structure = True
                        logger.info(f"Successfully aligned structure_df. Length: {len(self.structure_df)}")
                    else:
                        logger.warning(
                            f"Failed to align structure_df with the common index of formula_df and attributes_df. "
                            f"Master index len: {len(master_index)}, structure_df after reindex & dropna len: {len(aligned_structure_df)}, "
                            f"Indices equal: {aligned_structure_df.index.equals(master_index) if not aligned_structure_df.empty else 'N/A (empty)'}. "
                            f"Structure data will NOT be used."
                        )
                        # self.structure_df remains None, self.actual_with_structure remains False
                else:  # loaded_structure_df was None or empty
                    logger.warning(
                        "Failed to load structure_desc_source or it was empty. Proceeding without structure data."
                    )
            else:  # structure_desc_source was None
                logger.warning(
                    "with_structure is True, but structure_desc_source is None. Proceeding without structure data."
                )
        else:  # with_structure is False
            logger.info("with_structure is False. Structure data will not be loaded or used.")
        # self.structure_df and self.actual_with_structure are set correctly by now.

        # Final state logging
        logger.info(f"Final aligned formula_df length: {len(self.formula_df)}")
        if self.attributes_df is not None:
            logger.info(f"Final aligned attributes_df length: {len(self.attributes_df)}")
        else:
            logger.info("Final attributes_df is None.")
        if self.structure_df is not None:  # Check if it's not None before logging length
            logger.info(f"Final aligned structure_df length: {len(self.structure_df)}")
        else:
            logger.info("Final aligned structure_df is None.")
        logger.info(f"Final actual_with_structure status: {self.actual_with_structure}")

        # self.task_configs was already assigned
        logger.info(f"DataModule initialized with {len(self.task_configs)} task configurations.")

    def _load_data(self, source: Union[pd.DataFrame, str], name: str) -> Optional[pd.DataFrame]:
        """Helper to load data from various sources."""
        logger.debug(f"Attempting to load '{name}' from source type: {type(source)}")
        if source is None:
            logger.warning(f"Source for '{name}' is None.")
            return None
        try:
            df = None
            if isinstance(source, str):
                logger.info(f"Loading '{name}' data from path: {source}")
                if source.endswith(".pkl"):
                    df = joblib.load(source)
                elif source.endswith((".pd", ".pd.z", ".pd.xz")):
                    df = pd.read_pickle(source)
                elif source.endswith(".pd.parquet"):
                    df = pd.read_parquet(source)
                elif source.endswith(".csv"):
                    df = pd.read_csv(source, index_col=0)  # Assuming first column as index
                else:
                    logger.error(
                        f"Unsupported file type for '{name}': {source}. Must be .pkl, .pd, .pd.z, .pd.xz, .pd.parquet, or .csv."
                    )
                    raise ValueError(f"Unsupported file type for {name}: {source}.")
            elif isinstance(source, pd.DataFrame):
                logger.info(f"Using provided pd.DataFrame for '{name}' data.")
                df = source.copy()  # Use a copy
            else:
                logger.error(f"Unsupported data type for '{name}': {type(source)}.")
                raise TypeError(f"Unsupported data type for {name}: {type(source)}.")

            if df is not None:
                logger.info(f"Successfully loaded '{name}'. Shape: {df.shape}")
            return df

        except FileNotFoundError:
            logger.error(f"File not found for '{name}': {source}")
            return None
        except (ValueError, TypeError) as e:  # Catch specific errors to re-raise
            # These are intentionally raised for unsupported types/files, so let them propagate
            raise e
        except Exception as e:  # Catch other, unexpected exceptions
            logger.error(f"Unexpected error loading '{name}' data from {source}: {e}", exc_info=True)
            return None

    def setup(self, stage: Optional[str] = None):
        """Prepare datasets for different stages (fit, test, predict)."""
        logger.info(f"--- Setting up DataModule for stage: {stage} ---")
        if self.formula_df is None:  # attributes_df can now be None
            logger.error("formula_df is None. Cannot proceed with setup.")
            raise ValueError("formula_df not loaded properly.")

        # Determine full_idx based on whether attributes_df exists
        if self.attributes_df is not None:
            full_idx = self.attributes_df.index  # Should be same as formula_df.index at this point
        else:  # attributes_df is None, use formula_df.index as the source of all samples
            full_idx = self.formula_df.index
        logger.info(
            f"Total samples available before splitting (from {'attributes_df' if self.attributes_df is not None else 'formula_df'} index): {len(full_idx)}"
        )

        if self.test_all:
            self.train_idx = pd.Index([])
            self.val_idx = pd.Index([])
            self.test_idx = full_idx
            logger.info("Data split strategy: Using all data for testing (test_all=True).")
        # Splitting logic relies on attributes_df for 'split' column or for random splits if attributes_df exists
        elif self.attributes_df is not None and "split" in self.attributes_df.columns:
            logger.info("Data split strategy: Using 'split' column from attributes_df.")
            split_counts = self.attributes_df["split"].value_counts()
            logger.info(f"Value counts in 'split' column: {split_counts.to_dict()}")
            if not all(s_type in ["train", "val", "test"] for s_type in split_counts.index):
                logger.error(f"Invalid values in 'split' column: {split_counts.index.tolist()}")
                raise ValueError("Invalid values in 'split' column. Must be 'train', 'val', or 'test'.")

            self.train_idx = self.attributes_df[self.attributes_df["split"] == "train"].index
            self.val_idx = self.attributes_df[self.attributes_df["split"] == "val"].index
            self.test_idx = self.attributes_df[self.attributes_df["split"] == "test"].index

            if self.val_idx.empty and not self.train_idx.empty:  # and self.attributes_df is not None implicitly
                logger.warning("No 'val' samples in 'split' column. Splitting 10% of 'train' for validation.")
                self.train_idx, self.val_idx = train_test_split(
                    self.train_idx, test_size=0.1, random_state=self.train_random_seed, shuffle=True
                )
        elif (
            self.attributes_df is not None
        ):  # Random split only if attributes_df exists to provide a basis for splitting
            logger.info(
                "Data split strategy: Performing random train/val/test splits based on full_idx (derived from attributes_df)."
            )
            logger.info(f"Test split ratio: {self.test_split}, Validation split ratio (of non-test): {self.val_split}")
            if self.test_split >= 1.0:
                logger.info(f"test_split is {self.test_split}, assigning all data to test set.")
                self.test_idx = full_idx
                train_val_idx = pd.Index([])
            elif self.test_split > 0:
                train_val_idx, self.test_idx = train_test_split(
                    full_idx,  # full_idx here is from attributes_df.index
                    test_size=self.test_split,
                    random_state=self.test_random_seed,
                    shuffle=True,
                )
                logger.info(
                    f"Split full data ({len(full_idx)}) into train_val ({len(train_val_idx)}) and test ({len(self.test_idx)}) using seed {self.test_random_seed}."
                )
            else:  # test_split is 0
                train_val_idx = full_idx
                self.test_idx = pd.Index([])
                logger.info("test_split is 0. All data used for train_val.")

            if self.val_split > 0 and len(train_val_idx) > 0:
                effective_val_split = self.val_split
                if (1.0 - self.test_split) > 1e-6:  # Avoid division by zero
                    effective_val_split = self.val_split / (1.0 - self.test_split)

                if effective_val_split >= 1.0:
                    logger.info(
                        f"Calculated effective_val_split ({effective_val_split:.3f}) >= 1.0. Assigning all train_val to validation."
                    )
                    self.val_idx = train_val_idx
                    self.train_idx = pd.Index([])
                else:
                    self.train_idx, self.val_idx = train_test_split(
                        train_val_idx,
                        test_size=effective_val_split,
                        random_state=self.train_random_seed,
                        shuffle=True,
                    )
                    logger.info(
                        f"Split train_val ({len(train_val_idx)}) into train ({len(self.train_idx)}) and val ({len(self.val_idx)}) using seed {self.train_random_seed}, effective_val_split {effective_val_split:.3f}."
                    )
            elif len(train_val_idx) > 0:  # val_split is 0
                self.train_idx = train_val_idx
                self.val_idx = pd.Index([])
                logger.info("val_split is 0. All remaining train_val data used for train. Validation set is empty.")
            else:  # train_val_idx is empty
                self.train_idx = pd.Index([])
                self.val_idx = pd.Index([])
                logger.info("train_val_idx is empty. Train and Validation sets are empty.")
        else:  # attributes_df is None, so no basis for 'split' column or random splitting.
            # All data is effectively for prediction or a single use case if not test_all.
            logger.info("attributes_df is None. No 'split' column or random splitting applied. All data in full_idx.")
            if not self.test_all:  # If test_all is true, it's already handled.
                # If not test_all, and no other split info, assume all for train, no val/test.
                # This might need adjustment based on how predict_idx is handled later.
                self.train_idx = full_idx
                self.val_idx = pd.Index([])
                self.test_idx = pd.Index([])
                logger.info("Using all data for train_idx as attributes_df is None and not test_all.")
            # If self.test_all is True, self.test_idx is already full_idx, train/val are empty.

        logger.info(
            f"Final dataset sizes after splitting: Train={len(self.train_idx)}, Validation={len(self.val_idx)}, Test={len(self.test_idx)}"
        )

        # Determine if structure data should be used for dataset instances
        logger.info(
            f"Passing use_structure_for_this_dataset={self.actual_with_structure} to CompoundDataset instances."
        )

        # Helper function to create attributes for CompoundDataset when self.attributes_df is None
        def get_attributes_for_dataset(indices):
            if self.attributes_df is not None:
                return self.attributes_df.loc[indices]
            # If attributes_df is None, create a minimal DataFrame with just the index
            # CompoundDataset is expected to handle this for non-sequence tasks by using placeholders.
            if self.formula_df is not None:
                return pd.DataFrame(index=self.formula_df.loc[indices].index)
            return pd.DataFrame()

        def get_structure_for_dataset(indices):
            if self.actual_with_structure and self.structure_df is not None and not indices.empty:
                return self.structure_df.loc[indices]
            return None

        if stage == "fit" or stage is None:
            logger.info("--- Creating 'fit' stage datasets (train/val) ---")
            if not self.train_idx.empty:
                logger.info(f"Creating train_dataset with {len(self.train_idx)} samples.")
                self.train_dataset = CompoundDataset(
                    formula_desc=self.formula_df.loc[self.train_idx],
                    attributes=get_attributes_for_dataset(self.train_idx),
                    task_configs=self.task_configs,
                    structure_desc=get_structure_for_dataset(self.train_idx),
                    use_structure_for_this_dataset=self.actual_with_structure,
                    task_masking_ratios=self.task_masking_ratios,
                    is_predict_set=False,
                    dataset_name="train_dataset",
                )
            else:
                logger.warning("Train index is empty. train_dataset will be None.")
                self.train_dataset = None

            if not self.val_idx.empty:
                logger.info(f"Creating val_dataset with {len(self.val_idx)} samples.")
                self.val_dataset = CompoundDataset(
                    formula_desc=self.formula_df.loc[self.val_idx],
                    attributes=get_attributes_for_dataset(self.val_idx),
                    task_configs=self.task_configs,
                    structure_desc=get_structure_for_dataset(self.val_idx),
                    use_structure_for_this_dataset=self.actual_with_structure,
                    task_masking_ratios=None,  # No masking for validation
                    is_predict_set=False,
                    dataset_name="val_dataset",
                )
            else:
                logger.info("Validation index is empty. val_dataset will be None.")
                self.val_dataset = None

        if stage == "test" or stage is None:
            logger.info("--- Creating 'test' stage dataset ---")
            if not self.test_idx.empty:
                logger.info(f"Creating test_dataset with {len(self.test_idx)} samples.")
                self.test_dataset = CompoundDataset(
                    formula_desc=self.formula_df.loc[self.test_idx],
                    attributes=get_attributes_for_dataset(self.test_idx),
                    task_configs=self.task_configs,
                    structure_desc=get_structure_for_dataset(self.test_idx),
                    use_structure_for_this_dataset=self.actual_with_structure,
                    task_masking_ratios=None,  # No masking for test
                    is_predict_set=False,  # Typically False for test, model might output targets for metrics
                    dataset_name="test_dataset",
                )
            else:
                logger.info("Test index is empty. test_dataset will be None.")
                self.test_dataset = None

        if stage == "predict":
            logger.info("--- Creating 'predict' stage dataset ---")
            if self.init_predict_idx is not None:
                # Validate user-provided predict_idx against the formula_df's master index
                valid_predict_indices = self.init_predict_idx.intersection(self.formula_df.index)
                if len(valid_predict_indices) != len(self.init_predict_idx):
                    logger.warning(
                        f"User-provided predict_idx contains indices not found in the loaded formula_df. "
                        f"Original: {len(self.init_predict_idx)}, Valid: {len(valid_predict_indices)}. Using valid subset."
                    )
                self.predict_idx = valid_predict_indices
                if self.predict_idx.empty:
                    logger.warning(
                        "User-provided predict_idx resulted in an empty set after validation. Predict_dataset will be None."
                    )
            else:  # init_predict_idx was None, derive from test_idx or full_idx
                self.predict_idx = self.test_idx if not self.test_idx.empty else full_idx

            logger.info(
                f"Using predict_idx with {len(self.predict_idx)} samples (derived from {'user input, ' if self.init_predict_idx is not None else ''}test_idx or full_idx)."
            )

            if not self.predict_idx.empty:
                self.predict_dataset = CompoundDataset(
                    formula_desc=self.formula_df.loc[self.predict_idx],
                    attributes=get_attributes_for_dataset(self.predict_idx),
                    task_configs=self.task_configs,
                    structure_desc=get_structure_for_dataset(self.predict_idx),
                    use_structure_for_this_dataset=self.actual_with_structure,
                    task_masking_ratios=None,  # No masking for predict
                    is_predict_set=True,
                    dataset_name="predict_dataset",
                )
            else:
                logger.warning("Predict index is empty after processing. predict_dataset will be None.")
                self.predict_dataset = None
        logger.info(f"--- DataModule setup for stage '{stage}' complete ---")

    def train_dataloader(self):
        if not hasattr(self, "train_dataset") or self.train_dataset is None:
            logger.warning("train_dataloader: Train dataset is None or not initialized. Returning None.")
            return None
        if len(self.train_dataset) == 0:
            logger.warning("train_dataloader: Train dataset is empty (length 0). Returning None.")
            return None

        # Create custom collate function for ExtendRegression tasks
        collate_fn = create_collate_fn_with_task_info(self.task_configs)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset") or self.val_dataset is None:
            logger.info("val_dataloader: Validation dataset is None or not initialized. Returning None.")
            return None
        if len(self.val_dataset) == 0:
            logger.info("val_dataloader: Validation dataset is empty (length 0). Returning None.")
            return None

        # Create custom collate function for ExtendRegression tasks
        collate_fn = create_collate_fn_with_task_info(self.task_configs)

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        if not hasattr(self, "test_dataset") or self.test_dataset is None:
            logger.info("test_dataloader: Test dataset is None or not initialized. Returning None.")
            return None
        if len(self.test_dataset) == 0:
            logger.info("test_dataloader: Test dataset is empty (length 0). Returning None.")
            return None

        # Create custom collate function for ExtendRegression tasks
        collate_fn = create_collate_fn_with_task_info(self.task_configs)

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        if not hasattr(self, "predict_dataset") or self.predict_dataset is None:
            logger.info("predict_dataloader: Predict dataset is None or not initialized. Returning None.")
            return None
        if len(self.predict_dataset) == 0:
            logger.info("predict_dataloader: Predict dataset is empty (length 0). Returning None.")
            return None

        # Create custom collate function for ExtendRegression tasks
        collate_fn = create_collate_fn_with_task_info(self.task_configs)

        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
