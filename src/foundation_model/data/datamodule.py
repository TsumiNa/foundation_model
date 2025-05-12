# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, List, Optional, Union

import lightning as L
import numpy as np
import pandas as pd

# import torch # Not directly used in this file anymore for temps
from sklearn.model_selection import train_test_split  # For data splitting
from torch.utils.data import DataLoader

from .dataset import CompoundDataset

logger = logging.getLogger(__name__)


class CompoundDataModule(L.LightningDataModule):
    def __init__(
        self,
        formula_desc_source: Union[pd.DataFrame, np.ndarray, str],
        attributes_source: Union[pd.DataFrame, str],
        task_configs: List,
        structure_desc_source: Optional[Union[pd.DataFrame, np.ndarray, str]] = None,
        with_structure: bool = False,
        task_masking_ratios: Optional[Dict[str, float]] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        train_random_seed: int = 42,
        test_random_seed: int = 24,
        test_all: bool = False,
        # predict_idx: np.ndarray = None, # predict_idx will be derived from test_idx or all data
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        """
        Initialize the data module.

        Parameters
        ----------
        formula_desc_source : Union[pd.DataFrame, np.ndarray, str]
            Formula descriptors (DataFrame, NumPy array, or path to pickle/CSV).
        attributes_source : Union[pd.DataFrame, str]
            Task target attributes, sequence data, temps, and 'split' column (DataFrame or path).
        task_configs : List
            List of task configurations.
        structure_desc_source : Optional[Union[pd.DataFrame, np.ndarray, str]], optional
            Structure descriptors. Defaults to None.
        with_structure : bool, optional
            If True, attempt to load and use structure descriptors. Defaults to False.
        task_masking_ratios : Optional[Dict[str, float]], optional
            Ratios for random masking per task in training set. Defaults to None.
        val_split : float, optional
            Proportion of non-test data to use for validation. Defaults to 0.1.
        test_split : float, optional
            Proportion of data to use for testing. Defaults to 0.1.
        train_random_seed : int, optional
            Random seed for splitting train and validation sets. Defaults to 42.
        test_random_seed : int, optional
            Random seed for splitting test set from the rest. Defaults to 24.
        test_all : bool, optional
            If True, all data is used for the test set. Defaults to False.
        batch_size : int, optional
            Batch size for dataloaders. Defaults to 32.
        num_workers : int, optional
            Number of workers for dataloaders. Defaults to 0.
        """
        super().__init__()
        logger.info("Initializing CompoundDataModule...")
        self.save_hyperparameters(ignore=["formula_desc_source", "attributes_source", "structure_desc_source"])

        # --- Load Data ---
        logger.info("--- Loading Data ---")
        self.formula_df = self._load_data(formula_desc_source, "formula_desc")
        self.attributes_df = self._load_data(attributes_source, "attributes")

        if self.formula_df is None or self.attributes_df is None:
            # This check is if _load_data returned None (e.g. file not found)
            raise ValueError(
                "formula_desc_source and attributes_source must be successfully loaded (e.g., file not found or unreadable)."
            )

        if self.formula_df.empty or self.attributes_df.empty:
            # This check is if the loaded DataFrames are empty
            raise ValueError("Formula and attributes DataFrames cannot be empty after loading.")

        self.structure_df = None
        self.actual_with_structure = False  # Tracks if structure data is actually used
        if self.hparams.with_structure:
            logger.info("Attempting to load structure_desc as with_structure is True.")
            if structure_desc_source is not None:
                self.structure_df = self._load_data(structure_desc_source, "structure_desc")
                if self.structure_df is not None:
                    if len(self.structure_df) != len(self.attributes_df):
                        logger.warning(
                            f"Length mismatch: attributes_df ({len(self.attributes_df)}) vs structure_df ({len(self.structure_df)}). "
                            "Structure data will NOT be used."
                        )
                        self.structure_df = None
                    else:
                        self.actual_with_structure = True  # Tentatively True
                else:  # structure_df is None after loading attempt
                    logger.warning("Failed to load structure_desc_source. Proceeding without structure data.")
            else:  # structure_desc_source is None
                logger.warning(
                    "with_structure is True, but structure_desc_source is None. Proceeding without structure data."
                )
        else:
            logger.info("with_structure is False. Structure data will not be loaded or used.")

        logger.info(
            f"Initial loaded formula_df length: {len(self.formula_df)}, attributes_df length: {len(self.attributes_df)}"
        )
        if self.structure_df is not None:
            logger.info(f"Initial loaded structure_df length: {len(self.structure_df)}")

        # --- Align DataFrames ---
        logger.info("--- Aligning DataFrames by attributes_df index ---")
        common_index = self.attributes_df.index

        original_formula_len = len(self.formula_df)
        self.formula_df = self.formula_df.reindex(common_index)  # Use reindex to handle missing indices with NaNs
        if len(self.formula_df) != original_formula_len or self.formula_df.isnull().values.any():
            logger.warning(
                f"Formula_df reindexed. Original length: {original_formula_len}, new length: {len(self.formula_df)}. Check for NaNs if lengths differ or if unexpected."
            )
        # Drop rows that became all NaN after reindexing (if any, though attributes_df is the master)
        self.formula_df.dropna(how="all", inplace=True)

        # Re-align attributes_df with the potentially modified formula_df index (after its own dropna)
        # This ensures both are clean before structure_df alignment
        common_index = self.attributes_df.index.intersection(self.formula_df.index)
        self.attributes_df = self.attributes_df.loc[common_index]
        self.formula_df = self.formula_df.loc[common_index]

        if self.actual_with_structure and self.structure_df is not None:
            original_struct_len = len(self.structure_df)
            self.structure_df = self.structure_df.reindex(common_index)
            if len(self.structure_df) != original_struct_len or self.structure_df.isnull().values.any():
                logger.warning(
                    f"Structure_df reindexed. Original length: {original_struct_len}, new length: {len(self.structure_df)}. Check for NaNs."
                )
            self.structure_df.dropna(how="all", inplace=True)

            # Final check for structure alignment
            final_common_index = self.attributes_df.index.intersection(self.structure_df.index)
            if len(final_common_index) != len(self.attributes_df):
                logger.warning(
                    f"After reindexing and dropna, structure_df index does not fully match attributes_df. "
                    f"Attributes length: {len(self.attributes_df)}, Structure length: {len(self.structure_df)}, "
                    f"Intersection length: {len(final_common_index)}. Structure data will NOT be used."
                )
                self.structure_df = None
                self.actual_with_structure = False
            else:
                self.attributes_df = self.attributes_df.loc[final_common_index]
                self.formula_df = self.formula_df.loc[final_common_index]
                self.structure_df = self.structure_df.loc[final_common_index]
        else:  # Structure was not loaded or deemed unusable earlier
            self.actual_with_structure = False
            self.structure_df = None

        logger.info(f"Final aligned formula_df length: {len(self.formula_df)}")
        logger.info(f"Final aligned attributes_df length: {len(self.attributes_df)}")
        if self.structure_df is not None:
            logger.info(f"Final aligned structure_df length: {len(self.structure_df)}")
        logger.info(f"Final actual_with_structure status: {self.actual_with_structure}")

        self.task_configs = task_configs
        logger.info(f"DataModule initialized with {len(self.task_configs)} task configurations.")

    def _load_data(self, source: Union[pd.DataFrame, np.ndarray, str], name: str) -> Optional[pd.DataFrame]:
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
                    df = pd.read_pickle(source)
                elif source.endswith(".csv"):
                    df = pd.read_csv(source, index_col=0)  # Assuming first column as index
                else:
                    logger.error(f"Unsupported file type for '{name}': {source}. Must be .pkl or .csv.")
                    raise ValueError(f"Unsupported file type for {name}: {source}.")
            elif isinstance(source, np.ndarray):
                logger.info(f"Converting '{name}' data from np.ndarray to pd.DataFrame.")
                df = pd.DataFrame(source)
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
        except Exception as e:
            logger.error(f"Error loading '{name}' data from {source}: {e}", exc_info=True)
            return None

    def setup(self, stage: str = None):
        """Prepare datasets for different stages (fit, test, predict)."""
        logger.info(f"--- Setting up DataModule for stage: {stage} ---")
        if self.attributes_df is None or self.formula_df is None:
            logger.error("attributes_df or formula_df is None. Cannot proceed with setup.")
            raise ValueError("attributes_df or formula_df not loaded properly.")

        full_idx = self.attributes_df.index
        logger.info(f"Total samples available before splitting: {len(full_idx)}")

        if self.hparams.test_all:
            self.train_idx = pd.Index([])
            self.val_idx = pd.Index([])
            self.test_idx = full_idx
            logger.info("Data split strategy: Using all data for testing (test_all=True).")
        elif "split" in self.attributes_df.columns:
            logger.info("Data split strategy: Using 'split' column from attributes_df.")
            split_counts = self.attributes_df["split"].value_counts()
            logger.info(f"Value counts in 'split' column: {split_counts.to_dict()}")
            if not all(s_type in ["train", "val", "test"] for s_type in split_counts.index):
                logger.error(f"Invalid values in 'split' column: {split_counts.index.tolist()}")
                raise ValueError("Invalid values in 'split' column. Must be 'train', 'val', or 'test'.")

            self.train_idx = self.attributes_df[self.attributes_df["split"] == "train"].index
            self.val_idx = self.attributes_df[self.attributes_df["split"] == "val"].index
            self.test_idx = self.attributes_df[self.attributes_df["split"] == "test"].index

            if self.val_idx.empty and not self.train_idx.empty:
                logger.warning("No 'val' samples in 'split' column. Splitting 10% of 'train' for validation.")
                self.train_idx, self.val_idx = train_test_split(
                    self.train_idx, test_size=0.1, random_state=self.hparams.train_random_seed, shuffle=True
                )
        else:
            logger.info("Data split strategy: Performing random train/val/test splits.")
            logger.info(
                f"Test split ratio: {self.hparams.test_split}, Validation split ratio (of non-test): {self.hparams.val_split}"
            )
            if self.hparams.test_split > 0:
                train_val_idx, self.test_idx = train_test_split(
                    full_idx,
                    test_size=self.hparams.test_split,
                    random_state=self.hparams.test_random_seed,
                    shuffle=True,
                )
                logger.info(
                    f"Split full data ({len(full_idx)}) into train_val ({len(train_val_idx)}) and test ({len(self.test_idx)}) using seed {self.hparams.test_random_seed}."
                )
            else:
                train_val_idx = full_idx
                self.test_idx = pd.Index([])
                logger.info("test_split is 0. All data used for train_val.")

            if self.hparams.val_split > 0 and len(train_val_idx) > 0:
                # Adjust val_split relative to the size of train_val_idx
                effective_val_split = self.hparams.val_split
                if (1.0 - self.hparams.test_split) > 1e-6:  # Avoid division by zero if test_split is ~1.0
                    effective_val_split = self.hparams.val_split / (1.0 - self.hparams.test_split)

                if effective_val_split >= 1.0:  # Safety net if val_split is too large relative to remaining data
                    logger.warning(
                        f"Calculated effective_val_split ({effective_val_split:.3f}) is >= 1.0. Capping at 0.1 of train_val data."
                    )
                    effective_val_split = 0.1

                self.train_idx, self.val_idx = train_test_split(
                    train_val_idx,
                    test_size=effective_val_split,
                    random_state=self.hparams.train_random_seed,
                    shuffle=True,
                )
                logger.info(
                    f"Split train_val data ({len(train_val_idx)}) into train ({len(self.train_idx)}) and val ({len(self.val_idx)}) using seed {self.hparams.train_random_seed} and effective_val_split {effective_val_split:.3f}."
                )
            elif len(train_val_idx) > 0:
                self.train_idx = train_val_idx
                self.val_idx = pd.Index([])
                logger.info(
                    "val_split is 0 or train_val_idx was empty. All remaining train_val data used for train. Validation set is empty."
                )
            else:  # train_val_idx is empty
                self.train_idx = pd.Index([])
                self.val_idx = pd.Index([])
                logger.info("train_val_idx is empty. Train and Validation sets are empty.")

        logger.info(
            f"Final dataset sizes: Train={len(self.train_idx)}, Validation={len(self.val_idx)}, Test={len(self.test_idx)}"
        )

        # Determine if structure data should be used for dataset instances
        # self.actual_with_structure was set during __init__
        current_structure_df_for_dataset = self.structure_df if self.actual_with_structure else None
        logger.info(
            f"Passing use_structure_for_this_dataset={self.actual_with_structure} to CompoundDataset instances."
        )

        if stage == "fit" or stage is None:
            logger.info("--- Creating 'fit' stage datasets (train/val) ---")
            if not self.train_idx.empty:
                logger.info(f"Creating train_dataset with {len(self.train_idx)} samples.")
                self.train_dataset = CompoundDataset(
                    formula_desc=self.formula_df.loc[self.train_idx],
                    attributes=self.attributes_df.loc[self.train_idx],
                    task_configs=self.task_configs,
                    structure_desc=current_structure_df_for_dataset.loc[self.train_idx]
                    if current_structure_df_for_dataset is not None
                    else None,
                    use_structure_for_this_dataset=self.actual_with_structure,
                    task_masking_ratios=self.hparams.task_masking_ratios,
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
                    attributes=self.attributes_df.loc[self.val_idx],
                    task_configs=self.task_configs,
                    structure_desc=current_structure_df_for_dataset.loc[self.val_idx]
                    if current_structure_df_for_dataset is not None
                    else None,
                    use_structure_for_this_dataset=self.actual_with_structure,
                    task_masking_ratios=None,
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
                    attributes=self.attributes_df.loc[self.test_idx],
                    task_configs=self.task_configs,
                    structure_desc=current_structure_df_for_dataset.loc[self.test_idx]
                    if current_structure_df_for_dataset is not None
                    else None,
                    use_structure_for_this_dataset=self.actual_with_structure,
                    task_masking_ratios=None,
                    is_predict_set=False,
                    dataset_name="test_dataset",
                )
            else:
                logger.info("Test index is empty. test_dataset will be None.")
                self.test_dataset = None

        if stage == "predict":
            logger.info("--- Creating 'predict' stage dataset ---")
            self.predict_idx = self.test_idx if not self.test_idx.empty else full_idx
            logger.info(f"Using predict_idx with {len(self.predict_idx)} samples (derived from test_idx or full_idx).")
            if not self.predict_idx.empty:
                self.predict_dataset = CompoundDataset(
                    formula_desc=self.formula_df.loc[self.predict_idx],
                    attributes=self.attributes_df.loc[self.predict_idx],
                    task_configs=self.task_configs,
                    structure_desc=current_structure_df_for_dataset.loc[self.predict_idx]
                    if current_structure_df_for_dataset is not None
                    else None,
                    use_structure_for_this_dataset=self.actual_with_structure,
                    task_masking_ratios=None,
                    is_predict_set=True,
                    dataset_name="predict_dataset",
                )
            else:
                logger.warning("Predict index is empty. predict_dataset will be None.")
                self.predict_dataset = None
        logger.info(f"--- DataModule setup for stage '{stage}' complete ---")

    def train_dataloader(self):
        if not hasattr(self, "train_dataset") or self.train_dataset is None or len(self.train_dataset) == 0:
            logger.warning("train_dataloader: Train dataset is empty or not initialized. Returning None.")
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,  # Added for potential speedup
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset") or self.val_dataset is None or len(self.val_dataset) == 0:
            logger.info("Validation dataset is empty or not initialized. Returning None for val_dataloader.")
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if not hasattr(self, "test_dataset") or self.test_dataset is None or len(self.test_dataset) == 0:
            logger.info("Test dataset is empty or not initialized. Returning None for test_dataloader.")
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        if not hasattr(self, "predict_dataset") or self.predict_dataset is None or len(self.predict_dataset) == 0:
            logger.info("Predict dataset is empty or not initialized. Returning None for predict_dataloader.")
            return None
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
