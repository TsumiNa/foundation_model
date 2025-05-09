# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

from typing import List

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import CompoundDataset


class CompoundDataModule(L.LightningDataModule):
    def __init__(
        self,
        descriptor: pd.DataFrame,
        attributes: pd.DataFrame,
        task_configs: List,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray = None,
        predict_idx: np.ndarray = None,
        temps: np.ndarray = None,  # Temperature points for sequence tasks
        batch_size=32,
        num_workers=0,
    ):
        """
        Initialize the data module.

        Parameters
        ----------
        descriptor : pd.DataFrame
            Input features for the compounds
        attributes : pd.DataFrame
            Target attributes for the compounds (already preprocessed)
        task_configs : List
            List of task configurations defining all tasks to be processed
        train_idx : np.ndarray
            Indices for training data
        val_idx : np.ndarray
            Indices for validation data
        test_idx : np.ndarray, optional
            Indices for test data. If None, will use validation dataset for testing
        predict_idx : np.ndarray, optional
            Indices for prediction data. If None, will use test dataset for prediction
        temps : np.ndarray, optional
            Temperature points for sequence prediction tasks
        batch_size : int, optional
            Batch size for dataloaders, by default 32
        num_workers : int, optional
            Number of workers for dataloaders, by default 0
        """
        super().__init__()
        self.descriptor = descriptor
        self.attributes = attributes
        self.task_configs = task_configs
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.predict_idx = predict_idx
        self.temps = torch.tensor(temps, dtype=torch.float32) if temps is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            # Create train dataset
            self.train_dataset = CompoundDataset(
                self.descriptor.iloc[self.train_idx],
                self.attributes.iloc[self.train_idx],
                self.task_configs,
                temps=self.temps,
            )

            # Create validation dataset if validation indices are provided
            if len(self.val_idx) > 0:
                self.val_dataset = CompoundDataset(
                    self.descriptor.iloc[self.val_idx],
                    self.attributes.iloc[self.val_idx],
                    self.task_configs,
                    temps=self.temps,
                )
            else:
                self.val_dataset = None

        if stage == "test" or stage is None:
            # Create test dataset only if test indices are provided
            if self.test_idx is not None and len(self.test_idx) > 0:
                self.test_dataset = CompoundDataset(
                    self.descriptor.iloc[self.test_idx],
                    self.attributes.iloc[self.test_idx],
                    self.task_configs,
                    temps=self.temps,
                )
            elif hasattr(self, "val_dataset") and self.val_dataset is not None:
                self.test_dataset = self.val_dataset
            else:
                self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if hasattr(self, "val_dataset"):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return None

    def test_dataloader(self):
        if hasattr(self, "test_dataset"):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return None
