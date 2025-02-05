import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .dataset import CompoundDataset


class CompoundDataModule(L.LightningDataModule):
    def __init__(
        self,
        descriptor: pd.DataFrame,
        attributes: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray = None,
        predict_idx: np.ndarray = None,
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
        train_idx : np.ndarray
            Indices for training data
        val_idx : np.ndarray
            Indices for validation data
        test_idx : np.ndarray, optional
            Indices for test data. If None, will use validation dataset for testing
        predict_idx : np.ndarray, optional
            Indices for prediction data. If None, will use test dataset for prediction
        batch_size : int, optional
            Batch size for dataloaders, by default 32
        num_workers : int, optional
            Number of workers for dataloaders, by default 0
        """
        super().__init__()
        self.descriptor = descriptor
        self.attributes = attributes
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.predict_idx = predict_idx
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        # Create train dataset
        self.train_dataset = CompoundDataset(
            self.descriptor.iloc[self.train_idx],
            self.attributes.iloc[self.train_idx],
        )

        # Create validation dataset if validation indices are provided
        if len(self.val_idx) > 0:
            self.val_dataset = CompoundDataset(
                self.descriptor.iloc[self.val_idx],
                self.attributes.iloc[self.val_idx],
            )

        # Create test dataset only if test indices are provided
        if self.test_idx is not None:
            self.test_dataset = CompoundDataset(
                self.descriptor.iloc[self.test_idx],
                self.attributes.iloc[self.test_idx],
            )

        # Create prediction dataset only if prediction indices are provided
        if self.predict_idx is not None:
            self.predict_dataset = CompoundDataset(
                self.descriptor.iloc[self.predict_idx],
                self.attributes.iloc[self.predict_idx],
            )

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
        elif hasattr(self, "val_dataset"):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return None

    def predict_dataloader(self):
        if hasattr(self, "predict_dataset"):
            return DataLoader(
                self.predict_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif hasattr(self, "test_dataset"):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif hasattr(self, "val_dataset"):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return None
