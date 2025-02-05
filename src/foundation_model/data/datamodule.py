from typing import Callable, Dict

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from .dataset import CompoundDataset
from .splitter import MultiTaskSplitter


class CompoundDataModule(L.LightningDataModule):
    def __init__(
        self,
        descriptor: pd.DataFrame,
        attributes: pd.DataFrame,
        splitter: Callable,
        attribute_rates: Dict[str, float],
        filter_attributes: bool = True,
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
            Target attributes for the compounds
        splitter : Callable
            Function to split data into train/val/test sets
        attribute_rates : Dict[str, float]
            Dictionary specifying what fraction of data to use for each attribute
            e.g., {"attribute_name": 0.8} means use 80% of available data for that attribute
        filter_attributes : bool, optional
            If True (default), only keeps attributes specified in attribute_rates.
            If False, keeps all attributes and only applies masking.
        batch_size : int, optional
            Batch size for dataloaders, by default 32
        num_workers : int, optional
            Number of workers for dataloaders, by default 0
        """
        super().__init__()
        self.descriptor = descriptor
        self.attributes = attributes
        self.attribute_rates = attribute_rates
        self.filter_attributes = filter_attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        if isinstance(splitter, MultiTaskSplitter):
            self.splitter = splitter
        else:
            # Default to MultiTaskSplitter if a custom splitter is not provided
            self.splitter = MultiTaskSplitter(train_ratio=0.9, val_ratio=0.1)

    def setup(self, stage: str = None):
        # Split indices using MultiTaskSplitter
        indices = self.splitter.split(self.attributes)
        if len(indices) < 2:
            raise ValueError("Splitter must return at least two sets of indices")
        if len(indices) == 2:
            train_indices, test_indices = indices
            val_indices = []
        else:
            train_indices, val_indices, test_indices = indices
        self.train_dataset = CompoundDataset(
            self.descriptor.iloc[train_indices],
            self.attributes.iloc[train_indices],
            filter_attributes=self.filter_attributes,
            **self.attribute_rates,
        )
        # Create validation dataset if validation indices are provided
        if len(val_indices) > 0:
            # For validation, only filter attributes but don't apply rates
            val_rates = {
                attr: 1.0 for attr, rate in self.attribute_rates.items() if rate > 0
            }
            self.val_dataset = CompoundDataset(
                self.descriptor.iloc[val_indices],
                self.attributes.iloc[val_indices],
                filter_attributes=self.filter_attributes,
                **val_rates,
            )

        # Create test dataset if test indices are provided
        if len(test_indices) > 0:
            # For testing, only filter attributes but don't apply rates
            test_rates = {
                attr: 1.0 for attr, rate in self.attribute_rates.items() if rate > 0
            }
            self.test_dataset = CompoundDataset(
                self.descriptor.iloc[test_indices],
                self.attributes.iloc[test_indices],
                filter_attributes=self.filter_attributes,
                **test_rates,
            )
        else:
            # When using validation set as test set, only filter attributes but don't apply rates
            test_rates = {
                attr: 1.0 for attr, rate in self.attribute_rates.items() if rate > 0
            }
            self.test_dataset = CompoundDataset(
                self.descriptor.iloc[val_indices],
                self.attributes.iloc[val_indices],
                filter_attributes=self.filter_attributes,
                **test_rates,
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
        return None

    def predict_dataloader(self):
        if hasattr(self, "test_dataset"):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return None
