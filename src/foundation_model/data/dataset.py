import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CompoundDataset(Dataset):
    def __init__(
        self,
        descriptor: pd.DataFrame,
        attributes: pd.DataFrame,
    ):
        """
        Custom dataset for compounds.

        Parameters
        ----------
        descriptor : pd.DataFrame
            Input features for the compounds
        attributes : pd.DataFrame
            Target attributes for the compounds
        """
        # Basic validation
        if not descriptor.index.equals(attributes.index):
            raise ValueError("descriptor and attributes must have matching indices")

        if not list(attributes.columns):
            raise ValueError("attributes DataFrame must have at least one column")

        # Convert data and create masks
        self.x = torch.tensor(descriptor.values, dtype=torch.float32)
        self.y = attributes.values.astype(np.float32)
        self.mask = (~np.isnan(self.y)).astype(int)

        # Clean up nan values
        self.y = np.nan_to_num(self.y)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.mask = torch.tensor(self.mask, dtype=torch.float32)

        self._attribute_names = list(attributes.columns)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx], self._attribute_names

    @property
    def attribute_names(self) -> list:
        """
        Get the list of attribute names.

        Returns
        -------
        list
            List of attribute names in the dataset.
        """
        return self._attribute_names.copy()
