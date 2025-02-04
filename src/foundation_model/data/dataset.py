from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CompoundDataset(Dataset):
    def __init__(
        self,
        descriptor: pd.DataFrame,
        property: pd.DataFrame,
        **property_fractions,
    ):
        """
        Custom dataset for compounds.

        Parameters
        ----------
        descriptor : pd.DataFrame
            Input features for the compounds
        property : pd.DataFrame
            Target properties for the compounds
        property_fractions : dict
            Dictionary specifying what fraction of data to use for each property
            e.g., {"property_name": 0.8} means use 80% of available data for that property
        """
        # Ensure descriptor and property have matching indices
        if not descriptor.index.equals(property.index):
            raise ValueError("descriptor and property must have matching indices")

        # Get attributes from property columns
        self.attributes = list(property.columns)
        if not self.attributes:
            raise ValueError("property DataFrame must have at least one column")

        # Input features
        self.x = descriptor.values

        # Output attributes - select all columns from property
        self.y = property.values.astype(np.float32)

        # Create initial masks based on non-nan values
        self.mask = (~np.isnan(self.y)).astype(int)

        # Initialize property fractions with default values (use all available data)
        self._property_fractions = {attr: 1.0 for attr in self.attributes}

        # Validate and update property fractions if provided
        if property_fractions:
            # Validate attributes
            invalid_attrs = set(property_fractions.keys()) - set(self.attributes)
            if invalid_attrs:
                raise ValueError(
                    f"Invalid attributes in property_fractions: {invalid_attrs}. "
                    f"Valid attributes are: {self.attributes}"
                )

            # Validate percentages
            invalid_percents = [
                (attr, percent)
                for attr, percent in property_fractions.items()
                if not 0 <= percent <= 1
            ]
            if invalid_percents:
                raise ValueError(
                    "Percentages must be between 0 and 1. Invalid values: "
                    f"{invalid_percents}"
                )

            # Update with provided values
            self._property_fractions.update(property_fractions)

            # Apply fractions only to non-nan values
            for attr_name, fraction in self._property_fractions.items():
                attr_idx = self.attributes.index(attr_name)
                # Get indices where values are not nan
                valid_indices = np.where(~np.isnan(self.y[:, attr_idx]))[0]
                if len(valid_indices) > 0:  # Only proceed if there are valid values
                    # Calculate number of samples to use based on fraction
                    num_to_use = int(len(valid_indices) * fraction)
                    # Randomly select indices to keep
                    keep_indices = np.random.choice(
                        valid_indices, num_to_use, replace=False
                    )
                    # Mask all valid indices except those we keep
                    mask_indices = np.setdiff1d(valid_indices, keep_indices)
                    self.mask[mask_indices, attr_idx] = 0

        # Fill all nan to 0 after masking
        self.y = np.nan_to_num(self.y)

        # Convert to tensors
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.mask = torch.tensor(self.mask, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]

    @property
    def fractions(self) -> Dict[str, float]:
        """
        Get the fraction of data used for each property.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the fraction of data used for each property,
            where 1.0 means using all available data and 0.5 means using half.
        """
        return self._property_fractions.copy()
