from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CompoundDataset(Dataset):
    def __init__(
        self,
        descriptor: pd.DataFrame,
        attributes: pd.DataFrame,
        filter_attributes: bool = False,
        **attribute_rates,
    ):
        """
        Custom dataset for compounds.

        Parameters
        ----------
        descriptor : pd.DataFrame
            Input features for the compounds
        attributes : pd.DataFrame
            Target attributes for the compounds
        filter_attributes : bool, optional
            If True, only keeps attributes specified in attribute_rates.
            If False (default), keeps all attributes and only applies masking.
        attribute_rates : dict
            Dictionary specifying what fraction of data to use for each attribute
            e.g., {"attribute_name": 0.8} means use 80% of available data for that attribute
            Setting rate to 0 will remove that attribute from the dataset
        """
        # Basic validation
        if not descriptor.index.equals(attributes.index):
            raise ValueError("descriptor and attributes must have matching indices")

        initial_attributes = list(attributes.columns)
        if not initial_attributes:
            raise ValueError("attributes DataFrame must have at least one column")

        # Process attribute rates and determine which attributes to keep
        if attribute_rates:
            # Validate rates
            invalid_attrs = set(attribute_rates.keys()) - set(initial_attributes)
            if invalid_attrs:
                raise ValueError(
                    f"Invalid attributes in rates: {invalid_attrs}. "
                    f"Valid attributes are: {initial_attributes}"
                )

            invalid_rates = [
                (attr, rate)
                for attr, rate in attribute_rates.items()
                if not 0 <= rate <= 1
            ]
            if invalid_rates:
                raise ValueError(
                    f"Rates must be between 0 and 1. Invalid values: {invalid_rates}"
                )

            if filter_attributes:
                # Only keep attributes specified in attribute_rates
                self.attributes = list(attribute_rates.keys())
                attributes = attributes[self.attributes]
            else:
                # Keep all attributes but apply masking
                self.attributes = initial_attributes

            # Initialize rates with defaults (1.0) for attributes not in attribute_rates
            self._attribute_rates = {attr: 1.0 for attr in self.attributes}
            # Update with provided rates
            self._attribute_rates.update(attribute_rates)

            # Remove attributes with zero rates
            zero_rate_attrs = [
                attr for attr, rate in self._attribute_rates.items() if rate == 0
            ]
            if zero_rate_attrs:
                keep_attrs = [
                    attr for attr in self.attributes if attr not in zero_rate_attrs
                ]
                attributes = attributes[keep_attrs]
                self.attributes = keep_attrs
                self._attribute_rates = {
                    attr: rate
                    for attr, rate in self._attribute_rates.items()
                    if rate > 0
                }
        else:
            self.attributes = initial_attributes
            self._attribute_rates = {attr: 1.0 for attr in self.attributes}

        # Convert data and create masks
        self.x = descriptor.values
        self.y = attributes.values.astype(np.float32)
        self.mask = (~np.isnan(self.y)).astype(int)

        # Apply rates to mask
        for attr_name, rate in self._attribute_rates.items():
            if rate < 1.0:  # Only process attributes that need data reduction
                attr_idx = self.attributes.index(attr_name)
                valid_indices = np.where(self.mask[:, attr_idx])[0]
                if len(valid_indices) > 0:
                    num_to_use = int(len(valid_indices) * rate)
                    keep_indices = np.random.choice(
                        valid_indices, num_to_use, replace=False
                    )
                    mask_indices = np.setdiff1d(valid_indices, keep_indices)
                    self.mask[mask_indices, attr_idx] = 0

        # Clean up invalid columns (attributes with all zeros in mask)
        valid_columns = np.any(self.mask, axis=0)
        if not np.all(valid_columns):
            self.y = self.y[:, valid_columns]
            self.mask = self.mask[:, valid_columns]
            self.attributes = [
                attr for i, attr in enumerate(self.attributes) if valid_columns[i]
            ]
            self._attribute_rates = {
                attr: rate
                for attr, rate in self._attribute_rates.items()
                if attr in self.attributes
            }

        # Clean up invalid rows (samples with all zeros in mask)
        valid_rows = np.any(self.mask, axis=1)
        if not np.all(valid_rows):
            self.x = self.x[valid_rows]
            self.y = self.y[valid_rows]
            self.mask = self.mask[valid_rows]

        # Final processing
        self.y = np.nan_to_num(self.y)
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.mask = torch.tensor(self.mask, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx], self.attributes

    @property
    def rates(self) -> Dict[str, float]:
        """
        Get the rate of data used for each attribute.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the rate of data used for each attribute,
            where 1.0 means using all available data and 0.5 means using half.
        """
        return self._attribute_rates.copy()
