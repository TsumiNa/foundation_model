from typing import Dict

import numpy as np
import pandas as pd


class AttributePreprocessor:
    def __init__(self, attribute_rates: Dict[str, float]):
        """
        Initialize the preprocessor with attribute rates.

        Parameters
        ----------
        attribute_rates : Dict[str, float]
            Dictionary specifying what fraction of data to use for each attribute
            e.g., {"attribute_name": 0.8} means use 80% of available data for that attribute
            Setting rate to 0 will remove that attribute
        """
        self.attribute_rates = attribute_rates

    def process(self, attributes: pd.DataFrame) -> pd.DataFrame:
        """
        Process attributes by filtering and applying rates.

        Parameters
        ----------
        attributes : pd.DataFrame
            Target attributes for the compounds

        Returns
        -------
        pd.DataFrame
            Processed attributes with filtering and rates applied
        """
        # Validate attributes exist in DataFrame
        invalid_attrs = set(self.attribute_rates.keys()) - set(attributes.columns)
        if invalid_attrs:
            raise ValueError(
                f"Invalid attributes in rates: {invalid_attrs}. "
                f"Valid attributes are: {list(attributes.columns)}"
            )

        # Validate rates are between 0 and 1
        invalid_rates = [
            (attr, rate)
            for attr, rate in self.attribute_rates.items()
            if not 0 <= rate <= 1
        ]
        if invalid_rates:
            raise ValueError(
                f"Rates must be between 0 and 1. Invalid values: {invalid_rates}"
            )

        # Drop attributes with rate=0 or not in attribute_rates
        valid_attrs = [attr for attr, rate in self.attribute_rates.items() if rate > 0]
        processed = attributes[valid_attrs].copy()

        # Apply rates by masking data
        for attr, rate in self.attribute_rates.items():
            if rate < 1.0:
                mask = ~np.isnan(processed[attr])
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > 0:
                    num_to_use = int(len(valid_indices) * rate)
                    mask_indices = np.random.choice(
                        valid_indices, len(valid_indices) - num_to_use, replace=False
                    )
                    processed.loc[processed.index[mask_indices], attr] = np.nan

        return processed
