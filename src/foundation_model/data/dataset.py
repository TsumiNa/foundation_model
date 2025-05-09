from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CompoundDataset(Dataset):
    def __init__(
        self,
        descriptor: pd.DataFrame,
        attributes: pd.DataFrame,
        task_configs: List,
        temps: torch.Tensor = None,  # Temperature points for sequence tasks
    ):
        """
        Multi-task compound dataset.

        Parameters
        ----------
        descriptor : pd.DataFrame
            Input features for the compounds
        attributes : pd.DataFrame
            Target attributes for the compounds
        task_configs : List
            List of task configurations defining all tasks to be processed
        temps : torch.Tensor, optional
            Temperature points for sequence prediction tasks, expected shape (seq_len,) or (1, seq_len)
        """
        # Basic validation
        if not descriptor.index.equals(attributes.index):
            raise ValueError("descriptor and attributes must have matching indices")

        # Store temperature points
        self.temps = temps

        # Filter enabled tasks and organize by type
        self.task_configs = {}
        self.task_names = []

        # Group tasks by type
        self.regression_tasks = []
        self.classification_tasks = []
        self.sequence_tasks = []

        for cfg in task_configs:
            if not hasattr(cfg, "enabled") or not cfg.enabled:
                continue

            name = cfg.name
            self.task_configs[name] = cfg
            self.task_names.append(name)

            # Categorize tasks by type
            if hasattr(cfg, "type"):
                if cfg.type.name == "REGRESSION":
                    self.regression_tasks.append(name)
                elif cfg.type.name == "CLASSIFICATION":
                    self.classification_tasks.append(name)
                elif cfg.type.name == "SEQUENCE":
                    self.sequence_tasks.append(name)

        # Check which task columns exist in attributes
        existing_tasks = [task for task in self.task_names if task in attributes.columns]
        if not existing_tasks:
            raise ValueError("No enabled task columns found in attributes DataFrame")

        # Drop rows where all task values are NaN
        valid_mask = attributes[existing_tasks].notna().any(axis=1)
        if not valid_mask.all():
            descriptor = descriptor[valid_mask]
            attributes = attributes[valid_mask]

        # Convert descriptor data
        self.x = torch.tensor(descriptor.values, dtype=torch.float32)

        # Build y_dict and task_masks
        self.y_dict = {}
        task_masks_list = []

        # Process each task
        for task_name in self.task_names:
            if task_name not in attributes.columns:
                # Create empty placeholder for missing tasks
                self.y_dict[task_name] = torch.zeros((len(self.x), 1), dtype=torch.float32)
                task_masks_list.append(np.zeros((len(self.x), 1), dtype=np.float32))
                continue

            # Get task values and create mask
            task_values = attributes[task_name].values

            # Handle different task types
            if task_name in self.sequence_tasks:
                # For sequence tasks, special handling might be needed
                # This is a placeholder - actual implementation depends on your data format
                if isinstance(task_values[0], (list, np.ndarray)) or (
                    hasattr(task_values[0], "shape") and len(task_values[0].shape) > 0
                ):
                    # Already sequence data
                    pass
                else:
                    # Convert to sequence format if needed
                    task_values = task_values.reshape(-1, 1)

                mask = (~np.isnan(task_values).all(axis=1)).astype(np.float32).reshape(-1, 1)
            else:
                # For regression/classification tasks
                task_values = task_values.reshape(-1, 1)
                mask = (~np.isnan(task_values)).astype(np.float32)

            # Clean up NaN values
            cleaned_values = np.nan_to_num(task_values, nan=0.0)

            # Store data and masks
            self.y_dict[task_name] = torch.tensor(cleaned_values, dtype=torch.float32)
            task_masks_list.append(mask)

        # Combine all task masks into one tensor
        self.task_masks = torch.tensor(np.hstack(task_masks_list), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        Get a data sample.

        Returns
        -------
        tuple
            (x, y_dict, task_masks, temps)
            - x: Input features
            - y_dict: Dictionary with all task labels
            - task_masks: Task masks indicating which tasks have valid data
            - temps: Temperature points (if applicable)
        """
        # Build y_dict for this sample
        sample_y_dict = {task_name: self.y_dict[task_name][idx] for task_name in self.task_names}

        return self.x[idx], sample_y_dict, self.task_masks[idx], self.temps

    @property
    def attribute_names(self) -> list:
        """
        Get the list of attribute names.

        Returns
        -------
        list
            List of task names in the dataset.
        """
        return self.task_names.copy()
