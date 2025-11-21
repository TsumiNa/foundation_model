import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class MultiTaskSplitter:
    """
    A data splitter for multi-task learning that ensures all tasks have representation
    in train/val/test splits while handling NaN values and limited data scenarios.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the splitter with desired split ratios.

        Parameters
        ----------
        train_ratio : float, default=0.8
            Ratio of data to use for training
        val_ratio : float, default=0.1
            Ratio of data to use for validation
        test_ratio : float, default=0.1
            Ratio of data to use for testing
        random_state : int, optional
            Random seed for reproducibility
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio} ({train_ratio}, {val_ratio}, {test_ratio})"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def _get_task_data_counts(self, data: pd.DataFrame) -> pd.Series:
        """
        Count number of non-NaN values for each task.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing task data with possible NaN values

        Returns
        -------
        pd.Series
            Series containing non-NaN counts for each task, sorted ascending
        """
        return data.notna().sum().sort_values()

    def _get_required_counts(self, total_count: int) -> Tuple[int, int, int]:
        """
        Calculate required counts for each split based on ratios.

        Parameters
        ----------
        total_count : int
            Total number of samples available

        Returns
        -------
        Tuple[int, int, int]
            Required counts for train, val, and test splits
        """
        train_count = int(np.floor(total_count * self.train_ratio))
        val_count = int(np.floor(total_count * self.val_ratio))
        test_count = total_count - train_count - val_count
        return train_count, val_count, test_count

    def split(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data ensuring all tasks have representation in splits.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame where each column is a task and may contain NaN values

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Arrays of indices for train, validation, and test splits
        """
        # Get task counts and sort by available data (ascending)
        task_counts = self._get_task_data_counts(data)

        # Initialize split sets for train/val/test
        train_indices = set()
        val_indices = set()
        test_indices = set()

        # Track allocated indices for each task
        allocated_indices = set()

        for task, count in task_counts.items():
            # Get valid indices for this task (non-NaN values)
            task_valid_indices = np.where(data[task].notna())[0]

            # Remove already allocated indices
            available_indices = np.array([idx for idx in task_valid_indices if idx not in allocated_indices])

            if len(available_indices) == 0:
                warnings.warn(f"Task {task} has no unallocated data points. Using already allocated indices.")
                available_indices = task_valid_indices

            # Shuffle available indices
            np.random.shuffle(available_indices)

            # Calculate required counts
            train_needed, val_needed, test_needed = self._get_required_counts(len(available_indices))

            # Calculate target samples for val/test
            available_count = len(available_indices)
            val_target = int(np.floor(available_count * self.val_ratio))
            test_target = int(np.floor(available_count * self.test_ratio))
            holdout_size = val_target + test_target

            # First allocate validation and test sets
            if holdout_size > 0:
                # Reserve validation set
                if self.val_ratio > 0:
                    val_indices.update(available_indices[-val_target:])

                # Reserve test set (can overlap with val if needed)
                if self.test_ratio > 0:
                    test_indices.update(available_indices[:test_target])

                # Remaining data goes to train (ensuring no overlap with val/test)
                if self.val_ratio > 0:
                    train_indices.update(available_indices[:-val_target])
                else:
                    train_indices.update(available_indices[test_target:])
            else:
                # No validation or test needed, use all for training
                train_indices.update(available_indices)

            # Update allocated indices
            allocated_indices.update(available_indices)

        # Convert sets to sorted arrays
        train_indices_arr = np.array(sorted(train_indices))
        val_indices_arr = np.array(sorted(val_indices))
        test_indices_arr = np.array(sorted(test_indices))

        # Verify we have data in required splits
        if len(train_indices_arr) == 0:
            raise ValueError("No training data available after splitting")
        if self.val_ratio > 0 and len(val_indices_arr) == 0:
            raise ValueError("No validation data available after splitting")

        return train_indices_arr, val_indices_arr, test_indices_arr
