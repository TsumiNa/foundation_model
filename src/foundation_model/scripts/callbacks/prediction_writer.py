# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Callback to save model predictions as a pandas DataFrame.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, List, Literal, Sequence

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger


class PredictionDataFrameWriter(BasePredictionWriter):
    output_path: Path  # Explicitly type hint instance variables
    write_interval: Literal["batch", "epoch", "batch_and_epoch"]  # Explicitly type hint inherited instance variable

    """
    A PyTorch Lightning callback to save model predictions to a pandas DataFrame.

    This callback collects predictions from all batches within an epoch and
    saves them as a single DataFrame in both CSV and compressed pickle formats.
    It handles various prediction structures, including scalars, lists, and
    nested lists (e.g., for sequence predictions), ensuring that the original
    structure is preserved in the DataFrame.

    Parameters
    ----------
    output_path : str
        The directory path where prediction result files will be saved.
        For example, if `output_path` is "outputs/predictions", files will be
        saved as "outputs/predictions/predictions.csv" and "outputs/predictions/predictions.pd.xz".
    write_interval : str, optional
        When to write predictions. Supports "batch" and "epoch".
        Defaults to "epoch", which is recommended for this callback as it
        processes all predictions for an epoch at once.
    """

    def __init__(
        self, output_path: str, write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch"
    ):  # Update type hint
        super().__init__(write_interval)
        self.output_path = Path(output_path)
        # Ensure the output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _process_predictions(self, predictions: List[STEP_OUTPUT]) -> pd.DataFrame:
        """
        Processes a list of batch predictions into a single pandas DataFrame.

        Parameters
        ----------
        predictions : List[STEP_OUTPUT]
            A list where each element is the output of `predict_step` for a batch.
            Each `STEP_OUTPUT` is expected to be a dictionary where keys are
            prediction names (e.g., "task_name_value") and values are numpy arrays.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each column corresponds to a prediction type and
            each row represents a sample.
        """
        accumulated_data = defaultdict(list)

        if not predictions:
            return pd.DataFrame()

        # `predictions` is a list of dictionaries, one for each batch
        # Each dictionary contains {pred_name: np.ndarray_of_shape_batch_x_features}
        for batch_output in predictions:  # Renamed for clarity
            if not isinstance(batch_output, dict):
                # This can happen if predict_step returns a single tensor instead of a dict,
                # or if the dataloader was empty for a particular batch index.
                logger.warning(
                    f"Expected batch output to be a dict, got {type(batch_output)}. Skipping this batch's output."
                )
                continue

            # Now, batch_output is confirmed to be a dict
            for key, numpy_array_values in batch_output.items():
                # Convert numpy array to list of lists/values to preserve structure
                # This handles scalars, 1D arrays, and 2D arrays (sequences)
                if isinstance(numpy_array_values, torch.Tensor):
                    numpy_array_values = numpy_array_values.cpu().numpy()

                if not isinstance(numpy_array_values, np.ndarray):
                    # If it's already a list or scalar, extend directly
                    # This might happen if predict_step already returns lists
                    if isinstance(numpy_array_values, list):
                        accumulated_data[key].extend(numpy_array_values)
                    else:  # scalar
                        accumulated_data[key].append(numpy_array_values)
                    continue

                if numpy_array_values.ndim == 0:  # Scalar
                    accumulated_data[key].append(numpy_array_values.item())
                elif numpy_array_values.ndim == 1:  # Shape (B,): B scalar values
                    accumulated_data[key].extend(numpy_array_values.tolist())
                elif (
                    numpy_array_values.ndim == 2 and numpy_array_values.shape[1] == 1
                ):  # Shape (B, 1): Scalar per sample
                    accumulated_data[key].extend(numpy_array_values.squeeze(axis=1).tolist())
                else:  # Higher-dimensional array (e.g., list of sequences, or (B, N) with N>=2)
                    # Convert each row to a list
                    accumulated_data[key].extend([row.tolist() for row in numpy_array_values])

        # Check if all lists have the same length for DataFrame creation
        # This is a basic check; more complex validation might be needed
        # if different tasks produce different numbers of samples per batch,
        # which shouldn't happen with typical dataloaders.
        if accumulated_data:
            expected_len = -1
            for key, data_list in accumulated_data.items():
                if expected_len == -1:
                    expected_len = len(data_list)
                elif len(data_list) != expected_len:
                    # This indicates an issue, possibly with how batches are aggregated
                    # or if predict_step doesn't return consistent batch sizes for all keys.
                    # For now, we'll proceed, but pandas might raise an error or fill with NaNs.
                    logger.warning(
                        f"Inconsistent data lengths for DataFrame construction. "
                        f"Key '{key}' has length {len(data_list)}, expected {expected_len}."
                    )
                    # Attempt to pad or truncate, or let pandas handle it.
                    # Simplest is to let pandas handle, it will raise ValueError if lengths mismatch.
                    # To avoid error, one might need to ensure all lists are of the same length,
                    # e.g., by finding the max length and padding with None/NaN.
                    # However, this situation usually points to a problem upstream.

            try:
                df = pd.DataFrame(accumulated_data)
            except ValueError as e:
                logger.error(
                    f"Error creating DataFrame: {e}. This might be due to inconsistent list lengths for columns."
                )
                # Fallback: create DataFrame from each column individually and concat,
                # which might be more robust to length mismatches but can be slow.
                # Or, identify the max length and pad shorter lists.
                # For now, we'll re-raise if pandas can't handle it.
                # A more robust solution would be to ensure all lists in `accumulated_data`
                # are of the same length before calling pd.DataFrame.
                # This often means the total number of samples processed.

                # Let's find the maximum length among all lists
                max_len = 0
                if accumulated_data:
                    max_len = max(len(v) for v in accumulated_data.values())

                # Pad shorter lists with None (or np.nan)
                padded_data = {}
                for key, data_list in accumulated_data.items():
                    if len(data_list) < max_len:
                        padded_data[key] = data_list + [None] * (max_len - len(data_list))
                    else:
                        padded_data[key] = data_list

                if not padded_data and accumulated_data:  # if padding was not needed
                    df = pd.DataFrame(accumulated_data)
                elif padded_data:
                    df = pd.DataFrame(padded_data)
                else:  # if accumulated_data was empty
                    df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        return df

    def write_on_epoch_end(
        self,
        trainer: "L.Trainer",  # type: ignore
        pl_module: "L.LightningModule",  # type: ignore
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ) -> None:
        """
        Called at the end of the prediction epoch to save all collected predictions.

        Parameters
        ----------
        trainer : L.Trainer
            The PyTorch Lightning Trainer instance.
        pl_module : L.LightningModule
            The PyTorch Lightning module.
        predictions : Sequence[Any]
            A list containing the outputs of `predict_step` from all batches in the epoch.
        batch_indices : Sequence[Any]
            A list of batch indices corresponding to the predictions.
        """
        if not predictions:
            logger.info("No predictions to save.")
            return

        # `predictions` here is a list of what `predict_step` returned for each batch.
        # If `predict_step` returns a dict, then `predictions` is List[dict].
        df = self._process_predictions(predictions)  # type: ignore

        if df.empty:
            logger.warning("Processed DataFrame is empty. No data saved.")
            return

        # Save to CSV
        csv_path = self.output_path / "predictions.csv"
        try:
            df.to_csv(csv_path, index=True)
            logger.info(f"Predictions saved to {csv_path}")
        except Exception as e:
            logger.error(f"Error saving predictions to CSV {csv_path}: {e}")

        # Save to compressed Pickle
        pickle_path = self.output_path / "predictions.pd.xz"
        try:
            df.to_pickle(pickle_path, compression="xz")
            logger.info(f"Predictions saved to {pickle_path}")
        except Exception as e:
            logger.error(f"Error saving predictions to Pickle {pickle_path}: {e}")

    # write_on_batch_end is not needed when write_interval="epoch"
    # as Lightning accumulates predictions automatically.
    def write_on_batch_end(
        self,
        trainer: "L.Trainer",  # type: ignore
        pl_module: "L.LightningModule",  # type: ignore
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Called after each batch if `write_interval` is "batch".
        This implementation does nothing if `write_interval` is "epoch".
        """
        if self.write_interval == "batch":
            # This would be for saving each batch individually.
            # For the current requirement of a single DataFrame, this is not used.
            # If batch-wise saving was needed, logic similar to _process_predictions
            # for a single batch would go here.
            pass
        return
