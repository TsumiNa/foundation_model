import ast  # For safely evaluating string representations of lists/arrays
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Import SequenceTaskConfig to check its instance and access steps_column
from ..configs.model_config import SequenceTaskConfig, TaskType

logger = logging.getLogger(__name__)


# Helper function to parse elements that might be scalars, lists, or string representations of lists
def _parse_structured_element(
    element, task_name: str, column_name: str, dataset_name: str, expected_shape_for_nan: tuple
):
    """
    Parses an element from a pandas Series.
    Handles scalars, lists, numpy arrays, string representations of lists/arrays, and NaNs.
    Returns a numpy array. For NaNs or parsing errors, returns np.full(expected_shape_for_nan, np.nan).
    """
    if isinstance(element, (list, np.ndarray)):
        return np.asarray(element)
    elif isinstance(element, str):
        try:
            parsed_element = ast.literal_eval(element)
            # Ensure it's an array, even if ast.literal_eval returns a scalar
            return (
                np.array(parsed_element, ndmin=1)
                if not isinstance(parsed_element, (list, tuple))
                else np.asarray(parsed_element)
            )
        except (ValueError, SyntaxError, TypeError):
            logger.warning(
                f"[{dataset_name}] Task '{task_name}', column '{column_name}': Could not parse string element '{element}'. Using NaN placeholder."
            )
            return np.full(expected_shape_for_nan, np.nan)
    elif pd.isna(element):
        return np.full(expected_shape_for_nan, np.nan)
    else:  # Scalar number
        return np.array([element])  # Ensure it's an array


class CompoundDataset(Dataset):
    def __init__(
        self,
        formula_desc: Union[pd.DataFrame, np.ndarray],
        attributes: pd.DataFrame,  # Contains targets, series, temps, masks
        task_configs: List,  # List of task configuration objects
        structure_desc: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        use_structure_for_this_dataset: bool = False,
        task_masking_ratios: Optional[Dict[str, float]] = None,
        is_predict_set: bool = False,
        dataset_name: str = "dataset",
    ):
        self.dataset_name = dataset_name
        logger.info(f"[{self.dataset_name}] Initializing CompoundDataset...")
        logger.info(
            f"[{self.dataset_name}] is_predict_set: {is_predict_set}, use_structure: {use_structure_for_this_dataset}"
        )

        self.is_predict_set = is_predict_set
        self.use_structure_for_this_dataset = use_structure_for_this_dataset
        self.task_masking_ratios = task_masking_ratios

        # --- Input Validation (remains largely the same) ---
        if not isinstance(formula_desc, (pd.DataFrame, np.ndarray)):
            logger.error(f"[{self.dataset_name}] formula_desc type error: {type(formula_desc)}")
            raise TypeError("formula_desc must be pd.DataFrame or np.ndarray")
        if hasattr(formula_desc, "empty") and formula_desc.empty:
            raise ValueError("formula_desc DataFrame cannot be empty.")
        if isinstance(formula_desc, np.ndarray) and formula_desc.size == 0:
            raise ValueError("formula_desc np.ndarray cannot be empty.")

        if not isinstance(attributes, pd.DataFrame):
            logger.error(f"[{self.dataset_name}] attributes type error: {type(attributes)}")
            raise TypeError("attributes must be pd.DataFrame")

        if len(attributes.index) == 0 and len(attributes.columns) == 0 and len(formula_desc) > 0:
            logger.error(f"[{self.dataset_name}] attributes is a 0x0 DataFrame, but formula_desc has samples.")
            raise ValueError("attributes DataFrame cannot be empty when formula_desc is not.")
        if len(attributes.index) == 0 and len(attributes.columns) > 0 and len(formula_desc) > 0:
            logger.error(f"[{self.dataset_name}] attributes has columns but 0 rows (empty index).")
            raise ValueError("attributes DataFrame has columns but an empty index (no samples).")
        if not task_configs:
            raise ValueError("task_configs list cannot be empty.")

        # Convert formula_desc to tensor
        if isinstance(formula_desc, pd.DataFrame):
            self.x_formula = torch.tensor(formula_desc.values, dtype=torch.float32)
        else:  # np.ndarray
            self.x_formula = torch.tensor(formula_desc, dtype=torch.float32)

        # Handle structure_desc
        self.x_struct = None
        if self.use_structure_for_this_dataset and structure_desc is not None:
            # ... (validation and conversion for structure_desc remains the same) ...
            if not isinstance(structure_desc, (pd.DataFrame, np.ndarray)):
                raise TypeError("structure_desc must be pd.DataFrame or np.ndarray if provided")
            if hasattr(structure_desc, "empty") and structure_desc.empty:
                raise ValueError(
                    "structure_desc cannot be an empty DataFrame if use_structure_for_this_dataset is True."
                )
            if isinstance(structure_desc, np.ndarray) and structure_desc.size == 0:
                raise ValueError(
                    "structure_desc cannot be an empty np.ndarray if use_structure_for_this_dataset is True."
                )

            if isinstance(structure_desc, pd.DataFrame):
                self.x_struct = torch.tensor(structure_desc.values, dtype=torch.float32)
            else:  # np.ndarray
                self.x_struct = torch.tensor(structure_desc, dtype=torch.float32)
            if len(self.x_formula) != len(self.x_struct):
                raise ValueError("formula_desc and structure_desc must have the same number of samples.")
        elif self.use_structure_for_this_dataset and structure_desc is None:
            logger.warning(f"[{self.dataset_name}] use_structure_for_this_dataset is True, but structure_desc is None.")

        logger.info(f"[{self.dataset_name}] Final x_formula shape: {self.x_formula.shape}")
        if self.x_struct is not None:
            logger.info(f"[{self.dataset_name}] Final x_struct shape: {self.x_struct.shape}")

        self.y_dict: Dict[str, torch.Tensor] = {}
        self.temps_dict: Dict[str, torch.Tensor] = {}  # For steps data of sequence tasks
        self.task_masks_dict: Dict[str, torch.Tensor] = {}
        self.enabled_task_names: List[str] = []

        num_samples = len(self.x_formula)

        for cfg in task_configs:
            if not hasattr(cfg, "enabled") or not cfg.enabled:
                continue

            task_name = cfg.name
            task_type = cfg.type  # This is TaskType Enum
            self.enabled_task_names.append(task_name)
            logger.info(f"[{self.dataset_name}] Processing enabled task '{task_name}' (type: {task_type.name})")

            # --- Primary Data Loading (y_dict) using cfg.data_column ---
            data_col_for_task = cfg.data_column
            current_task_values_list = []
            current_task_mask_list = []

            expected_data_dim = 1  # Default for scalar
            if task_type == TaskType.REGRESSION and hasattr(cfg, "dims") and cfg.dims:
                expected_data_dim = cfg.dims[-1]
            elif task_type == TaskType.CLASSIFICATION:  # Target is class index, so dim is 1
                expected_data_dim = 1
            elif task_type == TaskType.SEQUENCE:
                # For sequence, data_column provides the sequence itself.
                # Expected dim is sequence length.
                expected_data_dim = getattr(cfg, "seq_len", None) or (
                    cfg.dims[-1] if hasattr(cfg, "dims") and cfg.dims and len(cfg.dims) > 1 else 1
                )

            if data_col_for_task and data_col_for_task in attributes.columns:
                logger.debug(
                    f"[{self.dataset_name}] Task '{task_name}': Loading data from column '{data_col_for_task}'."
                )
                raw_column_data = attributes[data_col_for_task]
                for element in raw_column_data:
                    parsed_val = _parse_structured_element(
                        element, task_name, data_col_for_task, self.dataset_name, (expected_data_dim,)
                    )
                    current_task_values_list.append(parsed_val)
                    current_task_mask_list.append(not np.all(np.isnan(parsed_val)))

                try:
                    processed_values_np = np.stack(current_task_values_list)
                except ValueError as e:  # Stacking failed, likely inconsistent shapes
                    logger.error(
                        f"[{self.dataset_name}] Task '{task_name}', column '{data_col_for_task}': Error stacking parsed data - {e}. Check data consistency. Using NaN placeholder."
                    )
                    processed_values_np = np.full((num_samples, expected_data_dim), np.nan)
                    current_task_mask_list = [False] * num_samples  # All masked due to error

            else:
                logger.warning(
                    f"[{self.dataset_name}] Task '{task_name}': data_column '{data_col_for_task}' not specified or not found in attributes. Using placeholder."
                )
                processed_values_np = np.full((num_samples, expected_data_dim), np.nan)
                current_task_mask_list = [False] * num_samples

            base_mask_np = np.array(current_task_mask_list, dtype=bool)

            if task_type == TaskType.CLASSIFICATION:
                self.y_dict[task_name] = torch.tensor(
                    np.nan_to_num(processed_values_np, nan=-1).astype(np.int64), dtype=torch.long
                )
            else:  # REGRESSION or SEQUENCE
                self.y_dict[task_name] = torch.tensor(np.nan_to_num(processed_values_np, nan=0.0), dtype=torch.float32)

            logger.debug(
                f"[{self.dataset_name}] Task '{task_name}': y_dict shape {self.y_dict[task_name].shape}, base_mask valid count: {np.sum(base_mask_np)}"
            )

            # --- Steps Data Loading (temps_dict for SEQUENCE tasks) using cfg.steps_column ---
            if isinstance(cfg, SequenceTaskConfig):
                steps_col_for_task = cfg.steps_column
                expected_steps_len = self.y_dict[task_name].shape[1]  # Should match sequence length from y_dict

                if steps_col_for_task and steps_col_for_task in attributes.columns:
                    logger.debug(
                        f"[{self.dataset_name}] Task '{task_name}': Loading steps from column '{steps_col_for_task}'."
                    )
                    raw_steps_data = attributes[steps_col_for_task]
                    current_task_steps_list = []
                    for element in raw_steps_data:
                        parsed_step = _parse_structured_element(
                            element, task_name, steps_col_for_task, self.dataset_name, (expected_steps_len,)
                        )
                        current_task_steps_list.append(parsed_step)

                    try:
                        processed_steps_np = np.stack(current_task_steps_list)
                    except ValueError as e:
                        logger.error(
                            f"[{self.dataset_name}] Task '{task_name}', steps column '{steps_col_for_task}': Error stacking parsed steps data - {e}. Using NaN placeholder."
                        )
                        processed_steps_np = np.full((num_samples, expected_steps_len), np.nan)

                elif steps_col_for_task:  # Specified in config but not found in attributes
                    logger.error(
                        f"[{self.dataset_name}] Task '{task_name}': Steps column '{steps_col_for_task}' "
                        f"is specified in config but not found in attributes DataFrame. This is required."
                    )
                    raise ValueError(
                        f"Steps column '{steps_col_for_task}' for task '{task_name}' not found in attributes data."
                    )
                else:  # steps_column not specified
                    logger.info(
                        f"[{self.dataset_name}] Task '{task_name}': No steps_column specified. "
                        f"Using zero placeholder for steps (temps_dict)."
                    )
                    processed_steps_np = np.zeros(
                        (num_samples, expected_steps_len)
                    )  # Default to zeros if not specified

                # Ensure steps are [N, L, 1] for compatibility with some models
                if processed_steps_np.ndim == 2:
                    processed_steps_np = np.expand_dims(processed_steps_np, axis=-1)

                self.temps_dict[task_name] = torch.tensor(
                    np.nan_to_num(processed_steps_np, nan=0.0), dtype=torch.float32
                )
                logger.debug(
                    f"[{self.dataset_name}] Task '{task_name}': temps_dict shape {self.temps_dict[task_name].shape}"
                )

            # --- Task Masking (final_mask_np) ---
            final_mask_np = base_mask_np.copy()  # Start with NaN-based mask
            num_valid_after_nan_mask = np.sum(final_mask_np)

            if self.task_masking_ratios and task_name in self.task_masking_ratios and not self.is_predict_set:
                # ... (random masking logic remains the same as original) ...
                ratio_to_keep = self.task_masking_ratios[task_name]
                logger.info(
                    f"[{self.dataset_name}] Task '{task_name}': Applying random masking "
                    f"with keep_ratio={ratio_to_keep}. Initial valid "
                    f"samples: {num_valid_after_nan_mask}"
                )
                if 0.0 <= ratio_to_keep < 1.0:
                    valid_indices = np.where(final_mask_np)[0]
                    if len(valid_indices) > 0:
                        num_to_keep = int(np.round(len(valid_indices) * ratio_to_keep))
                        if num_to_keep < len(valid_indices):
                            indices_to_set_false = np.random.choice(
                                valid_indices, size=len(valid_indices) - num_to_keep, replace=False
                            )
                            final_mask_np[indices_to_set_false] = False
                    # ... (logging for masking)
                # ... (handle ratio_to_keep == 0.0 or >= 1.0)

            if final_mask_np.ndim == 1:
                final_mask_np = final_mask_np.reshape(-1, 1)
            self.task_masks_dict[task_name] = torch.tensor(final_mask_np, dtype=torch.bool)
            logger.debug(
                f"[{self.dataset_name}] Task '{task_name}': final task_mask shape "
                f"{self.task_masks_dict[task_name].shape}, final valid count: "
                f"{torch.sum(self.task_masks_dict[task_name]).item()}"
            )

        logger.info(
            f"[{self.dataset_name}] CompoundDataset initialization complete. "
            f"Processed {len(self.enabled_task_names)} enabled tasks."
        )

    def __len__(self):
        return len(self.x_formula)

    def __getitem__(self, idx):
        current_x_formula = self.x_formula[idx]
        model_input_x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]

        if self.use_structure_for_this_dataset and self.x_struct is not None:
            model_input_x = (current_x_formula, self.x_struct[idx])
        else:
            model_input_x = current_x_formula

        sample_y_dict = {name: self.y_dict[name][idx] for name in self.enabled_task_names if name in self.y_dict}
        sample_task_masks_dict = {
            name: self.task_masks_dict[name][idx] for name in self.enabled_task_names if name in self.task_masks_dict
        }
        sample_temps_dict = {  # This is effectively steps_dict now
            name: self.temps_dict[name][idx] for name in self.enabled_task_names if name in self.temps_dict
        }

        return model_input_x, sample_y_dict, sample_task_masks_dict, sample_temps_dict

    @property
    def attribute_names(self) -> List[str]:
        return self.enabled_task_names[:]
