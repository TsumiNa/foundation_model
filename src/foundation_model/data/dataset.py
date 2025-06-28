import ast  # For safely evaluating string representations of lists/arrays
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from foundation_model.models.model_config import TaskType


# Helper function to parse elements that might be scalars, lists, or string representations of lists
def _parse_structured_element(
    element, task_name: str, column_name: str, dataset_name: str, expected_shape_for_nan: tuple
):
    """
    Parses an element from a pandas Series.
    Handles scalars, lists, numpy arrays, string representations of lists/arrays, and NaNs.
    Returns a numpy array. For NaNs or parsing errors, returns np.full(expected_shape_for_nan, np.nan).
    """
    if isinstance(element, np.ndarray) and np.issubdtype(element.dtype, np.number):
        return element.astype(float)  # Ensure float type for consistency
    elif isinstance(element, (list, tuple)) or (isinstance(element, np.ndarray) and element.dtype == object):
        # Handle lists, tuples, or object arrays that might contain mixed types
        processed_list = []
        for item in element:
            if isinstance(item, str):
                try:
                    # Attempt to evaluate if it's a string representation of a number
                    eval_item = ast.literal_eval(item)
                    if isinstance(eval_item, (int, float, bool)):  # bool is subclass of int
                        processed_list.append(float(eval_item))
                    else:  # ast.literal_eval returned a list/tuple/dict, treat as unparsable for this context
                        logger.debug(
                            f"[{dataset_name}] Task '{task_name}', column '{column_name}': Item '{item}' in list evaluated to non-scalar {type(eval_item)}. Using NaN."
                        )
                        processed_list.append(np.nan)
                except (ValueError, SyntaxError, TypeError):
                    logger.warning(  # Changed from debug to warning
                        f"[{dataset_name}] Task '{task_name}', column '{column_name}': Could not parse string element '{item}' in list. Using NaN."  # Changed "item" to "element"
                    )
                    processed_list.append(np.nan)
            elif isinstance(item, (int, float, bool)):
                processed_list.append(float(item))
            elif pd.isna(item):
                processed_list.append(np.nan)
            else:  # Other non-numeric types within the list
                logger.debug(
                    f"[{dataset_name}] Task '{task_name}', column '{column_name}': Non-numeric item '{item}' (type: {type(item)}) in list. Using NaN."
                )
                processed_list.append(np.nan)
        return np.array(processed_list, dtype=float)
    elif isinstance(element, str):
        try:
            # For string elements that are supposed to be lists/arrays themselves
            parsed_element = ast.literal_eval(element)
            # Ensure it's an array, even if ast.literal_eval returns a scalar
            # This branch now primarily handles strings that represent entire lists/arrays like "[1, 2, 3]"
            if isinstance(parsed_element, (list, tuple)):
                # Recursively call to process items within the parsed list/tuple
                return _parse_structured_element(
                    parsed_element, task_name, column_name, dataset_name, expected_shape_for_nan
                )
            elif isinstance(parsed_element, (int, float, bool)):
                return np.array([float(parsed_element)])  # Single numeric value from string
            else:  # Parsed to something else unexpected
                logger.warning(
                    f"[{dataset_name}] Task '{task_name}', column '{column_name}': String element '{element}' evaluated to unexpected type {type(parsed_element)}. Using NaN placeholder."
                )
                return np.full(expected_shape_for_nan, np.nan)
        except (ValueError, SyntaxError, TypeError):
            logger.warning(
                f"[{dataset_name}] Task '{task_name}', column '{column_name}': Could not parse string element '{element}' as a list/scalar. Using NaN placeholder."
            )
            return np.full(expected_shape_for_nan, np.nan)
    elif pd.isna(element):
        return np.full(expected_shape_for_nan, np.nan)
    elif isinstance(element, (int, float, bool)):  # Scalar number
        return np.array([float(element)])
    else:  # Other unhandled type, treat as error / NaN
        logger.warning(
            f"[{dataset_name}] Task '{task_name}', column '{column_name}': Unhandled element type '{type(element)}' ('{element}'). Using NaN placeholder."
        )
        return np.full(expected_shape_for_nan, np.nan)


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
        self.t_sequences_dict: Dict[str, torch.Tensor] = {}  # For t-parameter sequences of ExtendRegression tasks
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
            if task_type == TaskType.REGRESSION:
                # For regression targets, the expected dimension is typically 1 (a single scalar value).
                # If the data_column contains multiple values (e.g., a list for multi-output regression),
                # _parse_structured_element and np.stack should handle creating an (N, M) array.
                # cfg.dims is for the model head architecture, not for target data shape.
                expected_data_dim = 1
            elif task_type == TaskType.CLASSIFICATION:  # Target is class index, so dim is 1
                expected_data_dim = 1

            # Strict validation for data_column
            if not data_col_for_task:  # Not specified
                logger.error(f"[{self.dataset_name}] Task '{task_name}': data_column is not specified in config.")
                raise ValueError(f"data_column for task '{task_name}' must be specified.")
            elif data_col_for_task not in attributes.columns:  # Specified but not found
                logger.error(
                    f"[{self.dataset_name}] Task '{task_name}': data_column '{data_col_for_task}' "
                    f"is specified in config but not found in attributes DataFrame."
                )
                raise ValueError(
                    f"Data column '{data_col_for_task}' for task '{task_name}' not found in attributes data."
                )
            else:  # Column exists, load data
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
                        f"[{self.dataset_name}] Task '{task_name}', column '{data_col_for_task}': Error stacking parsed data - {e}. Check data consistency."
                    )
                    raise ValueError(f"Error processing data column '{data_col_for_task}' for task '{task_name}': {e}")

            base_mask_np = np.array(current_task_mask_list, dtype=bool)

            if task_type == TaskType.CLASSIFICATION:
                self.y_dict[task_name] = torch.tensor(
                    np.nan_to_num(processed_values_np, nan=-1).astype(np.int64), dtype=torch.long
                )
            else:  # REGRESSION or ExtendRegression
                self.y_dict[task_name] = torch.tensor(np.nan_to_num(processed_values_np, nan=0.0), dtype=torch.float32)

            logger.debug(
                f"[{self.dataset_name}] Task '{task_name}': y_dict shape {self.y_dict[task_name].shape}, base_mask valid count: {np.sum(base_mask_np)}"
            )

            # --- T-parameter Data Loading (t_sequences_dict for ExtendRegression tasks) using cfg.t_column ---
            if task_type == TaskType.ExtendRegression:
                t_col_for_task = cfg.t_column
                expected_t_len = self.y_dict[task_name].shape[1]  # Should match sequence length from y_dict

                # Strict validation for t_column
                if not t_col_for_task:  # Not specified
                    logger.error(f"[{self.dataset_name}] Task '{task_name}': t_column is not specified in config.")
                    raise ValueError(f"t_column for ExtendRegression task '{task_name}' must be specified.")
                elif t_col_for_task not in attributes.columns:  # Specified but not found
                    logger.error(
                        f"[{self.dataset_name}] Task '{task_name}': t_column '{t_col_for_task}' "
                        f"is specified in config but not found in attributes DataFrame."
                    )
                    raise ValueError(
                        f"T-parameter column '{t_col_for_task}' for task '{task_name}' not found in attributes data."
                    )
                else:  # Column exists, load data
                    logger.debug(
                        f"[{self.dataset_name}] Task '{task_name}': Loading t-parameters from column '{t_col_for_task}'."
                    )
                    raw_t_data = attributes[t_col_for_task]
                    current_task_t_list = []
                    for element in raw_t_data:
                        parsed_t = _parse_structured_element(
                            element, task_name, t_col_for_task, self.dataset_name, (expected_t_len,)
                        )
                        current_task_t_list.append(parsed_t)

                    try:
                        processed_t_np = np.stack(current_task_t_list)
                    except ValueError as e:
                        logger.error(
                            f"[{self.dataset_name}] Task '{task_name}', t-parameter column '{t_col_for_task}': Error stacking parsed t-parameter data - {e}."
                        )
                        raise ValueError(
                            f"Error processing t-parameter column '{t_col_for_task}' for task '{task_name}': {e}"
                        )

                # Store t-parameter data without extra dimension expansion
                self.t_sequences_dict[task_name] = torch.tensor(
                    np.nan_to_num(processed_t_np, nan=0.0), dtype=torch.float32
                )
                logger.debug(
                    f"[{self.dataset_name}] Task '{task_name}': t_sequences_dict shape {self.t_sequences_dict[task_name].shape}"
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
        sample_t_sequences_dict = {  # T-parameter sequences for ExtendRegression tasks
            name: self.t_sequences_dict[name][idx] for name in self.enabled_task_names if name in self.t_sequences_dict
        }

        return model_input_x, sample_y_dict, sample_task_masks_dict, sample_t_sequences_dict

    @property
    def attribute_names(self) -> List[str]:
        return self.enabled_task_names[:]
