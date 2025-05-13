import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Assuming TaskType enum is available, e.g., from model_config or a shared types module
# from ..models.model_config import TaskType # Or appropriate path

logger = logging.getLogger(__name__)


class CompoundDataset(Dataset):
    def __init__(
        self,
        formula_desc: Union[pd.DataFrame, np.ndarray],
        attributes: pd.DataFrame,  # Contains targets, series data, temps data, and potentially masks
        task_configs: List,  # List of task configuration objects
        structure_desc: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        use_structure_for_this_dataset: bool = False,
        task_masking_ratios: Optional[Dict[str, float]] = None,  # Key: task_name, Value: keep_ratio (0.0 to 1.0)
        is_predict_set: bool = False,
        dataset_name: str = "dataset",  # For logging purposes
    ):
        """
        Multi-task compound dataset.

        Parameters
        ----------
        formula_desc : Union[pd.DataFrame, np.ndarray]
            Input formula features for the compounds. Assumed to be aligned with attributes.
        attributes : pd.DataFrame
            Contains target attributes, sequence data, temps, etc. Indexed like formula_desc.
        task_configs : List
            List of task configurations defining all tasks to be processed.
        structure_desc : Optional[Union[pd.DataFrame, np.ndarray]], optional
            Input structure features. Assumed to be aligned. Defaults to None.
        use_structure_for_this_dataset : bool, optional
            Whether to use structure_desc if available. Defaults to False.
        task_masking_ratios : Optional[Dict[str, float]], optional
            Ratios for randomly masking valid samples per task (for training).
            1.0 means use all valid samples, 0.0 means mask all. Defaults to None (no random masking).
        is_predict_set : bool, optional
            If True, __getitem__ will only return formula_desc as model input 'x'. Defaults to False.
        dataset_name : str, optional
            A name for this dataset instance, used in logging. Defaults to "dataset".
        """
        self.dataset_name = dataset_name
        logger.info(f"[{self.dataset_name}] Initializing CompoundDataset...")
        logger.info(
            f"[{self.dataset_name}] is_predict_set: {is_predict_set}, use_structure_for_this_dataset: {use_structure_for_this_dataset}"
        )

        self.is_predict_set = is_predict_set
        self.use_structure_for_this_dataset = use_structure_for_this_dataset

        # Log shapes of input data
        formula_shape = formula_desc.shape if hasattr(formula_desc, "shape") else "N/A"
        logger.info(f"[{self.dataset_name}] Received formula_desc with shape: {formula_shape}")
        attributes_shape = attributes.shape if hasattr(attributes, "shape") else "N/A"
        logger.info(f"[{self.dataset_name}] Received attributes with shape: {attributes_shape}")

        if isinstance(formula_desc, pd.DataFrame):
            self.x_formula = torch.tensor(formula_desc.values, dtype=torch.float32)
        elif isinstance(formula_desc, np.ndarray):
            self.x_formula = torch.tensor(formula_desc, dtype=torch.float32)
        else:
            logger.error(f"[{self.dataset_name}] formula_desc type error: {type(formula_desc)}")
            raise TypeError("formula_desc must be pd.DataFrame or np.ndarray")

        if self.use_structure_for_this_dataset and structure_desc is not None:
            struct_shape = structure_desc.shape if hasattr(structure_desc, "shape") else "N/A"
            logger.info(f"[{self.dataset_name}] Received structure_desc with shape: {struct_shape}")
            if isinstance(structure_desc, pd.DataFrame):
                self.x_struct = torch.tensor(structure_desc.values, dtype=torch.float32)
            elif isinstance(structure_desc, np.ndarray):
                self.x_struct = torch.tensor(structure_desc, dtype=torch.float32)
            else:
                logger.error(f"[{self.dataset_name}] structure_desc type error: {type(structure_desc)}")
                raise TypeError("structure_desc must be pd.DataFrame or np.ndarray if provided")
            if len(self.x_formula) != len(self.x_struct):
                logger.error(
                    f"[{self.dataset_name}] Length mismatch: formula_desc ({len(self.x_formula)}) vs structure_desc ({len(self.x_struct)})"
                )
                raise ValueError("formula_desc and structure_desc must have the same number of samples.")
        else:
            self.x_struct = None
            if self.use_structure_for_this_dataset and structure_desc is None:
                logger.warning(
                    f"[{self.dataset_name}] use_structure_for_this_dataset is True, but structure_desc is None."
                )

        logger.info(f"[{self.dataset_name}] Final x_formula shape: {self.x_formula.shape}")
        if self.x_struct is not None:
            logger.info(f"[{self.dataset_name}] Final x_struct shape: {self.x_struct.shape}")

        self.y_dict: Dict[str, torch.Tensor] = {}
        self.temps_dict: Dict[str, torch.Tensor] = {}
        self.task_masks_dict: Dict[str, torch.Tensor] = {}

        self.enabled_task_names: List[str] = []
        logger.info(f"[{self.dataset_name}] Processing {len(task_configs)} provided task configurations.")

        for cfg_idx, cfg in enumerate(task_configs):
            if not hasattr(cfg, "enabled") or not cfg.enabled:
                logger.debug(
                    f"[{self.dataset_name}] Task config {cfg_idx} ('{getattr(cfg, 'name', 'N/A')}') is not enabled, skipping."
                )
                continue

            task_name = cfg.name
            task_type_enum_name = cfg.type.name  # e.g. "SEQUENCE"
            task_type_str = task_type_enum_name.lower()
            logger.info(f"[{self.dataset_name}] Processing enabled task '{task_name}' (type: {task_type_enum_name})")
            self.enabled_task_names.append(task_name)

            # --- Target Value / Series Data ---
            target_col_name = f"{task_name}_{task_type_str}_value"
            series_col_name = f"{task_name}_{task_type_str}_series"
            logger.debug(
                f"[{self.dataset_name}] Task '{task_name}': expected value col '{target_col_name}', series col '{series_col_name}'"
            )

            if task_type_enum_name == "SEQUENCE":
                if series_col_name in attributes.columns:
                    logger.debug(f"[{self.dataset_name}] Task '{task_name}': Found series column '{series_col_name}'.")
                    task_data_pd_series = attributes[series_col_name]
                else:
                    logger.warning(
                        f"[{self.dataset_name}] Task '{task_name}': Series column '{series_col_name}' not found in attributes. Will use '{target_col_name}' if available or placeholder."
                    )
                    task_data_pd_series = None  # Fallback to target_col_name or placeholder
            else:  # REGRESSION, CLASSIFICATION
                task_data_pd_series = None  # Not primary for these types

            if task_data_pd_series is not None:  # Primarily for SEQUENCE using _series
                # Handle cases where cells might be lists/ndarrays
                if task_data_pd_series.dtype == "object" and isinstance(
                    task_data_pd_series.iloc[0], (list, np.ndarray)
                ):
                    logger.debug(f"[{self.dataset_name}] Task '{task_name}': Stacking object series data.")
                    processed_values = np.stack(task_data_pd_series.values)
                else:
                    processed_values = task_data_pd_series.values

                if processed_values.ndim == 1:
                    logger.debug(f"[{self.dataset_name}] Task '{task_name}': Reshaping 1D series data to 2D.")
                    processed_values = processed_values.reshape(-1, 1)
                self.y_dict[task_name] = torch.tensor(np.nan_to_num(processed_values, nan=0.0), dtype=torch.float32)
                base_mask_np = ~task_data_pd_series.apply(
                    lambda x: np.all(np.isnan(x)) if isinstance(x, (list, np.ndarray)) else pd.isna(x)
                ).values.astype(bool)
                logger.debug(
                    f"[{self.dataset_name}] Task '{task_name}': y_dict shape {self.y_dict[task_name].shape}, base_mask valid count: {np.sum(base_mask_np)}"
                )

            elif target_col_name in attributes.columns:  # For REG/CLASS or SEQUENCE fallback to _value
                logger.debug(f"[{self.dataset_name}] Task '{task_name}': Found value column '{target_col_name}'.")
                task_values = attributes[target_col_name].values
                if task_values.ndim == 1:
                    task_values = task_values.reshape(-1, 1)

                # Determine dtype based on task type
                if task_type_enum_name == "CLASSIFICATION":
                    # For classification, target should be long integers (class indices)
                    # nan_to_num might be problematic if NaNs are present in integer labels;
                    # assume classification targets are clean or handle NaNs appropriately before this step.
                    # If NaNs are impossible for classification targets, nan_to_num can be removed for this path.
                    # For now, keep nan_to_num but ensure long type.
                    self.y_dict[task_name] = torch.tensor(
                        np.nan_to_num(task_values, nan=-1).astype(np.int64), dtype=torch.long
                    )
                    # Use -1 for NaN in integer arrays if they can occur, then handle in loss via ignore_index
                else:  # REGRESSION or SEQUENCE (if using _value column)
                    self.y_dict[task_name] = torch.tensor(np.nan_to_num(task_values, nan=0.0), dtype=torch.float32)

                base_mask_np = ~np.isnan(attributes[target_col_name].values).astype(bool)
                logger.debug(
                    f"[{self.dataset_name}] Task '{task_name}': y_dict shape {self.y_dict[task_name].shape}, base_mask valid count: {np.sum(base_mask_np)}"
                )
            else:
                logger.warning(
                    f"[{self.dataset_name}] Task '{task_name}': Neither series column '{series_col_name}' nor value column '{target_col_name}' found. "
                    f"Using zero placeholder."
                )
                placeholder_dim = 1
                if hasattr(cfg, "dims") and cfg.dims:
                    placeholder_dim = cfg.dims[-1]
                elif hasattr(cfg, "num_classes"):  # This is for ClassificationTaskConfig
                    placeholder_dim = cfg.num_classes

                # Determine dtype for placeholder based on task type
                if task_type_enum_name == "CLASSIFICATION":
                    # Placeholder for classification should be long, e.g., filled with a specific ignore_index like -1
                    self.y_dict[task_name] = torch.full(
                        (len(self.x_formula), placeholder_dim), fill_value=-1, dtype=torch.long
                    )
                else:  # REGRESSION or SEQUENCE
                    self.y_dict[task_name] = torch.zeros((len(self.x_formula), placeholder_dim), dtype=torch.float32)
                base_mask_np = np.zeros(len(self.x_formula), dtype=bool)  # All masked for placeholder
                logger.debug(
                    f"[{self.dataset_name}] Task '{task_name}': y_dict (placeholder) shape {self.y_dict[task_name].shape}, base_mask valid count: {np.sum(base_mask_np)}"
                )

            # --- Temps Data (for sequence tasks) ---
            if task_type_enum_name == "SEQUENCE":
                temps_col_name = f"{task_name}_temps"
                logger.debug(f"[{self.dataset_name}] Task '{task_name}': Expected temps column '{temps_col_name}'")
                if temps_col_name in attributes.columns:
                    logger.debug(f"[{self.dataset_name}] Task '{task_name}': Found temps column '{temps_col_name}'.")
                    temps_data_pd_series = attributes[temps_col_name]
                    if temps_data_pd_series.dtype == "object" and isinstance(
                        temps_data_pd_series.iloc[0], (list, np.ndarray)
                    ):
                        logger.debug(f"[{self.dataset_name}] Task '{task_name}': Stacking object temps data.")
                        processed_temps = np.stack(temps_data_pd_series.values)
                    else:
                        processed_temps = temps_data_pd_series.values

                    if processed_temps.ndim == 1:
                        logger.debug(f"[{self.dataset_name}] Task '{task_name}': Reshaping 1D temps data to 2D.")
                        processed_temps = processed_temps.reshape(-1, 1)
                    if processed_temps.ndim == 2:  # Add channel dim: [N, L] -> [N, L, 1]
                        logger.debug(f"[{self.dataset_name}] Task '{task_name}': Adding channel dim to temps data.")
                        processed_temps = np.expand_dims(processed_temps, axis=-1)
                    self.temps_dict[task_name] = torch.tensor(processed_temps, dtype=torch.float32)
                    logger.debug(
                        f"[{self.dataset_name}] Task '{task_name}': temps_dict shape {self.temps_dict[task_name].shape}"
                    )
                else:
                    logger.warning(
                        f"[{self.dataset_name}] Task '{task_name}': Temps column '{temps_col_name}' not found for sequence task."
                    )
                    # self.temps_dict[task_name] will not be populated, model needs to handle this.

            # --- Task Masking ---
            final_mask_np = base_mask_np.copy()
            num_valid_after_nan_mask = np.sum(final_mask_np)

            if task_masking_ratios and task_name in task_masking_ratios and not self.is_predict_set:
                ratio_to_keep = task_masking_ratios[task_name]
                logger.info(
                    f"[{self.dataset_name}] Task '{task_name}': Applying random masking with keep_ratio={ratio_to_keep}. Initial valid samples: {num_valid_after_nan_mask}"
                )
                if 0.0 <= ratio_to_keep < 1.0:
                    valid_indices = np.where(final_mask_np)[0]
                    if len(valid_indices) > 0:
                        num_to_keep = int(np.round(len(valid_indices) * ratio_to_keep))
                        if num_to_keep < len(valid_indices):  # only mask if we are keeping fewer than all valid
                            indices_to_set_false = np.random.choice(
                                valid_indices, size=len(valid_indices) - num_to_keep, replace=False
                            )
                            final_mask_np[indices_to_set_false] = False
                            logger.info(
                                f"[{self.dataset_name}] Task '{task_name}': {len(valid_indices) - num_to_keep} samples further masked. Kept {num_to_keep}."
                            )
                        else:
                            logger.info(
                                f"[{self.dataset_name}] Task '{task_name}': Keep ratio {ratio_to_keep} results in keeping all {len(valid_indices)} valid samples. No random masking applied."
                            )
                    else:
                        logger.info(
                            f"[{self.dataset_name}] Task '{task_name}': No valid samples after NaN masking to apply random masking to."
                        )

                elif ratio_to_keep == 0.0:
                    final_mask_np[:] = False
                    logger.info(f"[{self.dataset_name}] Task '{task_name}': All samples masked due to keep_ratio=0.0.")
                elif ratio_to_keep >= 1.0:
                    logger.info(
                        f"[{self.dataset_name}] Task '{task_name}': Keep ratio {ratio_to_keep} >= 1.0. No random masking applied beyond NaN masking."
                    )

            if final_mask_np.ndim == 1:  # Ensure mask is [N, 1]
                final_mask_np = final_mask_np.reshape(-1, 1)
            self.task_masks_dict[task_name] = torch.tensor(final_mask_np, dtype=torch.bool)
            logger.debug(
                f"[{self.dataset_name}] Task '{task_name}': final task_mask shape {self.task_masks_dict[task_name].shape}, final valid count: {torch.sum(self.task_masks_dict[task_name]).item()}"
            )

        logger.info(
            f"[{self.dataset_name}] CompoundDataset initialization complete. Processed {len(self.enabled_task_names)} enabled tasks."
        )

    def __len__(self):
        return len(self.x_formula)

    def __getitem__(self, idx):
        current_x_formula = self.x_formula[idx]
        model_input_x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]

        if self.is_predict_set:
            model_input_x = current_x_formula
        else:
            if self.use_structure_for_this_dataset and self.x_struct is not None:
                model_input_x = (current_x_formula, self.x_struct[idx])
            else:
                model_input_x = current_x_formula

        sample_y_dict = {name: self.y_dict[name][idx] for name in self.enabled_task_names if name in self.y_dict}
        sample_task_masks_dict = {
            name: self.task_masks_dict[name][idx] for name in self.enabled_task_names if name in self.task_masks_dict
        }
        sample_temps_dict = {
            name: self.temps_dict[name][idx] for name in self.enabled_task_names if name in self.temps_dict
        }

        return model_input_x, sample_y_dict, sample_task_masks_dict, sample_temps_dict

    @property
    def attribute_names(self) -> List[str]:
        """
        Get the list of enabled task names.
        """
        return self.enabled_task_names[:]  # Return a copy
