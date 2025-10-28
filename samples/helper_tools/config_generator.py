# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Generates a base model configuration YAML from an attributes.csv file.
It infers task names and types (REGRESSION/CLASSIFICATION) from columns
prefixed with 'target_'.
"""

import argparse
import copy
import re
from pathlib import Path

import pandas as pd
import yaml  # Using PyYAML

# Default paths relative to this script's parent's parent (i.e., samples/ directory)
DEFAULT_ATTRIBUTES_PATH = Path(__file__).resolve().parent.parent / "fake_data" / "attributes.csv"
DEFAULT_FORMULA_FEATURES_PATH = Path(__file__).resolve().parent.parent / "fake_data" / "formula_features.csv"
DEFAULT_OUTPUT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "generated_configs" / "generated_model_config.yaml"
)
BASE_MODEL_CONFIG_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "model_configs" / "base_model.yaml"
)

# Threshold for unique values to consider a column as classification if dtype is int-like
CLASSIFICATION_UNIQUE_THRESHOLD = 10


def load_base_template(template_path: Path) -> dict:
    """Loads the base model configuration template."""
    if not template_path.exists():
        print(f"Warning: Base template {template_path} not found. Using a minimal hardcoded template.")
        # Fallback to a very basic structure if base_model.yaml is missing
        # This should be more robust in a real scenario, perhaps erroring out or having a more complete default.
        return {
            "experiment_name": "generated_experiment",
            "log_dir": "results/logs/${experiment_name}",
            "datamodule": {
                "class_path": "foundation_model.data.datamodule.CompoundDataModule",
                "batch_size": 64,
                "num_workers": 4,
                "val_split": 0.1,
                "test_split": 0.1,
            },
            "model": {
                "class_path": "foundation_model.models.flexible_multi_task_model.FlexibleMultiTaskModel",
                "shared_block_dims": [256, 512, 512],
                "task_configs": [],
                "shared_block_optimizer": {"optimizer_type": "AdamW", "lr": 0.001, "weight_decay": 0.01},
            },
            "trainer": {
                "class_path": "lightning.pytorch.Trainer",
                "max_epochs": 50,
                "accelerator": "auto",
                "devices": "auto",
                "logger": [  # Logger is now a list
                    {"class_path": "lightning.pytorch.loggers.CSVLogger", "save_dir": "${log_dir}", "name": ""},
                    {"class_path": "lightning.pytorch.loggers.TensorBoardLogger", "save_dir": "${log_dir}", "name": ""},
                ],
                "callbacks": [
                    {
                        "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "monitor": "val_total_loss",
                        "mode": "min",
                    },
                    {
                        "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                        "monitor": "val_total_loss",
                        "patience": 10,
                    },
                ],
            },
        }
    with open(template_path, "r") as f:
        return yaml.safe_load(f)


def infer_task_type_and_details(series: pd.Series) -> tuple[str | None, dict | None]:
    """Infers task type and details from a pandas Series."""
    task_details = {}
    # Try to convert to numeric, coercing errors. This helps identify mixed-type columns.
    numeric_series = pd.to_numeric(series, errors="coerce")

    if numeric_series.isnull().all():  # All values failed to convert to numeric
        print(f"Warning: Column {series.name} contains non-numeric data and cannot be a target. Skipping.")
        return None, None

    # Check if the original series (before numeric conversion for type check) looks like integers
    is_integer_like = (numeric_series.dropna() % 1 == 0).all()

    if is_integer_like:
        unique_values = numeric_series.dropna().unique()
        if len(unique_values) <= CLASSIFICATION_UNIQUE_THRESHOLD and len(unique_values) > 1:
            task_type = "CLASSIFICATION"
            # Ensure classes are 0-indexed if possible, or map them.
            # For simplicity, assume they are somewhat contiguous or can be mapped.
            # A more robust approach would involve sklearn.preprocessing.LabelEncoder
            min_val = int(numeric_series.min())
            max_val = int(numeric_series.max())
            if min_val == 0:
                task_details["num_classes"] = max_val + 1
            else:
                # If not starting at 0, this is a simplification.
                # Real scenario might need re-labeling or more complex num_classes logic.
                task_details["num_classes"] = len(unique_values)
                print(
                    f"Warning: Classification task {series.name} does not start from 0. num_classes set to {len(unique_values)}. Consider re-labeling."
                )

        else:  # Too many unique int values, or only one unique value (not a task)
            if len(unique_values) <= 1:
                print(f"Warning: Column {series.name} has only one unique value. Skipping as a task.")
                return None, None
            task_type = "REGRESSION"
    else:  # Float values
        task_type = "REGRESSION"

    return task_type, task_details


def generate_model_config(
    attributes_csv_path: Path, formula_features_csv_path: Path, output_config_path: Path, base_template: dict
):
    """
    Generates a model configuration YAML based on the attributes CSV.
    """
    if not attributes_csv_path.exists():
        print(f"Error: Attributes CSV file not found at {attributes_csv_path}")
        return

    attributes_df = pd.read_csv(attributes_csv_path)
    config = copy.deepcopy(base_template)

    # Update datamodule paths
    config["datamodule"]["attributes_source"] = str(attributes_csv_path.resolve())
    config["datamodule"]["formula_desc_source"] = str(formula_features_csv_path.resolve())
    # Ensure task_configs in datamodule points to model's task_configs
    config["datamodule"]["task_configs"] = "${model.task_configs}"

    # Infer tasks
    model_task_configs = []
    # Get default task config from template if available, otherwise use a minimal one
    default_task_template_reg = next(
        (tc for tc in base_template.get("model", {}).get("task_configs", []) if tc.get("type") == "REGRESSION"), None
    )
    if not default_task_template_reg:  # Fallback minimal regression task
        default_task_template_reg = {
            "type": "REGRESSION",
            "dims": [512, 128, 1],
            "norm": True,
            "weight": 1.0,
            "optimizer": {"optimizer_type": "AdamW", "lr": 0.001},
        }

    default_task_template_cls = next(
        (tc for tc in base_template.get("model", {}).get("task_configs", []) if tc.get("type") == "CLASSIFICATION"),
        None,
    )
    if not default_task_template_cls:  # Fallback minimal classification task
        default_task_template_cls = {
            "type": "CLASSIFICATION",
            "dims": [512, 64, 2],
            "num_classes": 2,
            "norm": True,
            "weight": 0.5,
            "optimizer": {"optimizer_type": "AdamW", "lr": 0.001},
        }

    for col_name in attributes_df.columns:
        if col_name.startswith("target_"):
            task_name = re.sub(r"^target_", "", col_name)
            task_type, task_details = infer_task_type_and_details(attributes_df[col_name])

            if task_type:
                print(f"Found task: {task_name} (Type: {task_type})")
                new_task_config = {}
                if task_type == "REGRESSION":
                    new_task_config = copy.deepcopy(default_task_template_reg)
                elif task_type == "CLASSIFICATION":
                    new_task_config = copy.deepcopy(default_task_template_cls)
                    new_task_config["num_classes"] = task_details.get("num_classes", 2)  # Default to 2 if not inferred
                    # Adjust last dim based on num_classes
                    if new_task_config.get("dims"):
                        new_task_config["dims"][-1] = new_task_config["num_classes"]

                new_task_config["name"] = task_name
                # Update monitor paths if they exist in the template
                if "optimizer" in new_task_config and "monitor" in new_task_config["optimizer"]:
                    new_task_config["optimizer"]["monitor"] = f"val_{task_name}_loss"  # Or a more specific metric
                model_task_configs.append(new_task_config)

    if not model_task_configs:
        print("Warning: No tasks were inferred from the attributes file. Model config will have no tasks.")

    config["model"]["task_configs"] = model_task_configs

    # Ensure output directory exists
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False, indent=2)

    print(f"Model configuration generated and saved to: {output_config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a model configuration YAML from an attributes CSV file.")
    parser.add_argument(
        "--attributes_csv",
        type=Path,
        default=DEFAULT_ATTRIBUTES_PATH,
        help=f"Path to the attributes CSV file (default: {DEFAULT_ATTRIBUTES_PATH})",
    )
    parser.add_argument(
        "--formula_features_csv",
        type=Path,
        default=DEFAULT_FORMULA_FEATURES_PATH,
        help=f"Path to the formula features CSV file (default: {DEFAULT_FORMULA_FEATURES_PATH})",
    )
    parser.add_argument(
        "--output_config",
        type=Path,
        default=DEFAULT_OUTPUT_CONFIG_PATH,
        help=f"Path to save the generated model config YAML (default: {DEFAULT_OUTPUT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--base_template",
        type=Path,
        default=BASE_MODEL_CONFIG_TEMPLATE_PATH,
        help=f"Path to the base model config YAML template (default: {BASE_MODEL_CONFIG_TEMPLATE_PATH})",
    )
    args = parser.parse_args()

    print(f"Loading base template from: {args.base_template}")
    base_template_config = load_base_template(args.base_template)

    print(f"Generating model config using attributes from: {args.attributes_csv}")
    generate_model_config(args.attributes_csv, args.formula_features_csv, args.output_config, base_template_config)
    print("Config generation complete.")
    # To run this script:
    # python samples/helper_tools/config_generator.py
    #
    # To install dependencies if not already present (using uv):
    # uv add pandas pyyaml
