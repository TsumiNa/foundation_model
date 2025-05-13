# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for helper scripts in the samples directory.
"""

import argparse
import copy
from pathlib import Path

import yaml  # PyYAML


def modify_task_masking_ratio_in_config(
    base_config_path: Path, task_to_scale: str, ratio: float, output_config_path: Path
):
    """
    Loads a base YAML configuration, modifies the task_masking_ratios for a specific task,
    and saves it to a new path. Other tasks will have their ratios set to 1.0.

    Args:
        base_config_path (Path): Path to the base YAML configuration file.
        task_to_scale (str): The name of the task whose data ratio is to be scaled.
        ratio (float): The masking ratio to apply to the task_to_scale (0.0 to 1.0).
        output_config_path (Path): Path to save the modified YAML configuration.
    """
    if not base_config_path.exists():
        print(f"Error: Base config file not found at {base_config_path}")
        raise FileNotFoundError(f"Base config file not found at {base_config_path}")

    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        print(f"Error: Could not load or parse YAML from {base_config_path}")
        raise ValueError(f"Could not load or parse YAML from {base_config_path}")

    modified_config = copy.deepcopy(config)

    # Ensure datamodule and task_masking_ratios keys exist
    if "datamodule" not in modified_config:
        modified_config["datamodule"] = {}

    task_masking_ratios = {}

    # Get all task names from the model configuration to ensure all are covered
    all_task_names = []
    if "model" in modified_config and "task_configs" in modified_config["model"]:
        for task_cfg in modified_config["model"]["task_configs"]:
            if "name" in task_cfg:
                all_task_names.append(task_cfg["name"])

    if not all_task_names:
        print(
            f"Warning: No task names found in model.task_configs in {base_config_path}. "
            f"Setting ratio for '{task_to_scale}' only."
        )

    # Set ratio for the target task and 1.0 for others
    for name in all_task_names:
        if name == task_to_scale:
            task_masking_ratios[name] = float(ratio)
        else:
            task_masking_ratios[name] = 1.0

    # If task_to_scale was not in all_task_names (e.g. typo or not in base config's model tasks)
    # still add it, as it might be a valid task the user wants to control.
    if task_to_scale not in task_masking_ratios and all_task_names:  # only if all_task_names was populated
        print(
            f"Warning: Task '{task_to_scale}' not found in model.task_configs. Adding it to task_masking_ratios anyway."
        )
        task_masking_ratios[task_to_scale] = float(ratio)
    elif not all_task_names:  # if no tasks were found in model config, just set the one requested
        task_masking_ratios[task_to_scale] = float(ratio)

    modified_config["datamodule"]["task_masking_ratios"] = task_masking_ratios

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w") as f:
        yaml.dump(modified_config, f, sort_keys=False, indent=2)

    print(f"Modified config for task '{task_to_scale}' with ratio {ratio} saved to: {output_config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify task_masking_ratio in a YAML config file for a specific task.")
    parser.add_argument("--base_config", type=Path, required=True, help="Path to the base YAML configuration file.")
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task to scale.")
    parser.add_argument(
        "--ratio", type=float, required=True, help="Masking ratio for the specified task (e.g., 0.5 for 50%% data)."
    )
    parser.add_argument(
        "--output_config", type=Path, required=True, help="Path to save the modified YAML configuration."
    )
    args = parser.parse_args()

    modify_task_masking_ratio_in_config(
        base_config_path=args.base_config,
        task_to_scale=args.task_name,
        ratio=args.ratio,
        output_config_path=args.output_config,
    )
    # Example usage:
    # python samples/helper_tools/utils.py \
    #   --base_config samples/generated_configs/generated_model_config.yaml \
    #   --task_name regression_1 \
    #   --ratio 0.2 \
    #   --output_config samples/generated_configs/temp_scaling_configs/config_regression_1_ratio_0.2.yaml
