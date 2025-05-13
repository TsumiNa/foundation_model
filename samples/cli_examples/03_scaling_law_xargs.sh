#!/bin/bash
# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

# --- CLI Example 3: Scaling Law Study with Task Masking Ratios ---
#
# This script demonstrates how to run multiple training experiments to study
# the scaling law for a specific task by varying its data availability
# using `datamodule.task_masking_ratios`.
#
# It uses a loop to iterate through different ratios and calls a Python helper
# script (`samples/helper_tools/utils.py`) to generate a temporary config
# file for each ratio.
#
# Prerequisites:
# 1. Fake data must be generated:
#    python samples/helper_tools/fake_data_generator.py
# 2. A base model configuration must be generated:
#    python samples/helper_tools/config_generator.py \
#      --attributes_csv samples/fake_data/attributes.csv \
#      --formula_features_csv samples/fake_data/formula_features.csv \
#      --output_config samples/generated_configs/default_config.yaml
# 3. The utility script for modifying configs must exist:
#    samples/helper_tools/utils.py
#
# Ensure your Python environment and the foundation_model package are active.

# --- Configuration ---
# Path to the base model configuration file
BASE_CONFIG_FILE="samples/generated_configs/default_config.yaml"

# Task name (from attributes.csv, without 'target_') to study scaling for.
# This MUST match a task name present in the $BASE_CONFIG_FILE model.task_configs section.
# Example: If attributes.csv has 'target_regression_1', this should be 'regression_1'.
TASK_TO_SCALE="regression_1" # ADJUST THIS BASED ON YOUR GENERATED TASKS

# Directory to store logs for this series of scaling experiments
LOG_SERIES_DIR="samples/example_logs/scaling_law_${TASK_TO_SCALE}"

# Directory to store temporary configuration files generated for each ratio
TEMP_CONFIG_DIR="samples/generated_configs/temp_scaling_configs_${TASK_TO_SCALE}"

# Ratios of data to use for the TASK_TO_SCALE
# Other tasks will default to using 100% of their data (ratio 1.0) as per utils.py logic.
RATIOS_TO_TEST=(0.1 0.25 0.5 0.75 1.0)
# RATIOS_TO_TEST=(0.05 0.1 0.2 0.4 0.6 0.8 1.0) # More granular example

# Python interpreter (if not in PATH or using a specific venv)
PYTHON_EXEC="python" # Or specify path e.g., /path/to/venv/bin/python

# Path to the utility script
CONFIG_MODIFIER_SCRIPT="samples/helper_tools/utils.py"

# --- Script Execution ---
echo "--------------------------------------------------"
echo "Starting Scaling Law Experiment Series"
echo "Task to scale: ${TASK_TO_SCALE}"
echo "Base Config: ${BASE_CONFIG_FILE}"
echo "Log Series Dir: ${LOG_SERIES_DIR}"
echo "Temp Config Dir: ${TEMP_CONFIG_DIR}"
echo "Ratios to test: ${RATIOS_TO_TEST[*]}"
echo "--------------------------------------------------"

# Validate prerequisites
if [ ! -f "$BASE_CONFIG_FILE" ]; then
    echo "Error: Base configuration file '$BASE_CONFIG_FILE' not found."
    echo "Please run prerequisite generation scripts first."
    exit 1
fi
if [ ! -f "$CONFIG_MODIFIER_SCRIPT" ]; then
    echo "Error: Config modifier script '$CONFIG_MODIFIER_SCRIPT' not found."
    exit 1
fi
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo "Error: Python executable '$PYTHON_EXEC' not found."
    exit 1
fi

# Create directories
mkdir -p "$LOG_SERIES_DIR"
mkdir -p "$TEMP_CONFIG_DIR"

# Loop through each ratio and run training
for RATIO in "${RATIOS_TO_TEST[@]}"; do
    echo ""
    echo "--- Processing Ratio: $RATIO for Task: $TASK_TO_SCALE ---"

    # Define paths for this specific run
    RUN_ID="ratio_$(printf "%.2f" "$RATIO" | sed 's/\.//g')" # e.g., ratio_025
    CURRENT_TEMP_CONFIG_FILE="${TEMP_CONFIG_DIR}/config_${TASK_TO_SCALE}_${RUN_ID}.yaml"
    CURRENT_LOG_DIR="${LOG_SERIES_DIR}/${RUN_ID}"
    EXPERIMENT_NAME="scaling_${TASK_TO_SCALE}_${RUN_ID}_$(date +%Y%m%d_%H%M%S)"

    mkdir -p "$CURRENT_LOG_DIR"

    echo "Generating temporary config: $CURRENT_TEMP_CONFIG_FILE"
    $PYTHON_EXEC "$CONFIG_MODIFIER_SCRIPT" \
        --base_config "$BASE_CONFIG_FILE" \
        --task_name "$TASK_TO_SCALE" \
        --ratio "$RATIO" \
        --output_config "$CURRENT_TEMP_CONFIG_FILE"

    if [ ! -f "$CURRENT_TEMP_CONFIG_FILE" ]; then
        echo "Error: Failed to generate temporary config file for ratio $RATIO."
        continue # Skip to next ratio
    fi

    echo "Starting training run for ratio $RATIO..."
    echo "Logs will be in: $CURRENT_LOG_DIR"

    $PYTHON_EXEC -m foundation_model.scripts.train fit \
        --config "$CURRENT_TEMP_CONFIG_FILE" \
        trainer.logger.save_dir="$CURRENT_LOG_DIR" \
        trainer.logger.name="" \
        trainer.callbacks[0].dirpath="$CURRENT_LOG_DIR/checkpoints/" \
        trainer.max_epochs=3 \
        experiment_name="$EXPERIMENT_NAME" # Set max_epochs low for quick demo runs

    echo "Finished training run for ratio $RATIO."
done

echo ""
echo "--------------------------------------------------"
echo "All Scaling Law Experiment Runs Submitted."
echo "Log series stored in: ${LOG_SERIES_DIR}"
echo "Temporary configs stored in: ${TEMP_CONFIG_DIR}"
echo "After completion, use samples/helper_tools/scaling_law_analyzer.py (to be created)"
echo "e.g., python samples/helper_tools/scaling_law_analyzer.py --log_dir ${LOG_SERIES_DIR} --task_name ${TASK_TO_SCALE}"
echo "--------------------------------------------------"

# Note:
# - The `TASK_TO_SCALE` variable must accurately reflect a task name in your generated config.
# - `trainer.max_epochs=3` is set for quick demonstration. For actual scaling law studies,
#   you'd use a much larger number of epochs or rely on early stopping.
# - The `scaling_law_analyzer.py` script, mentioned for post-processing, still needs to be created.
# - Consider using `nohup` or a job scheduler for long-running experiment series.
# - If using xargs was a strict requirement, this loop could be adapted,
#   but this direct loop is often clearer for managing individual log directories and configs.
