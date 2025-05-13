#!/bin/bash
# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

# --- CLI Example 1: Basic Training Run ---
#
# This script demonstrates a basic training execution using the Lightning CLI.
#
# Prerequisites:
# 1. Fake data must be generated:
#    python samples/helper_tools/fake_data_generator.py
# 2. A model configuration must be generated from the fake data:
#    python samples/helper_tools/config_generator.py \
#      --attributes_csv samples/fake_data/attributes.csv \
#      --formula_features_csv samples/fake_data/formula_features.csv \
#      --output_config samples/generated_configs/generated_model_config.yaml
#
# Ensure your Python environment with necessary dependencies (PyTorch, Lightning, etc.)
# and the foundation_model package are active.

# --- Configuration ---
# Path to the generated model configuration file
CONFIG_FILE="samples/generated_configs/generated_model_config.yaml"

# Directory to store logs for this specific run
LOG_DIR_BASE="samples/example_logs/basic_run"
EXPERIMENT_NAME="basic_experiment_$(date +%Y%m%d_%H%M%S)" # Unique experiment name
LOG_DIR="${LOG_DIR_BASE}/${EXPERIMENT_NAME}"

# --- Script Execution ---
echo "--------------------------------------------------"
echo "Starting Basic Training Run"
echo "--------------------------------------------------"
echo "Using Config File: ${CONFIG_FILE}"
echo "Logging to: ${LOG_DIR}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found."
    echo "Please run the prerequisite generation scripts first."
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Execute the training script
# The `fit` subcommand is standard for PyTorch Lightning CLI.
# We override logger.save_dir and callbacks[0].dirpath (ModelCheckpoint)
# to ensure logs and checkpoints for this example run are stored in a specific location.
# Note: The exact index for ModelCheckpoint in callbacks might vary if base_model.yaml changes.
# We also set trainer.logger.name to an empty string so that version_X is created directly under LOG_DIR.
# The YAML now handles pathing based on trainer.default_root_dir.
# We pass the fully resolved LOG_DIR (which includes the timestamped experiment name)
# directly as the default_root_dir for the trainer.
python -m foundation_model.scripts.train fit \
    --config "$CONFIG_FILE" \
    --trainer.default_root_dir="$LOG_DIR"

# The trainer.default_root_dir in the YAML is a static fallback.
# The CLI argument --trainer.default_root_dir="$LOG_DIR" overrides it.
# Loggers and ModelCheckpoint will then use this resolved default_root_dir.

echo "--------------------------------------------------"
echo "Basic Training Run Command Executed."
echo "Check logs and results in: ${LOG_DIR}"
echo "--------------------------------------------------"

# Note:
# - This script assumes `foundation_model.scripts.train` is the entry point for your Lightning CLI.
# - Adjust paths and Python command if your project structure or entry point differs.
# - For actual training, you might want to remove `trainer.max_epochs` override or set it higher
#   if it's defined low in the generated_model_config.yaml for quick testing.
