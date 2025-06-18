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
#      --output_config samples/configs/test_t_depends/generated_model_config.yaml
#
# Ensure your Python environment with necessary dependencies (PyTorch, Lightning, etc.)
# and the foundation_model package are active.

# --- Configuration ---
# Path to the generated model configuration file
CONFIG_FILE="samples/configs/test_t_depends/fit_config.yaml"

# Directory to store logs for this specific run
LOG_DIR_BASE="samples/example_logs/basic_run"
EXPERIMENT_NAME="basic_experiment_$(date +%Y%m%d_%H%M%S)" # Unique experiment name
LOG_DIR="${LOG_DIR_BASE}/${EXPERIMENT_NAME}"

# 允许用户通过参数指定 LOG_DIR 和执行阶段
USER_LOG_DIR=""
USER_STAGES="fit,test,predict"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --log_dir)
            USER_LOG_DIR="$2"
            shift; shift
            ;;
        --stages)
            USER_STAGES="$2"
            shift; shift
            ;;
        *)
            echo "Unknown option: $1"; exit 1
            ;;
    esac
done

if [ -n "$USER_LOG_DIR" ]; then
    LOG_DIR="$USER_LOG_DIR"
fi
# 如果用户未指定 --log_dir，则 LOG_DIR 保持自动生成的默认值

IFS=',' read -ra STAGES <<< "$USER_STAGES"

# --- Script Execution ---
echo "--------------------------------------------------"
echo "Starting training"
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

# Set environment variables for logging
export LOG_DIR="$LOG_DIR"

for stage in "${STAGES[@]}"; do
    if [ "$stage" = "fit" ]; then
        CONFIG_FILE="samples/configs/test_t_depends/fit_config.yaml"
        echo "--------------------------------------------------"
        echo "Starting training"
        echo "--------------------------------------------------"
        echo "Using Config File: ${CONFIG_FILE}"
        echo "Logging to: ${LOG_DIR}"
        # Check if config file exists
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "Error: Configuration file '$CONFIG_FILE' not found."
            echo "Please run the prerequisite generation scripts first."
            exit 1
        fi
        mkdir -p "$LOG_DIR"
        export LOG_DIR="$LOG_DIR"
        fm-trainer fit --config "$CONFIG_FILE"
            # --trainer.default_root_dir "$LOG_DIR" 
            # --trainer.logger.1.init_args.save_dir "$LOG_DIR"
    fi

    if [ "$stage" = "test" ] || [ "$stage" = "predict" ]; then
        # 获取最新的 checkpoint
        CKPT_DIR="$LOG_DIR/fit/checkpoints"
        LATEST_CKPT=$(ls -t $CKPT_DIR/model-*-val_final_loss*.ckpt 2>/dev/null | head -1)
        if [ -z "$LATEST_CKPT" ]; then
            echo "No checkpoint found, using last.ckpt"
            LATEST_CKPT="$CKPT_DIR/last.ckpt"
        fi
        echo "Using checkpoint: $LATEST_CKPT"
    fi

    if [ "$stage" = "test" ]; then
        CONFIG_FILE="samples/configs/test_t_depends/test_config.yaml"
        echo "--------------------------------------------------"
        echo "Starting testing"
        echo "--------------------------------------------------"
        echo "Using Config File: ${CONFIG_FILE}"
        echo "Logging to: ${LOG_DIR}"
        fm-trainer test --config "$CONFIG_FILE" --ckpt_path "$LATEST_CKPT"
            # --trainer.default_root_dir "$LOG_DIR" 
            # --trainer.logger.1.init_args.save_dir "$LOG_DIR"
    fi

    if [ "$stage" = "predict" ]; then
        CONFIG_FILE="samples/configs/test_t_depends/predict_config.yaml"
        echo "--------------------------------------------------"
        echo "Starting prediction"
        echo "--------------------------------------------------"
        echo "Using Config File: ${CONFIG_FILE}"
        echo "Logging to: ${LOG_DIR}"
        # 删除已存在的 config.yaml，避免 LightningCLI 报错
        PREDICT_CONFIG_DIR="$LOG_DIR/predict"
        PREDICT_CONFIG_FILE="$PREDICT_CONFIG_DIR/config.yaml"
        if [ -f "$PREDICT_CONFIG_FILE" ]; then
            echo "Removing existing $PREDICT_CONFIG_FILE to avoid LightningCLI overwrite error."
            rm "$PREDICT_CONFIG_FILE"
        fi
        fm-trainer predict --config "$CONFIG_FILE" --ckpt_path "$LATEST_CKPT"
            # --trainer.default_root_dir "$LOG_DIR" \
            # --trainer.logger.1.init_args.save_dir "$LOG_DIR"
    fi
done

echo "--------------------------------------------------"
echo "All tasks completed."
echo "Check logs and results in: ${LOG_DIR}"
echo "--------------------------------------------------"

# Note:
# - This script assumes `foundation_model.scripts.train` is the entry point for your Lightning CLI.
# - Adjust paths and Python command if your project structure or entry point differs.
# - For actual training, you might want to remove `trainer.max_epochs` override or set it higher
#   if it's defined low in the generated_model_config.yaml for quick testing.
