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



# Directory to store logs for this specific run
LOG_DIR_BASE="samples/example_logs/basic_run"
EXPERIMENT_NAME="basic_experiment_$(date +%Y%m%d_%H%M%S)" # Unique experiment name
LOG_DIR="${LOG_DIR_BASE}/${EXPERIMENT_NAME}"

# 允许用户通过参数指定 LOG_DIR、执行阶段、配置目录和checkpoint路径
USER_LOG_DIR=""
USER_STAGES="fit,test,predict"
USER_CONFIG_DIR=""
USER_CKPT_PATH=""

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
        --config_dir)
            USER_CONFIG_DIR="$2"
            shift; shift
            ;;
        --ckpt_path)
            USER_CKPT_PATH="$2"
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

if [ -n "$USER_CONFIG_DIR" ]; then
    CONFIG_DIR="$USER_CONFIG_DIR"
else
    CONFIG_DIR="samples/configs/test_t_depends"
fi

IFS=',' read -ra STAGES <<< "$USER_STAGES"

# Save the base LOG_DIR for consistent reference throughout the script
BASE_LOG_DIR="$LOG_DIR"

# Helper function to ensure LOG_DIR is properly set
ensure_log_dir() {
    if [ -z "$LOG_DIR" ]; then
        LOG_DIR="$BASE_LOG_DIR"
        export LOG_DIR="$LOG_DIR"
        echo "LOG_DIR was empty, restored to: $LOG_DIR"
    fi
}

# Function to find the best checkpoint according to the specified logic
find_best_checkpoint() {
    local ckpt_dir="${LOG_DIR}/fit/checkpoints"
    
    # 1. 如果用户指定了checkpoint路径，直接使用
    if [ -n "$USER_CKPT_PATH" ]; then
        if [ -f "$USER_CKPT_PATH" ]; then
            echo "$USER_CKPT_PATH"
            return 0
        else
            echo "Error: User specified checkpoint not found: $USER_CKPT_PATH" >&2
            return 1
        fi
    fi
    
    # 2. 从${LOG_DIR}/fit/checkpoints寻找最佳checkpoint
    if [ -d "$ckpt_dir" ]; then
        # 优先查找best checkpoint (按val_final_loss排序)
        local best_ckpt=$(ls -t "$ckpt_dir"/model-*-val_final_loss*.ckpt 2>/dev/null | head -1)
        if [ -n "$best_ckpt" ]; then
            echo "$best_ckpt"
            return 0
        fi
        
        # 如果没有best checkpoint，查找last.ckpt
        local last_ckpt="$ckpt_dir/last.ckpt"
        if [ -f "$last_ckpt" ]; then
            echo "$last_ckpt"
            return 0
        fi
    fi
    
    # 3. 没有找到checkpoint
    return 1
}

# Create log directory
mkdir -p "$LOG_DIR"

# Set environment variables for logging
export LOG_DIR="$LOG_DIR"
echo "Initial LOG_DIR set to: $LOG_DIR"

for stage in "${STAGES[@]}"; do
    if [ "$stage" = "fit" ]; then
        # 检查是否有多个 fit*.yaml 配置文件，按编号排序
        FIT_CONFIG_FILES=($(ls "$CONFIG_DIR"/fit*.yaml 2>/dev/null | sort -V))
        if [ ${#FIT_CONFIG_FILES[@]} -eq 0 ]; then
            echo "Error: No fit*.yaml configuration files found in $CONFIG_DIR."
            echo "Please run the prerequisite generation scripts first."
            exit 1
        fi

        # 保存原始LOG_DIR值
        ORIGINAL_LOG_DIR="$LOG_DIR"
        
        # 设置全局CKPT_DIR，所有fit阶段共享
        CKPT_DIR="${LOG_DIR}/fit/checkpoints"
        export CKPT_DIR="$CKPT_DIR"
        mkdir -p "$CKPT_DIR"

        PREV_CKPT=""
        for ((i=0; i<${#FIT_CONFIG_FILES[@]}; i++)); do
            CONFIG_FILE="${FIT_CONFIG_FILES[$i]}"
            FIT_STAGE_NAME="step$((i+1))"  # step1, step2, etc.
            FIT_LOG_DIR="${ORIGINAL_LOG_DIR}/fit/${FIT_STAGE_NAME}"
            echo "--------------------------------------------------"
            echo "Starting training: $FIT_STAGE_NAME ($(basename "$CONFIG_FILE"))"
            echo "--------------------------------------------------"
            echo "Using Config File: ${CONFIG_FILE}"
            echo "Logging to: ${FIT_LOG_DIR}"
            echo "Checkpoints to: ${CKPT_DIR}"

            mkdir -p "$FIT_LOG_DIR"

            # 检查配置文件是否存在
            if [ ! -f "$CONFIG_FILE" ]; then
                echo "Error: Configuration file '$CONFIG_FILE' not found."
                exit 1
            fi

            # 设置当前阶段的LOG_DIR环境变量
            export LOG_DIR="$FIT_LOG_DIR"
            
            # 从上一阶段的best checkpoint开始（除了第一阶段）
            if [ -n "$PREV_CKPT" ]; then
                echo "Resuming from checkpoint: $PREV_CKPT"
                fm-trainer fit --config "$CONFIG_FILE" --ckpt_path "$PREV_CKPT"
            elif [ $i -eq 0 ] && [ -n "$USER_CKPT_PATH" ]; then
                # 第一阶段且用户指定了checkpoint
                echo "Starting first stage from user-specified checkpoint: $USER_CKPT_PATH"
                fm-trainer fit --config "$CONFIG_FILE" --ckpt_path "$USER_CKPT_PATH"
            else
                echo "Starting training from scratch"
                fm-trainer fit --config "$CONFIG_FILE"
            fi
            
            # 重置LOG_DIR环境变量
            unset LOG_DIR

            # 获取本阶段训练后的最佳ckpt，供下一个阶段使用
            BEST_CKPT=$(ls -t $CKPT_DIR/model-*-val_final_loss*.ckpt 2>/dev/null | head -1)
            if [ -z "$BEST_CKPT" ]; then
                BEST_CKPT="$CKPT_DIR/last.ckpt"
            fi
            PREV_CKPT="$BEST_CKPT"
            FINAL_FIT_LOG_DIR="$FIT_LOG_DIR"
            echo "Best checkpoint for next stage: $PREV_CKPT"
        done

        # 记录最后一个fit阶段的ckpt和log目录，供test/predict使用
        FINAL_FIT_CKPT="$PREV_CKPT"
        FINAL_FIT_LOG_DIR="$FINAL_FIT_LOG_DIR"
        
        # 清理fit阶段的环境变量并恢复原始LOG_DIR
        unset CKPT_DIR
        export LOG_DIR="$ORIGINAL_LOG_DIR"
    fi

    if [ "$stage" = "test" ]; then
        # 确保LOG_DIR正确设置
        ensure_log_dir
        
        CONFIG_FILE="$CONFIG_DIR/test_config.yaml"
        TEST_LOG_DIR="$LOG_DIR/test"
        mkdir -p "$TEST_LOG_DIR"
        echo "--------------------------------------------------"
        echo "Starting testing"
        echo "--------------------------------------------------"
        echo "Using Config File: ${CONFIG_FILE}"
        echo "Logging to: ${TEST_LOG_DIR}"
        
        # 查找checkpoint
        CHECKPOINT=$(find_best_checkpoint)
        if [ $? -ne 0 ]; then
            echo "Error: No checkpoint found for test stage."
            echo "Please ensure fit stage has been completed or specify --ckpt_path"
            exit 1
        fi
        echo "Using checkpoint: $CHECKPOINT"
        
        # 设置LOG_DIR环境变量在执行命令之前，确保配置文件能正确解析路径
        export LOG_DIR="$TEST_LOG_DIR"
        echo "LOG_DIR set to: $LOG_DIR"
        fm-trainer test --config "$CONFIG_FILE" --ckpt_path "$CHECKPOINT"
        unset LOG_DIR
        # 恢复原始LOG_DIR
        export LOG_DIR="$BASE_LOG_DIR"
    fi

    if [ "$stage" = "predict" ]; then
        # 确保LOG_DIR正确设置
        ensure_log_dir
        
        CONFIG_FILE="$CONFIG_DIR/predict_config.yaml"
        PREDICT_LOG_DIR="$LOG_DIR/predict"
        mkdir -p "$PREDICT_LOG_DIR"
        echo "--------------------------------------------------"
        echo "Starting prediction"
        echo "--------------------------------------------------"
        echo "Using Config File: ${CONFIG_FILE}"
        echo "Logging to: ${PREDICT_LOG_DIR}"
        
        # 查找checkpoint
        CHECKPOINT=$(find_best_checkpoint)
        if [ $? -ne 0 ]; then
            echo "Error: No checkpoint found for predict stage."
            echo "Please ensure fit stage has been completed or specify --ckpt_path"
            exit 1
        fi
        echo "Using checkpoint: $CHECKPOINT"
        
        # 删除已存在的 config.yaml，避免 LightningCLI 报错
        PREDICT_CONFIG_FILE="$PREDICT_LOG_DIR/config.yaml"
        if [ -f "$PREDICT_CONFIG_FILE" ]; then
            echo "Removing existing $PREDICT_CONFIG_FILE to avoid LightningCLI overwrite error."
            rm "$PREDICT_CONFIG_FILE"
        fi
        # 设置LOG_DIR环境变量在执行命令之前，确保配置文件中的回调能正确解析路径
        export LOG_DIR="$PREDICT_LOG_DIR"
        echo "LOG_DIR set to: $LOG_DIR"
        fm-trainer predict --config "$CONFIG_FILE" --ckpt_path "$CHECKPOINT"
        unset LOG_DIR
        # 恢复原始LOG_DIR
        export LOG_DIR="$BASE_LOG_DIR"
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
