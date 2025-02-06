#!/usr/bin/env bash

# Define base parameters
python=/data/.miniforge3/envs/xepy311/bin/python
max_epochs=300
log_dir="results/20250206_logs"
random_seed=1234
num_workers=0
devices=4

# Fixed parameters
shared_lr=0.001
task_lr=0.01
num_ensembles=5  # Number of runs for each configuration

# Define array for mp_attrs_rate
mp_attrs_rates=(0.05 0.0)

# Loop through parameter combinations and ensemble runs
for mp_rate in "${mp_attrs_rates[@]}"; do
    for ensemble_id in $(seq 1 $num_ensembles); do
        # Use ensemble_id in random seed to ensure different seeds
        current_seed=$((random_seed + ensemble_id))
        exp_name="mp_${mp_rate}-shared_lr=${shared_lr}_task_lr=${task_lr}"
        
        cmd="$python -m foundation_model.scripts.train \
            --max_epochs=$max_epochs \
            --devices=$devices \
            --num_workers=$num_workers \
            --mp_attrs_rate=$mp_rate \
            --log_dir=\"$log_dir\" \
            --exp_name \"$exp_name\""
        
        echo "Running experiment: $exp_name"
        eval $cmd
        
        # Optional: add sleep between runs if needed
        sleep 10
    done
done
