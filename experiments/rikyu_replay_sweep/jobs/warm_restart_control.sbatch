#!/bin/bash
#SBATCH --partition=1n1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/home/ea0094/jobs/task_scaling/%x_%j.out
#SBATCH --error=/home/ea0094/jobs/task_scaling/%x_%j.err

# Warm-restart control for the replay sweep's above-ceiling puzzle: same model structure, same
# FIXED full training data, single task, NO other tasks — retrain 10 times in a chain where each
# iteration is a fresh optimizer (fm finetune with freeze_encoder=false), mimicking the
# per-step optimizer reset of continual pretraining. If the metric climbs above the single-round
# early-stopping level here too, optimizer restarts (not multi-task transfer) explain the
# above-ceiling finals in the sweep.
#   it00 = single-task pretrain (the at-intro analogue); it01..it10 = warm restarts.
set -uo pipefail
PROJ=/home/ea0094/projects/foundation_model
CFG=$PROJ/experiments/rikyu_task_scaling/configs
FM=$PROJ/.venv/bin/fm
: "${TASK:?need TASK=<task name>}"
BASE=$PROJ/artifacts/warm_restart/${TASK}

module load nvhpc
cd "$PROJ"
echo "== $(hostname) $(date) warm-restart control task=$TASK =="
nvidia-smi --query-gpu=name --format=csv,noheader

rc=0
if [ ! -f "$BASE/it00/training/final_model.pt" ]; then
    "$FM" pretrain --config "$CFG/ws_n1000.toml" --output-dir "$BASE/it00" --seed 2025 \
        --set "pretrain.task_sequence=[\"$TASK\"]" 2>&1 || { echo "!! it00 FAILED"; exit 1; }
fi
prev=$BASE/it00/training/final_model.pt
for i in $(seq 1 10); do
    ii=$(printf "%02d" "$i")
    OUT=$BASE/it${ii}
    if [ ! -f "$OUT/training/finetune_summary.json" ]; then
        t0=$SECONDS
        "$FM" finetune --config "$CFG/ft.toml" --checkpoint "$prev" --output-dir "$OUT" --seed 2025 \
            --set "finetune.tasks=[\"$TASK\"]" --set finetune.freeze_encoder=false 2>&1
        if [ $? -ne 0 ]; then
            echo "!! it$ii FAILED"
            rc=1
            break
        fi
        echo "-- it$ii done in $((SECONDS - t0))s"
    else
        echo "-- it$ii already done, skip"
    fi
    prev=$OUT/training/final_model.pt
done
echo "== warm-restart $TASK exit=$rc =="
date
exit $rc
