#!/bin/bash
#SBATCH --partition=1n1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/home/ea0094/jobs/task_scaling/%x_%j.out
#SBATCH --error=/home/ea0094/jobs/task_scaling/%x_%j.err

# Task-scaling finetune worker for run ORD: for every pretrain step checkpoint k=1..21,
# (a) frozen-encoder finetune of the 3 dielectric targets (epochs<=100 + early stopping),
# (b) inverse design on the finetuned model. Skip-if-done per k; safe to resubmit.
set -uo pipefail
PROJ=/home/ea0094/projects/foundation_model
CFG=$PROJ/experiments/rikyu_task_scaling/configs
FM=$PROJ/.venv/bin/fm
: "${ORD:?need ORD=0|1|2}"
PRE=$PROJ/artifacts/task_scaling/pre_o${ORD}/training

module load nvhpc
cd "$PROJ"
echo "== $(hostname) $(date) ft ord=$ORD =="
nvidia-smi --query-gpu=name --format=csv,noheader

rc=0
for k in $(seq 1 21); do
    kk=$(printf "%02d" "$k")
    step_dir=$(ls -d "$PRE"/step${kk}_* 2>/dev/null | head -1)
    ckpt="$step_dir/checkpoint.pt"
    if [ ! -f "$ckpt" ]; then
        echo "!! missing pretrain checkpoint for k=$kk — skipping"
        rc=1
        continue
    fi
    OUT=$PROJ/artifacts/task_scaling/ft_o${ORD}/k${kk}
    t0=$SECONDS
    if [ -f "$OUT/training/finetune_summary.json" ]; then
        echo "-- k=$kk ft: already done, skip"
    else
        "$FM" finetune --config "$CFG/ft.toml" --checkpoint "$ckpt" \
            --output-dir "$OUT" --seed 2025 2>&1
        if [ $? -ne 0 ]; then
            echo "!! k=$kk ft FAILED"
            rc=1
            continue
        fi
        echo "-- k=$kk ft done in $((SECONDS - t0))s"
    fi
    INV=$OUT/inverse
    t1=$SECONDS
    if [ -f "$INV/inverse_design.json" ]; then
        echo "-- k=$kk inverse: already done, skip"
    else
        "$FM" inverse --config "$CFG/inv.toml" --checkpoint "$OUT/training/final_model.pt" \
            --output-dir "$INV" 2>&1
        if [ $? -ne 0 ]; then
            echo "!! k=$kk inverse FAILED"
            rc=1
            continue
        fi
        echo "-- k=$kk inverse done in $((SECONDS - t1))s"
    fi
done
echo "== ft ord=$ORD exit=$rc =="
date
exit $rc
