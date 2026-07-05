#!/bin/bash
#SBATCH --partition=1n1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/ea0094/jobs/task_scaling/%x_%j.out
#SBATCH --error=/home/ea0094/jobs/task_scaling/%x_%j.err

# Task-scaling pretrain worker: 21 tasks, replay n=1000, random order per ORD (0/1/2).
# --seed/--set replicate exactly what pretrain.n_runs=3 would produce for run index ORD,
# but as parallel Slurm jobs. --resume makes resubmission after a walltime kill safe
# (the seeded shuffle keeps the task order stable across resubmits).
set -uo pipefail
PROJ=/home/ea0094/projects/foundation_model
CFG=$PROJ/experiments/rikyu_task_scaling/configs
: "${ORD:?need ORD=0|1|2}"
N=${N:-1000}                 # replay amount branch; 1000 keeps the original (untagged) dirs
DIRTAG=""; [ "$N" != "1000" ] && DIRTAG="_n${N}"
OUT=$PROJ/artifacts/task_scaling/pre${DIRTAG}_o${ORD}

module load nvhpc            # mandatory: GPU is not allocated without a loaded module
cd "$PROJ"
echo "== $(hostname) $(date) ord=$ORD n=$N seed=$((2025 + ORD)) task_order_seed=$((7501 + ORD)) =="
nvidia-smi --query-gpu=name --format=csv,noheader

t0=$SECONDS
"$PROJ/.venv/bin/fm" pretrain --config "$CFG/pre_n${N}.toml" \
    --output-dir "$OUT" --resume \
    --seed $((2025 + ORD)) \
    --set pretrain.task_order_seed=$((7501 + ORD)) 2>&1
rc=$?
echo "== fm exit=$rc total wall=$((SECONDS - t0))s =="
date
exit $rc
