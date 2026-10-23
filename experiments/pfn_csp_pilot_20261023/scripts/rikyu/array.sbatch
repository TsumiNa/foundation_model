#!/bin/bash
# One array task = one GPU, 32 CPUs, PACK units of units.txt run concurrently.
#   sbatch --account=<project> --array=0-<N/PACK> scripts/rikyu/array.sbatch
#SBATCH --job-name=pfn-csp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
set -u
EXP="$SLURM_SUBMIT_DIR"
cd "$EXP"
PACK=${PACK:-16}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CATBOOST_THREADS=2
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
start=$((SLURM_ARRAY_TASK_ID * PACK))
UNITS_FILE=${UNITS_FILE:-scripts/rikyu/units.txt}
mapfile -t UNITS < <(sed -n "$((start + 1)),$((start + PACK))p" "$UNITS_FILE")
for u in "${UNITS[@]}"; do
  # shellcheck disable=SC2086
  .venv/bin/python scripts/run_experiment.py $u > "artifacts/logs/unit_$(echo $u | tr ' -' '__').log" 2>&1 &
done
wait
grep -L "wrote\|exists" artifacts/logs/unit_*.log 2>/dev/null | head
