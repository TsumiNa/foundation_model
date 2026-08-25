#!/bin/bash
#SBATCH --job-name=fm-full24
#SBATCH --partition=1n1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=full24_%j.out
#SBATCH --error=full24_%j.err

# Full 24-task continual-rehearsal reproduction on GB200. Full data, 100 epochs/step
# (early stopping). Output to persistent Lustre for rsync-back.
set -uo pipefail
PROJ=${PROJ:-$HOME/projects/foundation_model}
CFG=${CFG:-$HOME/jobs/rehearsal/pretrain_rehearsal_full24.toml}
OUT="$PROJ/artifacts/pretrain_rehearsal_full24_rikyu"

module load nvhpc            # expose the NVIDIA CUDA runtime and tools
cd "$PROJ"
echo "== node =="; hostname; date
echo "SLURM_JOB_ID=$SLURM_JOB_ID OUT=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== fm pretrain full24 =="
t0=$SECONDS
"$PROJ/.venv/bin/fm" pretrain --config "$CFG" --output-dir "$OUT" 2>&1
rc=$?
echo "== fm exit=$rc, total wall=$((SECONDS - t0))s =="

echo "== final metrics table =="
find "$OUT" -name metrics_table.csv -exec cat {} \; 2>/dev/null
echo "== done =="; date
exit $rc
