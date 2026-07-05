#!/bin/bash
#SBATCH --partition=1n1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/home/ea0094/jobs/rehearsal/sweep/%x_%j.out
#SBATCH --error=/home/ea0094/jobs/rehearsal/sweep/%x_%j.err

# Replay-sweep worker. Config selected via the SWEEP_CFG env var (--export). Each config's
# [output].dir points at its own artifacts/replay_sweep/replay_<tag>_rikyu directory.
set -uo pipefail
PROJ=/home/ea0094/projects/foundation_model
CFG=/home/ea0094/jobs/rehearsal/sweep/${SWEEP_CFG}

module load nvhpc            # mandatory: GPU is not allocated without a loaded module
cd "$PROJ"
echo "== $(hostname) $(date) =="
echo "cfg=$CFG"
grep -E "default_replay|^dir" "$CFG"
nvidia-smi --query-gpu=name --format=csv,noheader

t0=$SECONDS
"$PROJ/.venv/bin/fm" pretrain --config "$CFG" 2>&1
rc=$?
echo "== fm exit=$rc total wall=$((SECONDS - t0))s =="
date
exit $rc
