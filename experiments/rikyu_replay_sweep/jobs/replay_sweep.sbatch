#!/bin/bash
#SBATCH --partition=1n1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Replay-sweep worker. Config selected via the SWEEP_CFG env var (--export). Each config's
# [output].dir points at its own artifacts/replay_sweep/replay_<tag>_rikyu directory.
set -uo pipefail
PROJ=${PROJ:-$HOME/projects/foundation_model}
CFG=${CFG:-$HOME/jobs/rehearsal/sweep/${SWEEP_CFG}}

module load nvhpc            # expose the NVIDIA CUDA runtime and tools
cd "$PROJ"
echo "== $(hostname) $(date) =="
echo "cfg=$CFG"
grep -E "^amount|^dir" "$CFG"
nvidia-smi --query-gpu=name --format=csv,noheader

t0=$SECONDS
"$PROJ/.venv/bin/fm" pretrain --config "$CFG" 2>&1
rc=$?
echo "== fm exit=$rc total wall=$((SECONDS - t0))s =="
date
exit $rc
