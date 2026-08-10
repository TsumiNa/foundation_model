#!/bin/bash
#SBATCH --partition=ai-h200-brc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=24:00:00
#SBATCH --output=/home/users/u0001820/jobs/replay_epoch_p24/%x_%j.out
#SBATCH --error=/home/users/u0001820/jobs/replay_epoch_p24/%x_%j.err

# Hybrid-replay validation run: amount = max(1500, 0.3*N_task), m150 recipe (all baked into
# the config). ~68k labels/step at step 24 — expect ~22-24h; idempotent --resume on TIMEOUT.
source /etc/profile
set -euo pipefail
module load system/ai-h200-brc nvhpc

PROJ=/home/users/u0001820/projects/foundation_model-x86-cuda
cd "$PROJ"
flock .uv_sync.lock "$HOME/.local/bin/uv" sync --frozen --all-groups
.venv/bin/python -c 'import torch; print(torch.cuda.get_device_name(), torch.version.cuda, torch.cuda.is_available())'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "== $(hostname) $(date) commit=$(git rev-parse --short HEAD) =="
t0=$SECONDS
.venv/bin/fm pretrain \
  --config experiments/replay_epoch_sweep/configs/hybrid_full24.toml \
  --output-dir artifacts/replay_sweep_hybrid/hybrid_r03_f1500 \
  --resume
rc=$?
echo "== fm exit=$rc wall=$((SECONDS - t0))s =="
exit $rc
