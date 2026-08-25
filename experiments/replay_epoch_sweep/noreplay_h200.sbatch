#!/bin/bash
#SBATCH --partition=ai-h200-brc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Baseline arm A: sequential 24-task pretrain with NO replay (interval 999), patience 24,
# max_epochs 150 — the "what if we just don't replay" forgetting trajectory. Idempotent --resume.
source /etc/profile
set -euo pipefail
module load system/ai-h200-brc nvhpc

PROJ=${PROJ:-$HOME/projects/foundation_model-x86-cuda}
cd "$PROJ"
flock .uv_sync.lock "$HOME/.local/bin/uv" sync --frozen --all-groups
.venv/bin/python -c 'import torch; print(torch.cuda.get_device_name(), torch.version.cuda, torch.cuda.is_available())'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "== $(hostname) $(date) commit=$(git rev-parse --short HEAD) =="
t0=$SECONDS
.venv/bin/fm pretrain \
  --config experiments/replay_epoch_sweep/configs/noreplay_full24.toml \
  --output-dir artifacts/replay_sweep_baseline/noreplay_seq \
  --resume
rc=$?
echo "== fm exit=$rc wall=$((SECONDS - t0))s =="
exit $rc
