#!/bin/bash
#SBATCH --partition=ai-h200-brc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=22:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Variant worker: epoch resampling + patience 24 + max_epochs 150 (the p24 sweep showed 100%
# of steps hit the 100-epoch cap, so max_epochs — not patience — is the binding lever).
# One tag per job via TAG (--export); idempotent --resume.
source /etc/profile
set -euo pipefail
module load system/ai-h200-brc nvhpc

PROJ=${PROJ:-$HOME/projects/foundation_model-x86-cuda}
cd "$PROJ"
flock .uv_sync.lock "$HOME/.local/bin/uv" sync --frozen --all-groups
.venv/bin/python -c 'import torch; print(torch.cuda.get_device_name(), torch.version.cuda, torch.cuda.is_available())'

TAG=${TAG:?set via sbatch --export=ALL,TAG=nXXXX}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "== $(hostname) $(date) tag=$TAG commit=$(git rev-parse --short HEAD) =="
t0=$SECONDS
.venv/bin/fm pretrain \
  --config "experiments/rikyu_replay_sweep/configs/sweep_${TAG}.toml" \
  --output-dir "artifacts/replay_sweep_epoch_m150/replay_${TAG}_epoch_m150" \
  --set 'pretrain.replay.resample="epoch"' \
  --set 'training.early_stopping.patience=24' \
  --set 'training.max_epochs=150' \
  --resume
rc=$?
echo "== fm exit=$rc wall=$((SECONDS - t0))s =="
exit $rc
