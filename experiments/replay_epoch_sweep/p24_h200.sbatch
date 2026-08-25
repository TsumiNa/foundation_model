#!/bin/bash
#SBATCH --partition=ai-h200-brc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=18:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Variant worker: per-epoch replay resampling + 3x early-stopping patience (8 -> 24).
# One fixed-count tag per job, selected via TAG (--export). Idempotent: resubmit the same
# command with --resume to continue after a walltime kill.
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
  --output-dir "artifacts/replay_sweep_epoch_p24/replay_${TAG}_epoch_p24" \
  --set 'pretrain.replay.resample="epoch"' \
  --set 'training.early_stopping.patience=24' \
  --resume
rc=$?
echo "== fm exit=$rc wall=$((SECONDS - t0))s =="
exit $rc
