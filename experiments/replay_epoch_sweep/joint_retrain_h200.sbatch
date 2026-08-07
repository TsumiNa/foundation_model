#!/bin/bash
#SBATCH --partition=ai-h200-brc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=24:00:00
#SBATCH --output=/home/users/u0001820/jobs/replay_epoch_p24/%x_%j.out
#SBATCH --error=/home/users/u0001820/jobs/replay_epoch_p24/%x_%j.err

# Baseline arm B: full-data JOINT retrain of all 24 tasks (fm finetune, freeze_encoder=false),
# warm-started from arm A's final model. NOTE: fm finetune has no --resume — this must finish
# within walltime (est. 6-12h; 24h requested).
source /etc/profile
set -euo pipefail
module load system/ai-h200-brc nvhpc

PROJ=/home/users/u0001820/projects/foundation_model-x86-cuda
cd "$PROJ"
flock .uv_sync.lock "$HOME/.local/bin/uv" sync --frozen --all-groups
.venv/bin/python -c 'import torch; print(torch.cuda.get_device_name(), torch.version.cuda, torch.cuda.is_available())'

CKPT="$PROJ/artifacts/replay_sweep_baseline/noreplay_seq/training/final_model.pt"
test -f "$CKPT" || { echo "missing $CKPT — run noreplay_h200.sbatch first"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "== $(hostname) $(date) commit=$(git rev-parse --short HEAD) =="
t0=$SECONDS
.venv/bin/fm finetune \
  --config experiments/replay_epoch_sweep/configs/joint_retrain_full24.toml \
  --checkpoint "$CKPT" \
  --output-dir artifacts/replay_sweep_baseline/joint_retrain
rc=$?
echo "== fm exit=$rc wall=$((SECONDS - t0))s =="
exit $rc
