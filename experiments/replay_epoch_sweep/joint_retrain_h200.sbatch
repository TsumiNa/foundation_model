#!/bin/bash
#SBATCH --partition=ai-h200-brc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Baseline arm B: full-data JOINT retrain of all 24 tasks (fm finetune, freeze_encoder=false),
# warm-started from arm A's final model. EP (--export) sets the epoch cap; each cap is an
# independent run from the same checkpoint (early stopping patience 24 may fire below the cap —
# epochs_run in finetune_summary.json records where it stopped). The original 150-epoch run
# predates the EP parameter and lives in joint_retrain/ (not joint_retrain_m150/).
# NOTE: fm finetune has no --resume — each run must finish within walltime (~65 s/epoch observed).
source /etc/profile
set -euo pipefail
module load system/ai-h200-brc nvhpc

PROJ=${PROJ:-$HOME/projects/foundation_model-x86-cuda}
cd "$PROJ"
flock .uv_sync.lock "$HOME/.local/bin/uv" sync --frozen --all-groups
.venv/bin/python -c 'import torch; print(torch.cuda.get_device_name(), torch.version.cuda, torch.cuda.is_available())'

CKPT="$PROJ/artifacts/replay_sweep_baseline/noreplay_seq/training/final_model.pt"
test -f "$CKPT" || { echo "missing $CKPT — run noreplay_h200.sbatch first"; exit 1; }

EP=${EP:?set via sbatch --export=ALL,EP=200}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "== $(hostname) $(date) EP=$EP commit=$(git rev-parse --short HEAD) =="
t0=$SECONDS
.venv/bin/fm finetune \
  --config experiments/replay_epoch_sweep/configs/joint_retrain_full24.toml \
  --checkpoint "$CKPT" \
  --epochs "$EP" \
  --output-dir "artifacts/replay_sweep_baseline/joint_retrain_m${EP}"
rc=$?
echo "== fm exit=$rc wall=$((SECONDS - t0))s =="
exit $rc
