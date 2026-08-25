#!/bin/bash
#SBATCH --partition=ai-h200-brc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Smoke for the ratio + baseline extension (--sample 400, 1 epoch, real GPU environment):
#   1. ratio pretrain path  (sweep_0p50 + m150 overrides)
#   2. no-replay pretrain   (interval 999)
#   3. joint retrain        (fm finetune from smoke 2's final model, encoder unfrozen)
source /etc/profile
set -euo pipefail
module load system/ai-h200-brc nvhpc

PROJ=${PROJ:-$HOME/projects/foundation_model-x86-cuda}
cd "$PROJ"
flock .uv_sync.lock "$HOME/.local/bin/uv" sync --frozen --all-groups
.venv/bin/python -c 'import torch; print(torch.cuda.get_device_name(), torch.version.cuda, torch.cuda.is_available())'
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "== $(hostname) $(date) commit=$(git rev-parse --short HEAD) =="

rm -rf artifacts/smoke_ratio_baseline
echo "--- smoke 1/3: ratio 0p50 + m150 overrides ---"
.venv/bin/fm pretrain \
  --config experiments/rikyu_replay_sweep/configs/sweep_0p50.toml \
  --output-dir artifacts/smoke_ratio_baseline/ratio_0p50 \
  --set 'pretrain.replay.resample="epoch"' \
  --set 'training.early_stopping.patience=24' \
  --set 'training.max_epochs=150' \
  --sample 400 --max-epochs 1

echo "--- smoke 2/3: no-replay sequential ---"
.venv/bin/fm pretrain \
  --config experiments/replay_epoch_sweep/configs/noreplay_full24.toml \
  --output-dir artifacts/smoke_ratio_baseline/noreplay \
  --sample 400 --max-epochs 1

echo "--- smoke 3/3: joint retrain from smoke 2 ---"
.venv/bin/fm finetune \
  --config experiments/replay_epoch_sweep/configs/joint_retrain_full24.toml \
  --checkpoint artifacts/smoke_ratio_baseline/noreplay/training/final_model.pt \
  --output-dir artifacts/smoke_ratio_baseline/joint_retrain \
  --sample 400 --epochs 1

.venv/bin/python - <<'EOF'
import json
from pathlib import Path
s = json.load(open("artifacts/smoke_ratio_baseline/joint_retrain/training/finetune_summary.json"))
assert len(s["finetuned_heads"]) == 24 and s["freeze_encoder"] is False and not s["added_tasks"]
assert len(s["metrics_after"]) == 24
print("SMOKE PASS: joint retrain trained 24 heads, encoder unfrozen, all evaluated")
EOF
