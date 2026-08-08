#!/usr/bin/env bash
# Launch the baseline-B convergence probes (joint retrain, epoch caps 200/250/300) on
# ism-gpu-a100: one tmux session per cap on GPUs 0-2. Requires arm A's final model mirrored
# from R-CCS at artifacts/replay_sweep_baseline/noreplay_seq/training/final_model.pt.
# Run ON the remote machine. (The 150-cap run lives on R-CCS H200 — joint_retrain/.)
set -euo pipefail
REPO=${REPO:-/data/claude/foundation_model}
cd "$REPO"
CKPT=artifacts/replay_sweep_baseline/noreplay_seq/training/final_model.pt
test -f "$CKPT" || { echo "missing $CKPT — rsync it from R-CCS first"; exit 1; }
mkdir -p logs
git log -1 --format='%H %s' > artifacts/replay_sweep_baseline/COMMIT_joint_ism.txt
g=0
for ep in 200 250 300; do
  tmux new-session -d -s "joint_m${ep}" \
    "CUDA_VISIBLE_DEVICES=$g .venv/bin/fm finetune \
       --config experiments/replay_epoch_sweep/configs/joint_retrain_full24.toml \
       --checkpoint $CKPT \
       --epochs $ep \
       --output-dir artifacts/replay_sweep_baseline/joint_retrain_m${ep} \
       >> logs/joint_m${ep}.log 2>&1"
  g=$((g + 1))
done
tmux ls
