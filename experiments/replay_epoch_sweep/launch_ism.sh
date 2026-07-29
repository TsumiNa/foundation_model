#!/usr/bin/env bash
# Launch the full fixed-count epoch-resample sweep (n100..n2500, 7 runs) on ism-gpu-a100:
# 4 tmux workers (one per A100), shared heavy-first FIFO queue. Run ON the remote machine.
set -euo pipefail
REPO=${REPO:-/data/claude/foundation_model}
cd "$REPO"
mkdir -p logs artifacts/replay_sweep_epoch
printf '%s\n' n2500 n2000 n1500 n1000 n500 n200 n100 > artifacts/replay_sweep_epoch/queue.txt
git log -1 --format='%H %s' > artifacts/replay_sweep_epoch/COMMIT.txt
for g in 0 1 2 3; do
  tmux new-session -d -s "replay_epoch_gpu$g" \
    "bash experiments/replay_epoch_sweep/worker.sh $g >> logs/worker_gpu$g.log 2>&1"
done
tmux ls
