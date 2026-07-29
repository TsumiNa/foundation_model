#!/usr/bin/env bash
# One worker per GPU on ism-gpu-a100. Each worker pops the next replay tag off the shared
# FIFO queue (heavy-first order) and runs the canonical rikyu sweep config for that tag with
# the ONLY delta being pretrain.replay.resample="epoch" + a fresh output dir.
# Failed tags are not re-queued (workers move on); rerun manually — every run is
# idempotent via --resume.
set -uo pipefail
REPO=${REPO:-/data/claude/foundation_model}
QUEUE=$REPO/artifacts/replay_sweep_epoch/queue.txt
LOCK=$REPO/artifacts/replay_sweep_epoch/queue.lock
GPU=$1

cd "$REPO"
while :; do
  TAG=$(flock "$LOCK" bash -c "head -n1 '$QUEUE' 2>/dev/null; sed -i '1d' '$QUEUE' 2>/dev/null")
  [ -z "${TAG:-}" ] && break
  echo "== [gpu$GPU] $(date '+%F %T') start $TAG =="
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/fm pretrain \
    --config "experiments/rikyu_replay_sweep/configs/sweep_${TAG}.toml" \
    --output-dir "artifacts/replay_sweep_epoch/replay_${TAG}_epoch" \
    --set 'pretrain.replay.resample="epoch"' \
    --resume >> "logs/replay_${TAG}_epoch.log" 2>&1
  rc=$?
  echo "== [gpu$GPU] $(date '+%F %T') done $TAG rc=$rc =="
done
echo "== [gpu$GPU] $(date '+%F %T') queue empty, worker exit =="
