#!/usr/bin/env bash
# Pull the ratio-family + baseline results back to the local machine (idempotent — rerun as
# runs land):
#   1. mirror the R-CCS ratio runs + baseline arms and the ism joint-retrain caps
#      (step/lightning checkpoints excluded — metrics/logs/provenance only)
#   2. copy each ratio run's training/metrics_table.csv to results/mt_<tag>_epoch_m150.csv
#      (0p10 was --resume'd twice: its table is PARTIAL — rebuild afterwards with
#       analysis/rebuild_metrics_from_stepdirs.py, which overwrites the collected copy)
#   3. copy the no-replay table to results/mt_noreplay.csv and each joint-retrain
#      finetune_summary.json to results/joint_retrain_m<cap>.json
set -euo pipefail
cd "$(dirname "$0")"
RCCS=riken-login
RCCS_REPO=projects/foundation_model-x86-cuda
ISM=ism-gpu-a100
ISM_REPO=/data/claude/foundation_model

mkdir -p results ../../artifacts/replay_sweep_epoch_m150 ../../artifacts/replay_sweep_baseline

rsync -a --exclude 'checkpoint.pt' --exclude '*.ckpt' --exclude 'lightning/' \
  "$RCCS:$RCCS_REPO/artifacts/replay_sweep_epoch_m150/replay_0p10_epoch_m150" \
  "$RCCS:$RCCS_REPO/artifacts/replay_sweep_epoch_m150/replay_0p20_epoch_m150" \
  "$RCCS:$RCCS_REPO/artifacts/replay_sweep_epoch_m150/replay_0p30_epoch_m150" \
  "$RCCS:$RCCS_REPO/artifacts/replay_sweep_epoch_m150/replay_0p50_epoch_m150" \
  ../../artifacts/replay_sweep_epoch_m150/ 2>/dev/null || true

rsync -a --exclude 'checkpoint.pt' --exclude '*.ckpt' --exclude 'lightning/' \
  "$RCCS:$RCCS_REPO/artifacts/replay_sweep_baseline/" ../../artifacts/replay_sweep_baseline/

rsync -a -e 'ssh -o ClearAllForwardings=yes' \
  --exclude 'checkpoint.pt' --exclude '*.ckpt' --exclude 'lightning/' \
  "$ISM:$ISM_REPO/artifacts/replay_sweep_baseline/" ../../artifacts/replay_sweep_baseline/

for d in ../../artifacts/replay_sweep_epoch_m150/replay_0p*_epoch_m150; do
  tag=$(basename "$d"); tag=${tag#replay_}; tag=${tag%_epoch_m150}
  src="$d/training/metrics_table.csv"
  if [ -f "$src" ]; then
    cp "$src" "results/mt_${tag}_epoch_m150.csv"
    echo "collected results/mt_${tag}_epoch_m150.csv ($(wc -l <"$src") lines)"
  else
    echo "SKIP $tag: no metrics_table.csv yet"
  fi
done

nr=../../artifacts/replay_sweep_baseline/noreplay_seq/training/metrics_table.csv
if [ -f "$nr" ]; then
  cp "$nr" results/mt_noreplay.csv
  echo "collected results/mt_noreplay.csv ($(wc -l <"$nr") lines)"
fi

for d in ../../artifacts/replay_sweep_baseline/joint_retrain*; do
  cap=$(basename "$d"); cap=${cap#joint_retrain}; cap=${cap:-_m150}; cap=${cap#_}
  s="$d/training/finetune_summary.json"
  if [ -f "$s" ]; then
    cp "$s" "results/joint_retrain_${cap:-m150}.json"
    echo "collected results/joint_retrain_${cap}.json"
  fi
done
