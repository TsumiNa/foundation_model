#!/usr/bin/env bash
# Pull the epoch-sweep results from ism-gpu-a100 back to the local machine:
#   1. mirror artifacts/replay_sweep_epoch/ (checkpoints excluded — metrics/logs/provenance only)
#   2. copy each run's training/metrics_table.csv to results/mt_<tag>_epoch.csv
set -euo pipefail
cd "$(dirname "$0")"
REMOTE=ism-gpu-a100
RREPO=/data/claude/foundation_model

mkdir -p results
rsync -a -e 'ssh -o ClearAllForwardings=yes' \
  --exclude '*.ckpt' --exclude 'lightning/' \
  "$REMOTE:$RREPO/artifacts/replay_sweep_epoch/" ../../artifacts/replay_sweep_epoch/

for d in ../../artifacts/replay_sweep_epoch/replay_n*_epoch; do
  tag=$(basename "$d"); tag=${tag#replay_}; tag=${tag%_epoch}
  src="$d/training/metrics_table.csv"
  if [ -f "$src" ]; then
    cp "$src" "results/mt_${tag}_epoch.csv"
    echo "collected results/mt_${tag}_epoch.csv ($(wc -l <"$src") lines)"
  else
    echo "SKIP $tag: no metrics_table.csv yet"
  fi
done
