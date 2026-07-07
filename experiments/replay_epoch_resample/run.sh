#!/usr/bin/env bash
# Local (Mac Studio, MPS) validation sweep: does per-epoch replay resampling
# (pretrain.replay.resample = "epoch") lift retention at small replay n?
#
# Reuses the canonical rikyu replay-sweep configs unchanged; the only deltas are
# --set resample overrides and fresh output dirs (NEVER write into the recovered
# rikyu artifacts under artifacts/replay_sweep/).
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_ENABLE_MPS_FALLBACK=1  # KR-head ops not fully covered on MPS

for n in n200 n500; do
  .venv/bin/fm pretrain \
    --config "experiments/rikyu_replay_sweep/configs/sweep_${n}.toml" \
    --output-dir "artifacts/replay_sweep_epoch/replay_${n}_epoch" \
    --set 'pretrain.replay.resample="epoch"' \
    --resume &
done
wait
