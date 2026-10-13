#!/usr/bin/env bash
# Single-task QMOF runs: every task in TASKS x seeds 2025-2027, CPU, into experiments/qmof_learnability/artifacts/single/<task>_s<seed>.
# A finished run is marked with DONE and skipped on rerun.
#   bash experiments/qmof_learnability/scripts/run_single.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=experiments/qmof_learnability/artifacts/single
TASKS=(qmof_band_gap_pbe qmof_energy_per_atom_pbe qmof_cbm_pbe qmof_vbm_pbe)
for t in "${TASKS[@]}"; do
  for s in 2025 2026 2027; do
    r="$OUT/${t}_s$s"
    [ -f "$r/DONE" ] && continue
    uv run fm pretrain --config experiments/qmof_learnability/configs/single_qmof.toml --output-dir "$r" --seed "$s" \
      --set "pretrain.task_sequence=[\"$t\"]" > /dev/null 2>&1 && touch "$r/DONE"
    echo "$t s$s done"
  done
done
