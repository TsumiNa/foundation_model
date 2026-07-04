#!/bin/bash
#SBATCH --partition=1n1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/ea0094/jobs/task_scaling/%x_%j.out
#SBATCH --error=/home/ea0094/jobs/task_scaling/%x_%j.err

# GPU smoke of the full task-scaling chain on tiny compute: a 4-task mini pretrain
# (formation_energy pinned first, one classification + one KR task to exercise every head kind)
# -> warm-start + finetune from the step-2 checkpoint -> inverse on the warm-start output.
# Ends with explicit artifact checks; the last line prints SMOKE_PASS or SMOKE_FAIL.
set -uo pipefail
PROJ=/home/ea0094/projects/foundation_model
CFG=$PROJ/experiments/rikyu_task_scaling/configs
FM=$PROJ/.venv/bin/fm
B=$PROJ/artifacts/task_scaling/smoke

module load nvhpc
cd "$PROJ"
echo "== $(hostname) $(date) smoke =="
nvidia-smi --query-gpu=name --format=csv,noheader
rm -rf "$B"

set -e
"$FM" pretrain --config "$CFG/pre_n1000.toml" --output-dir "$B/pre" \
    --sample 2000 --max-epochs 2 --seed 2025 \
    --set 'pretrain.task_sequence=["formation_energy","density","material_type","zt"]' \
    --set 'pretrain.task_order_groups=[["formation_energy"],["density","material_type"],["zt"]]' 2>&1

step2=$(ls -d "$B"/pre/training/step02_* | head -1)
echo "== warm-start from $step2 =="
"$FM" pretrain --config "$CFG/ws_n1000.toml" --checkpoint "$step2/checkpoint.pt" \
    --output-dir "$B/ws" --sample 2000 --max-epochs 2 --seed 2025 2>&1

echo "== finetune from $step2 =="
"$FM" finetune --config "$CFG/ft.toml" --checkpoint "$step2/checkpoint.pt" \
    --output-dir "$B/ft" --sample 2000 --epochs 2 --seed 2025 2>&1

echo "== inverse on the warm-start output =="
"$FM" inverse --config "$CFG/inv.toml" --checkpoint "$B/ws/training/final_model.pt" \
    --output-dir "$B/inv" --steps 20 --sample 2000 2>&1
set +e

ok=1
for f in \
    "$B/pre/training/step04_zt/checkpoint.pt" \
    "$B/pre/training/metrics_table.csv" \
    "$B/ws/training/final_model.pt" \
    "$B/ws/training/metrics_table.csv" \
    "$B/ft/training/finetune_summary.json" \
    "$B/inv/inverse_design.json" \
    "$B/inv/fe_down_diel_up/summary.json" \
    "$B/inv/fe_down_diel_up/trajectories/latent_default.npz"; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f"
        ok=0
    fi
done
if [ "$ok" -eq 1 ]; then echo "SMOKE_PASS"; else echo "SMOKE_FAIL"; exit 1; fi
date
