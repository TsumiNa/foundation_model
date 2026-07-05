#!/bin/bash
#SBATCH --partition=1n1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/home/ea0094/jobs/task_scaling/%x_%j.out
#SBATCH --error=/home/ea0094/jobs/task_scaling/%x_%j.err

# k=0 baseline: train the 3 dielectric targets from scratch (no pretraining), 3 seeds.
# No inverse stage — the scratch model has no formation_energy head.
set -uo pipefail
PROJ=/home/ea0094/projects/foundation_model
CFG=$PROJ/experiments/rikyu_task_scaling/configs
FM=$PROJ/.venv/bin/fm

module load nvhpc
cd "$PROJ"
echo "== $(hostname) $(date) scratch baseline =="
nvidia-smi --query-gpu=name --format=csv,noheader

N=${N:-1000}
DIRTAG=""; [ "$N" != "1000" ] && DIRTAG="_n${N}"
rc=0
for S in 2025 2026 2027; do
    OUT=$PROJ/artifacts/task_scaling/scratch${DIRTAG}_s${S}
    if [ -f "$OUT/training/final_model.pt" ]; then
        echo "-- seed=$S already done, skip"
        continue
    fi
    t0=$SECONDS
    "$FM" pretrain --config "$CFG/ws_n${N}.toml" --resume --output-dir "$OUT" --seed "$S" 2>&1
    if [ $? -ne 0 ]; then
        echo "!! seed=$S FAILED"
        rc=1
        continue
    fi
    echo "-- seed=$S done in $((SECONDS - t0))s"
done
echo "== scratch exit=$rc =="
date
exit $rc
