#!/bin/bash
#SBATCH --partition=1n1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/ea0094/jobs/task_scaling/%x_%j.out
#SBATCH --error=/home/ea0094/jobs/task_scaling/%x_%j.err

# Task-scaling warm-start worker for run ORD: for every pretrain step checkpoint k=1..21,
# (a) warm-start the 3 dielectric targets jointly (replay n=1000), (b) run inverse design on
# the resulting model. Every k has skip-if-done + --resume, so the job is safe to resubmit.
set -uo pipefail
PROJ=/home/ea0094/projects/foundation_model
CFG=$PROJ/experiments/rikyu_task_scaling/configs
FM=$PROJ/.venv/bin/fm
: "${ORD:?need ORD=0|1|2}"
N=${N:-1000}
K_FROM=${K_FROM:-1}   # per-checkpoint range: lets a helper job take the slow tail in parallel
K_TO=${K_TO:-21}      # (per-k skip-if-done keeps two jobs from redoing each other's finished work)
DIRTAG=""; [ "$N" != "1000" ] && DIRTAG="_n${N}"
PRE=$PROJ/artifacts/task_scaling/pre${DIRTAG}_o${ORD}/training

module load nvhpc
cd "$PROJ"
echo "== $(hostname) $(date) ws ord=$ORD n=$N k=$K_FROM..$K_TO =="
nvidia-smi --query-gpu=name --format=csv,noheader

rc=0
for k in $(seq "$K_FROM" "$K_TO"); do
    kk=$(printf "%02d" "$k")
    step_dir=$(ls -d "$PRE"/step${kk}_* 2>/dev/null | head -1)
    ckpt="$step_dir/checkpoint.pt"
    if [ ! -f "$ckpt" ]; then
        echo "!! missing pretrain checkpoint for k=$kk — skipping"
        rc=1
        continue
    fi
    OUT=$PROJ/artifacts/task_scaling/ws${DIRTAG}_o${ORD}/k${kk}
    t0=$SECONDS
    if [ -f "$OUT/training/final_model.pt" ]; then
        echo "-- k=$kk ws: already done, skip"
    else
        "$FM" pretrain --config "$CFG/ws_n${N}.toml" --checkpoint "$ckpt" --resume \
            --output-dir "$OUT" --seed 2025 2>&1
        if [ $? -ne 0 ]; then
            echo "!! k=$kk ws FAILED"
            rc=1
            continue
        fi
        echo "-- k=$kk ws done in $((SECONDS - t0))s"
    fi
    INV=$OUT/inverse
    t1=$SECONDS
    if [ -f "$INV/inverse_design.json" ]; then
        echo "-- k=$kk inverse: already done, skip"
    else
        "$FM" inverse --config "$CFG/inv.toml" --checkpoint "$OUT/training/final_model.pt" \
            --output-dir "$INV" 2>&1
        if [ $? -ne 0 ]; then
            echo "!! k=$kk inverse FAILED"
            rc=1
            continue
        fi
        echo "-- k=$kk inverse done in $((SECONDS - t1))s"
    fi
done
echo "== ws ord=$ORD exit=$rc =="
date
exit $rc
