#!/bin/bash
# One GPU runs PACK independent routes. The source checkout is immutable during jobs.
#SBATCH --job-name=agis
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=18:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -euo pipefail
module purge
: "${PROJ:?}"
: "${IMAGE:?}"
: "${GRID:?}"
: "${OUTROOT:?}"
: "${MANIFEST:?}"
PACK=${PACK:-1}
test "$PACK" -ge 1
test "$PACK" -le 8
test -r "$IMAGE"
test -r "$PROJ/$MANIFEST"
test -r "$GRID"
mkdir -p "$OUTROOT"
START=$(( ${SLURM_ARRAY_TASK_ID:-0} * PACK ))
COUNT=$(wc -l < "$GRID")
printf 'job=%s pack=%s start=%s count=%s host=%s\n' "$SLURM_JOB_ID" "$PACK" "$START" "$COUNT" "$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GIT_SHA=$(git -C "$PROJ" rev-parse HEAD)
printf '%s\n' "$GIT_SHA" > "$OUTROOT/SOURCE_${SLURM_JOB_ID}.txt"
apptainer exec --nv --bind "$PROJ:/workspace:ro" --env PYTHONPATH=/workspace/src --pwd /workspace "$IMAGE" \
    python -c 'import importlib.metadata as m, foundation_model, torch; assert m.version("foundation-model") == "0.4.1"; assert foundation_model.__file__.startswith("/workspace/src/"); assert torch.cuda.device_count() == 1; print("Validated:", m.version("foundation-model"), foundation_model.__file__, torch.__version__, torch.cuda.get_device_name())'
pids=()
for ((k=0; k<PACK; k++)); do
    row=$((START + k))
    if (( row >= COUNT )); then break; fi
    INDEX=$(sed -n "$((row + 1))p" "$GRID")
    [[ "$INDEX" =~ ^[0-9]+$ ]]
    RUNROOT=/campaign
    if [[ ${CALIBRATE:-0} == 1 ]]; then RUNROOT="/campaign/replica_${row}"; fi
    apptainer exec --nv \
        --bind "$PROJ:/workspace:ro" --bind "$OUTROOT:/campaign" \
        --env PYTHONPATH=/workspace/src,MPLCONFIGDIR=/tmp/agis-mpl-$SLURM_JOB_ID-$k,OMP_NUM_THREADS=2,MKL_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2 \
        --pwd /workspace "$IMAGE" \
        python experiments/agis_transfer/campaign.py run --manifest "$MANIFEST" \
        --index "$INDEX" --output-root "$RUNROOT" \
        > "$OUTROOT/worker_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}_${k}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=$((failed + 1)); fi
done
printf 'Completed GPU task; failed routes=%s\n' "$failed"
test "$failed" -eq 0
