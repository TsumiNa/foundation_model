#!/bin/bash
#SBATCH --job-name=fm-few-label
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00

source /etc/profile
set -euo pipefail
module load "system/${FM_PARTITION}" nvhpc
cd "$FM_WORKSPACE"
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR="$FM_WORKSPACE/mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
image_hash=$(sha256sum "$FM_IMAGE" | cut -d ' ' -f1)
if [[ "$image_hash" != "$FM_IMAGE_HASH" ]]; then exit 2; fi
first=$((FIRST_CASE + ${SLURM_ARRAY_TASK_ID:-0} * PACK))
pids=()
for ((lane=0; lane<PACK; lane++)); do
    case_index=$((first + lane))
    if ((case_index >= 156)); then break; fi
    extra=()
    if [[ ${SMOKE:-0} == 1 ]]; then extra+=(--smoke); fi
    singularity exec --nv --cleanenv --bind "$FM_WORKSPACE:$FM_WORKSPACE" \
        --env OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=1,MPLCONFIGDIR="$MPLCONFIGDIR" \
        "$FM_IMAGE" python scripts/run.py --data "$FM_DATA" --output "$FM_OUTPUT" \
        --revision "$FM_REVISION" --image-hash "$image_hash" --case "$case_index" "${extra[@]}" \
        > "$FM_OUTPUT/worker-${case_index}.log" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
