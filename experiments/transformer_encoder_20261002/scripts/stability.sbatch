#!/bin/bash
#SBATCH --job-name=fm-stability
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=stability-%A_%a.out

# Runtime account/paths are supplied outside Git. PACK is calibrated per architecture.
# FIRST_CASE=0, PACK=1: 12 legacy cases. FIRST_CASE=12, PACK=3: 15 grouped cases.
set -euo pipefail
module purge
module load nvhpc
module list
APPTAINER=${APPTAINER:-/shared/software/apptainer/bin/apptainer}
: "${FM_WORKSPACE:?}" "${FM_IMAGE:?}" "${FM_DATA_DIR:?}" "${FM_OUTPUT_ROOT:?}"
: "${FM_BENCHMARK_REVISION:?}" "${SLURM_JOB_ID:?}"
PACK=${PACK:-1}
FIRST_CASE=${FIRST_CASE:-0}
[[ $PACK =~ ^[1-9][0-9]*$ && $PACK -le ${SLURM_CPUS_PER_TASK:-1} ]] || exit 2
[[ $FIRST_CASE =~ ^[0-9]+$ && $FM_BENCHMARK_REVISION =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ $FM_WORKSPACE = /* && $FM_DATA_DIR = "$FM_WORKSPACE"/* && $FM_OUTPUT_ROOT = "$FM_WORKSPACE"/* ]] || exit 2
[[ ! -d "$FM_WORKSPACE/src/foundation_model" ]] || { echo "FAIL scripts/data workspace required"; exit 2; }
EXP="$FM_WORKSPACE/experiments/transformer_encoder_20261002"
PROTOCOL="$EXP/configs/protocol.toml"
EXPECTED=$($APPTAINER exec --cleanenv --bind "$FM_WORKSPACE:$FM_WORKSPACE" "$FM_IMAGE" python -c \
    'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["benchmark"]["sif_sha256"])' "$PROTOCOL")
FM_SIF_SHA256=$(sha256sum "$FM_IMAGE" | cut -d ' ' -f 1)
[[ $FM_SIF_SHA256 = "$EXPECTED" ]] || { echo "FAIL SIF checksum"; exit 2; }
THREADS=$((SLURM_CPUS_PER_TASK / PACK))
[[ $THREADS -le 4 ]] || THREADS=4
mkdir -p "$FM_OUTPUT_ROOT"
PIDS=()
STATUS=0
for ((offset=0; offset<PACK; offset++)); do
    index=$((FIRST_CASE + ${SLURM_ARRAY_TASK_ID:-0} * PACK + offset))
    "$APPTAINER" exec --nv --cleanenv --bind "$FM_WORKSPACE:$FM_WORKSPACE" \
        --env "SLURM_JOB_ID=$SLURM_JOB_ID,CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},OMP_NUM_THREADS=$THREADS,OPENBLAS_NUM_THREADS=$THREADS,MPLCONFIGDIR=/tmp/fm-stability-$SLURM_JOB_ID-$index,FM_BENCHMARK_REVISION=$FM_BENCHMARK_REVISION,FM_SIF_SHA256=$FM_SIF_SHA256" \
        "$FM_IMAGE" python "$EXP/scripts/stability_probe.py" --protocol "$PROTOCOL" \
        --probe-config "$EXP/configs/stability.toml" --data-dir "$FM_DATA_DIR" \
        --output-root "$FM_OUTPUT_ROOT" --case "$index" \
        > "$FM_OUTPUT_ROOT/case-${index}-${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then STATUS=1; fi
done
if [[ $STATUS = 0 ]]; then echo "PASS stability array task"; else echo "FAIL stability array task"; fi
exit "$STATUS"
