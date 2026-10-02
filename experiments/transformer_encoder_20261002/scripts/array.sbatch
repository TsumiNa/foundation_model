#!/bin/bash
#SBATCH --job-name=fm-transfer
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%A_%a.out

# Submit from a directory for this campaign's Slurm logs. Account comes from sbatch --account.
# Required exported values: FM_WORKSPACE, FM_IMAGE, FM_WORKLIST, FM_DATA_DIR, FM_OUTPUT_ROOT,
# FM_BENCHMARK_REVISION. Optional: PACK (default 1), SOURCE_COUNT (7), MAX_EPOCHS,
# SOURCE_ONLY (0), INVERSE_SMOKE (0), FM_PROTOCOL (initial screen by default).
# No source package is bound into the image.
set -euo pipefail
module purge
module load nvhpc
module list
APPTAINER=${APPTAINER:-/shared/software/apptainer/bin/apptainer}
: "${FM_WORKSPACE:?}" "${FM_IMAGE:?}" "${FM_WORKLIST:?}" "${FM_DATA_DIR:?}"
: "${FM_OUTPUT_ROOT:?}" "${FM_BENCHMARK_REVISION:?}" "${SLURM_JOB_ID:?}"
PACK=${PACK:-1}
SOURCE_COUNT=${SOURCE_COUNT:-7}
[[ $PACK =~ ^[1-9][0-9]*$ && $PACK -le ${SLURM_CPUS_PER_TASK:-1} ]] || exit 2
[[ $FM_BENCHMARK_REVISION =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ $FM_WORKSPACE = /* && $FM_DATA_DIR = "$FM_WORKSPACE"/* && $FM_OUTPUT_ROOT = "$FM_WORKSPACE"/* ]] || exit 2
[[ ! -d "$FM_WORKSPACE/src/foundation_model" ]] || { echo "FAIL stage a scripts/data workspace only"; exit 2; }
EXP="$FM_WORKSPACE/experiments/transformer_encoder_20261002"
PROTOCOL=${FM_PROTOCOL:-"$EXP/configs/protocol.toml"}
[[ $PROTOCOL = "$FM_WORKSPACE"/* && -f $PROTOCOL ]] || exit 2
EXPECTED=$($APPTAINER exec --cleanenv --bind "$FM_WORKSPACE:$FM_WORKSPACE" "$FM_IMAGE" python -c \
    'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["benchmark"]["sif_sha256"])' "$PROTOCOL")
FM_SIF_SHA256=$(sha256sum "$FM_IMAGE" | cut -d ' ' -f 1)
[[ $FM_SIF_SHA256 = "$EXPECTED" ]] || { echo "FAIL SIF checksum"; exit 2; }
export FM_SIF_SHA256
printf 'image_sha256=%s benchmark_revision=%s pack=%s\n' "$FM_SIF_SHA256" "$FM_BENCHMARK_REVISION" "$PACK"
mkdir -p "$FM_OUTPUT_ROOT"
THREADS=$(( SLURM_CPUS_PER_TASK / PACK ))
[[ $THREADS -le 4 ]] || THREADS=4
ARGS=(--protocol "$PROTOCOL" --data-dir "$FM_DATA_DIR" --output-root "$FM_OUTPUT_ROOT"
    --source-count "$SOURCE_COUNT")
[[ ${SOURCE_ONLY:-0} = 0 ]] || ARGS+=(--source-only)
[[ -z ${MAX_EPOCHS:-} ]] || ARGS+=(--max-epochs "$MAX_EPOCHS")
mapfile -t LANES < "$FM_WORKLIST"
PIDS=()
STATUS=0
for ((offset=0; offset<PACK; offset++)); do
    index=$(( ${SLURM_ARRAY_TASK_ID:-0} * PACK + offset ))
    [[ $index -lt ${#LANES[@]} ]] || break
    IFS=$'\t' read -r arm seed <<< "${LANES[$index]}"
    [[ $arm =~ ^[a-z_]+$ && $seed =~ ^[0-9]+$ ]] || exit 2
    (
        # --cleanenv removes host PYTHONPATH; only experiment scripts and prepared data are bound.
        "$APPTAINER" exec --nv --cleanenv --bind "$FM_WORKSPACE:$FM_WORKSPACE" \
            --env "SLURM_JOB_ID=$SLURM_JOB_ID,CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},OMP_NUM_THREADS=$THREADS,OPENBLAS_NUM_THREADS=$THREADS,MPLCONFIGDIR=/tmp/fm-mpl-$SLURM_JOB_ID-$index,FM_BENCHMARK_REVISION=$FM_BENCHMARK_REVISION,FM_SIF_SHA256=$FM_SIF_SHA256" \
            "$FM_IMAGE" python "$EXP/scripts/run_lane.py" "${ARGS[@]}" --arm "$arm" --seed "$seed"
        if [[ ${INVERSE_SMOKE:-0} = 1 ]]; then
            "$APPTAINER" exec --nv --cleanenv --bind "$FM_WORKSPACE:$FM_WORKSPACE" \
                --env "SLURM_JOB_ID=$SLURM_JOB_ID,CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},OMP_NUM_THREADS=$THREADS,OPENBLAS_NUM_THREADS=$THREADS,MPLCONFIGDIR=/tmp/fm-mpl-$SLURM_JOB_ID-$index" \
                "$FM_IMAGE" python "$EXP/scripts/inverse_smoke.py" --protocol "$PROTOCOL" --data-dir "$FM_DATA_DIR" \
                --arm "$arm" --checkpoint "$FM_OUTPUT_ROOT/${arm}_s${seed}/targets/dielectric_total_f100_k${SOURCE_COUNT}_full/training/final_model.pt" \
                --output "$FM_OUTPUT_ROOT/${arm}_s${seed}/inverse"
        fi
    ) > "$FM_OUTPUT_ROOT/lane-${SLURM_JOB_ID}-${index}.log" 2>&1 &
    PIDS+=("$!")
done
[[ ${#PIDS[@]} -gt 0 ]] || { echo "FAIL empty array range"; exit 2; }
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then STATUS=1; fi
done
if [[ $STATUS = 0 ]]; then echo "PASS array task"; else echo "FAIL array task; inspect lane logs"; fi
exit "$STATUS"
