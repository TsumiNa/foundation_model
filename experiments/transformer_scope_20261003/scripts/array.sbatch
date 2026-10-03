#!/bin/bash
#SBATCH --job-name=fm-transformer-scope
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=scope-%A_%a.out
set -euo pipefail
module purge
module load nvhpc
module list
APPTAINER=${APPTAINER:-/shared/software/apptainer/bin/apptainer}
: "${FM_WORKSPACE:?}" "${FM_IMAGE:?}" "${FM_DATA_DIR:?}" "${FM_OUTPUT_ROOT:?}" "${FM_BENCHMARK_REVISION:?}" "${PHASE:?}"
PACK=${PACK:-1}; FIRST_CASE=${FIRST_CASE:-0}
[[ $PACK =~ ^[1-9][0-9]*$ && $PACK -le ${SLURM_CPUS_PER_TASK:-1} && $FIRST_CASE =~ ^[0-9]+$ ]] || exit 2
[[ $FM_BENCHMARK_REVISION =~ ^[0-9a-f]{40}$ && $FM_WORKSPACE = /* && $FM_DATA_DIR = "$FM_WORKSPACE"/* && $FM_OUTPUT_ROOT = "$FM_WORKSPACE"/* ]] || exit 2
[[ ! -d "$FM_WORKSPACE/src" && -z ${PYTHONPATH:-} ]] || { echo 'FAIL source overrides'; exit 2; }
EXP="$FM_WORKSPACE/experiments/transformer_scope_20261003"
CFG="$EXP/configs/study.toml"
EXPECTED=$($APPTAINER exec --cleanenv --bind "$FM_WORKSPACE:$FM_WORKSPACE" "$FM_IMAGE" python -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["study"]["sif_sha256"])' "$CFG")
FM_SIF_SHA256=$(sha256sum "$FM_IMAGE" | cut -d ' ' -f 1)
[[ $FM_SIF_SHA256 = "$EXPECTED" ]] || { echo 'FAIL SIF checksum'; exit 2; }
[[ $PHASE = pilot || $PHASE = formal ]] || exit 2
CASE_COUNT=$($APPTAINER exec --cleanenv --bind "$FM_WORKSPACE:$FM_WORKSPACE" "$FM_IMAGE" python -c \
    'import sys,tomllib; r=tomllib.load(open(sys.argv[1],"rb")); print(sum(len(v["learning_rates"]) for v in r["arms"].values())*len(r["study"]["pilot_seeds"]) if sys.argv[2]=="pilot" else len(r["arms"])*len(r["study"]["seeds"])*len(r["study"]["split_seeds"])*(1+len(r["study"]["source_sets"])*len(r["study"]["source_budgets"])))' "$CFG" "$PHASE")
[[ $((FIRST_CASE + ${SLURM_ARRAY_TASK_ID:-0} * PACK)) -lt $CASE_COUNT ]] || exit 2
SELECTION=()
if [[ $PHASE = formal ]]; then
    : "${FM_SELECTION:?}"
    [[ -f $FM_SELECTION && $FM_SELECTION = "$FM_WORKSPACE"/* ]] || exit 2
    SELECTION+=(--selection "$FM_SELECTION")
fi
EXTRA=(); [[ ${SMOKE:-0} = 0 ]] || EXTRA+=(--smoke)
THREADS=$((SLURM_CPUS_PER_TASK / PACK)); [[ $THREADS -le 4 ]] || THREADS=4
mkdir -p "$FM_OUTPUT_ROOT"
PIDS=(); STATUS=0
for ((offset=0; offset<PACK; offset++)); do
    index=$((FIRST_CASE + ${SLURM_ARRAY_TASK_ID:-0} * PACK + offset))
    [[ $index -lt $CASE_COUNT ]] || break
    "$APPTAINER" exec --nv --cleanenv --bind "$FM_WORKSPACE:$FM_WORKSPACE" \
        --env "SLURM_JOB_ID=$SLURM_JOB_ID,CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},OMP_NUM_THREADS=$THREADS,OPENBLAS_NUM_THREADS=$THREADS,FM_BENCHMARK_REVISION=$FM_BENCHMARK_REVISION,FM_SIF_SHA256=$FM_SIF_SHA256" \
        "$FM_IMAGE" python "$EXP/scripts/study.py" --config "$CFG" --data "$FM_DATA_DIR" \
        --output "$FM_OUTPUT_ROOT" --phase "$PHASE" --case "$index" ${SELECTION[@]+"${SELECTION[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} \
        > "$FM_OUTPUT_ROOT/case-$index-$SLURM_JOB_ID.log" 2>&1 &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do if ! wait "$pid"; then STATUS=1; fi; done
[[ $STATUS != 0 ]] || echo 'PASS controls array'
exit "$STATUS"
