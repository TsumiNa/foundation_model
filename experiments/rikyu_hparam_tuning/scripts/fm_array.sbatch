#!/bin/bash
# Generic RIKYU grid-array worker for the hyper-parameter tuning campaign.
#
# One array task = one grid point = one `fm pretrain` (or `fm finetune`) run inside the
# project's AArch64 container. Every unit is idempotent: a finished run drops a DONE marker
# and is skipped on resubmission, so the whole array can be resubmitted safely after a
# TIMEOUT or a partial failure.
#
# Grid file format (TAB-separated, one line per array index, 0-based -> line N+1):
#   <runid>\t<shell-quoted `fm` overrides>
# e.g.
#   a_l128_h256_lr5e-3<TAB>--set model.latent_dim=128 --set 'model.encoder_hidden_dims=[256]'
#
# Submit:
#   sbatch --array=0-$(($(wc -l < grid.txt)-1))%64 \
#          --export=ALL,GRID=...,CONFIG=...,OUTROOT=...,MODE=pretrain \
#          fm_array.sbatch
#
#SBATCH --job-name=fm-grid
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

module purge

PROJ=${PROJ:-$HOME/projects/foundation_model}
IMAGE=${IMAGE:-$HOME/containers/foundation-model_rikyu-0.2.1.sif}
MODE=${MODE:-pretrain}
EXTRA_FLAGS=${EXTRA_FLAGS:-}
: "${GRID:?set GRID=/abs/path/to/grid.txt}"
: "${CONFIG:?set CONFIG=repo-relative path to the TOML}"
: "${OUTROOT:?set OUTROOT=/abs/path for run outputs}"

test -r "$IMAGE"
test -r "$GRID"
test -r "$PROJ/$CONFIG"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GRID")
[ -n "$LINE" ] || { echo "FAIL empty grid line $SLURM_ARRAY_TASK_ID"; exit 1; }
RUNID=${LINE%%$'\t'*}
OVERRIDES=${LINE#*$'\t'}

OUT="$OUTROOT/$RUNID"
mkdir -p "$OUT"

if [ -f "$OUT/DONE" ]; then
    echo "SKIP $RUNID (DONE marker present)"
    echo "PASS $RUNID skipped"
    exit 0
fi

# Re-parse the shell-quoted override string into "$@" so quoted TOML values survive.
eval "set -- $OVERRIDES"

echo "host=$(hostname) run=$RUNID mode=$MODE gpus=${SLURM_JOB_GPUS:-?} cuda=${CUDA_VISIBLE_DEVICES:-?}"
echo "cmd: fm $MODE --config $CONFIG --output-dir /out/$RUNID $EXTRA_FLAGS $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

START=$SECONDS
set +e
apptainer exec --nv \
    --bind "$PROJ:/workspace:ro" \
    --bind "$OUTROOT:/out" \
    --env MPLCONFIGDIR=/tmp/mplconfig \
    --pwd /workspace \
    "$IMAGE" \
    fm "$MODE" --config "$CONFIG" --output-dir "/out/$RUNID" $EXTRA_FLAGS "$@"
RC=$?
set -e
ELAPSED=$((SECONDS - START))

printf '%s\t%s\t%s\t%s\n' "$RUNID" "$ELAPSED" "$RC" "$(hostname)" >> "$OUTROOT/_timing.tsv"

if [ "$RC" -eq 0 ]; then
    date -Is > "$OUT/DONE"
    echo "PASS $RUNID elapsed=${ELAPSED}s"
else
    echo "FAIL $RUNID rc=$RC elapsed=${ELAPSED}s"
fi
exit "$RC"
