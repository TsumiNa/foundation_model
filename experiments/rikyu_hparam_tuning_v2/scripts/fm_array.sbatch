#!/bin/bash
# Generic RIKYU grid-array worker for the v2 hyper-parameter campaign.
#
# One array task = one grid point = one `fm pretrain` (or `fm finetune`) run inside the project's
# AArch64 container. Every unit is idempotent: a finished run drops a DONE marker and is skipped
# on resubmission, so the whole array can be resubmitted safely after a TIMEOUT or a partial
# failure.
#
# Grid file format (TAB-separated, one line per array index, 0-based -> line N+1):
#   <runid>\t<shell-quoted `fm` overrides>
# e.g.
#   a1_L128_E5e-3_M1e-6<TAB>--set model.latent_dim=128 --set training.scheduler.min_lr=1e-06
#
# DIFFERENCE FROM v1's worker, and the reason this file exists:
#
#   v1's fm_array.sbatch defaulted IMAGE to the 0.2.1 container and v1's submit.sh never exported
#   IMAGE at all. For v1 that was harmless — 0.2.1 was the intended image. For v2 it is the one
#   mistake in this campaign that would be SILENT: 0.2.1 steps ReduceLROnPlateau once per BATCH
#   (fixed in PR #45), so every v2 run would train in exactly the broken-scheduler regime that v2
#   exists to escape, exit 0, drop a DONE marker, and report plausible numbers from the wrong
#   world. Nothing downstream would flag it.
#
#   So here IMAGE is REQUIRED, and the container's own package version is asserted against
#   EXPECT_VERSION before a single epoch runs. A wrong image is now a loud failure at second 5.
#   The resolved facts are written to $OUT/ENV.json so the claim is auditable after the fact
#   rather than resting on anyone having remembered.
#
#SBATCH --job-name=fm-grid-v2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

module purge

PROJ=${PROJ:?set PROJ=/abs/path to the repo checkout}
MODE=${MODE:-pretrain}
EXTRA_FLAGS=${EXTRA_FLAGS:-}
: "${IMAGE:?set IMAGE=/abs/path/to/foundation-model_rikyu-<version>.sif — never defaulted, see header}"
: "${EXPECT_VERSION:?set EXPECT_VERSION=<x.y.z> the container must report}"
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

# --- image identity gate ---------------------------------------------------------------------
# Ask the container what it actually is, before trusting it with 20 minutes of GPU time.
GOT_VERSION=$(apptainer exec "$IMAGE" python -c \
    'import importlib.metadata as m; print(m.version("foundation-model"))' 2>/dev/null || echo "UNKNOWN")

if [ "$GOT_VERSION" != "$EXPECT_VERSION" ]; then
    echo "FAIL $RUNID image-version mismatch: expected $EXPECT_VERSION, container reports $GOT_VERSION"
    echo "     IMAGE=$IMAGE"
    echo "     Refusing to train — this is the silent-wrong-regime failure this gate exists for."
    exit 3
fi

GIT_SHA=$(git -C "$PROJ" rev-parse --short HEAD 2>/dev/null || echo "unknown")
cat > "$OUT/ENV.json" <<JSON
{
  "runid": "$RUNID",
  "image": "$IMAGE",
  "image_mtime": "$(date -Is -r "$IMAGE" 2>/dev/null || echo unknown)",
  "fm_version": "$GOT_VERSION",
  "expected_version": "$EXPECT_VERSION",
  "git_sha": "$GIT_SHA",
  "config": "$CONFIG",
  "mode": "$MODE",
  "host": "$(hostname)",
  "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
  "started": "$(date -Is)"
}
JSON

# Re-parse the shell-quoted override string into "$@" so quoted TOML values survive.
eval "set -- $OVERRIDES"

echo "host=$(hostname) run=$RUNID mode=$MODE fm=$GOT_VERSION gpus=${SLURM_JOB_GPUS:-?} cuda=${CUDA_VISIBLE_DEVICES:-?}"
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
