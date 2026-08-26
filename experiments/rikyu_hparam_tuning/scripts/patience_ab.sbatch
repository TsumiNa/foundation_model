#!/bin/bash
# Patience A/B — does stepping ReduceLROnPlateau once per EPOCH (#45) change training?
#
# Deliberately standalone rather than a new stage in submit.sh/fm_array.sbatch: those two are
# bind-mounted into the stage-C jobs that are still running, and this experiment needs a
# per-run IMAGE, which fm_array.sbatch takes only as a single job-wide env var.
#
# Each grid line is: RUNID <TAB> IMAGE <TAB> CONFIG <TAB> SEED
# so the image is chosen per array index — that is the whole point, the arms differ by image.
#
#   pab_old_*   image 0.2.1 + probe3.toml            per-BATCH cadence, global weight_decay 1e-3
#   pab_new_*   image 0.3.2 + patience_ab_new.toml   per-EPOCH cadence, weight decays PINNED to
#                                                    0.2.1's values -> cadence is the only change
#   pab_asis_*  image 0.3.2 + probe3.toml            per-EPOCH cadence AND #42's new weight-decay
#                                                    defaults -> what "just switch images" does
#
# old vs new  = the patience question the experiment was asked to answer.
# new vs asis = the weight-decay side effect, separated out instead of blamed on patience.
#
# Idempotent: a DONE marker per run, so a resubmit after a walltime kill or a partial failure
# reruns only what is missing.
#
# Usage:
#   sbatch --array=0-8 experiments/rikyu_hparam_tuning/scripts/patience_ab.sbatch
#
#SBATCH --job-name=fm-pab
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=/home/rku00225/jobs/hparam/pab_%A_%a.out
#SBATCH --error=/home/rku00225/jobs/hparam/pab_%A_%a.err

set -euo pipefail
module purge

PROJ=${PROJ:-$HOME/projects/foundation_model}
OUTROOT=${OUTROOT:-/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning/patience_ab}
GRID=${GRID:-$PROJ/experiments/rikyu_hparam_tuning/configs/grid_patience_ab.txt}

test -r "$GRID" || { echo "FAIL missing grid $GRID"; exit 2; }
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GRID")
[ -n "$LINE" ] || { echo "FAIL empty grid line $SLURM_ARRAY_TASK_ID"; exit 1; }

RUNID=$(printf '%s' "$LINE" | cut -f1)
IMAGE=$(eval echo "$(printf '%s' "$LINE" | cut -f2)")   # the grid stores $HOME unexpanded
CONFIG=$(printf '%s' "$LINE" | cut -f3)
SEED=$(printf '%s' "$LINE" | cut -f4)

test -r "$IMAGE"       || { echo "FAIL missing image $IMAGE"; exit 2; }
test -r "$PROJ/$CONFIG" || { echo "FAIL missing config $PROJ/$CONFIG"; exit 2; }

OUT="$OUTROOT/$RUNID"
mkdir -p "$OUT"
if [ -f "$OUT/DONE" ]; then
    echo "SKIP $RUNID (DONE marker present)"
    echo "PASS $RUNID skipped"
    exit 0
fi

echo "host=$(hostname) run=$RUNID seed=$SEED"
echo "image=$IMAGE"
echo "config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

START=$SECONDS
set +e
apptainer exec --nv \
    --bind "$PROJ:/workspace:ro" \
    --bind "$OUTROOT:/out" \
    --env MPLCONFIGDIR=/tmp/mplconfig \
    --pwd /workspace \
    "$IMAGE" \
    fm pretrain --config "$CONFIG" --output-dir "/out/$RUNID" --seed "$SEED"
RC=$?
set -e
ELAPSED=$((SECONDS - START))

printf '%s\t%s\t%s\t%s\n' "$RUNID" "$ELAPSED" "$RC" "$(hostname)" >> "$OUTROOT/_timing.tsv"

if [ $RC -eq 0 ]; then
    touch "$OUT/DONE"
    echo "PASS $RUNID in ${ELAPSED}s"
else
    echo "FAIL $RUNID rc=$RC after ${ELAPSED}s"
fi
exit $RC
