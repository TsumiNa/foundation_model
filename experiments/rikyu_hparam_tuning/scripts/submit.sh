#!/bin/bash
# Submit one campaign stage as a Slurm array on RIKYU.
#
# Each stage knows which grid file, which probe config and which output root it belongs to, so a
# submission is `submit.sh <stage>` and the mapping lives in one place instead of in a shell
# history. Resubmitting a stage is always safe: finished grid points carry a DONE marker and are
# skipped, so this is also the recovery command after a TIMEOUT or a partial failure.
#
#   scripts/submit.sh a1
#   scripts/submit.sh breg --time 01:00:00 --throttle 120
#
# Run it on the login node from anywhere.

set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="$(cd "$EXP/../.." && pwd)"
OUTBASE=${OUTBASE:-/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning}
LOGDIR=${LOGDIR:-$HOME/jobs/hparam}

STAGE=${1:?usage: submit.sh <stage> [--time HH:MM:SS] [--throttle N] [extra sbatch args...]}
shift || true

TIME=""
THROTTLE=80
EXTRA_SBATCH=()
while [ $# -gt 0 ]; do
    case "$1" in
        --time) TIME="$2"; shift 2 ;;
        --throttle) THROTTLE="$2"; shift 2 ;;
        *) EXTRA_SBATCH+=("$1"); shift ;;
    esac
done

# stage -> probe config | output root | default walltime
case "$STAGE" in
    a1|a2|a3|a4|a6)   CONFIG=probe3.toml;      OUT=stage_a; DEFTIME=03:00:00 ;;
    breg|bkr|bclf)    CONFIG=single_task.toml; OUT=stage_b; DEFTIME=02:00:00 ;;
    bmtreg)           CONFIG=probe3.toml;      OUT=stage_b; DEFTIME=03:00:00 ;;
    bmtkr)            CONFIG=probe3_kr.toml;   OUT=stage_b; DEFTIME=03:00:00 ;;
    *) echo "unknown stage '$STAGE'" >&2; exit 2 ;;
esac
TIME=${TIME:-$DEFTIME}

GRID="$EXP/configs/grid_$STAGE.txt"
test -r "$GRID" || { echo "missing $GRID — generate it with scripts/make_grids.py $STAGE" >&2; exit 2; }
N=$(grep -c . "$GRID")

mkdir -p "$OUTBASE/$OUT" "$LOGDIR"
cd "$LOGDIR"

JID=$(sbatch --parsable \
    --job-name="$STAGE" \
    --time="$TIME" \
    --array="0-$((N - 1))%$THROTTLE" \
    --export=ALL,GRID="$GRID",CONFIG="experiments/rikyu_hparam_tuning/configs/$CONFIG",OUTROOT="$OUTBASE/$OUT",MODE=pretrain,PROJ="$PROJ" \
    "${EXTRA_SBATCH[@]}" \
    "$EXP/scripts/fm_array.sbatch")

DONE_ALREADY=$(cut -f1 "$GRID" | while read -r r; do [ -f "$OUTBASE/$OUT/$r/DONE" ] && echo x; done | grep -c . || true)
echo "stage=$STAGE job=$JID points=$N already_done=$DONE_ALREADY config=$CONFIG out=$OUTBASE/$OUT time=$TIME throttle=$THROTTLE"
echo "commit=$(git -C "$PROJ" rev-parse --short HEAD)"
