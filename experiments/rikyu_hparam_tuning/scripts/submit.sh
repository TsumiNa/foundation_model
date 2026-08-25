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

# stage -> probe config | output root | default walltime | fm subcommand
MODE=pretrain
case "$STAGE" in
    a1|a1b|a2|a3|a4|a6) CONFIG=probe3.toml;    OUT=stage_a; DEFTIME=03:00:00 ;;
    breg|bkr|bclf|b4) CONFIG=single_task.toml; OUT=stage_b; DEFTIME=02:00:00 ;;
    bmtreg)           CONFIG=probe3.toml;      OUT=stage_b; DEFTIME=03:00:00 ;;
    bmtkr)            CONFIG=probe3_kr.toml;   OUT=stage_b; DEFTIME=03:00:00 ;;
    # Stage C: two arms x two phases. `fm pretrain --resume` is idempotent, so a walltime kill is
    # recovered by resubmitting the identical command; `fm finetune` has no resume and gets its
    # budget in one go.
    cpre_base)   CONFIG=final_hybrid.toml;            OUT=stage_c; DEFTIME=48:00:00 ;;
    cpre_tuned)  CONFIG=final_hybrid_tuned.toml;      OUT=stage_c; DEFTIME=48:00:00 ;;
    ccon_base)   CONFIG=final_consolidate.toml;       OUT=stage_c; DEFTIME=10:00:00; MODE=finetune ;;
    ccon_tuned)  CONFIG=final_consolidate_tuned.toml; OUT=stage_c; DEFTIME=10:00:00; MODE=finetune ;;
    *) echo "unknown stage '$STAGE'" >&2; exit 2 ;;
esac
TIME=${TIME:-$DEFTIME}
test -r "$EXP/configs/$CONFIG" || { echo "missing $EXP/configs/$CONFIG" >&2; exit 2; }

GRID="$EXP/configs/grid_$STAGE.txt"
test -r "$GRID" || { echo "missing $GRID — generate it with scripts/make_grids.py $STAGE" >&2; exit 2; }
N=$(grep -c . "$GRID")

mkdir -p "$OUTBASE/$OUT" "$LOGDIR"
cd "$LOGDIR"

JID=$(sbatch --parsable \
    --job-name="$STAGE" \
    --time="$TIME" \
    --array="0-$((N - 1))%$THROTTLE" \
    --export=ALL,GRID="$GRID",CONFIG="experiments/rikyu_hparam_tuning/configs/$CONFIG",OUTROOT="$OUTBASE/$OUT",MODE="$MODE",PROJ="$PROJ" \
    "${EXTRA_SBATCH[@]}" \
    "$EXP/scripts/fm_array.sbatch")

DONE_ALREADY=$(cut -f1 "$GRID" | while read -r r; do [ -f "$OUTBASE/$OUT/$r/DONE" ] && echo x; done | grep -c . || true)
echo "stage=$STAGE job=$JID points=$N already_done=$DONE_ALREADY mode=$MODE config=$CONFIG out=$OUTBASE/$OUT time=$TIME throttle=$THROTTLE"
echo "commit=$(git -C "$PROJ" rev-parse --short HEAD)"
