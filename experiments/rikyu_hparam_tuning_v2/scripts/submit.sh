#!/bin/bash
# Submit one v2 campaign stage as a Slurm array on RIKYU.
#
# A submission is `submit.sh <stage>`; the stage -> (config, grid, output root, walltime, fm mode)
# mapping lives here so it is in one place rather than in a shell history. Resubmitting a stage is
# always safe: finished grid points carry a DONE marker and are skipped, so this is also the
# recovery command after a TIMEOUT or a partial failure.
#
#   scripts/submit.sh smoke
#   scripts/submit.sh s0
#   scripts/submit.sh a1 --throttle 200
#
# Run it on the login node from anywhere. Slurm needs a LOGIN shell to find its controller, so
# over ssh this must be wrapped: ssh rikyu-login 'bash -lc "..."' (PLAN §5 lesson 1).
#
# IMAGE AND VERSION ARE EXPORTED EXPLICITLY, ALWAYS. v1's submit.sh left IMAGE unset and its
# worker defaulted to the 0.2.1 container. Inheriting that here would silently run the entire v2
# campaign under the per-batch scheduler cadence that v2 exists to escape. The worker refuses to
# train unless the container reports EXPECT_VERSION, so a wrong image fails in seconds instead of
# quietly producing numbers from the wrong regime.

set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="$(cd "$EXP/../.." && pwd)"
OUTBASE=${OUTBASE:-/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning_v2}
LOGDIR=${LOGDIR:-$HOME/jobs/hparam_v2}

VERSION=${VERSION:-0.3.2}
IMAGE=${IMAGE:-$HOME/containers/foundation-model_rikyu-$VERSION.sif}

STAGE=${1:?usage: submit.sh <stage> [--time HH:MM:SS] [--throttle N] [extra sbatch args...]}
shift || true

TIME=""
THROTTLE=200
EXTRA_SBATCH=()
while [ $# -gt 0 ]; do
    case "$1" in
        --time) TIME="$2"; shift 2 ;;
        --throttle) THROTTLE="$2"; shift 2 ;;
        *) EXTRA_SBATCH+=("$1"); shift ;;
    esac
done

# stage -> probe config | output root | default walltime | fm subcommand | extra fm flags
MODE=pretrain
FLAGS=""
case "$STAGE" in
    # Full-chain smoke in the REAL execution environment before any long run. --sample 400
    # --max-epochs 1 keeps it to a couple of minutes; verify_smoke.sh then checks it actually
    # TRAINED rather than exiting early and reporting a false green (PLAN §5).
    smoke)      CONFIG=probe6.toml; OUT=smoke;    DEFTIME=00:40:00; FLAGS="--sample 400 --max-epochs 1" ;;
    # Stage 0: the anchor. Untuned defaults on 0.3.2, plus v1's adopted config, on probe6.
    # Without this point the v2 tuning gain cannot be separated from PR #45's gain.
    s0)         CONFIG=probe6.toml; OUT=stage0;   DEFTIME=03:00:00 ;;
    # Stage A': encoder x LR x scheduler, jointly.
    a1|a1r|a1b|a2|a3) CONFIG=probe6.toml; OUT=stage_a; DEFTIME=03:00:00 ;;
    # Stage B': multi-task joint head tuning on the A' base.
    breg|bkr|b3)      CONFIG=probe6.toml; OUT=stage_b; DEFTIME=03:00:00 ;;
    # Stage C': 24 tasks, 4 arms. `fm pretrain --resume` is idempotent, so a walltime kill is
    # recovered by resubmitting the identical command; `fm finetune` has NO resume and gets its
    # whole budget in one go.
    c2pre)      CONFIG=final_hybrid_v2.toml;      OUT=stage_c; DEFTIME=48:00:00 ;;
    c2con)      CONFIG=final_consolidate_v2.toml; OUT=stage_c; DEFTIME=10:00:00; MODE=finetune ;;
    *) echo "unknown stage '$STAGE'" >&2; exit 2 ;;
esac
TIME=${TIME:-$DEFTIME}
test -r "$EXP/configs/$CONFIG" || { echo "missing $EXP/configs/$CONFIG" >&2; exit 2; }
test -r "$IMAGE" || { echo "missing image $IMAGE" >&2; exit 2; }

GRID="$EXP/configs/grid_$STAGE.txt"
test -r "$GRID" || { echo "missing $GRID — generate it with scripts/make_grids.py $STAGE" >&2; exit 2; }
N=$(grep -c . "$GRID")

mkdir -p "$OUTBASE/$OUT" "$LOGDIR"
cd "$LOGDIR"

JID=$(sbatch --parsable \
    --job-name="v2$STAGE" \
    --time="$TIME" \
    --array="0-$((N - 1))%$THROTTLE" \
    --export=ALL,GRID="$GRID",CONFIG="experiments/rikyu_hparam_tuning_v2/configs/$CONFIG",OUTROOT="$OUTBASE/$OUT",MODE="$MODE",PROJ="$PROJ",IMAGE="$IMAGE",EXPECT_VERSION="$VERSION",EXTRA_FLAGS="$FLAGS" \
    "${EXTRA_SBATCH[@]}" \
    "$EXP/scripts/fm_array.sbatch")

DONE_ALREADY=$(cut -f1 "$GRID" | while read -r r; do [ -f "$OUTBASE/$OUT/$r/DONE" ] && echo x; done | grep -c . || true)
echo "stage=$STAGE job=$JID points=$N already_done=$DONE_ALREADY mode=$MODE config=$CONFIG"
echo "  out=$OUTBASE/$OUT time=$TIME throttle=$THROTTLE flags='$FLAGS'"
echo "  image=$IMAGE expect_version=$VERSION"
echo "  commit=$(git -C "$PROJ" rev-parse --short HEAD)"
