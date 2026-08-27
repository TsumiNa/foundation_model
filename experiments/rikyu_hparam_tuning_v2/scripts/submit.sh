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
PACK=1
EXTRA_SBATCH=()
while [ $# -gt 0 ]; do
    case "$1" in
        --time) TIME="$2"; shift 2 ;;
        --throttle) THROTTLE="$2"; shift 2 ;;
        --pack) PACK="$2"; shift 2 ;;
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
    s0)         CONFIG=probe6.toml; OUT=stage0;   DEFTIME=06:00:00 ;;
    # Packing calibration. Re-runs grid points that ALREADY completed unpacked, into a scratch
    # output root, so the speedup is measured against a known per-run baseline for the same
    # configurations rather than against an average over a different set of points. 12h because a
    # packed run is expected to be slower in wall clock even as throughput rises.
    packcal)    CONFIG=probe6.toml; OUT=packcal;  DEFTIME=12:00:00 ;;
    # Stage A' (encoder x LR x scheduler, jointly) and stage B' (joint head tuning on A's base).
    #
    # Six hours, not the three a probe6 run needs. Probe grid lines carry no `--resume` — probe
    # runs are short enough that resuming is not worth the partial-metrics_table.csv complication
    # it introduces — so a walltime kill throws the whole run away and it re-runs from zero.
    # Measured on stage 0: steps run 45-75 epochs each and get more expensive as replay
    # accumulates, putting a run at 1.0-1.5h. That is close enough to three hours that a slow
    # configuration could cross it, and with 217 idle nodes the over-request costs nothing.
    a1|a1r|a1b|a2|a3|a4) CONFIG=probe6.toml; OUT=stage_a; DEFTIME=06:00:00 ;;
    b|b3)             CONFIG=probe6.toml; OUT=stage_b; DEFTIME=06:00:00 ;;
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

# --pack N puts N grid points on ONE GPU. Slurm accounting says a single run uses ~9% of a GB200
# and 1.29 GB of its 189 GB, so one run per card wastes about nine tenths of the reservation. The
# runs are independent processes, so this trades wall clock per run for throughput and leaves the
# numbers untouched. Array size shrinks to ceil(N/PACK) and each task walks its own slice.
if [ "$PACK" -gt 1 ]; then
    WORKER="$EXP/scripts/fm_array_packed.sbatch"
    ARRAY_N=$(( (N + PACK - 1) / PACK ))
    # ~8 CPUs per co-tenant: measured usage is about one core each, and the headroom absorbs the
    # bursts of input preparation without letting the pack become CPU-bound.
    #
    # Hard-capped at 32 because the site enforces a per-GPU CPU limit and rejects the job
    # outright above it: "[AI4S] Requested CPUs (64 cpus-per-task x 1 tasks = 64) exceed the
    # per-GPU cap 32 (= 1 GPU x 32)". At PACK=8 that leaves 4 cores per co-tenant, still well
    # above the ~1 core a run was measured to use.
    CPUS=$(( PACK * 8 )); [ "$CPUS" -gt 32 ] && CPUS=32
    PACK_ARGS=(--cpus-per-task="$CPUS")
else
    WORKER="$EXP/scripts/fm_array.sbatch"
    ARRAY_N=$N
    PACK_ARGS=()
fi

JID=$(sbatch --parsable \
    --job-name="v2$STAGE" \
    --time="$TIME" \
    --array="0-$((ARRAY_N - 1))%$THROTTLE" \
    --export=ALL,GRID="$GRID",CONFIG="experiments/rikyu_hparam_tuning_v2/configs/$CONFIG",OUTROOT="$OUTBASE/$OUT",MODE="$MODE",PROJ="$PROJ",IMAGE="$IMAGE",EXPECT_VERSION="$VERSION",EXTRA_FLAGS="$FLAGS",PACK="$PACK" \
    "${PACK_ARGS[@]}" \
    "${EXTRA_SBATCH[@]}" \
    "$WORKER")

DONE_ALREADY=$(cut -f1 "$GRID" | while read -r r; do [ -f "$OUTBASE/$OUT/$r/DONE" ] && echo x; done | grep -c . || true)
echo "stage=$STAGE job=$JID points=$N already_done=$DONE_ALREADY mode=$MODE config=$CONFIG"
echo "  out=$OUTBASE/$OUT time=$TIME throttle=$THROTTLE flags='$FLAGS'"
echo "  pack=$PACK array_tasks=$ARRAY_N worker=$(basename "$WORKER") gpus_in_flight=$((THROTTLE < ARRAY_N ? THROTTLE : ARRAY_N))"
echo "  image=$IMAGE expect_version=$VERSION"
echo "  commit=$(git -C "$PROJ" rev-parse --short HEAD)"
