#!/bin/bash
# Release suite worker: PACK members share one GPU; each member = pretrain (--resume) -> consolidate.
#
# Members file: one line per member, "<member id>\t<task_order_seed>" (configs/members_<set>.txt).
# Array task i handles lines [i*PACK, i*PACK+PACK). Every stage of every member is idempotent: a
# finished stage drops a DONE marker and is skipped, and an interrupted pretrain resumes from its last
# completed step, so resubmitting the same array is always the recovery command.
#
# The only thing that differs between members is task_order_seed. The training seed, data, split
# manifest and every hyper-parameter come from the one config.
#
# Required env: PROJ IMAGE EXPECT_VERSION SET OUTROOT. Optional: SMOKE=1 (--sample 400, one epoch).
#
#SBATCH --job-name=fm-release
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -uo pipefail
module purge

: "${PROJ:?set PROJ=/abs/path to the repo checkout}"
: "${IMAGE:?set IMAGE=/abs/path/to/foundation-model_rikyu-<version>.sif}"
: "${EXPECT_VERSION:?set EXPECT_VERSION=<x.y.z>}"
: "${SET:?set SET=v1|full}"
: "${OUTROOT:?set OUTROOT=/abs/path for member outputs}"
PACK=${PACK:-5}
SMOKE=${SMOKE:-0}
EXP=experiments/release_suite_v1_20261108
MEMBERS="$PROJ/$EXP/configs/members_$SET.txt"
PRE_CFG="$EXP/configs/pretrain_$SET.toml"
CON_CFG="$EXP/configs/consolidate_$SET.toml"
for f in "$IMAGE" "$MEMBERS" "$PROJ/$PRE_CFG" "$PROJ/$CON_CFG"; do
    test -r "$f" || { echo "FAIL cannot read $f" >&2; exit 2; }
done

GOT=$(apptainer exec "$IMAGE" python -c 'import importlib.metadata as m; print(m.version("foundation-model"))' 2>/dev/null || echo UNKNOWN)
if [ "$GOT" != "$EXPECT_VERSION" ]; then
    echo "FAIL image reports foundation-model $GOT, expected $EXPECT_VERSION ($IMAGE)"
    exit 3
fi
GIT_SHA=$(git -C "$PROJ" rev-parse HEAD 2>/dev/null || echo unknown)
PRE_FLAGS=(); CON_FLAGS=()
if [ "$SMOKE" = 1 ]; then PRE_FLAGS=(--sample 400 --max-epochs 1); CON_FLAGS=(--sample 400 --epochs 1); fi
START=$((SLURM_ARRAY_TASK_ID * PACK))
echo "host=$(hostname) set=$SET pack=$PACK lines=[$START,$((START + PACK))) fm=$GOT git=$GIT_SHA smoke=$SMOKE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

fm_in() {  # fm_in <log> <fm args...>
    local log=$1; shift
    apptainer exec --nv --bind "$PROJ:/workspace:ro" --bind "$OUTROOT:/out" \
        --env MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}_$RANDOM" --pwd /workspace \
        "$IMAGE" fm "$@" >>"$log" 2>&1
}

run_member() {
    local idx=$1 line id seed out t0 rc
    line=$(sed -n "$((idx + 1))p" "$MEMBERS")
    [ -n "$line" ] || return 0
    id=${line%%$'\t'*}; seed=${line#*$'\t'}
    out="$OUTROOT/$id"; mkdir -p "$out"
    printf '{"member": "%s", "task_order_seed": %s, "image": "%s", "fm_version": "%s", "git": "%s", "slurm": "%s_%s", "pack": %s, "smoke": %s, "started": "%s"}\n' \
        "$id" "$seed" "$IMAGE" "$GOT" "$GIT_SHA" "${SLURM_ARRAY_JOB_ID:-}" "${SLURM_ARRAY_TASK_ID:-}" "$PACK" "$SMOKE" "$(date -Is)" \
        >>"$out/ENV.jsonl"

    if [ ! -f "$out/pretrain/DONE" ]; then
        t0=$SECONDS
        fm_in "$out/pretrain.log" pretrain --config "$PRE_CFG" --output-dir "/out/$id/pretrain" --resume \
            --set "pretrain.task_order_seed=$seed" "${PRE_FLAGS[@]}"
        rc=$?
        printf '%s\tpretrain\t%s\t%s\t%s\n' "$id" $((SECONDS - t0)) "$rc" "$(hostname)" >>"$OUTROOT/_timing.tsv"
        [ "$rc" -eq 0 ] || { echo "FAIL $id pretrain rc=$rc (see $out/pretrain.log)"; return "$rc"; }
        date -Is >"$out/pretrain/DONE"
    fi

    if [ ! -f "$out/consolidate/DONE" ]; then
        # fm finetune cannot resume: an interrupted attempt is set aside (never deleted) and redone.
        [ -e "$out/consolidate" ] && mv "$out/consolidate" "$out/consolidate.interrupted.$(date +%Y%m%dT%H%M%S)"
        t0=$SECONDS
        fm_in "$out/consolidate.log" finetune --config "$CON_CFG" --output-dir "/out/$id/consolidate" \
            --checkpoint "/out/$id/pretrain/training/final_model.pt" "${CON_FLAGS[@]}"
        rc=$?
        printf '%s\tconsolidate\t%s\t%s\t%s\n' "$id" $((SECONDS - t0)) "$rc" "$(hostname)" >>"$OUTROOT/_timing.tsv"
        [ "$rc" -eq 0 ] || { echo "FAIL $id consolidate rc=$rc (see $out/consolidate.log)"; return "$rc"; }
        date -Is >"$out/consolidate/DONE"
    fi
    echo "PASS $id"
}

pids=()
for k in $(seq 0 $((PACK - 1))); do
    run_member $((START + k)) &
    pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
echo "task done: $failed of ${#pids[@]} members failed"
exit $((failed > 0 ? 1 : 0))
