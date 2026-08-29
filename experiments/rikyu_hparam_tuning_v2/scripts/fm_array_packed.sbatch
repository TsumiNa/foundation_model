#!/bin/bash
# Packed RIKYU grid-array worker: PACK grid points share ONE GPU.
#
# Why this exists. Slurm's own accounting says every run in both campaigns uses about 9% of its
# GPU (v2 A'1: 8-10% across 394 runs; v1 A1: 6-8%; v1's 20-hour 24-task Stage C run: 7%), holds
# 1.29 GB of a GB200's 189 GB, and burns roughly one of its 18 allocated CPUs. The model is an
# MLP encoder at batch 256 — a single step is trivial for this hardware — so the GPU cannot be
# saturated by one run no matter how the input pipeline is tuned. Running several independent
# runs on the same card is the only lever that changes the arithmetic.
#
# The runs are separate processes with separate seeds, separate output directories and no shared
# state, so packing changes THROUGHPUT and nothing else about the numbers. What it does change is
# wall clock per run: packed runs contend, so a timing figure from a packed stage must never be
# put beside one from an unpacked stage.
#
# Grid file format is unchanged, so the same grid files work under either worker:
#   <runid>\t<shell-quoted `fm` overrides>
#
# Array indexing differs: task i handles grid lines [i*PACK, i*PACK+PACK), so the array is
# ceil(N/PACK) tasks rather than N. submit.sh computes that.
#
#SBATCH --job-name=fm-grid-v2p
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -uo pipefail

module purge

PROJ=${PROJ:?set PROJ=/abs/path to the repo checkout}
MODE=${MODE:-pretrain}
EXTRA_FLAGS=${EXTRA_FLAGS:-}
PACK=${PACK:-4}
: "${IMAGE:?set IMAGE=/abs/path/to/foundation-model_rikyu-<version>.sif}"
: "${EXPECT_VERSION:?set EXPECT_VERSION=<x.y.z> the container must report}"
: "${GRID:?set GRID=/abs/path/to/grid.txt}"
: "${CONFIG:?set CONFIG=repo-relative path to the TOML}"
: "${OUTROOT:?set OUTROOT=/abs/path for run outputs}"

test -r "$IMAGE"
test -r "$GRID"
test -r "$PROJ/$CONFIG"

# SRC_OVERRIDE runs the container's environment against a DIFFERENT source tree, by putting it
# ahead of the installed package on PYTHONPATH. Used to probe a change that is not in any image
# yet: the dependencies stay byte-identical to every other run in the campaign, so the only
# difference is the code under test. Verified on the login node — the unpatched container reports
# the package from /opt/foundation-model/.venv, and with the override it reports it from the bound
# tree at the same version.
SRC_ARGS=()
if [ -n "${SRC_OVERRIDE:-}" ]; then
    test -d "$SRC_OVERRIDE/src/foundation_model"
    SRC_ARGS=(--bind "$SRC_OVERRIDE:/patched:ro" --env PYTHONPATH=/patched/src)
    echo "SOURCE OVERRIDE: $SRC_OVERRIDE (container env, patched code)"
fi

# --- image identity gate, once per allocation rather than once per run -------------------------
# Same gate as the unpacked worker and for the same reason: a wrong image would train every point
# in this allocation under 0.2.1's per-batch scheduler cadence, exit 0, and drop DONE markers.
GOT_VERSION=$(apptainer exec "$IMAGE" python -c \
    'import importlib.metadata as m; print(m.version("foundation-model"))' 2>/dev/null || echo "UNKNOWN")
if [ "$GOT_VERSION" != "$EXPECT_VERSION" ]; then
    echo "FAIL image-version mismatch: expected $EXPECT_VERSION, container reports $GOT_VERSION"
    echo "     IMAGE=$IMAGE"
    echo "     Refusing to train — this is the silent-wrong-regime failure this gate exists for."
    exit 3
fi

GIT_SHA=$(git -C "$PROJ" rev-parse --short HEAD 2>/dev/null || echo unknown)
START_IDX=$((SLURM_ARRAY_TASK_ID * PACK))

echo "host=$(hostname) pack=$PACK lines=[$START_IDX,$((START_IDX + PACK))) fm=$GOT_VERSION"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_one() {
    local idx=$1
    local line runid overrides out start elapsed rc
    line=$(sed -n "$((idx + 1))p" "$GRID")
    [ -n "$line" ] || return 0            # past the end of the grid: nothing to do

    runid=${line%%$'\t'*}
    overrides=${line#*$'\t'}
    out="$OUTROOT/$runid"
    mkdir -p "$out"

    if [ -f "$out/DONE" ]; then
        echo "SKIP $runid (DONE marker present)"
        return 0
    fi

    cat > "$out/ENV.json" <<JSON
{
  "runid": "$runid",
  "image": "$IMAGE",
  "fm_version": "$GOT_VERSION",
  "expected_version": "$EXPECT_VERSION",
  "git_sha": "$GIT_SHA",
  "config": "$CONFIG",
  "mode": "$MODE",
  "host": "$(hostname)",
  "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
  "packed_with": $PACK,
  "src_override": "${SRC_OVERRIDE:-}",
  "started": "$(date -Is)"
}
JSON

    # Re-parse the shell-quoted override string into positional args so quoted TOML values survive.
    eval "set -- $overrides"

    start=$SECONDS
    apptainer exec --nv \
        "${SRC_ARGS[@]}" \
        --bind "$PROJ:/workspace:ro" \
        --bind "$OUTROOT:/out" \
        --env MPLCONFIGDIR="/tmp/mplconfig_$runid" \
        --pwd /workspace \
        "$IMAGE" \
        fm "$MODE" --config "$CONFIG" --output-dir "/out/$runid" $EXTRA_FLAGS "$@" \
        > "$out/run_stdout.log" 2>&1
    rc=$?
    elapsed=$((SECONDS - start))

    # One short line appended O_APPEND from at most PACK concurrent writers on one node: under
    # PIPE_BUF this is atomic, so the file does not need locking.
    printf '%s\t%s\t%s\t%s\t%s\n' "$runid" "$elapsed" "$rc" "$(hostname)" "packed$PACK" \
        >> "$OUTROOT/_timing.tsv"

    if [ "$rc" -eq 0 ]; then
        date -Is > "$out/DONE"
        echo "PASS $runid elapsed=${elapsed}s"
    else
        echo "FAIL $runid rc=$rc elapsed=${elapsed}s (see $out/run_stdout.log)"
    fi
    return "$rc"
}

pids=()
for k in $(seq 0 $((PACK - 1))); do
    run_one $((START_IDX + k)) &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done

echo "task done: $failed of ${#pids[@]} runs failed"
# Non-zero when any packed run failed, so a stage's failures stay visible in sacct rather than
# being averaged away by its co-tenants succeeding.
exit $((failed > 0 ? 1 : 0))
