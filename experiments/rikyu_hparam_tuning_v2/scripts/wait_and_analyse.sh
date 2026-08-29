#!/bin/bash
# Block until the named Slurm jobs leave the queue, then verify and score the stages they fed.
#
#   scripts/wait_and_analyse.sh 53185 52398 52500
#
# Verification runs BEFORE scoring and its exit status is reported, because a stage that exited 0
# is not the same as a stage that trained: stage 0's first attempt returned eleven green runs and
# seven that had been killed at the walltime, and only the per-run check tells those apart.
#
# Scoring stage A' needs stage 0's summary — the untuned anchor is the reference every A' margin
# is quoted against — so the two run in that order and A' is skipped if the anchor is not there.

set -uo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="$(cd "$EXP/../.." && pwd)"
OUTBASE=${OUTBASE:-/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning_v2}

for job in "$@"; do
    while squeue -h -j "$job" -o %i 2>/dev/null | grep -q .; do sleep 60; done
    echo "job $job left the queue"
done

cd "$PROJ"
mkdir -p "$EXP/summary"

echo "=== stage 0: verify ==="
python3 "$EXP/scripts/verify_runs.py" "$OUTBASE/stage0" --expect-runs 18
s0_ok=$?
echo "verify exit=$s0_ok"

# Verification GATES scoring rather than merely reporting. stage0.py drops incomplete runs
# silently, so a failed verify followed by a successful score overwrites the anchor with a partial
# sample — and every later stage quotes its margins against that anchor. ALLOW_UNVERIFIED=1 exists
# for a deliberate re-score of runs already known to be good.
if [ "$s0_ok" -ne 0 ] && [ "${ALLOW_UNVERIFIED:-0}" != "1" ]; then
    echo "stage 0 verification FAILED (exit $s0_ok) — refusing to overwrite the anchor."
    echo "Fix the runs, or set ALLOW_UNVERIFIED=1 to score anyway."
    exit "$s0_ok"
fi

echo "=== stage 0: score ==="
python3 "$EXP/analysis/stage0.py" --runs "$OUTBASE/stage0" -o "$EXP/summary/stage0.json"

if [ ! -r "$EXP/summary/stage0.json" ]; then
    echo "no stage0 summary — stage A' cannot be scored against an anchor that does not exist"
    exit 1
fi

echo "=== stage A': verify (sampled) ==="
# Every run, but only the failure lines: 1080 'ok' lines would bury the ones that matter.
# PIPESTATUS, not $?, because the pipe's exit status is grep's.
python3 "$EXP/scripts/verify_runs.py" "$OUTBASE/stage_a" 2>&1 | grep -vE '^ok ' | tail -40
sa_ok=${PIPESTATUS[0]}
if [ "$sa_ok" -ne 0 ] && [ "${ALLOW_UNVERIFIED:-0}" != "1" ]; then
    echo "stage A' verification FAILED (exit $sa_ok) — refusing to score."
    exit "$sa_ok"
fi

echo "=== stage A': score ==="
python3 "$EXP/analysis/stage_a.py" --runs "$OUTBASE/stage_a" \
    --stage0 "$EXP/summary/stage0.json" -o "$EXP/summary/stage_a.json" --top 8
echo "stage_a exit=$?  (2 = short list is edge-bound, an a1b round is required)"
