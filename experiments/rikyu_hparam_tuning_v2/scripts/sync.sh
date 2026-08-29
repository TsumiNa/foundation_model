#!/bin/bash
# Push the v2 experiment directory to RIKYU (code and configs only), or pull results back.
#
#   scripts/sync.sh push      # local -> RIKYU checkout (configs, scripts, analysis)
#   scripts/sync.sh pull      # RIKYU raw run output -> local results/raw/
#
# Run from the local machine. Results never travel by git — `experiments/*/results/`, `*.png` and
# `*.csv` are all gitignored — so this is the transport for them. Executable bits do not survive
# rsync's default mode here, which is why push re-applies them; without that the next
# `submit.sh` fails with a bare "Permission denied" that looks like a cluster problem.

set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST=${HOST:-rikyu-login}
REMOTE_EXP=${REMOTE_EXP:-projects/foundation_model_v2/experiments/rikyu_hparam_tuning_v2}
OUTBASE=${OUTBASE:-/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning_v2}

case "${1:-push}" in
push)
    rsync -az --partial \
        --exclude 'results/' --exclude '*.png' --exclude '*.csv' --exclude '__pycache__/' \
        "$EXP/" "$HOST:$REMOTE_EXP/"
    ssh "$HOST" "chmod +x $REMOTE_EXP/scripts/*.sh $REMOTE_EXP/scripts/*.sbatch $REMOTE_EXP/scripts/*.py"
    # Commit the mirror so `git rev-parse HEAD` there identifies the scripts that actually ran.
    # Every run records that sha in its ENV.json, and an uncommitted mirror would make the
    # recorded sha point at whatever the checkout happened to be sitting on instead — provenance
    # that looks precise and means nothing.
    ssh "$HOST" "cd \$(dirname $REMOTE_EXP)/.. && git add -A experiments/rikyu_hparam_tuning_v2 && \
        (git diff --cached --quiet || git -c user.name='v2 mirror' -c user.email='noreply@local' \
         commit -q -m 'exp(v2): mirror of the local working tree' ) && git rev-parse --short HEAD"
    echo "pushed $EXP -> $HOST:$REMOTE_EXP"
    ;;
pull)
    mkdir -p "$EXP/results/raw"
    rsync -az --partial "$HOST:$OUTBASE/" "$EXP/results/raw/"
    # Summary JSON is the one artefact that must also reach git (PLAN §9.3).
    rsync -az --partial "$HOST:$REMOTE_EXP/summary/" "$EXP/summary/" 2>/dev/null || true
    echo "pulled $HOST:$OUTBASE -> $EXP/results/raw"
    du -sh "$EXP/results/raw"
    ;;
*)
    echo "usage: sync.sh [push|pull]" >&2
    exit 2
    ;;
esac
