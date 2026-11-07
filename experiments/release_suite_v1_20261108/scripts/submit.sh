#!/bin/bash
# Submit the release suite on RIKYU. Run on the login node (over ssh: ssh rikyu-login 'bash -lc "..."').
#
#   scripts/submit.sh smoke      2 members, --sample 400, one epoch; then the export check on them
#   scripts/submit.sh suite      all members (configs/members_$SET.txt), PACK per GPU; then export
#   scripts/submit.sh export     export only (every member must have a finished consolidation)
#
# Resubmitting `suite` is the recovery command after a TIMEOUT: finished stages are skipped and an
# interrupted pretrain resumes from its last completed step. The export job waits on the suite with
# afterok; if the suite fails, cancel the pending export and resubmit both.
#
# Env: SET (v1), VERSION (1.0.0), PACK (5), RIKYU_ACCOUNT (optional, never stored in the repo).

set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="$(cd "$EXP/../.." && pwd)"
SET=${SET:-v1}
VERSION=${VERSION:-1.0.0}
PACK=${PACK:-5}
IMAGE=${IMAGE:-$HOME/containers/foundation-model_rikyu-$VERSION.sif}
OUTBASE=${OUTBASE:-/data1/rkp00067/rku00225/fm/release_suite_$SET}
LOGDIR=${LOGDIR:-$HOME/jobs/release_suite_$SET}
ACCOUNT=()
[ -n "${RIKYU_ACCOUNT:-}" ] && ACCOUNT=(--account="$RIKYU_ACCOUNT")
STAGE=${1:?usage: submit.sh smoke|suite|export}
mkdir -p "$LOGDIR"
cd "$LOGDIR"

common="ALL,PROJ=$PROJ,IMAGE=$IMAGE,EXPECT_VERSION=$VERSION,SET=$SET"
n=$(grep -c . "$EXP/configs/members_$SET.txt")
case "$STAGE" in
    smoke)
        out="$OUTBASE/smoke"; mkdir -p "$out"
        jid=$(sbatch --parsable "${ACCOUNT[@]}" --array=0 --time=02:00:00 --job-name=fm-release-smoke \
            --export="$common,OUTROOT=$out,PACK=2,SMOKE=1" "$EXP/scripts/suite.sbatch")
        eid=$(sbatch --parsable "${ACCOUNT[@]}" --dependency=afterok:"$jid" --time=01:00:00 \
            --export="$common,OUTROOT=$out,MEMBER_LIMIT=2" "$EXP/scripts/export.sbatch")
        ;;
    suite)
        out="$OUTBASE/members"; mkdir -p "$out"
        jid=$(sbatch --parsable "${ACCOUNT[@]}" --array=0-$(((n + PACK - 1) / PACK - 1)) \
            --export="$common,OUTROOT=$out,PACK=$PACK" "$EXP/scripts/suite.sbatch")
        eid=$(sbatch --parsable "${ACCOUNT[@]}" --dependency=afterok:"$jid" \
            --export="$common,OUTROOT=$out" "$EXP/scripts/export.sbatch")
        ;;
    export)
        out="$OUTBASE/members"; jid=-
        eid=$(sbatch --parsable "${ACCOUNT[@]}" --export="$common,OUTROOT=$out" "$EXP/scripts/export.sbatch")
        ;;
    *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
echo "stage=$STAGE set=$SET members=$n pack=$PACK image=$IMAGE out=$out train_job=$jid export_job=$eid git=$(git -C "$PROJ" rev-parse --short HEAD)"
