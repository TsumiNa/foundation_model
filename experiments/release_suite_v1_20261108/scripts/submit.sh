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
# Env: SET (v1); TRAIN_VERSION (1.0.0, the image v1 was trained with); EXPORT_VERSION (1.0.1: the first
# release that guards bounded inverses, so older images must not export); PACK (5); OUTBASE, or
# RIKYU_GROUP_DIR to derive it; RIKYU_ACCOUNT (optional). Accounts and group paths are never stored here.

set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="$(cd "$EXP/../.." && pwd)"
SET=${SET:-v1}
TRAIN_VERSION=${TRAIN_VERSION:-1.0.0}
EXPORT_VERSION=${EXPORT_VERSION:-1.0.1}
PACK=${PACK:-5}
TRAIN_IMAGE=${TRAIN_IMAGE:-$HOME/containers/foundation-model_rikyu-$TRAIN_VERSION.sif}
EXPORT_IMAGE=${EXPORT_IMAGE:-$HOME/containers/foundation-model_rikyu-$EXPORT_VERSION.sif}
OUTBASE=${OUTBASE:-${RIKYU_GROUP_DIR:?set OUTBASE, or RIKYU_GROUP_DIR to the group storage}/fm/release_suite_$SET}
case "$EXPORT_VERSION" in 0.* | 1.0.0) echo "EXPORT_VERSION $EXPORT_VERSION has no inverse guards; use 1.0.1+" >&2; exit 2 ;; esac
LOGDIR=${LOGDIR:-$HOME/jobs/release_suite_$SET}
ACCOUNT=()
[ -n "${RIKYU_ACCOUNT:-}" ] && ACCOUNT=(--account="$RIKYU_ACCOUNT")
STAGE=${1:?usage: submit.sh smoke|suite|export}
mkdir -p "$LOGDIR"
cd "$LOGDIR"

common="ALL,PROJ=$PROJ,SET=$SET"
# --export=ALL carries the login shell's variables into the jobs, so a SMOKE or MEMBER_LIMIT left over
# from a debug run is set explicitly on every job (each key once: Slurm does not promise which duplicate wins).
train_env="$common,IMAGE=$TRAIN_IMAGE,EXPECT_VERSION=$TRAIN_VERSION"
export_env="$common,IMAGE=$EXPORT_IMAGE,EXPECT_VERSION=$EXPORT_VERSION"
n=$(grep -c . "$EXP/configs/members_$SET.txt")
case "$STAGE" in
    smoke)
        out="$OUTBASE/smoke"; mkdir -p "$out"
        jid=$(sbatch --parsable "${ACCOUNT[@]}" --array=0 --time=02:00:00 --job-name=fm-release-smoke \
            --export="$train_env,OUTROOT=$out,PACK=2,SMOKE=1" "$EXP/scripts/suite.sbatch")
        eid=$(sbatch --parsable "${ACCOUNT[@]}" --dependency=afterok:"$jid" --time=01:00:00 \
            --export="$export_env,OUTROOT=$out,MEMBER_LIMIT=2" "$EXP/scripts/export.sbatch")
        ;;
    suite)
        out="$OUTBASE/members"; mkdir -p "$out"
        jid=$(sbatch --parsable "${ACCOUNT[@]}" --array=0-$(((n + PACK - 1) / PACK - 1)) \
            --export="$train_env,OUTROOT=$out,PACK=$PACK,SMOKE=0" "$EXP/scripts/suite.sbatch")
        eid=$(sbatch --parsable "${ACCOUNT[@]}" --dependency=afterok:"$jid" \
            --export="$export_env,OUTROOT=$out,MEMBER_LIMIT=" "$EXP/scripts/export.sbatch")
        ;;
    export)
        out="$OUTBASE/members"; jid=-
        eid=$(sbatch --parsable "${ACCOUNT[@]}" --export="$export_env,OUTROOT=$out,MEMBER_LIMIT=" "$EXP/scripts/export.sbatch")
        ;;
    *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
echo "stage=$STAGE set=$SET members=$n pack=$PACK train_image=$TRAIN_IMAGE export_image=$EXPORT_IMAGE out=$out train_job=$jid export_job=$eid git=$(git -C "$PROJ" rev-parse --short HEAD)"
