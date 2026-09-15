#!/bin/bash
# Recompute everything that depends on the transfer stage, once it has finished at n=10.
#
# Written while the runs were still going, so that finishing is one command rather than a sequence
# reconstructed from memory at the end.
#
# The work is split by dependency, not by preference. xfer.py, position.py and the library builder
# import nothing outside the standard library, so they run on RIKYU where the 240-run tree lives and
# only their JSON output travels back. matched_test.py needs pandas, plots.py needs matplotlib and
# the deck needs python-pptx -- none of which exist on the login node -- so those run here, against
# prediction files pulled down on demand.
#
# Every step refuses to run on incomplete data rather than silently scoring a partial stage. That
# failure mode already cost this campaign one scoring pass against step 23 of an unfinished arm.
#
#   bash scripts/finalise_xfer.sh              # verify, analyse, plot, build deck and library
#   EXPECT=240 bash scripts/finalise_xfer.sh   # override the expected run count
set -uo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST=${HOST:-rikyu-login}
OUT=${OUTBASE:-/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning_v2}
X="$OUT/stage_xfer"
SUM="$EXP/summary"
RES="$EXP/results"
PRED="$EXP/.preds"          # scratch for the parquet files matched_test needs
EXPECT=${EXPECT:-240}
DATE=${DATE:-$(date +%Y%m%d)}

cd "$EXP" || exit 1
fail() { echo "ABORT: $*" >&2; exit 1; }
# The remote script arrives on stdin, not as an argument. Passing it as `bash -lc '...'` makes the
# outer quotes strip the inner ones, which silently broke a sed expression in the completeness check.
remote() { ssh -o ConnectTimeout=30 "$HOST" "bash -l -s"; }

# ---- 0. the stage must actually be finished ---------------------------------------------------
echo "== checking the stage is complete =="
state=$(remote <<REMOTE
X=$X
done_n=\$(ls -d \$X/*/DONE 2>/dev/null | wc -l)
short=0
for d in \$X/*/; do
  [ -e "\$d/DONE" ] || continue
  n=\$(ls \$d/training 2>/dev/null | sed -n 's/^step0*\([0-9]*\)_.*/\1/p' | sort -n | tail -1)
  [ "\${n:-0}" -lt 24 ] && short=\$((short+1))
done
echo "\$done_n \$short"
REMOTE
) || fail "cannot reach $HOST"
done_n=$(echo "$state" | awk '{print $1+0}')
short=$(echo "$state" | awk '{print $2+0}')
echo "  $done_n / $EXPECT runs carry a DONE marker; $short of them have fewer than 24 steps"
[ "$done_n" -ge "$EXPECT" ] || fail "only $done_n of $EXPECT finished — refusing to score a partial stage"
# A DONE marker on a short run would silently shrink the position curve rather than error.
[ "$short" -eq 0 ] || fail "$short run(s) carry DONE but have fewer than 24 steps"

# ---- 1. stdlib-only analyses, run where the data is -------------------------------------------
# The remote checkout is a mirror, not a clone that pulls: a script added here after the last
# mirror does not exist there, which is how the library step failed the first time this ran.
echo "== syncing the analysis scripts to $HOST =="
# Relative, so it resolves against the REMOTE home. An unquoted ~ expands locally at assignment
# time, which pointed the first attempt at this machine's own home directory.
REMOTE_EXP=projects/foundation_model_v2/experiments/rikyu_hparam_tuning_v2
rsync -aq analysis/ "$HOST:$REMOTE_EXP/analysis/" || fail "rsync analysis/"
rsync -aq scripts/build_model_library.py "$HOST:$REMOTE_EXP/scripts/" || fail "rsync library builder"

echo "== transfer and position, on $HOST =="
remote <<REMOTE || fail "remote analyses"
set -e
cd ~/projects/foundation_model_v2/experiments/rikyu_hparam_tuning_v2
python3 analysis/xfer.py --runs $X --ceilings summary/ceilings_adopted.json \
    -o summary/transfer_xfer.json
python3 analysis/position.py --runs $X --ceilings summary/ceilings_adopted.json \
    -o summary/position.json
REMOTE

echo "== model library, on $HOST =="
# --copy so the library is self-contained: referencing checkpoints in place leaves it broken the
# moment the run tree is pruned.
remote <<REMOTE || fail "model library"
set -e
cd ~/projects/foundation_model_v2/experiments/rikyu_hparam_tuning_v2
python3 scripts/build_model_library.py --runs $X -o $OUT/model_library --copy
REMOTE

echo "== pulling results back =="
for f in transfer_xfer.json position.json; do
    scp -q "$HOST:~/projects/foundation_model_v2/experiments/rikyu_hparam_tuning_v2/summary/$f" \
        "$SUM/$f" || fail "scp $f"
done
scp -q "$HOST:$OUT/model_library/MANIFEST.json" "$SUM/model_library_manifest.json" \
    || echo "  (manifest not copied — it is large; it stays on RIKYU)"

# ---- 2. matched-row comparison, which needs pandas --------------------------------------------
echo "== pulling predictions for the matched comparison =="
mkdir -p "$PRED"
# A step directory holds predictions for EVERY task trained so far, but the comparison needs only
# the one task the run was built to test. Pulling everything moved 10 GB on the first attempt; the
# per-task patterns below cut that to a fraction, at the cost of one rsync per task.
mapfile -t TASKS < <(ssh -o ConnectTimeout=30 "$HOST" \
    "ls -d $X/xf_*_o0 2>/dev/null | sed 's|.*/xf_||; s|_o0$||'" | sort -u)
[ "${#TASKS[@]}" -gt 0 ] || fail "could not list the tasks under test"
echo "  ${#TASKS[@]} tasks to pull"
for task in "${TASKS[@]}"; do
    rsync -aq --prune-empty-dirs \
        --include='*/' --include="${task}_pred.parquet" --include='DONE' --exclude='*' \
        "$HOST:$OUT/stage_single/stA_${task}_s*/" "$PRED/stage_single/" 2>/dev/null
    rsync -aq --prune-empty-dirs --relative \
        --include='*/' --include="step24_${task}/${task}_pred.parquet" --include='DONE' \
        --exclude='*' \
        "$HOST:$X/./xf_${task}_o*/" "$PRED/stage_xfer/" 2>/dev/null
done
# matched_test.py gates on the DONE marker, so a filter that pulls only predictions makes every run
# look unfinished and every task report "missing runs" — a filter bug wearing a data-shaped error.
du -sh "$PRED" | awk '{print "  pulled " $1}'

# Local steps run through uv: the project's pandas and matplotlib live in its environment, not in
# the system python3. python-pptx is not a project dependency, so the deck asks for it explicitly.
echo "== matched comparison =="
uv run python analysis/matched_test.py --single "$PRED/stage_single" \
    --multi-glob "$PRED/stage_xfer/xf_{task}_o*" --label xfer \
    -o "$SUM/matched_xfer.json" || fail "matched_test.py"

# ---- 3. figures and deck ----------------------------------------------------------------------
echo "== figures =="
uv run python analysis/plots.py --summary "$SUM" -o "$RES" || fail "plots.py"

echo "== deck =="
uv run --with python-pptx python build_report_pptx.py --date "${DATE:0:4}-${DATE:4:2}-${DATE:6:2}" \
    -o "$RES/REPORT_v2_$DATE.pptx" || fail "build_report_pptx.py"

echo
echo "=== done ==="
echo "  summary:  $SUM/{transfer_xfer,matched_xfer,position}.json"
echo "  deck:     $RES/REPORT_v2_$DATE.pptx"
echo "  library:  $HOST:$OUT/model_library/{MANIFEST.json,INDEX.md,models/}"
echo
echo "  Numbers in the report prose are NOT regenerated — reread the sections that quote"
echo "  transfer, position or library figures and update them against the JSON above."
