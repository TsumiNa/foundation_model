#!/bin/bash
# Score the warm-start fine-tune stage once all 480 runs are in: guard, analyse on RIKYU, pull back.
#
# Same shape as finalise_xfer.sh and for the same reason: the analysis refuses a partial stage rather
# than scoring one, because a DONE count that looks complete has lied to this campaign before.
#
#   bash scripts/finalise_ft.sh              # expects 480 DONE (24 tasks x 10 orderings x 2 arms)
#   EXPECT=480 bash scripts/finalise_ft.sh
set -uo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST=${HOST:-rikyu-login}
OUT=${OUTBASE:-/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning_v2}
EXPECT=${EXPECT:-480}
REMOTE_EXP=projects/foundation_model_v2/experiments/rikyu_hparam_tuning_v2
cd "$EXP" || exit 1
fail() { echo "ABORT: $*" >&2; exit 1; }
remote() { ssh -o ConnectTimeout=30 "$HOST" "bash -l -s"; }

echo "== checking the stage is complete =="
state=$(remote <<REMOTE
B=$OUT/stage_ft
d=\$(ls -d \$B/ft*_o*/DONE 2>/dev/null | wc -l)
# a DONE marker without the task's metrics file is a run that exited without evaluating
m=0
for r in \$B/ft*_o*/; do
  [ -e "\$r/DONE" ] || continue
  t=\$(basename \$r); t=\${t#ft?_}; t=\${t%_o*}
  [ -f "\$r/training/finetune/\${t}_metrics.json" ] || m=\$((m+1))
done
echo "\$d \$m"
REMOTE
) || fail "cannot reach $HOST"
done_n=$(echo "$state" | awk '{print $1+0}'); missing=$(echo "$state" | awk '{print $2+0}')
echo "  $done_n / $EXPECT runs carry DONE; $missing of them lack a metrics file"
[ "$done_n" -ge "$EXPECT" ] || fail "only $done_n of $EXPECT finished — refusing to score a partial stage"
[ "$missing" -eq 0 ] || fail "$missing run(s) carry DONE but never wrote metrics"

echo "== syncing analysis to $HOST and scoring =="
rsync -aq analysis/ "$HOST:$REMOTE_EXP/analysis/" || fail "rsync analysis/"
remote <<REMOTE || fail "ft.py"
set -e
cd $REMOTE_EXP
python3 analysis/ft.py --runs $OUT/stage_ft --ceilings summary/ceilings_adopted.json \
    --xfer summary/matched_xfer.json -o summary/ft.json
REMOTE
scp -q "$HOST:$REMOTE_EXP/summary/ft.json" summary/ft.json || fail "scp ft.json"
echo "=== done ===  summary/ft.json"
