#!/bin/bash
# Campaign status at a glance: queue, per-stage completion, timing, and the image every run used.
#
#   scripts/status.sh            # all stages
#   scripts/status.sh stage0     # one stage
#
# Run it on RIKYU. Over ssh it needs a login shell for Slurm to find its controller:
#   ssh rikyu-login 'bash -lc "projects/foundation_model_v2/.../status.sh"'

set -uo pipefail
OUTBASE=${OUTBASE:-/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning_v2}
STAGES=${*:-smoke stage0 stage_a stage_b stage_c}

echo "=== queue ==="
squeue -u "$USER" -o "%.10i %.12j %.2t %.11M %.11L %R" 2>/dev/null | head -20
running=$(squeue -h -u "$USER" -t R 2>/dev/null | wc -l)
pending=$(squeue -h -u "$USER" -t PD 2>/dev/null | wc -l)
echo "  running=$running pending=$pending"

for s in $STAGES; do
    D="$OUTBASE/$s"
    [ -d "$D" ] || continue
    total=$(find "$D" -maxdepth 1 -mindepth 1 -type d | wc -l)
    done_n=$(find "$D" -maxdepth 2 -name DONE | wc -l)
    echo
    echo "=== $s : $done_n/$total runs carry a DONE marker ==="

    if [ -r "$D/_timing.tsv" ]; then
        awk -F'\t' '
            $3 == 0 { n++; t += $2; if ($2 > mx) mx = $2; if (mn == 0 || $2 < mn) mn = $2 }
            $3 != 0 { bad++ }
            END {
                if (n) printf "  ok=%d  mean=%.2fh  min=%.2fh  max=%.2fh  total=%.1f GPU-h\n",
                              n, t/n/3600, mn/3600, mx/3600, t/3600
                if (bad) printf "  FAILED runs recorded: %d\n", bad
            }' "$D/_timing.tsv"
    fi

    # Which image did these runs actually use? The campaign's central risk is a stage having
    # quietly run on 0.2.1's per-batch scheduler cadence, so this is asserted, not assumed.
    vers=$(find "$D" -maxdepth 2 -name ENV.json -exec sed -n 's/.*"fm_version": "\([^"]*\)".*/\1/p' {} \; 2>/dev/null | sort | uniq -c)
    if [ -n "$vers" ]; then
        echo "  fm versions in this stage:"
        echo "$vers" | sed 's/^/    /'
        if [ "$(echo "$vers" | wc -l)" -gt 1 ]; then
            echo "    ^^ MIXED IMAGES IN ONE STAGE — these runs are not comparable."
        fi
    fi

    # In-flight progress: how far through its task sequence each unfinished run is.
    inflight=$(find "$D" -maxdepth 1 -mindepth 1 -type d ! -exec test -e '{}/DONE' \; -print 2>/dev/null | head -4)
    if [ -n "$inflight" ]; then
        echo "  in flight (steps completed):"
        for r in $inflight; do
            n=$(find "$r/training" -maxdepth 1 -name 'step*' -type d 2>/dev/null | wc -l)
            last=$(find "$r/training" -maxdepth 1 -name 'step*' -type d 2>/dev/null | sort | tail -1)
            echo "    $(basename "$r"): $n steps, latest=$(basename "${last:-none}")"
        done
    fi
done
