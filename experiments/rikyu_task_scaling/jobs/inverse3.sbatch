#!/bin/bash
#SBATCH --partition=1n1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/home/ea0094/jobs/task_scaling/%x_%j.out
#SBATCH --error=/home/ea0094/jobs/task_scaling/%x_%j.err

# Rerun inverse design with the corrected 3-scenario objective (formation_energy down paired
# with EACH dielectric target up separately) on every existing eval model of one
# (MODE, ORD, N) slice. Writes to <model>/inverse3/ — the original joint-scenario results under
# <model>/inverse/ are kept. Skip-if-done per k; missing models (e.g. ws tail still running)
# are skipped with a warning and picked up by a later resubmission.
set -uo pipefail
PROJ=/home/ea0094/projects/foundation_model
CFG=$PROJ/experiments/rikyu_task_scaling/configs
FM=$PROJ/.venv/bin/fm
: "${MODE:?need MODE=ws|ft}"
: "${ORD:?need ORD=0|1|2}"
N=${N:-1000}
DIRTAG=""; [ "$N" != "1000" ] && DIRTAG="_n${N}"

module load nvhpc
cd "$PROJ"
echo "== $(hostname) $(date) inverse3 mode=$MODE ord=$ORD n=$N =="
nvidia-smi --query-gpu=name --format=csv,noheader

rc=0
for k in $(seq 1 21); do
    kk=$(printf "%02d" "$k")
    MODEL=$PROJ/artifacts/task_scaling/${MODE}${DIRTAG}_o${ORD}/k${kk}/training/final_model.pt
    if [ ! -f "$MODEL" ]; then
        echo "!! k=$kk model missing — skipping (resubmit later)"
        rc=1
        continue
    fi
    OUT=$PROJ/artifacts/task_scaling/${MODE}${DIRTAG}_o${ORD}/k${kk}/inverse3
    t0=$SECONDS
    if [ -f "$OUT/inverse_design.json" ]; then
        echo "-- k=$kk already done, skip"
        continue
    fi
    "$FM" inverse --config "$CFG/inv.toml" --checkpoint "$MODEL" --output-dir "$OUT" 2>&1
    if [ $? -ne 0 ]; then
        echo "!! k=$kk inverse3 FAILED"
        rc=1
        continue
    fi
    echo "-- k=$kk done in $((SECONDS - t0))s"
done
echo "== inverse3 $MODE o$ORD n=$N exit=$rc =="
date
exit $rc
