#!/bin/bash
# Submit the task-scaling production jobs (run AFTER jobs/smoke.sbatch has printed SMOKE_PASS):
#   3x pretrain (parallel) -> each triggers its own ws + ft eval job via afterok -> + 1 scratch.
# If a pretrain job fails, its dependents go DependencyNeverSatisfied: resubmit the pretrain
# (all workers are skip-if-done + --resume idempotent), then resubmit the two eval jobs.
set -euo pipefail
PROJ=/home/ea0094/projects/foundation_model
JOBS=$PROJ/experiments/rikyu_task_scaling/jobs
N=${1:-1000}   # replay branch: bash submit_all.sh [1000|1500]
TAG=""; [ "$N" != "1000" ] && TAG="_n${N}"
mkdir -p /home/ea0094/jobs/task_scaling

for ORD in 0 1 2; do
    pre=$(sbatch --parsable --job-name=ts_pre${TAG}_o${ORD} --export=ALL,ORD=$ORD,N=$N "$JOBS/pretrain_scaling.sbatch")
    ws=$(sbatch --parsable --dependency=afterok:$pre --job-name=ts_ws${TAG}_o${ORD} --export=ALL,ORD=$ORD,N=$N "$JOBS/ws_scaling.sbatch")
    ft=$(sbatch --parsable --dependency=afterok:$pre --job-name=ts_ft${TAG}_o${ORD} --export=ALL,ORD=$ORD,N=$N "$JOBS/ft_scaling.sbatch")
    echo "ord$ORD: pre=$pre ws=$ws ft=$ft"
done
sc=$(sbatch --parsable --job-name=ts_scratch${TAG} --export=ALL,N=$N "$JOBS/scratch_baseline.sbatch")
echo "scratch=$sc"
