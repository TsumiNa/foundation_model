---
name: hpc-training-env
description: Deploy the fm training environment on a Slurm supercomputer and run/babysit experiment fleets there (setup, data transfer, job patterns, monitoring, result recovery). Use when asked to set up a cluster environment or manage cluster jobs.
---

# Training environment on a Slurm cluster

Cluster-specific facts (login alias, partitions, module names, current account state) live in
`.github/instructions/rikyu-supercomputer.instructions.md` — read it first and **update it in
place** when the cluster changes (never delete it; it survives machine migrations).

## Environment setup (new machine / new account)

```bash
# 1. SSH alias in ~/.ssh/config (key auth), then:
git clone <repo> && cd <repo>
uv sync --frozen --all-groups          # exact locked env; NEVER bare `uv sync` (relocks)
.venv/bin/fm --help                    # entry point sanity
# 2. Data: parquets are NOT in git — rsync from a machine that has them:
rsync -az <src>:.../data/ data/        # then rerun any derived-data prep scripts
# 3. GPU smoke of the full chain AS A JOB (never CPU on the login node):
sbatch experiments/<exp>/jobs/smoke.sbatch   # pretrain→ws→ft→inverse at --sample 400
```

Data/results policy: **experiment results never travel through git** (only configs, job scripts,
analysis code are tracked). Share results between machines/people via rsync.

## Slurm job patterns (all learned the hard way)

- Job scripts: set partition/GPU/time in `#SBATCH`, `module load nvhpc` (or the cluster's CUDA
  module), absolute paths (`sbatch` from anywhere), env-var parameterisation
  (`--export=ALL,MODE=ws,ORD=0`), per-unit **skip-if-done markers** so every job is idempotent
  and resubmission is always safe. Print a final PASS/FAIL line.
- **`--dependency=afterok` semantics (measured)**: success releases dependents; a FAILED parent
  leaves them PENDING (DependencyNeverSatisfied) **forever** — recovery is
  `scancel <dependents>` + resubmit with a fresh dependency. Never wait on them.
- Independent work units (per-checkpoint evals, per-task controls) → **disjoint-range parallel
  jobs**, not one serial job. A 10–14 h serial plan became ~3 h wall this way.
- After code/schema changes while checkpoints are in flight: hold dependents
  (`scontrol hold`), run a small **GPU compat-probe job** (real environment, real checkpoint,
  tiny chain, PASS/FAIL), release only on PASS.
- Login-node `sbatch`/`sacct` fail intermittently (DNS) — every remote query needs a retry loop;
  never interpret a failed query as "no jobs".

## SSH from the local orchestration side

- `$var`/`$(...)` inside `ssh 'bash -lc "..."'` are expanded by the LOCAL shell — escape as
  `\$var`, or transfer scripts via `ssh 'cat > file' < local_file` and execute.
- fish shell: bare `===`-style separators are parse errors; quote them.
- Babysitting: background until-loops polling `sacct` (sleep 240–600 s), one waiter per job
  fleet; kill stale monitors when fleets change.

## Result recovery (continuous, not end-of-run)

- Periodic `rsync -az --partial` of finished outputs to the local mirror (every ~30 min during
  active fleets); exclude `*smoke*` always; exclude `*.pt`/`lightning/`/`logs/` for routine
  mirrors.
- Before any shutdown/migration deadline: full sweep **including checkpoints** (they are gone
  forever otherwise and are usually small — verify total size first), plus job logs; reconcile
  file counts on both sides; audit `artifacts/` for directories from older experiments.
