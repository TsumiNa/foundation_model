---
name: hpc-training-env
description: Decide WHERE a workload runs (local/remote/container selection protocol + mandatory preflight), deploy the fm environment on machines and Slurm clusters, and run/babysit experiment fleets (setup, data transfer, job patterns, monitoring, result recovery). Use when picking an execution target, setting up an environment, or managing cluster jobs.
---

# Execution targets & the training environment

This skill owns everything machine-specific — the workflow skills (fm-pretrain, fm-eval-ws-ft,
fm-inverse) stay machine-independent and defer here. Machine-specific facts (login alias,
partitions, module names, quotas, account state) live in `.github/instructions/
{rikyu,riken-rccs,ism-gpu-a100}-*.instructions.md` — read the relevant one first and **update
it in place** when a machine changes (never delete; they survive migrations).

## Choosing where to run (protocol)

1. An **explicit user instruction** for this task wins.
2. Else follow the **preference order in AGENTS.md / .github/instructions** (RIKYU is the
   preferred compute platform when available).
3. Else **default to the LOCAL machine**.

- On **RIKYU and RIKEN R-CCS, prefer the project container** over a bare uv env: the GHCR image
  built by `.github/workflows/rikyu-container.yml` (tags `ghcr.io/<owner>/<repo>:rikyu-*`);
  fall back to `uv sync --frozen` only where the container runtime is unavailable.
- **Preflight EVERY chosen target before committing work — local included.** Minimum checks:
  reachable; repo at the intended commit; env resolves (container pulls / `uv sync --frozen`
  exits clean); GPU visible (`python -c "import torch; print(torch.cuda.is_available())"` or MPS
  locally); required `data/` files present; disk headroom; for Slurm also partition state
  (`sinfo`) and quota headroom. If the env is new or changed, run the standard `--sample 400`
  smoke of the full intended chain as a real job before any long run.

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
