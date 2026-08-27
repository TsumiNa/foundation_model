---
description: "Use when running, training, or submitting jobs on the RIKEN R-CCS RIKYU supercomputer (Early Access Phase 2, NVIDIA GB200 NVL4). Covers Phase 2 SSH access, the unified Slurm gpu partition, supported GPU/node sizes, project charging, modules, CUDA-13 PyTorch, persistent and local storage, and the project .venv conventions."
name: "RIKYU Supercomputer Usage"
applyTo: "**"
---

# RIKYU Supercomputer Usage (RIKEN R-CCS Early Access Phase 2)

Reference for running this project on **RIKYU**. Official Phase 2 manual:
https://docs.r-ccs.riken.jp/rikyu/en/

Early Access Phase 2 is scheduled to continue through the end of September 2026. Phase 1 job
scripts and environment variables are not compatible with Phase 2; the official manual recommends
recreating job scripts rather than carrying them over.

## Access and current project layout

- Log in only through the local alias: `ssh rikyu-login`.
- Keep the host name, user name, identity file, and other credentials in the local `~/.ssh/config`;
  never add them to this repository.
- Phase 2 login node verified as `c000`, Ubuntu, AArch64.
- Home is available as `$HOME` on Lustre.
- The conventional repo clone is `$HOME/projects/foundation_model`.
- The conventional job/log directory is `$HOME/jobs`.
- Scheduler: Slurm.

Do not use the old Phase 1 hosts `login01.ai.r-ccs.riken.jp` or
`ar08n01-m.ai.r-ccs.riken.jp`.

## Login-node and non-interactive SSH rules

Do not run training, benchmarks, long builds, or other heavy computation on a login node. The Phase
2 login node exposes GB200 devices, so `nvidia-smi` and even `torch.cuda.is_available()` may succeed
there; this does **not** authorize computation or prove that a Slurm job has allocated a GPU.

Phase 2 sets `SLURM_CONF_SERVER` through shell initialization. A non-login remote command can fail
to find the Slurm controller:

```bash
ssh rikyu-login 'sinfo'                    # may fail
ssh rikyu-login 'bash -lc "sinfo"'         # correct
ssh rikyu-login 'bash -lc "module avail"'  # correct
```

Wrap all non-interactive remote Slurm and module commands in `bash -lc`.

## Project/account readiness check

Jobs are charged to a Phase 2 project account. If a user belongs to multiple projects, select the
project with `--account=PROJECT_NAME` / `-A PROJECT_NAME`.

Before the first submission, verify the Slurm association and group storage:

```bash
sacctmgr show user "$USER" withassoc
id
ls -ld /data1/rkp*
```

Do not record a personal or project account in the repository. Supply it when submitting, for
example `sbatch --account="$RIKYU_ACCOUNT" job.sbatch`. Use `$RIKYU_GROUP_DIR` for the assigned
group-storage path when a job needs `/data1`.

## System and job resources

RIKYU consists of 400 NVIDIA GB200 NVL4 compute nodes (1,600 GPUs total). Each node has:

- 2 Grace CPUs and 4 B200 GPUs.
- 960 GiB CPU memory.
- 173.2 GiB HBM3e per GPU.
- InfiniBand XDR (`800 Gbps × 4`).
- One 7.68 TB local NVMe SSD.

Slurm uses one default partition named `gpu`. Check the live schedulable capacity rather than
hard-coding a node count:

```bash
sinfo
scontrol show partition gpu
```

Supported job sizes are:

| GPUs | Nodes | Maximum CPU cores/node | Maximum memory/node | Maximum walltime |
|---:|---:|---:|---:|---:|
| 1 | 1 | 36 | 400 GB | 96 h |
| 2 | 1 | 72 | 800 GB | 96 h |
| 3 | 1 | 108 | 1,200 GB | 96 h |
| 4 | 1 | 144 | 1,600 GB | 96 h |
| 8 | 2 | 144 | 1,600 GB | 96 h |
| 12 | 3 | 144 | 1,600 GB | 96 h |
| 16 | 4 | 144 | 1,600 GB | 96 h |

**The CPU column above is the manual's figure, not what the scheduler enforces.** A
`--cpus-per-task` above **32 per GPU** is rejected outright, so size CPU requests from 32 per GPU
(128 on a four-GPU node) rather than from the table:

```
[AI4S] Requested CPUs (64 cpus-per-task x 1 tasks = 64) exceed the per-GPU cap 32 (= 1 GPU x 32)
```

The table is left as the manual states it; the cap is what was observed. Values between 32 and the
table's 36 were not tested — the rejection message names 32 explicitly.

The live partition default time is 12 hours and the maximum is 4 days. Specify `--time` explicitly.
For more than 4 GPUs, Phase 2 allocates whole four-GPU nodes, so the GPU count must be a multiple of
4. A five-GPU request was verified to be rejected by Slurm.

The memory figures are estimates combining usable CPU and GPU memory. CPU and GPU memory have
different performance characteristics even though GB200 connects them coherently through NVLink-C2C.

## Measure GPU utilisation before assuming one run per GPU

**A GB200 is far larger than this project's models. One run per GPU wastes most of the card, and
nothing will tell you.** The jobs complete, exit 0, drop their markers and log nothing unusual;
Slurm does not warn about an idle GPU. It has to be checked deliberately.

Measured on this repository's training workload — an MLP encoder at `batch_size = 256` — across
394 grid runs, plus v1's campaign for comparison:

| campaign / stage | `gres/gpuutil` | device memory used |
|---|---:|---:|
| v2 stage A' grid (394 runs) | 8–10%, mode **9%** | 1.29 GB of 189 GB |
| v1 stage A probe (66 runs) | 6–8% | 998 MB |
| v1 stage C, 24 tasks, ~20 h | **7%** | — |

A run is also allocated 18 CPUs and 400 GB of host RAM and uses about **one core** and 17.6 GB. So
the reservation is correct — one GPU per run, never more — and the reserved GPU idles through
roughly nine tenths of it.

Read it from Slurm's own accounting rather than a sampled `nvidia-smi`:

```bash
ssh rikyu-login 'bash -lc "sacct -j <jobid> --format=JobID,TRESUsageInAve -n -P"'
# ... gres/gpumem=1290M,gres/gpuutil=9,mem=17639M ...
```

### Pack independent runs onto one GPU

Where a workload cannot saturate a card, run several independent processes on it. They are separate
processes with separate seeds and no shared state, so **packing changes throughput and nothing else
about the numbers.** Measured against the same grid points run one-per-GPU:

| | wall clock per run | runs per GPU | throughput |
|---|---:|---:|---:|
| unpacked | 2.19 h | 1 | 1× |
| `PACK=4` | 2.28 h | 4 | **3.8×** |
| `PACK=8` | 2.46 h | 8 | **7.1×** |

Twelve percent slower per run for seven times the throughput. At `PACK=8` every other dimension
still has room: 10 GB of 189 GB device memory, 141 GB of the 400 GB host allocation, and 8 of the
32 permitted CPUs.

Do **not** quote a "production" ratio computed as run-hours ÷ pack size. That divides by the pack
size by construction and returns it unchanged, so it measures nothing — it will report exactly
`PACK` however the runs actually behaved. The calibration above is the measurement.

The shape is one array task holding one GPU and running `PACK` grid points concurrently, each with
its own output directory and completion marker; the array shrinks to `ceil(N / PACK)`:

```bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32          # the enforced per-GPU cap; ~4 cores per co-tenant at PACK=8
PACK=${PACK:-8}
START=$((SLURM_ARRAY_TASK_ID * PACK))
for k in $(seq 0 $((PACK - 1))); do
    run_one $((START + k)) &        # own output dir, own DONE marker, own log
done
wait                                 # fail the task if any co-tenant failed
```

`run_one` reads its grid line, skips if its DONE marker exists, and runs one `fm` invocation. Submit
with `--array=0-$(((N + PACK - 1) / PACK - 1))%<throttle>`.

Three things to get right:

- **Calibrate, do not extrapolate.** Re-run points that already completed unpacked, into a scratch
  output root, so the speedup is measured against a known baseline for the *same* configurations.
  Contention is not linear; `PACK=16` was never measured here and host RAM binds first.
- **Wall-clock comparisons do not cross the boundary.** Packed runs contend, so a timing figure
  from a packed stage cannot be placed beside one from an unpacked stage. Accuracy is unaffected.
- **Do not pack a workload that already fills the card.** Check `gres/gpuutil` first. This is a
  remedy for small models on large GPUs, not a general speed-up.

## Usage fees

Early Access Phase 2 compute jobs cost **300 JPY per GPU-hour**, plus consumption tax, billed to the
selected project after Phase 2. Estimate cost before submission:

```text
estimated cost (JPY, before tax) = requested GPUs × requested/actual hours × 300
```

Examples:

- 1 GPU for 1 hour: 300 JPY.
- 4 GPUs for 5 hours: 6,000 JPY.
- 16 GPUs for 24 hours: 115,200 JPY.

Use the RIKYU Portal to inspect project usage:
https://portal.rikyu.r-ccs.riken.jp/en/

## Modules

Phase 2 provides NVIDIA HPC SDK module families:

- `nvhpc`
- `nvhpc-nompi`
- `nvhpc-hpcx`
- `nvhpc-hpcx-cuda13`
- `nvhpc-byo-compiler`

Use:

```bash
module purge
module avail
module load nvhpc
module list
```

The manual documents 26.3. The live system also exposed 26.5 when checked in 2026-07; unversioned
`module load nvhpc` selected 26.3 at that time. Pin a version only when compiler/runtime
reproducibility requires it, and always record `module list` in run provenance.

`module load nvhpc` configures compilers, CUDA, MPI, NCCL, and related development libraries. It
does **not** allocate a GPU; GPU allocation comes from Slurm `--gpus=N`.

## Spack-provided software

System-provided scientific applications are managed through a public Spack 1.2.0 instance:

```bash
. /shared/software/spack-1.2.0/share/spack/setup-env.sh
spack find -x
spack find -lx
spack load <package>
```

Include the setup line inside a batch job when using Spack software. For MPI-enabled Spack
applications, confirm the package's MPI build and normally launch it with `srun`. Do not carry over
the Phase 1 UCX/NCCL environment-variable block unless a Phase 2-specific test proves it is needed.

## Python, uv, and isolated project environment

- Phase 2 system Python was 3.12.3 when checked.
- A user-installed `uv` is conventionally available as `$HOME/.local/bin/uv`.
- The project venv is conventionally `$HOME/projects/foundation_model/.venv`.
- The existing project venv was verified as Python 3.13.14 with
  `torch==2.12.1+cu130`, `torch.version.cuda == "13.0"`, and AArch64 wheels.
- Never copy this `.venv` to an x86_64 machine.

Use the project environment explicitly:

```bash
cd "$HOME/projects/foundation_model"
uv sync --frozen --all-groups
.venv/bin/python --version
.venv/bin/python -c \
  'import platform, torch; print(platform.machine(), torch.__version__, torch.version.cuda)'
```

Programs under `.venv/bin` are not automatically added to `PATH`. In batch jobs, use absolute paths
or activate the venv explicitly. Absolute paths are preferred for reproducibility.

The repository config selects CUDA-13 PyTorch on Linux:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

RIKYU's system `containers/image` configuration routes `ghcr.io` through an internal mirror. If the
mirror does not contain this repository, use `scripts/rikyu_pull_container.sh`; it temporarily
bypasses the mirror without overwriting an existing user registry configuration.

## Storage

Phase 2 has three storage areas:

| Area | Path | Default capacity | Filesystem | Lifetime/access |
|---|---|---:|---|---|
| Home | `/home/USER` | 50 GB/user | Lustre, SSD-backed | Persistent; owner only |
| Group | `/data1/GROUP` | 1 TB/group | Lustre, HDD-backed | Persistent; group members |
| Scratch | `/tmp` | 1.5 TB/requested GPU | XFS on local NVMe | Deleted when the job ends |

Use home for configuration, code, and small persistent files. Use `/data1/rkpNNNNN` for large
persistent project data after the Phase 2 group is assigned. Check quotas with the official
project-ID form:

```bash
lfs quota -h -p "$(lfs project -d "$HOME" | awk '{print $1}')" /home
lfs quota -h -p "$(lfs project -d /data1/GROUP | awk '{print $1}')" /data1
```

The home quota command currently needs re-validation after the Phase 2 project/account association
is fixed; the live account returned project ID `0`.

## Local scratch staging

The old Phase 1 `USER_SCRATCH_DIR=/scratch/job-<jobid>` convention is removed. Phase 2 scratch is
the job node's `/tmp`.

Files in `/tmp` are deleted at job completion. Copy outputs to `/home` or `/data1/GROUP` before the
job exits. For multi-node jobs, `/tmp` is local to each node and is not shared.

Example single-node pattern:

```bash
SCRATCH="/tmp/$USER/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"

# Stage inputs into "$SCRATCH", run there, then persist results:
cp -a "$SCRATCH/output/." /data1/GROUP/results/
```

The exact job-time scratch quota, cleanup behavior, and environment variables are pending a Phase 2
smoke test after the Slurm account association is fixed.

## Minimal Phase 2 batch template (1 GPU)

This template follows the official Phase 2 resource model.

```bash
#!/bin/bash
#SBATCH --job-name=fm-job
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

module purge
module load nvhpc

PROJ=${PROJ:-$HOME/projects/foundation_model}
VENV_PY="$PROJ/.venv/bin/python"
PERSISTENT_OUT="$PROJ/artifacts/<run-name>"

cd "$PROJ"

"$VENV_PY" -c \
  'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(), torch.version.cuda)'
"$VENV_PY" -m foundation_model.cli.main pretrain \
  --config <toml> \
  --output-dir "$PERSISTENT_OUT"
```

The module is loaded for a consistent development/runtime environment, not to request the GPU.

## Submitting and controlling jobs

Run these on a login shell:

```bash
JID=$(sbatch --parsable job.sbatch)
squeue -j "$JID"
scontrol show job "$JID"
sacct -j "$JID" -X --format=JobID,Account,State,Elapsed,ExitCode,AllocTRES
scancel "$JID"
```

Interactive examples:

```bash
salloc --gpus=1 --time=00:10:00
srun hostname

# Or directly:
srun --gpus=1 --time=00:10:00 --pty bash
```

Add `--account=PROJECT_NAME` when required.

## Lightning GPU selection

The `fm` CLI drives a Lightning `Trainer`; `[training].accelerator` and `[training].devices` select
among GPUs that Slurm has allocated.

For a single-node job using 1–4 GPUs:

```toml
[training]
accelerator = "auto"  # or "gpu"
devices = -1          # use all GPUs allocated to this job
```

Lightning can launch single-node DDP itself; do not wrap the `fm` command in `mpirun`. The current
`fm` CLI does not expose `num_nodes`, so do not request 8/12/16 GPUs for `fm` training until a
Phase 2 multi-node launch path is implemented and tested.

`fm predict` and `fm inverse` remain single-device workflows.

## Long runs and resume

The maximum walltime is 96 hours. Store checkpoints and output on persistent storage, never solely
under `/tmp`.

`fm pretrain --resume` can continue from the latest completed task-step by reusing the same
`--output-dir`. A step interrupted mid-fit restarts from the previous completed step checkpoint;
optimizer state is not restored because each task-step starts a new optimizer.

Because Phase 2 is billed per GPU-hour, do not introduce an unbounded self-resubmission loop.
Estimate the cost, bound the number of submissions, and check completion and exit status after each
job.

## Phase 2 execution validation

Run short one-GPU smoke jobs and record:

- Slurm project account and group-storage path, supplied outside the repository.
- `SLURM_JOB_GPUS`, `CUDA_VISIBLE_DEVICES`, and other changed Phase 2 job variables.
- PyTorch device count, name, memory, and CUDA version.
- Actual `/tmp` quota for one GPU and cleanup after job completion.
- `module load nvhpc` versus the standalone cu130 PyTorch environment.
- Single-node Lightning runs with 1, 2, and 4 GPUs.
- `sacct` accounting and billed GPU-hours.

Do not reintroduce Phase 1 environment variables or partition names while these tests are pending.

## Keeping macOS and RIKYU in sync

`pyproject.toml` and `uv.lock` are shared through git (`origin = TsumiNa/foundation_model`).
Commit on one clone, pull the same commit on the other, then run `uv sync --frozen --all-groups`.
Never copy `.venv` between macOS and RIKYU.
