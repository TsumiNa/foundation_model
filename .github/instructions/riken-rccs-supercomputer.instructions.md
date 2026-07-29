---
description: "Use when running, training, or submitting jobs on the experimental RIKEN R-CCS Cloud supercomputer. Covers the mandatory rikyu-first policy, SSH access through riken-login, Slurm, heterogeneous CPU/GPU partitions, GPU queue preference and exclusions, the 24-hour walltime, modules, storage, and project environment constraints."
name: "RIKEN R-CCS Cloud Supercomputer Usage"
applyTo: "**"
---

# RIKEN R-CCS Cloud Supercomputer Usage

Reference for running this project on the experimental, heterogeneous **R-CCS Cloud** system.
Official manual: https://riken-rccs.github.io/ondemand_rccs_cloud/

The access, scheduler, modules, storage membership, and partition limits below were also checked on
the live system in 2026-07.

## Mandatory platform preference: use rikyu first

**Always prefer the `rikyu` supercomputer over R-CCS Cloud.** rikyu has abundant, homogeneous GB200
capacity; even jobs using dozens of GPUs often start without queueing and it should be treated almost
like a local resource.

Use R-CCS Cloud only when rikyu is unavailable, unsuitable for the hardware experiment, or the user
explicitly requests R-CCS Cloud. Do not move a routine job to R-CCS Cloud merely because its devices
are interesting: this system is an internal experimental facility with scarce and highly heterogeneous
hardware.

The separate rikyu instructions are in
`.github/instructions/rikyu-supercomputer.instructions.md`.

## Usage-policy boundaries

R-CCS Cloud is for small-scale performance evaluation and research/development. Its official policy
prohibits production use such as large-scale LLM training or large production simulations, continuous
services, and personal use.

- Do not run long or high-load work on a login node. Training, long compilation, Jupyter/VS Code
  servers, and continuous AI-assisted tools must run on compute nodes.
- Do not occupy resources for extended periods by repeatedly resubmitting jobs.
- Request only the nodes, GPUs, and time actually needed.
- There is no backup. Keep irreplaceable code and results elsewhere.
- The system is best-effort and resources or access may change without notice.

## Access & layout

- SSH: `ssh riken-login` (already configured).
- User: `u0001820`.
- Login host: `login1.cloud.r-ccs.riken.jp`.
- Home: `/home/users/u0001820`, physically mounted at `/hs/work0/home/users/u0001820`.
- No `foundation_model` clone was present when these instructions were written. Use separate clones
  for incompatible execution environments:
  - GH200: `/home/users/u0001820/projects/foundation_model-gh200`
  - H200/H100/A100: `/home/users/u0001820/projects/foundation_model-x86-cuda`
  - MI300A: `/home/users/u0001820/projects/foundation_model-mi300a`
- Group: `agis-fmms` (project ID `11036`).
- Shared group storage: `/lvs0/agis-fmms`.
- Scheduler: **Slurm 24.05** (`sbatch`, `srun`, `squeue`, `sinfo`, `sacct`, `scancel`).
- `uv`: `/home/users/u0001820/.local/bin/uv`.
- Login-node system Python is 3.9 and is too old for this project; use the project `.venv`.

The home quota is 1 TB. Although the manual's generic group-directory default is 5 TB and
10,000,000 files, `agis-fmms` was live-verified at **50 TB and 20,000,000 files** in 2026-07 and was
already close to both limits. Check quotas before a large run:

```bash
lfs quota -h -u "$(whoami)" /hs/work0
lfs quota -hp 11036 /lvs0
```

## Non-interactive SSH and modules

Initialize a login shell when inspecting modules remotely:

```bash
ssh riken-login 'bash -lc "module avail"'
```

Every batch script should initialize the system profile before loading the partition-specific modules.
If strict shell mode is used, **source the profile before enabling `nounset`**: `/etc/profile` reads
`HISTCONTROL` before defining it, so the reverse order exits immediately.

```bash
source /etc/profile
set -euo pipefail
module load system/<partition> <toolchain>
```

The login node exposes the `system/...` modulefiles but may not expose the toolchain modulefiles added
by them. Verify the complete module pair on a compute node, not by loading `nvhpc`/`rocm` on the
login node.

Do not run training directly through `ssh riken-login '<training command>'`; submit it through Slurm.

## Walltime and submission limits

The official operational policy and live Slurm configuration agree:

- Maximum reservation time per normal job: **24 hours** (`1-00:00:00`).
- `ai-h100l-pu` is stricter: **30 minutes**.
- Per-user submitted work (`nodes × requested time`) is limited to
  `nodes in the target partition × 72 hours`.
- Lower cumulative usage receives higher scheduling priority.

Use an explicit `#SBATCH --time=...` and never request more than the partition limit. For work that
cannot finish in 24 hours, split it into meaningful restartable stages with persistent checkpoints.
Do not create an automatic resubmit loop that effectively monopolizes a partition.

Slurm currently accepts an over-limit submission and assigns it a job ID, but leaves it pending with
`Reason=PartitionTimeLimit`; it does not clamp the requested time or fail `sbatch`. After submission,
check `squeue` or `scontrol show job "$JID"` rather than treating a returned job ID as validation.
Slurm also rounds a request containing seconds up to the next whole minute.

## GPU selection policy

Before submission, inspect current availability:

```bash
sinfo -p qc-gh200,ai-h200-brc,ai-h100l-pu,fs-mi300a,qc-a100 \
  -o "%18P %10l %6D %8t %N %G"
squeue -p qc-gh200,ai-h200-brc,ai-h100l-pu,fs-mi300a,qc-a100
```

For GPU work, use this priority order, considering both queue state and software compatibility:

1. `qc-gh200`
2. `ai-h200-brc`
3. `ai-h100l-pu` — public queue, but jobs are limited to **30 minutes**
4. `fs-mi300a`
5. `qc-a100`

### Forbidden partition

**Never use `ai-h100l`.** It is exclusive to the “High Performance Big Data Research Team” and the
“Data Management Platform Development Unit”. The similarly named `ai-h100l-pu` partition is public
and is the only H100L partition allowed here.

### Preferred GPU resources

| Priority | Partition | Nodes / accelerators per node | Node memory | Job limit | Modules |
|---:|---|---|---:|---:|---|
| 1 | `qc-gh200` | 8 documented; 1 NVIDIA GH200/node | 512 GB | 24 h | `system/qc-gh200 nvhpc` |
| 2 | `ai-h200-brc` | 1 node; 8 NVIDIA H200 | 1,536 GB | 24 h | `system/ai-h200-brc nvhpc` |
| 3 | `ai-h100l-pu` | 2 shared nodes; H100 NVL in MIG mode | 256 GB | 30 min | `system/ai-h100l nvhpc` |
| 4 | `fs-mi300a` | 1 node; 4 AMD MI300A | 512 GB | 24 h | `system/fs-mi300a rocm` |
| 5 | `qc-a100` | 2 nodes; 8 NVIDIA A100/node | 4,096 GB | 24 h | `system/qc-a100 nvhpc` |

Resource counts are experimental and can drift. The official resource page and live `sinfo` may
temporarily disagree while nodes are under maintenance; trust live Slurm for schedulable capacity.

For CPU-only work, choose any compatible available CPU partition. `genoa` is the ordinary x86_64
choice (`module load system/genoa mpi/openmpi-x86_64`); `fx700` is AArch64/A64FX and may require
cross-platform dependency handling.

### Queue observation record

The following deliberately tiny smoke jobs were submitted on 2026-07-29 JST to measure real queue
delay. These are point-in-time observations, not scheduling guarantees:

| Partition | Job ID | Submitted | Initial estimated start | Initial estimated wait |
|---|---:|---|---|---:|
| `qc-a100` | `262879` | 16:29:00 | 19:58:13 | 3:29:13 |
| `qc-gh200` | `262866` | 16:22:27 | 20:01:30 | 3:39:03 |
| `fs-mi300a` | `262865` | 16:22:27 | next day 10:30:57 | 18:08:30 |

Slurm job `262883` has an `afterany` dependency on all three tests and will write their final
`Submit`, `Eligible`, `Start`, `End`, state, requested resources, and allocated resources to:

```text
/home/users/u0001820/jobs/instructions-smoke/queue_wait_2026-07-29.psv
```

Replace the estimates above with actual `Start - Submit` durations after the recorder completes.

## GPU allocation syntax

- `qc-gh200`: select one node without `--gpus`. Slurm automatically adds `gres/gpu=1` to the request.
  `--gpus=1` is accepted too, but is redundant.
- `ai-h200-brc`: explicitly request `#SBATCH --gpus=<1-8>`. A one-GPU job was verified to set
  `SLURM_JOB_GPUS=1` and `CUDA_VISIBLE_DEVICES=0`.
- `ai-h100l-pu`: select one node **without** `--gpus`; `--gpus=1` is rejected. The public queue
  currently assigns one `NVIDIA H100 NVL MIG 1g.12gb` device to PyTorch (about 10.75 GiB usable),
  even though `nvidia-smi -L` lists the parent H100 and all six MIG instances.
- `fs-mi300a`: select one node without `--gpus`; `--gpus=1` is rejected.
- `qc-a100`: use `#SBATCH --gres=gpu:<1-8>`. Despite the manual's `--gpus=<n>` example,
  untyped `--gpus=1` is rejected by the live Slurm configuration for this partition.
- Never infer allocation from the parent GPU shown by `nvidia-smi`. Validate with PyTorch device
  count/name/memory as well as `SLURM_JOB_GPUS` and `CUDA_VISIBLE_DEVICES`; the latter two are unset
  on `ai-h100l-pu`.
- Lightning uses the devices exposed by Slurm. For multi-GPU training, set `[training].devices = -1`
  or the desired allocated count; do not request eight GPUs and then silently use one.

## Python, PyTorch, CUDA, and ROCm

The repository currently resolves Linux PyTorch from the **CUDA 13 (`cu130`)** index. This is the
normal environment for the NVIDIA partitions (`qc-gh200`, `ai-h200-brc`, `ai-h100l-pu`,
`qc-a100`) on both AArch64 and x86_64:

```bash
cd /home/users/u0001820/projects/<foundation_model-gh200-or-foundation_model-x86-cuda>
uv sync --frozen --all-groups
.venv/bin/python -c \
  'import platform, torch; print(platform.machine(), torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Treat these as three hard environment boundaries:

| Environment | ISA | Accelerator backend | Partitions |
|---|---|---|---|
| GH200 | AArch64 | NVIDIA CUDA | `qc-gh200` |
| x86 CUDA | x86_64 | NVIDIA CUDA | `ai-h200-brc`, `ai-h100l-pu`, `qc-a100` |
| MI300A | x86_64 | AMD ROCm | `fs-mi300a` |

**Never share or copy `.venv` across these boundaries.** An AArch64 venv copied to x86_64 was
verified to fail immediately with `Exec format error`. CUDA and ROCm environments are also
incompatible even when both hosts are x86_64. Keep a separate clone and `.venv` for each row rather
than repeatedly mutating one environment in place.

Run `uv sync --frozen --all-groups` separately in the GH200 clone and the x86-CUDA clone; do not
reuse the resulting `.venv`.

`fs-mi300a` is AMD/ROCm. The repository's default Linux lock selects CUDA PyTorch and **cannot be
used unchanged on MI300A**. Only select `fs-mi300a` after preparing and validating a dedicated
ROCm-compatible project configuration and lock in the MI300A clone. Never overwrite either CUDA
environment with an ad-hoc ROCm installation. Inside an MI300A allocation, verify
`torch.version.hip`, `torch.cuda.is_available()`, device count, device name, and device memory before
starting real work.

## Minimal NVIDIA batch templates

### GH200 (first choice)

```bash
#!/bin/bash
#SBATCH --job-name=fm-gh200
#SBATCH --partition=qc-gh200
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/u0001820/jobs/%x_%j.out
#SBATCH --error=/home/users/u0001820/jobs/%x_%j.err

source /etc/profile
set -euo pipefail
module load system/qc-gh200 nvhpc

PROJ=/home/users/u0001820/projects/foundation_model-gh200
cd "$PROJ"

.venv/bin/python -c \
  'import torch; print(torch.cuda.get_device_name(), torch.version.cuda, torch.cuda.is_available())'
.venv/bin/fm pretrain --config <toml> --output-dir <persistent-output-dir>
```

### H200 (second choice; also adapt for A100)

```bash
#!/bin/bash
#SBATCH --job-name=fm-h200
#SBATCH --partition=ai-h200-brc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/u0001820/jobs/%x_%j.out
#SBATCH --error=/home/users/u0001820/jobs/%x_%j.err

source /etc/profile
set -euo pipefail
module load system/ai-h200-brc nvhpc

PROJ=/home/users/u0001820/projects/foundation_model-x86-cuda
cd "$PROJ"

.venv/bin/python -c \
  'import torch; print(torch.cuda.get_device_name(), torch.version.cuda, torch.cuda.is_available())'
.venv/bin/fm pretrain --config <toml> --output-dir <persistent-output-dir>
```

For `qc-a100`, change the partition and module to `qc-a100` and `system/qc-a100 nvhpc`, and replace
`#SBATCH --gpus=1` with `#SBATCH --gres=gpu:1`.
For `ai-h100l-pu`, change the partition/module to `ai-h100l-pu` and `system/ai-h100l nvhpc`, remove
`--gpus`, and set `--time` to no more than `00:30:00`.

Create `/home/users/u0001820/jobs` before the first submission. Keep outputs and checkpoints under
home or `/lvs0/agis-fmms`; do not rely on node-local paths surviving job completion.

## Submitting and controlling jobs

```bash
JID=$(sbatch --parsable job.sbatch)
squeue -j "$JID"
scontrol show job "$JID"
sacct -j "$JID" -X --format=JobID,Partition,State,Elapsed,ExitCode,AllocTRES
scancel "$JID"
```

Interactive inspection must also request a compute node:

```bash
srun --nodes=1 --partition=qc-gh200 --time=00:30:00 --pty bash -l
```

After entering an interactive allocation, load the matching system/toolchain modules before running
GPU commands.

## Keeping local, rikyu, and R-CCS clones in sync

Share source and lock files through git; never copy `.venv` between systems or architectures.
Before a run, record the commit and resolved environment in the output directory. After changing
dependencies locally, commit `pyproject.toml` and `uv.lock`, pull the same commit remotely, and
recreate/sync the target-specific environment.
