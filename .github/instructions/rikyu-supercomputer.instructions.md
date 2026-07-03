---
description: "Use when running, training, or submitting jobs on the RIKEN R-CCS rikyu supercomputer (AI4S early-access, NVIDIA GB200). Covers SSH access, Slurm sbatch on the 1n1gpu partition, the nvhpc module, the CUDA-13 (cu130) PyTorch setup, local scratch staging, and the .venv PATH conventions."
name: "rikyu Supercomputer Usage"
applyTo: "**"
---

# rikyu Supercomputer Usage (RIKEN R-CCS AI4S early-access)

Reference for running this project on **rikyu**. Official docs:
https://riken-rccs.github.io/ai4s_early_access/ja/ — the whole guide is a single page.
All facts below were verified on the live system (2026-07).

## Access & layout

- SSH: `ssh rikyu-login` (already configured; user `ea0094`, login node `ar08n01-m.ai.r-ccs.riken.jp`, aarch64).
- Repo clone on rikyu: `/home/ea0094/projects/foundation_model` (home is Lustre `/work/hps0`).
- Job assets & example scripts live in `/home/ea0094/jobs/` (see `jobs/smoke/smoke_test.sbatch`, `jobs/smoke/gpu_verify.sbatch`).
- Scheduler: **Slurm** (`/usr/bin/{sbatch,squeue,sinfo,scancel,salloc,srun}`).
- GPU: **NVIDIA GB200** (Grace-Blackwell, compute capability `sm_100`, ~186 GB HBM, driver 595.45.04, max CUDA 13.2).

## Non-interactive SSH gotcha (modules)

`module` is a shell function loaded by the login profile. A bare `ssh rikyu-login 'module ...'`
will NOT find it. Always wrap remote commands in a **login shell**:

```bash
ssh rikyu-login 'bash -lc "module load nvhpc && module list"'
```

When authoring a remote script, do **not** pipe complex heredocs with `(` `)` through `ssh '...'`
(the outer shell mis-parses them). Write the file locally, then transfer it:

```bash
ssh rikyu-login 'cat > /home/ea0094/jobs/smoke/job.sbatch' < ./job.sbatch
```

## Partitions

| Partition   | Nodes | GPU/node | Cores/node | Mem/node | Max walltime |
|-------------|-------|----------|------------|----------|--------------|
| `1n1gpu` *  | 1     | 1        | 36         | 400 GB   | 4 days       |
| `1n2gpu`    | 1     | 2        | 72         | 800 GB   | 4 days       |
| `1n4gpu`    | 1     | 4        | 144        | 1600 GB  | 4 days       |
| `2n4gpu`    | 2     | 4        | 144        | 1600 GB  | 4 days       |
| `4n4gpu`    | 4     | 4        | 144        | 1600 GB  | 4 days       |
| `4n4gpu-p`  | 4     | 4        | 144        | ∞        | ∞            |

`1n1gpu` is the default and our main partition. Check availability: `sinfo -p 1n1gpu`.
Note: on `1n1gpu`, `nvidia-smi` may show another user's process on the same physical GB200 —
GPU memory is large but do not assume the card is exclusively idle; check before large allocations.

## Long pre-training past the walltime (`fm pretrain --resume`)

Partitions cap at **4 days**. A `fm pretrain` run whose `task_sequence` won't finish in one job can
be re-submitted with **`--resume`**: it warm-starts from the output dir's latest step checkpoint and
continues at the next task in place (finished runs are skipped). Use the **same** `--output-dir` and
put it on persistent storage (not per-job `/scratch`, which is deleted at job end). A resubmit-until-
done pattern: make the sbatch script re-queue itself while `final_model.pt` is absent, e.g.

```bash
OUT=/home/ea0094/projects/foundation_model/artifacts/pretrain_big
.venv/bin/fm pretrain --config pretrain.toml --output-dir "$OUT" --resume
test -f "$OUT/training/final_model.pt" || sbatch "$0"   # (n_runs=1; adjust the path for sweeps)
```

Resume is per completed task-step (a step killed mid-fit restarts from the previous step's
checkpoint); optimizer state is not restored, which is fine since each step trains a fresh optimizer.

## Modules

```bash
module avail                 # nvhpc/26.3 (default), nvhpc-nompi, nvhpc-hpcx, nvhpc-hpcx-cuda13,
                             # nvhpc-byo-compiler, cuda/11.8, cuda/13.2
module load nvhpc            # our default; provides CUDA/compilers/MPI toolchain
```

**Always load a module in every GPU job** — without one the GPU is not allocated. The documented
batch example always includes `module load nvhpc`, so treat it as mandatory: put `module load nvhpc`
in every job script. It does not conflict with the cu130 torch wheels used here (PyTorch ships its
own CUDA runtime), and it also provides the toolchain for MPI/NCCL and compiled CUDA.

## Python / uv / .venv — PATH conventions

- System Python is 3.9 (too old — never use it). Use the project venv only.
- `uv` is at `~/.local/bin/uv` (on PATH); the project venv Python is 3.13 (`.venv/bin/python`).
- **Programs under `.venv/bin` are not on PATH inside a batch job.** Use one of:
  1. **Absolute path** (preferred in job scripts): `/home/ea0094/projects/foundation_model/.venv/bin/python`
     (the `fm` console script carries an absolute-path shebang, so it also works standalone).
  2. **Symlink into a PATH dir**: `~/.local/bin` is on PATH; symlink the `fm` command there.
     Recreate with:
     ```bash
     ln -sfn /home/ea0094/projects/foundation_model/.venv/bin/fm ~/.local/bin/fm
     ```
- Sync deps: `cd /home/ea0094/projects/foundation_model && uv sync` (set `UV_HTTP_TIMEOUT=300`
  for the large CUDA wheels).

## PyTorch: CUDA 13 on Linux, default on macOS

rikyu needs a **CUDA-13** torch (`cu130`); a plain PyPI aarch64 wheel is **CPU-only** and will
silently fail to use the GPU (`torch.cuda.is_available() == False`). This is configured in
`pyproject.toml` so it is platform-split and does not affect local macOS resolution:

```toml
# dependencies:  "torch>=2.9.1, <3.0"   # cu130 aarch64 wheels start at 2.9.1

[tool.uv.sources]
torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

Result: Linux → `torch==2.x+cu130` (cu13 runtime, GB200 `sm_100`); macOS → `torch==2.x` (MPS/CPU).
Quick GPU check inside a job: `torch.version.cuda` should be `13.0` and `torch.cuda.is_available()` `True`.

## Local scratch (fast I/O)

Each job gets a per-job NVMe scratch dir, auto-deleted at job end (measured ~6 GB/s):

- `USER_SCRATCH_DIR` = `/scratch/job-<jobid>` (set only inside a job; empty on the login node).
- `SLURM_SUBMIT_DIR` = the directory you submitted from (persistent Lustre).
- Pattern: stage inputs into `$USER_SCRATCH_DIR`, write outputs there, then copy results back
  to persistent storage before the job exits:
  ```bash
  cp -r "$USER_SCRATCH_DIR/output" "$SLURM_SUBMIT_DIR/results/"
  ```

## Submitting & controlling jobs

```bash
JID=$(sbatch --parsable job.sbatch)   # capture the job id
squeue -j "$JID"                       # or: squeue -u $USER
scancel "$JID"                         # cancel
sacct -j "$JID" -X --format=JobID,State,Elapsed,ExitCode   # after completion
```

## Minimal sbatch template (1 GPU)

```bash
#!/bin/bash
#SBATCH --job-name=fm-job
#SBATCH --partition=1n1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/ea0094/jobs/%x_%j.out
#SBATCH --error=/home/ea0094/jobs/%x_%j.err
set -uo pipefail

PROJ=/home/ea0094/projects/foundation_model
VENV_PY="$PROJ/.venv/bin/python"

module load nvhpc            # mandatory: GPU is not allocated without a loaded module
cd "$PROJ"

# ... optionally stage data into "$USER_SCRATCH_DIR" ...

"$VENV_PY" -m foundation_model.cli.main pretrain --config <toml> --output-dir <persistent-dir>
# or, with the symlink on PATH:  fm pretrain --config <toml> --output-dir <persistent-dir>

# ... copy results from "$USER_SCRATCH_DIR" back to "$SLURM_SUBMIT_DIR" ...
```

## GPU selection (`[training].accelerator` / `devices`)

The `fm` CLI drives a Lightning `Trainer`, so the GPU is chosen by the `[training]` config, not by
env vars. Both default to **`"auto"`**, so on any GPU job (`--gpus-per-node>=1` + `module load nvhpc`)
`fm pretrain` / `fm finetune` use the allocated GPU(s) automatically — no config change needed on
`1n1gpu`. To pin devices on a multi-GPU partition (`1n2gpu` / `1n4gpu`), set `[training].devices`:

```toml
[training]
accelerator = "auto"   # or "gpu"
devices = -1            # all allocated GPUs (single-node DDP); or an int count, or e.g. [0, 1]
```

Lightning spawns single-node DDP itself (NCCL) — you do **not** launch it under `srun`/`mpirun` or
set the MPI env below for single-node multi-GPU. `fm predict` / `fm inverse` are single-device
(`accelerator = "auto" | "cpu"`, no `devices`).

## Multi-node MPI env (generic; the `fm` CLI is single-node)

The `fm` CLI does not expose `num_nodes`, so it targets a **single node**. These env vars are for
hand-rolled MPI/NCCL jobs on the multi-node partitions (`2n4gpu` / `4n4gpu`), per the rikyu docs —
not needed for the single-node DDP above:

```bash
export OMPI_MCA_pml=ucx
export UCX_CUDA_COPY_DMABUF=no
export UCX_MAX_RNDV_RAILS=4
export NCCL_DMABUF_ENABLE=0
export NCCL_NET_GDR_LEVEL=SYS
export UCX_PROTO_ENABLE=n
# one process per GPU:
export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
```

## Keeping macOS ↔ rikyu in sync

`pyproject.toml` / `uv.lock` are shared via git (`origin` = `TsumiNa/foundation_model`).
Commit on one clone, `git pull` on the other, then `uv sync` on each. `.venv` is git-ignored.
