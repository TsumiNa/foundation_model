---
description: 'Use when running neural-network training jobs on the ism-gpu-a100 remote A100 machine. Covers SSH access, remote working directory, GPU checks, long-running process handling, logs, and artifact locations without project-specific training optimizations.'
applyTo: '**'
---

# ism-gpu-a100 Neural Network Training

Use this when a model-training job should run on the remote A100 machine instead of the local Mac.

## Machine

- Connect with `ssh -o ClearAllForwardings=yes ism-gpu-a100`.
- Host alias `ism-gpu-a100` resolves to `megalith3`.
- Hardware: 128 CPU cores and 4x A100-40GB GPUs.
- Repository workdir for this project: `/data/claude/geo_pfn_20260710/repo`.
- `uv` is available at `~/.local/bin/uv`.

## SSH Rules

- Prefer `ssh -o ClearAllForwardings=yes ism-gpu-a100` for scripted commands. The user's SSH config may define a `LocalForward` on port 8025; stale interactive sessions can make normal `ssh ism-gpu-a100` print bind errors or exit 255.
- Do not rely on piped stdout from remote background launches through the gateway. Write remote output to log files, then inspect those files with `cat`, `tail`, or `grep`.
- For long jobs, start an interactive remote session and run the job inside `tmux` or an equivalent persistent session so training survives local disconnects.

## Before Launching

1. SSH to the machine:

    ```bash
    ssh -o ClearAllForwardings=yes ism-gpu-a100
    ```

2. Enter the repository:

    ```bash
    cd /data/claude/geo_pfn_20260710/repo
    ```

3. Check GPU availability before starting:

    ```bash
    nvidia-smi
    ```

4. Confirm the branch, commit, and environment state you intend to train from:

    ```bash
    git status --short --branch
    ~/.local/bin/uv sync
    ```

## Launch Pattern

Run training with the project's normal CLI, explicitly selecting CUDA when the CLI supports it. Keep command-specific hyperparameters in the project script or experiment notes, not in this infrastructure instruction.

```bash
mkdir -p logs checkpoints
tmux new -s train-<name>
~/.local/bin/uv run python -m <training_module> <training_args> --device cuda \
  --out checkpoints/<name>.pt > logs/<name>.log 2>&1
```

Detach from `tmux` with `Ctrl-b d`. Reattach later with:

```bash
tmux attach -t train-<name>
```

If the training entry point does not use `--out`, still write checkpoints under `checkpoints/` or another clearly named experiment directory, and log stdout/stderr under `logs/`.

## Monitoring

Use separate remote commands or another `tmux` pane:

```bash
nvidia-smi
tail -f logs/<name>.log
```

Record the checkpoint path, log path, command, branch, and commit hash in the experiment notes or result artifact so the run can be reproduced.
