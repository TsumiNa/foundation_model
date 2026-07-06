---
name: fm-pretrain
description: Design and run fm pretrain experiments — continual multi-task pretraining with replay, constrained-random task orders, step checkpoints. Use when asked to set up, smoke-test, launch or babysit a pretraining run/sweep.
---

# Continual pretraining with `fm pretrain`

Reference configs: `experiments/rikyu_task_scaling/make_configs.py` (generator — edit the script,
never the generated TOML) and `experiments/rikyu_replay_sweep/configs/`. Entry point:
`.venv/bin/fm pretrain --config <toml> [--seed N] [--set key=value] [--output-dir D] [--sample N]
[--max-epochs N] [--resume] [--checkpoint ckpt.pt]`.

## Config anatomy

Shared sections (`[data] [descriptor] [datasets] [tasks] [model] [training]`) are derived from a
base config — keep them identical across compared runs. Then:

```toml
[pretrain]
task_sequence = ["formation_energy", ...]   # the tasks, in declaration order
n_runs = 1
task_order = "random"        # or "fixed"
task_order_seed = 7501       # run i shuffles with seed+i (reproducible replicates)
task_order_groups = [ [...], [...], [...] ]  # optional; must partition task_sequence:
                             # each group is shuffled internally, groups concatenate in order

[pretrain.replay]
interval = 1                 # replay every step
amount = 1000                # labels per seen task per step; per_task = {...} to override

[output]
dir = "artifacts/<experiment>/<run>"
```

## Facts that shape experiment design (measured on the 24-task set)

- **Replay n ≥ 1000 essentially resists forgetting**; returns diminish beyond (mean R²
  0.371@n100 → 0.600@n2500). Forgetting is event-driven (step collapses when specific
  interfering tasks arrive), not gradual.
- **Kernel-regression tasks dominate replay wall-time** — with random orders, pin them to a
  final group so early checkpoints stay cheap.
- **Pin tasks needed downstream to step 1** via `task_order_groups` (e.g. formation_energy when
  every checkpoint must support an inverse objective).
- After step 1 a task only sees replay-sized data again — full-data training happens once, at
  its own introduction. This drives representation effects downstream (see fm-eval-ws-ft).
- Expectation setting: on this task set, multi-task co-training gave NO net benefit over
  single-task training (interference); don't promise transfer.

## Outputs

`<out>/training/stepNN_<task>/checkpoint.pt` (one per introduction step — these feed ws/ft/
inverse), `training/final_model.pt`, `training/metrics_table.csv` (long format: step, task,
new_task, r2, mae, samples; the realized task order is recoverable from rows where
task == new_task — never hardcode orders in analysis).

## Workflow

1. **Smoke first, in the real environment**: `--sample 400 --max-epochs 1` as a GPU job on the
   cluster (never a CPU smoke on the login node — the user requires the real job environment).
2. Launch runs with per-run overrides: `--seed $((2025+i)) --set pretrain.task_order_seed=$((7501+i))`.
3. Long runs: `--resume` continues from the last completed step; job scripts must be
   idempotent (skip completed steps) so resubmission is always safe.
4. If code/schema changes while checkpoints are pending consumption, run a **GPU compat-probe
   job** (tiny ws→ft→inverse chain from a real old checkpoint, PASS/FAIL line) before mass jobs.
5. Cluster patterns: see the hpc-training-env skill.
