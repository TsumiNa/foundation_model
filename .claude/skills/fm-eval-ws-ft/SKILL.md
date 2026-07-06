---
name: fm-eval-ws-ft
description: Evaluate pretrained checkpoints on new target tasks via warm-start (full-model) and frozen-encoder finetune. Use when asked to measure transfer/scaling from checkpoints, add target tasks to a model, or run scratch baselines.
---

# Checkpoint evaluation: warm-start vs finetune

Two complementary probes per checkpoint — always report both, they answer different questions:

| mode | command | what it measures |
|---|---|---|
| **ws** (warm-start) | `fm pretrain --config ws.toml --checkpoint <ckpt>` with `task_sequence = [targets]` + replay | end-to-end value of the checkpoint as an *initialisation* (whole model retrains) |
| **ft** (finetune) | `fm finetune --config ft.toml --checkpoint <ckpt>` with `freeze_encoder = true` | *representation quality* — the frozen encoder cannot hide damage |
| scratch baseline | the ws config **without** `--checkpoint` (3 seeds) | the k=0 reference band |

Reference configs: `experiments/rikyu_task_scaling/make_configs.py` (`ws_n*.toml`, `ft.toml`).

## Config essentials

- **ft**: `freeze_encoder = true`, `add_new_tasks = true`, `epochs = 100` **with early stopping**
  — the default 20 epochs undertrains and confounds "better pretraining" with "longer training".
  `freeze_encoder = false` = full retrain (that variant is the warm-restart control protocol:
  chained fresh-optimizer retrains isolate optimizer effects).
- Train multiple target tasks **jointly** (one run, all targets in the sequence) unless the user
  explicitly wants per-target runs.
- Warm-start carries over the checkpoint's old heads (logged as `warm-started ... with heads
  [...]`); new target heads are added (`Adding new head '<task>'`). Watch for missing-key
  warnings — there should be none.

## Outputs

ws: standard pretrain outputs (`metrics_table.csv`, final_model.pt — the model used by inverse).
ft: `training/finetune_summary.json` with `metrics_before` / `metrics_after` per task.
Collector: `experiments/rikyu_task_scaling/analysis/collect.py` → `scaling.csv`;
curves: `analysis/plot_scaling.py`.

## Execution & scale

- Evaluations from different checkpoints are **independent — parallelize** (disjoint k-range
  jobs with per-k skip-if-done; never one serial job when a deadline exists).
- ~10× wall-time asymmetry: ws (replay + full model) ≫ ft. Budget accordingly.
- After any code/schema change mid-experiment: GPU compat-probe before the fleet (see fm-pretrain).

## Interpretation guide (validated on the dielectric targets)

- ws ≈ scratch everywhere is the *expected* null result — full retraining repairs representation
  damage, so ws differences are small (±0.03).
- ft is the sensitive probe: dips after k=1 are real (dominant-task dilution to replay size +
  interference from small unrelated tasks; dip depth tracks *which* task arrived).
- A ws edge at small k = initialisation prior from the first (large, fundamental) task; erodes
  as weakly-related tasks accumulate.
- Compare against the **single-task baseline** (task trained alone on full data), not only
  at-intro levels — at-intro is depressed by multi-task interference.
