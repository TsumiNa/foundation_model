---
name: fm-pretrain
description: Continual pretraining with fm pretrain — the default hybrid-replay recipe, config anatomy, replay facts, outputs and machine-independent workflow. Use when asked to pretrain, continue a task sequence, or design a pretraining experiment.
---

# Continual pretraining with `fm pretrain`

This skill is **machine-independent**. Where to execute (local / remote / container) is decided
per the protocol in the hpc-training-env skill; folder layout for a new experiment follows the
experiment-layout skill. Entry point: `.venv/bin/fm pretrain --config <toml> [--seed N]
[--set key=value] [--output-dir D] [--sample N] [--max-epochs N] [--resume] [--checkpoint ckpt.pt]`.

## DEFAULT training recipe (validated 2026-08, replay_epoch_sweep campaign)

Two steps — continual pretraining with hybrid replay, then one full-data consolidation:

```toml
[training]
max_epochs = 250                # m250 default; steps saturate near ~100-150, cap is headroom

[training.early_stopping]
patience = 20                   # resampling delays the val plateau — do not lower this

[pretrain.replay]
interval = 1
resample = "epoch"              # redraw each old task's replay subset every epoch — never hurt at any budget
amount = 0.30                   # hybrid rule: amount_t = max(1500, 0.3 * N_t)
# floor: EVERY task with 0.3*N < 1500 (i.e. N < 5000) gets per_task = 1500 (engine clamps to N,
# so small tasks are auto-full-coverage). Recompute this list when the task set changes.
per_task = { <each task with N < 5000> = 1500 }
```

```bash
fm pretrain --config <exp>/configs/pretrain.toml --output-dir <out> --resume   # idempotent
fm finetune --config <exp>/configs/consolidate.toml \
  --checkpoint <out>/training/final_model.pt --epochs 250 --output-dir <out>_joint
```

The consolidation config = same catalog + `[finetune]` with all task heads listed,
`freeze_encoder = false`, masking 1.0 (reference: `experiments/replay_epoch_sweep/configs/
joint_retrain_full24.toml`). From a healthy replay run it early-stops far below the cap
(measured: 76 epochs) and mainly benefits big tasks. `fm finetune` has **no --resume** — give
it walltime once.

Why these choices (all measured; see experiments/replay_epoch_sweep/HYBRID_RECIPE.md):
- epoch resampling: +0.022…+0.126 mean R² over frozen subsets at every budget; zero regressions.
- hybrid amounts: fixed-count starves big tasks (deficit plateau ~0.045 → 0.025–0.031 hybrid);
  pure ratio starves small tasks (r0.5: 0.085 vs ≤0.01 at full coverage). max(floor, r·N) is the
  only setting that wins every size group.
- generous patience/cap: with resampling, early stopping is the binding constraint (at patience 8
  every step stopped early; at 24, 100% hit the epoch cap).
- consolidation: replay DURING training is mandatory (skip it and even a converged end retrain
  lands 0.055–0.08 below every replay arm); the end pass is cheap polish on top (+0.006 mean,
  big-task deficit 0.031 → 0.022).

## Config anatomy

Shared sections (`[data] [descriptor] [datasets] [tasks] [model] [training]`) derive from a base
config — keep them identical across compared runs. `[pretrain]`: `task_sequence`, `n_runs`,
`task_order = "fixed"|"random"` (+ `task_order_seed`, `task_order_groups` — groups must
partition the sequence; pin expensive kernel-regression tasks to a final block, and pin tasks
needed downstream to step 1). `[output] dir`. Overrides at launch via `--set` keep canonical
configs untouched (`--set 'pretrain.replay.resample="epoch"'` etc.).

## Facts that shape experiment design (measured on the 24-task set)

- Replay subsets draw from independent RNG streams `(seed, task, epoch)`; `resample = "epoch"`
  is incompatible with `data.persistent_workers = true` (validated at config build).
- Without replay the backbone collapses within ~5 steps (only the newest task survives;
  forgotten tasks go deeply negative R², median −23 by step 24). Never run `interval > 1`
  sequences on code older than PR #36 (kernel-regression heads crashed on replay-free steps).
- Kernel-regression tasks dominate replay wall-time; late steps are the expensive ones.
  Measured wall (24 tasks, 1×H200): hybrid recipe ≈ 22 h; consolidation ≈ 1.5 h.
- After step 1 a task only sees replay-sized data again — full-data training happens once, at
  its own introduction; the consolidation pass is what closes the remaining big-task gap.
- Multi-task co-training gave NO net transfer benefit on this task set; don't promise transfer.
- material_type (classification) is insensitive to replay settings (±0.005) — not a tuning target.

## Outputs

`<out>/training/stepNN_<task>/checkpoint.pt` (per introduction step — these feed ws/ft/inverse),
`training/final_model.pt`, `training/metrics_table.csv` (long format; realized task order =
rows where task == new_task). Consolidation adds `training/finetune_summary.json`
(metrics_before/after, epochs_run). CAVEAT: a `--resume`d pretrain writes a PARTIAL
metrics_table (upstream issue) — rebuild from the authoritative per-step
`stepNN_*/<task>_metrics.json` (see experiments/replay_epoch_sweep/analysis/
rebuild_metrics_from_stepdirs.py).

## Workflow (machine-independent)

1. Lay out the experiment folder per the experiment-layout skill; configs + scripts land there.
2. Pick the execution target and PREFLIGHT it (hpc-training-env skill) — never skip the check.
3. **Smoke first, in the real execution environment**: `--sample 400 --max-epochs 1` through the
   full intended chain (pretrain → consolidate → downstream) before any long run.
4. Long runs are idempotent: launch with `--resume`, recover walltime kills by resubmitting the
   same command; per-run overrides via `--seed $((2025+i)) --set pretrain.task_order_seed=...`.
5. If code/schema changes while checkpoints are pending consumption, run a compat-probe
   (tiny ws→ft→inverse chain from a real old checkpoint) before mass jobs.
