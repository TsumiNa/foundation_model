# `fm` configuration reference

Every `fm` subcommand is driven by a single **TOML** file. This document is the authoritative
schema: every section, every key, its type, default, constraint, and meaning. It is generated from
and kept in sync with the config dataclasses in `src/foundation_model/workflows/` (`task_catalog.py`,
`_sections.py`, `pretrain.py`, `finetune.py`, `inverse.py`, `predict.py`).

- Ready-to-copy templates live in [`samples/`](../samples/) (`*.toml` formal, `*_smoke.toml` fast).
- **Unknown keys are rejected** at load time with a message naming the offending key and the allowed
  set — so a typo fails fast rather than being silently ignored.
- Types follow TOML: strings must be quoted, floats need a decimal point (`10.0` is a float, `10`
  an int — this matters where a field is int-only), arrays are `[...]`, tables are `[section]`,
  arrays-of-tables are `[[section]]`.

## How a config is loaded

The CLI (`fm <cmd> --config x.toml`) does exactly this: parse the TOML → apply `--set` and
first-class-flag overrides onto the raw tree → build a validated `@dataclass` → write
`run_provenance.json` (the fully resolved config) → run. No values are hidden; the provenance file
records what actually ran.

Override precedence (later wins): **TOML file** → `--set SECTION.KEY=VALUE` → dedicated flags
(`--seed`, `--accelerator`, `--sample`, and per-command flags). `--checkpoint` / `--output-dir` are
passed to the builder directly and take precedence over `[…].checkpoint` / `[output].dir`.

## Which sections each subcommand reads

| Section | `pretrain` | `finetune` | `inverse` | `predict` |
|---|:--:|:--:|:--:|:--:|
| `[data]` | ✓ | ✓ | ✓ | ✓ |
| `[descriptor]` | ✓ | ✓ | ✓ | ✓ |
| `[datasets.*]` | ✓ | ✓ | ✓ | ✓ |
| `[[tasks]]` | ✓ | ✓ | ✓ | ✓ |
| `[model]` | ✓ | ✓ | ✓ | ✓ |
| `[training]` (+ sub-tables) | ✓ | ✓ | — | — |
| `[output]` | ✓ | ✓ | ✓ | ✓ |
| command section | `[pretrain]` | `[finetune]` | `[inverse]` | `[predict]` |

`inverse` and `predict` do **not** read `[training]`; their seed/accelerator live in `[inverse]` /
`[predict]` instead.

---

# Shared sections

## `[data]` — data loading + splitting

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `composition_column` | str | `"composition"` | | Column holding the composition string / formula key. |
| `val_split` | float | `0.1` | `[0, 1)` | Fraction held out for validation. |
| `test_split` | float | `0.1` | `[0, 1)`, `val+test < 1` | Fraction held out for test. |
| `split_random_seed` | int | `42` | | Seed for the random train/val/test split (when no `split` column is present). |
| `batch_size` | int | `256` | `>= 1` | Training/eval batch size. |
| `num_workers` | int | `0` | `>= 0` | DataLoader worker processes. |
| `persistent_workers` | bool | `false` | requires `num_workers >= 1` | Keep DataLoader workers alive across epochs (saves per-epoch worker startup). Incompatible with `pretrain.replay.resample = "epoch"` — persistent workers hold a one-time dataset copy and would never see the per-epoch replay redraws; the config builder rejects the combination. |
| `pin_memory` | bool | `true` | | Page-locked host memory for faster async host→GPU copies. Only has an effect on CUDA; harmless elsewhere. |
| `prefetch_factor` | int | unset | `>= 1`; requires `num_workers >= 1` | Batches prefetched per worker. Unset = torch default (2). |
| `multiprocessing_context` | str | unset | `"fork"` \| `"spawn"` \| `"forkserver"`; requires `num_workers >= 1` | Worker start method. Unset = platform default (fork on Linux, spawn on macOS). |

If a dataset file has a `split` column (`train`/`val`/`test`), it is honored directly and the
random split is not used.

## `[descriptor]` — how composition descriptors are produced

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `kind` | str | `"kmd"` | `kmd` \| `precomputed` | `kmd` = on-the-fly invertible KMD descriptor (required for the composition inverse-design path); `precomputed` = load a descriptor table. |
| `n_grids` | int | `8` | `>= 2` (kmd only) | KMD grid resolution; the descriptor width scales with it. |
| `path` | str (path) | — | required iff `kind = "precomputed"` | File of precomputed per-composition descriptors. |

## `[datasets.<name>]` — one composition-keyed data file (array of named tables)

Define one table per data file; `<name>` is the key each `[[tasks]]` references via its `dataset`
field. At least one dataset is required.

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `path` | str (path) | — | required; name must end in `.csv` / `.parquet` / `.pd` / `.pd.z` / `.pd.xz` / `.pkl` | The data file (composition-keyed rows). |
| `preprocessing_path` | str (path) | `None` | | Optional joblib object with a `"dropped_idx"` (the qc dataset drops rows). |
| `min_elements` | int | `None` | `>= 1` | Keep only compositions with at least this many elements. |
| `sample` | int | `None` | `>= 1` | Row cap (smoke runs); `--sample N` sets this for every dataset. |

## `[[tasks]]` — supervised tasks (array of tables)

One entry per prediction head. At least one is required; names must be unique.

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `name` | str | — | required, unique | Task/head name; used as the head key and in output filenames. |
| `kind` | str | — | `regression` \| `kernel_regression` \| `classification` | Head type. (Legacy aliases `reg`/`kr`/`clf` also accepted.) |
| `dataset` | str | — | must match a `[datasets.<name>]` | Which dataset supplies this task's column(s). |
| `column` | str | — | required | Target column. |
| `t_column` | str | `None` | required iff `kind = kernel_regression`; forbidden otherwise | The sequence x-axis column (e.g. energies for DOS, temperatures for ZT). |
| `num_classes` | int | `None` | required iff `kind = classification`, `>= 2`; forbidden otherwise | Number of classes. |
| `lr` | float | `None` | `> 0` | Per-task learning-rate override (else `[training]`'s LR for this head's kind). |
| `weight_decay` | float | `None` | `>= 0` | Per-task weight-decay override (else `[training]`'s weight decay for this head's kind). |
| `hidden_dims` | list[int] | `None` | positive ints; reg/clf only | Override `[model].head_hidden_dims` for this head. |
| `x_hidden_dims` | list[int] | `None` | positive ints; KR only | Override `[model].kr_x_hidden_dims` (value branch). |
| `t_hidden_dims` | list[int] | `None` | positive ints; KR only | Override `[model].kr_t_hidden_dims` (coordinate branch). |
| `n_kernel` | int | `None` | positive int; KR only | Override `[model].n_kernel` for this head. |

### `[[tasks]].scaler` — optional inverse-transform for reporting

A nested table on a task. Used only to inverse-transform predictions to human-readable units.

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `path` | str (path) | — | required | Fitted scaler (joblib). |
| `key` | str | `None` | | Key inside a dict-of-scalers pickle; `None` = the whole object is the scaler. |

## Python-layer fields vs TOML keys

`BaseTaskConfig` is what the model consumes; the config layer builds it from `[datasets.*]` +
`[[tasks]]`, and the names differ. When reading the source, this is the mapping:

| `BaseTaskConfig` field | TOML equivalent | Notes |
|---|---|---|
| `data_files` | `[datasets.<name>].path`, selected by `[[tasks]].dataset` | TOML groups tasks under named datasets instead of repeating a path per task. |
| `data_column` | `[[tasks]].column` | |
| `t_column` | `[[tasks]].t_column` | Kernel regression only. |
| `composition_column` | `[data].composition_column` | Global in TOML; the per-task override is Python-only. |
| `split_column` | — | Honoured inside the data file (a `split` column); not a TOML key. |
| `task_masking_ratio` | no direct key; set by the pretrain loop | Each step sets `1.0` for the newly introduced task and the ratio resolved from `[pretrain.replay].amount` / `.per_task` for every replaying task. Not settable per task for a scaling-law sweep — use `[datasets.<name>].sample`. |
| `predict_idx` | `[predict].split` / `[predict].compositions` | Set for every task at once by `fm predict`, not per task. |
| `optimizer` | `[training]` group settings + `[[tasks]].lr` / `.weight_decay` | See [`[training]`](#training--optimization-pretrain--finetune-only). |

## `[model]` — network architecture

The `*_hidden_dims` lists are the **interior hidden widths**; the input (descriptor width for the
encoder, `latent_dim` for the heads) is prepended and the output (`1` / `num_classes` / kernel
projection) appended. Example: `encoder_hidden_dims = [512, 256]` builds
`descriptor_dim → 512 → 256 → latent_dim`. Any `[[tasks]]` entry may override its own head (see
above).

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `latent_dim` | int | `128` | positive int | Encoder output width = every head's input width. |
| `encoder_hidden_dims` | list[int] | `[256]` | positive ints (may be empty) | Encoder hidden layers. Empty = a single `descriptor_dim → latent_dim` layer. |
| `head_hidden_dims` | list[int] | `[64]` | positive ints, non-empty | Default hidden layers for regression/classification heads. |
| `kr_x_hidden_dims` | list[int] | `[128, 64]` | positive ints, non-empty | Default KR value-branch hidden layers. |
| `kr_t_hidden_dims` | list[int] | `[16, 8]` | positive ints, non-empty | Default KR coordinate-branch hidden layers. |
| `n_kernel` | int | `15` | positive int | Default number of KR Gaussian kernel centers. |

## `[training]` — optimization (pretrain + finetune only)

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `max_epochs` | int | `100` | `>= 1` | Max epochs per training step. |
| `encoder_lr` | float | `0.005` | `> 0` | Shared-encoder learning rate. |
| `encoder_weight_decay` | float | `0.01` | `>= 0` | Shared-encoder weight decay. |
| `head_lr` | float | `0.005` | `> 0` | Regression/classification head learning rate. |
| `head_weight_decay` | float | `1e-05` | `>= 0` | Regression/classification head weight decay. |
| `kr_lr` | float | `0.0005` | `> 0` | Kernel-regression head learning rate. |
| `kr_weight_decay` | float | `5e-05` | `>= 0` | Kernel-regression head weight decay. |
| `ae_lr` | float | `0.005` | `> 0` | AutoEncoder head learning rate (the AE head always trains). |
| `ae_weight_decay` | float | `0.001` | `>= 0` | AutoEncoder head weight decay. |
| `learnable_loss_balancer` | bool | `false` | | Uncertainty weighting (Kendall/Gal/Cipolla, CVPR 2018): learn one log σ per supervised task and combine losses as Σᵢ [ 0.5·exp(−2 log σᵢ)·Lᵢ + log σᵢ ] instead of the static `[[tasks]].loss_weight`. Off by default — see the note below. |
| `accelerator` | str | `"auto"` | | Lightning accelerator (`auto` / `cpu` / `gpu` / …). |
| `devices` | int \| list[int] \| str | `"auto"` | **one device only** | Passed to Lightning `Trainer(devices=...)`. While distributed training is out (see the note below), only single-device forms are accepted: `1`, `"auto"`, `[0]`, `"0"`. `-1`, `2`, `[1, 3]`, `"1,3"` and `"0-3"` are rejected at config time, and a `"auto"` that Lightning resolves onto several GPUs is refused before the fit starts. |
| `seed` | int | `2025` | | Global seed (`--seed` overrides). |

> **One device.** Distributed training was removed with its output half never written —
> the sampler and metric side was built, but nothing guarded writes by rank, so every rank would
> concurrently overwrite the same checkpoint, metrics JSON and prediction parquet. There is a
> second reason it cannot simply be switched back on: the training logs no longer carry
> `sync_dist=True`, so each rank would hand `ReduceLROnPlateau` its own shard's
> `train_final_loss_epoch` and the learning rates would diverge across ranks even though the
> gradients are synchronised — a run that finishes, looks normal, and is wrong. Both halves are
> named in [ARCHITECTURE.md](../ARCHITECTURE.md)'s distributed-training section, which also records the
> measured reason to doubt DDP is the right axis for this project at all.

The model builds **one AdamW with one parameter group per role** — shared encoder, regression/
classification heads, kernel-regression heads, and the always-on autoencoder head — so each group
has its own learning rate and weight decay. The four defaults span three orders of magnitude
(`1e-2` encoder / `1e-3` AE / `5e-5` KR / `1e-5` reg+clf); they were call-site constants before
`0.3.0` and are configuration now.

> **Changed.** It used to be one AdamW *instance* per group. Lightning drives at most one optimizer
> automatically, so that shape forced `automatic_optimization = False`, and under manual
> optimization Lightning stops driving schedulers too — which is how stepping them became the
> model's job, and how the #45 per-batch bug got in. The collapse changes no learning rate: every
> group's scheduler was built from the one `[training.scheduler]` block below and monitored the
> same metric, so the N schedulers already decided identically. Per-group `lr` / `weight_decay` /
> `min_lr` survive as parameter groups and a list-valued `min_lr`.

A single `[[tasks]]` entry may override its own head with `lr` / `weight_decay`, which win over
these group defaults. Everything else about the optimizer comes from the two sub-tables below and
is shared by every group.

> **`learnable_loss_balancer` has never been switched on in any run.** The model has implemented
> uncertainty weighting since before `[training]` existed, but nothing routed a value to it, so it
> defaulted off and stayed there. Exposing the key does not turn it on; it makes the A/B runnable.
> Note when you do run it: the log σ parameters join the shared-encoder group, so they take
> `encoder_weight_decay` — AdamW's decoupled decay pulls each log σ toward 0 (σ = 1, i.e. the
> unweighted objective), which is a mild bias against the balancer that is worth being aware of
> when reading the comparison.

### `[training.optimizer]` → AdamW numerics (shared by every group)

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `betas` | list[float] | `[0.9, 0.999]` | two values in `[0, 1)` | AdamW running-average coefficients. |
| `eps` | float | `1e-06` | `> 0` | Added to the denominator for numerical stability. |

AdamW is the only optimizer. Earlier revisions carried `Adam` and `SGD` branches that no config
key could reach; they were removed in `0.3.0` rather than left as untested dead paths.

### `[training.scheduler]` → `ReduceLROnPlateau` (shared by every group)

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `enabled` | bool | `true` | | `false` = constant learning rate; no scheduler is constructed. |
| `mode` | str | `"min"` | `min` \| `max` | Whether a lower or higher monitored value is better. |
| `factor` | float | `0.5` | `(0, 1)` | Multiplier applied to the LR on plateau. |
| `patience` | int | `5` | `>= 0` | Epochs without improvement before reducing. |
| `min_lr` | float | `0.0001` | `>= 0`; `< lr` when `enabled = true` | **Floor** for the reduced LR — see the warning below. |
| `monitor` | str | `"train_final_loss_epoch"` | non-empty; must exist at epoch end | Metric the plateau is measured on. Must be logged with `on_epoch=True` during **training**; a missing key raises at the end of the first epoch rather than silently skipping the LR step. |

`ReduceLROnPlateau` is the only scheduler; `StepLR` and the `"None"` selector were removed in
`0.3.0`, the latter replaced by `enabled`.

There is **one** scheduler, over the single AdamW, and **Lightning** drives it: the model declares
`interval = "epoch"` and Lightning steps it once per epoch on the epoch-aggregated `monitor`
metric. The model does not touch it, so the cadence cannot drift back to per-batch. There is no
`interval` / `frequency` key here and no field behind one — a plateau scheduler stepped per batch
is the bug below, not a setting.

These five keys are the scheduling **decision**, which reads only the monitored metric, so they
apply to every parameter group; `lr` / `weight_decay` / `min_lr` remain per-group. A per-group
scheduler *policy* is therefore not expressible — it never was from this file, which has always
been one block feeding every group.

> **Changed in 0.3.1.** Schedulers previously stepped inside `training_step`, i.e. once per
> *batch*, which made `patience` count batches — on a 24k-row task at `batch_size = 256`
> (~90 batches/epoch) the LR reached `min_lr` inside the first epoch. `monitor` was ignored
> entirely, and its old default `train_total_loss` named a metric that does not exist. Runs before
> 0.3.1 annealed far faster than their config implies.

> **`min_lr` interacts with every learning rate.** It is a floor, so a low LR plus the default
> `1e-4` floor leaves almost no room to anneal: at `lr = 2e-4` the scheduler can halve once and
> then stops. `min_lr >= lr` is rejected at config time, because in that case the scheduler runs
> but can never change the LR — a no-op that is invisible in logs. When lowering a learning rate
> below ~`1e-3`, lower `min_lr` with it or set `enabled = false` deliberately.

### `[training.early_stopping]` → Lightning `EarlyStopping` (on by default)

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `enabled` | bool | `true` | | Turn early stopping on/off. |
| `monitor` | str | `"val_final_loss"` | | Metric to monitor. |
| `mode` | str | `"min"` | `min` \| `max` | Whether lower or higher is better. |
| `patience` | int | `8` | `>= 1` | Epochs without improvement before stopping. |
| `min_delta` | float | `0.0001` | | Minimum change counted as improvement. |

### `[training.checkpoint]` → Lightning `ModelCheckpoint` (opt-in)

Off by default — the run recorder already writes replay-schema checkpoints that
finetune/inverse/predict consume. Enable to *also* emit Lightning `.ckpt` files.

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `enabled` | bool | `false` | | Emit Lightning `.ckpt` files. |
| `monitor` | str | `"val_final_loss"` | | Metric to monitor. |
| `mode` | str | `"min"` | `min` \| `max` | Direction of improvement. |
| `save_top_k` | int | `1` | | How many best checkpoints to keep. |
| `save_last` | bool | `false` | | Also save the last-epoch checkpoint. |
| `filename` | str | `None` | | Optional Lightning filename template. |

### `[training.logging]` → Lightning loggers (opt-in)

| Key | Type | Default | Description |
|---|---|---|---|
| `csv` | bool | `false` | Write a `CSVLogger` metrics table under `<output.dir>/logs/`. |
| `tensorboard` | bool | `false` | Write a `TensorBoardLogger` under `<output.dir>/logs/`. |

## `[output]` — where the run writes

| Key | Type | Default | Description |
|---|---|---|---|
| `dir` | str (path) | — | Run output directory. Required unless `--output-dir` is passed (which overrides it). Other keys under `[output]` are ignored. |

---

# Command sections

## `[pretrain]` — replay-based continual pre-training

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `task_sequence` | list[str] | `[]` → `[[tasks]]` order | tasks must exist | Order tasks are introduced across steps. |
| `n_runs` | int | `1` | `>= 1` | Independent repeats (different seeds), written to `runs/runNN/`. |
| `task_order` | str | `"fixed"` | `fixed` \| `random` | `fixed` = `task_sequence` order; `random` = per-run shuffle (reproducible: run *i* shuffles with `numpy` seed `[training].seed + i`, or `task_order_seed + i` when set). |
| `task_order_seed` | int | `None` | requires `task_order = "random"` | Decouple the order shuffle from the training seed: run *i* (0-based) shuffles with `task_order_seed + i`, so the same orders can be replayed under different training seeds (and vice versa). `None` = derive from the run seed (`[training].seed + i`). |
| `task_order_groups` | list[list[str]] | `[]` | requires `task_order = "random"`; must exactly partition `task_sequence` | Constrained shuffle: tasks are shuffled within each group and the groups are concatenated in the listed order — e.g. keep expensive kernel-regression tasks in a final block while still randomizing within blocks. |
| `checkpoint` | str (path) | `None` | | **Warm-start**: load this checkpoint's encoder + heads as the starting point (`--checkpoint` overrides). Its tasks count as already-learned and are skipped as new steps (they still take part in replay + evaluation); training continues with the `task_sequence` tasks the checkpoint doesn't already contain. Errors if a checkpoint task isn't in the catalog, or if every `task_sequence` task is already in the checkpoint. |
| `resume` | bool | `false` | | **Kill-restart** (`--resume` sets it): on start, if a run's output dir already holds step checkpoints, warm-start from the latest and continue **in place** at the next task; a run whose `final_model.pt` exists is skipped. For long pre-training that can exceed a scheduler's job time — re-submit the same command and it picks up where it stopped. Resume granularity is one completed task-step; optimizer state is not restored (each step trains a fresh optimizer regardless). |

### `[pretrain.replay]`

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `interval` | int | `1` | `>= 1` | Already-learned tasks rejoin training every Nth step; `1` = always replay. |
| `amount` | float \| int | `0.05` | float in `(0,1)` or int `>= 1` | Replay amount per old task: a fraction of its labels, or an absolute label count. |
| `per_task` | table (str→num) | `{}` | keys must be tasks; same value rule | Override `amount` for named tasks, e.g. `per_task = { density = 0.2 }`. |
| `resample` | str | `"step"` | `"step"` \| `"epoch"` | `"step"`: one frozen replay subset per training step (historical behavior). `"epoch"`: redraw the subset (same size) at every epoch start, so over E epochs a step's replay coverage approaches `N·(1−(1−n/N)^E)` of the task's N labelled rows at unchanged per-epoch cost. Draws are deterministic per `([data].split_random_seed, task, epoch)` and independent across tasks. Incompatible with `persistent_workers=True` dataloaders (redraws would never reach the frozen worker copies) — the run fails fast with a clear error if one is detected. |

## `[finetune]` — frozen-encoder head fine-tuning

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `checkpoint` | str (path) | — | required (or `--checkpoint`) | Checkpoint to fine-tune from. |
| `tasks` | list[str] | — | required, non-empty | Heads to fine-tune; other heads stay frozen (the AE head always trains). |
| `epochs` | int | `20` | | Fine-tune epochs (distinct from `[training].max_epochs`). |
| `freeze_encoder` | bool | `true` | | Freeze the shared encoder + non-target heads (BatchNorm buffers included). |
| `add_new_tasks` | bool | `true` | | If a target task isn't in the checkpoint, add a fresh head for it. |

## `[predict]` — evaluate / predict with a checkpoint

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `checkpoint` | str (path) | — | required (or `--checkpoint`) | Checkpoint to load. |
| `tasks` | list[str] | `[]` → all heads | must exist in the checkpoint | Heads to predict; empty = every checkpoint head. |
| `split` | str | `"test"` | `train` \| `val` \| `test` \| `all` | Which split to predict on. |
| `compositions` | list[str] | `[]` | | Explicit compositions to predict; **overrides `split`** when given. |
| `with_metrics` | bool | `true` | | Compute metrics when true targets are available (`--no-metrics` disables). |
| `seed` | int | `2025` | | RNG seed (`--seed` overrides). |
| `accelerator` | str | `"auto"` | `auto` \| `cpu` | Device: `auto` uses CUDA if available, else CPU (`--accelerator` overrides). |

## `[inverse]` — inverse design (scenarios × algorithm paths)

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `checkpoint` | str (path) | — | required (or `--checkpoint`) | Trained checkpoint to inverse-design from. |
| `steps` | int | `300` | | Gradient-optimization steps (`--steps` overrides). |
| `lr` | float | `0.05` | | Optimizer learning rate. |
| `record_trajectory` | bool | `true` | | Record + emit optimization trajectories (`--no-trajectory` disables). |
| `per_seed_trajectories` | bool | `false` | | Also emit per-seed trajectory plots (capped at 20). |
| `animation_formats` | list[str] | `["gif"]` | ⊆ `{gif, html, svg}` | Trajectory animation formats (`--animation-formats` overrides). |
| `seed` | int | `2025` | | Global RNG seed (`--seed` overrides). |
| `accelerator` | str | `"auto"` | `auto` \| `cpu` | Device (`--accelerator` overrides). |

`[inverse]` also contains one `[inverse.seeds]` table and the `[[inverse.scenarios]]` /
`[[inverse.paths]]` arrays below. If `[[inverse.paths]]` is omitted, a built-in set of **11 default
paths** (3 latent + 8 composition) is used.

### `[inverse.seeds]` — how starting compositions are chosen

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `strategy` | str | `"top_objective"` | `top_objective` \| `weighted_random` \| `random` \| `explicit` | Seed-selection algorithm. `top_objective` ranks the candidate pool (the target tasks' rows in the chosen split) by the scenario's objective score — the exact weighted loss the optimizers minimise — and takes the best (lowest) `n`. `weighted_random` samples without replacement with probability proportional to the rank of `weight_task`'s true label (higher label = more likely). |
| `weight_task` | str | `None` | required iff `strategy = "weighted_random"`; must be a regression task | Task whose TRUE labels weight the sampling. |
| `weight_direction` | str | `"high"` | `high` \| `low`; `weighted_random` only; exclusive with `weight_value` | `high`: higher label = more likely; `low`: lower label = more likely. |
| `weight_value` | float | `None` | `weighted_random` only; exclusive with `weight_direction` | Sample by closeness: labels nearer this value are more likely. |
| `n` | int | `20` | `>= 1` | Total seed compositions to return. |
| `split` | str | `"test"` | `train`/`val`/`test`/`all` | Split to draw candidates from. |
| `explicit` | list[str] | `[]` | required (non-empty) when `strategy = explicit` | Explicit candidate pool. |
| `explicit_append` | list[str] | `[]` | each must have a computable descriptor | Extra seeds appended to every strategy's output. |
| `dedup_by_element_system` | bool | `true` | | Keep only one composition per element system (set of element symbols). |

### `[[inverse.scenarios]]` — design objectives (array of tables)

At least one required; names unique. Each scenario is `name` + a non-empty
`[[inverse.scenarios.targets]]` array; every task named in a target must have a head in the
checkpoint.

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `name` | str | — | required, unique | Scenario identifier (→ output subdir). |
| `targets` | array of tables | — | required, non-empty; one entry per task, no duplicates | The objective terms (below). |

### `[[inverse.scenarios.targets]]` — one objective term (array of tables)

The target kind derives from the task's `[[tasks]].kind`; the allowed keys are kind-conditional:

| Key | Type | Applies to | Description |
|---|---|---|---|
| `task` | str | all | Task name (must exist in the catalog + checkpoint). |
| `value` | float | regression | Steer the prediction toward this value (MSE). Exactly one of `value`/`direction` per regression target. |
| `direction` | str (`"high"` \| `"low"`) | regression, classification | Regression: push the prediction up/down with **no fixed goal** — the objective is unbounded, so the achieved magnitude scales with `steps × lr`; use `weight` to balance it against the bounded terms. Classification: push `P(classes)` up (default) or down. |
| `points` | list of `[t, y]` pairs | kernel_regression | Target curve: the head is evaluated at the given `t` values and pulled toward the given `y` values (MSE over the points). `t` outside the head's trained range extrapolates. |
| `classes` | list[int] | classification | Label indices whose combined probability is steered. Must be a **strict subset** of the head's classes (the full set makes the objective constant/undefined). |
| `weight` | float | all | `> 0`, default `1.0`. Scales this term against the scenario's other targets. |

```toml
[[inverse.scenarios]]
name = "fe_down_diel_up"

[[inverse.scenarios.targets]]
task = "formation_energy"   # regression, value mode
value = -1.0

[[inverse.scenarios.targets]]
task = "dielectric_total"   # regression, direction-only mode
direction = "high"
weight = 2.0

[[inverse.scenarios.targets]]
task = "dos_density"        # kernel_regression: target curve
points = [[-2.0, 0.5], [0.0, 1.2], [2.0, 0.8]]

[[inverse.scenarios.targets]]
task = "material_type"      # classification (optional — no default class objective)
classes = [1, 3]
direction = "low"
```

### `[[inverse.paths]]` — algorithm variants (array of tables)

Each path is one optimization recipe. `method` selects the family; the other keys are validated
against it (a composition-only key on a `latent` path — or `ae_align_scale` on a `composition`
path — is rejected).

| Key | Type | Default | Applies to | Description |
|---|---|---|---|---|
| `name` | str | — | both | Path identifier (→ output filenames). |
| `method` | str | — | both | `latent` (optimize the latent vector) or `composition` (optimize the recipe). |
| `ae_align_scale` | float | `0.5` | latent | AE-manifold alignment weight (sweet spot ≈ 0.5). |
| `init` | str | `"seed"` | composition | `seed` (blend seed weights) or `random` (random starts). |
| `n_starts` | int | `None` | composition (`init=random`) | Number of random starts. |
| `seed_blend` | float | `0.95` | composition (`init=seed`) | Fraction of the seed kept; the rest is uniform over the whitelist. |
| `allowed_elements` | list[str] \| `"all"` | `"all"` | composition | Hard element whitelist. |
| `diversity_scale` | float | `1.0` | composition | Per-output element-diversity penalty (`1.0` = none). |
| `max_elements` | int | `None` | composition | Cardinality cap: at most K elements per recipe. |
| `element_step_scale` | float \| table | `1.0` | composition | Per-element gradient scaling (`0` hard-locks an element to its seed value). |
| `fixed_amounts` | table (str→float) | `{}` | composition | Pin specific elements to absolute amounts, e.g. `{ Au = 0.65 }`. |
| `annealing_scale` | float | `0.5` | composition | Softness of the K-hot annealing schedule. |
| `annealing_schedule` | table | `None` | composition | Advanced override of the annealing schedule. |

For the design intent behind each knob, see
[docs/inverse_design_algorithms.md](inverse_design_algorithms.md).

---

# CLI flags

Every subcommand shares these (from `common_options`):

| Flag | Applies to | Description |
|---|---|---|
| `--config PATH` | all | The TOML config file (**required**). |
| `--output-dir DIR` | all | Override `[output].dir`. |
| `--set SECTION.KEY=VALUE` | all | Override one TOML value (repeatable); VALUE is TOML syntax, so quote strings: `--set 'data.composition_column="formula"'`. |
| `--seed N` | all | Override the run seed (routed to the right section per subcommand). |
| `--accelerator X` | all | Override the accelerator (`auto` / `cpu`). |
| `--sample N` | all | Cap rows for every `[datasets.*]` (fast smoke runs). |

Per-subcommand flags:

| Subcommand | Flags |
|---|---|
| `pretrain` | `--max-epochs N` (→ `training.max_epochs`), `--checkpoint PATH` (warm-start / continue a sequence), `--resume` (continue after a kill from the output dir's latest step checkpoint) |
| `finetune` | `--checkpoint PATH`, `--tasks a,b` (→ `finetune.tasks`), `--epochs N` |
| `inverse` | `--checkpoint PATH`, `--scenario NAME` (repeatable; run only these), `--steps N`, `--no-trajectory`, `--animation-formats gif,html,svg` |
| `predict` | `--checkpoint PATH`, `--tasks a,b`, `--split train\|val\|test\|all`, `--compositions "Fe2 O3,Al2 O3"` (overrides split), `--no-metrics` |

Run `fm <subcommand> --help` for the exact list.

---

# Minimal examples

Pre-train two regression heads + one classifier, then fine-tune and predict:

```toml
# pretrain.toml
[descriptor]
kind = "kmd"
n_grids = 12

[datasets.qc]
path = "data/qc.parquet"

[[tasks]]
name = "density"
kind = "regression"
dataset = "qc"
column = "density"

[[tasks]]
name = "material_type"
kind = "classification"
dataset = "qc"
column = "mtype"
num_classes = 5

[model]
latent_dim = 128
encoder_hidden_dims = [256]

[training]
max_epochs = 100
[training.early_stopping]
patience = 8
[training.logging]
csv = true

[pretrain]
task_sequence = ["density", "material_type"]
[pretrain.replay]
interval = 1
amount = 0.05

[output]
dir = "artifacts/pretrain"
```

```bash
fm pretrain --config pretrain.toml
fm finetune --config finetune.toml --checkpoint artifacts/pretrain/runs/run00/training/final_model.pt
fm predict  --config predict.toml  --checkpoint artifacts/finetune/training/final_model.pt --split test
```

See [`samples/`](../samples/) for complete formal + smoke configs of all four subcommands.
