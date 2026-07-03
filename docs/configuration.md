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
| `lr` | float | `None` | | Per-task learning-rate override (else the section LR for its kind). |
| `replay` | float \| int | `None` | float in `(0,1)` or int `>= 1` | Per-task rehearsal amount (pretrain); see `[pretrain.rehearsal]`. |
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
| `encoder_lr` | float | `0.005` | | Shared-encoder learning rate. |
| `head_lr` | float | `0.005` | | Regression/classification head learning rate. |
| `kr_lr` | float | `0.0005` | | Kernel-regression head learning rate. |
| `kr_weight_decay` | float | `5e-05` | | Weight decay for KR heads (reg/clf heads use a fixed `1e-5`). |
| `ae_lr` | float | `0.005` | | AutoEncoder head learning rate (the AE head always trains). |
| `accelerator` | str | `"auto"` | | Lightning accelerator (`auto` / `cpu` / `gpu` / …). |
| `devices` | int \| list[int] \| str | `"auto"` | | Passed to Lightning `Trainer(devices=...)`: `"auto"` (all devices for the accelerator), an int count (`-1` = all), a list of device indices (`[1, 3]`), or a string (`"1,3"` / `"0-3"`). |
| `seed` | int | `2025` | | Global seed (`--seed` overrides). |

### `[training.early_stopping]` → Lightning `EarlyStopping` (on by default)

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `enabled` | bool | `true` | | Turn early stopping on/off. |
| `monitor` | str | `"val_final_loss"` | | Metric to monitor. |
| `mode` | str | `"min"` | `min` \| `max` | Whether lower or higher is better. |
| `patience` | int | `8` | `>= 1` | Epochs without improvement before stopping. |
| `min_delta` | float | `0.0001` | | Minimum change counted as improvement. |

### `[training.checkpoint]` → Lightning `ModelCheckpoint` (opt-in)

Off by default — the run recorder already writes rehearsal-schema checkpoints that
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

## `[pretrain]` — continual-rehearsal pre-training

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `task_sequence` | list[str] | `[]` → `[[tasks]]` order | tasks must exist | Order tasks are introduced across steps. |
| `n_runs` | int | `1` | `>= 1` | Independent repeats (different seeds), written to `runs/runNN/`. |
| `task_order` | str | `"fixed"` | `fixed` \| `random` | `fixed` = `task_sequence` order; `random` = per-run shuffle. |
| `checkpoint` | str (path) | `None` | | **Warm-start**: load this checkpoint's encoder + heads as the starting point (`--checkpoint` overrides). Its tasks count as already-learned and are skipped as new steps (they still take part in rehearsal + evaluation); training continues with the `task_sequence` tasks the checkpoint doesn't already contain. Errors if a checkpoint task isn't in the catalog, or if every `task_sequence` task is already in the checkpoint. |

### `[pretrain.rehearsal]`

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `interval` | int | `1` | `>= 1` | Already-learned tasks rejoin training every Nth step; `1` = always replay. |
| `default_replay` | float \| int | `0.05` | float in `(0,1)` or int `>= 1` | Replay amount per old task: a fraction of its labels, or an absolute label count. |
| `per_task` | table (str→num) | `{}` | keys must be tasks; same value rule | Override `default_replay` for named tasks, e.g. `per_task = { density = 0.2 }`. |

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
| `class_weight` | float | `5.0` | | Weight of the classification objective vs. the regression targets. |
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
| `strategy` | str | `"top_qc"` | `top_qc` \| `random` \| `explicit` | Seed-selection algorithm. |
| `n` | int | `20` | `>= 1` | Total seed compositions to return. |
| `split` | str | `"test"` | `train`/`val`/`test`/`all` | Split to draw candidates from. |
| `explicit` | list[str] | `[]` | required (non-empty) when `strategy = explicit` | Explicit candidate pool. |
| `explicit_append` | list[str] | `[]` | each must have a computable descriptor | Extra seeds appended to every strategy's output. |
| `dedup_by_element_system` | bool | `true` | | Keep only one composition per element system (set of element symbols). |

### `[[inverse.scenarios]]` — design objectives (array of tables)

At least one required; names unique.

| Key | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `name` | str | — | required, unique | Scenario identifier (→ output subdir). |
| `reg_tasks` | list[str] | — | required, non-empty, `len == len(reg_targets)` | Regression heads to steer. |
| `reg_targets` | list[float] | — | required, paired with `reg_tasks` | Desired value per regression task. |
| `class_task` | str | `"material_type"` | | Classification head defining the objective. |
| `class_target` | int | `None` | | Target class index; `None` = the default class `[1]`. |

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
| `pretrain` | `--max-epochs N` (→ `training.max_epochs`), `--checkpoint PATH` (warm-start / continue a sequence) |
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
[pretrain.rehearsal]
interval = 1
default_replay = 0.05

[output]
dir = "artifacts/pretrain"
```

```bash
fm pretrain --config pretrain.toml
fm finetune --config finetune.toml --checkpoint artifacts/pretrain/runs/run00/training/final_model.pt
fm predict  --config predict.toml  --checkpoint artifacts/finetune/training/final_model.pt --split test
```

See [`samples/`](../samples/) for complete formal + smoke configs of all four subcommands.
