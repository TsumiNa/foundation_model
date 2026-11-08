# Recent Updates

## 2026-11-09 (2.1.1 — quieter, clearer output for users of the released models)

Found while building a demo of the released models for colleagues:

- **Quieter logs.** Building a model logged its structure and every head at INFO, hundreds of lines per
  member, so `yammi predict` and `load_suite` flooded the terminal. These messages are DEBUG now;
  `YAMMI_LOG_LEVEL=DEBUG` (or `WARNING`, ...) sets the level.
- **No download progress bars for cached members.** A member already complete in the Hugging Face cache
  is read from it without asking the Hub, so loading 10 cached members no longer prints 10 "Fetching N
  files" bars. The revision is still resolved to a commit first, so a moved tag is still followed.
- **`yammi_version` replaces the leftover `fm_version`** in `config.json`, `prediction_info.json`, a
  fine-tune's `derived_from` record and `PretrainedModel.derived_from()`. Bundles written before read as
  before; `BundleConfig.fm_version` remains as a deprecated alias.
- **`yammi inverse --pretrained` reports in physical units.** A scenario's `SUMMARY.md` lists the targets
  as written (value and unit, class names) and adds the achieved mean ± std of each regression target in
  physical units. It used to show only the training-scale value (a 2.5 eV target read `band_gap→1.09964`).
- **Install instructions never name the PyPI placeholder.** The missing-`hub`-extra errors said
  `pip install 'yammi[hub]'`, which resolves to the empty PyPI placeholder; they now give the GitHub URL
  (or `uv sync --extra hub` in a clone). README, docs, the model card template and the `yammi-pretrained` skill
  show both ways: `pip install` from the GitHub URL, or a clone with `uv sync --frozen` for the locked versions.

## 2026-11-09 (2.1.0 — findings from testing the released models)

Tested `TsumiNa/yammi@v1.0.0` as outside users would: input formats, every head and member, curve grids,
error paths, CLI vs API, fine-tuning on three unseen tasks and a new classification head, inverse design,
offline use. Fixes:

- **Out-of-range predictions are NaN, not the training extreme.** For heads whose inverse is bounded
  (15 of v1's 41), 0.4–3.6 % of held-out predictions per member landed past the bound and came back as
  the most extreme training value, which was almost always far off (`total_magnetization` R² 0.25 → −24).
  `predict` / `predict_curve` now return NaN there with a `SaturationWarning`; `saturated="clip"` keeps
  the old behaviour.
- **Combined predictions.** `Suite.predict` / `Suite.predict_curve`: mean of the members that stay in
  range, plus `<task>:std` (member spread). It had a higher held-out R² than the default member on 27 of 29
  regression heads. `yammi predict --pretrained owner/name` combines all members unless `#member` is given
  (`prediction_info.json` adds `members` and `combined`; `member` is null when members are combined).
- **New heads fine-tuned from a bundle stay invertible.** `fit_default` keeps Yeo-Johnson λ in [0, 2].
  A zero-inflated target (HEA Curie temperature, 72 % zeros) had fitted λ = −3: test R² −2.4, now 0.82.
  The fine-tune log warns when its metrics leave out non-finite predictions.
- **`num_classes` may be left out** for classification tasks when fine-tuning a bundle: an existing head
  takes the bundle's count, a new head counts its labels (it failed before).
- **`PretrainedModel.derived_from()`** reports where a derived bundle came from (None for an original). It
  used to describe the model itself.
- **Clear Hub errors**: gated repository without access, unknown revision (lists the release tags), unknown
  or private repository — instead of raw 401/404 errors with request ids.
- Model card: the combined usage first, an "Accuracy on held-out compositions" section (via
  `--card-extra`), limitations for unseen elements and unbounded heads, and how to install the agent skill.
- New skill `yammi-pretrained` (`.claude/skills`, `.agents/skills`) for using released models; the other
  skills point to it and note the 2.x behaviour. README: how to give the skill to an agent.

## 2026-11-08 (2.0.1 — one multi-arch image; the container is the default way in; offline loading)

- `ghcr.io/tsumina/yammi:<version>` / `:latest` / `:sha-<sha>` now hold both `linux/amd64` and
  `linux/arm64`; Docker / Apptainer / SingularityCE pull the host's variant and only that one. The
  separate `rikyu-*` tags are gone (RIKYU pulls `:<version>` like every other machine).
  `scripts/rikyu_pull_container.sh` writes `~/containers/yammi-<version>.sif`.
- The image now installs the `[hub]` extra (`huggingface_hub`), so `yammi … --pretrained owner/name`
  works inside it. Before, it failed with a missing-module error. `docs/container.md` shows how to pass
  the token and cache.
- **Loading from the cache offline works.** `load_pretrained` / `--pretrained owner/name@rev` asked the Hub
  which commit a revision points to even when the files were cached, so `HF_HUB_OFFLINE=1` (or a compute
  node without network) failed. When the Hub cannot be reached it now reads the commit from the local
  cache, and says so plainly when the revision is not cached either.
- README, `docs/pretrained_models.md` and model cards lead with the container (Docker / Apptainer); a pip
  install is for scripting from Python or developing on yammi.

## 2026-11-08 (2.0.0 — the project is now YAMMI)

**YAMMI — Yet Another Materials Model, Inverse-ready** (pronounced like *yummy*). The PyPI name
`foundation-model` belongs to an unrelated project, and "foundation model" said nothing about materials.

- Distribution, import package and command are all `yammi`: `pip install 'yammi[hub] @ git+https://github.com/TsumiNa/yammi'`,
  `import yammi`, `yammi pretrain|finetune|inverse|predict|export|publish`. **The `fm` command is gone.**
- `import foundation_model` (and `foundation_model.<module>`) still works and warns: it resolves to the same
  module objects as `yammi`, so 1.x code and checkpoints pickled under the old module paths load unchanged.
- The GitHub repository is `TsumiNa/yammi`, the released models `TsumiNa/yammi` on the Hugging Face Hub, the
  containers `ghcr.io/tsumina/yammi`. GitHub and the Hub redirect the old addresses.
- Bundle format unchanged (schema 2). Cards of bundles written before 2.0 name the writer as
  "foundation-model 1.x (now yammi)".
- Skills are `yammi-pretrain`, `yammi-eval-ws-ft`, `yammi-inverse`. Experiment folders are records and keep the
  names and container versions they ran with.

## 2026-11-08 (1.0.2 — gated model cards upload; install line)

- `fm publish --visibility gated` was refused by the Hub: the licence checkbox put the licence id
  (`cc-by-nc-sa-4.0`) in a metadata key, and the Hub rejects keys containing "." or "$". The checkbox now
  reads "I accept the licence terms stated above"; the gating prompt above it names the licence.
- The model card, README and `docs/pretrained_models.md` said `pip install 'foundation-model[hub]'`, which
  installs an unrelated PyPI project of that name. They now install from the GitHub repository.

## 2026-11-08 (1.0.1 — predictions past a transform's range; local member selection)

- **No more silent NaN.** A Yeo-Johnson step with λ < 0 (or > 2), and any Box-Cox step, has an inverse that
  is undefined past a bound; the training maximum often sits on it (14 of the 41 release tasks). A
  prediction past it came back NaN with no warning.
  - `fm export` and fine-tuned new heads append a `clip` at the training-split range on each such side
    (`foundation_model.data.targets.guard_inverse_domain`). Predictions there saturate at the most extreme
    training value; sides whose inverse is defined everywhere still extrapolate. These
    clips are marked `guard: true`, so they are never confused with a clip that is part of a target's
    normalisation. That key makes this **bundle schema v2**; readers accept v1 bundles, which have no guards.
  - `predict` / `predict_curve` raise the new `SaturationWarning` when values saturate, and when a bundle
    exported with 1.0.0 still yields NaN (re-export it).
  - Labels are normalised with the unguarded transform (`strip_inverse_guard`), so held-out values past the
    training range keep their values when fine-tuning; a fine-tuned existing head's guard widens to its new
    training range.
  - Per-step evaluation with a configured scaler computed R² over NaN and stopped the run. Metrics and
    plots now leave out non-finite predictions, and the metrics record how many (`nonfinite`).
- `--pretrained` / `load_pretrained` accept `<bundle dir>#member` for local bundles, as Hub sources do.
- Model cards group the heads table by kind, abbreviate class lists longer than 12, and note saturation.
- The release repository is `TsumiNa/fm`; examples use it.

## 2026-11-04 (1.0.0 — store, publish and reuse pretrained models)

foundation-model 1.0 gives models a stable home outside the training repository. The public interface is
bundle schema v1, `foundation_model.hub` and the `--pretrained` CLI paths. It follows semantic versioning
from here (`docs/pretrained_models.md`). The entries below, from target transforms to `fm publish`, make
up the release; the design is in `docs/refactor/pretrained-release/00-overview.md`.

- Store: `fm export`.
- Publish: `fm publish`, with model cards and gating.
- Reuse:
  - `load_pretrained` / `fm predict --pretrained`, in physical units;
  - `fm finetune --pretrained`, with derived bundles that inherit the licence;
  - `fm inverse --pretrained`, with physical-unit targets.
- Legacy checkpoints and catalog-based commands keep working unchanged.

## 2026-11-04 (publish: `fm publish`, model cards)

- `fm publish BUNDLE --repo owner/name --tag v1.0.0 --visibility private|gated|public`. Every member is
  loaded back first. The command creates the repository, sets gating (manual approval), uploads the bundle
  and tags the release; `--dry-run` writes the card only.
- `foundation_model.hub.render_card` generates `README.md`:
  - YAML header: licence (Hub id or `other`), `library_name: foundation-model`, tags, and for gated repos
    an access form with non-commercial and licence checkboxes;
  - body: the heads table (units, shape heads, `t` ranges), members, usage (Python + CLI), limitations,
    licence terms (including the scratch-trained exemption), lineage, and any `--card-extra` Markdown.

## 2026-11-04 (reuse: `fm inverse --pretrained`)

- `fm inverse --pretrained SOURCE` (or `[inverse].pretrained`) designs from a released bundle with no
  catalog or data.
  - Targets are written in physical units and class names, and converted with the bundle's transforms;
    shape-only heads stay in shape units.
  - Reports add physical-unit regression channels, and `scenario.json` keeps the targets as written.
  - Given the same seeds and targets, the result matches the catalog path's.
- New seed strategy `elements`: random compositions over an element set, available with or without a bundle.

## 2026-11-04 (reuse: `fm finetune --pretrained`, derived bundles)

- `fm finetune --pretrained SOURCE` (or `[finetune].pretrained`) fine-tunes a released bundle. The TOML
  lists only the user's tasks, with data in physical units:
  - existing heads use the bundle's transform, class names or shape normalisation;
  - new heads fit their own transform on the training split.
- The run writes `<output>/bundle`, a derived bundle with the source licence and `derived_from`.
  Lineage chains across repeated fine-tunes.
- `PretrainedModel.save(path)` writes a one-member derived bundle the same way.
- `TaskCatalog(target_inputs=…)` maps physical-unit columns onto the training scale: transforms,
  shape normalisation, and class names to indices.
- `fm finetune`'s fit and evaluation are factored into one helper, shared by the checkpoint path and the
  bundle path. The checkpoint path is unchanged.

## 2026-11-04 (using released models: `load_pretrained`, `fm predict --pretrained`)

- `foundation_model.hub.load_pretrained(source, member=, revision=)` loads one member, strictly, from a
  local bundle or a Hub repository `owner/name[@revision][#member]`. On the Hub only the metadata and the
  requested member are downloaded.
- `load_suite` gives all members, loaded on first use.
- `PretrainedModel` methods:
  - `predict`: scalar heads in physical units; classification as class names, with optional
    probabilities;
  - `predict_curve`: the default grid, one shared grid, or one grid per composition; out-of-range `t`
    raises `ExtrapolationWarning`;
  - also `latent`, `descriptors` and `units`.
- `fm predict --pretrained SOURCE --compositions … --output-dir …` needs no config or data.
- New optional extra `foundation-model[hub]` (`huggingface_hub`).

## 2026-11-04 (release bundles: `fm export`)

- New `foundation_model.hub` package with the release bundle, schema v1.
  - `config.json`: descriptor with an element-features hash, architecture, and per-head metadata (resolved
    widths, plain-data target transform or class names, units, shape-only marker, `t` range and default
    grid). It also carries the licence and provenance.
  - `manifest.json` and per-member `model.safetensors`.
  - `build_model` rebuilds the architecture from `config.json` alone; `load_member` loads with `strict=True`.
- `fm export --config --run ... --out --license`: one member per run.
  - Refuses members whose heads or widths differ, heads without a transform (unless `output_scale = "shape"`)
    and classification heads without class names.
  - Verifies that every member's outputs are bit-identical to its checkpoint.
- `[[tasks]]` gains release metadata `unit`, `t_unit`, `output_scale`. `[tasks.scaler]` also accepts JSON
  transform records. New dependency: `safetensors`.

## 2026-11-04 (target transforms as data)

- `foundation_model.data.targets`: a target transform is an ordered list of plain-number steps,
  with forward, inverse and JSON round trip and no pickle:
  - steps: `standard_scaler`, `power_transformer`, `log1p`, `arcsinh`, `clip`, `identity`;
  - `from_sklearn` converts every scaler in the project's preprocessing files and the AGIS
    pipeline, and rejects other estimators by name;
  - `fit_default` fits the default recipe;
  - `ClassLabels` holds classification label names;
  - `normalize_max_abs` gives the per-row normalisation of shape-only heads.
- `data/scripts/record_nemad_phonix_transforms_20261104.py` records the recovered transforms of the
  NEMAD and phonix-db `_norm` tables. It writes them only after checking them against every row, to
  within 1e-12.

## 2026-11-04 (pretrain: fixed run seed)

- `[pretrain] run_seed = "fixed"` keeps `[training].seed` for every run of an `n_runs` sweep, so a
  random-order sweep (`task_order = "random"` + `task_order_seed`) differs only in task order. The
  default `"per_run"` keeps the historical `[training].seed + i`.

## 2026-11-04 (licence: Apache-2.0)

- The code is licensed under Apache-2.0. `pyproject.toml` declared MIT and pointed to a missing `LICENSE`
  file, while the source headers already said Apache-2.0; `LICENSE` and `NOTICE` are now in the repository
  and the package metadata (`Apache-2.0 AND BSD-3-Clause`, for the bundled XenonPy element table) matches
  them. The licence does not cover pretrained weights or training data (`docs/licensing/`); the tracked
  teaching table is CC BY 4.0 (`notebooks/teaching_inverse_design/data/README.md`).

## 2026-10-02 (0.5.0 — selectable feature-token encoders)

- Expose Transformer encoders through TOML for pretraining, fine-tuning, prediction and inverse
  design, while retaining the MLP default and the original shared-token Python defaults.
- Add feature-specific scalar and grouped descriptor tokenization, CLS/mean/concatenation pooling,
  an independent latent output width, and a residual token-wise MLP control without attention.
  Feature/group encoders default to Pre-LN/GELU and initialize attention layers independently.
- Add optional explicit reconstruction decoder dimensions for controlled encoder comparisons.
- Validate configuration combinations and cover gradients, pooling, checkpoint restoration and
  workflow construction. Fix sinusoidal position encoding for odd Transformer widths.
- Publish new versioned container images before the transfer-learning benchmark runs on RIKYU.

## 2026-08-27 (0.4.0 — one optimizer, and a much smaller model class)

Breaking, in the order the changes read:

- **One `AdamW` with per-role parameter groups**, and `automatic_optimization` back on. The model
  built one optimizer *and* one `ReduceLROnPlateau` per group, which forced manual optimization —
  and manual optimization means Lightning drives no schedulers either, which is how stepping them
  became the model's job and how #45 (`patience` counting batches) got in. Verified lossless:
  replaying one metric trajectory through both shapes gives byte-identical learning rates over 20
  epochs. Per-group `lr` / `weight_decay` / `min_lr` survive; a per-group scheduler *policy* does
  not, and disagreement now raises rather than silently honouring one group.
- **`[training] devices` accepts one device only.** Distributed training was removed with its
  output half never written; multi-device configurations are rejected rather than merely
  undocumented, with a runtime guard for `devices = "auto"` on a multi-GPU node. See
  `ARCHITECTURE.md`, which also records the measured reason to doubt DDP is the right axis here.
- **`OptimizerConfig.interval` / `.frequency` removed.** Read by nothing but their own validator,
  and `interval = "step"` *is* #45.

Also in this release:

- `flexible_multi_task_model.py` **3296 → ~1100 lines**; inverse design and the KR batch plumbing
  moved out, and `models/inverse_design/` and `workflows/inverse/` are packages now — one module
  per question, `optimize_composition` down from 975 lines (709 of code) to 489 (223 of code).
- **`[training] learnable_loss_balancer`** (default `false`): uncertainty weighting existed on the
  model since before the `[training]` section but nothing ever routed a value to it, so it had
  never run. Supervised heads only — the autoencoder is excluded.
- The checkpoint machinery nothing read back is gone; the load path the CLIs actually use now has
  tests against a real model.
- Five duplicated implementations collapsed, found by an AST-normalised sweep of every function
  body; the run seed reaches inverse-design seed selection; the KR regroup validates its lengths.

## 2026-07-04 (Unified `fm` CLI refactor)

- **Single `fm` command** replaces the three legacy console scripts. A thin
  [click](https://click.palletsprojects.com/) group (`cli/main.py`) with four subcommands, each a
  `build_*_config(toml) → run(cfg, recorder)` dispatch — no business logic in the CLI:
  - `fm pretrain` — continual-rehearsal pre-training with a rehearsal-**interval** replay schedule
    (old tasks join only every Nth step), fraction/count replay amounts, and an `n_runs` sweep.
    Supports **warm-start** from a checkpoint (`--checkpoint` / `[pretrain].checkpoint`): loads the
    checkpoint's encoder + heads, treats its tasks as already-learned, and continues the sequence
    with the remaining tasks — i.e. extend a trained model with new tasks. (`fm finetune` already
    warm-starts, since it loads a checkpoint to fine-tune.) With `--resume` (`[pretrain].resume`),
    a run killed mid-sequence (e.g. a scheduler job timeout) picks up in place from its output dir's
    latest step checkpoint on re-submit; finished runs are skipped. Resume granularity is one
    completed task-step (each step already trains a fresh optimizer, so no optimizer state is kept).
  - `fm finetune` — frozen-encoder fine-tuning of selected heads; the AE head **stays trainable**
    (at `ae_lr`) and `task_log_sigmas` are frozen; non-target heads are disabled for the fit and
    restored before saving so the checkpoint keeps every head.
  - `fm inverse` — scenario × path inverse design (11 default paths: 3 latent α + 8 composition),
    seed selection (top_qc / random / explicit + element-system dedup), per-scenario figures +
    trajectory `.npz`/plots/animations.
  - `fm predict` — arbitrary-checkpoint evaluation/prediction (`strict=False` load, split overlay
    via `CompoundDataModule`, masked metrics, per-task scaler inverse-transform).
- **Config is TOML + dataclasses** (no YAML / LightningCLI / omegaconf): `[data]`/`[descriptor]`/
  `[datasets.*]`/`[[tasks]]`/`[model]`/`[training]` shared sections normalized into validated
  `@dataclass` configs (`workflows/`), unknown keys rejected by name. `--set section.key=value`
  overrides + first-class flags (`--seed`/`--accelerator`/`--sample`/…).
- **Configurable model architecture**: `[model]` sets depth + per-layer widths via `*_hidden_dims`
  lists (`encoder_hidden_dims`, `head_hidden_dims`, `kr_x_hidden_dims`, `kr_t_hidden_dims`,
  `n_kernel`) — the input (descriptor width / `latent_dim`) is prepended and the output appended.
  Each `[[tasks]]` may override its own head (`hidden_dims` for reg/clf; `x_hidden_dims` /
  `t_hidden_dims` / `n_kernel` for KR), falling back to the `[model]` defaults. Replaces the fixed
  single-hidden-layer `encoder_hidden` / `head_hidden_dim` scalars.
- **Configurable Lightning callbacks/loggers**: `[training.early_stopping]` / `[training.checkpoint]`
  (Lightning `EarlyStopping` / `ModelCheckpoint`) and `[training.logging]` (`CSVLogger` /
  `TensorBoardLogger`) — a flexible subset of each. EarlyStopping is on by default; the Lightning
  checkpoint + loggers are opt-in so the `RunRecorder` stays the default checkpoint/log writer.
- **Provenance on every run**: `run_provenance.json` (resolved config + package versions + git +
  argv + seeds) and `run.log`, written by `workflows/recording.py::RunRecorder` — the single
  artifact writer for the training/predict flows.
- **Config reference + fixes**: added [`docs/configuration.md`](docs/configuration.md), an
  authoritative per-key schema for all four subcommands. `fm predict` now accepts `--seed` /
  `--accelerator` (routed to `[predict]`, previously crashed with `unknown key 'training'`) and
  runs on the chosen device; `--config` help text added; pretrain/finetune command-section fields
  documented inline. `[training].devices` now accepts Lightning's full form — an int count,
  a list of device indices (`[1, 3]`), or a string (`"auto"` / `"1,3"` / `"0-3"`) — with validation.
- **New package layout**: `workflows/` (task_catalog, recording, `_sections`/`_engine`, pretrain,
  finetune, inverse, inverse_trajectory, plots, predict) + `cli/`; colocated `<module>_test.py`.
- **Removed**: `src/foundation_model/scripts/` in full (`train.py`/LightningCLI, the
  continual-rehearsal runners, `dynamic_task_suite`, `multi_task_progressive_clf`,
  `finetune_inverse_heads`, `paper_inverse_*`, `eval_inverse_methods`, `prediction_writer`), the
  root `run_*.sh` wrappers, legacy `samples/` configs, stale docs, the `fm-trainer`/`fm-pretrain-suite`/
  `fm-progressive-clf` entry points, and the `omegaconf` dependency. Stale workflow-demo notebooks
  were pruned and replaced by `notebooks/fm_cli_workflows.ipynb`.

## 2025-11-27 (Multi-GPU reliability)

- **Distributed training fixes**:
  - `CompoundDataModule` now records its `DistributedSampler` and calls `set_epoch` every epoch, restoring proper shuffling in DDP runs.
  - Lightning `Trainer` construction in `dynamic_task_suite.py` enables `sync_batchnorm` and scales learning rates with `world_size`, keeping gradient statistics consistent regardless of the number of GPUs.
- **Prediction + metrics correctness**:
  - `FlexibleMultiTaskModel` switches to `torchmetrics.R2Score` with stage-scoped modules, deduplicates DistributedSampler padding, and logs aggregated R² per task without bias.
  - Prediction plotting writes per-rank Parquet shards that are merged on rank 0, eliminating OOM risks while guaranteeing deterministic ordering and deduplication.
- **Testing & diagnostics**:
  - Added exhaustive tests in `dynamic_task_suite_test.py` for deduplication, sampler index math, and per-rank file merging.
  - Created troubleshooting scripts (`diagnose_multi_gpu.py`, `verify_multi_gpu_fixes.py`, `test_set_epoch_fix.py`) plus written analyses (`MULTI_GPU_FIXES_SUMMARY.md`, `multi_gpu_analysis.md`, `multi_gpu_additional_issues.md`) documenting the investigation and verification steps.
- **optimize_latent features & docs**:
  - Added multi-target optimization: `task_targets={task: target}` jointly minimizes MSE across multiple regression heads and returns per-task scores.
  - Reshaped outputs: default returns `optimized_input` (B, R, D) and `optimized_target` (B, R, T); `return_details=True` yields per-task trajectories and initial scores.
  - Removed `SUMMARY.md` and `DOCUMENTATION_INDEX.md`; consolidated guidance into `README_OPTIMIZATION_CORE.md` (usage, targets/extrema, multi-restart, multi-target; `initial_input` required).
- **Demo notebook alignment**: `notebooks/advanced_optimization_demo.ipynb` now trains a density task on sample polymer data (`descriptor_path`/`pretrain_data_path`) before optimization, seeds from real descriptors, and updates text/examples accordingly.

## 2025-11-25

- **Simplified Architecture - Deposit Layer Cleanup**:
  - **Removed deposit layer Linear transformation**: Encoder now directly outputs latent representations that are transformed with `torch.tanh()` at the `FlexibleMultiTaskModel` level, eliminating the intermediate Linear layer bottleneck.
  - **Performance improvement**: Simplified architecture achieves 2x better optimization scores (5.0 vs 2.5) with smoother convergence curves due to stronger gradient flow and reduced constraints.
  - **Bug fixes**:
    - Fixed `AttributeError` in `flexible_multi_task_model.py:525` where code referenced non-existent `encoder.deposit.parameters()`.
    - Removed redundant type-checking logic in `foundation_encoder.py:170-174`.
  - **Code cleanup**:
    - Renamed `deposit_dim` → `latent_dim` throughout codebase for clarity.
    - Updated all docstrings to replace "deposit layer" references with "task heads" or "latent representations".
    - Updated `_TransformerBackbone` documentation to reflect direct gradient flow to task heads.
  - **Verification and comparison**:
    - Added `compare_input_vs_latent.py` script comparing input space vs latent space optimization strategies (250 steps, both converge with latent space 29% better).
    - Added `verify_current_architecture.py` to validate simplified architecture.
    - Added `diagnose_architecture_difference.py` to analyze performance improvements.
  - **Documentation**:
    - Created `ARCHITECTURE_CLEANUP_FINAL.md` with comprehensive cleanup report.
    - Created `SIMPLIFIED_ARCHITECTURE_CLEANUP.md` with detailed fix summary.
    - Updated `notebooks/advanced_optimization_demo.ipynb` to reflect simplified architecture (7 cells updated).

## 2025-11-06

- **Encoder configuration refactor**:
  - Added `EncoderType`, `BaseEncoderConfig`, and concrete `MLPEncoderConfig` / `TransformerEncoderConfig` dataclasses for declarative encoder selection.
  - Updated `FoundationEncoder` and `FlexibleMultiTaskModel` to consume the unified `encoder_config`, covering `[CLS]` and mean-pooling aggregation options.
  - Refreshed `ARCHITECTURE.md`, `README.md`, and sample configs to document how transformer tokens propagate gradients and how to choose between encoder backbones.

## 2025-05-14

- **Component cleanup**:
  - Removed obsolete `GatedFusion`, `LoRAAdapter`, and `StructureEncoder` modules from the components package.
  - Simplified classification and regression heads to operate without LoRA adapters.
  - Updated documentation to reflect the removal of gated fusion and LoRA features.

## 2025-05-13

- **Documentation & Code Consistency**:
  - Verified and confirmed that Sequence Heads in `FlexibleMultiTaskModel` correctly receive `h_task` (output from the deposit layer) as per the implementation.
  - Updated diagrams and descriptions in `README.md` and `ARCHITECTURE.md` to accurately reflect that Sequence Heads use `h_task` as their primary input and to use the new `task_sequence_data_batch` variable name.
  - Renamed the `temps_batch` parameter to `task_sequence_data_batch` across `FlexibleMultiTaskModel` methods (`forward`, `training_step`, `validation_step`, `predict_step`) for improved clarity.
  - Updated `flexible_multi_task_model_test.py` to align with the renaming of `temps_batch`, resolving a test failure.
- **Model & Training Refinements**:
  - Implemented manual optimization in `FlexibleMultiTaskModel`.
  - Updated classification and regression task heads, including `predict` method enhancements in `ClassificationHead`.
  - Refactored task configuration handling and head implementations, including adding `d_in` to `BaseTaskConfig` and removing `task_config.py`.
  - Streamlined model architecture by removing the legacy `multi_task_flexible.py` module.
  - Enhanced training and validation steps in `FlexibleMultiTaskModel` to better support self-supervised learning components.
  - Reorganized logging setup in `FlexibleMultiTaskModel`.
- **Data Handling Enhancements (`CompoundDataset` & `CompoundDataModule`)**:
  - Improved attribute validation, input validation, and error handling in `CompoundDataset` for robustness.
  - Enhanced `CompoundDataset` for predict mode and general data parsing.
  - Improved parameter handling in `CompoundDataModule` and `CompoundDataset`.
  - Refined classification target handling in `CompoundDataset`.
  - Updated `CompoundDataModule` to handle dictionary-based temperature/sequence inputs.
- **Testing & Dependencies**:
  - Added `pytest-mock` dependency for improved testing capabilities.
  - Added and enhanced unit tests for `CompoundDataModule`, `CompoundDataset`, and `FlexibleMultiTaskModel`.
- **Configuration**:
  - Updated `configs/model_configs/base_model.yaml` with enhanced base model configuration.
  - Added `ARCHITECTURE.md` for detailed model architecture documentation.

## 2025‑05‑10

- **Code Refactoring & Enhancements**:
  - Added foundation encoder and self-supervised learning components
  - Enhanced data handling in CompoundDataModule and CompoundDataset for multi-task support
  - Updated modality dropout handling and adjusted self-supervised loss computation
  - Removed LoRA parameters from model and task head, integrated freezing logic into optimizer config
  - Updated deposit layer handling and improved documentation for task-specific representations
- **Documentation & Organization**:
  - Moved update history from README.md to changes.md
  - Added copyright and license headers to model configuration and data module files
  - Enhanced docstrings for masked feature modeling and contrastive loss methods
  - Added model configuration and optimizer settings for flexible multi-task model

## 2025‑05‑09

- **Major Code Refactoring**:
  - Implemented a more modular architecture with separate components
  - Created a task head abstraction hierarchy for different task types
  - Moved components to dedicated modules (adapter/fusion layers)
  - Added configuration system using Pydantic models
  - Added YAML-based configuration support
- **Package Management**:
  - Switched from pip to uv for package management
  - Added proper dependency specifications in pyproject.toml

## 2025‑05‑08

- **Dual‑modality encoder** (formula + structure) introduced.
- New `--pretrain` flag enables contrastive, cross‑reconstruction, masked‑feature, and optional property‑supervision losses.
- **Encoder control flag** `--freeze_encoder` freezes shared / structure encoders.
- Added five selectable sequence heads: `rnn`, `vec`, `transformer` (Flash‑Attention), `tcn`, `hybrid`.
- CLI accepts `--sequence_mode` for custom recipes; each task config now supports `loss_weight` (replacing the legacy global `loss_weights` flag).
- `FlexibleMultiTaskModel.add_task` now accepts multiple task configs in one call for bulk head registration.
