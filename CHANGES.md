# Recent Updates

## 2026-07-04 (Unified `fm` CLI refactor)

- **Single `fm` command** replaces the three legacy console scripts. A thin
  [click](https://click.palletsprojects.com/) group (`cli/main.py`) with four subcommands, each a
  `build_*_config(toml) → run(cfg, recorder)` dispatch — no business logic in the CLI:
  - `fm pretrain` — continual-rehearsal pre-training with a rehearsal-**interval** replay schedule
    (old tasks join only every Nth step), fraction/count replay amounts, and an `n_runs` sweep.
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
