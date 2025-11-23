# Recent Updates

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

## 2025-11-27

- **optimize_latent features & docs**:
  - Added multi-target optimization: `task_targets={task: target}` jointly minimizes MSE across multiple regression heads and returns per-task scores.
  - Reshaped outputs: default returns `optimized_input` (B, R, D) and `optimized_target` (B, R, T); `return_details=True` yields per-task trajectories and initial scores.
  - Removed `SUMMARY.md` and `DOCUMENTATION_INDEX.md`; consolidated guidance into `README_OPTIMIZATION_CORE.md` (usage, targets/extrema, multi-restart, multi-target; `initial_input` required).
- **Demo notebook alignment**: `notebooks/advanced_optimization_demo.ipynb` now trains a density task on sample polymer data (`descriptor_path`/`pretrain_data_path`) before optimization, seeds from real descriptors, and updates text/examples accordingly.
