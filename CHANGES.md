# Recent Updates

### 2025-05-13
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

### 2025‑05‑10
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

### 2025‑05‑09
- **Major Code Refactoring**:
  - Implemented a more modular architecture with separate components
  - Created a task head abstraction hierarchy for different task types
  - Moved components to dedicated modules (StructureEncoder, LoRAAdapter, GatedFusion)
  - Added configuration system using Pydantic models
  - Added YAML-based configuration support
- **Package Management**:
  - Switched from pip to uv for package management
  - Added proper dependency specifications in pyproject.toml

### 2025‑05‑08
- **Dual‑modality encoder** (formula + structure) with gated fusion.
- New `--pretrain` flag enables contrastive, cross‑reconstruction, masked‑feature, and optional property‑supervision losses.
- **Encoder control flags**
  - `--freeze_encoder` freezes shared / structure encoders
  - `--lora_rank` adds LoRA adapters for lightweight fine‑tuning
- Added five selectable sequence heads: `rnn`, `vec`, `transformer` (Flash‑Attention), `tcn`, `hybrid`.
- CLI accepts `--sequence_mode`, `--loss_weights` for custom recipes.
