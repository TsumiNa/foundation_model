# Recent Updates

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
