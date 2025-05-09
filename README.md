# Foundation Model for Material Properties

A multi-task learning model for predicting various material properties.

## Project Structure

```
foundation_model/
├── src/
│   └── foundation_model/    # Main package
│       ├── models/          # Neural network models
│       │   ├── fc_layers.py # Basic neural network layers
│       │   ├── flexible_multi_task_model.py # Refactored multi-task model
│       │   ├── multi_task_flexible.py # Legacy multi-task property predictor
│       │   ├── task_config.py # Task configuration classes
│       │   ├── components/   # Model components
│       │   │   ├── structure_encoder.py # Structure encoding module
│       │   │   ├── lora_adapter.py # Low-Rank Adaptation module
│       │   │   └── gated_fusion.py # Gated fusion module
│       │   └── task_head/    # Task-specific heads
│       │       ├── base.py   # Base task head interfaces
│       │       ├── regression.py # Regression task head
│       │       ├── classification.py # Classification task head
│       │       └── sequence/ # Sequence prediction heads
│       │           ├── rnn.py # RNN-based sequence head
│       │           ├── fixed_vec.py # Fixed vector sequence head
│       │           ├── tcn_film.py # TCN with FiLM modulation
│       │           └── transformer.py # Transformer-based sequence head
│       │
│       ├── data/            # Data handling
│       │   ├── dataset.py   # Dataset implementation
│       │   ├── datamodule.py # Lightning data module
│       │   └── splitter.py  # Data splitting utilities
│       │
│       ├── utils/           # Utility functions
│       │   ├── training.py  # Training utilities
│       │   └── plotting.py  # Visualization utilities
│       │
│       ├── configs/         # Configuration files
│       │   └── model_config.py # Model and experiment configs
│       │
│       └── scripts/         # Execution scripts
│           └── train.py     # Main training script
│
├── configs/                 # YAML configuration files
│   └── model_configs/       # Model configuration files
│       └── base_model.yaml  # Base model configuration
│
├── data/                    # Data directory
│   └── raw/                 # Raw data files
│
├── results/                 # Output directory
│   ├── models/             # Saved models
│   ├── logs/               # Training logs
│   └── figures/            # Generated plots
│
└── notebooks/              # Jupyter notebooks
    └── experiments/        # Experimental notebooks
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/foundation_model.git
cd foundation_model
```

2. Install the package using uv:
```bash
uv sync --frozen --all-groups
```

This will install all dependencies as defined in the pyproject.toml and uv.lock files, including both production and development dependencies, and ensure exact version matching. This method is preferred for reproducible installations.


If you need to add additional dependencies, use:
```bash
uv add <package_name>
# or for development dependencies
uv add --dev <package_name>
```

After adding new dependencies, update the lock file:
```bash
uv pip freeze > uv.lock
```

## Usage

### Training

To train the model with default settings:

```bash
# From the project root directory
python src/foundation_model/scripts/train.py
```

The training script is installed as part of the package, so you can also run it from anywhere after installation:

```bash
python -m foundation_model.scripts.train
```

### Configuration

There are two ways to configure the model:

1. **Command-line arguments**:

Available command line arguments:

- `--pretrain` (bool): enable contrastive / cross / mask losses  
- `--with_structure` (bool): expect structure descriptors in the batch  
- `--freeze_encoder` (bool): stop gradient for shared (and structure) encoders  
- `--lora_rank` (int): LoRA adapter rank (0 = off)  
- `--sequence_mode` [str]: `rnn|vec|transformer|tcn|hybrid` (default: transformer)  
- `--loss_weights` JSON string: e.g. `'{"con":1,"cross":1,"mask":1,"attr":0}'`  
- `--max_epochs`, `--accelerator`, `--devices`, `--strategy`, `--default_root_dir`  
- `--batch_size`, `--num_workers`

2. **YAML configuration**:

You can also use YAML configuration files for more complex setups. See `configs/model_configs/base_model.yaml` for an example.

#### Pre‑training vs. Fine‑tuning

| Mode | Arguments | What trains |
|------|-----------|-------------|
| **Encoder frozen** | `--freeze_encoder --lora_rank 0` | Only task heads (linear probe) |
| **LoRA micro‑tune** | `--freeze_encoder --lora_rank 8` | Heads + small LoRA adapters |
| **Full fine‑tune** | *(default)* | Encoder + heads |
| **Pre‑train** | `--pretrain --with_structure` | Encoder with dual‑modality losses |

> Tip: combine `--pretrain` with `--loss_weights '{"attr":0,"seq":0}'` if you
> don't want downstream heads to affect pre‑training.


## Recent Updates

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


## Features

- Multi‑task learning for material property prediction  
- **Dual‑modality support**: formula descriptors **+** optional structure descriptors with gated fusion  
- **Pre‑training & downstream in one model**  
  - Pre‑train losses: contrastive, cross‑reconstruction, masked‑feature, property supervision  
  - `--pretrain` flag toggles extra losses; same architecture used for fine‑tune  
- **Flexible sequence heads**: `rnn`, `vec`, `transformer`, `tcn`, `hybrid` (Flash‑Attention inside)  
- **Encoder control**: `--freeze_encoder` to lock shared layers; add *LoRA* adapters with `--lora_rank`  
- Handles missing values via masking & modality dropout  
- Comprehensive logging and visualization tools  
- Configurable data splitting strategies  
- Early stopping and model checkpointing

## Model Architecture

The model consists of:
1. A shared foundation encoder for learning common features
2. An intermediate deposit layer
3. Task-specific heads for different prediction tasks:
   - Regression heads for continuous properties
   - Classification heads for categorical properties
   - Sequence heads for time series or sequential data

## Data Handling

- Supports multiple material properties
- Handles missing values through masking
- Configurable data splitting ratios
- Property-specific sampling fractions

### Quick Examples

##### Example 1 – Pre‑train with formula+structure

```bash
python -m foundation_model.scripts.train \
  --pretrain --with_structure \
  --loss_weights '{"attr":0,"seq":0}' \
  --max_epochs 60
```

##### Example 2 – Fine‑tune only heads with LoRA (encoder frozen)

```bash
python -m foundation_model.scripts.train \
  --freeze_encoder --lora_rank 8 \
  --sequence_mode rnn
```

##### Example 3 – Full fine‑tune, Flash‑Attention transformer head

```bash
python -m foundation_model.scripts.train \
  --sequence_mode transformer --d_model 256 --nhead 4
```

##### Example 4 – Partial fine‑tune (encoder unlocked, LoRA off)

```bash
python -m foundation_model.scripts.train \
  --freeze_encoder False --lora_rank 0 \
  --sequence_mode vec --seq_len 256
```
