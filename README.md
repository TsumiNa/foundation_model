# Foundation Model for Material Properties

A multi-task learning model for predicting various material properties.

## Project Structure

```
foundation_model/
├── src/
│   └── foundation_model/    # Main package
│       ├── models/          # Neural network models
│       │   ├── layers.py    # Basic neural network layers
│       │   └── multi_task.py # Multi-task property predictor
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

2. Install the package in development mode:
```bash
pip install -e .
```

This will automatically install all required dependencies.

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

Available command line arguments:

- `--max_epochs`: Maximum number of training epochs (default: 100)
- `--accelerator`: Hardware accelerator to use (default: "auto")
- `--devices`: Number of devices to use (default: 4)
- `--strategy`: Training strategy (default: "auto")
- `--default_root_dir`: Directory for logs and checkpoints
- `--batch_size`: Training batch size (default: 128)
- `--num_workers`: Number of data loading workers (default: 0)

### Model Configuration

Model and training configurations can be modified in `configs/model_config.py`:

- `ModelConfig`: Neural network architecture and training parameters
- `ExperimentConfig`: Experiment settings and paths

## Features

- Multi-task learning for material property prediction
- Flexible architecture with shared and task-specific layers
- Support for handling missing values in the dataset
- Comprehensive logging and visualization tools
- Configurable data splitting strategies
- Early stopping and model checkpointing

## Model Architecture

The model consists of:
1. A shared block for learning common features
2. An intermediate deposit layer
3. Task-specific blocks for each property prediction

## Data Handling

- Supports multiple material properties
- Handles missing values through masking
- Configurable data splitting ratios
- Property-specific sampling fractions
