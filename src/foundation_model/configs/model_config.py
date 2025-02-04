from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelConfig:
    # Model architecture configuration
    shared_block_dims: List[int] = field(
        default_factory=lambda: [None, 128, 64, 32]
    )  # None will be replaced with input_dim at runtime
    task_block_dims: List[int] = field(default_factory=lambda: [32, 16, 1])
    norm_shared: bool = True
    norm_tasks: bool = True
    residual_shared: bool = False
    residual_tasks: bool = False

    # Optimizer configuration
    shared_block_lr: float = 0.005
    task_block_lr: float = 0.01

    # Training configuration
    batch_size: int = 128
    num_workers: int = 0
    max_epochs: int = 100

    # Dataset configuration
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    random_seed: int = 42

    # Property sampling ratios
    property_fractions: Dict[str, float] = field(
        default_factory=lambda: {
            "Seebeck coefficient": 1.0,
            "Thermal conductivity": 1.0,
            "Electrical resistivity": 1.0,
            "Magnetic susceptibility": 1.0,
            "Specific heat capacity": 1.0,
            "Electrical conductivity": 1.0,
            "ZT": 1.0,
            "Hall coefficient": 1.0,
            "Power factor": 1.0,
            "Carrier concentration": 1.0,
            "Thermopower": 1.0,
            "Lattice thermal conductivity": 1.0,
            "Hall mobility": 1.0,
            "Electronic contribution": 1.0,
            "Electronic thermal conductivity": 1.0,
            "Band gap": 1.0,
            "Density": 1.0,
            "Efermi": 1.0,
            "Final energy per atom": 1.0,
            "Formation energy per atom": 1.0,
            "Total magnetization": 1.0,
            "Volume": 1.0,
        }
    )

    # Trainer configuration
    trainer_config: Dict = field(
        default_factory=lambda: {
            "accelerator": "auto",
            "devices": 4,
            "strategy": "auto",
        }
    )

    # Early stopping and checkpoint configuration
    early_stopping_config: Dict = field(
        default_factory=lambda: {
            "monitor": "val_loss",
            "patience": 10,
            "mode": "min",
            "min_delta": 1e-4,
        }
    )

    checkpoint_config: Dict = field(
        default_factory=lambda: {
            "monitor": "val_loss",
            "save_top_k": 1,
            "mode": "min",
            "save_last": True,
        }
    )


@dataclass
class ExperimentConfig:
    # Experiment path configuration
    data_dir: str = "data/raw"
    results_dir: str = "results"
    model_save_dir: str = "results/models"
    log_dir: str = "results/logs"
    figures_dir: str = "results/figures"

    # Feature extraction configuration
    xenonpy_featurizers: List[str] = field(
        default_factory=lambda: [
            "WeightedAverage",
            "WeightedVariance",
            "MaxPooling",
            "MinPooling",
        ]
    )
