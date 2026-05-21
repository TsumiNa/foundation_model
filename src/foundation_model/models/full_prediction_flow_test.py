from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import pytest
import torch

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import KernelRegressionTaskConfig, MLPEncoderConfig, RegressionTaskConfig, TaskType

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_DATA_PATHS: Final[list[Path]] = [
    _PROJECT_ROOT / "data/qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet",
    _PROJECT_ROOT / "data/qc_ac_te_mp_dos_reformat_20250615.pd.parquet",
]
_DATA_AVAILABLE: Final[bool] = all(path.exists() for path in _DATA_PATHS)

pytestmark = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="integration dataset not available locally (expected DOS parquet files under data/).",
)


@pytest.mark.integration
def test_full_prediction_flow_produces_kernel_regression_outputs() -> None:
    """Ensure predict_step returns kernel regression outputs for the integration dataset."""
    task_configs = [
        RegressionTaskConfig(
            name="density",
            type=TaskType.REGRESSION,
            data_column="Density (normalized)",
            dims=[128, 64, 32, 1],
            enabled=True,
            predict_idx="all",
        ),
        KernelRegressionTaskConfig(
            name="dos",
            type=TaskType.KERNEL_REGRESSION,
            data_column="DOS density (normalized)",
            t_column="DOS energy",
            x_dim=[128, 32, 16],
            t_dim=[32, 16],
            t_encoding_method="fc",
            enabled=True,
            predict_idx="all",
        ),
    ]

    # Descriptor frame and attributes frame are both composition-indexed parquet files.
    descriptor_df = pd.read_parquet(_DATA_PATHS[0])
    attributes_df = pd.read_parquet(_DATA_PATHS[1])

    def descriptor_fn(compositions):
        present = [c for c in compositions if c in descriptor_df.index]
        return descriptor_df.loc[present]

    datamodule = CompoundDataModule(
        task_configs=task_configs,
        descriptor_fn=descriptor_fn,
        task_frames={cfg.name: attributes_df for cfg in task_configs},
        composition_column="id",  # join key is the 'id' index (mp-*), not the 'composition' formula column
        batch_size=8,
        num_workers=0,
    )
    datamodule.setup(stage="predict")
    assert datamodule.predict_dataset is not None

    predict_loader = datamodule.predict_dataloader()
    assert predict_loader is not None

    batch = next(iter(predict_loader))
    model = FlexibleMultiTaskModel(
        encoder_config=MLPEncoderConfig(hidden_dims=[290, 128]),
        task_configs=task_configs,
    )

    with torch.no_grad():
        predictions = model.predict_step(batch, batch_idx=0)

    assert "dos_value" in predictions, "Expected DOS kernel regression output to be present"
    dos_output = predictions["dos_value"]
    assert isinstance(dos_output, list), "KernelRegression predictions should be returned as a list per sample"
    assert len(dos_output) > 0, "Prediction list should contain at least one sample"
