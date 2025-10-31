from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import pytest
import torch

from foundation_model.models.model_config import KernelRegressionTaskConfig
from foundation_model.models.task_head.kernel_regression import KernelRegressionHead
from foundation_model.scripts.callbacks.prediction_writer import PredictionDataFrameWriter

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_INTEGRATION_DATA_PATHS: Final[list[Path]] = [
    _PROJECT_ROOT / "data/qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet",
    _PROJECT_ROOT / "data/qc_ac_te_mp_dos_reformat_20250615.pd.parquet",
]
_EXAMPLE_CHECKPOINT: Final[Path] = (
    _PROJECT_ROOT / "samples/example_logs/basic_run/basic_experiment_20250702_003437/fit/checkpoints/last.ckpt"
)


@pytest.fixture(scope="module")
def kernel_head() -> KernelRegressionHead:
    config = KernelRegressionTaskConfig(
        name="test_dos_prediction",
        x_dim=[64, 32, 16],
        t_dim=[32, 16, 8],
        kernel_num_centers=5,
        t_encoding_method="fourier",
    )
    return KernelRegressionHead(config)


@pytest.mark.parametrize("batch_sizes", [(4, 3, 3)])
def test_prediction_writer_kernel_regression(tmp_path: Path, kernel_head: KernelRegressionHead, batch_sizes) -> None:
    torch.manual_seed(0)
    writer = PredictionDataFrameWriter(output_path=str(tmp_path), write_interval="epoch")

    batch_predictions = []
    for size in batch_sizes:
        batch_output = torch.randn(size, 1)
        batch_predictions.append(kernel_head.predict(batch_output))

    df = writer._process_predictions(batch_predictions)

    expected_rows = sum(batch_sizes)
    expected_column = "test_dos_prediction_value"

    assert len(df) == expected_rows
    assert expected_column in df.columns
    assert df[expected_column].dtype == float
    assert not df[expected_column].isna().any()

    csv_path = tmp_path / "predictions.csv"
    pickle_path = tmp_path / "predictions.pd.xz"

    df.to_csv(csv_path, index=False)
    df.to_pickle(pickle_path, compression="xz")

    reloaded_csv = pd.read_csv(csv_path)
    reloaded_pickle = pd.read_pickle(pickle_path)

    pd.testing.assert_frame_equal(reloaded_csv, df.reset_index(drop=True))
    pd.testing.assert_frame_equal(reloaded_pickle.reset_index(drop=True), df.reset_index(drop=True))


def test_prediction_writer_multiple_kernel_tasks(tmp_path: Path) -> None:
    torch.manual_seed(1)
    head_one = KernelRegressionHead(
        KernelRegressionTaskConfig(name="dos_prediction", x_dim=[64, 32], t_dim=[16, 8], kernel_num_centers=4)
    )
    head_two = KernelRegressionHead(
        KernelRegressionTaskConfig(name="bandStructure", x_dim=[64, 16], t_dim=[8, 4], kernel_num_centers=3)
    )

    writer = PredictionDataFrameWriter(output_path=str(tmp_path))

    batch_output = torch.randn(5, 1)
    combined_pred = {**head_one.predict(batch_output), **head_two.predict(batch_output)}

    df = writer._process_predictions([combined_pred])

    assert len(df) == 5
    assert "dos_prediction_value" in df.columns
    assert "band_structure_value" in df.columns


def test_prediction_writer_process_predictions_handles_empty_input(tmp_path: Path) -> None:
    writer = PredictionDataFrameWriter(output_path=str(tmp_path))
    df = writer._process_predictions([])
    assert df.empty


@pytest.mark.integration
@pytest.mark.skipif(
    torch.cuda.device_count() < 2
    or not _EXAMPLE_CHECKPOINT.exists()
    or not all(path.exists() for path in _INTEGRATION_DATA_PATHS),
    reason="requires 2 GPUs, example checkpoint, and integration dataset parquet files",
)
def test_prediction_writer_multi_gpu_integration(tmp_path: Path) -> None:
    import lightning as L

    from foundation_model.data.datamodule import CompoundDataModule
    from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
    from foundation_model.models.model_config import KernelRegressionTaskConfig, TaskType

    task_configs = [
        KernelRegressionTaskConfig(
            name="dos",
            type=TaskType.KERNEL_REGRESSION,
            data_column="DOS density (normalized)",
            t_column="DOS energy",
            x_dim=[128, 32, 16],
            t_dim=[32, 16],
            t_encoding_method="fc",
            norm=True,
            residual=False,
            weight=1.0,
            enabled=True,
        )
    ]

    datamodule = CompoundDataModule(
        formula_desc_source=str(_INTEGRATION_DATA_PATHS[0]),
        attributes_source=str(_INTEGRATION_DATA_PATHS[1]),
        task_configs=task_configs,
        batch_size=256,
        num_workers=0,
        predict_idx="all",
        val_split=0.1,
        test_split=0.1,
        random_seed=42,
    )

    model = FlexibleMultiTaskModel.load_from_checkpoint(str(_EXAMPLE_CHECKPOINT), strict=True)

    writer = PredictionDataFrameWriter(output_path=str(tmp_path), write_interval="epoch")

    trainer = L.Trainer(
        accelerator="auto",
        devices=2,
        logger=False,
        callbacks=[writer],
    )

    trainer.predict(model, datamodule=datamodule)

    output_csv = tmp_path / "predictions.csv"
    assert output_csv.exists(), "Distributed prediction should produce a CSV output"

    df = pd.read_csv(output_csv)
    assert len(df) == 48998, "Expected full dataset predictions to be saved without loss"
