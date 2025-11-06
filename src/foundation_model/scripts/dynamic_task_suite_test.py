from pathlib import Path
import textwrap

import pytest

from foundation_model.models.model_config import OptimizerConfig
from foundation_model.scripts.dynamic_task_suite import (
    DynamicTaskSuiteRunner,
    SuiteConfig,
    parse_arguments,
)


def _make_suite_config(tmp_path: Path, **overrides) -> SuiteConfig:
    base_kwargs = {
        "descriptor_path": tmp_path / "descriptors.parquet",
        "pretrain_data_path": tmp_path / "pretrain.parquet",
        "finetune_data_path": tmp_path / "finetune.parquet",
        "output_dir": tmp_path / "outputs",
    }
    base_kwargs.update(overrides)
    return SuiteConfig(**base_kwargs)


def test_build_regression_task_with_head_lr(tmp_path: Path):
    config = _make_suite_config(tmp_path, head_lr=2e-3)
    runner = DynamicTaskSuiteRunner(config)

    task_config = runner._build_regression_task("density", "density_column")

    assert task_config.optimizer is not None
    assert isinstance(task_config.optimizer, OptimizerConfig)
    assert task_config.optimizer.lr == pytest.approx(2e-3)
    assert task_config.optimizer.weight_decay == pytest.approx(1e-5)
    assert task_config.dims[0] == runner.encoder_latent_dim
    assert task_config.dims[-1] == 1


def test_build_regression_task_without_head_lr(tmp_path: Path):
    config = _make_suite_config(tmp_path)
    runner = DynamicTaskSuiteRunner(config)

    task_config = runner._build_regression_task("density", "density_column")

    assert task_config.optimizer is None
    assert task_config.dims[0] == runner.encoder_latent_dim


def test_parse_arguments_respects_overrides(tmp_path: Path):
    args = [
        "--descriptor-path",
        str(tmp_path / "descriptors.parquet"),
        "--pretrain-data-path",
        str(tmp_path / "pretrain.parquet"),
        "--finetune-data-path",
        str(tmp_path / "finetune.parquet"),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--head-lr",
        "5e-4",
        "--disable-deposit-layer",
    ]

    suite_config = parse_arguments(args)

    assert suite_config.head_lr == pytest.approx(5e-4)
    assert suite_config.use_deposit_layer is False


def test_parse_arguments_enable_deposit_layer(tmp_path: Path):
    args = [
        "--descriptor-path",
        str(tmp_path / "descriptors.parquet"),
        "--pretrain-data-path",
        str(tmp_path / "pretrain.parquet"),
        "--finetune-data-path",
        str(tmp_path / "finetune.parquet"),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--enable-deposit-layer",
    ]

    suite_config = parse_arguments(args)

    assert suite_config.use_deposit_layer is True


def test_parse_arguments_from_toml_config(tmp_path: Path):
    config_path = tmp_path / "suite.toml"
    descriptor_path = tmp_path / "desc.parquet"
    pretrain_path = tmp_path / "pre.parquet"
    finetune_path = tmp_path / "fin.parquet"
    output_dir = tmp_path / "out"
    config_path.write_text(
        textwrap.dedent(
            f"""
            descriptor_path = "{descriptor_path}"
            pretrain_data_path = "{pretrain_path}"
            finetune_data_path = "{finetune_path}"
            output_dir = "{output_dir}"
            head_lr = 0.0005
            use_deposit_layer = false
            shared_block_dims = [190, 256, 128]
            encoder_config = { type = "transformer", d_model = 256, nhead = 4 }
            pretrain_tasks = ["density", "Rg"]
            finetune_tasks = ["density"]
            """
        ).strip(),
        encoding="utf-8",
    )

    suite_config = parse_arguments(["--config-file", str(config_path)])

    assert suite_config.descriptor_path == descriptor_path
    assert suite_config.head_lr == pytest.approx(5e-4)
    assert suite_config.use_deposit_layer is False
    assert suite_config.pretrain_tasks == ["density", "Rg"]
    assert suite_config.finetune_tasks == ["density"]
    assert suite_config.encoder_config.type.value == "transformer"
    assert suite_config.encoder_config.latent_dim == 256


def test_config_file_overridden_by_cli(tmp_path: Path):
    config_path = tmp_path / "suite.toml"
    descriptor_path = tmp_path / "desc.parquet"
    pretrain_path = tmp_path / "pre.parquet"
    finetune_path = tmp_path / "fin.parquet"
    output_dir = tmp_path / "out"
    config_path.write_text(
        textwrap.dedent(
            f"""
            descriptor_path = "{descriptor_path}"
            pretrain_data_path = "{pretrain_path}"
            finetune_data_path = "{finetune_path}"
            output_dir = "{output_dir}"
            head_hidden_dim = 128
            head_lr = 0.0005
            use_deposit_layer = false
            """
        ).strip(),
        encoding="utf-8",
    )

    suite_config = parse_arguments(
        [
            "--config-file",
            str(config_path),
            "--head-hidden-dim",
            "64",
            "--enable-deposit-layer",
        ]
    )

    assert suite_config.head_hidden_dim == 64
    assert suite_config.head_lr == pytest.approx(5e-4)
    assert suite_config.use_deposit_layer is True
