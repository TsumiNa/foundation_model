import pytest

from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import ClassificationTaskConfig, KernelRegressionTaskConfig, RegressionTaskConfig
from foundation_model.models.task_head.classification import ClassificationHead
from foundation_model.models.task_head.kernel_regression import KernelRegressionHead
from foundation_model.models.task_head.regression import RegressionHead


def _build_base_model():
    base_task = RegressionTaskConfig(
        name="base_reg",
        dims=[64, 32, 1],
        norm=True,
        residual=False,
    )

    model = FlexibleMultiTaskModel(
        shared_block_dims=[32, 64],
        task_configs=[base_task],
        norm_shared=True,
        residual_shared=False,
    )
    return model


def test_add_and_remove_tasks_dynamically():
    model = _build_base_model()

    assert "base_reg" in model.task_heads
    assert isinstance(model.task_heads["base_reg"], RegressionHead)
    assert model.has_regression

    cls_cfg = ClassificationTaskConfig(
        name="cls_task",
        dims=[64, 32],
        num_classes=3,
        norm=True,
        residual=False,
    )
    returned = model.add_task(cls_cfg)
    assert returned is model

    assert "cls_task" in model.task_heads
    assert isinstance(model.task_heads["cls_task"], ClassificationHead)
    assert model.has_classification
    assert model._get_task_static_weight("cls_task") == pytest.approx(1.0)
    assert "cls_task" in model.task_configs_map
    if model.enable_learnable_loss_balancer:
        assert "cls_task" in model.task_log_sigmas

    returned = model.remove_tasks("cls_task")
    assert returned is model
    assert "cls_task" not in model.task_heads
    assert "cls_task" not in model.task_configs_map
    if model.enable_learnable_loss_balancer:
        assert "cls_task" not in model.task_log_sigmas
    assert not model.has_classification

    kr_cfg = KernelRegressionTaskConfig(
        name="kernel_task",
        x_dim=[64, 32],
        t_dim=[16, 8],
        kernel_num_centers=4,
        t_encoding_method="fc",
        loss_weight=0.5,
    )
    model.add_task(kr_cfg)

    assert "kernel_task" in model.task_heads
    assert isinstance(model.task_heads["kernel_task"], KernelRegressionHead)
    assert model.has_kernel_regression
    assert model._get_task_static_weight("kernel_task") == pytest.approx(0.5)
    if model.enable_learnable_loss_balancer:
        assert "kernel_task" in model.task_log_sigmas

    model.remove_tasks("base_reg")
    assert "base_reg" not in model.task_heads
    assert "base_reg" not in model.task_configs_map
    assert model.has_regression is False


def test_add_task_validates_dimensions():
    model = _build_base_model()

    bad_cls_cfg = ClassificationTaskConfig(
        name="bad_cls",
        dims=[32, 16],  # deposit_dim is 64 -> mismatch
        num_classes=2,
        norm=True,
        residual=False,
    )

    with pytest.raises(ValueError):
        model.add_task(bad_cls_cfg)

    bad_kernel_cfg = KernelRegressionTaskConfig(
        name="bad_kernel",
        x_dim=[32, 16],  # deposit_dim is 64 -> mismatch
        t_dim=[16, 8],
        kernel_num_centers=4,
        t_encoding_method="fc",
    )

    with pytest.raises(ValueError):
        model.add_task(bad_kernel_cfg)


def test_add_and_remove_chain_returns_self():
    model = _build_base_model()

    cls_chain_cfg = ClassificationTaskConfig(
        name="cls_chain",
        dims=[64, 32],
        num_classes=2,
        norm=True,
        residual=False,
    )
    extra_reg_cfg = RegressionTaskConfig(
        name="extra_reg",
        dims=[64, 32, 1],
        norm=True,
        residual=False,
    )

    chained_model = model.add_task(cls_chain_cfg, extra_reg_cfg)
    assert chained_model is model
    assert {"cls_chain", "extra_reg"}.issubset(model.task_heads.keys())

    model.remove_tasks("cls_chain").remove_tasks("extra_reg")
    assert "cls_chain" not in model.task_configs_map
    assert "extra_reg" not in model.task_configs_map
