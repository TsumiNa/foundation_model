# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for FlexibleMultiTaskModel, including integration with CompoundDataModule.
"""

from pathlib import Path  # To use Path objects for directory manipulation
from types import SimpleNamespace

import lightning as L
import numpy as np
import pandas as pd
from typing import Any, cast

from torchmetrics import R2Score

import pytest
from dataclasses import dataclass
import torch
import torch.nn as nn
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel, OptimizationTarget
from foundation_model.models.task_head.base import BaseTaskHead
from foundation_model.models.model_config import (
    TaskConfigType,
    BaseEncoderConfig,
    ClassificationTaskConfig,
    EncoderType,
    KernelRegressionTaskConfig,
    MLPEncoderConfig,
    OptimizerConfig,
    RegressionTaskConfig,
    TaskType,
    TransformerEncoderConfig,
)
from foundation_model.models.task_head.classification import ClassificationHead
from foundation_model.models.task_head.regression import RegressionHead

# from ...data.dataset import CompoundDataset


# --- Fixtures ---


@pytest.fixture
def model_config_mixed_tasks():
    """
    Provides a configuration for FlexibleMultiTaskModel with regression and classification tasks.
    Focuses on a simple setup without structure fusion or self-supervised learning.
    """
    shared_dims = [64, 128, 256]  # Input -> hidden -> latent
    latent_dim = shared_dims[-1]  # Tanh-activated latent representation (input to task heads)

    task_configs_list = [
        RegressionTaskConfig(
            name="regr_task_1",
            type=TaskType.REGRESSION,
            dims=[latent_dim, 64, 1],
            data_column="regr_task_1",
            optimizer=OptimizerConfig(lr=1e-4, scheduler_enabled=False),
            loss_weight=1.0,
        ),
        ClassificationTaskConfig(
            name="clf_task_1",
            type=TaskType.CLASSIFICATION,
            dims=[latent_dim, 64, 3],
            data_column="clf_task_1_classification_value",
            num_classes=3,
            optimizer=OptimizerConfig(lr=1e-4, scheduler_enabled=False),
            loss_weight=1.0,
        ),
        RegressionTaskConfig(
            name="regr_task_2",
            type=TaskType.REGRESSION,
            dims=[latent_dim, 32, 2],
            data_column="regr_task_2",
            optimizer=OptimizerConfig(lr=1e-4, scheduler_enabled=False),
            loss_weight=0.5,
        ),
    ]

    config_dict = {
        "shared_block_dims": shared_dims,
        "task_configs": task_configs_list,
        "encoder_config": MLPEncoderConfig(hidden_dims=shared_dims, norm=True, residual=False),
        "shared_block_optimizer": OptimizerConfig(lr=1e-3, scheduler_enabled=False),
    }
    return SimpleNamespace(**config_dict)


@pytest.fixture
def sample_batch_mixed_tasks(model_config_mixed_tasks):
    """
    Generates a sample batch for mixed regression and classification tasks.
    Output format: (x_formula, y_dict_batch, task_masks_batch, temps_batch)
    """
    batch_size = 4
    config = model_config_mixed_tasks
    formula_input_dim = config.shared_block_dims[0]

    x_formula = torch.randn(batch_size, formula_input_dim)
    y_dict_batch = {}
    task_masks_batch = {}

    for i, task_cfg in enumerate(config.task_configs):
        # Create a [B, 1] boolean mask for each task
        # For simplicity, mask one different sample for each task if batch_size allows
        mask_task = torch.ones(batch_size, 1, dtype=torch.bool)
        if batch_size > 0:
            mask_idx = i % batch_size
            mask_task[mask_idx, 0] = False
        task_masks_batch[task_cfg.name] = mask_task

        if task_cfg.type == TaskType.REGRESSION:
            assert isinstance(task_cfg, RegressionTaskConfig)
            task_output_dim = task_cfg.dims[-1]  # Last element of dims is output_dim for the head
            y_task = torch.randn(batch_size, task_output_dim)
        elif task_cfg.type == TaskType.CLASSIFICATION:
            assert isinstance(task_cfg, ClassificationTaskConfig)
            # Target for classification is class indices (long tensor), shape (B,)
            y_task = torch.randint(0, task_cfg.num_classes, (batch_size,), dtype=torch.long)
        else:
            # This fixture is not designed for sequence tasks
            raise ValueError(f"Unsupported task type {task_cfg.type} in sample_batch_mixed_tasks fixture.")
        y_dict_batch[task_cfg.name] = y_task

    # temps_batch is an empty dict as no sequence tasks are included
    temps_batch: dict[str, list[torch.Tensor]] = {}
    return (x_formula, y_dict_batch, task_masks_batch, temps_batch)


# --- Unit Tests for Model Components ---


def test_model_initialization(model_config_mixed_tasks):
    """Test model initialization with mixed regression and classification tasks."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )

    assert model.encoder is not None, "Encoder should be initialized"
    assert hasattr(model.encoder, "shared"), "Encoder should have 'shared' attribute"

    assert isinstance(model.task_heads, nn.ModuleDict), "task_heads should be an nn.ModuleDict"

    enabled_tasks_in_config = [tc for tc in config.task_configs if tc.enabled]
    assert len(model.task_heads) == len(enabled_tasks_in_config), (
        f"Expected {len(enabled_tasks_in_config)} task heads, got {len(model.task_heads)}"
    )

    for task_cfg in enabled_tasks_in_config:
        assert task_cfg.name in model.task_heads, f"Task head {task_cfg.name} not found in model.task_heads"
        head_module = model.task_heads[task_cfg.name]
        if task_cfg.type == TaskType.REGRESSION:
            assert isinstance(head_module, RegressionHead), (
                f"Task {task_cfg.name} should be RegressionHead, got {type(head_module)}"
            )
        elif task_cfg.type == TaskType.CLASSIFICATION:
            assert isinstance(head_module, ClassificationHead), (
                f"Task {task_cfg.name} should be ClassificationHead, got {type(head_module)}"
            )
        # No sequence tasks in this fixture

    # Automatic optimization, deliberately. The model returns one optimizer, so Lightning owns the
    # backward pass, the optimizer step AND the scheduler step. It was manual only because the old
    # per-group optimizers made Lightning refuse to drive them, and that split is what let the
    # scheduler be stepped per batch instead of per epoch (#45).
    assert model.automatic_optimization, "FlexibleMultiTaskModel must use automatic optimization"


def test_model_forward_pass(model_config_mixed_tasks, sample_batch_mixed_tasks):
    """Test the forward pass for mixed regression and classification predictions."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    model.eval()  # Set model to evaluation mode

    x_formula, _, _, temps_batch = sample_batch_mixed_tasks  # y_dict and masks not needed for forward pass directly

    # Forward pass expects x_formula (or (x_formula, x_struct)) and task_sequence_data_batch
    output = model(x_formula, t_sequences=temps_batch)

    assert isinstance(output, dict), "Output should be a dictionary"

    enabled_tasks_in_config = [tc for tc in config.task_configs if tc.enabled]
    assert len(output.keys()) == len(enabled_tasks_in_config), (
        f"Expected {len(enabled_tasks_in_config)} keys in output, got {len(output.keys())}"
    )

    for task_cfg in enabled_tasks_in_config:
        assert task_cfg.name in output, f"Output dictionary should contain '{task_cfg.name}' key"
        task_pred = output[task_cfg.name]
        assert isinstance(task_pred, torch.Tensor), f"{task_cfg.name} predictions should be a Tensor"

        batch_size = x_formula.shape[0]
        if task_cfg.type == TaskType.REGRESSION:
            assert isinstance(task_cfg, RegressionTaskConfig)
            expected_task_output_dim = task_cfg.dims[-1]
            expected_shape = (batch_size, expected_task_output_dim)
            assert task_pred.shape == expected_shape, (
                f"{task_cfg.name} (Regression) predictions shape mismatch. Expected {expected_shape}, got {task_pred.shape}"
            )
        elif task_cfg.type == TaskType.CLASSIFICATION:
            assert isinstance(task_cfg, ClassificationTaskConfig)
            expected_num_classes = task_cfg.num_classes
            expected_shape = (
                batch_size,
                expected_num_classes,
            )  # Output of classification head is typically (B, num_classes) logits
            assert task_pred.shape == expected_shape, (
                f"{task_cfg.name} (Classification) predictions shape mismatch. Expected {expected_shape}, got {task_pred.shape}"
            )


def test_model_training_step(model_config_mixed_tasks, sample_batch_mixed_tasks, mocker):
    """Test the training_step for mixed regression and classification tasks."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    model.train()  # Set model to training mode

    # Mock trainer and strategy for manual_backward
    mock_trainer = mocker.MagicMock(spec=L.Trainer)
    mock_strategy = mocker.MagicMock()
    mock_strategy.backward = mocker.MagicMock()
    mock_trainer.strategy = mock_strategy
    model.trainer = mock_trainer

    # Mock optimizers
    # The model's configure_optimizers returns a list of dicts or optimizers
    # For simplicity in this unit test, we'll mock the self.optimizers() call
    # which is what training_step uses internally.
    # Number of optimizers = 1 (shared) + num_enabled_tasks
    num_enabled_tasks = sum(1 for tc in config.task_configs if tc.enabled)
    # No optimizer mocking: under automatic optimization training_step neither fetches optimizers
    # nor steps them. It computes the objective and hands it back, and that is the whole contract.

    mock_log_dict = mocker.patch.object(model, "log_dict")
    mock_log = mocker.patch.object(model, "log")

    # sample_batch_mixed_tasks provides (x_formula, y_dict_batch, task_masks_batch, temps_batch)
    # This matches the expected input for training_step
    loss = model.training_step(sample_batch_mixed_tasks, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Loss should be a Tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    # It must carry the graph: Lightning backwards exactly what this returns, so a detached loss
    # would train nothing while still looking like a healthy step.
    assert loss.requires_grad, "the returned loss is what Lightning backwards, so it needs a graph"

    # The model must NOT touch backward or the optimizer itself any more.
    mock_strategy.backward.assert_not_called()

    # Check that optimizer steps and zero_grads were called
    mock_log_dict.assert_called()
    logged_metrics = mock_log_dict.call_args[0][0]

    assert "train_final_supervised_loss" in logged_metrics

    enabled_tasks_in_config = [tc for tc in config.task_configs if tc.enabled]
    for task_cfg in enabled_tasks_in_config:
        assert f"train_{task_cfg.name}_raw_loss" in logged_metrics
        assert isinstance(logged_metrics[f"train_{task_cfg.name}_raw_loss"], torch.Tensor)
        assert f"train_{task_cfg.name}_final_loss_contrib" in logged_metrics
        assert f"train_{task_cfg.name}_static_weight" in logged_metrics
        assert isinstance(logged_metrics[f"train_{task_cfg.name}_static_weight"], torch.Tensor)

    assert "train_mfm_loss" not in logged_metrics
    assert "train_contrastive_loss" not in logged_metrics
    assert "train_cross_recon_loss" not in logged_metrics
    assert "train_modality_dropout_applied" not in logged_metrics


def test_model_validation_step(model_config_mixed_tasks, sample_batch_mixed_tasks, mocker):
    """Test the validation_step for mixed regression and classification tasks."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    model.eval()  # Set model to evaluation mode

    mock_log_dict = mocker.patch.object(model, "log_dict")
    mock_log = mocker.patch.object(model, "log")

    # validation_step logs metrics via self.log_dict and returns nothing; the `-> None`
    # annotation is what enforces that now, so asserting on the value is both redundant and
    # rejected by the type checker.
    model.validation_step(sample_batch_mixed_tasks, batch_idx=0)

    mock_log_dict.assert_called()
    logged_metrics = mock_log_dict.call_args[0][0]

    assert "val_final_supervised_loss" in logged_metrics

    enabled_tasks_in_config = [tc for tc in config.task_configs if tc.enabled]
    for task_cfg in enabled_tasks_in_config:
        assert f"val_{task_cfg.name}_raw_loss" in logged_metrics
        assert isinstance(logged_metrics[f"val_{task_cfg.name}_raw_loss"], torch.Tensor)
        assert f"val_{task_cfg.name}_final_loss_contrib" in logged_metrics
        assert f"val_{task_cfg.name}_static_weight" in logged_metrics
        assert isinstance(logged_metrics[f"val_{task_cfg.name}_static_weight"], torch.Tensor)

    assert "val_mfm_loss" not in logged_metrics
    assert "val_contrastive_loss" not in logged_metrics
    assert "val_cross_recon_loss" not in logged_metrics


def test_model_predict_step_all_tasks(model_config_mixed_tasks, sample_batch_mixed_tasks):
    """Test the predict_step for all enabled tasks."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    model.eval()  # Set model to evaluation mode

    x_formula, y_dict, task_masks, temps_batch = sample_batch_mixed_tasks

    # predict_step expects a batch tuple: (x_input, y_dict, masks_dict, temps_dict)
    # For prediction, y_dict and masks_dict can be the ones from the sample batch or None.
    # x_input for predict_step is typically just x_formula.
    predict_batch_tuple = (x_formula, y_dict, task_masks, temps_batch)

    output = model.predict_step(predict_batch_tuple, batch_idx=0)

    assert isinstance(output, dict), "Predict output should be a dictionary"

    enabled_tasks_in_config = [tc for tc in config.task_configs if tc.enabled]

    # Each task head's predict method might return multiple keys (e.g., _value, _logits, _labels)
    # So, the number of keys in output might be >= number of tasks.
    # We will check for specific expected keys.

    for task_cfg in enabled_tasks_in_config:
        task_name_snake = task_cfg.name.replace("-", "_")  # Basic snake_case conversion

        if task_cfg.type == TaskType.REGRESSION:
            assert isinstance(task_cfg, RegressionTaskConfig)
            expected_key_value = f"{task_name_snake}_value"
            assert expected_key_value in output, f"Predict output should contain key '{expected_key_value}'"
            raw_value = output[expected_key_value]
            pred_value = raw_value if isinstance(raw_value, torch.Tensor) else torch.as_tensor(raw_value)
            expected_shape = (x_formula.shape[0], task_cfg.dims[-1])
            assert pred_value.shape == expected_shape, (
                f"Shape mismatch for {expected_key_value}. Expected {expected_shape}, got {pred_value.shape}"
            )

        elif task_cfg.type == TaskType.CLASSIFICATION:
            assert isinstance(task_cfg, ClassificationTaskConfig)
            expected_key_proba = f"{task_name_snake}_proba"
            expected_key_label = f"{task_name_snake}_label"

            assert expected_key_proba in output, f"Predict output should contain key '{expected_key_proba}'"
            raw_proba_value = output[expected_key_proba]

            proba_value = (
                raw_proba_value if isinstance(raw_proba_value, torch.Tensor) else torch.as_tensor(raw_proba_value)
            )
            expected_proba_shape = (x_formula.shape[0], task_cfg.num_classes)
            assert proba_value.shape == expected_proba_shape, (
                f"Shape mismatch for {expected_key_proba}. Expected {expected_proba_shape}, got {proba_value.shape}"
            )

            assert expected_key_label in output, f"Predict output should contain key '{expected_key_label}'"
            raw_label_value = output[expected_key_label]

            label_value = (
                raw_label_value if isinstance(raw_label_value, torch.Tensor) else torch.as_tensor(raw_label_value)
            )
            expected_label_shape = (x_formula.shape[0],)
            assert label_value.shape == expected_label_shape, (
                f"Shape mismatch for {expected_key_label}. Expected {expected_label_shape}, got {label_value.shape}"
            )
            assert label_value.dtype == torch.long, f"{expected_key_label} should be of type torch.long"


def test_model_configure_optimizers(model_config_mixed_tasks):
    """ONE AdamW with one param group per role, and at most one ReduceLROnPlateau.

    The previous shape — one optimizer and one scheduler per group — is what forced manual
    optimization, since Lightning drives at most one optimizer on its own. Collapsing it gives up
    nothing that was reachable: every group's scheduler came from the same [training.scheduler]
    block and monitored the same metric, so the N schedulers stepped in lockstep. What genuinely
    varies per group — lr, weight_decay, min_lr — survives as param groups and a list-valued
    min_lr.
    """
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=OptimizerConfig(lr=1e-3, weight_decay=1e-2, min_lr=1e-6),
    )
    for task_config in model.task_configs_map.values():
        task_config.optimizer = OptimizerConfig(lr=5e-3, weight_decay=1e-5, min_lr=1e-7)

    result = cast(dict[str, Any], model.configure_optimizers())
    optimizer = result["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW), "exactly one AdamW, not one per group"

    # One group for the encoder plus one per head with trainable parameters.
    expected_groups = 1 + len(
        [n for n in model.task_heads if any(p.requires_grad for p in model._head(n).parameters())]
    )
    assert len(optimizer.param_groups) == expected_groups

    # Per-group hyper-parameters must survive the collapse — this is the whole reason param groups
    # exist rather than a single flat parameter list.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-2)
    assert all(g["lr"] == pytest.approx(5e-3) for g in optimizer.param_groups[1:])
    assert all(g["weight_decay"] == pytest.approx(1e-5) for g in optimizer.param_groups[1:])

    # Every trainable parameter is optimized exactly once — no gaps, no double counting.
    optimized = [id(p) for g in optimizer.param_groups for p in g["params"]]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(optimized) == len(set(optimized)), "a parameter appears in more than one group"
    assert set(optimized) == trainable

    # One scheduler, driven by Lightning at epoch end — the model no longer steps it, which is
    # what makes `patience` count epochs by construction rather than by convention.
    lr_config = result["lr_scheduler"]
    assert isinstance(lr_config["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert lr_config["interval"] == "epoch"
    assert lr_config["monitor"] == "train_final_loss_epoch"
    # min_lr stays per group: it is a floor on each group's own learning rate.
    assert lr_config["scheduler"].min_lrs == [1e-6] + [1e-7] * (expected_groups - 1)


def test_configure_optimizers_rejects_groups_that_disagree_on_scheduler_policy(model_config_mixed_tasks):
    """A single optimizer carries a single scheduler, so a mixed policy has to be an error.

    It cannot arise from configuration — one [training.scheduler] block feeds every group — so a
    model that reaches here with mixed policies was assembled in Python. Silently honouring one
    group's patience while discarding another's is the class of invisible divergence that made the
    v1 campaign unusable.
    """
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=OptimizerConfig(lr=1e-3, patience=5, min_lr=1e-6),
    )
    for task_config in model.task_configs_map.values():
        task_config.optimizer = OptimizerConfig(lr=5e-3, patience=20, min_lr=1e-6)

    with pytest.raises(ValueError, match="must share one LR-scheduler policy"):
        model.configure_optimizers()


def test_configure_optimizers_returns_a_bare_optimizer_when_the_scheduler_is_off(model_config_mixed_tasks):
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=OptimizerConfig(lr=1e-3, scheduler_enabled=False),
    )
    for task_config in model.task_configs_map.values():
        task_config.optimizer = OptimizerConfig(lr=5e-3, scheduler_enabled=False)

    assert isinstance(model.configure_optimizers(), torch.optim.AdamW)


def test_scheduler_policy_is_only_compared_where_it_decides_something(model_config_mixed_tasks):
    """With the scheduler off everywhere, mode/factor/patience/monitor decide nothing.

    No scheduler is constructed in that case, so those fields are dead values — rejecting the model
    over them would refuse to build an optimizer on the strength of settings no code path reads.
    The inference-only builder in workflows/_engine.py assembles exactly this shape.
    """
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=OptimizerConfig(lr=1e-3, scheduler_enabled=False, patience=5, factor=0.5),
    )
    for task_config in model.task_configs_map.values():
        # Same switch, wildly different dead values.
        task_config.optimizer = OptimizerConfig(lr=5e-3, scheduler_enabled=False, patience=99, factor=0.1, mode="max")

    assert isinstance(model.configure_optimizers(), torch.optim.AdamW)


def test_configure_optimizers_rejects_a_scheduler_that_is_on_for_some_groups_only(model_config_mixed_tasks):
    """Mixed on/off is still an error: there the settings do decide something.

    One optimizer carries one scheduler, so "annealed" and "constant" cannot both be honoured —
    and picking either silently is how a group ends up training under a schedule nobody chose.
    """
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=OptimizerConfig(lr=1e-3, scheduler_enabled=False),
    )
    for task_config in model.task_configs_map.values():
        task_config.optimizer = OptimizerConfig(lr=5e-3, min_lr=1e-6, scheduler_enabled=True)

    with pytest.raises(ValueError, match="must agree on whether the LR scheduler is enabled"):
        model.configure_optimizers()


# --- Fixtures for DataModule and Trainer Integration ---


@pytest.fixture
def dummy_compound_datamodule(model_config_mixed_tasks, tmp_path):
    """
    Creates a CompoundDataModule instance with dummy data for mixed tasks.
    Uses tmp_path for any file-based operations if CompoundDataModule were to save/load intermediate.
    For this version, we'll pass DataFrames directly.
    """
    config = model_config_mixed_tasks
    num_samples = 20  # e.g., 10 train, 5 val, 5 test
    batch_size = 4
    formula_input_dim = config.shared_block_dims[0]

    # Create dummy formula descriptors
    formula_df = create_dummy_dataframe(num_samples, formula_input_dim, index_prefix="s")

    # Create dummy attributes DataFrame
    # It needs columns for each task's target and a 'split' column.
    # One column per task, of whatever dtype that task needs — float arrays, int label arrays,
    # lists of lists for multi-output regression. Inference would fix it to the first one.
    attributes_data: dict[str, Any] = {}
    sample_indices = formula_df.index

    for task_cfg in config.task_configs:
        if task_cfg.type == TaskType.REGRESSION:
            assert isinstance(task_cfg, RegressionTaskConfig)
            # Regression target(s)
            num_outputs = task_cfg.dims[-1]
            if num_outputs == 1:
                attributes_data[task_cfg.name] = np.random.rand(num_samples, num_outputs).squeeze()
            else:
                # For multi-output regression, store as list of lists for pandas DataFrame
                multi_output_data = np.random.rand(num_samples, num_outputs)
                attributes_data[task_cfg.name] = [list(row) for row in multi_output_data]
        elif task_cfg.type == TaskType.CLASSIFICATION:
            assert isinstance(task_cfg, ClassificationTaskConfig)
            # Classification target (indices)
            # Column name should match what CompoundDataset expects for classification values
            col_name = f"{task_cfg.name}_classification_value"
            attributes_data[col_name] = np.random.randint(0, task_cfg.num_classes, num_samples)

    attributes_df = pd.DataFrame(attributes_data, index=sample_indices)

    # Add 'split' column for deterministic train/val/test splits
    # e.g., 60% train, 20% val, 20% test
    splits = []
    num_train = int(num_samples * 0.6)
    num_val = int(num_samples * 0.2)
    num_test = num_samples - num_train - num_val

    splits.extend(["train"] * num_train)
    splits.extend(["val"] * num_val)
    splits.extend(["test"] * num_test)
    np.random.shuffle(splits)  # Shuffle to distribute, though CompoundDataModule might re-split
    # Using 'split' column directly is more robust for tests.

    # Ensure splits array matches num_samples if rounding caused issues
    if len(splits) < num_samples:
        splits.extend(["train"] * (num_samples - len(splits)))  # Add remaining to train
    attributes_df["split"] = splits[:num_samples]

    def descriptor_fn(compositions):
        present = [c for c in compositions if c in formula_df.index]
        return formula_df.loc[present]

    # Each supervised task reads its own data_column from the shared attributes frame
    # (composition-indexed). AUTOENCODER tasks need no frame.
    task_frames = {cfg.name: attributes_df for cfg in config.task_configs if cfg.type != TaskType.AUTOENCODER}

    dm = CompoundDataModule(
        task_configs=config.task_configs,
        descriptor_fn=descriptor_fn,
        task_frames=task_frames,
        batch_size=batch_size,
        num_workers=0,
    )
    dm.setup()  # Call setup to prepare datasets
    return dm


# --- Integration Test with Trainer ---


def test_trainer_integration_mixed_tasks(model_config_mixed_tasks, dummy_compound_datamodule, tmp_path):
    """
    Test the model with pytorch_lightning.Trainer and CompoundDataModule
    for mixed regression and classification tasks, including logger functionality.
    """
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )

    # Using integer for version, and a slightly different name for clarity
    csv_logger = CSVLogger(save_dir=str(tmp_path), name="pytest_csv_logs", version=0)
    trainer = L.Trainer(
        logger=csv_logger,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    # The main assertion is that this runs without errors.
    try:
        trainer.fit(model, datamodule=dummy_compound_datamodule)
        csv_logger.finalize("success")  # Ensure logs are flushed

        # Construct the expected path based on CSVLogger's default behavior with name and version
        # CSVLogger creates save_dir / name / version_X / metrics.csv
        # Here, save_dir is tmp_path, name is "pytest_csv_logs", version is 0.
        # Use the logger's reported log_dir to be more robust
        actual_log_dir = Path(csv_logger.log_dir)
        log_file_path = actual_log_dir / "metrics.csv"

        # --- Start Debugging Prints ---
        print("\n--- Debugging CSVLogger ---")
        print(f"tmp_path: {tmp_path}")
        print(f"CSVLogger save_dir: {csv_logger.save_dir}")
        print(f"CSVLogger name: {csv_logger.name}")
        print(f"CSVLogger version: {csv_logger.version}")
        print(f"CSVLogger reported log_dir: {csv_logger.log_dir}")
        print(f"Constructed actual_log_dir: {actual_log_dir}")
        print(f"Constructed log_file_path: {log_file_path}")

        if tmp_path.exists():
            print(f"Contents of tmp_path ({tmp_path}):")
            for item in tmp_path.rglob("*"):
                print(f"  {item}")
        else:
            print(f"tmp_path ({tmp_path}) does not exist.")

        if actual_log_dir.exists():
            print(f"Contents of actual_log_dir ({actual_log_dir}):")
            for item in actual_log_dir.glob("*"):
                print(f"  {item}")
        else:
            print(f"actual_log_dir ({actual_log_dir}) does not exist (before assertion).")
        # --- End Debugging Prints ---

        assert log_file_path.is_file(), f"metrics.csv not found at {log_file_path}"

        # Read the CSV and check its content
        metrics_df = pd.read_csv(log_file_path)
        assert not metrics_df.empty, "metrics.csv is empty"

        # When on_epoch=True (default for training_step), CSVLogger appends "_epoch"
        # For on_step=True, it does not append "_step" but uses the "step" column.
        # Validation step logs with on_epoch=True, on_step=False by default.
        expected_train_cols = ["epoch", "step", "train_final_loss_epoch", "train_final_supervised_loss_epoch"]
        expected_val_cols = ["val_final_loss", "val_final_supervised_loss"]

        enabled_tasks_in_config = [tc for tc in config.task_configs if tc.enabled]
        for task_cfg in enabled_tasks_in_config:
            expected_train_cols.append(f"train_{task_cfg.name}_raw_loss_epoch")
            expected_train_cols.append(f"train_{task_cfg.name}_final_loss_contrib_epoch")
            expected_val_cols.append(f"val_{task_cfg.name}_raw_loss")
            expected_val_cols.append(f"val_{task_cfg.name}_final_loss_contrib")

        all_expected_cols = set(expected_train_cols + expected_val_cols)

        # Check if all expected columns are present (some might be NaN if not logged in a particular step/epoch)
        for col in all_expected_cols:
            assert col in metrics_df.columns, f"Expected column '{col}' not found in metrics.csv"

        # With fast_dev_run, we expect at least one row for training metrics and one for validation metrics
        # (though they might be combined or logged at different steps by CSVLogger)
        # A simple check for non-empty columns for key metrics:
        assert metrics_df["train_final_loss_epoch"].notna().any(), "train_final_loss_epoch column has no data"
        assert metrics_df["val_final_loss"].notna().any(), "val_final_loss column has no data"

        # If fast_dev_run is True, it also runs validation and test loops if defined.
        # And predict_loop if predict_dataloaders are available.

        # Explicitly test predict if predict_dataloader is set up by dm.setup()
        if dummy_compound_datamodule.predict_dataloader() is not None:
            predictions = trainer.predict(
                model, datamodule=dummy_compound_datamodule, ckpt_path=None
            )  # ckpt_path=None for fresh predict
            assert predictions is not None
            assert isinstance(predictions, list)
            if len(predictions) > 0:
                assert isinstance(predictions[0], dict)  # Each item in list is output of predict_step for a batch
    except Exception as e:
        pytest.fail(f"Trainer integration test failed: {e}")


def test_model_predict_step_specific_tasks(model_config_mixed_tasks, sample_batch_mixed_tasks, mocker):
    """Test the predict_step with specific tasks requested."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    model.eval()
    x_formula, y_dict, task_masks, temps_batch = sample_batch_mixed_tasks
    predict_batch_tuple = (x_formula, y_dict, task_masks, temps_batch)

    # Mock logger to check warnings
    mock_logger = mocker.patch("foundation_model.models.flexible_multi_task_model.logger")

    # 1. Predict a single existing task
    task_to_predict_single = [config.task_configs[0].name]
    output_single = model.predict_step(predict_batch_tuple, batch_idx=0, tasks_to_predict=task_to_predict_single)
    assert isinstance(output_single, dict)
    # Check that only keys related to task_to_predict_single are present
    for key in output_single.keys():
        assert task_to_predict_single[0].replace("-", "_") in key
    # Verify that other tasks are not in the output
    if len(config.task_configs) > 1:
        other_task_name_snake = config.task_configs[1].name.replace("-", "_")
        assert not any(other_task_name_snake in key for key in output_single.keys()), (
            f"Predictions for '{other_task_name_snake}' should not be in output when only '{task_to_predict_single[0]}' is requested."
        )

    # 2. Predict multiple existing tasks
    if len(config.task_configs) > 1:
        tasks_to_predict_multiple = [config.task_configs[0].name, config.task_configs[1].name]
        output_multiple = model.predict_step(
            predict_batch_tuple, batch_idx=0, tasks_to_predict=tasks_to_predict_multiple
        )
        assert isinstance(output_multiple, dict)
        # Check that keys related to both tasks are present
        task0_snake = tasks_to_predict_multiple[0].replace("-", "_")
        task1_snake = tasks_to_predict_multiple[1].replace("-", "_")
        assert any(task0_snake in key for key in output_multiple.keys())
        assert any(task1_snake in key for key in output_multiple.keys())
        if len(config.task_configs) > 2:  # If there's a third task, ensure it's not predicted
            third_task_snake = config.task_configs[2].name.replace("-", "_")
            assert not any(third_task_snake in key for key in output_multiple.keys())

    # 3. Predict a non-existent task (should log warning and return empty dict or only valid tasks)
    tasks_to_predict_non_existent = ["non_existent_task", config.task_configs[0].name]
    output_non_existent = model.predict_step(
        predict_batch_tuple, batch_idx=0, tasks_to_predict=tasks_to_predict_non_existent
    )
    mock_logger.warning.assert_any_call(
        "Task 'non_existent_task' requested for prediction but not found or not enabled in the model. Skipping."
    )
    # Output should only contain predictions for the valid task
    task0_snake = config.task_configs[0].name.replace("-", "_")
    assert all(task0_snake in key for key in output_non_existent.keys())
    assert not any("non_existent_task" in key for key in output_non_existent.keys())

    # 4. Predict with an empty list (should return empty dict)
    output_empty_list = model.predict_step(predict_batch_tuple, batch_idx=0, tasks_to_predict=[])
    assert isinstance(output_empty_list, dict)
    assert len(output_empty_list) == 0


def test_model_registered_tasks_info_property(model_config_mixed_tasks):
    """Test the registered_tasks_info property."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )

    df_info = model.registered_tasks_info
    assert isinstance(df_info, pd.DataFrame)
    assert list(df_info.columns) == ["name", "type", "enabled"]
    assert len(df_info) == len(config.task_configs)

    for i, task_cfg_from_model in enumerate(config.task_configs):
        assert df_info.loc[i, "name"] == task_cfg_from_model.name
        assert df_info.loc[i, "type"] == task_cfg_from_model.type.value  # Enum value
        assert df_info.loc[i, "enabled"] == task_cfg_from_model.enabled


def test_r2_metric_updates_respect_masks(model_config_mixed_tasks):
    """Validate that masked samples do not influence the logged R² metric."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    preds = torch.tensor([[1.0], [3.0], [5.0]])
    targets = torch.tensor([[1.0], [10.0], [5.0]])
    sample_mask = torch.tensor([[1], [0], [1]], dtype=torch.bool)

    model._update_r2_metric(
        stage="val",
        task_name="regr_task_1",
        preds=preds,
        targets=targets,
        sample_mask=sample_mask,
    )

    assert "regr_task_1" in model._metrics_updated["val"]
    computed = cast(R2Score, model.val_r2_metrics["regr_task_1"]).compute()
    assert torch.isclose(computed, torch.tensor(1.0))


@pytest.mark.parametrize("stage", ["val", "test"])
def test_r2_metric_uses_each_tasks_own_tensors(model_config_mixed_tasks, sample_batch_mixed_tasks, mocker, stage):
    """Every task's R² must be fed that task's own predictions, not the last task's.

    The two loops in ``validation_step`` / ``test_step`` are separate: losses are computed in the
    first, R² updated in the second. Reading the first loop's tensors from the second scored every
    task against whichever task iterated last — silently wrong when the shapes happened to agree
    (``regr_task_1`` is (B, 1), ``regr_task_2`` is (B, 2)) and a hard ``RuntimeError`` from
    torchmetrics when they did not (a classification head last).
    """
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    model.eval()
    mocker.patch.object(model, "log_dict")
    mocker.patch.object(model, "log")
    spy = mocker.spy(model, "_update_r2_metric")

    step = model.validation_step if stage == "val" else model.test_step
    step(sample_batch_mixed_tasks, batch_idx=0)

    seen = {call.kwargs["task_name"]: call.kwargs for call in spy.call_args_list}
    batch_size = sample_batch_mixed_tasks[0].shape[0]
    # Head output widths differ per task (dims[-1]), which is what pins the tensors to the task.
    assert seen["regr_task_1"]["preds"].shape == (batch_size, 1)
    assert seen["regr_task_2"]["preds"].shape == (batch_size, 2)
    assert seen["clf_task_1"]["preds"].shape == (batch_size, 3)
    for name, kwargs in seen.items():
        expected = sample_batch_mixed_tasks[1][name]
        assert torch.equal(kwargs["targets"], expected), f"{name} scored against another task's targets"


def test_classification_task_last_does_not_break_r2(model_config_mixed_tasks, sample_batch_mixed_tasks, mocker):
    """A classification head as the final task used to crash the R² update with a shape mismatch."""
    config = model_config_mixed_tasks
    reordered = [tc for tc in config.task_configs if tc.name != "clf_task_1"]
    reordered.append(next(tc for tc in config.task_configs if tc.name == "clf_task_1"))
    model = FlexibleMultiTaskModel(
        task_configs=reordered,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    model.eval()
    mocker.patch.object(model, "log_dict")
    mocker.patch.object(model, "log")

    model.validation_step(sample_batch_mixed_tasks, batch_idx=0)  # used to raise RuntimeError

    # Only the regression tasks own an R² metric; both must have been updated.
    assert model._metrics_updated["val"] == {"regr_task_1", "regr_task_2"}


# Helper for creating dummy dataframes
def create_dummy_dataframe(num_samples, num_features, index_prefix="sample_"):
    data = np.random.rand(num_samples, num_features)
    index = [f"{index_prefix}{i}" for i in range(num_samples)]
    return pd.DataFrame(data, index=index, columns=[f"feat_{j}" for j in range(num_features)])


# ---------------------------------------------------------------------------
# TestAutoEncoder — new enable_autoencoder interface
# ---------------------------------------------------------------------------

INPUT_DIM = 20
LATENT_DIM = 8


def _make_model(nonnegative=False, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
    enc = MLPEncoderConfig(hidden_dims=[input_dim, 16, latent_dim])
    task = RegressionTaskConfig(name="prop", data_column="prop", dims=[latent_dim, 4, 1])
    return FlexibleMultiTaskModel(
        task_configs=[task],
        encoder_config=enc,
        enable_autoencoder=True,
        autoencoder_nonnegative=nonnegative,
    )


def test_enable_autoencoder_creates_head():
    model = _make_model()
    assert "__reconstruction__" in model.task_heads


def test_enable_autoencoder_mlp_dims():
    model = _make_model()
    head = model.task_heads["__reconstruction__"]
    # First layer input == latent_dim; output == input_dim
    first = next(iter(head.net.parameters()))
    assert first.shape[1] == LATENT_DIM
    # Config dims should be reversed hidden_dims
    cfg = model.task_configs_map["__reconstruction__"]
    assert cfg.dims == [LATENT_DIM, 16, INPUT_DIM]


def test_enable_autoencoder_transformer_dims():
    # Transformer AE dims should be [latent_dim, input_dim] — a single linear projection
    enc = TransformerEncoderConfig(input_dim=INPUT_DIM, d_model=LATENT_DIM)
    task = RegressionTaskConfig(name="prop", data_column="prop", dims=[LATENT_DIM, 4, 1])
    model = FlexibleMultiTaskModel(
        task_configs=[task],
        encoder_config=enc,
        enable_autoencoder=True,
    )
    cfg = model.task_configs_map["__reconstruction__"]
    assert cfg.dims == [LATENT_DIM, INPUT_DIM]
    # forward produces the right output shape
    x = torch.randn(4, INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert out["__reconstruction__"].shape == (4, INPUT_DIM)


def test_enable_autoencoder_not_in_task_configs_by_default():
    enc = MLPEncoderConfig(hidden_dims=[INPUT_DIM, 16, LATENT_DIM])
    task = RegressionTaskConfig(name="prop", data_column="prop", dims=[LATENT_DIM, 4, 1])
    model = FlexibleMultiTaskModel(task_configs=[task], encoder_config=enc)
    assert "__reconstruction__" not in model.task_heads


def test_autoencoder_forward_runs():
    model = _make_model()
    x = torch.randn(4, INPUT_DIM)
    out = model(x)
    assert "__reconstruction__" in out
    assert out["__reconstruction__"].shape == (4, INPUT_DIM)


def test_autoencoder_nonnegative_output():
    model = _make_model(nonnegative=True)
    x = torch.randn(32, INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert out["__reconstruction__"].min().item() >= 0


def test_autoencoder_linear_output_can_be_negative():
    model = _make_model(nonnegative=False)
    torch.manual_seed(0)
    x = torch.randn(128, INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert out["__reconstruction__"].min().item() < 0


def test_optimize_latent_space_requires_ae():
    enc = MLPEncoderConfig(hidden_dims=[INPUT_DIM, 16, LATENT_DIM])
    task = RegressionTaskConfig(name="prop", data_column="prop", dims=[LATENT_DIM, 4, 1])
    model = FlexibleMultiTaskModel(task_configs=[task], encoder_config=enc)
    model.eval()
    with pytest.raises(ValueError, match="enable_autoencoder"):
        model.optimize_latent(
            task_name="prop",
            initial_input=torch.randn(2, INPUT_DIM),
            optimize_space="latent",
        )


# --- optimize_latent classification objective (class_targets) ---------------


def _make_reg_clf_model():
    enc = MLPEncoderConfig(hidden_dims=[INPUT_DIM, 16, LATENT_DIM])
    tasks: list[TaskConfigType] = [
        RegressionTaskConfig(name="prop", data_column="prop", dims=[LATENT_DIM, 8, 1]),
        ClassificationTaskConfig(name="cls", data_column="cls", num_classes=3, dims=[LATENT_DIM, 8, 3]),
    ]
    return FlexibleMultiTaskModel(task_configs=tasks, encoder_config=enc, enable_autoencoder=True)


def _target_class_prob(model, x, classes):
    with torch.no_grad():
        h = torch.tanh(model.encoder(x))
        probs = torch.softmax(model.task_heads["cls"](h), dim=-1)
        return probs[:, classes].sum(dim=-1).mean().item()


def test_optimize_latent_class_target_input_space_increases_prob():
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    model.eval()  # match optimize_latent's internal eval mode (consistent BatchNorm stats)
    x = torch.randn(8, INPUT_DIM)
    target_classes = [2]
    before = _target_class_prob(model, x, target_classes)
    res = model.optimize_latent(
        initial_input=x, class_targets={"cls": target_classes}, optimize_space="input", steps=100, lr=0.2
    )
    after = _target_class_prob(model, res.optimized_input[:, 0, :], target_classes)
    assert after > before  # objective drives the target-class probability up


def test_optimize_latent_combined_reg_and_class_targets():
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    x = torch.randn(5, INPUT_DIM)
    res = model.optimize_latent(
        initial_input=x,
        task_targets={"prop": 1.0},
        class_targets={"cls": [0, 1]},
        optimize_space="latent",
        steps=20,
    )
    assert res.optimized_input.shape == (5, 1, INPUT_DIM)  # reconstructed via AE
    assert res.optimized_target.shape == (5, 1, 2)  # one regression + one class channel


def test_optimize_latent_class_targets_rejects_regression_task():
    model = _make_reg_clf_model()
    with pytest.raises(ValueError, match="accepts value/direction"):
        model.optimize_latent(
            initial_input=torch.randn(2, INPUT_DIM),
            class_targets={"prop": [0]},
            optimize_space="input",
        )


def test_optimize_latent_class_targets_rejects_out_of_range_index():
    model = _make_reg_clf_model()  # "cls" head has num_classes=3 → valid indices [0, 3)
    for bad in ([3], [-1]):
        with pytest.raises(ValueError, match="out of range"):
            model.optimize_latent(
                initial_input=torch.randn(2, INPUT_DIM),
                class_targets={"cls": bad},
                optimize_space="input",
            )


def test_optimize_latent_class_targets_only_no_regression():
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    x = torch.randn(4, INPUT_DIM)
    res = model.optimize_latent(initial_input=x, class_targets={"cls": [1]}, optimize_space="input", steps=10)
    assert res.optimized_input.shape == (4, 1, INPUT_DIM)
    assert res.optimized_target.shape == (4, 1, 1)  # the class channel (P(classes)) is tracked


def test_optimize_latent_ae_align_validates_range():
    """ae_align_scale lives in [0, 1] — out-of-range values are rejected."""
    model = _make_reg_clf_model()
    with pytest.raises(ValueError, match=r"ae_align_scale must be in \[0, 1\]"):
        model.optimize_latent(
            initial_input=torch.randn(2, INPUT_DIM),
            task_targets={"prop": 1.0},
            optimize_space="latent",
            ae_align_scale=-0.1,
        )
    with pytest.raises(ValueError, match=r"ae_align_scale must be in \[0, 1\]"):
        model.optimize_latent(
            initial_input=torch.randn(2, INPUT_DIM),
            task_targets={"prop": 1.0},
            optimize_space="latent",
            ae_align_scale=1.5,
        )


def test_optimize_latent_ae_align_runs_in_latent_space():
    torch.manual_seed(0)
    model = _make_reg_clf_model()  # enable_autoencoder=True, so AE head is available
    x = torch.randn(4, INPUT_DIM)
    res = model.optimize_latent(
        initial_input=x,
        targets=[
            OptimizationTarget(task="prop", value=1.0),
            OptimizationTarget(task="cls", classes=[1], weight=3.0),
        ],
        ae_align_scale=0.5,  # default empirical sweet spot
        optimize_space="latent",
        steps=10,
    )
    assert res.optimized_input.shape == (4, 1, INPUT_DIM)
    assert res.optimized_target.shape == (4, 1, 2)


def test_optimize_latent_target_weight_rejects_nonpositive():
    model = _make_reg_clf_model()
    with pytest.raises(ValueError, match="weight must be > 0"):
        model.optimize_latent(
            initial_input=torch.randn(2, INPUT_DIM),
            targets=[OptimizationTarget(task="cls", classes=[1], weight=0.0)],
            optimize_space="input",
        )


def test_optimize_latent_target_weight_runs_with_combined_objectives():
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    x = torch.randn(4, INPUT_DIM)
    res = model.optimize_latent(
        initial_input=x,
        targets=[
            OptimizationTarget(task="prop", value=1.0),
            OptimizationTarget(task="cls", classes=[1], weight=5.0),  # class prob is primary
        ],
        optimize_space="input",
        steps=10,
    )
    assert res.optimized_input.shape == (4, 1, INPUT_DIM)
    assert res.optimized_target.shape == (4, 1, 2)  # regression channel + class channel


def test_optimize_latent_restores_requires_grad_after_call():
    """Regression test for the requires_grad leak: optimize_latent must leave every model
    parameter's ``requires_grad`` flag as it was before the call. Previously only ``training``
    mode was restored, so subsequent ``model.fit(...)`` calls silently froze the encoder /
    heads and "training stopped moving the weights" became annoying to bisect.
    """
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    # Snapshot whatever pattern the caller had (all True by default, but the test should hold
    # for any non-trivial pattern too).
    expected = [p.requires_grad for p in model.parameters()]
    model.optimize_latent(
        initial_input=torch.randn(3, INPUT_DIM),
        task_targets={"prop": 1.0},
        class_targets={"cls": [1]},
        optimize_space="input",
        steps=5,
    )
    actual = [p.requires_grad for p in model.parameters()]
    assert actual == expected


# --- OptimizationTarget kinds (direction / curve / class-low) -----------------


def _make_full_model():
    """Regression + classification + kernel-regression heads (AE on) for target-kind tests."""
    enc = MLPEncoderConfig(hidden_dims=[INPUT_DIM, 16, LATENT_DIM])
    tasks: list[TaskConfigType] = [
        RegressionTaskConfig(name="prop", data_column="prop", dims=[LATENT_DIM, 8, 1]),
        ClassificationTaskConfig(name="cls", data_column="cls", num_classes=3, dims=[LATENT_DIM, 8, 3]),
        KernelRegressionTaskConfig(
            name="curve",
            data_column="curve",
            t_column="curve_t",
            x_dim=[LATENT_DIM, 8, 4],
            t_dim=[4, 4],
            t_encoding_method="fc",
        ),
    ]
    return FlexibleMultiTaskModel(task_configs=tasks, encoder_config=enc, enable_autoencoder=True)


def test_direction_target_moves_prediction_both_ways():
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    model.eval()
    x = torch.randn(6, INPUT_DIM)
    up = model.optimize_latent(
        initial_input=x, targets=[OptimizationTarget(task="prop", direction="high")], steps=50, lr=0.1
    )
    down = model.optimize_latent(
        initial_input=x, targets=[OptimizationTarget(task="prop", direction="low")], steps=50, lr=0.1
    )
    assert up.optimized_target[:, 0, 0].mean() > up.initial_score[:, 0, 0].mean()
    assert down.optimized_target[:, 0, 0].mean() < down.initial_score[:, 0, 0].mean()


def test_class_low_direction_decreases_probability():
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    model.eval()
    x = torch.randn(8, INPUT_DIM)
    res = model.optimize_latent(
        initial_input=x, targets=[OptimizationTarget(task="cls", classes=[1], direction="low")], steps=80, lr=0.1
    )
    # Channel is P(classes) for both directions; "low" must push it down.
    assert res.optimized_target[:, 0, 0].mean() < res.initial_score[:, 0, 0].mean()


def test_class_target_covering_all_classes_rejected():
    model = _make_reg_clf_model()
    for direction in ("high", "low"):
        with pytest.raises(ValueError, match="strict subset"):
            model.optimize_latent(
                initial_input=torch.randn(2, INPUT_DIM),
                targets=[OptimizationTarget(task="cls", classes=[0, 1, 2], direction=direction)],
            )


def test_curve_target_reduces_rmse_channel():
    torch.manual_seed(0)
    model = _make_full_model()
    model.eval()
    x = torch.randn(5, INPUT_DIM)
    points = [[0.1, 0.5], [0.5, 1.0], [0.9, 0.2]]
    res = model.optimize_latent(
        initial_input=x, targets=[OptimizationTarget(task="curve", points=points)], steps=60, lr=0.1
    )
    assert res.optimized_target.shape == (5, 1, 1)
    assert res.trajectory.shape == (5, 1, 60, 1)
    # Channel is RMSE-to-curve — must decrease.
    assert res.optimized_target[:, 0, 0].mean() < res.initial_score[:, 0, 0].mean()


def test_regression_target_field_matrix_validation():
    model = _make_full_model()
    x = torch.randn(2, INPUT_DIM)
    # both value and direction
    with pytest.raises(ValueError, match="exactly one of value or direction"):
        model.optimize_latent(initial_input=x, targets=[OptimizationTarget(task="prop", value=1.0, direction="high")])
    # neither
    with pytest.raises(ValueError, match="exactly one of value or direction"):
        model.optimize_latent(initial_input=x, targets=[OptimizationTarget(task="prop")])
    # curve task without points
    with pytest.raises(ValueError, match="non-empty points"):
        model.optimize_latent(initial_input=x, targets=[OptimizationTarget(task="curve")])
    # malformed points
    with pytest.raises(ValueError, match=r"\[t, y\] pairs"):
        model.optimize_latent(initial_input=x, targets=[OptimizationTarget(task="curve", points=[[1.0, 2.0, 3.0]])])
    # classification without classes
    with pytest.raises(ValueError, match="non-empty classes"):
        model.optimize_latent(initial_input=x, targets=[OptimizationTarget(task="cls")])
    # duplicate task
    with pytest.raises(ValueError, match="Duplicate"):
        model.optimize_latent(
            initial_input=x,
            targets=[OptimizationTarget(task="prop", value=1.0), OptimizationTarget(task="prop", direction="high")],
        )
    # targets is exclusive with the sugar kwargs
    with pytest.raises(ValueError, match="mutually exclusive"):
        model.optimize_latent(
            initial_input=x, targets=[OptimizationTarget(task="prop", value=1.0)], task_targets={"prop": 1.0}
        )


def test_evaluate_targets_matches_initial_score_and_hand_computation():
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    model.eval()
    x = torch.randn(4, INPUT_DIM)
    targets = [
        OptimizationTarget(task="prop", value=2.0, weight=3.0),
        OptimizationTarget(task="cls", classes=[1], weight=5.0),
    ]
    channels, objective = model.evaluate_targets(x, targets)
    assert channels.shape == (4, 2)
    assert objective.shape == (4,)
    # Channels equal a plain forward; objective equals the weighted per-sample loss sum.
    with torch.no_grad():
        h = torch.tanh(model.encoder(x))
        pred = model.task_heads["prop"](h).squeeze(-1)
        probs = torch.softmax(model.task_heads["cls"](h), dim=-1)
        p_sel = probs[:, 1]
        expected_obj = 3.0 * (pred - 2.0) ** 2 + 5.0 * (-torch.log(p_sel))
    assert torch.allclose(channels[:, 0], pred, atol=1e-5)
    assert torch.allclose(channels[:, 1], p_sel, atol=1e-5)
    assert torch.allclose(objective, expected_obj, atol=1e-4)
    # And the same channels come back as optimize_latent's initial_score.
    res = model.optimize_latent(initial_input=x, targets=targets, steps=1)
    assert torch.allclose(res.initial_score[:, 0, :], channels, atol=1e-5)


def test_optimize_composition_mixed_target_kinds_smoke():
    torch.manual_seed(0)
    model = _make_full_model()
    kernel = torch.randn(6, INPUT_DIM)
    res = model.optimize_composition(
        kernel,
        targets=[
            OptimizationTarget(task="prop", direction="low"),
            OptimizationTarget(task="cls", classes=[2]),
            OptimizationTarget(task="curve", points=[[0.2, 0.5], [0.8, 1.5]]),
        ],
        n_starts=3,
        steps=8,
    )
    assert res.optimized_target.shape == (3, 3)
    assert res.trajectory.shape == (8, 3, 3)
    assert res.initial_score.shape == (3, 3)


# --- optimize_composition (differentiable KMD) --------------------------------


def test_optimize_composition_runs_and_returns_simplex_weights():
    torch.manual_seed(0)
    model = _make_reg_clf_model()  # INPUT_DIM=20
    n_components = 6
    kmd_kernel = torch.randn(n_components, INPUT_DIM)
    res = model.optimize_composition(
        kmd_kernel,
        targets=[
            OptimizationTarget(task="prop", value=1.0),
            OptimizationTarget(task="cls", classes=[1], weight=3.0),
        ],
        n_starts=4,
        steps=10,
    )
    assert res.optimized_weights.shape == (4, n_components)
    # Output is a simplex: non-negative, rows sum to 1.
    assert (res.optimized_weights >= 0).all()
    assert torch.allclose(res.optimized_weights.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert res.optimized_descriptor.shape == (4, INPUT_DIM)
    # Descriptor matches the matmul exactly (no round-trip).
    assert torch.allclose(res.optimized_descriptor, res.optimized_weights @ kmd_kernel, atol=1e-5)
    assert res.optimized_target.shape == (4, 2)
    assert res.trajectory.shape == (10, 4, 2)


def test_optimize_composition_validates_kernel_and_objectives():
    model = _make_reg_clf_model()
    # kernel must be 2D
    with pytest.raises(ValueError, match="2D torch.Tensor"):
        model.optimize_composition(torch.randn(6), task_targets={"prop": 1.0}, n_starts=2, steps=2)
    # kernel's x_dim must match encoder.input_dim
    with pytest.raises(ValueError, match="encoder.input_dim"):
        model.optimize_composition(torch.randn(6, INPUT_DIM + 1), task_targets={"prop": 1.0}, n_starts=2, steps=2)
    # at least one objective required
    with pytest.raises(ValueError, match="at least one of"):
        model.optimize_composition(torch.randn(6, INPUT_DIM), n_starts=2, steps=2)


def test_optimize_composition_increases_target_class_probability():
    """Optimising for a class with high class_target_weight raises P(target) from a uniform seed."""
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    model.eval()
    kmd_kernel = torch.randn(6, INPUT_DIM)
    target = [1]
    init_w = torch.full((4, 6), 1.0 / 6)

    def _prob(w):
        with torch.no_grad():
            logits = model.task_heads["cls"](torch.tanh(model.encoder(w @ kmd_kernel)))
            return torch.softmax(logits, dim=-1)[:, target].sum(dim=-1).mean().item()

    res = model.optimize_composition(
        kmd_kernel,
        initial_weights=init_w,
        targets=[OptimizationTarget(task="cls", classes=target, weight=5.0)],
        steps=200,
        lr=0.2,
    )
    assert _prob(res.optimized_weights) > _prob(init_w)


def test_optimize_composition_rejects_negative_initial_weights():
    model = _make_reg_clf_model()
    kernel = torch.randn(6, INPUT_DIM)
    bad = torch.tensor([[1.0, -0.1, 0.2, 0.2, 0.2, 0.5]])
    with pytest.raises(ValueError, match="non-negative"):
        model.optimize_composition(kernel, initial_weights=bad, task_targets={"prop": 0.0}, steps=2)
    zero_row = torch.zeros(1, 6)
    with pytest.raises(ValueError, match="positive sum"):
        model.optimize_composition(kernel, initial_weights=zero_row, task_targets={"prop": 0.0}, steps=2)


def test_optimize_composition_does_not_reset_global_rng():
    """The method must not rewind the global RNG (would defeat n_starts diversity)."""
    torch.manual_seed(42)
    model = _make_reg_clf_model()
    kernel = torch.randn(6, INPUT_DIM)
    state_before = torch.random.get_rng_state().clone()
    model.optimize_composition(kernel, task_targets={"prop": 0.0}, n_starts=4, steps=2)
    state_after = torch.random.get_rng_state()
    # The RNG must have advanced (some random was consumed), not been reset.
    assert not torch.equal(state_before, state_after)


def test_optimize_composition_trajectory_shape_when_zero_steps():
    """Empty trajectory still carries the target-channel width T (not 0)."""
    model = _make_reg_clf_model()
    kernel = torch.randn(6, INPUT_DIM)
    res = model.optimize_composition(
        kernel, task_targets={"prop": 0.0}, class_targets={"cls": [1]}, n_starts=3, steps=0
    )
    # targets = [prop (value), cls (class)] → T == 2
    assert res.trajectory.shape == (0, 3, 2)


def test_optimize_composition_does_not_populate_module_grads():
    """Encoder/head .grad must NOT be touched; only logits is optimised."""
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    kernel = torch.randn(6, INPUT_DIM)
    # Ensure no pre-existing grads.
    for p in model.parameters():
        p.grad = None
    model.optimize_composition(kernel, task_targets={"prop": 0.5}, n_starts=3, steps=4)
    # After the call, no encoder/head parameter should have an accumulated .grad.
    for name, p in model.named_parameters():
        assert p.grad is None, f"parameter {name} unexpectedly has .grad after optimize_composition"


def test_optimize_composition_restores_model_state_on_error():
    """A validation raised inside the call must still leave the model in its original mode and
    with parameter requires_grad flags untouched (try/finally invariant)."""
    model = _make_reg_clf_model()
    model.train()  # put model into training mode
    before_mode = model.training
    before_req_grad = [p.requires_grad for p in model.parameters()]
    kernel = torch.randn(6, INPUT_DIM)
    # Force a failure deep in the optimisation (mismatched class target index).
    with pytest.raises(ValueError):
        model.optimize_composition(kernel, class_targets={"cls": [99]}, n_starts=2, steps=2)
    # Mode and requires_grad must be exactly as we left them.
    assert model.training == before_mode
    assert [p.requires_grad for p in model.parameters()] == before_req_grad


def _build_aligned_model_and_kernel():
    """Helper for symbol-based tests: a tiny model + kernel whose first dim == len(DEFAULT_ELEMENTS).

    Symbol-based ``allowed_elements`` / ``element_step_scale`` require the kernel to align with
    the bundled element registry. The kernel is random (matmul correctness is irrelevant here);
    we just need the right shape so the symbol→index mapping is unambiguous.
    """
    from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS

    n_components = len(DEFAULT_ELEMENTS)
    enc = MLPEncoderConfig(hidden_dims=[INPUT_DIM, 16, LATENT_DIM])
    tasks: list[TaskConfigType] = [
        RegressionTaskConfig(name="prop", data_column="prop", dims=[LATENT_DIM, 8, 1]),
        ClassificationTaskConfig(name="cls", data_column="cls", num_classes=3, dims=[LATENT_DIM, 8, 3]),
    ]
    model = FlexibleMultiTaskModel(task_configs=tasks, encoder_config=enc, enable_autoencoder=True)
    kernel = torch.randn(n_components, INPUT_DIM)
    return model, kernel, DEFAULT_ELEMENTS


def test_optimize_composition_allowed_elements_symbol_whitelist():
    """A list of element symbols restricts w to those elements; the rest stay at exactly 0."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()
    whitelist = ["Mg", "Al", "Cu", "Ni"]
    res = model.optimize_composition(
        kernel,
        targets=[
            OptimizationTarget(task="prop", value=1.0),
            OptimizationTarget(task="cls", classes=[1], weight=3.0),
        ],
        n_starts=3,
        allowed_elements=whitelist,
        steps=15,
        lr=0.2,
    )
    w = res.optimized_weights
    allowed_idx = [elements.index(s) for s in whitelist]
    forbidden_idx = [i for i in range(len(elements)) if i not in allowed_idx]
    assert torch.all(w[:, forbidden_idx] == 0)
    assert torch.allclose(w[:, allowed_idx].sum(dim=-1), torch.ones(3), atol=1e-5)


def test_optimize_composition_allowed_elements_default_all():
    """The default ``allowed_elements='all'`` imposes no constraint."""
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    kernel = torch.randn(6, INPUT_DIM)  # any kernel size works when no symbols are used
    res = model.optimize_composition(kernel, task_targets={"prop": 0.5}, n_starts=2, steps=5)
    # All columns can carry weight; nothing should be forced to zero by the default.
    assert (res.optimized_weights > 0).all()


def test_optimize_composition_allowed_elements_validation():
    model, kernel, _ = _build_aligned_model_and_kernel()
    # "all" is the only acceptable string.
    with pytest.raises(ValueError, match="must be 'all'"):
        model.optimize_composition(kernel, task_targets={"prop": 0.0}, allowed_elements="everything", steps=2)
    # Empty list rejected.
    with pytest.raises(ValueError, match="non-empty"):
        model.optimize_composition(kernel, task_targets={"prop": 0.0}, allowed_elements=[], steps=2)
    # Unknown symbol rejected.
    with pytest.raises(ValueError, match="Unknown element symbol"):
        model.optimize_composition(kernel, task_targets={"prop": 0.0}, allowed_elements=["Mg", "NotAnElement"], steps=2)
    # Wrong type rejected.
    with pytest.raises(TypeError, match="non-empty list"):
        model.optimize_composition(kernel, task_targets={"prop": 0.0}, allowed_elements=42, steps=2)  # type: ignore[arg-type]
    # Symbols with a non-aligned kernel rejected.
    small_kernel = torch.randn(6, INPUT_DIM)
    with pytest.raises(ValueError, match="align with DEFAULT_ELEMENTS"):
        model.optimize_composition(
            small_kernel, task_targets={"prop": 0.0}, allowed_elements=["Mg", "Al"], n_starts=2, steps=2
        )


def test_optimize_composition_element_step_scale_locks_symbols():
    """A symbol→0.0 mapping freezes those elements' weights at their **absolute** seed values.

    The previous version of this test only checked that the locked elements' ratio stayed at 1.0
    (which holds even if both drift together, since their logits move in lockstep). That doesn't
    actually verify "frozen": with the bare gradient-zeroing implementation, ``w[Mg]`` drifts
    because the softmax denominator changes whenever other (unlocked) logits move. This test
    now asserts each locked element holds its **un-blended seed value** to within float tolerance.
    """
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()

    # Seed: asymmetric mass on 4 specific symbols, zero on the rest. The asymmetry matters —
    # equal-mass locks would survive ratio-only checks even if both drift together.
    locked_syms = ["Mg", "Al"]
    free_syms = ["Cu", "Ni"]
    init_w = torch.zeros(1, len(elements))
    init_w[0, elements.index("Mg")] = 0.30
    init_w[0, elements.index("Al")] = 0.20
    init_w[0, elements.index("Cu")] = 0.30
    init_w[0, elements.index("Ni")] = 0.20

    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 5.0},
        initial_weights=init_w,
        element_step_scale={s: 0.0 for s in locked_syms},
        steps=80,
        lr=0.5,  # large enough that any drift in locked weights would show up
    )
    w = res.optimized_weights[0]
    mg, al = elements.index("Mg"), elements.index("Al")
    assert torch.isclose(w[mg], torch.tensor(0.30, dtype=w.dtype), atol=1e-4)
    assert torch.isclose(w[al], torch.tensor(0.20, dtype=w.dtype), atol=1e-4)
    # And the unlocked elements share the remaining 0.50 mass.
    free_total = w.sum() - w[mg] - w[al]
    assert torch.isclose(free_total, torch.tensor(0.50, dtype=w.dtype), atol=1e-4)


def test_optimize_composition_element_step_scale_locks_with_unlocked_drift():
    """Locked elements stay at seed even while unlocked elements actually move."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()
    init_w = torch.zeros(1, len(elements))
    init_w[0, elements.index("Mg")] = 0.40  # locked
    init_w[0, elements.index("Cu")] = 0.30  # free
    init_w[0, elements.index("Ni")] = 0.30  # free

    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 5.0},
        initial_weights=init_w,
        element_step_scale={"Mg": 0.0},
        steps=80,
        lr=0.5,
    )
    w = res.optimized_weights[0]
    # Mg held exactly.
    assert torch.isclose(w[elements.index("Mg")], torch.tensor(0.40, dtype=w.dtype), atol=1e-4)
    # The unlocked elements ended up in different ratios than they started (proves they moved).
    cu0, ni0 = init_w[0, elements.index("Cu")], init_w[0, elements.index("Ni")]
    cu_f, ni_f = w[elements.index("Cu")], w[elements.index("Ni")]
    assert not torch.isclose(cu_f / ni_f, cu0 / ni0, atol=1e-3), "unlocked weights didn't actually move"
    # And the unlocked mass equals 1 - locked mass.
    assert torch.isclose(w.sum() - w[elements.index("Mg")], torch.tensor(0.60, dtype=w.dtype), atol=1e-4)


def test_optimize_composition_element_step_scale_lock_requires_initial_weights():
    """A hard lock with random init is rejected (no seed to lock to)."""
    model, kernel, _ = _build_aligned_model_and_kernel()
    with pytest.raises(ValueError, match="hard lock.*initial_weights"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            element_step_scale={"Mg": 0.0},
            n_starts=2,
            steps=2,
        )


def test_optimize_composition_element_step_scale_lock_must_be_allowed():
    """Locking an element that's not in allowed_elements is contradictory and rejected."""
    model, kernel, elements = _build_aligned_model_and_kernel()
    init_w = torch.zeros(1, len(elements))
    init_w[0, elements.index("Mg")] = 1.0
    with pytest.raises(ValueError, match="must also be in allowed_elements"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            initial_weights=init_w,
            allowed_elements=["Al", "Cu"],
            element_step_scale={"Mg": 0.0},
            steps=2,
        )


def test_optimize_composition_element_step_scale_uniform_scalar():
    """A scalar element_step_scale=0 freezes every element at the seed (uniform behaviour)."""
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    kernel = torch.randn(6, INPUT_DIM)
    init_w = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]])
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 5.0},
        initial_weights=init_w,
        element_step_scale=0.0,  # everything frozen
        seed_blend=1.0,  # strict seed → no uniform mixing, so w should match init_w exactly
        steps=30,
        lr=0.5,
    )
    # With every element frozen and equal seed proportions kept, w should match init_w (normalised).
    assert torch.allclose(res.optimized_weights, init_w, atol=1e-5)


def test_optimize_composition_element_step_scale_validation():
    model, kernel, _ = _build_aligned_model_and_kernel()
    # Negative scalar rejected.
    with pytest.raises(ValueError, match=">= 0"):
        model.optimize_composition(kernel, task_targets={"prop": 0.0}, element_step_scale=-0.5, steps=2)
    # Unknown symbol rejected.
    with pytest.raises(ValueError, match="Unknown element symbol"):
        model.optimize_composition(
            kernel, task_targets={"prop": 0.0}, element_step_scale={"Mg": 0.5, "NotAnElement": 0.0}, steps=2
        )
    # Negative value in mapping rejected.
    with pytest.raises(ValueError, match="values must be >= 0"):
        model.optimize_composition(
            kernel, task_targets={"prop": 0.0}, element_step_scale={"Mg": 0.5, "Al": -0.1}, steps=2
        )
    # Wrong type rejected.
    with pytest.raises(TypeError, match="non-negative float or a mapping"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            element_step_scale=[1.0, 1.0],
            steps=2,  # type: ignore[arg-type]
        )
    # Symbol dict with a non-aligned kernel rejected.
    small_kernel = torch.randn(6, INPUT_DIM)
    with pytest.raises(ValueError, match="align with DEFAULT_ELEMENTS"):
        model.optimize_composition(
            small_kernel, task_targets={"prop": 0.0}, element_step_scale={"Mg": 0.0}, n_starts=2, steps=2
        )


def test_optimize_composition_seed_blend_validates_range():
    """seed_blend must be in [0, 1]."""
    model, kernel, elements = _build_aligned_model_and_kernel()
    w = torch.zeros(1, len(elements))
    w[0, 0] = 1.0
    with pytest.raises(ValueError, match=r"seed_blend must be in \[0, 1\]"):
        model.optimize_composition(kernel, initial_weights=w, task_targets={"prop": 0.0}, seed_blend=-0.1, steps=2)
    with pytest.raises(ValueError, match=r"seed_blend must be in \[0, 1\]"):
        model.optimize_composition(kernel, initial_weights=w, task_targets={"prop": 0.0}, seed_blend=1.5, steps=2)


def test_optimize_composition_seed_blend_strict_freezes_support_set():
    """seed_blend=1.0 reproduces the old strict-seed behaviour: non-seed elements stay ~0."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()

    # Seed places all mass on Mg + Al; with seed_blend=1.0 every other element starts at logit
    # log(1e-12) ≈ −27.6 and can't escape in a handful of steps.
    init_w = torch.zeros(1, len(elements))
    init_w[0, elements.index("Mg")] = 0.6
    init_w[0, elements.index("Al")] = 0.4

    res = model.optimize_composition(
        kernel,
        initial_weights=init_w,
        task_targets={"prop": 5.0},
        seed_blend=1.0,
        steps=40,
        lr=0.1,
    )
    w = res.optimized_weights[0]
    seed_mass = w[elements.index("Mg")] + w[elements.index("Al")]
    # Strict seed: non-seed elements never recruited — essentially all mass stays on Mg+Al.
    assert seed_mass > 0.999


def test_optimize_composition_seed_blend_allows_new_elements():
    """seed_blend<1.0 lifts non-seed logits enough that Adam can recruit new elements."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()

    init_w = torch.zeros(1, len(elements))
    init_w[0, elements.index("Mg")] = 0.6
    init_w[0, elements.index("Al")] = 0.4

    res = model.optimize_composition(
        kernel,
        initial_weights=init_w,
        task_targets={"prop": 5.0},
        seed_blend=0.5,  # heavy blend so the test is robust to model init
        steps=80,
        lr=0.2,
    )
    w = res.optimized_weights[0]
    non_seed = sum(w[i].item() for i, s in enumerate(elements) if s not in {"Mg", "Al"})
    # Some non-seed mass should accumulate (the toy model has no specific preference, so we
    # only require the floor to be measurably above zero — the strict-seed test above shows
    # the same setup gives ~0 when seed_blend=1.0).
    assert non_seed > 0.05


def test_optimize_composition_random_init_uses_n_starts():
    """initial_weights=None falls back to n_starts random simplex points; allowed_elements still binds."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()
    allowed = ["Mg", "Al", "Cu", "Ni"]
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        n_starts=5,
        allowed_elements=allowed,
        steps=5,
    )
    assert res.optimized_weights.shape == (5, len(elements))
    # Disallowed elements stay at exactly zero (mask is applied at every step).
    disallowed = [i for i, s in enumerate(elements) if s not in allowed]
    assert torch.allclose(res.optimized_weights[:, disallowed], torch.zeros_like(res.optimized_weights[:, disallowed]))


def test_optimize_composition_diversity_scale_validates_range():
    """diversity_scale lives in [0, 1] — out-of-range values are rejected."""
    model, kernel, _ = _build_aligned_model_and_kernel()
    with pytest.raises(ValueError, match=r"diversity_scale must be in \[0, 1\]"):
        model.optimize_composition(kernel, task_targets={"prop": 0.0}, diversity_scale=-0.1, n_starts=2, steps=2)
    with pytest.raises(ValueError, match=r"diversity_scale must be in \[0, 1\]"):
        model.optimize_composition(kernel, task_targets={"prop": 0.0}, diversity_scale=1.5, n_starts=2, steps=2)


def test_optimize_composition_diversity_scale_endpoints_run():
    """Both endpoints (0 = max penalty, 1 = no penalty default) run cleanly and stay on the simplex."""
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    for scale in (0.0, 0.5, 1.0):
        res = model.optimize_composition(kernel, task_targets={"prop": 1.0}, n_starts=3, diversity_scale=scale, steps=5)
        assert res.optimized_weights.shape[0] == 3
        assert torch.allclose(res.optimized_weights.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_optimize_composition_diversity_scale_direction():
    """diversity_scale=1 (no penalty) keeps a higher per-output entropy than diversity_scale=0 (max penalty)."""
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    res_peaky = model.optimize_composition(
        kernel, task_targets={"prop": 1.0}, n_starts=4, diversity_scale=0.0, steps=60, lr=0.2
    )
    torch.manual_seed(0)
    res_spread = model.optimize_composition(
        kernel, task_targets={"prop": 1.0}, n_starts=4, diversity_scale=1.0, steps=60, lr=0.2
    )

    def _mean_entropy(w):
        return float(-(w * w.clamp(min=1e-12).log()).sum(dim=-1).mean())

    assert _mean_entropy(res_spread.optimized_weights) > _mean_entropy(res_peaky.optimized_weights)


def test_optimize_composition_uses_kmd_kernel_torch():
    """End-to-end: a real KMD's kernel_torch flows into optimize_composition."""
    from foundation_model.utils.kmd_plus import KMD

    rng = np.random.default_rng(0)
    # 1d with n_features=5, n_grids=4 → x_dim = 20 (matches INPUT_DIM).
    cf = rng.normal(size=(7, 5))
    kmd = KMD(cf, method="1d", n_grids=4)
    model = _make_reg_clf_model()
    kernel = kmd.kernel_torch()
    assert kernel.shape == (7, INPUT_DIM)
    res = model.optimize_composition(kernel, task_targets={"prop": 0.5}, n_starts=3, steps=10)
    assert res.optimized_weights.shape == (3, 7)
    assert torch.allclose(res.optimized_weights.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_optimize_composition_max_elements_enforces_K_cardinality():
    """max_elements=K → final composition has *at most* K non-zero elements per row.

    The hard top-K projection picks K positions, but if any of those has zero ``w_soft`` mass
    (can happen when the optimiser drove other logits very negative), it stays at zero after
    renormalisation — so the contract is "≤ K", not "= K". On a non-degenerate synthetic
    setup K is usually saturated; on a real-model load some rows can land below K.
    """
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    K = 3
    res = model.optimize_composition(
        kernel,
        targets=[
            OptimizationTarget(task="prop", value=1.0),
            OptimizationTarget(task="cls", classes=[1], weight=2.0),
        ],
        n_starts=4,
        max_elements=K,
        steps=120,
        lr=0.2,
    )
    w = res.optimized_weights
    # Simplex preserved.
    assert torch.allclose(w.sum(dim=-1), torch.ones(w.shape[0]), atol=1e-5)
    assert (w >= 0).all()
    # At most K non-zero positions per row.
    nz = (w > 1e-6).sum(dim=-1)
    assert torch.all(nz <= K), f"expected ≤ {K} non-zero per row, got {nz.tolist()}"
    # On this toy setup with uniform-ish init we additionally expect saturation at K.
    assert torch.all(nz == K), f"toy model should saturate at K={K}, got {nz.tolist()}"


def test_optimize_composition_max_elements_full_is_noop():
    """max_elements == n_components disables the constraint (results match the unconstrained run)."""
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    kernel = torch.randn(6, INPUT_DIM)
    init = torch.full((2, 6), 1.0 / 6)
    base = model.optimize_composition(kernel, task_targets={"prop": 1.0}, initial_weights=init, steps=30, lr=0.1)
    torch.manual_seed(0)
    constrained = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        initial_weights=init,
        max_elements=6,  # == n_components → no-op
        steps=30,
        lr=0.1,
    )
    # max_elements == n_components ⇒ no soft top-K, no hard projection ⇒ identical trajectory.
    assert torch.allclose(base.optimized_weights, constrained.optimized_weights, atol=1e-5)


def test_optimize_composition_max_elements_with_allowed_elements():
    """When ``allowed_elements`` whitelists the support, top-K picks from inside the whitelist."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()
    whitelist = ["Mg", "Al", "Cu", "Ni", "Fe"]
    K = 3
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        n_starts=3,
        allowed_elements=whitelist,
        max_elements=K,
        steps=80,
        lr=0.2,
    )
    w = res.optimized_weights
    nz = (w > 1e-6).sum(dim=-1)
    assert torch.all(nz == K)
    # Non-whitelisted positions still exactly zero.
    forbidden = [i for i, s in enumerate(elements) if s not in whitelist]
    assert (w[:, forbidden] == 0).all()


def test_optimize_composition_max_elements_keeps_locked_in_support():
    """A hard-locked element must remain non-zero (and at its seed value) even with top-K."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()
    # Lock Mg at 0.30; allow the optimiser to pick the other K-1 freely.
    init_w = torch.zeros(1, len(elements))
    init_w[0, elements.index("Mg")] = 0.30
    init_w[0, elements.index("Al")] = 0.25
    init_w[0, elements.index("Cu")] = 0.25
    init_w[0, elements.index("Ni")] = 0.20
    K = 3
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 5.0},
        initial_weights=init_w,
        element_step_scale={"Mg": 0.0},  # hard lock
        max_elements=K,
        steps=120,
        lr=0.3,
    )
    w = res.optimized_weights[0]
    # Mg is held at its un-blended seed value.
    assert torch.isclose(w[elements.index("Mg")], torch.tensor(0.30, dtype=w.dtype), atol=1e-4)
    # At most K non-zero (Mg + ≤K-1 free, saturated to K on this non-degenerate setup).
    nz = int((w > 1e-6).sum().item())
    assert nz <= K, f"expected ≤ {K} non-zero with Mg locked, got {nz}"
    assert nz == K, f"toy model with Mg locked should saturate at K={K}, got {nz}"


def test_optimize_composition_max_elements_validation():
    """All max_elements / topk_* validation errors fire before model state is touched."""
    model, kernel, elements = _build_aligned_model_and_kernel()
    with pytest.raises(ValueError, match=r"max_elements must be in \[1, n_components"):
        model.optimize_composition(kernel, task_targets={"prop": 0.0}, max_elements=0, n_starts=2, steps=2)
    with pytest.raises(ValueError, match=r"max_elements must be in \[1, n_components"):
        model.optimize_composition(kernel, task_targets={"prop": 0.0}, max_elements=999, n_starts=2, steps=2)
    with pytest.raises(TypeError, match="max_elements must be an int"):
        model.optimize_composition(kernel, task_targets={"prop": 0.0}, max_elements=2.5, n_starts=2, steps=2)  # type: ignore[arg-type]
    # max_elements > |allowed_elements| → rejected with a specific message.
    with pytest.raises(ValueError, match="exceeds the number of allowed elements"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            allowed_elements=["Mg", "Al"],
            max_elements=5,
            n_starts=2,
            steps=2,
        )
    # max_elements < n_locked → rejected.
    init_w = torch.zeros(1, len(elements))
    init_w[0, elements.index("Mg")] = 0.3
    init_w[0, elements.index("Al")] = 0.3
    init_w[0, elements.index("Cu")] = 0.4
    with pytest.raises(ValueError, match="must be > total locked elements"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            initial_weights=init_w,
            element_step_scale={"Mg": 0.0, "Al": 0.0, "Cu": 0.0},
            max_elements=2,
            steps=2,
        )
    # Bad annealing_scale.
    with pytest.raises(ValueError, match=r"annealing_scale must be in \[0, 1\]"):
        model.optimize_composition(
            kernel, task_targets={"prop": 0.0}, max_elements=2, annealing_scale=-0.1, n_starts=2, steps=2
        )
    with pytest.raises(ValueError, match=r"annealing_scale must be in \[0, 1\]"):
        model.optimize_composition(
            kernel, task_targets={"prop": 0.0}, max_elements=2, annealing_scale=1.5, n_starts=2, steps=2
        )
    # Bad annealing_schedule dict.
    with pytest.raises(ValueError, match="annealing_schedule missing required keys"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            max_elements=2,
            annealing_schedule={"step": [0.5], "scale": [0.5]},  # no annealing_func
            n_starts=2,
            steps=2,
        )
    with pytest.raises(ValueError, match="annealing_schedule lists must be the same length"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            max_elements=2,
            annealing_schedule={"step": [0.5, 1.0], "scale": [0.5], "annealing_func": ["geometric"]},
            n_starts=2,
            steps=2,
        )
    with pytest.raises(ValueError, match=r"annealing_schedule\['step'\] entries must be in \(0, 1\]"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            max_elements=2,
            annealing_schedule={"step": [0.0, 1.0], "scale": [0.5, 0.0], "annealing_func": ["geometric", "geometric"]},
            n_starts=2,
            steps=2,
        )
    with pytest.raises(ValueError, match=r"annealing_schedule\['step'\] must be strictly increasing"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            max_elements=2,
            annealing_schedule={"step": [0.5, 0.5], "scale": [0.5, 0.3], "annealing_func": ["geometric", "geometric"]},
            n_starts=2,
            steps=2,
        )
    with pytest.raises(ValueError, match=r"annealing_schedule\['scale'\] entries must be in \[0, 1\]"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            max_elements=2,
            annealing_schedule={"step": [1.0], "scale": [1.5], "annealing_func": ["geometric"]},
            n_starts=2,
            steps=2,
        )
    with pytest.raises(ValueError, match=r"annealing_schedule\['annealing_func'\] entries must be one of"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            max_elements=2,
            annealing_schedule={"step": [1.0], "scale": [0.5], "annealing_func": ["exponential_decay"]},
            n_starts=2,
            steps=2,
        )


def test_optimize_composition_max_elements_trajectory_softens_to_hard():
    """The per-step trajectory shows annealing: early steps are softer (more nonzeros) than late."""
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    K = 3
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        n_starts=2,
        max_elements=K,
        steps=60,
        lr=0.2,
        record_weights_trajectory=True,
    )
    traj = res.weights_trajectory  # (steps, B, n)
    assert traj is not None and traj.shape[0] == 60
    early_nz = (traj[2] > 1e-3).sum(dim=-1).float().mean().item()  # avg #non-zero at step 2
    late_nz = (traj[-1] > 1e-3).sum(dim=-1).float().mean().item()  # avg #non-zero at last step
    # Early (large τ) should carry more spread mass; late (small τ) should be near K.
    assert early_nz > late_nz, f"annealing not visible in trajectory: early={early_nz}, late={late_nz}"
    # Late state should be at most a hair above K (in soft state; final returned is hard-projected).
    assert late_nz <= K + 1, f"final soft state too diffuse: {late_nz} non-zero (target K={K})"


def test_optimize_composition_max_elements_constant_schedule_no_anneal():
    """An ``annealing_func='constant'`` segment covering the full run holds τ; hard-projection still gives K."""
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    K = 4
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        n_starts=2,
        max_elements=K,
        annealing_scale=0.3,  # initial scale; segment will hold this
        annealing_schedule={"step": [1.0], "scale": [0.3], "annealing_func": ["constant"]},
        steps=40,
        lr=0.2,
    )
    w = res.optimized_weights
    nz = (w > 1e-6).sum(dim=-1)
    assert torch.all(nz == K)


def test_optimize_composition_annealing_scale_endpoints():
    """annealing_scale=0 and annealing_scale=1 both run cleanly and enforce K (the two endpoints
    of the user-facing knob; calibration: 0→τ_start=1, 0.5→5, 1→25)."""
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    K = 3
    for scale in (0.0, 1.0):
        res = model.optimize_composition(
            kernel,
            task_targets={"prop": 1.0},
            n_starts=2,
            max_elements=K,
            annealing_scale=scale,
            steps=30,
            lr=0.2,
        )
        nz = (res.optimized_weights > 1e-6).sum(dim=-1)
        assert torch.all(nz <= K), f"scale={scale}: nz={nz.tolist()}"


def test_optimize_composition_annealing_schedule_dict_overrides_front():
    """A dict with step[-1] < 1.0 takes over the front; the tail falls back to default."""
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    K = 3
    # Use a two-segment dict that only covers the first 50% of steps.
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        n_starts=2,
        max_elements=K,
        annealing_scale=0.5,
        annealing_schedule={
            "step": [0.2, 0.5],
            "scale": [0.9, 0.7],
            "annealing_func": ["linear", "cosine"],
        },
        steps=60,
        lr=0.2,
        record_weights_trajectory=True,
    )
    # Hard-projected final still has exactly K (this run is non-degenerate).
    nz = (res.optimized_weights > 1e-6).sum(dim=-1)
    assert torch.all(nz <= K)
    # Sanity: the trajectory was recorded — used by visualisation downstream.
    assert res.weights_trajectory is not None and res.weights_trajectory.shape[0] == 60


def test_optimize_composition_max_elements_gradient_flows_to_all_logits():
    """The soft top-K must let gradient flow back to logits at *all* positions, not just the chosen K.

    This is the qualitative difference vs. a post-hoc projection: at any τ > 0, all positions
    carry non-trivial gradient — so the optimiser can re-select which K to include.
    """
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    n = kernel.shape[0]
    # Manually replicate one step at a moderate τ to peek at the gradient pattern.
    logits = torch.zeros(1, n, requires_grad=True)
    w_soft = torch.softmax(logits, dim=-1)
    # Build the soft top-K mask inline (mirrors the production code).
    K, tau = 3, 0.5
    alpha = logits.clone()
    m = torch.zeros_like(logits)
    for _ in range(K):
        p = torch.softmax(alpha / tau, dim=-1)
        m = m + p
        alpha = alpha + torch.log((1.0 - p).clamp(min=1e-12))
    w = (w_soft * m) / (w_soft * m).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    # Loss against an arbitrary target → gradient should populate everywhere.
    target = torch.zeros_like(w)
    target[0, 5] = 1.0
    loss = ((w - target) ** 2).mean()
    loss.backward()
    assert logits.grad is not None
    # All entries (not just K) should have non-trivial gradient.
    abs_grad = logits.grad.abs()
    n_nontrivial = int((abs_grad > 1e-8).sum().item())
    assert n_nontrivial == n, f"expected gradient at all {n} positions, got {n_nontrivial}"


def test_optimize_composition_fixed_amounts_pins_single_symbol():
    """fixed_amounts={'Au': 0.65} holds Au at exactly 0.65; the remaining 0.35 spreads to others."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        fixed_amounts={"Au": 0.65},
        n_starts=3,
        steps=40,
        lr=0.2,
    )
    w = res.optimized_weights
    au = elements.index("Au")
    # Au at exactly the pinned value across the batch (atol=1e-4 same as other lock tests).
    assert torch.allclose(w[:, au], torch.full((3,), 0.65, dtype=w.dtype), atol=1e-4)
    # Remaining mass is 1 - 0.65 = 0.35 across the non-Au columns.
    rest_sum = w.sum(dim=-1) - w[:, au]
    assert torch.allclose(rest_sum, torch.full((3,), 0.35, dtype=w.dtype), atol=1e-4)


def test_optimize_composition_fixed_amounts_multi_symbol():
    """Two pinned elements both hold at their assigned values; the rest sum to 1 - Σ fixed."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        fixed_amounts={"Au": 0.65, "Ga": 0.20},
        n_starts=2,
        steps=40,
        lr=0.2,
    )
    w = res.optimized_weights
    au, ga = elements.index("Au"), elements.index("Ga")
    assert torch.allclose(w[:, au], torch.full((2,), 0.65, dtype=w.dtype), atol=1e-4)
    assert torch.allclose(w[:, ga], torch.full((2,), 0.20, dtype=w.dtype), atol=1e-4)
    rest = w.sum(dim=-1) - w[:, au] - w[:, ga]
    assert torch.allclose(rest, torch.full((2,), 0.15, dtype=w.dtype), atol=1e-4)


def test_optimize_composition_fixed_amounts_works_without_initial_weights():
    """fixed_amounts does not require initial_weights (unlike element_step_scale=0)."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()
    # No ``initial_weights`` → uses n_starts random init. Should succeed.
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        fixed_amounts={"Au": 0.5},
        n_starts=4,
        steps=20,
        lr=0.2,
    )
    au = elements.index("Au")
    assert torch.allclose(
        res.optimized_weights[:, au], torch.full((4,), 0.5, dtype=res.optimized_weights.dtype), atol=1e-4
    )


def test_optimize_composition_fixed_amounts_with_max_elements():
    """K=3 + 2 fixed → exactly 3 non-zero per row, with the 2 fixed at their pinned values."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()
    K = 3
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        fixed_amounts={"Au": 0.65, "Ga": 0.20},
        max_elements=K,
        n_starts=2,
        steps=60,
        lr=0.2,
    )
    w = res.optimized_weights
    nz = (w > 1e-6).sum(dim=-1)
    assert torch.all(nz <= K), f"got nz={nz.tolist()}, expected ≤ {K}"
    au, ga = elements.index("Au"), elements.index("Ga")
    assert torch.allclose(w[:, au], torch.full((2,), 0.65, dtype=w.dtype), atol=1e-4)
    assert torch.allclose(w[:, ga], torch.full((2,), 0.20, dtype=w.dtype), atol=1e-4)


def test_optimize_composition_fixed_amounts_with_allowed_elements():
    """A fixed element not in allowed_elements is contradictory and rejected."""
    model, kernel, _ = _build_aligned_model_and_kernel()
    with pytest.raises(ValueError, match="not in allowed_elements"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            fixed_amounts={"Au": 0.65},
            allowed_elements=["Mg", "Ga", "Cu"],  # Au absent
            n_starts=2,
            steps=2,
        )


def test_optimize_composition_fixed_amounts_mutex_with_element_step_scale_zero():
    """The same symbol in both fixed_amounts and element_step_scale=0 is ambiguous → reject."""
    model, kernel, elements = _build_aligned_model_and_kernel()
    init_w = torch.zeros(1, len(elements))
    init_w[0, elements.index("Au")] = 0.40
    init_w[0, elements.index("Cu")] = 0.60
    with pytest.raises(ValueError, match="appear in both element_step_scale=0 and"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            initial_weights=init_w,
            fixed_amounts={"Au": 0.65},
            element_step_scale={"Au": 0.0},
            steps=2,
        )


def test_optimize_composition_fixed_amounts_validation():
    """Sum<1, value range, type, unknown symbols — every guard fires before model state is touched."""
    model, kernel, _ = _build_aligned_model_and_kernel()
    # Sum >= 1.0 rejected (no leftover mass).
    with pytest.raises(ValueError, match="must be strictly less than 1.0"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            fixed_amounts={"Au": 0.7, "Ga": 0.4},
            n_starts=2,
            steps=2,
        )
    # Value out of (0, 1).
    with pytest.raises(ValueError, match="must be strictly between 0 and 1"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            fixed_amounts={"Au": 0.0},
            n_starts=2,
            steps=2,
        )
    with pytest.raises(ValueError, match="must be strictly between 0 and 1"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            fixed_amounts={"Au": 1.0},
            n_starts=2,
            steps=2,
        )
    # Unknown symbol.
    with pytest.raises(ValueError, match="Unknown element symbol"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            fixed_amounts={"NotAnElement": 0.5},
            n_starts=2,
            steps=2,
        )
    # Empty mapping.
    with pytest.raises(ValueError, match="must be non-empty"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            fixed_amounts={},
            n_starts=2,
            steps=2,
        )
    # Wrong type.
    with pytest.raises(TypeError, match="must be a mapping"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            fixed_amounts=[("Au", 0.5)],  # type: ignore[arg-type]
            n_starts=2,
            steps=2,
        )
    # max_elements <= n_locked when fixed_amounts present.
    with pytest.raises(ValueError, match="must be > total locked elements"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            fixed_amounts={"Au": 0.4, "Ga": 0.3},
            max_elements=2,
            n_starts=2,
            steps=2,
        )


def test_optimize_composition_max_elements_K_equals_one_one_hot():
    """K=1 produces a one-hot recipe (the smallest cardinality; exercises the n_iter=1 branch
    of the iterative softmax).
    """
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        n_starts=3,
        max_elements=1,
        steps=40,
        lr=0.2,
    )
    w = res.optimized_weights
    nz = (w > 1e-6).sum(dim=-1)
    assert torch.all(nz == 1), f"expected one-hot per row, got nz={nz.tolist()}"
    assert torch.allclose(w.sum(dim=-1), torch.ones(3, dtype=w.dtype), atol=1e-5)
    # The single non-zero must be exactly 1.0.
    max_per_row = w.max(dim=-1).values
    assert torch.allclose(max_per_row, torch.ones(3, dtype=w.dtype), atol=1e-5)


def test_optimize_composition_combined_locks_exceeding_simplex_rejected():
    """element_step_scale=0 (locking seed-heavy elements) + fixed_amounts together cannot
    claim more than 100% of the simplex — the combined-lock runtime check catches this."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()
    # Seed with Mg=0.50 (locked) + Cu=0.50 (free); fixed_amounts={"Au": 0.65}. Combined locked
    # mass would be 0.50 (Mg) + 0.65 (Au) = 1.15 > 1.0.
    init_w = torch.zeros(1, len(elements))
    init_w[0, elements.index("Mg")] = 0.50
    init_w[0, elements.index("Cu")] = 0.50
    with pytest.raises(ValueError, match="Combined locked mass exceeds 1.0"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            initial_weights=init_w,
            element_step_scale={"Mg": 0.0},
            fixed_amounts={"Au": 0.65},
            steps=2,
        )


def test_optimize_composition_max_elements_equals_n_locked_rejected():
    """``max_elements == n_locked`` is rejected (no unlocked slot to absorb leftover mass).

    Previously this combination passed validation but silently produced rows with sum < 1
    when the locked seed values summed to less than 1 (e.g. Mg=Al=Cu=0.20, max_elements=3).
    The validation now enforces strict ``max_elements > n_locked`` for both lock paths.
    """
    model, kernel, elements = _build_aligned_model_and_kernel()
    init_w = torch.zeros(1, len(elements))
    init_w[0, elements.index("Mg")] = 0.30
    init_w[0, elements.index("Al")] = 0.30
    init_w[0, elements.index("Cu")] = 0.40
    with pytest.raises(ValueError, match="must be > total locked elements"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            initial_weights=init_w,
            element_step_scale={"Mg": 0.0, "Al": 0.0, "Cu": 0.0},
            max_elements=3,
            steps=2,
        )


def test_optimize_composition_min_nonzero_weight_drops_traces():
    """A floor of 0.1 makes every non-zero unlocked element ≥ 0.1 (no trace amounts)."""
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        n_starts=4,
        max_elements=5,
        min_nonzero_weight=0.1,
        steps=80,
        lr=0.2,
    )
    w = res.optimized_weights
    # Every non-zero weight is at least the floor (within float tolerance).
    nz_mask = w > 0
    assert (w[nz_mask] >= 0.1 - 1e-5).all(), f"floor violated: smallest non-zero = {w[nz_mask].min().item():.4f}"
    # Rows still sum to 1.
    assert torch.allclose(w.sum(dim=-1), torch.ones(4, dtype=w.dtype), atol=1e-5)


def test_optimize_composition_min_nonzero_weight_zero_is_noop():
    """min_nonzero_weight=0.0 (default) produces identical output to omitting the kwarg."""
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    kernel = torch.randn(6, INPUT_DIM)
    init = torch.full((2, 6), 1.0 / 6)
    base = model.optimize_composition(kernel, task_targets={"prop": 1.0}, initial_weights=init, steps=20, lr=0.1)
    torch.manual_seed(0)
    floored = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        initial_weights=init,
        min_nonzero_weight=0.0,
        steps=20,
        lr=0.1,
    )
    assert torch.allclose(base.optimized_weights, floored.optimized_weights, atol=1e-6)


def test_optimize_composition_min_nonzero_weight_with_fixed_amounts_below_floor():
    """A floor higher than any fixed amount is contradictory — caught at kwarg time."""
    model, kernel, _ = _build_aligned_model_and_kernel()
    with pytest.raises(ValueError, match="below min_nonzero_weight"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            fixed_amounts={"Au": 0.05, "Ga": 0.20},
            min_nonzero_weight=0.10,
            n_starts=2,
            steps=2,
        )


def test_optimize_composition_min_nonzero_weight_with_fixed_amounts_compatible():
    """A floor ≤ all fixed amounts works; fixed Au stays at 0.05 even with floor=0.05."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        fixed_amounts={"Au": 0.30, "Ga": 0.20},
        min_nonzero_weight=0.10,
        max_elements=4,
        n_starts=2,
        steps=40,
        lr=0.2,
    )
    w = res.optimized_weights
    au = elements.index("Au")
    assert torch.allclose(w[:, au], torch.full((2,), 0.30, dtype=w.dtype), atol=1e-4)
    nz_mask = w > 0
    assert (w[nz_mask] >= 0.10 - 1e-5).all()


def test_optimize_composition_min_nonzero_weight_with_locked_seed_below_floor():
    """An element_step_scale=0 lock pinning at a seed value below the floor is rejected."""
    model, kernel, elements = _build_aligned_model_and_kernel()
    init_w = torch.zeros(1, len(elements))
    init_w[0, elements.index("Mg")] = 0.05  # locked below floor
    init_w[0, elements.index("Cu")] = 0.95
    with pytest.raises(ValueError, match="locked element.*below min_nonzero_weight"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            initial_weights=init_w,
            element_step_scale={"Mg": 0.0},
            min_nonzero_weight=0.10,
            steps=2,
        )


def test_optimize_composition_min_nonzero_weight_validation():
    """floor out of [0,1], floor > 1/max_elements — all rejected pre-state."""
    model, kernel, _ = _build_aligned_model_and_kernel()
    with pytest.raises(ValueError, match=r"min_nonzero_weight must be in \[0, 1\]"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            min_nonzero_weight=-0.1,
            n_starts=2,
            steps=2,
        )
    with pytest.raises(ValueError, match=r"min_nonzero_weight must be in \[0, 1\]"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            min_nonzero_weight=1.5,
            n_starts=2,
            steps=2,
        )
    # floor > 1/K: K=3 → 1/K=0.333; floor=0.5 > 0.333 → reject.
    with pytest.raises(ValueError, match="exceeds 1 / max_elements"):
        model.optimize_composition(
            kernel,
            task_targets={"prop": 0.0},
            min_nonzero_weight=0.5,
            max_elements=3,
            n_starts=2,
            steps=2,
        )


def test_optimize_composition_min_nonzero_weight_fallback_preserves_simplex():
    """When the floor would empty a row's unlocked mass, the row falls back to unfloored
    (preserves sum=1 instead of breaking the simplex)."""
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    # Use a high floor + small K so dropping below-floor positions could empty unlocked.
    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 1.0},
        n_starts=4,
        max_elements=3,
        min_nonzero_weight=0.33,  # at the edge: needs all 3 ≥ 0.33 → exactly 1/3 each
        steps=40,
        lr=0.2,
    )
    w = res.optimized_weights
    # Whatever the floor did, the simplex must be preserved.
    assert torch.allclose(w.sum(dim=-1), torch.ones(4, dtype=w.dtype), atol=1e-5)
    assert (w >= 0).all()


def test_optimize_latent_space_with_ae():
    model = _make_model()
    model.eval()
    x = torch.randn(2, INPUT_DIM)
    result = model.optimize_latent(
        task_name="prop",
        initial_input=x,
        target_value=1.0,
        steps=5,
        num_restarts=2,
        optimize_space="latent",
    )
    assert result.optimized_input.shape == (2, 2, INPUT_DIM)
    assert result.optimized_target.shape == (2, 2, 1)
    assert result.trajectory.shape == (2, 2, 5, 1)


# --- LR scheduler cadence ---------------------------------------------------------------------


class _SchedulerStepCounter(FlexibleMultiTaskModel):
    """Records every scheduler ``step()`` call so a test can assert the cadence.

    A subclass rather than patching the bound method: Lightning checks that
    ``configure_optimizers`` is *defined* on the model, and a MagicMock does not satisfy that
    check, so patching it makes the Trainer refuse to run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_calls: list[object] = []
        self.n_schedulers = 0

    def configure_optimizers(self):
        result = super().configure_optimizers()
        entries = result if isinstance(result, list) else [result]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            scheduler = entry.get("lr_scheduler", {}).get("scheduler")
            if scheduler is None:
                continue
            self.n_schedulers += 1
            original_step = scheduler.step

            def counting_step(*args, _orig=original_step, **kwargs):
                self.step_calls.append(args[0] if args else None)
                return _orig(*args, **kwargs)

            scheduler.step = counting_step
        return result


def test_scheduler_steps_once_per_epoch_not_per_batch(model_config_mixed_tasks, dummy_compound_datamodule):
    """``patience`` must count epochs, so the scheduler must be stepped once per epoch.

    History this guards against: the model once used manual optimization — forced by having one
    optimizer per parameter group, since Lightning drives at most one automatically — and manual
    optimization means Lightning does not drive schedulers either, so the model stepped them
    itself. It did that inside ``training_step``, once per *batch*, which made
    ReduceLROnPlateau's ``patience`` count batches: on a 24k-row task at ``batch_size = 256``
    (~90 batches/epoch) the LR reached ``min_lr`` inside the first epoch, and a whole tuning
    campaign was run and discarded before anyone noticed.

    The optimizers are now collapsed into one, so Lightning owns the scheduler and steps it at the
    interval configure_optimizers declares. The model cannot get the cadence wrong because it no
    longer chooses it — but this still asserts the RUNTIME cadence rather than that declaration,
    because last time the config read perfectly while the runtime did the wrong thing.
    """
    config = model_config_mixed_tasks
    model = _SchedulerStepCounter(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=OptimizerConfig(lr=1e-3, min_lr=1e-6),
    )
    # Every head gets a live scheduler too, so the count below covers all parameter groups.
    for task_config in model.task_configs_map.values():
        task_config.optimizer = OptimizerConfig(lr=1e-3, min_lr=1e-6)

    epochs, batches_per_epoch = 2, 3
    trainer = L.Trainer(
        logger=False,
        max_epochs=epochs,
        limit_train_batches=batches_per_epoch,
        limit_val_batches=1,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dummy_compound_datamodule)

    assert model.n_schedulers > 0, "no scheduler was created, so the cadence is untested"
    assert len(model.step_calls) == model.n_schedulers * epochs, (
        f"expected {model.n_schedulers} scheduler(s) x {epochs} epochs = "
        f"{model.n_schedulers * epochs} steps; got {len(model.step_calls)} over "
        f"{batches_per_epoch} batches/epoch — per-batch stepping would give "
        f"{model.n_schedulers * epochs * batches_per_epoch}"
    )
    # Each step receives the epoch-aggregated monitored metric, not a raw per-batch loss.
    assert all(value is not None for value in model.step_calls)


def test_scheduler_monitor_missing_raises(model_config_mixed_tasks, dummy_compound_datamodule):
    """A monitor that never appears must fail loudly — a scheduler that silently never anneals
    is invisible in logs, which is the failure mode this whole area kept producing.

    Every group has to carry the SAME live scheduler here. The fixture's heads switch theirs off,
    so leaving them alone makes the model fail in configure_optimizers on the mixed on/off policy
    instead — an error that also happens to quote the monitor name, so the assertion below would
    pass without a single epoch ever running.
    """
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=OptimizerConfig(lr=1e-3, min_lr=1e-6, monitor="no_such_metric"),
    )
    for task_config in model.task_configs_map.values():
        task_config.optimizer = OptimizerConfig(lr=1e-3, min_lr=1e-6, monitor="no_such_metric")

    trainer = L.Trainer(
        logger=False,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(MisconfigurationException, match="no_such_metric"):
        trainer.fit(model, datamodule=dummy_compound_datamodule)


def test_no_optimizer_step_when_loss_has_no_graph(model_config_mixed_tasks, sample_batch_mixed_tasks, mocker):
    """A batch whose loss carries no graph must be skipped, not backwarded.

    Under automatic optimization the way to skip a batch is to return None — Lightning then
    performs no backward and no optimizer step. Returning the graph-free tensor instead would have
    Lightning try to backward a graph that does not exist.

    The manual-optimization version of this test guarded a subtler bug: the else-branch called
    opt.step() on every optimizer right after logging that it was *skipping* the step, which was a
    no-op only because zero_grad(set_to_none=True) leaves grads as None. Handing the loop to
    Lightning removes the branch and the bug with it.
    """
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    mocker.patch.object(model, "log_dict")
    mocker.patch.object(model, "log")

    assert model.training_step(sample_batch_mixed_tasks, 0) is None, (
        "a graph-free batch must be skipped by returning None, not by returning a detached loss"
    )


# --- characterization: the three steps' observable output must not drift -----------------------


def _run_step_and_capture(model, batch, step_name, mocker):
    """Every value the step publishes: its return, and the full logged key/value set."""
    captured: dict[str, float] = {}

    def record_dict(d, *args, **kwargs):
        for key, value in d.items():
            captured[key] = float(value)

    def record_one(key, value, *args, **kwargs):
        captured[key] = float(value)

    mocker.patch.object(model, "log_dict", side_effect=record_dict)
    mocker.patch.object(model, "log", side_effect=record_one)
    # No optimizer or backward mocking: under automatic optimization every step is a pure function
    # of the batch that logs and returns, and touches neither.
    result = getattr(model, step_name)(batch, 0)
    return (None if result is None else float(result)), captured


@pytest.mark.parametrize("step_name", ["training_step", "validation_step", "test_step"])
def test_step_output_is_stable(model_config_mixed_tasks, sample_batch_mixed_tasks, mocker, step_name):
    """Pin what each step returns and logs.

    These functions have been rewritten several times and carry the loss that actually trains the
    model, so a refactor has to prove it changed nothing observable. This captures the return value
    and every logged key/value from a fixed batch with fixed weights; any drift in loss
    composition, weighting, masking or metric naming fails here rather than silently changing
    training.
    """
    torch.manual_seed(0)
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.01)

    returned, logged = _run_step_and_capture(model, sample_batch_mixed_tasks, step_name, mocker)

    prefix = {"training_step": "train", "validation_step": "val", "test_step": "test"}[step_name]
    task_names = [c.name for c in config.task_configs]

    # Structure: one raw loss, one all_missing flag, one weight and one contribution per task that
    # produced a loss, plus the aggregate keys.
    for name in task_names:
        assert f"{prefix}_{name}_all_missing" in logged, f"missing all_missing flag for {name}"
    assert f"{prefix}_final_supervised_loss" in logged
    assert f"{prefix}_final_loss" in logged

    # Values: the aggregate must equal the sum of the per-task contributions it reports.
    contributions = sum(v for k, v in logged.items() if k.endswith("_final_loss_contrib"))
    assert logged[f"{prefix}_final_supervised_loss"] == pytest.approx(contributions, rel=1e-5)
    assert logged[f"{prefix}_final_loss"] == pytest.approx(contributions, rel=1e-5)
    if returned is not None:
        assert returned == pytest.approx(contributions, rel=1e-5)

    # Weighting: with no learnable balancer, each contribution is weight x raw loss.
    for name in task_names:
        raw_key, weight_key = f"{prefix}_{name}_raw_loss", f"{prefix}_{name}_static_weight"
        contrib_key = f"{prefix}_{name}_final_loss_contrib"
        if raw_key in logged:
            assert logged[contrib_key] == pytest.approx(logged[raw_key] * logged[weight_key], rel=1e-5)


# --- encoder-config normalisation and the typed head accessor ---------------------------------


def test_encoder_config_accepts_a_concrete_config_unchanged():
    """A ready-made config must reach the encoder as-is.

    __init__ used to branch on isinstance(..., BaseEncoderConfig) to skip build_encoder_config;
    that branch is gone because build_encoder_config already passes such a config through. This
    pins the behaviour the removed branch used to provide.
    """
    encoder_config = MLPEncoderConfig(hidden_dims=[5, 8, 4])
    model = FlexibleMultiTaskModel(
        task_configs=[RegressionTaskConfig(name="t", data_column="c", dims=[4, 3, 1])],
        encoder_config=encoder_config,
        shared_block_optimizer=OptimizerConfig(lr=1e-3),
    )
    assert model.encoder_config is encoder_config
    assert model.latent_dim == 4


def test_encoder_config_accepts_a_mapping():
    """A mapping is normalised into a dataclass by the same single call."""
    model = FlexibleMultiTaskModel(
        task_configs=[RegressionTaskConfig(name="t", data_column="c", dims=[4, 3, 1])],
        encoder_config={"hidden_dims": [5, 8, 4]},
        shared_block_optimizer=OptimizerConfig(lr=1e-3),
    )
    assert isinstance(model.encoder_config, MLPEncoderConfig)
    assert model.latent_dim == 4


def test_encoder_config_rejects_a_subclass_the_encoder_cannot_use():
    """The removed branch tested the ABSTRACT base, so a third subclass passed it and then failed
    deeper in FoundationEncoder. Routing through build_encoder_config rejects it at construction."""

    @dataclass(kw_only=True)
    class _ThirdKind(BaseEncoderConfig):
        type: EncoderType = EncoderType.MLP

        @property
        def latent_dim(self) -> int:
            return 4

    with pytest.raises(TypeError, match="MLPEncoderConfig"):
        FlexibleMultiTaskModel(
            task_configs=[RegressionTaskConfig(name="t", data_column="c", dims=[4, 3, 1])],
            encoder_config=_ThirdKind(),
            shared_block_optimizer=OptimizerConfig(lr=1e-3),
        )


def test_head_accessor_returns_the_registered_head_for_every_kind(model_config_mixed_tasks):
    """_head must resolve to the registered instance for each supported head type.

    It replaces 13 raw ModuleDict lookups whose type a checker could not resolve, so it is the one
    place the "every value here is a BaseTaskHead" invariant is stated.
    """
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    assert model.task_heads, "fixture must register at least one head"
    for name in model.task_heads:
        head = model._head(name)
        assert head is model.task_heads[name]
        assert isinstance(head, BaseTaskHead)
        assert callable(head.compute_loss)


# --- learnable loss balancer (uncertainty weighting) -------------------------------------------


def test_loss_balancer_registers_one_log_sigma_per_supervised_task(model_config_mixed_tasks):
    """Kendall/Gal/Cipolla uncertainty weighting needs one learnable log sigma per task.

    The feature has been implemented on the model since before the [training] section existed, but
    nothing routed a value to it, so it had never run. These two tests are the on/off comparison
    that absence made impossible.
    """
    config = model_config_mixed_tasks
    off = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        enable_learnable_loss_balancer=False,
    )
    on = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        enable_learnable_loss_balancer=True,
    )

    assert len(off.task_log_sigmas) == 0, "no sigmas may exist while the balancer is off"
    supervised = [n for n in on.task_heads if n != "__reconstruction__"]
    assert sorted(on.task_log_sigmas) == sorted(supervised)
    # They must be learnable, and start at log sigma = 0 (sigma = 1), i.e. the unweighted objective.
    assert all(p.requires_grad for p in on.task_log_sigmas.parameters())
    assert all(p.detach().item() == 0.0 for p in on.task_log_sigmas.values())

    # And they must reach an optimizer, or they would never move. configure_optimizers puts them
    # in the main (encoder) group, so look for them there rather than trusting that it did.
    # The fixture gives some heads scheduler_enabled=False while the encoder has it on — a mix a
    # single scheduler cannot represent, and which configure_optimizers now rejects. Pin one policy
    # here: this test is about the balancer, not about that constraint.
    for task_config in on.task_configs_map.values():
        task_config.optimizer = OptimizerConfig(lr=1e-3, min_lr=1e-6)

    result = cast(dict[str, Any], on.configure_optimizers())
    optimised = {id(p) for group in result["optimizer"].param_groups for p in group["params"]}
    assert all(id(p) in optimised for p in on.task_log_sigmas.parameters()), (
        "log sigmas exist but no optimizer owns them — they would stay at their init value forever"
    )


def test_loss_balancer_changes_the_objective_and_its_gradient(model_config_mixed_tasks):
    """At log sigma = 0 the balanced objective is HALF the static one, plus a log-sigma penalty.

    That factor is not cosmetic: it halves every gradient reaching the shared encoder on the first
    step, which is exactly the kind of silent scale change that has to be visible in a test before
    anyone reads an A/B of this feature.
    """
    config = model_config_mixed_tasks
    losses = {"regr_task_1": torch.tensor(4.0, requires_grad=True)}

    off = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        enable_learnable_loss_balancer=False,
    )
    on = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        enable_learnable_loss_balancer=True,
    )

    logs_off: dict[str, torch.Tensor] = {}
    logs_on: dict[str, torch.Tensor] = {}
    total_off = off._weighted_total_loss(
        stage="train", raw_losses=losses, logs=logs_off, device=torch.device("cpu"), keep_graph=True
    )
    total_on = on._weighted_total_loss(
        stage="train", raw_losses=losses, logs=logs_on, device=torch.device("cpu"), keep_graph=True
    )

    assert float(total_off) == pytest.approx(4.0)
    # 0.5 * exp(-2*0) * 4.0 + 0 = 2.0
    assert float(total_on) == pytest.approx(2.0)
    # The balancer reports the sigma it is using; the static path does not.
    assert "train_regr_task_1_sigma_t" in logs_on
    assert "train_regr_task_1_sigma_t" not in logs_off
    assert float(logs_on["train_regr_task_1_sigma_t"]) == pytest.approx(1.0)

    # The log sigma must carry gradient, or the weighting could never adapt.
    total_on.backward()
    assert on.task_log_sigmas["regr_task_1"].grad is not None
    assert float(on.task_log_sigmas["regr_task_1"].grad) != 0.0
