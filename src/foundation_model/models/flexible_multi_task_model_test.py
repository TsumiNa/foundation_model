# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for FlexibleMultiTaskModel, including integration with CompoundDataModule.
"""

from types import SimpleNamespace

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import (
    ClassificationTaskConfig,
    OptimizerConfig,
    RegressionTaskConfig,
    TaskType,
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
    deposit_dim = shared_dims[-1]  # Input to task heads

    task_configs_list = [
        RegressionTaskConfig(
            name="regr_task_1",
            type=TaskType.REGRESSION,
            dims=[deposit_dim, 64, 1],  # deposit_dim -> hidden_in_head -> output_dim (1 for this task)
            optimizer=OptimizerConfig(lr=1e-4, scheduler_type="None"),
        ),
        ClassificationTaskConfig(
            name="clf_task_1",
            type=TaskType.CLASSIFICATION,
            dims=[deposit_dim, 64, 3],  # deposit_dim -> hidden_in_head -> num_classes (3 for this task)
            num_classes=3,
            optimizer=OptimizerConfig(lr=1e-4, scheduler_type="None"),
        ),
        RegressionTaskConfig(
            name="regr_task_2",  # Another regression task
            type=TaskType.REGRESSION,
            dims=[deposit_dim, 32, 2],  # deposit_dim -> hidden_in_head -> output_dim (2 for this task)
            optimizer=OptimizerConfig(lr=1e-4, scheduler_type="None"),
        ),
    ]

    config_dict = {
        "shared_block_dims": shared_dims,
        "task_configs": task_configs_list,
        "norm_shared": True,
        "residual_shared": False,
        "shared_block_optimizer": OptimizerConfig(lr=1e-3, scheduler_type="None"),
        "with_structure": False,
        "struct_block_dims": None,
        "modality_dropout_p": 0.0,
        "enable_self_supervised_training": False,
        "loss_weights": None,  # Default weights will be used
        "mask_ratio": 0.15,  # Default, not used if SSL is false
        "temperature": 0.07,  # Default, not used if SSL is false
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
    temps_batch = {}
    return (x_formula, y_dict_batch, task_masks_batch, temps_batch)


# --- Unit Tests for Model Components ---


def test_model_initialization(model_config_mixed_tasks):
    """Test model initialization with mixed regression and classification tasks."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_configs=config.task_configs,
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        shared_block_optimizer=config.shared_block_optimizer,
        with_structure=config.with_structure,
        struct_block_dims=config.struct_block_dims,
        modality_dropout_p=config.modality_dropout_p,
        enable_self_supervised_training=config.enable_self_supervised_training,
        loss_weights=config.loss_weights,
        mask_ratio=config.mask_ratio,
        temperature=config.temperature,
    )

    assert model.encoder is not None, "Encoder should be initialized"
    if not config.with_structure:
        assert hasattr(model.encoder, "shared"), "Encoder should have 'shared' attribute"
        assert hasattr(model.encoder, "deposit"), "Encoder should have 'deposit' attribute"

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

    assert model.with_structure == config.with_structure
    assert model.enable_self_supervised_training == config.enable_self_supervised_training
    assert model.automatic_optimization, "automatic_optimization should be True by default"


def test_model_forward_pass(model_config_mixed_tasks, sample_batch_mixed_tasks):
    """Test the forward pass for mixed regression and classification predictions."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_configs=config.task_configs,
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        shared_block_optimizer=config.shared_block_optimizer,
        with_structure=config.with_structure,
        struct_block_dims=config.struct_block_dims,
        modality_dropout_p=config.modality_dropout_p,
        enable_self_supervised_training=config.enable_self_supervised_training,
        loss_weights=config.loss_weights,
        mask_ratio=config.mask_ratio,
        temperature=config.temperature,
    )
    model.eval()  # Set model to evaluation mode

    x_formula, _, _, temps_batch = sample_batch_mixed_tasks  # y_dict and masks not needed for forward pass directly

    # Forward pass expects x_formula (or (x_formula, x_struct)) and temps_batch
    output = model(x_formula, temps_batch=temps_batch)

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
        shared_block_dims=config.shared_block_dims,
        task_configs=config.task_configs,
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        shared_block_optimizer=config.shared_block_optimizer,
        with_structure=config.with_structure,
        struct_block_dims=config.struct_block_dims,
        modality_dropout_p=config.modality_dropout_p,
        enable_self_supervised_training=config.enable_self_supervised_training,  # False for this config
        loss_weights=config.loss_weights,
        mask_ratio=config.mask_ratio,
        temperature=config.temperature,
    )
    model.train()  # Set model to training mode

    mock_log_dict = mocker.patch.object(model, "log_dict")

    # sample_batch_mixed_tasks provides (x_formula, y_dict_batch, task_masks_batch, temps_batch)
    # This matches the expected input for training_step
    loss = model.training_step(sample_batch_mixed_tasks, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Loss should be a Tensor"
    assert loss.requires_grad, "Loss should require gradients for backpropagation"
    assert loss.ndim == 0, "Loss should be a scalar"

    mock_log_dict.assert_called()
    logged_metrics = mock_log_dict.call_args[0][0]

    assert "train_total_loss" in logged_metrics, "train_total_loss should be logged"

    enabled_tasks_in_config = [tc for tc in config.task_configs if tc.enabled]
    for task_cfg in enabled_tasks_in_config:
        assert f"train_{task_cfg.name}_loss" in logged_metrics, f"train_{task_cfg.name}_loss should be logged"
        assert isinstance(logged_metrics[f"train_{task_cfg.name}_loss"], torch.Tensor), (
            f"train_{task_cfg.name}_loss should be a Tensor"
        )
        assert f"train_{task_cfg.name}_loss_weighted" in logged_metrics, (
            f"train_{task_cfg.name}_loss_weighted should be logged"
        )

    # Ensure SSL losses are not logged as enable_self_supervised_training=False
    assert "train_mfm_loss" not in logged_metrics
    assert "train_contrastive_loss" not in logged_metrics
    assert "train_cross_recon_loss" not in logged_metrics
    # train_modality_dropout_applied is only logged if SSL and with_structure are true
    assert "train_modality_dropout_applied" not in logged_metrics


def test_model_validation_step(model_config_mixed_tasks, sample_batch_mixed_tasks, mocker):
    """Test the validation_step for mixed regression and classification tasks."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_configs=config.task_configs,
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        shared_block_optimizer=config.shared_block_optimizer,
        with_structure=config.with_structure,
        struct_block_dims=config.struct_block_dims,
        modality_dropout_p=config.modality_dropout_p,
        enable_self_supervised_training=config.enable_self_supervised_training,  # False for this config
        loss_weights=config.loss_weights,
        mask_ratio=config.mask_ratio,
        temperature=config.temperature,
    )
    model.eval()  # Set model to evaluation mode

    mock_log_dict = mocker.patch.object(model, "log_dict")

    # validation_step now returns None and logs metrics via self.log_dict
    result = model.validation_step(sample_batch_mixed_tasks, batch_idx=0)
    assert result is None, "validation_step should return None"

    mock_log_dict.assert_called()
    logged_metrics = mock_log_dict.call_args[0][0]

    assert "val_total_loss" in logged_metrics, "val_total_loss should be logged"

    enabled_tasks_in_config = [tc for tc in config.task_configs if tc.enabled]
    for task_cfg in enabled_tasks_in_config:
        assert f"val_{task_cfg.name}_loss" in logged_metrics, f"val_{task_cfg.name}_loss should be logged"
        assert isinstance(logged_metrics[f"val_{task_cfg.name}_loss"], torch.Tensor), (
            f"val_{task_cfg.name}_loss should be a Tensor"
        )

    # Ensure SSL losses are not logged as enable_self_supervised_training=False
    assert "val_mfm_loss" not in logged_metrics
    assert "val_contrastive_loss" not in logged_metrics
    assert "val_cross_recon_loss" not in logged_metrics


def test_model_predict_step(model_config_mixed_tasks, sample_batch_mixed_tasks):
    """Test the predict_step for mixed regression and classification tasks."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_configs=config.task_configs,
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        shared_block_optimizer=config.shared_block_optimizer,
        with_structure=config.with_structure,
        struct_block_dims=config.struct_block_dims,
        modality_dropout_p=config.modality_dropout_p,
        enable_self_supervised_training=config.enable_self_supervised_training,
        loss_weights=config.loss_weights,
        mask_ratio=config.mask_ratio,
        temperature=config.temperature,
    )
    model.eval()  # Set model to evaluation mode

    x_formula, y_dict, task_masks, temps_batch = sample_batch_mixed_tasks

    # predict_step expects a batch tuple: (x_input, y_dict, masks_dict, temps_dict)
    # For prediction, y_dict and masks_dict can be the ones from the sample batch or None.
    # x_input for predict_step is typically just x_formula.
    predict_batch_tuple = (x_formula, y_dict, task_masks, temps_batch)

    output = model.predict_step(predict_batch_tuple, batch_idx=0, additional_output=True)

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
            pred_tensor = output[expected_key_value]
            assert isinstance(pred_tensor, torch.Tensor)
            expected_shape = (x_formula.shape[0], task_cfg.dims[-1])
            assert pred_tensor.shape == expected_shape, (
                f"Shape mismatch for {expected_key_value}. Expected {expected_shape}, got {pred_tensor.shape}"
            )

        elif task_cfg.type == TaskType.CLASSIFICATION:
            assert isinstance(task_cfg, ClassificationTaskConfig)
            expected_key_proba = f"{task_name_snake}_proba"  # Changed from _logits
            expected_key_label = f"{task_name_snake}_label"  # Changed from _labels

            assert expected_key_proba in output, f"Predict output should contain key '{expected_key_proba}'"
            proba_tensor = output[expected_key_proba]
            assert isinstance(proba_tensor, torch.Tensor)
            expected_proba_shape = (x_formula.shape[0], task_cfg.num_classes)
            assert proba_tensor.shape == expected_proba_shape, (
                f"Shape mismatch for {expected_key_proba}. Expected {expected_proba_shape}, got {proba_tensor.shape}"
            )

            assert expected_key_label in output, f"Predict output should contain key '{expected_key_label}'"
            label_tensor = output[expected_key_label]
            assert isinstance(label_tensor, torch.Tensor)
            expected_label_shape = (x_formula.shape[0],)  # Predicted labels are (B,)
            assert label_tensor.shape == expected_label_shape, (
                f"Shape mismatch for {expected_key_label}. Expected {expected_label_shape}, got {label_tensor.shape}"
            )
            assert label_tensor.dtype == torch.long, f"{expected_key_label} should be of type torch.long"


def test_model_configure_optimizers(model_config_mixed_tasks):
    """Test configure_optimizers for mixed regression and classification tasks."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_configs=config.task_configs,
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        shared_block_optimizer=config.shared_block_optimizer,
        with_structure=config.with_structure,  # False in this fixture
        struct_block_dims=config.struct_block_dims,
        modality_dropout_p=config.modality_dropout_p,
        enable_self_supervised_training=config.enable_self_supervised_training,  # False in this fixture
        loss_weights=config.loss_weights,
        mask_ratio=config.mask_ratio,
        temperature=config.temperature,
    )

    optimizers_and_schedulers = model.configure_optimizers()

    assert isinstance(optimizers_and_schedulers, list), "configure_optimizers should return a list"

    num_optimizers = 0
    encoder_opt_found = False
    task_head_opts_found_count = 0

    all_optimized_param_ids = set()
    enabled_tasks_in_config = [tc for tc in config.task_configs if tc.enabled]

    for item in optimizers_and_schedulers:
        num_optimizers += 1
        optimizer = item["optimizer"] if isinstance(item, dict) else item
        assert isinstance(optimizer, torch.optim.Optimizer), "Optimizer item is not a torch.optim.Optimizer"

        current_opt_param_ids = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                assert p.requires_grad, "Optimizer should only manage parameters that require gradients"
                current_opt_param_ids.add(id(p))

        assert all_optimized_param_ids.isdisjoint(current_opt_param_ids), (
            "Optimizers should manage disjoint sets of parameters"
        )
        all_optimized_param_ids.update(current_opt_param_ids)

        # Check if this optimizer handles encoder parameters
        # model.encoder contains all encodable parts (shared, deposit, struct_enc, fusion)
        encoder_params_ids = {id(p) for p in model.encoder.parameters() if p.requires_grad}
        if not encoder_params_ids.isdisjoint(current_opt_param_ids):
            # This optimizer is for the encoder (or part of it if encoder params are split, though not expected here)
            # For this simple config (no SSL, no structure), model.encoder params are what we expect for shared_block_optimizer
            assert current_opt_param_ids.issubset(
                encoder_params_ids
            )  # Optimizer params should be a subset of all encoder params
            encoder_opt_found = True  # Mark that an optimizer for encoder parts was found
            continue

        # Check if this optimizer handles one of the task heads
        found_task_head_for_this_opt = False
        for task_name, head in model.task_heads.items():
            head_params_ids = {id(p) for p in head.parameters() if p.requires_grad}
            if not head_params_ids.isdisjoint(current_opt_param_ids):
                assert current_opt_param_ids == head_params_ids, (
                    f"Optimizer does not manage all and only parameters for task head {task_name}"
                )
                task_head_opts_found_count += 1
                found_task_head_for_this_opt = True
                break

        if not found_task_head_for_this_opt and not encoder_opt_found:
            # If it wasn't an encoder optimizer, it must be a task head optimizer
            # This assertion might be too strict if encoder_params are split across multiple optimizers by configure_optimizers
            # For now, assume one optimizer for encoder, one for each head.
            pass  # Allow falling through if it's an encoder optimizer already marked

    # Expected optimizers: 1 for encoder + 1 for each enabled task
    expected_num_optimizers = 1 + len(enabled_tasks_in_config)
    assert num_optimizers == expected_num_optimizers, (
        f"Expected {expected_num_optimizers} optimizers, got {num_optimizers}"
    )
    assert encoder_opt_found, "No optimizer found for the encoder components"
    assert task_head_opts_found_count == len(enabled_tasks_in_config), (
        f"Expected {len(enabled_tasks_in_config)} optimizers for task heads, found {task_head_opts_found_count}"
    )

    # Verify all trainable model parameters are covered
    all_model_trainable_params_ids = {id(p) for p in model.parameters() if p.requires_grad}
    # Since SSL is false, ssl_module parameters are not included.
    assert all_optimized_param_ids == all_model_trainable_params_ids, (
        "Not all trainable model parameters are covered by the optimizers"
    )


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
    attributes_data = {}
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
            attributes_data[task_cfg.name] = np.random.randint(0, task_cfg.num_classes, num_samples)

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

    dm = CompoundDataModule(
        formula_desc_source=formula_df,
        attributes_source=attributes_df,
        task_configs=config.task_configs,
        with_structure=False,  # Matching model_config_mixed_tasks
        batch_size=batch_size,
        num_workers=0,
        # test_all=False, # Default
        # val_split and test_split are used if 'split' column is not present.
        # Since we provide 'split', these will be ignored by the DataModule's logic.
    )
    dm.setup()  # Call setup to prepare datasets
    return dm


# --- Integration Test with Trainer ---


def test_trainer_integration_mixed_tasks(model_config_mixed_tasks, dummy_compound_datamodule):
    """
    Test the model with pytorch_lightning.Trainer and CompoundDataModule
    for mixed regression and classification tasks.
    """
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_configs=config.task_configs,
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        shared_block_optimizer=config.shared_block_optimizer,
        with_structure=config.with_structure,
        struct_block_dims=config.struct_block_dims,
        modality_dropout_p=config.modality_dropout_p,
        enable_self_supervised_training=config.enable_self_supervised_training,
        loss_weights=config.loss_weights,
        mask_ratio=config.mask_ratio,
        temperature=config.temperature,
    )

    trainer = L.Trainer(
        fast_dev_run=True,  # Runs 1 batch for train, val, test and predict
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,  # Keep test output clean
    )

    # The main assertion is that this runs without errors.
    try:
        trainer.fit(model, datamodule=dummy_compound_datamodule)
        # If fast_dev_run is True, it also runs validation and test loops if defined.
        # And predict_loop if predict_dataloaders are available.
        # We can add more specific assertions on logged metrics if needed,
        # but fast_dev_run is a good smoke test.

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


# Helper for creating dummy dataframes
def create_dummy_dataframe(num_samples, num_features, index_prefix="sample_"):
    data = np.random.rand(num_samples, num_features)
    index = [f"{index_prefix}{i}" for i in range(num_samples)]
    return pd.DataFrame(data, index=index, columns=[f"feat_{j}" for j in range(num_features)])
