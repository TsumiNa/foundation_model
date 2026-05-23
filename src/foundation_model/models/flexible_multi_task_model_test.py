# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for FlexibleMultiTaskModel, including integration with CompoundDataModule.
"""

from pathlib import Path  # To use Path objects for directory manipulation
from types import SimpleNamespace

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from lightning.pytorch.loggers import CSVLogger

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import (
    ClassificationTaskConfig,
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
            optimizer=OptimizerConfig(lr=1e-4, scheduler_type="None"),
            loss_weight=1.0,
        ),
        ClassificationTaskConfig(
            name="clf_task_1",
            type=TaskType.CLASSIFICATION,
            dims=[latent_dim, 64, 3],
            data_column="clf_task_1_classification_value",
            num_classes=3,
            optimizer=OptimizerConfig(lr=1e-4, scheduler_type="None"),
            loss_weight=1.0,
        ),
        RegressionTaskConfig(
            name="regr_task_2",
            type=TaskType.REGRESSION,
            dims=[latent_dim, 32, 2],
            data_column="regr_task_2",
            optimizer=OptimizerConfig(lr=1e-4, scheduler_type="None"),
            loss_weight=0.5,
        ),
    ]

    config_dict = {
        "shared_block_dims": shared_dims,
        "task_configs": task_configs_list,
        "encoder_config": MLPEncoderConfig(hidden_dims=shared_dims, norm=True, residual=False),
        "shared_block_optimizer": OptimizerConfig(lr=1e-3, scheduler_type="None"),
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

    assert not model.automatic_optimization, "automatic_optimization should be False for FlexibleMultiTaskModel"


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
    mock_optimizers_list = []
    for _ in range(1 + num_enabled_tasks):
        opt = mocker.MagicMock(spec=torch.optim.Optimizer)
        opt.step = mocker.MagicMock()
        opt.zero_grad = mocker.MagicMock()
        mock_optimizers_list.append(opt)

    # Patch model.optimizers() to return our list of mock optimizers
    # Note: model.optimizers() is a property that calls self.trainer.optimizers,
    # so we need to ensure model.trainer.optimizers returns our mocks.
    # A more direct way if model.optimizers() is called:
    mocker.patch.object(model, "optimizers", return_value=mock_optimizers_list)
    # If training_step directly accesses self.trainer.optimizers:
    # mock_trainer.optimizers = mock_optimizers_list # This might be needed if self.optimizers() is complex

    mock_log_dict = mocker.patch.object(model, "log_dict")
    mock_log = mocker.patch.object(model, "log")

    # sample_batch_mixed_tasks provides (x_formula, y_dict_batch, task_masks_batch, temps_batch)
    # This matches the expected input for training_step
    loss = model.training_step(sample_batch_mixed_tasks, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Loss should be a Tensor"
    # With manual optimization, the returned loss might not directly have requires_grad=True
    # if it's detached or cloned before returning. The important part is that the original
    # computed loss that was passed to manual_backward had requires_grad=True.
    # We'll assert that backward was called on a tensor.
    # assert loss.requires_grad, "Loss should require gradients for backpropagation"
    assert loss.ndim == 0, "Loss should be a scalar"

    # Check that manual_backward was called (via strategy)
    mock_strategy.backward.assert_called_once()
    # Check that backward was called with a tensor that requires grad
    backward_loss_arg = mock_strategy.backward.call_args[0][0]
    assert isinstance(backward_loss_arg, torch.Tensor)
    assert backward_loss_arg.requires_grad, "Loss passed to backward should require gradients"

    # Check that optimizer steps and zero_grads were called
    for opt_mock in mock_optimizers_list:
        opt_mock.step.assert_called_once()
        opt_mock.zero_grad.assert_called_once()

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

    # validation_step now returns None and logs metrics via self.log_dict
    result = model.validation_step(sample_batch_mixed_tasks, batch_idx=0)
    assert result is None, "validation_step should return None"

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
            pred_value = output[expected_key_value]
            if not isinstance(pred_value, torch.Tensor):
                pred_value = torch.as_tensor(pred_value)
            expected_shape = (x_formula.shape[0], task_cfg.dims[-1])
            assert pred_value.shape == expected_shape, (
                f"Shape mismatch for {expected_key_value}. Expected {expected_shape}, got {pred_value.shape}"
            )

        elif task_cfg.type == TaskType.CLASSIFICATION:
            assert isinstance(task_cfg, ClassificationTaskConfig)
            expected_key_proba = f"{task_name_snake}_proba"
            expected_key_label = f"{task_name_snake}_label"

            assert expected_key_proba in output, f"Predict output should contain key '{expected_key_proba}'"
            proba_value = output[expected_key_proba]
            if not isinstance(proba_value, torch.Tensor):
                proba_value = torch.as_tensor(proba_value)
            expected_proba_shape = (x_formula.shape[0], task_cfg.num_classes)
            assert proba_value.shape == expected_proba_shape, (
                f"Shape mismatch for {expected_key_proba}. Expected {expected_proba_shape}, got {proba_value.shape}"
            )

            assert expected_key_label in output, f"Predict output should contain key '{expected_key_label}'"
            label_value = output[expected_key_label]
            if not isinstance(label_value, torch.Tensor):
                label_value = torch.as_tensor(label_value)
            expected_label_shape = (x_formula.shape[0],)
            assert label_value.shape == expected_label_shape, (
                f"Shape mismatch for {expected_key_label}. Expected {expected_label_shape}, got {label_value.shape}"
            )
            assert label_value.dtype == torch.long, f"{expected_key_label} should be of type torch.long"


def test_model_configure_optimizers(model_config_mixed_tasks):
    """Test configure_optimizers for mixed regression and classification tasks."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
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
        # model.encoder contains all encodable parts (shared, struct_enc, fusion)
        encoder_params_ids = {id(p) for p in model.encoder.parameters() if p.requires_grad}
        log_sigma_param_ids = {id(p) for p in model.task_log_sigmas.parameters() if p.requires_grad}
        encoder_related_ids = encoder_params_ids.union(log_sigma_param_ids)
        if not encoder_related_ids.isdisjoint(current_opt_param_ids):
            assert current_opt_param_ids.issubset(encoder_related_ids)
            encoder_opt_found = True
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


def test_stage_index_tracker_masks_duplicates(model_config_mixed_tasks):
    """Ensure distributed index tracker flags duplicates introduced by sampler padding."""
    config = model_config_mixed_tasks
    model = FlexibleMultiTaskModel(
        task_configs=config.task_configs,
        encoder_config=config.encoder_config,
        shared_block_optimizer=config.shared_block_optimizer,
    )
    tracker = {"indices": [0, 1, 0, 2], "cursor": 0, "seen": set()}
    model._stage_index_trackers["val"] = tracker

    mask_info = model._get_batch_valid_mask(stage="val", batch_size=2, device=torch.device("cpu"))
    assert mask_info is not None
    mask_tensor, mask_flags = mask_info
    assert mask_tensor.tolist() == [True, True]
    assert mask_flags == [True, True]

    mask_info = model._get_batch_valid_mask(stage="val", batch_size=2, device=torch.device("cpu"))
    assert mask_info is not None
    mask_tensor, mask_flags = mask_info
    assert mask_tensor.tolist() == [False, True]
    assert mask_flags == [False, True]


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
    computed = model.val_r2_metrics["regr_task_1"].compute()
    assert torch.isclose(computed, torch.tensor(1.0))


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
    tasks = [
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
    assert res.optimized_target.shape == (5, 1, 1)  # one regression task tracked


def test_optimize_latent_class_targets_rejects_regression_task():
    model = _make_reg_clf_model()
    with pytest.raises(ValueError, match="must be a classification task"):
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
    assert res.optimized_target.shape == (4, 1, 0)  # no regression tasks tracked


def test_optimize_latent_ae_cycle_rejects_negative():
    model = _make_reg_clf_model()
    with pytest.raises(ValueError, match="ae_cycle_weight must be >= 0"):
        model.optimize_latent(
            initial_input=torch.randn(2, INPUT_DIM),
            task_targets={"prop": 1.0},
            optimize_space="latent",
            ae_cycle_weight=-0.1,
        )


def test_optimize_latent_ae_cycle_runs_in_latent_space():
    torch.manual_seed(0)
    model = _make_reg_clf_model()  # enable_autoencoder=True, so AE head is available
    x = torch.randn(4, INPUT_DIM)
    res = model.optimize_latent(
        initial_input=x,
        task_targets={"prop": 1.0},
        class_targets={"cls": [1]},
        class_target_weight=3.0,
        ae_cycle_weight=0.5,  # pull latent toward AE-reconstructible fixed set
        optimize_space="latent",
        steps=10,
    )
    assert res.optimized_input.shape == (4, 1, INPUT_DIM)
    assert res.optimized_target.shape == (4, 1, 1)


def test_optimize_latent_class_target_weight_rejects_nonpositive():
    model = _make_reg_clf_model()
    with pytest.raises(ValueError, match="class_target_weight must be > 0"):
        model.optimize_latent(
            initial_input=torch.randn(2, INPUT_DIM),
            class_targets={"cls": [1]},
            class_target_weight=0.0,
            optimize_space="input",
        )


def test_optimize_latent_class_target_weight_runs_with_combined_objectives():
    torch.manual_seed(0)
    model = _make_reg_clf_model()
    x = torch.randn(4, INPUT_DIM)
    res = model.optimize_latent(
        initial_input=x,
        task_targets={"prop": 1.0},
        class_targets={"cls": [1]},
        class_target_weight=5.0,  # class probability is the primary objective
        optimize_space="input",
        steps=10,
    )
    assert res.optimized_input.shape == (4, 1, INPUT_DIM)
    assert res.optimized_target.shape == (4, 1, 1)  # one regression task tracked


# --- optimize_composition (differentiable KMD) --------------------------------


def test_optimize_composition_runs_and_returns_simplex_weights():
    torch.manual_seed(0)
    model = _make_reg_clf_model()  # INPUT_DIM=20
    n_components = 6
    kmd_kernel = torch.randn(n_components, INPUT_DIM)
    res = model.optimize_composition(
        kmd_kernel,
        task_targets={"prop": 1.0},
        class_targets={"cls": [1]},
        class_target_weight=3.0,
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
    assert res.optimized_target.shape == (4, 1)
    assert res.trajectory.shape == (10, 4, 1)


def test_optimize_composition_validates_kernel_and_objectives():
    model = _make_reg_clf_model()
    # kernel must be 2D
    with pytest.raises(ValueError, match="2D torch.Tensor"):
        model.optimize_composition(torch.randn(6), task_targets={"prop": 1.0}, n_starts=2, steps=2)
    # kernel's x_dim must match encoder.input_dim
    with pytest.raises(ValueError, match="encoder.input_dim"):
        model.optimize_composition(torch.randn(6, INPUT_DIM + 1), task_targets={"prop": 1.0}, n_starts=2, steps=2)
    # at least one objective required
    with pytest.raises(ValueError, match="at least one of task_targets"):
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
        class_targets={"cls": target},
        class_target_weight=5.0,
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
    """Empty trajectory still carries the regression-task width T (not 0)."""
    model = _make_reg_clf_model()
    kernel = torch.randn(6, INPUT_DIM)
    res = model.optimize_composition(
        kernel, task_targets={"prop": 0.0}, class_targets={"cls": [1]}, n_starts=3, steps=0
    )
    # tasks_for_optimization = ["prop"] → T == 1
    assert res.trajectory.shape == (0, 3, 1)


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
    tasks = [
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
        task_targets={"prop": 1.0},
        class_targets={"cls": [1]},
        class_target_weight=3.0,
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
    """A symbol→0.0 mapping freezes those elements' weights at their seed values."""
    torch.manual_seed(0)
    model, kernel, elements = _build_aligned_model_and_kernel()

    # Seed: equal mass on 4 specific symbols, zero on the rest.
    locked_syms = ["Mg", "Al"]
    free_syms = ["Cu", "Ni"]
    seed_syms = locked_syms + free_syms
    init_w = torch.zeros(1, len(elements))
    for s in seed_syms:
        init_w[0, elements.index(s)] = 0.25

    res = model.optimize_composition(
        kernel,
        task_targets={"prop": 5.0},
        initial_weights=init_w,
        element_step_scale={s: 0.0 for s in locked_syms},
        steps=50,
        lr=0.3,
    )
    w = res.optimized_weights
    # The two locked symbols had equal seed weight; their logits don't move, so the softmax ratio
    # stays at 1.0 regardless of how other elements grow.
    mg, al = elements.index("Mg"), elements.index("Al")
    assert torch.isclose(w[0, mg] / w[0, al], torch.tensor(1.0), atol=1e-4)


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


def test_optimize_composition_entropy_weight_rejects_negative():
    model, kernel, _ = _build_aligned_model_and_kernel()
    with pytest.raises(ValueError, match="entropy_weight must be >= 0"):
        model.optimize_composition(kernel, task_targets={"prop": 0.0}, entropy_weight=-0.1, n_starts=2, steps=2)


def test_optimize_composition_entropy_weight_runs():
    """entropy_weight>0 just needs to run cleanly and still produce simplex rows."""
    torch.manual_seed(0)
    model, kernel, _ = _build_aligned_model_and_kernel()
    res = model.optimize_composition(kernel, task_targets={"prop": 1.0}, n_starts=3, entropy_weight=0.5, steps=5)
    assert res.optimized_weights.shape[0] == 3
    assert torch.allclose(res.optimized_weights.sum(dim=-1), torch.ones(3), atol=1e-5)


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
