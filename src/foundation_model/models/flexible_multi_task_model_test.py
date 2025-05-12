from types import SimpleNamespace

import pytest
import torch

from .flexible_multi_task_model import FlexibleMultiTaskModel
from .model_config import OptimizerConfig, RegressionTaskConfig, TaskType  # Added imports

# --- Fixtures ---


@pytest.fixture
def model_config_simple_attr_only():
    """
    Provides a basic configuration for FlexibleMultiTaskModel focusing on attribute tasks.
    - No structure fusion (with_structure=False)
    - No sequence head (sequence_mode="none")
    - No pretraining options enabled
    - No LoRA
    """
    shared_dims = [64, 128, 256]  # Input -> hidden -> latent
    deposit_dim = shared_dims[-1]  # Deposit layer input is latent dim

    # Define task configurations for 3 attribute regression tasks
    task_configs_list = []
    n_attr_tasks = 3  # Define n_attr_tasks here for clarity
    for i in range(n_attr_tasks):
        task_configs_list.append(
            RegressionTaskConfig(
                name=f"attr_task_{i}",
                type=TaskType.REGRESSION,
                dims=[deposit_dim, 128, 1],  # deposit_dim -> hidden -> output (1 for regression)
                optimizer=OptimizerConfig(lr=1e-4, optimizer_type="Adam", scheduler_type="None"),  # Basic optimizer
            )
        )

    config = {
        "shared_block_dims": shared_dims,
        "task_configs": task_configs_list,
        "norm_shared": True,
        "residual_shared": False,
        # norm_tasks and residual_tasks are part of individual TaskConfigs if needed.
        # For LinearBlock used in RegressionTaskConfig, these are defaulted.
        "shared_block_optimizer": OptimizerConfig(lr=1e-3, optimizer_type="Adam", scheduler_type="None"),
        # sequence_mode is not a direct param for __init__ if no SequenceTaskConfig
        "with_structure": False,
        "struct_block_dims": None,
        "modality_dropout_p": 0.0,
        "enable_self_supervised_training": False,
        "loss_weights": None,
        "mask_ratio": 0.0,
        "temperature": 0.07,
    }
    return SimpleNamespace(**config)  # Convert dict to SimpleNamespace


@pytest.fixture
def sample_batch_attr_only(model_config_simple_attr_only):
    """
    Generates a sample batch for attribute-only tasks.
    x_formula: (B, D_formula_in)
    y_dict_batch: dict of {task_name: (B, task_output_dim)}
    task_masks_batch: dict of {task_name: (B, 1)}
    temps_batch is an empty dict.
    """
    batch_size = 4
    config = model_config_simple_attr_only
    formula_input_dim = config.shared_block_dims[0]

    x_formula = torch.randn(batch_size, formula_input_dim)

    y_dict_batch = {}
    task_masks_batch = {}

    for i, task_cfg in enumerate(config.task_configs):
        # Assuming all are regression tasks with output_dim 1 as per fixture
        task_output_dim = task_cfg.dims[-1]
        y_task = torch.randn(batch_size, task_output_dim)
        # Create a [B, 1] boolean mask for each task
        mask_task = torch.ones(batch_size, 1, dtype=torch.bool)
        if batch_size > 0:  # Mask one sample for each task for variability
            mask_idx = i % batch_size  # Cycle through samples to mask for different tasks
            mask_task[mask_idx, 0] = False

        y_dict_batch[task_cfg.name] = y_task
        task_masks_batch[task_cfg.name] = mask_task

    # For attribute-only, x_struct is None. temps_batch is an empty dict.
    # The model's training_step expects: (x, y_dict_batch, task_masks_batch, temps_batch)
    # x can be x_formula or (x_formula, x_struct)
    return (x_formula, y_dict_batch, task_masks_batch, {})  # Empty dict for temps_batch


# --- Test Cases ---


def test_model_initialization_attr_only(model_config_simple_attr_only):
    """Test model initialization for attribute-only tasks."""
    config = model_config_simple_attr_only
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_configs=config.task_configs,  # Use task_configs
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

    assert model.shared is not None, "Shared block (via encoder.shared) should be initialized"
    assert model.deposit is not None, "Deposit layer (via encoder.deposit) should be initialized"

    # Task heads are now in model.task_heads (a ModuleDict)
    assert model.task_heads is not None
    assert len(model.task_heads) == len(config.task_configs), f"Expected {len(config.task_configs)} task heads"

    # Check if specific task heads from config are present
    for task_cfg in config.task_configs:
        assert task_cfg.name in model.task_heads, f"Task head {task_cfg.name} not found in model.task_heads"

    # No direct seq_head attribute anymore if not created by create_task_heads
    # We check by ensuring no SequenceTaskConfig was passed and thus no sequence head in model.task_heads
    assert not any(
        isinstance(head, torch.nn.Module) and "sequence" in head.__class__.__name__.lower()
        for head in model.task_heads.values()
    ), "No sequence head should be present in task_heads for this config"

    assert not model.with_structure, "with_structure should be False"
    if hasattr(model, "struct_enc"):  # struct_enc might not exist if with_structure is False from the start
        assert model.struct_enc is None, "Structure encoder should be None if with_structure=False"
    if hasattr(model, "fusion"):
        assert model.fusion is None, "Fusion layer should be None if with_structure=False"

    assert not model.enable_self_supervised_training, "enable_self_supervised_training should be False"

    # LoRA is not part of __init__, it's applied if lora_rank > 0 in the old model.
    # The new model doesn't seem to have direct LoRA params in __init__.
    # We'll assume no LoRA for this simple test based on config.

    # automatic_optimization is not explicitly set in the new model's __init__
    # It defaults to True in L.LightningModule. If it's meant to be False, it needs to be set.
    # For now, let's remove this check or adapt if the model explicitly sets it.
    # assert not model.automatic_optimization, "automatic_optimization should be False"
    # Based on the provided model code, automatic_optimization is not set, so it defaults to True.
    # If the old model had it False, this is a change. For now, we test the current state.
    assert model.automatic_optimization, "automatic_optimization should be True by default in L.LightningModule"


def test_model_forward_pass_attr_only(model_config_simple_attr_only, sample_batch_attr_only):
    """Test the forward pass for attribute-only predictions."""
    config = model_config_simple_attr_only
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
    model.eval()

    x_formula, _, _, temps_batch = sample_batch_attr_only  # Unpack, temps_batch is {}

    # When with_structure is False, model's forward expects x_formula and temps_batch
    output = model(x_formula, temps_batch=temps_batch)

    assert isinstance(output, dict), "Output should be a dictionary"

    # Check outputs for each configured task
    assert len(output.keys()) == len(config.task_configs), (
        f"Expected {len(config.task_configs)} keys in output, got {len(output.keys())}"
    )

    for task_cfg in config.task_configs:
        assert task_cfg.name in output, f"Output dictionary should contain '{task_cfg.name}' key"
        task_pred = output[task_cfg.name]
        assert isinstance(task_pred, torch.Tensor), f"{task_cfg.name} predictions should be a Tensor"

        # Assuming regression tasks with output dim 1 as per fixture
        # task_cfg.dims is like [deposit_dim, hidden_dim, output_dim_for_task_head]
        expected_task_output_dim = task_cfg.dims[-1]
        expected_task_shape = (x_formula.shape[0], expected_task_output_dim)
        assert task_pred.shape == expected_task_shape, (
            f"{task_cfg.name} predictions shape mismatch. Expected {expected_task_shape}, got {task_pred.shape}"
        )


def test_model_training_step_attr_only(model_config_simple_attr_only, sample_batch_attr_only, mocker):
    """Test the training_step for attribute-only tasks."""
    config = model_config_simple_attr_only
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
    model.train()

    mock_log_dict = mocker.patch.object(model, "log_dict")

    # The model uses automatic_optimization=True by default.
    # training_step itself doesn't call optimizer steps or manual_backward.
    # PyTorch Lightning handles that.

    loss = model.training_step(sample_batch_attr_only, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Loss should be a Tensor"
    assert loss.requires_grad, "Loss should require gradients for backpropagation"
    assert loss.ndim == 0, "Loss should be a scalar"

    mock_log_dict.assert_called()
    logged_metrics = mock_log_dict.call_args[0][0]

    # Check that losses for each attribute task are logged
    for task_cfg in config.task_configs:
        assert f"train_{task_cfg.name}_loss" in logged_metrics, f"train_{task_cfg.name}_loss should be logged"
        assert isinstance(logged_metrics[f"train_{task_cfg.name}_loss"], torch.Tensor), (
            f"train_{task_cfg.name}_loss should be a Tensor"
        )
        assert f"train_{task_cfg.name}_loss_weighted" in logged_metrics, (
            f"train_{task_cfg.name}_loss_weighted should be logged"
        )

    assert "train_total_loss" in logged_metrics, "train_total_loss should be logged"

    # Ensure SSL losses are not logged as enable_self_supervised_training=False
    assert "train_mfm_loss" not in logged_metrics
    assert "train_contrastive_loss" not in logged_metrics
    assert "train_cross_recon_loss" not in logged_metrics
    assert "train_modality_dropout_applied" not in logged_metrics  # Only logged if SSL and with_structure


def test_model_validation_step_attr_only(model_config_simple_attr_only, sample_batch_attr_only, mocker):
    """Test the validation_step for attribute-only tasks."""
    config = model_config_simple_attr_only
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_configs=config.task_configs,
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        shared_block_optimizer=config.shared_block_optimizer,
        with_structure=config.with_structure,
        struct_block_dims=config.struct_block_dims,
        modality_dropout_p=config.modality_dropout_p,
        enable_self_supervised_training=config.enable_self_supervised_training,  # False
        loss_weights=config.loss_weights,
        mask_ratio=config.mask_ratio,
        temperature=config.temperature,
    )
    model.eval()

    mock_log_dict = mocker.patch.object(model, "log_dict")

    # validation_step now returns None and logs metrics
    result = model.validation_step(sample_batch_attr_only, batch_idx=0)
    assert result is None, "validation_step should return None"

    mock_log_dict.assert_called()
    logged_metrics = mock_log_dict.call_args[0][0]

    # Check that losses for each attribute task are logged
    for task_cfg in config.task_configs:
        assert f"val_{task_cfg.name}_loss" in logged_metrics, f"val_{task_cfg.name}_loss should be logged"
        assert isinstance(logged_metrics[f"val_{task_cfg.name}_loss"], torch.Tensor), (
            f"val_{task_cfg.name}_loss should be a Tensor"
        )

    assert "val_total_loss" in logged_metrics, "val_total_loss should be logged"

    # Ensure SSL losses are not logged
    assert "val_mfm_loss" not in logged_metrics
    assert "val_contrastive_loss" not in logged_metrics
    assert "val_cross_recon_loss" not in logged_metrics


def test_model_predict_step_attr_only(model_config_simple_attr_only, sample_batch_attr_only):
    """Test the predict_step for attribute-only tasks."""
    config = model_config_simple_attr_only
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
    model.eval()

    # sample_batch_attr_only returns (x_formula, y_dict_batch, task_masks_batch, temps_batch)
    # predict_step expects batch[0] to be x_formula and batch[3] to be temps_batch
    x_formula, _, _, temps_batch = sample_batch_attr_only

    # Construct the batch as expected by predict_step: (x_input, y_dict, masks_dict, temps_dict)
    # For predict_step, y_dict and masks_dict can be None. x_input is x_formula.
    predict_batch_tuple = (x_formula, None, None, temps_batch)

    output = model.predict_step(predict_batch_tuple, batch_idx=0)

    assert isinstance(output, dict), "Predict output should be a dictionary"

    # The output keys are now like "attr_task_0_value" (from head.predict())
    # instead of just "attr_task_0".
    # We need to check for the presence of keys corresponding to each task.
    # Each regression task (as per fixture) should produce one output key (e.g., task_name_value).

    num_expected_output_keys = 0
    for task_cfg in config.task_configs:
        # RegressionTaskHead.predict by default returns {f"{self.snake_case_name}_value": y_pred_processed}
        # So we expect one key per task.
        num_expected_output_keys += 1
        # More specific check:
        # from foundation_model.models.task_head.base import _to_snake_case
        # expected_key = f"{_to_snake_case(task_cfg.name)}_value"
        # assert expected_key in output, f"Predict output should contain key '{expected_key}'"
        # For simplicity, just check one of them if names are "attr_task_0", "attr_task_1", etc.
        if task_cfg.name == "attr_task_0":  # Check for one specific task's output format
            assert "attr_task_0_value" in output, "Predict output should contain 'attr_task_0_value'"

    assert len(output.keys()) == num_expected_output_keys, (
        f"Expected {num_expected_output_keys} keys in predict output, got {len(output.keys())}"
    )

    # Example check for one task's output tensor properties
    if "attr_task_0_value" in output:
        pred_tensor = output["attr_task_0_value"]
        assert isinstance(pred_tensor, torch.Tensor)
        # Assuming output dim is 1 for regression tasks in fixture
        task_0_config = next(tc for tc in config.task_configs if tc.name == "attr_task_0")
        expected_shape = (x_formula.shape[0], task_0_config.dims[-1])
        assert pred_tensor.shape == expected_shape, (
            f"Shape mismatch for attr_task_0_value. Expected {expected_shape}, got {pred_tensor.shape}"
        )


def test_model_configure_optimizers_attr_only(model_config_simple_attr_only):
    """Test configure_optimizers for attribute-only tasks."""
    config = model_config_simple_attr_only
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

    optimizers_and_schedulers = model.configure_optimizers()

    # Expected: 1 optimizer for the encoder, and 1 for each task head (3 tasks in fixture)
    # Total = 1 (encoder) + 3 (task heads) = 4 optimizers
    # Each can optionally have a scheduler, so items in the list can be an optimizer or a dict.

    assert isinstance(optimizers_and_schedulers, list), "configure_optimizers should return a list"

    num_optimizers = 0
    encoder_opt_found = False
    task_head_opts_found_count = 0

    # Collect all parameters managed by the optimizers
    all_optimized_param_ids = set()

    for item in optimizers_and_schedulers:
        num_optimizers += 1
        optimizer = item["optimizer"] if isinstance(item, dict) else item
        assert isinstance(optimizer, torch.optim.Optimizer), "Optimizer item is not a torch.optim.Optimizer"

        current_opt_param_ids = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                assert p.requires_grad, "Optimizer should only manage parameters that require gradients"
                current_opt_param_ids.add(id(p))

        # Check for disjointness with already processed parameters
        assert all_optimized_param_ids.isdisjoint(current_opt_param_ids), (
            "Optimizers should manage disjoint sets of parameters"
        )
        all_optimized_param_ids.update(current_opt_param_ids)

        # Check if this optimizer handles encoder parameters
        encoder_params_ids = {id(p) for p in model.encoder.parameters() if p.requires_grad}
        if not encoder_params_ids.isdisjoint(current_opt_param_ids):
            assert current_opt_param_ids == encoder_params_ids, (
                "An optimizer should manage all and only encoder parameters"
            )
            encoder_opt_found = True
            continue  # Move to next optimizer

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
        assert found_task_head_for_this_opt, "Optimizer does not seem to correspond to encoder or any task head"

    expected_num_optimizers = 1 + len(config.task_configs)  # 1 for encoder + N for task heads
    assert num_optimizers == expected_num_optimizers, (
        f"Expected {expected_num_optimizers} optimizers, got {num_optimizers}"
    )
    assert encoder_opt_found, "No optimizer found for the encoder"
    assert task_head_opts_found_count == len(config.task_configs), (
        f"Expected {len(config.task_configs)} optimizers for task heads, found {task_head_opts_found_count}"
    )

    # Verify all trainable parameters are covered
    all_model_trainable_params_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if model.enable_self_supervised_training and hasattr(model, "ssl_module"):  # ssl_module might not exist
        # SSL module params are optimized by a separate optimizer in the list
        # This check might need adjustment if ssl_module params are part of encoder_params
        pass  # Covered by the loop checking disjoint sets and counts
    else:
        assert all_optimized_param_ids == all_model_trainable_params_ids, (
            "Not all trainable model parameters are covered by the optimizers"
        )
