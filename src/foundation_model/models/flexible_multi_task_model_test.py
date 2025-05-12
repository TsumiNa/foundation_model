from types import SimpleNamespace

import pytest
import torch

from .flexible_multi_task_model import FlexibleMultiTaskModel

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
    config = {
        "shared_block_dims": [64, 128, 256],  # Input dim (e.g., from formula_desc) -> hidden -> latent
        "task_block_dims": [256, 128, 1],  # Latent dim from deposit -> hidden -> output (1 for regression)
        "n_attr_tasks": 3,
        "norm_shared": True,
        "residual_shared": False,
        "norm_tasks": True,
        "residual_tasks": False,
        "shared_block_lr": 1e-3,
        "task_block_lr": 1e-3,
        "seq_head_lr": 1e-3,  # Will be ignored as seq_head is None
        "sequence_mode": "none",
        "seq_len": None,  # Not used
        "with_structure": False,
        "struct_block_dims": None,  # Not used
        "modality_dropout_p": 0.0,  # Not used
        "pretrain": False,
        "loss_weights": None,
        "mask_ratio": 0.0,  # Not used
        "temperature": 0.07,  # Not used
        "freeze_encoder": False,
        "lora_rank": 0,  # LoRA off
        "lora_alpha": 1.0,
    }
    return SimpleNamespace(**config)


@pytest.fixture
def sample_batch_attr_only(model_config_simple_attr_only):
    """
    Generates a sample batch for attribute-only tasks.
    x_formula: (B, D_formula_in)
    y_attr: (B, n_attr_tasks)
    mask_attr: (B, n_attr_tasks)
    temps, y_seq, mask_seq are None.
    """
    batch_size = 4
    formula_input_dim = model_config_simple_attr_only.shared_block_dims[0]
    n_attr_tasks = model_config_simple_attr_only.n_attr_tasks

    x_formula = torch.randn(batch_size, formula_input_dim)
    y_attr = torch.randn(batch_size, n_attr_tasks)
    # Mask some attributes (e.g., last task for first sample, first task for second sample)
    mask_attr = torch.ones_like(y_attr)
    if batch_size > 1 and n_attr_tasks > 0:
        mask_attr[0, -1] = 0
        if batch_size > 1 and n_attr_tasks > 0:  # ensure n_attr_tasks > 0
            mask_attr[1, 0] = 0

    # For attribute-only, x_struct, temps, y_seq, mask_seq are effectively None or not used
    # The model's training_step expects a tuple of 6 items.
    return (x_formula, y_attr, mask_attr, None, None, None)


# --- Test Cases ---


def test_model_initialization_attr_only(model_config_simple_attr_only):
    """Test model initialization for attribute-only tasks."""
    config = model_config_simple_attr_only
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_block_dims=config.task_block_dims,
        n_attr_tasks=config.n_attr_tasks,
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        norm_tasks=config.norm_tasks,
        residual_tasks=config.residual_tasks,
        shared_block_lr=config.shared_block_lr,
        task_block_lr=config.task_block_lr,
        seq_head_lr=config.seq_head_lr,
        sequence_mode=config.sequence_mode,
        with_structure=config.with_structure,
        pretrain=config.pretrain,
        freeze_encoder=config.freeze_encoder,
        lora_rank=config.lora_rank,
    )

    assert model.shared is not None, "Shared block should be initialized"
    assert model.deposit is not None, "Deposit layer should be initialized"
    assert model.attr_heads is not None and len(model.attr_heads) == config.n_attr_tasks, (
        f"Expected {config.n_attr_tasks} attribute heads"
    )

    assert model.seq_head is None, "Sequence head should be None for sequence_mode='none'"
    assert not hasattr(model, "struct_enc") or model.struct_enc is None, (
        "Structure encoder should not be initialized if with_structure=False"
    )
    assert not hasattr(model, "fusion") or model.fusion is None, (
        "Fusion layer should not be initialized if with_structure=False"
    )
    assert not model.pretrain, "Pretrain flag should be False"
    assert model.lora_rank == 0, "LoRA rank should be 0 (off)"

    # Check automatic_optimization is False as per model's __init__
    assert not model.automatic_optimization, "automatic_optimization should be False"


def test_model_forward_pass_attr_only(model_config_simple_attr_only, sample_batch_attr_only):
    """Test the forward pass for attribute-only predictions."""
    config = model_config_simple_attr_only
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_block_dims=config.task_block_dims,
        n_attr_tasks=config.n_attr_tasks,
        # Pass other necessary params from config, simplified for brevity if not directly affecting forward structure
        sequence_mode=config.sequence_mode,
        with_structure=config.with_structure,
    )
    model.eval()  # Set to evaluation mode

    x_formula, _, _, _, _, _ = sample_batch_attr_only  # We only need x_formula for this forward pass type

    # When with_structure is False, model expects x_formula directly
    output = model(x_formula)

    assert isinstance(output, dict), "Output should be a dictionary"
    assert "attr" in output, "Output dictionary should contain 'attr' key"

    attr_preds = output["attr"]
    assert isinstance(attr_preds, torch.Tensor), "Attribute predictions should be a Tensor"

    expected_shape = (x_formula.shape[0], config.n_attr_tasks)  # (batch_size, n_attr_tasks)
    assert attr_preds.shape == expected_shape, (
        f"Attribute predictions shape mismatch. Expected {expected_shape}, got {attr_preds.shape}"
    )

    # Since sequence_mode is 'none', 'seq' key should not be in output
    assert "seq" not in output, "Output should not contain 'seq' key when sequence_mode is 'none'"


def test_model_training_step_attr_only(model_config_simple_attr_only, sample_batch_attr_only, mocker):
    """Test the training_step for attribute-only tasks."""
    config = model_config_simple_attr_only
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_block_dims=config.task_block_dims,
        n_attr_tasks=config.n_attr_tasks,
        sequence_mode=config.sequence_mode,
        with_structure=config.with_structure,
        pretrain=config.pretrain,  # Ensure pretrain is False for this test
        # Other necessary params
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        norm_tasks=config.norm_tasks,
        residual_tasks=config.residual_tasks,
        shared_block_lr=config.shared_block_lr,
        task_block_lr=config.task_block_lr,
        seq_head_lr=config.seq_head_lr,
        freeze_encoder=config.freeze_encoder,
        lora_rank=config.lora_rank,
    )
    model.train()  # Set to training mode

    # Mock log_dict to check logged values
    mock_log_dict = mocker.patch.object(model, "log_dict")

    # Mock optimizers and their methods (step, zero_grad) as manual_optimization is True
    # The model's training_step calls zero_grad, manual_backward, and step on optimizers
    mock_opt_shared = mockerMock = mocker.MagicMock()
    mock_opt_attr = mocker.MagicMock()

    # The model's configure_optimizers returns a list.
    # For attr-only and no structure, it's [opt_shared, opt_attr]
    mocker.patch.object(model, "optimizers", return_value=[mock_opt_shared, mock_opt_attr])
    mocker.patch.object(model, "manual_backward")

    loss = model.training_step(sample_batch_attr_only, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Loss should be a Tensor"
    assert loss.requires_grad, "Loss should require gradients for backpropagation"
    assert loss.ndim == 0, "Loss should be a scalar"

    # Check that log_dict was called
    mock_log_dict.assert_called()

    # Get the arguments log_dict was called with
    # log_dict(logs, prog_bar=True, on_step=True, on_epoch=False)
    logged_metrics = mock_log_dict.call_args[0][0]

    assert "train_attr_loss" in logged_metrics, "train_attr_loss should be logged"
    assert isinstance(logged_metrics["train_attr_loss"], torch.Tensor), "train_attr_loss should be a Tensor"

    # Ensure pretrain-specific losses are not logged as pretrain=False
    assert "train_con" not in logged_metrics, "Contrastive loss should not be logged"
    assert "train_cross" not in logged_metrics, "Cross-reconstruction loss should not be logged"
    assert "train_mask" not in logged_metrics, "Masked-feature loss should not be logged"

    # Ensure sequence loss is not logged
    assert "train_seq_loss" not in logged_metrics, "Sequence loss should not be logged"

    # Check optimizer steps
    model.manual_backward.assert_called_once_with(loss)
    mock_opt_shared.step.assert_called_once()
    mock_opt_attr.step.assert_called_once()
    mock_opt_shared.zero_grad.assert_called_once()
    mock_opt_attr.zero_grad.assert_called_once()


def test_model_validation_step_attr_only(model_config_simple_attr_only, sample_batch_attr_only, mocker):
    """Test the validation_step for attribute-only tasks."""
    config = model_config_simple_attr_only
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_block_dims=config.task_block_dims,
        n_attr_tasks=config.n_attr_tasks,
        sequence_mode=config.sequence_mode,
        with_structure=config.with_structure,
        pretrain=config.pretrain,  # Ensure pretrain is False
        # Other necessary params
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        norm_tasks=config.norm_tasks,
        residual_tasks=config.residual_tasks,
        shared_block_lr=config.shared_block_lr,
        task_block_lr=config.task_block_lr,
        seq_head_lr=config.seq_head_lr,
        freeze_encoder=config.freeze_encoder,
        lora_rank=config.lora_rank,
    )
    model.eval()  # Set to evaluation mode for validation_step

    # Mock log_dict to check logged values
    mock_log_dict = mocker.patch.object(model, "log_dict")

    loss = model.validation_step(sample_batch_attr_only, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Validation loss should be a Tensor"
    assert loss.ndim == 0, "Validation loss should be a scalar"

    # Check that log_dict was called
    mock_log_dict.assert_called()
    logged_metrics = mock_log_dict.call_args[0][0]

    assert "val_attr_loss" in logged_metrics, "val_attr_loss should be logged"
    assert isinstance(logged_metrics["val_attr_loss"], torch.Tensor), "val_attr_loss should be a Tensor"

    # Ensure pretrain-specific and sequence losses are not logged
    assert "val_con" not in logged_metrics
    assert "val_cross" not in logged_metrics
    assert "val_mask" not in logged_metrics
    assert "val_seq_loss" not in logged_metrics


def test_model_predict_step_attr_only(model_config_simple_attr_only, sample_batch_attr_only):
    """Test the predict_step for attribute-only tasks."""
    config = model_config_simple_attr_only
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_block_dims=config.task_block_dims,
        n_attr_tasks=config.n_attr_tasks,
        sequence_mode=config.sequence_mode,
        with_structure=config.with_structure,
        # Other necessary params
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        norm_tasks=config.norm_tasks,
        residual_tasks=config.residual_tasks,
        shared_block_lr=config.shared_block_lr,
        task_block_lr=config.task_block_lr,
        seq_head_lr=config.seq_head_lr,
        pretrain=config.pretrain,
        freeze_encoder=config.freeze_encoder,
        lora_rank=config.lora_rank,
    )
    model.eval()  # Set to evaluation mode

    # predict_step takes (self, batch, batch_idx, dataloader_idx=0)
    # For attr-only, batch is (x_formula, y_attr, mask_attr, None, None, None)
    # We only need x_formula from the batch for prediction in this mode.
    x_formula, _, _, _, _, _ = sample_batch_attr_only

    # The model's predict_step internally calls self.forward((x_formula, x_struct)...)
    # or self.forward(x_formula...)
    # In attr-only mode (with_structure=False), it expects just x_formula in the batch for forward.
    # The batch for predict_step is the full 6-tuple.

    # Create a batch suitable for predict_step: (x, y_attr, mask_attr, temps, y_seq, mask_seq)
    # where x is x_formula for attr-only mode.
    predict_batch = (x_formula, None, None, None, None, None)  # y_attr etc. are not used by predict_step

    output = model.predict_step(predict_batch, batch_idx=0)

    assert isinstance(output, dict), "Predict output should be a dictionary"
    assert "attr" in output, "Predict output dictionary should contain 'attr' key"

    attr_preds = output["attr"]
    assert isinstance(attr_preds, torch.Tensor), "Attribute predictions should be a Tensor"

    expected_shape = (x_formula.shape[0], config.n_attr_tasks)  # (batch_size, n_attr_tasks)
    assert attr_preds.shape == expected_shape, (
        f"Attribute predictions shape mismatch. Expected {expected_shape}, got {attr_preds.shape}"
    )

    assert "seq" not in output, "Predict output should not contain 'seq' key when sequence_mode is 'none'"


def test_model_configure_optimizers_attr_only(model_config_simple_attr_only):
    """Test configure_optimizers for attribute-only tasks."""
    config = model_config_simple_attr_only
    model = FlexibleMultiTaskModel(
        shared_block_dims=config.shared_block_dims,
        task_block_dims=config.task_block_dims,
        n_attr_tasks=config.n_attr_tasks,
        sequence_mode=config.sequence_mode,  # "none"
        with_structure=config.with_structure,  # False
        # Other necessary params
        norm_shared=config.norm_shared,
        residual_shared=config.residual_shared,
        norm_tasks=config.norm_tasks,
        residual_tasks=config.residual_tasks,
        shared_block_lr=config.shared_block_lr,
        task_block_lr=config.task_block_lr,
        seq_head_lr=config.seq_head_lr,
        pretrain=config.pretrain,
        freeze_encoder=config.freeze_encoder,
        lora_rank=config.lora_rank,
    )

    optimizers = model.configure_optimizers()

    # For attr-only, no structure, no seq_head: expects 2 optimizers
    # 1 for shared block, 1 for attribute heads
    assert isinstance(optimizers, list), "configure_optimizers should return a list"
    assert len(optimizers) == 2, f"Expected 2 optimizers, got {len(optimizers)}"

    # Check types (optional, but good for sanity)
    assert isinstance(optimizers[0], torch.optim.Adam), "First optimizer should be Adam for shared block"
    assert isinstance(optimizers[1], torch.optim.Adam), "Second optimizer should be Adam for attribute heads"

    # Check parameters managed by each optimizer
    # Optimizer 0: shared parameters
    shared_params_ids = {id(p) for p in model.shared.parameters() if p.requires_grad}
    opt0_params_ids = {id(p) for group in optimizers[0].param_groups for p in group["params"]}
    assert shared_params_ids == opt0_params_ids, "Optimizer 0 does not manage all shared parameters"

    # Optimizer 1: attribute heads parameters
    attr_heads_params_ids = {id(p) for p in model.attr_heads.parameters() if p.requires_grad}
    opt1_params_ids = {id(p) for group in optimizers[1].param_groups for p in group["params"]}
    assert attr_heads_params_ids == opt1_params_ids, "Optimizer 1 does not manage all attribute head parameters"

    # Ensure no overlap between optimizer parameters
    assert opt0_params_ids.isdisjoint(opt1_params_ids), "Optimizers should manage disjoint sets of parameters"
