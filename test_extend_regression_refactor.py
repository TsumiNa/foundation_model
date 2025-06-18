#!/usr/bin/env python3
"""
Simple test script to validate the ExtendRegressionHead refactoring.
"""

import torch

from src.foundation_model.models.model_config import ExtendRegressionTaskConfig
from src.foundation_model.models.task_head.extend_regression import ExtendRegressionHead


def test_extend_regression_head():
    """Test the refactored ExtendRegressionHead."""
    print("Testing ExtendRegressionHead refactoring...")

    # Test configuration
    config = ExtendRegressionTaskConfig(
        name="test_extend_regression",
        x_dim=[64, 32, 16],
        t_dim=[32, 16, 8],
        interaction_dim=16,
        t_encoding_method="fourier",
        norm=True,
        residual=False,
    )

    # Create model
    model = ExtendRegressionHead(config)
    print(f"✓ Model created with config: {config.name}")
    print(f"  - x_dim: {config.x_dim}")
    print(f"  - t_dim: {config.t_dim}")
    print(f"  - t_encoding_method: {config.t_encoding_method}")
    print(f"  - interaction_dim: {config.interaction_dim}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, config.x_dim[0])  # (4, 64)
    t = torch.randn(batch_size)  # (4,) - scalar per sample

    # Forward pass
    output = model.forward(x, t)
    print("✓ Forward pass successful:")
    print(f"  - Input x shape: {x.shape}")
    print(f"  - Input t shape: {t.shape}")
    print(f"  - Output shape: {output.shape}")

    # Test loss computation
    target = torch.randn(batch_size, 1)
    loss, per_dim_loss = model.compute_loss(output, target)
    print("✓ Loss computation successful:")
    print(f"  - Total loss: {loss.item():.4f}")
    print(f"  - Per-dim loss shape: {per_dim_loss.shape}")

    # Test prediction
    pred_dict = model._predict_impl(output)
    print("✓ Prediction successful:")
    print(f"  - Prediction keys: {list(pred_dict.keys())}")
    print(f"  - Prediction shape: {pred_dict['value'].shape}")

    # Test with FC encoding
    config_fc = ExtendRegressionTaskConfig(
        name="test_extend_regression_fc",
        x_dim=[64, 32, 16],
        t_dim=[32, 16, 8],
        interaction_dim=16,
        t_encoding_method="fc",
        norm=True,
        residual=False,
    )

    model_fc = ExtendRegressionHead(config_fc)
    output_fc = model_fc.forward(x, t)
    print("✓ FC encoding test successful:")
    print(f"  - Output shape: {output_fc.shape}")

    print("\n🎉 All tests passed! Refactoring is successful.")


if __name__ == "__main__":
    test_extend_regression_head()
