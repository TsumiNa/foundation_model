# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Feature tokenization, pooling and differentiable encoder contracts."""

import pytest
import torch

from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import RegressionTaskConfig, TransformerEncoderConfig

from .foundation_encoder import FoundationEncoder, _FeatureBackbone, _TokenMLPBlock


@pytest.mark.parametrize("tokenization, group_size", [("shared", 1), ("feature", 1), ("grouped", 4)])
@pytest.mark.parametrize("pooling", ["cls", "mean", "concat"])
def test_all_tokenizers_and_pooling_preserve_output_and_input_gradients(tokenization, group_size, pooling):
    torch.manual_seed(42)
    encoder = FoundationEncoder(
        TransformerEncoderConfig(
            input_dim=12,
            d_model=8,
            nhead=2,
            num_layers=2,
            dropout=0.0,
            tokenization=tokenization,
            group_size=group_size,
            pooling=pooling,
            output_dim=6,
            norm_first=True,
            activation="gelu",
        )
    )
    x = torch.randn(3, 12, requires_grad=True)
    y = encoder(x)
    assert y.shape == (3, 6)
    y.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert (x.grad.abs().sum(dim=0) > 0).all()


def test_grouped_encoder_reordering_values_and_identities_preserves_prediction():
    encoder = FoundationEncoder(
        TransformerEncoderConfig(
            input_dim=12,
            d_model=8,
            nhead=2,
            num_layers=2,
            tokenization="grouped",
            group_size=4,
            dropout=0,
            pooling="mean",
        )
    ).eval()
    x = torch.randn(3, 12)
    order = torch.tensor([2, 0, 1])
    original = encoder(x)
    tokenizer = encoder.shared
    assert isinstance(tokenizer, _FeatureBackbone)
    with torch.no_grad():
        tokenizer.feature_weight.copy_(tokenizer.feature_weight[order].clone())
        tokenizer.feature_bias.copy_(tokenizer.feature_bias[order].clone())
    torch.testing.assert_close(original, encoder(x.reshape(3, 3, 4)[:, order].reshape(3, 12)))
    assert not torch.allclose(original, encoder(x))


def test_concat_projection_consumes_all_tokens_in_column_order():
    encoder = FoundationEncoder(
        TransformerEncoderConfig(
            input_dim=12,
            d_model=8,
            nhead=2,
            num_layers=2,
            tokenization="grouped",
            group_size=4,
            pooling="concat",
            output_dim=6,
        )
    )
    assert isinstance(encoder.shared, _FeatureBackbone)
    projection = encoder.shared.output_projection
    assert isinstance(projection, torch.nn.Linear)
    assert projection.in_features == 3 * 8
    assert projection.out_features == 6


def test_feature_attention_qkv_layers_are_independently_initialized_in_full_model():
    model = FlexibleMultiTaskModel(
        [RegressionTaskConfig(name="y", dims=[8, 4, 1])],
        encoder_config=TransformerEncoderConfig(input_dim=12, d_model=8, nhead=2, num_layers=2, tokenization="feature"),
    )
    assert isinstance(model.encoder.shared, _FeatureBackbone)
    assert isinstance(model.encoder.shared.transformer, torch.nn.TransformerEncoder)
    a, b = model.encoder.shared.transformer.layers
    assert not torch.equal(a.self_attn.in_proj_weight, b.self_attn.in_proj_weight)


def test_no_attention_control_is_trainable_and_deterministic_when_frozen():
    encoder = FoundationEncoder(
        TransformerEncoderConfig(
            input_dim=12,
            d_model=8,
            nhead=2,
            num_layers=2,
            tokenization="grouped",
            group_size=4,
            pooling="mean",
            use_attention=False,
        )
    ).eval()
    x = torch.randn(3, 12, requires_grad=True)
    torch.testing.assert_close(encoder(x), encoder(x))
    encoder(x).square().sum().backward()
    assert x.grad is not None and (x.grad.abs().sum(dim=0) > 0).all()


@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_no_attention_control_honors_normalization_order_and_activation(norm_first, activation):
    config = TransformerEncoderConfig(
        input_dim=12,
        d_model=8,
        nhead=2,
        tokenization="grouped",
        group_size=4,
        pooling="mean",
        use_attention=False,
        norm_first=norm_first,
        activation=activation,
        dropout=0,
    )
    block = _TokenMLPBlock(config, 16).eval()
    assert isinstance(block.net[1], torch.nn.ReLU if activation == "relu" else torch.nn.GELU)
    x = torch.randn(2, 3, 8)
    expected = x + block.net(block.norm(x)) if norm_first else block.norm(x + block.net(x))
    torch.testing.assert_close(block(x), expected)


def test_full_model_preserves_modern_encoder_xavier_initialization():
    model = FlexibleMultiTaskModel(
        [RegressionTaskConfig(name="y", dims=[6, 4, 1])],
        encoder_config=TransformerEncoderConfig(
            input_dim=12,
            d_model=8,
            nhead=2,
            tokenization="feature",
            pooling="concat",
            output_dim=6,
        ),
    )
    for layer in model.encoder.modules():
        if isinstance(layer, torch.nn.Linear):
            bound = (6 / (layer.in_features + layer.out_features)) ** 0.5
            assert layer.weight.abs().max() <= bound


def test_shared_sinusoidal_encoder_supports_odd_valid_width():
    encoder = FoundationEncoder(TransformerEncoderConfig(input_dim=4, d_model=9, nhead=3))
    assert encoder(torch.ones(2, 4)).shape == (2, 9)


@pytest.mark.parametrize("shape", [(2, 11), (2, 3, 4)])
def test_feature_encoder_rejects_wrong_input_shape(shape):
    encoder = FoundationEncoder(TransformerEncoderConfig(input_dim=12, d_model=8, nhead=2, tokenization="feature"))
    with pytest.raises(ValueError, match="shape|dimension"):
        encoder(torch.zeros(shape))


def test_feature_encoder_state_roundtrip_preserves_predictions():
    config = TransformerEncoderConfig(
        input_dim=12, d_model=8, nhead=2, tokenization="grouped", group_size=4, pooling="concat", output_dim=6
    )
    a, b = FoundationEncoder(config).eval(), FoundationEncoder(config).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 12)
    torch.testing.assert_close(a(x), b(x))
