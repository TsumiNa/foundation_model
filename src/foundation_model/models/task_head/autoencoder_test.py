# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from foundation_model.models.model_config import _AEConfig
from foundation_model.models.task_head.autoencoder import AutoEncoderHead


@pytest.fixture
def linear_head():
    cfg = _AEConfig(dims=[8, 16, 4], nonnegative=False)
    return AutoEncoderHead(cfg)


@pytest.fixture
def nonneg_head():
    cfg = _AEConfig(dims=[8, 16, 4], nonnegative=True)
    return AutoEncoderHead(cfg)


def test_linear_output_shape(linear_head):
    x = torch.randn(5, 8)
    out = linear_head(x)
    assert out.shape == (5, 4)


def test_linear_output_can_be_negative(linear_head):
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    out = linear_head(x)
    assert out.min().item() < 0, "linear head should produce negative values for some inputs"


def test_nonneg_output_shape(nonneg_head):
    x = torch.randn(5, 8)
    out = nonneg_head(x)
    assert out.shape == (5, 4)


def test_nonneg_output_all_positive(nonneg_head):
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    out = nonneg_head(x)
    assert out.min().item() >= 0, "Softplus output must be non-negative"


def test_empty_dims_raises():
    cfg = _AEConfig(dims=[], nonnegative=False)
    with pytest.raises(ValueError, match="non-empty"):
        AutoEncoderHead(cfg)


def test_single_dim_raises():
    cfg = _AEConfig(dims=[8], nonnegative=False)
    with pytest.raises(ValueError, match="output dimension"):
        AutoEncoderHead(cfg)


def test_two_dim_direct_projection():
    # dims=[latent, input_dim] — no hidden layer, single linear projection (Transformer AE case)
    cfg = _AEConfig(dims=[8, 20], nonnegative=False)
    head = AutoEncoderHead(cfg)
    x = torch.randn(5, 8)
    out = head(x)
    assert out.shape == (5, 20)
