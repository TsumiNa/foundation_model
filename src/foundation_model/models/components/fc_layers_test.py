# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch

from .fc_layers import LinearLayer


def test_single_composition_uses_running_statistics_and_retains_gradients() -> None:
    torch.manual_seed(7)
    layer = LinearLayer(3, 5).train()
    reference = deepcopy(layer).eval()
    x = torch.randn(1, 3, requires_grad=True)
    buffers = {k: v.clone() for k, v in layer.named_buffers()}
    actual = layer(x)
    torch.testing.assert_close(actual, reference(x), rtol=0, atol=0)
    actual.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    for key, value in layer.named_buffers():
        torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
    assert layer.normal is not None
    assert layer.training and layer.normal.training


@pytest.mark.parametrize("training", [True, False])
def test_multiple_compositions_preserve_batchnorm_computation(training: bool) -> None:
    layer = LinearLayer(3, 5).train(training)
    reference = deepcopy(layer)
    assert reference.normal is not None
    x = torch.randn(7, 3)
    actual = layer(x)
    expected = reference.activation(reference.normal(reference.layer(x)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key, value in layer.named_buffers():
        torch.testing.assert_close(value, dict(reference.named_buffers())[key], rtol=0, atol=0)
