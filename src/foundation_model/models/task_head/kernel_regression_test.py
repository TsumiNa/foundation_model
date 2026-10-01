# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the kernel-regression batch plumbing.

``expand_for_kernel_regression`` / ``reshape_kernel_regression_predictions`` were methods on
FlexibleMultiTaskModel that never touched ``self``, and had no direct tests there — they were
only ever exercised through a full forward pass. Now that they are functions with importers in
two workflow modules, their contract is worth pinning on its own: a KR sample is one composition
paired with a variable-length t-sequence, so the batch is flattened for the head and regrouped
afterwards, and the flatten/regroup pair has to be each other's inverse.
"""

from copy import deepcopy

import numpy as np
import pytest
import torch

from foundation_model.models.model_config import KernelRegressionTaskConfig
from foundation_model.models.task_head.kernel_regression import (
    KernelRegressionHead,
    expand_for_kernel_regression,
    reshape_kernel_regression_predictions,
)


@pytest.fixture
def batchnorm_head():
    torch.manual_seed(20261001)
    return KernelRegressionHead(
        KernelRegressionTaskConfig(
            name="rho", x_dim=[3, 6], t_dim=[6, 4], norm=True, residual=False, kernel_num_centers=3
        )
    ).train()


def test_masked_points_do_not_change_valid_predictions_statistics_or_gradients(batchnorm_head):
    reference = deepcopy(batchnorm_head)
    valid_x = torch.randn(4, 3, requires_grad=True)
    valid_t = torch.tensor([0.0, 6.0, 50.0, 290.0])
    x = torch.cat([valid_x.detach(), torch.full((96, 3), float("nan"))]).requires_grad_()
    t = torch.cat([valid_t, torch.full((96,), float("nan"))])
    mask = torch.arange(100) < 4
    expected = reference(valid_x, valid_t)
    actual = batchnorm_head(x, t, mask=mask)
    torch.testing.assert_close(actual[mask], expected, rtol=0, atol=0)
    assert torch.equal(actual[~mask], torch.zeros(96, 1))
    for name, buffer in reference.named_buffers():
        torch.testing.assert_close(dict(batchnorm_head.named_buffers())[name], buffer, rtol=0, atol=0)
    actual[mask].sum().backward()
    expected.sum().backward()
    assert x.grad is not None and valid_x.grad is not None
    torch.testing.assert_close(x.grad[:4], valid_x.grad, rtol=0, atol=0)
    assert torch.equal(x.grad[4:], torch.zeros(96, 3))
    for name, parameter in reference.named_parameters():
        actual_grad = dict(batchnorm_head.named_parameters())[name].grad
        if parameter.grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, parameter.grad, rtol=0, atol=0)


def test_all_missing_replay_batches_leave_head_statistics_and_inference_unchanged(batchnorm_head):
    x = torch.randn(300, 3)
    t = torch.linspace(6, 290, 300)
    batchnorm_head(x, t)
    reference = deepcopy(batchnorm_head).eval()
    buffers = {name: value.clone() for name, value in batchnorm_head.named_buffers()}
    for _ in range(60):
        result = batchnorm_head(torch.randn(16, 3), torch.zeros(16), mask=torch.zeros(16, dtype=torch.bool))
        assert torch.equal(result, torch.zeros(16, 1))
    for name, value in batchnorm_head.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    batchnorm_head.eval()
    torch.testing.assert_close(batchnorm_head(x, t), reference(x, t), rtol=0, atol=0)


def test_one_valid_point_uses_stored_batchnorm_statistics_without_updating_them(batchnorm_head):
    reference = deepcopy(batchnorm_head).eval()
    x = torch.randn(2, 3, requires_grad=True)
    t = torch.tensor([0.0, float("nan")])
    buffers = {name: value.clone() for name, value in batchnorm_head.named_buffers()}
    result = batchnorm_head(x, t, mask=torch.tensor([True, False]))
    torch.testing.assert_close(result[:1], reference(x[:1], t[:1]), rtol=0, atol=0)
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert x.grad is not None
    for name, value in batchnorm_head.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    assert batchnorm_head.training
    assert all(m.training for m in batchnorm_head.modules() if isinstance(m, torch.nn.BatchNorm1d))


@pytest.mark.parametrize("constant_branch", ["composition", "temperature"])
def test_repeated_single_branch_input_uses_stored_statistics(batchnorm_head, constant_branch):
    reference = deepcopy(batchnorm_head)
    x = torch.randn(16, 3)
    t = torch.linspace(6, 290, 16)
    branches: tuple[torch.nn.Module, ...]
    if constant_branch == "composition":
        x = x[:1].expand_as(x).clone()
        branches = (reference.beta_net, reference.mu1_net)
    else:
        t = torch.zeros(16)
        branches = (reference.mu2_net,)
    for branch in branches:
        for module in branch.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.eval()
    padded_x = torch.cat([x, torch.full((16, 3), float("nan"))])
    padded_t = torch.cat([t, torch.full((16,), float("nan"))])
    mask = torch.arange(32) < 16
    for _ in range(60):
        result = batchnorm_head(padded_x, padded_t, mask=mask)[:16]
        expected = reference(x, t)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for name, value in batchnorm_head.named_buffers():
        torch.testing.assert_close(value, dict(reference.named_buffers())[name], rtol=0, atol=0)
    result.sum().backward()
    assert any(p.grad is not None for p in batchnorm_head.parameters())


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("constant_inputs", [True, False])
def test_all_valid_mask_preserves_unmasked_behavior(batchnorm_head, training, constant_inputs):
    batchnorm_head.train(training)
    reference = deepcopy(batchnorm_head)
    x = torch.randn(4, 3)
    t = torch.tensor([0.0, 6.0, 50.0, 290.0])
    if constant_inputs:
        x = x[:1].expand_as(x).clone()
        t = torch.zeros_like(t)
    actual = batchnorm_head(x, t, mask=torch.ones(4, 1, dtype=torch.bool))
    expected = reference(x, t)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    for name, parameter in reference.named_parameters():
        actual_grad = dict(batchnorm_head.named_parameters())[name].grad
        if parameter.grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, parameter.grad, rtol=0, atol=0)


def test_singleton_forward_restores_norm_modes_after_failure(batchnorm_head, monkeypatch):
    batchnorm_head.mu1_net.eval()
    modes = {name: module.training for name, module in batchnorm_head.named_modules()}

    def fail(_):
        raise RuntimeError("injected branch failure")

    monkeypatch.setattr(batchnorm_head.mu2_net, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        batchnorm_head(torch.randn(2, 3), torch.zeros(2), mask=torch.tensor([True, False]))
    assert {name: module.training for name, module in batchnorm_head.named_modules()} == modes


def test_empty_mask_preserves_empty_point_layout(batchnorm_head):
    result = batchnorm_head(torch.empty(0, 3), torch.empty(0), mask=torch.empty(0, dtype=torch.bool))
    assert result.shape == (0, 1)
    for module in batchnorm_head.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            assert module.num_batches_tracked is not None
            assert int(module.num_batches_tracked) == 0


@pytest.mark.parametrize("mask", [torch.ones(3, dtype=torch.bool), torch.ones(4), torch.ones(2, 2, dtype=torch.bool)])
def test_kernel_head_rejects_malformed_point_masks(batchnorm_head, mask):
    with pytest.raises(ValueError, match="mask"):
        batchnorm_head(torch.randn(4, 3), torch.zeros(4), mask=mask)


def test_expand_replicates_each_row_once_per_t_value():
    h_task = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    t_sequence = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4])]

    h_expanded, t_expanded = expand_for_kernel_regression(h_task, t_sequence)

    assert h_expanded.shape == (4, 2)  # 3 + 1 t-values, feature dim preserved
    assert torch.equal(h_expanded, torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(t_expanded, torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_expand_accepts_the_legacy_padded_tensor_form():
    h_task = torch.tensor([[1.0], [2.0]])
    t_sequence = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    h_expanded, t_expanded = expand_for_kernel_regression(h_task, t_sequence)

    assert torch.equal(h_expanded, torch.tensor([[1.0], [1.0], [2.0], [2.0]]))
    assert torch.equal(t_expanded, torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_expand_rejects_a_t_sequence_that_does_not_match_the_batch():
    with pytest.raises(ValueError, match="Mismatch between batch_size"):
        expand_for_kernel_regression(torch.zeros(3, 2), [torch.tensor([0.1])])


@pytest.mark.parametrize(
    "h_task, t_sequence",
    [
        pytest.param(torch.zeros(0, 4), [], id="empty-list"),
        pytest.param(torch.zeros(0, 4), torch.zeros(0, 3), id="empty-tensor"),
        pytest.param(torch.zeros(2, 4), [torch.zeros(0), torch.zeros(0)], id="rows-with-no-t-values"),
    ],
)
def test_expand_returns_empty_tensors_rather_than_raising(h_task, t_sequence):
    """An empty batch must come back as empty tensors of the right shape.

    The empty-list case used to read ``t_sequence[0].dtype`` to pick the output dtype, which is an
    IndexError on the one input that actually reaches it.
    """
    h_expanded, t_expanded = expand_for_kernel_regression(h_task, t_sequence)

    assert h_expanded.shape == (0, 4)
    assert t_expanded.shape == (0,)
    assert t_expanded.dtype == h_task.dtype


def test_reshape_is_the_inverse_of_the_flattening():
    lengths = [3, 1, 2]
    flat = np.arange(sum(lengths), dtype=float).reshape(-1, 1)

    reshaped = reshape_kernel_regression_predictions({"dos_value": flat}, lengths)

    assert [len(part) for part in reshaped["dos_value"]] == lengths
    # (N, 1) is squeezed to (N,) so a written row reads [1.23, 4.56], not [[1.23], [4.56]].
    assert all(part.ndim == 1 for part in reshaped["dos_value"])
    assert np.array_equal(np.concatenate(reshaped["dos_value"]), flat.squeeze(axis=1))


def test_reshape_keeps_a_slot_for_a_zero_length_sample():
    """Regrouping is positional, so a sample with no t-values still has to occupy its index."""
    reshaped = reshape_kernel_regression_predictions({"dos_value": np.array([[1.0], [2.0]])}, [1, 0, 1])

    assert len(reshaped["dos_value"]) == 3
    assert reshaped["dos_value"][1].size == 0


@pytest.mark.parametrize("lengths", [[3, 1], [3, 2, 2]])
def test_reshape_rejects_lengths_that_do_not_match_the_predictions(lengths):
    """A mismatch used to pass silently, handing back short arrays and dropping the tail.

    The regroup is positional, so nothing downstream can tell a truncated sequence from a genuine
    one — the two inputs have to come from the same batch, and saying so is the only way to notice
    when they do not.
    """
    flat = np.arange(6, dtype=float).reshape(-1, 1)  # 6 rows

    with pytest.raises(ValueError, match="sequence_lengths sums to"):
        reshape_kernel_regression_predictions({"dos_value": flat}, lengths)
