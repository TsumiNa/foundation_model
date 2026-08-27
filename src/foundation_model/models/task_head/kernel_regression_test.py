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

import numpy as np
import pytest
import torch

from foundation_model.models.task_head.kernel_regression import (
    expand_for_kernel_regression,
    reshape_kernel_regression_predictions,
)


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
