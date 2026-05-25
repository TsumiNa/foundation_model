# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from foundation_model.models.model_config import ClassificationTaskConfig
from foundation_model.models.task_head.classification import ClassificationHead


def _head(class_weights=None, num_classes=3):
    cfg = ClassificationTaskConfig(
        name="cls", data_column="cls", dims=[8, 16, num_classes], num_classes=num_classes, class_weights=class_weights
    )
    return ClassificationHead(cfg)


def test_class_weights_none_matches_unweighted_cross_entropy():
    head = _head(class_weights=None)
    pred = torch.randn(5, 3)
    target = torch.tensor([0, 1, 2, 1, 0])
    loss = head.compute_loss(pred, target)
    expected = F.cross_entropy(pred, target)
    assert torch.allclose(loss, expected)


def test_class_weights_applied_in_loss():
    weights = [1.0, 5.0, 0.2]
    head = _head(class_weights=weights)
    pred = torch.randn(5, 3)
    target = torch.tensor([0, 1, 2, 1, 0])
    loss = head.compute_loss(pred, target)
    # The head averages weighted per-sample losses by sample count (its masking convention),
    # not by the sum of weights (F.cross_entropy's default "mean").
    per_sample = F.cross_entropy(pred, target, weight=torch.tensor(weights), reduction="none")
    expected = per_sample.sum() / target.numel()
    assert torch.allclose(loss, expected)
    # The weights buffer follows the module (saved/moved with it).
    assert "class_weights" in dict(head.named_buffers())


def test_class_weights_length_must_match_num_classes():
    with pytest.raises(ValueError, match="class_weights length"):
        _head(class_weights=[1.0, 2.0], num_classes=3)


def test_class_weights_state_dict_key_present_when_unset():
    """Whether class_weights is configured or not, the ``class_weights`` buffer key must exist
    in the state_dict — so a checkpoint saved with weights can strict-load into a head built
    without them (and vice versa). Without ``register_buffer("class_weights", None)`` the key
    only appears when weights are set, which breaks cross-config checkpoint compatibility."""
    head_unweighted = _head(class_weights=None)
    head_weighted = _head(class_weights=[1.0, 2.0, 0.5])
    assert "class_weights" in head_unweighted.state_dict()
    assert "class_weights" in head_weighted.state_dict()
    # And strict-loading across configs works in both directions (the missing/present None case).
    head_unweighted.load_state_dict(head_weighted.state_dict(), strict=True)
    head_weighted.load_state_dict(head_unweighted.state_dict(), strict=True)
