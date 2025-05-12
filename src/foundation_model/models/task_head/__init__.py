# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Task heads for the FlexibleMultiTaskModel.
"""

# create_task_heads function removed as its logic is now in FlexibleMultiTaskModel._build_task_heads

# Optionally, define __all__ if this module is intended for wildcard imports,
# or just let imports be direct from submodules.
# For example:
from .base import BaseTaskHead
from .classification import ClassificationHead
from .regression import RegressionHead
from .sequence import SequenceBaseHead, create_sequence_head  # and specific sequence head types if needed

__all__ = [
    "BaseTaskHead",
    "ClassificationHead",
    "RegressionHead",
    "SequenceBaseHead",
    "create_sequence_head",
]
