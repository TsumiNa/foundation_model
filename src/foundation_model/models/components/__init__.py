# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Model components package for foundation model architecture.

This package contains modular components used to build the foundation model architecture,
including encoders, fusion mechanisms, and self-supervised learning modules.
"""

from .foundation_encoder import FoundationEncoder
from .gated_fusion import GatedFusion
from .self_supervised import SelfSupervisedModule
from .structure_encoder import StructureEncoder

__all__ = [
    "FoundationEncoder",
    "GatedFusion",
    "SelfSupervisedModule",
    "StructureEncoder",
]
