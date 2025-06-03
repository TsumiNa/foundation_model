# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Callbacks for PyTorch Lightning training scripts.
"""

from .prediction_writer import PredictionDataFrameWriter

__all__ = ["PredictionDataFrameWriter"]
