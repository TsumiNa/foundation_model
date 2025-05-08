"""
Base sequence task head interface for the FlexibleMultiTaskModel.
"""

from ..base import SequenceBaseHead

# This module re-exports the SequenceBaseHead class from ..base
# with the appropriate import path for sequence head modules
# to avoid circular imports

__all__ = ["SequenceBaseHead"]
