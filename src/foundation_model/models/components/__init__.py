"""
Component modules for the FlexibleMultiTaskModel.
"""

from .gated_fusion import GatedFusion
from .lora_adapter import LoRAAdapter
from .structure_encoder import StructureEncoder

__all__ = ["StructureEncoder", "LoRAAdapter", "GatedFusion"]
