"""
Classification task head for the FlexibleMultiTaskModel.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.lora_adapter import LoRAAdapter
from ..fc_layers import LinearBlock
from .base import BaseTaskHead


class ClassificationHead(BaseTaskHead):
    """
    Classification task head for predicting categorical values.

    Parameters
    ----------
    d_in : int
        Input dimension to the task head.
    name : str
        Name of the classification task.
    dims : list[int]
        Hidden dimensions of the classification network.
    num_classes : int
        Number of classes to predict.
    norm : bool
        Whether to apply layer normalization.
    residual : bool
        Whether to use residual connections.
    lora_rank : Optional[int]
        If not None, apply LoRA adaptation with the specified rank.
    lora_alpha : float
        Scaling factor for LoRA adaptation.
    """

    def __init__(
        self,
        d_in: int,
        name: str,
        dims: list[int],
        num_classes: int,
        norm: bool = True,
        residual: bool = False,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 1.0,
    ):
        super().__init__(d_in, name)

        # Ensure at least 2 classes for classification
        if num_classes < 2:
            raise ValueError("Number of classes must be at least 2")

        # Construct the network using LinearBlock for hidden layers
        self.hidden_layers = LinearBlock(
            [d_in] + dims,
            normalization=norm,
            residual=residual,
        )

        # Separate output layer for classification
        self.output_layer = nn.Linear(dims[-1], num_classes)

        # Apply LoRA to the output layer if requested
        if lora_rank is not None and lora_rank > 0:
            self.output_layer = LoRAAdapter(
                self.output_layer, r=lora_rank, alpha=lora_alpha, freeze_base=True
            )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the classification head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, d_in).
        **kwargs : dict
            Additional arguments (not used in classification head).

        Returns
        -------
        torch.Tensor
            Predicted logits, shape (B, num_classes).
        """
        features = self.hidden_layers(x)
        logits = self.output_layer(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, d_in).

        Returns
        -------
        torch.Tensor
            Predicted probabilities, shape (B, num_classes).
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, d_in).

        Returns
        -------
        torch.Tensor
            Predicted class indices, shape (B,).
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-entropy loss for classification.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted logits, shape (B, num_classes).
        target : torch.Tensor
            Target class indices, shape (B,).
        mask : torch.Tensor, optional
            Binary mask indicating valid samples, shape (B,).
            If None, all samples are considered valid.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (total_loss, per_class_loss) where total_loss is a scalar tensor
            and per_class_loss contains loss per class.
        """
        if mask is None:
            mask = torch.ones(pred.size(0), device=pred.device)

        # Individual sample losses
        losses = F.cross_entropy(pred, target, reduction="none") * mask

        # Compute per-class losses
        per_class_loss = torch.zeros(self.num_classes, device=pred.device)
        for c in range(self.num_classes):
            class_mask = (target == c) & (mask.bool())
            if class_mask.sum() > 0:
                per_class_loss[c] = losses[class_mask].mean()

        # Total loss (average over valid samples)
        total_loss = losses.sum() / mask.sum().clamp_min(1.0)

        return total_loss, per_class_loss
