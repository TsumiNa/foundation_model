"""
Classification task head for the FlexibleMultiTaskModel.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.lora_adapter import LoRAAdapter
from ..fc_layers import LinearBlock
from ..model_config import ClassificationTaskConfig  # Changed import
from .base import BaseTaskHead


class ClassificationHead(BaseTaskHead):
    """
    Classification task head for predicting categorical values.

    Parameters
    ----------
    config : ClassificationTaskConfig
        Configuration object containing parameters like input dimension (`d_in`),
        task name (`name`), hidden dimensions (`dims`), number of classes (`num_classes`),
        normalization (`norm`), residual connections (`residual`), LoRA rank (`lora_rank`),
        and LoRA alpha (`lora_alpha`).
    """

    def __init__(self, config: ClassificationTaskConfig):  # Changed signature
        super().__init__(config)

        if not hasattr(config, "dims") or not config.dims:
            raise ValueError("ClassificationHead config must have 'dims' attribute, and it cannot be empty.")
        d_in = config.dims[0]  # d_in sourced from config
        hidden_dims = config.dims[1:]  # hidden_dims are the rest

        num_classes = config.num_classes
        norm = config.norm
        residual = config.residual

        # LoRA specific attributes from config
        lora_enabled = getattr(config, "lora_enabled", False)
        lora_rank = getattr(config, "lora_rank", 0)
        lora_alpha = getattr(config, "lora_alpha", 1.0)
        lora_freeze_base = getattr(config, "lora_freeze_base", True)

        # Ensure at least 2 classes for classification
        if num_classes < 2:
            raise ValueError("Number of classes must be at least 2 for classification.")

        # Determine input dimension for the output layer
        last_hidden_dim = hidden_dims[-1] if hidden_dims else d_in

        # Construct the network using LinearBlock for hidden layers if hidden_dims are provided
        if hidden_dims:
            self.hidden_layers = LinearBlock(
                [d_in] + hidden_dims,  # Use d_in and hidden_dims
                normalization=norm,
                residual=residual,
            )
        else:
            self.hidden_layers = nn.Identity()

        # Separate output layer for classification
        self.output_layer = nn.Linear(last_hidden_dim, num_classes)

        # Apply LoRA to the output layer if requested
        if lora_enabled and lora_rank > 0:  # Check enabled flag and rank
            self.output_layer = LoRAAdapter(
                self.output_layer, r=lora_rank, alpha=lora_alpha, freeze_base=lora_freeze_base
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
            # Ensure mask has the same shape as target for broadcasting with losses
            mask = torch.ones_like(target, device=pred.device, dtype=torch.float)

        # Ensure target is long type for cross_entropy
        target = target.long()

        # Individual sample losses - apply mask after loss calculation
        losses = F.cross_entropy(pred, target, reduction="none")
        masked_losses = losses * mask  # Apply mask here

        # Compute per-class losses (average loss for samples belonging to each class)
        per_class_loss = torch.zeros(self.num_classes, device=pred.device)
        # Use scatter_add_ for efficient aggregation if possible, otherwise loop
        # Note: scatter_add_ requires target to be long and indices within range
        # Ensure target indices are valid before using scatter_add_
        valid_mask = mask > 0
        if valid_mask.any():
            # Calculate sum of losses per class for valid samples
            class_loss_sum = torch.zeros(self.num_classes, device=pred.device).scatter_add_(
                0, target[valid_mask], masked_losses[valid_mask]
            )
            # Calculate count per class for valid samples
            class_count = torch.zeros(self.num_classes, device=pred.device).scatter_add_(
                0, target[valid_mask], torch.ones_like(target[valid_mask], dtype=torch.float)
            )
            # Calculate average loss per class, handle division by zero
            per_class_loss = torch.nan_to_num(class_loss_sum / class_count.clamp_min(1.0))

        # Total loss (average over valid samples)
        total_loss = masked_losses.sum() / mask.sum().clamp_min(1.0)

        return total_loss, per_class_loss

    def _predict_impl(self, x: torch.Tensor, additional: bool = False) -> dict[str, torch.Tensor]:
        """
        Core prediction logic for classification.

        Parameters
        ----------
        x : torch.Tensor
            Raw logits from the forward pass.
        additional : bool, optional
            If True, return both labels and probabilities.
            If False, return only labels. Defaults to False.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing prediction results:
            - {"labels": labels} if additional is False.
            - {"labels": labels, "probabilities": probabilities} if additional is True.
        """
        probabilities = F.softmax(x, dim=-1)
        labels = torch.argmax(probabilities, dim=-1)

        if additional:
            return {"labels": labels, "probabilities": probabilities}
        else:
            return {"labels": labels}
