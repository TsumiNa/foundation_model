"""
Classification task head for the FlexibleMultiTaskModel.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray

from foundation_model.models.components.lora_adapter import LoRAAdapter
from foundation_model.models.fc_layers import LinearBlock
from foundation_model.models.model_config import ClassificationTaskConfig  # Changed import

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
        # pred is (B, C), target is (B, 1) from dataloader, mask is (B, 1) from dataloader

        # 1. Process target to be (B,) and long type for F.cross_entropy
        if target.ndim == 2 and target.shape[1] == 1:
            final_target_for_loss = target.squeeze(-1).long()
        elif target.ndim == 1:
            final_target_for_loss = target.long()
        else:
            raise ValueError(f"Classification target has unexpected shape: {target.shape}. Expected (B, 1) or (B,).")

        if final_target_for_loss.shape[0] != pred.shape[0] or final_target_for_loss.ndim != 1:
            raise ValueError(
                f"Internal shape error for classification target. "
                f"Pred shape: {pred.shape}, Processed target shape: {final_target_for_loss.shape}. Expected target (B,)."
            )

        # 2. Process mask to be (B,) and float type for multiplication
        if mask is None:
            # If no mask provided, all samples are valid. Create a 1D mask of shape (B,).
            mask_1d = torch.ones(pred.shape[0], device=pred.device, dtype=torch.float)
        elif mask.ndim == 2 and mask.shape[1] == 1:  # Expected (B, 1)
            mask_1d = mask.squeeze(-1).float()
        elif mask.ndim == 1:  # Allow (B,)
            mask_1d = mask.float()
        else:
            raise ValueError(f"Classification mask has unexpected shape: {mask.shape}. Expected (B, 1) or (B,).")

        if mask_1d.shape[0] != pred.shape[0] or mask_1d.ndim != 1:
            raise ValueError(
                f"Internal shape error for classification mask. "
                f"Pred shape: {pred.shape}, Processed mask shape: {mask_1d.shape}. Expected mask (B,)."
            )

        # 3. Individual sample losses
        losses = F.cross_entropy(pred, final_target_for_loss, reduction="none", ignore_index=-1)  # losses is (B,)
        masked_losses = losses * mask_1d  # Apply 1D mask, result is (B,)

        # 4. Compute per-class losses
        per_class_loss = torch.zeros(self.num_classes, device=pred.device)
        valid_mask_1d = mask_1d > 0  # boolean, shape (B,)

        if valid_mask_1d.any():
            target_valid = final_target_for_loss[valid_mask_1d]  # (N_valid,)
            masked_losses_valid = masked_losses[valid_mask_1d]  # (N_valid,)

            class_loss_sum = torch.zeros(self.num_classes, device=pred.device).scatter_add_(
                0, target_valid, masked_losses_valid
            )
            class_count = torch.zeros(self.num_classes, device=pred.device).scatter_add_(
                0, target_valid, torch.ones_like(target_valid, dtype=torch.float)
            )
            per_class_loss = torch.nan_to_num(class_loss_sum / class_count.clamp_min(1.0))

        # 5. Total loss
        total_loss = masked_losses.sum() / mask_1d.sum().clamp_min(1.0)

        return total_loss, per_class_loss

    def _predict_impl(self, x: torch.Tensor, additional: bool = False) -> dict[str, ndarray]:
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
            - {"label": label} if additional is False.
            - {"label": label, "proba": proba} if additional is True.
        """
        proba = F.softmax(x, dim=-1)
        label = torch.argmax(proba, dim=-1)

        if additional:
            return {"label": label.detach().cpu().numpy(), "proba": proba.detach().cpu().numpy()}
        else:
            return {"label": label.detach().cpu().numpy()}
