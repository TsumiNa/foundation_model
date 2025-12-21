"""
AutoEncoder task head for the FlexibleMultiTaskModel.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from numpy import ndarray

from foundation_model.models.components.fc_layers import LinearBlock
from foundation_model.models.model_config import AutoEncoderTaskConfig

from .base import BaseTaskHead


class AutoEncoderHead(BaseTaskHead):
    """
    AutoEncoder task head for reconstructing the input.

    Parameters
    ----------
    config : AutoEncoderTaskConfig
        Configuration object containing parameters like input dimension (`d_in`),
        task name (`name`), hidden dimensions (`dims`), normalization (`norm`),
        and residual connections (`residual`).
    """

    def __init__(self, config: AutoEncoderTaskConfig):
        super().__init__(config)

        if not hasattr(config, "dims") or not config.dims:
            raise ValueError("AutoEncoderHead config must have 'dims' attribute, and it cannot be empty.")
        d_in = config.dims[0]
        head_internal_dims = config.dims[1:]
        if not head_internal_dims:
            raise ValueError(
                "AutoEncoderHead config 'dims' must include at least an output dimension after the input dimension."
            )

        norm = config.norm
        residual = config.residual

        self.net = LinearBlock(
            [d_in] + head_internal_dims[:-1],
            normalization=norm,
            residual=residual,
            dim_output_layer=head_internal_dims[-1],
            output_active=torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the autoencoder head (decoder).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (latent representation), shape (B, d_in).
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        torch.Tensor
            Reconstructed input, shape (B, output_dim).
        """
        return self.net(x)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute masked MSE loss for reconstruction.

        Parameters
        ----------
        pred : torch.Tensor
            Reconstructed values, shape (B, output_dim).
        target : torch.Tensor
            Original input values, shape (B, output_dim).
        mask : torch.Tensor, optional
            Binary mask indicating valid targets.

        Returns
        -------
        torch.Tensor | None
            Total loss as a scalar tensor, or None if no valid samples.
        """
        if mask is None:
            mask = torch.ones_like(target)

        # Check if there are any valid samples
        valid_count = mask.sum()
        if valid_count == 0:
            # No valid samples in this batch for this task
            return None

        # Apply mask to both predictions and targets
        losses = F.mse_loss(pred, target, reduction="none") * mask

        # Compute total loss
        total_loss = losses.sum() / valid_count

        return total_loss

    def _predict_impl(self, x: torch.Tensor) -> Dict[str, ndarray]:
        """
        Core prediction logic for autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Raw output from the forward pass (reconstructed values).

        Returns
        -------
        Dict[str, ndarray]
            A dictionary containing the reconstructed values: {"reconstruction": x}.
        """
        return {"reconstruction": x.detach().cpu().numpy()}
