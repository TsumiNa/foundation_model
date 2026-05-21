"""
AutoEncoder task head for the FlexibleMultiTaskModel.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from numpy import ndarray

from foundation_model.models.components.fc_layers import LinearBlock
from foundation_model.models.model_config import _AEConfig

from .base import BaseTaskHead


class AutoEncoderHead(BaseTaskHead):
    """
    AutoEncoder task head for reconstructing the input.

    Parameters
    ----------
    config : _AEConfig
        Internal configuration object. ``dims`` must be fully resolved before
        instantiation (first element = latent_dim, last element = input_dim).
        ``nonnegative=True`` applies Softplus to the output; ``False`` uses a
        linear output (no activation).
    """

    def __init__(self, config: _AEConfig):
        super().__init__(config)

        if not config.dims:
            raise ValueError("AutoEncoderHead: config.dims must be resolved (non-empty) before instantiation.")
        d_in = config.dims[0]
        head_internal_dims = config.dims[1:]
        if not head_internal_dims:
            raise ValueError(
                "AutoEncoderHead: config.dims must include at least an output dimension after the input dimension."
            )

        output_act = torch.nn.Softplus() if config.nonnegative else None

        if len(head_internal_dims) == 1:
            # Direct projection with no hidden layers (e.g. Transformer AE: [latent_dim, input_dim]).
            # Mirrors the dim_output_layer path which also skips normalization on the final layer.
            self.net = LinearBlock(
                [d_in, head_internal_dims[0]],
                normalization=False,
                output_active=output_act,
            )
        else:
            self.net = LinearBlock(
                [d_in] + head_internal_dims[:-1],
                normalization=config.norm,
                residual=config.residual,
                dim_output_layer=head_internal_dims[-1],
                output_active=output_act,
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
