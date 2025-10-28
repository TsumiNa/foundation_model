# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Kernel regression head for sequence regression tasks.

This implementation follows the internal research prototype in
``notebooks/KRFD_model.py`` and adapts it to the FlexibleMultiTaskModel
infrastructure. It decomposes predictions into kernel-weighted terms and
low-dimensional baselines over the shared representation and the sequencing
parameter ``t``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray

from foundation_model.models.fc_layers import LinearBlock
from foundation_model.models.model_config import KernelRegressionTaskConfig

from .base import BaseTaskHead


class _GaussianKernel(nn.Module):
    """
    Gaussian kernel with learnable centres/sigmas.
    """

    def __init__(
        self,
        centers: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        learnable_centers: bool,
        learnable_sigmas: bool,
        min_sigma: float,
    ):
        super().__init__()

        centers = centers.flatten().float()
        sigmas = sigmas.flatten().float()
        if centers.ndim != 1 or sigmas.ndim != 1:
            raise ValueError("centers and sigmas must be 1D tensors.")
        if centers.numel() != sigmas.numel():
            raise ValueError("centers and sigmas must have the same length.")
        if centers.numel() == 0:
            raise ValueError("Kernel requires at least one center.")

        self.min_sigma = float(max(min_sigma, 1e-6))

        if learnable_centers:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers, persistent=True)

        safe_sigmas = sigmas.clamp_min(self.min_sigma)
        log_sigma = safe_sigmas.log()
        if learnable_sigmas:
            self.log_sigmas = nn.Parameter(log_sigma)
        else:
            self.register_buffer("log_sigmas", log_sigma, persistent=True)

    @property
    def sigmas(self) -> torch.Tensor:
        return self.log_sigmas.exp().clamp_min(self.min_sigma)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        t = t.view(-1, 1)
        diff = t - self.centers.view(1, -1)
        denom = 2.0 * (self.sigmas.view(1, -1) ** 2)
        return torch.exp(-(diff**2) / denom)


def _build_dims(input_dim: int, hidden_override: Optional[List[int]], fallback: List[int]) -> List[int]:
    dims = [input_dim]
    if hidden_override is not None:
        dims.extend(hidden_override)
    else:
        dims.extend(fallback)

    if len(dims) < 2:
        # Ensure there is at least one hidden layer
        dims.append(max(1, input_dim))
    return dims


class KernelRegressionHead(BaseTaskHead):
    """
    Regression head that models ``y(X, t)`` via Gaussian kernels:

        y(X, t) = Σ_i k(t, c_i) β_i(X) + μ₁(X) + μ₂(t) + μ₃(X, t)

    where ``k`` is a Gaussian kernel with centres ``c_i`` and learnable (or fixed)
    bandwidths, β(X) produces kernel weights, μ₁/μ₂/μ₃ are low-capacity baselines.
    """

    def __init__(self, config: KernelRegressionTaskConfig):
        super().__init__(config)

        if not config.x_dim:
            raise ValueError("KernelRegressionTaskConfig.x_dim must contain at least the input dimension.")
        if not config.t_dim:
            raise ValueError("KernelRegressionTaskConfig.t_dim must contain at least one dimension.")
        if config.kernel_num_centers <= 0:
            raise ValueError("kernel_num_centers must be positive.")

        self.enable_mu3 = config.enable_mu3
        self.n_kernels = int(config.kernel_num_centers)

        # --- Raw t input (align with research prototype) ---
        t_input_dim = 1

        # --- Kernel initialisation ---
        centers = self._init_centers_tensor(config)
        sigmas = self._init_sigmas_tensor(config)
        self.kernel = _GaussianKernel(
            centers,
            sigmas,
            learnable_centers=config.kernel_learnable_centers,
            learnable_sigmas=config.kernel_learnable_sigmas,
            min_sigma=config.kernel_min_sigma,
        )

        # --- β(X) branch ---
        shared_x_hidden = config.x_dim[1:]
        beta_dims = _build_dims(config.x_dim[0], config.beta_hidden_dims, shared_x_hidden)
        self.beta_net = LinearBlock(
            beta_dims + [self.n_kernels],
            normalization=config.norm,
            residual=config.residual,
        )

        # --- μ₁(X) branch ---
        mu1_dims = _build_dims(config.x_dim[0], config.mu1_hidden_dims, shared_x_hidden)
        self.mu1_net = LinearBlock(
            mu1_dims + [1],
            normalization=config.norm,
            residual=config.residual,
        )

        # --- μ₂(t) branch ---
        shared_t_hidden = config.t_dim
        mu2_dims = _build_dims(t_input_dim, config.mu2_hidden_dims, shared_t_hidden)
        self.mu2_net = LinearBlock(
            mu2_dims + [1],
            normalization=config.norm,
            residual=config.residual,
        )

        # --- μ₃(X, t) branch (optional) ---
        self.mu3_net: Optional[LinearBlock]
        if self.enable_mu3:
            mu3_hidden = config.mu3_hidden_dims if config.mu3_hidden_dims is not None else shared_x_hidden
            mu3_dims = _build_dims(config.x_dim[0] + t_input_dim, mu3_hidden, shared_x_hidden)
            self.mu3_net = LinearBlock(
                mu3_dims + [1],
                normalization=config.norm,
                residual=config.residual,
            )
        else:
            self.mu3_net = None

    def _init_centers_tensor(self, config: KernelRegressionTaskConfig) -> torch.Tensor:
        if config.kernel_centers_init is not None:
            centers = torch.as_tensor(config.kernel_centers_init, dtype=torch.float32)
            if centers.numel() != self.n_kernels:
                raise ValueError(
                    f"kernel_centers_init length ({centers.numel()}) must equal kernel_num_centers ({self.n_kernels})."
                )
            return centers

        start, end = config.kernel_init_range
        if start == end:
            centers = torch.full((self.n_kernels,), float(start), dtype=torch.float32)
        else:
            centers = torch.linspace(float(start), float(end), steps=self.n_kernels)
        return centers

    def _init_sigmas_tensor(self, config: KernelRegressionTaskConfig) -> torch.Tensor:
        if config.kernel_sigmas_init is not None:
            sigmas = torch.as_tensor(config.kernel_sigmas_init, dtype=torch.float32)
            if sigmas.numel() != self.n_kernels:
                raise ValueError(
                    f"kernel_sigmas_init length ({sigmas.numel()}) must equal kernel_num_centers ({self.n_kernels})."
                )
            return sigmas.clamp_min(config.kernel_min_sigma)

        return torch.full((self.n_kernels,), float(config.kernel_init_sigma), dtype=torch.float32)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)

        beta = self.beta_net(x)
        mu1 = self.mu1_net(x)
        mu2 = self.mu2_net(t)

        k_t = self.kernel(t.squeeze(1))
        kernel_term = (k_t * beta).sum(dim=1, keepdim=True)

        if self.mu3_net is not None:
            xt = torch.cat([x, t], dim=1)
            mu3 = self.mu3_net(xt)
        else:
            mu3 = 0.0

        return kernel_term + mu1 + mu2 + mu3

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if pred.dim() == 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 2 and target.shape[1] == 1:
            target = target.squeeze(1)

        if mask is None:
            mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
        elif mask.dim() == 2 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        valid_count = mask.sum()
        if valid_count == 0:
            return None

        losses = F.mse_loss(pred, target, reduction="none") * mask
        return losses.sum() / valid_count

    def _predict_impl(self, x: torch.Tensor) -> dict[str, ndarray]:
        return {"value": x.detach().cpu().numpy()}


__all__ = ["KernelRegressionHead"]
