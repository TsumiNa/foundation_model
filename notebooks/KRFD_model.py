# -*- coding: utf-8 -*-
"""
Lightning implementation:

Final model equation:
    y(X, t) = Σ_i k(t, t_i) β_i(X) + μ1(X) + μ2(t) + μ3(X, t)

Where:
- k(t, t_i) = exp(-(t - t_i)^2 / (2σ_i^2)) is the Gaussian kernel
- β(X): coefficients depending on X (with optional L2 regularization)
- μ1(X): baseline depending only on X
- μ2(t): baseline depending only on t
- μ3(X,t): baseline depending on interaction of X and t

Features:
- Implemented with `lightning` (new API)
- Gaussian kernel: centers and σ can be learnable
- β(X), μ1(X), μ2(t), μ3(X,t) are configurable MLPs
- Includes synthetic dataset, DataLoader, training/validation/testing pipeline
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from lightning import LightningModule
from torch import nn

# -------------------------
# Utility: build MLP
# -------------------------


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: Tuple[int, ...] = (128, 128),
    activation: str = "gelu",
    dropout: float = 0.0,
    use_batchnorm: bool = False,
) -> nn.Module:
    """Build a simple MLP with optional activation, batch norm and dropout."""
    acts = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(acts.get(activation, nn.GELU)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def init_weights_xavier_uniform(module: nn.Module):
    """Apply xavier_uniform initialization to all linear layers in a module."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# -------------------------
# Model: Gaussian kernel
# -------------------------
class _GaussianKernel(nn.Module):
    """
    Gaussian kernel: k(t, t_i) = exp(-(t - t_i)^2 / (2σ_i^2))
    """

    def __init__(
        self,
        centers: torch.Tensor,
        sigmas: torch.Tensor,
        learnable_centers: bool = False,
        learnable_sigmas: bool = False,
        min_sigma: float = 1e-3,
    ):
        super().__init__()
        M = centers.numel()
        if centers.ndim != 1:
            raise ValueError("centers must be 1D [M]")
        if sigmas.ndim == 0:
            sigmas = sigmas.expand(M)
        elif sigmas.ndim != 1 or sigmas.numel() != M:
            raise ValueError("sigmas must be scalar or 1D with same length as centers")

        self.min_sigma = float(min_sigma)
        if learnable_centers:
            self.centers = nn.Parameter(centers.clone())
        else:
            self.register_buffer("centers", centers.clone())
        if learnable_sigmas:
            self.log_sigma = nn.Parameter(sigmas.clamp_min(self.min_sigma).log())
        else:
            self.register_buffer("log_sigma", sigmas.clamp_min(self.min_sigma).log())

    @property
    def sigmas(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp_min(self.min_sigma)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix for inputs t against all centers."""
        t = t.view(-1, 1)
        diff = t - self.centers.view(1, -1)
        denom = 2.0 * (self.sigmas.view(1, -1) ** 2)
        K = torch.exp(-(diff**2) / denom)
        return K


# -------------------------
# Main model: β(X), μ1(X), μ2(t), μ3(X,t)
# -------------------------


@dataclass
class ModelConfig:
    # property
    prop_name: str
    # input/output dimensions
    x_dim: int
    n_kernels: int
    # β(X)
    beta_hidden: Tuple[int, ...] = (128, 128)
    beta_activation: str = "gelu"
    beta_dropout: float = 0.0
    beta_batchnorm: bool = False
    # independent learning rate and weight decay for beta
    beta_weight_decay: float | None = None
    beta_lr: float | None = None  # defaults to lr if None
    # independent learning rate and weight decay for mu3
    mu3_weight_decay: float | None = None  # defaults to other_weight_decay if None
    mu3_lr: float | None = None  # defaults to lr if None
    # other/global lr & decay
    other_weight_decay: float = 1e-4
    lr: float = 1e-3
    # μ1(X)
    mu1_hidden: Tuple[int, ...] = (128,)
    mu1_activation: str = "gelu"
    mu1_dropout: float = 0.0
    mu1_batchnorm: bool = False
    # μ2(t)
    mu2_hidden: Tuple[int, ...] = (64,)
    mu2_activation: str = "gelu"
    mu2_dropout: float = 0.0
    mu2_batchnorm: bool = False
    # μ3(X,t)
    enable_mu3: bool = True
    mu3_hidden: Tuple[int, ...] = (128,)
    mu3_activation: str = "gelu"
    mu3_dropout: float = 0.0
    mu3_batchnorm: bool = False
    # kernel options
    learnable_centers: bool = False
    learnable_sigmas: bool = True
    init_sigma: float = 0.15
    # logging options
    log_points: int = 5  # number of points to log across kernels


class KernelRegression(LightningModule):
    """
    LightningModule implementing the kernel regression model:

    Final model equation:
        y(X, t) = Σ_i k(t, t_i) β_i(X) + μ1(X) + μ2(t) + μ3(X, t)

    Where:
    - k(t, t_i) = exp(-(t - t_i)^2 / (2σ_i^2)) is the Gaussian kernel
    - β(X): coefficients depending on X (with optional L2 regularization)
    - μ1(X): baseline depending only on X
    - μ2(t): baseline depending only on t
    - μ3(X,t): baseline depending on interaction of X and t

    Features:
    - Gaussian kernel: centers and σ can be learnable
    - β(X), μ1(X), μ2(t), μ3(X,t) are configurable MLPs
    - Training/validation/testing pipeline with logging of kernel parameters
    - Parameter groups for optimizer with configurable learning rates and weight decays
    """

    def __init__(
        self, cfg: ModelConfig, kernel_centers: torch.Tensor, kernel_sigmas: torch.Tensor, init_weights: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kernel_centers", "kernel_sigmas"])
        self.cfg = cfg

        M = cfg.n_kernels
        # validate kernel hyperparameters shape
        if kernel_centers.ndim != 1 or kernel_centers.numel() != M:
            raise ValueError("kernel_centers must be 1D tensor of length n_kernels")
        if kernel_sigmas.ndim != 1 or kernel_sigmas.numel() != M:
            raise ValueError("kernel_sigmas must be 1D tensor of length n_kernels")

        # Gaussian kernel module
        self.kernel = _GaussianKernel(
            centers=kernel_centers.float(),
            sigmas=kernel_sigmas.float(),
            learnable_centers=cfg.learnable_centers,
            learnable_sigmas=cfg.learnable_sigmas,
        )

        # β(X) branch outputs per-kernel weights
        self.beta_net = make_mlp(
            cfg.x_dim, M, cfg.beta_hidden, cfg.beta_activation, cfg.beta_dropout, cfg.beta_batchnorm
        )

        # μ1(X) branch
        self.mu1_net = make_mlp(cfg.x_dim, 1, cfg.mu1_hidden, cfg.mu1_activation, cfg.mu1_dropout, cfg.mu1_batchnorm)

        # μ2(t) branch
        self.mu2_net = make_mlp(1, 1, cfg.mu2_hidden, cfg.mu2_activation, cfg.mu2_dropout, cfg.mu2_batchnorm)

        # μ3(X,t) optional interaction branch
        if cfg.enable_mu3:
            self.mu3_net = make_mlp(
                cfg.x_dim + 1, 1, cfg.mu3_hidden, cfg.mu3_activation, cfg.mu3_dropout, cfg.mu3_batchnorm
            )
        else:
            self.mu3_net = None

        # Apply xavier_uniform initialization if requested
        if init_weights:
            init_weights_xavier_uniform(self.beta_net)
            init_weights_xavier_uniform(self.mu1_net)
            init_weights_xavier_uniform(self.mu2_net)
            if self.mu3_net is not None:
                init_weights_xavier_uniform(self.mu3_net)

        # metrics
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing y(X,t).

        Args:
            x: [batch_size, x_dim] input features
            t: [batch_size] or [batch_size, 1] time values

        Returns:
            y: [batch_size, 1] predicted output
        """
        K = self.kernel(t)
        beta = self.beta_net(x)
        mu1 = self.mu1_net(x)
        mu2 = self.mu2_net(t.view(-1, 1))
        kbeta = (K * beta).sum(dim=1, keepdim=True)
        if self.mu3_net is not None:
            xt = torch.cat([x, t.view(-1, 1)], dim=1)
            mu3 = self.mu3_net(xt)
        else:
            mu3 = 0.0
        y = kbeta + mu1 + mu2 + mu3
        return y

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: tuple of (x, t, y)
            batch_idx: index

        Returns:
            loss
        """
        x, t, y = batch
        y_hat = self(x, t)
        loss = self.mse(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Log selected kernel parameters (centers and sigmas) to avoid slowdown.
        # We log at indices {0, s, 2s, ..., N} where N=M-1 and s≈N/(log_points-1).
        M = int(self.kernel.centers.numel())
        N = M - 1
        P = max(2, int(self.cfg.log_points))  # ensure at least {0, N}
        step = max(1, round(N / (P - 1)))
        idxs = sorted({min(i * step, N) for i in range(P)})

        if self.cfg.learnable_centers:
            centers_np = self.kernel.centers.detach().cpu().numpy()
            for i in idxs:
                self.log(f"center/{i}", float(centers_np[i]), prog_bar=False, on_step=False, on_epoch=True)
        if self.cfg.learnable_sigmas:
            sig_np = self.kernel.sigmas.detach().cpu().numpy()
            for i in idxs:
                self.log(f"sigma/{i}", float(sig_np[i]), prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: tuple of (x, t, y)
            batch_idx: index
        """
        x, t, y = batch
        y_hat = self(x, t)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        self.log_dict({"val_mse": mse, "val_mae": mae}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: tuple of (x, t, y)
            batch_idx: index
        """
        x, t, y = batch
        y_hat = self(x, t)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        self.log_dict({"test_mse": mse, "test_mae": mae}, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for Lightning prediction loop.

        Args:
            batch: tuple of (x, t, y)
            batch_idx: index
            dataloader_idx: index

        Returns:
            tuple of (t, y, y_hat)
        """
        x, t, y = batch
        y_hat = self(x, t)
        return t, y, y_hat

    def configure_optimizers(self):
        """
        AdamW with parameter groups:
        - β(X) branch: beta_weight_decay and beta_lr (defaults to lr if None)
        - other branches (μ1/μ2/kernel): other_weight_decay and lr
        - optional μ3 branch: mu3_weight_decay/mu3_lr (inherit if None)
        Bias and normalization parameters are excluded from decay.

        Returns:
            optimizer
        """

        def is_no_decay(name: str, p: torch.Tensor) -> bool:
            return p.ndim == 1 or ("bias" in name.lower()) or ("bn" in name.lower()) or ("norm" in name.lower())

        beta_decay, beta_no_decay = [], []
        for n, p in self.beta_net.named_parameters():
            (beta_no_decay if is_no_decay(n, p) else beta_decay).append(p)

        other_modules = [("mu1", self.mu1_net), ("mu2", self.mu2_net), ("kernel", self.kernel)]
        other_decay, other_no_decay = [], []
        for mname, mod in other_modules:
            for n, p in mod.named_parameters():
                (other_no_decay if is_no_decay(n, p) else other_decay).append(p)

        param_groups = []
        beta_lr = self.cfg.beta_lr if self.cfg.beta_lr is not None else self.cfg.lr
        mu3_lr = self.cfg.mu3_lr if self.cfg.mu3_lr is not None else self.cfg.lr
        mu3_wd = self.cfg.mu3_weight_decay if self.cfg.mu3_weight_decay is not None else self.cfg.other_weight_decay

        if beta_decay:
            param_groups.append({"params": beta_decay, "weight_decay": self.cfg.beta_weight_decay, "lr": beta_lr})
        if beta_no_decay:
            param_groups.append({"params": beta_no_decay, "weight_decay": 0.0, "lr": beta_lr})
        if other_decay:
            param_groups.append({"params": other_decay, "weight_decay": self.cfg.other_weight_decay, "lr": self.cfg.lr})
        if other_no_decay:
            param_groups.append({"params": other_no_decay, "weight_decay": 0.0, "lr": self.cfg.lr})

        # μ3 branch (optional)
        if self.mu3_net is not None:
            mu3_decay, mu3_no_decay = [], []
            for n, p in self.mu3_net.named_parameters():
                (mu3_no_decay if is_no_decay(n, p) else mu3_decay).append(p)
            if mu3_decay:
                param_groups.append({"params": mu3_decay, "weight_decay": mu3_wd, "lr": mu3_lr})
            if mu3_no_decay:
                param_groups.append({"params": mu3_no_decay, "weight_decay": 0.0, "lr": mu3_lr})

        optimizer = torch.optim.AdamW(param_groups)
        return optimizer
