# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from lightning import LightningModule
from torch.utils.data import Dataset

torch.set_float32_matmul_precision("medium")  # Recommended option


# ==============================================================================
# Helper modules and base class
# ==============================================================================


class FourierFeatures(nn.Module):
    """
    Encode scalar t into Fourier features
    """

    def __init__(self, input_dim: int, mapping_size: int, scale: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.B = nn.Parameter(torch.randn((input_dim, mapping_size)) * scale, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # Ensure t is float type
        t = t.float()
        # Core operation: (batch_size, 1) @ (1, mapping_size) -> (batch_size, mapping_size)
        x_proj = 2 * math.pi * t @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class BaseModel(LightningModule):
    """
    Base model class with common training and optimization logic
    """

    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, t, y = batch
        y_hat = self.forward(x, t)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t, y = batch
        y_hat = self.forward(x, t)
        loss = self.loss_fn(y_hat, y)
        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, t, y = batch
        y_hat = self.forward(x, t)
        loss = self.loss_fn(y_hat, y)
        self.log("test/loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, t, y = batch
        y_hat = self.forward(x, t)
        # v is the ground truth y
        return t, y, y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# ==============================================================================
# DecompositionModel
# ==============================================================================


class FourierDecompositionModel(BaseModel):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        interaction_dim: int,
        t_encoding_method: str = "fourier",  # 'fourier', 'fc', or 'none'
        fourier_mapping_size: int = 32,
        t_embedding_dim: int = 64,
        learning_rate: float = 1e-3,
    ):
        super().__init__(learning_rate)
        self.save_hyperparameters()  # Save all hyperparameters

        # --- Dynamically define t encoder and input dimension ---
        self.t_encoder = None
        t_input_dim = 1

        if t_encoding_method == "fourier":
            self.t_encoder = FourierFeatures(input_dim=1, mapping_size=fourier_mapping_size)
            t_input_dim = fourier_mapping_size * 2
            print(f"Using Fourier feature encoding for t, encoded dimension: {t_input_dim}")
        elif t_encoding_method == "fc":
            self.t_encoder = nn.Sequential(nn.Linear(1, t_embedding_dim), nn.ReLU())
            t_input_dim = t_embedding_dim
            print(f"Using learnable FC layer encoding for t, encoded dimension: {t_input_dim}")
        elif t_encoding_method == "none":
            print("No encoding for t, using t directly.")
        else:
            raise ValueError("t_encoding_method must be 'fourier', 'fc', or 'none'")

        # --- Define the rest of the model ---
        # f_x(X) part
        self.f_x = nn.Sequential(nn.Linear(x_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        # f_t and g_t parts, their input dimension is determined above
        self.f_t = nn.Sequential(nn.Linear(t_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.g_x = nn.Sequential(nn.Linear(x_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, interaction_dim))
        self.g_t = nn.Sequential(nn.Linear(t_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, interaction_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Ensure t has correct dimension
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # --- Encode t according to the selected method ---
        if self.t_encoder is not None:
            t_encoded = self.t_encoder(t)
        else:
            t_encoded = t  # 'none' mode

        # --- Compute model output ---
        fx_out = self.f_x(x)
        ft_out = self.f_t(t_encoded)
        gx_out = self.g_x(x)
        gt_out = self.g_t(t_encoded)

        interaction = (gx_out * gt_out).sum(dim=1, keepdim=True)

        y_hat = fx_out + ft_out + interaction
        return y_hat


class DOSDataset(Dataset):
    """
    Expand desc(DataFrame), dos_energy(Series of list), dos(Series of list) into (D_j, t^j_i, v^j_i) samples
    """

    def __init__(self, desc: pd.DataFrame, dos_energy: pd.Series, dos: pd.Series):
        super().__init__()
        self.samples = []
        # Iterate over all samples
        for idx in desc.index.intersection(dos.index).intersection(dos_energy.index):
            D_j = torch.tensor(desc.loc[idx].values, dtype=torch.float32)
            t_list = dos_energy.loc[idx]
            v_list = dos.loc[idx]
            # Ensure lengths are consistent
            if len(t_list) != len(v_list):
                continue
            for t_i, v_i in zip(t_list, v_list):
                t_tensor = torch.tensor([t_i], dtype=torch.float32)  # (1,)
                v_tensor = torch.tensor([v_i], dtype=torch.float32)  # (1,)
                self.samples.append((D_j, t_tensor, v_tensor))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]
