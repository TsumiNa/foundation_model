# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from lightning import LightningModule
from torch.utils.data import Dataset

torch.set_float32_matmul_precision("medium")  # 推荐选项


# ==============================================================================
# 依赖的辅助模块和基类
# ==============================================================================


class FourierFeatures(nn.Module):
    """
    将标量t编码为傅里叶特征
    """

    def __init__(self, input_dim: int, mapping_size: int, scale: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.B = nn.Parameter(torch.randn((input_dim, mapping_size)) * scale, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # 确保 t 是 float 类型
        t = t.float()
        # 核心操作: (batch_size, 1) @ (1, mapping_size) -> (batch_size, mapping_size)
        x_proj = 2 * math.pi * t @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class BaseModel(LightningModule):
    """
    模型基类，包含通用的训练和优化逻辑
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
        # v 代表真值y
        return t, y, y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# ==============================================================================
# FourierMLPModel
# ==============================================================================


class FourierMLPModel(BaseModel):
    def __init__(
        self, x_dim: int, hidden_dim: int, num_layers: int, fourier_mapping_size: int, learning_rate: float = 1e-3
    ):
        super().__init__(learning_rate)
        self.save_hyperparameters()  # 保存所有超参数

        # 1. 创建t的傅里叶编码器
        self.fourier_encoder = FourierFeatures(input_dim=1, mapping_size=fourier_mapping_size)

        # 傅里叶特征的输出维度是 mapping_size * 2
        fourier_output_dim = fourier_mapping_size * 2

        # 2. 计算拼接后送入MLP的总维度
        total_input_dim = x_dim + fourier_output_dim

        # 3. 动态构建MLP层
        layers = []
        # 输入层
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # 输出层
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # a. 对 t 进行编码
        t_encoded = self.fourier_encoder(t)

        # b. 拼接 X 和编码后的 t
        combined_input = torch.cat([x, t_encoded], dim=-1)

        # c. 通过 MLP 进行预测
        y_hat = self.mlp(combined_input)
        return y_hat


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
        self.save_hyperparameters()  # 保存所有超参数

        # --- 动态定义 t 的编码器和输入维度 ---
        self.t_encoder = None
        t_input_dim = 1

        if t_encoding_method == "fourier":
            self.t_encoder = FourierFeatures(input_dim=1, mapping_size=fourier_mapping_size)
            t_input_dim = fourier_mapping_size * 2
            print(f"使用傅里叶特征编码t，编码后维度: {t_input_dim}")
        elif t_encoding_method == "fc":
            self.t_encoder = nn.Sequential(nn.Linear(1, t_embedding_dim), nn.ReLU())
            t_input_dim = t_embedding_dim
            print(f"使用可学习的FC层编码t，编码后维度: {t_input_dim}")
        elif t_encoding_method == "none":
            print("不使用任何编码，直接输入t。")
        else:
            raise ValueError("t_encoding_method 必须是 'fourier', 'fc', 或 'none'")

        # --- 定义模型的其余部分 ---
        # f_x(X) 部分
        self.f_x = nn.Sequential(nn.Linear(x_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        # f_t 和 g_t 部分，它们的输入维度由上面的逻辑决定
        self.f_t = nn.Sequential(nn.Linear(t_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.g_x = nn.Sequential(nn.Linear(x_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, interaction_dim))
        self.g_t = nn.Sequential(nn.Linear(t_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, interaction_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 确保 t 的维度正确
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # --- 根据所选方法对 t 进行编码 ---
        if self.t_encoder is not None:
            t_encoded = self.t_encoder(t)
        else:
            t_encoded = t  # 'none' 模式

        # --- 计算模型输出 ---
        fx_out = self.f_x(x)
        ft_out = self.f_t(t_encoded)
        gx_out = self.g_x(x)
        gt_out = self.g_t(t_encoded)

        interaction = (gx_out * gt_out).sum(dim=1, keepdim=True)

        y_hat = fx_out + ft_out + interaction
        return y_hat


class DOSDataset(Dataset):
    """
    将 desc(DataFrame), dos_energy(Series of list), dos(Series of list) 展开为 (D_j, t^j_i, v^j_i) 样本
    """

    def __init__(self, desc: pd.DataFrame, dos_energy: pd.Series, dos: pd.Series):
        super().__init__()
        self.samples = []
        # 遍历所有样本
        for idx in desc.index.intersection(dos.index).intersection(dos_energy.index):
            D_j = torch.tensor(desc.loc[idx].values, dtype=torch.float32)
            t_list = dos_energy.loc[idx]
            v_list = dos.loc[idx]
            # 保证长度一致
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
