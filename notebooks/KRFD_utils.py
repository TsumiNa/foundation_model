# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import lightning as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def init_centers_and_sigmas(
    t_train: List[float] | np.ndarray | torch.Tensor | None,
    n_kernels: int,
    method: str = "quantile",
    sigma_alpha: float = 0.5,
    t_range: Tuple[float, float] = (0.0, 1.0),
    inverse_density: bool = False,
    density_smoothing: float = 0.3,
    **kwargs,
):
    """
    Initialize kernel centers and sigmas from training data distribution.

    Args:
        t_train: Array-like training time values (list, numpy array, tensor, etc.) or None
        n_kernels: Number of kernel centers to create
        method: 'quantile' | 'hist'
        sigma_alpha: Scale factor to convert local spacing -> sigma
        t_range: Fallback range when t_train is None
        inverse_density: If False (default), sample more in high-density areas.
                        If True, sample more in low-density (sparse) areas.
        density_smoothing: Float in [0,1]. Prevents over-concentration in extreme density regions.
                          When inverse_density=False: prevents over-concentration in high-density areas.
                          When inverse_density=True: prevents over-concentration in low-density areas.
                          0=no smoothing (may over-concentrate), 1=full smoothing (nearly uniform).
        **kwargs: Additional parameters (bins, density_bins, seed)

    Returns:
        centers: torch.Tensor[n_kernels] - kernel center positions
        sigmas: torch.Tensor[n_kernels] - kernel bandwidths
    """
    if t_train is None:
        centers = torch.linspace(t_range[0], t_range[1], n_kernels)
        sigmas = torch.full((n_kernels,), float(max(1e-3, sigma_alpha)))
        return centers, sigmas

    # Convert to numpy array regardless of input type
    t = np.asarray(t_train).flatten()

    if method == "quantile":
        if inverse_density:
            # 反向密度采样：在低密度区域多放kernels
            bins = kwargs.get("density_bins", 64)
            hist, edges = np.histogram(t, bins=bins, density=True)

            # 计算反密度
            pdf = 1.0 / (hist + 1e-8)

            # 应用平滑以防止过分集中在极低密度区域
            if density_smoothing > 0:
                smooth_power = 1.0 - density_smoothing
                pdf = pdf**smooth_power

            pdf = pdf / pdf.sum()  # 归一化

            # 根据平滑后的反密度创建累积分布并采样
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            cum_pdf = np.cumsum(pdf)
            cum_pdf = cum_pdf / cum_pdf[-1]

            qs = np.linspace(0.0, 1.0, num=n_kernels)
            centers = np.interp(qs, cum_pdf, bin_centers)
        else:
            # 正常密度采样：在高密度区域多放kernels
            bins = kwargs.get("density_bins", 64)
            hist, edges = np.histogram(t, bins=bins, density=True)

            # 使用原始密度
            pdf = hist + 1e-8

            # 应用平滑以防止过分集中在极高密度区域
            if density_smoothing > 0:
                smooth_power = 1.0 - density_smoothing
                pdf = pdf**smooth_power

            pdf = pdf / pdf.sum()  # 归一化

            # 根据平滑后的密度创建累积分布并采样
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            cum_pdf = np.cumsum(pdf)
            cum_pdf = cum_pdf / cum_pdf[-1]

            qs = np.linspace(0.0, 1.0, num=n_kernels)
            centers = np.interp(qs, cum_pdf, bin_centers)

    elif method == "hist":
        bins = kwargs.get("bins", 64)
        hist, edges = np.histogram(t, bins=bins, density=True)
        mids = 0.5 * (edges[:-1] + edges[1:])

        if inverse_density:
            # 反向密度采样：在低密度区域多放kernels
            pdf = 1.0 / (hist + 1e-8)
        else:
            # 正常密度采样：在高密度区域多放kernels
            pdf = hist + 1e-8

        # 应用平滑以防止过分集中
        if density_smoothing > 0:
            smooth_power = 1.0 - density_smoothing
            pdf = pdf**smooth_power

        pdf = pdf / pdf.sum()  # 归一化
        rng = np.random.default_rng(kwargs.get("seed", 0))
        idx = rng.choice(len(mids), size=n_kernels, replace=True, p=pdf)
        centers = mids[idx]

    else:
        raise ValueError("Unknown method for init_centers_and_sigmas. Use 'quantile' or 'hist'.")

    centers = torch.tensor(np.sort(centers), dtype=torch.float32)

    # derive sigma from neighbor spacing
    c = centers.detach().cpu().numpy()
    M = len(c)
    if M == 1:
        sigmas = torch.tensor([max(1e-3, sigma_alpha)], dtype=torch.float32)
        return centers, sigmas
    d = np.zeros(M)
    d[0] = c[1] - c[0]
    d[-1] = c[-1] - c[-2]
    for i in range(1, M - 1):
        d[i] = 0.5 * ((c[i] - c[i - 1]) + (c[i + 1] - c[i]))
    sigma = np.maximum(1e-3, sigma_alpha * d)
    sigmas = torch.tensor(sigma, dtype=torch.float32)
    return centers, sigmas


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


class DOSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        descriptor: pd.DataFrame,
        dos_energy: pd.Series,
        dos: pd.Series,
        serial: Optional[pd.Series] = None,  # index->"train"/"val"/"test"
        batch_size: int = 32,
        random_seed: int = 42,
        train_ratio: float = 1.0,  # 新增选项，控制train数量比例
    ):
        super().__init__()
        self.desc = descriptor
        self.dos_energy = dos_energy
        self.dos = dos
        self.serial = serial
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.train_ratio = train_ratio

    def setup(self, stage=None):
        rng = np.random.RandomState(self.random_seed)
        if self.serial is not None:
            train_idx = self.serial[self.serial == "train"].index
            val_idx = self.serial[self.serial == "val"].index
            test_idx = self.serial[self.serial == "test"].index

            # 按train_ratio采样train
            n_train = int(len(train_idx) * self.train_ratio)
            if n_train < len(train_idx):
                perm = rng.permutation(len(train_idx))
                train_idx = train_idx[perm[:n_train]]
        else:
            # 自动划分
            all_idx = np.array(list(self.desc.index.intersection(self.dos.index).intersection(self.dos_energy.index)))
            perm = rng.permutation(len(all_idx))
            n = len(all_idx)
            n_train = int(n * 0.7 * self.train_ratio)
            n_val = int(n * 0.1)
            train_idx = all_idx[perm[:n_train]]
            val_idx = all_idx[perm[n_train : n_train + n_val]]
            test_idx = all_idx[perm[n_train + n_val :]]

        self.train_dataset = DOSDataset(
            self.desc.loc[train_idx], self.dos_energy.loc[train_idx], self.dos.loc[train_idx]
        )
        self.val_dataset = DOSDataset(self.desc.loc[val_idx], self.dos_energy.loc[val_idx], self.dos.loc[val_idx])
        self.test_dataset = DOSDataset(self.desc.loc[test_idx], self.dos_energy.loc[test_idx], self.dos.loc[test_idx])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
