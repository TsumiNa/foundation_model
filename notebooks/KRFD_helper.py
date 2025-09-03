# -*- coding: utf-8 -*-


import random

import matplotlib.pyplot as plt
import numpy as np


def swap_train_val_split(df, swap_ratio=0.1, random_seed=None):
    """
    随机交换 train 和 val 的 split 标签，返回新的 split Series。
    参数:
        df: 输入的 DataFrame，需包含 'split' 列
        swap_ratio: 交换比例（float，0~1），表示 train/val 中要交换的样本比例
        random_seed: 随机种子，默认为None表示每次都随机
    返回:
        pd.Series，index与输入一致，内容为调整后的 split
    """
    split = df["split"].copy()
    train_idx = split[split == "train"].index
    val_idx = split[split == "val"].index
    n_swap = int(min(len(train_idx), len(val_idx)) * swap_ratio)
    if n_swap == 0:
        return split

    rng = np.random.default_rng(random_seed)
    swap_train = rng.choice(train_idx, n_swap, replace=False)
    swap_val = rng.choice(val_idx, n_swap, replace=False)

    split.loc[swap_train] = "val"
    split.loc[swap_val] = "train"
    return split


def plot_prediction_pairs(
    samples,
    *,
    n=9,
    seed=42,
    prop_name="",
    title_prefix="Sample",
    show_true=True,
    show_pred=True,
    serial_name="",
    save_dir=None,
):
    """
    输入 samples: List of (t, v_true) 或 (t, v_true, v_pred)
    自动判断有无 pred，进行单曲线或对比绘图
    只取每条曲线中间90%的数据进行绘制
    show_true: 是否绘制真实曲线
    show_pred: 是否绘制预测曲线
    xlabel: x轴标签，默认为""
    额外输出本次抽中的indices
    save_dir: 图片保存路径（str或Path），默认None表示不保存
    """
    random.seed(seed)  # 固定随机种子以确保可重复性

    indices = random.sample(range(len(samples)), n)
    print("plot_prediction_pairs indices:", indices)  # 输出抽中的indices
    n_cols = int(np.sqrt(n))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        entry = samples[idx]
        if len(entry) == 2:
            t, v_true = entry
            v_pred = None
        elif len(entry) == 3:
            t, v_true, v_pred = entry
        else:
            raise ValueError("Each sample must be (t, v_true) or (t, v_true, v_pred)")

        t = t.squeeze(-1).cpu().numpy()
        v_true = v_true.squeeze(-1).cpu().numpy()
        n_points = len(t)
        start = int(n_points * 0.05)
        end = int(n_points * 0.95)
        t_mid = t[start:end]
        v_true_mid = v_true[start:end]

        ax = axes[i]
        if show_true:
            ax.plot(t_mid, v_true_mid, label="True", linewidth=1.5)
        if show_pred and v_pred is not None:
            v_pred = v_pred.squeeze(-1).cpu().numpy()
            v_pred_mid = v_pred[start:end]
            ax.plot(t_mid, v_pred_mid, label="Pred", linewidth=1.5)
        ax.set_title(f"{title_prefix} #{idx}")
        ax.set_xlabel(serial_name)
        ax.set_ylabel(prop_name)
        ax.legend()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches="tight")
        print(f"Saved figure to {save_dir}")
    plt.show()
