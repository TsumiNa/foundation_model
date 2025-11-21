import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def plot_test_loss(df, save_dir=None, dpi=150):
    """
    绘制并保存所有 (test_loss) 相关的图表。

    参数：
    df : DataFrame
        包含 `mp_rate` 和 `xxx (test_loss)` 数据的 DataFrame。
    save_dir : str, 可选
        如果提供目录路径，则会保存图像到该目录，否则仅显示图像。
    dpi : int, 默认 150
        图片分辨率，适用于 PPT。
    """
    # **去除 mp_rate=0.8 的数据**
    df_filtered = df[df["mp_rate"] != 0.8]

    # **获取所有 `(test_loss)` 相关的列**
    test_loss_columns = [col for col in df_filtered.columns if "(test_loss)" in col]

    # **创建存储目录（如果需要）**
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # **遍历所有 `(test_loss)` 相关的列**
    for loss_col in test_loss_columns:
        # **移除该列的 NaN**
        df_cleaned = df_filtered.dropna(subset=[loss_col])

        # 按 mp_rate 分组
        grouped = df_cleaned.groupby("mp_rate")[loss_col]

        # 计算均值、标准差和样本数
        means = grouped.mean()
        stds = grouped.std()
        counts = grouped.count()

        # **移除 std 计算中产生的 NaN**
        stds = stds.fillna(0)

        # 计算 95% 置信区间
        ci95 = 1.96 * stds / np.sqrt(counts)

        # **移除 NaN 值的 mp_rate**
        valid_mask = means.notna() & ci95.notna()
        means = means[valid_mask]
        ci95 = ci95[valid_mask]

        # **确保所有数据都是浮点数**
        means_np = means.to_numpy()
        ci95_np = ci95.to_numpy()
        mp_rate_np = means.index.to_numpy()

        # **绘制 PPT 友好的图表**
        plt.figure(figsize=(8, 6), dpi=dpi)
        sns.lineplot(x=mp_rate_np, y=means_np, marker="o", label="Mean", linewidth=2.5)
        plt.fill_between(
            mp_rate_np,
            means_np - ci95_np,
            means_np + ci95_np,
            alpha=0.2,
            label="95% CI",
        )

        # **设置坐标轴标签**
        plt.xlabel("# of Material Project (%)", fontsize=16)
        plt.ylabel(f"{loss_col.replace('(test_loss)', '(MSE)')}", fontsize=16)

        # **移除标题**
        plt.legend(fontsize=14)
        plt.grid(True, linestyle="--", linewidth=1.2)
        plt.gca().spines["top"].set_linewidth(1.5)
        plt.gca().spines["right"].set_linewidth(1.5)

        # **保存或显示**
        if save_dir:
            img_path = os.path.join(
                save_dir,
                f"{loss_col.replace(' ', '_').replace('(test_loss)', '').strip()}.png",
            )
            plt.savefig(img_path, bbox_inches="tight", dpi=dpi)
            plt.close()
        else:
            plt.show()


def plot_scatter_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str | None = None,
    return_stat: bool = False,
    ax: plt.Axes | None = None,
) -> Tuple[plt.Figure, plt.Axes] | Tuple[plt.Figure, plt.Axes, Dict]:
    """
    Create a scatter plot comparing predicted vs true values.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    title : str, optional
        Title for the plot
    return_stat : bool, optional
        Whether to return statistics dictionary, by default False

    Returns
    -------
    Tuple[plt.Figure, plt.Axes] | Tuple[plt.Figure, plt.Axes, Dict]
        Figure and axes objects, and optionally statistics dictionary
    """
    # Calculate statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    if not ax:
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    else:
        fig = ax.get_figure()

    # Plot scatter points
    ax.scatter(y_true, y_pred, alpha=0.5)

    # Plot diagonal line
    ax_min = min(y_true.min(), y_pred.min())
    ax_max = max(y_true.max(), y_pred.max())
    ax.plot([ax_min, ax_max], [ax_min, ax_max], "k--", alpha=0.5)

    # Plot regression line
    x_line = np.linspace(ax_min, ax_max, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r-", alpha=0.5)

    # Add statistics text
    stats_text = (
        f"R2 = {r_value**2:.4f}\nMSE = {mse:.4f}\nMAE = {mae:.4f}\nSlope = {slope:.4f}\nIntercept = {intercept:.4f}"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Set labels and title
    ax.set_xlabel("Observation")
    ax.set_ylabel("Prediction")
    if title:
        ax.set_title(title)

    # Make plot square with equal axes
    ax.set_aspect("equal", adjustable="box")

    if return_stat:
        stats_dict = {
            "r2": r_value**2,
            "mse": mse,
            "mae": mae,
            "slope": slope,
            "intercept": intercept,
            "std_err": std_err,
            "p_value": p_value,
        }
        return fig, ax, stats_dict
    return fig, ax


def plot_predictions(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    all_masks: np.ndarray,
    attributes: list[str],
    *,
    savefig: Optional[str] = None,
    suffix: Optional[str] = None,
    no_show: bool = False,
    return_stat: bool = False,
) -> Optional[Dict]:
    """
    Generate scatter plots comparing predicted and target values for multiple attributes.

    Parameters
    ----------
    all_preds : np.ndarray
        Array containing prediction values
    all_targets : np.ndarray
        Array containing target values
    all_masks : np.ndarray
        Array containing mask indicators (1 for valid entries)
    attributes : list[str]
        List of attribute names
    savefig : str, optional
        Directory path to save plots
    suffix : str, optional
        String suffix for save directory
    no_show : bool, optional
        Whether to clear figures after creation
    return_stat : bool, optional
        Whether to return statistics dictionary

    Returns
    -------
    Optional[Dict]
        Dictionary containing statistics for each attribute if return_stat is True
    """
    all_stats = []
    for m in range(len(attributes)):
        mask_m = all_masks[:, m] == 1
        preds_m = all_preds[mask_m, m]
        targets_m = all_targets[mask_m, m]

        try:
            # Create scatter plot
            fig, _, stat = plot_scatter_comparison(targets_m, preds_m, title=attributes[m], return_stat=True)
        except ValueError:
            continue

        # Save figure if path provided
        if savefig and isinstance(savefig, str):
            from pathlib import Path

            savefig_ = f"{savefig}/{suffix if suffix else ''}"
            _ = Path(savefig_).mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{savefig_}/{attributes[m]}.png", bbox_inches="tight")

        stat["attribute"] = attributes[m]
        all_stats.append(stat)

        if no_show:
            plt.cla()
            plt.clf()
            plt.close()

    if return_stat:
        import pandas as pd

        return pd.DataFrame(all_stats)
    return None
