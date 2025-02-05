from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_scatter_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: Optional[str] = None,
    return_stat: bool = False,
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

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

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
        f"R2 = {r_value**2:.4f}\n"
        f"MSE = {mse:.4f}\n"
        f"MAE = {mae:.4f}\n"
        f"Slope = {slope:.4f}\n"
        f"Intercept = {intercept:.4f}"
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
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
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

        # Create scatter plot
        fig, _, stat = plot_scatter_comparison(
            targets_m, preds_m, title=attributes[m], return_stat=True
        )

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
