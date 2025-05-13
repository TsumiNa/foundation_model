# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Analyzes experiment logs to plot scaling laws.

This script expects a log directory structure where each subdirectory
represents an experiment run with a specific data ratio for a task.
The subdirectory names should ideally indicate the ratio (e.g., 'ratio_010' for 0.1).

It reads 'metrics.csv' from each run, extracts a specified metric for a given task,
and plots this metric against the data ratios.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "scaling_law_plots"


def extract_ratio_from_dirname(dirname: str) -> float | None:
    """Extracts data ratio from directory names like 'ratio_010', 'ratio_0.1', '0.1'."""
    # Matches patterns like "ratio_010", "ratio_0.1", "0.1x", "data_0.1"
    match = re.search(r"ratio_?(\d+[\.,]?\d*)", dirname, re.IGNORECASE)
    if match:
        val_str = match.group(1).replace(",", ".")
        # Handle cases like "010" for 0.1 if it's a fixed format, or assume it's 10 if not scaled by 100
        # For "ratio_010" meaning 0.1, we might need to divide by 100 if it's always 3 digits
        if len(val_str) == 3 and "." not in val_str and val_str.startswith("0"):  # e.g. "010" -> 0.10
            return float(val_str) / 100.0
        return float(val_str)  # e.g. "0.1" or "1.0"

    match = re.search(r"(\d+[\.,]?\d*)x", dirname, re.IGNORECASE)  # e.g. "0.1x"
    if match:
        return float(match.group(1).replace(",", "."))

    # Fallback for just numbers like "0.1", "1.0"
    match = re.fullmatch(r"(\d+[\.,]?\d*)", dirname, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", "."))
    print(f"Warning: Could not extract ratio from directory name: {dirname}")
    return None


def power_law(x, a, b):
    """Power law function: y = a * x^b"""
    return a * np.power(x, b)


def analyze_scaling_laws(
    log_series_dir: Path,
    task_name: str,
    metric_key_suffix: str,  # e.g., "_loss", "_mae", "_accuracy"
    output_dir: Path,
    plot_title_prefix: str = "Scaling Law",
    lower_is_better: bool = True,  # For choosing 'best' vs 'last' metric
):
    """
    Parses logs, extracts metrics, plots scaling law, and attempts to fit a power law.
    """
    if not log_series_dir.is_dir():
        print(f"Error: Log series directory not found: {log_series_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []  # List of (ratio, metric_value, last_epoch)

    # Dynamically construct the metric name to look for in CSV
    # Assumes val metrics are logged as "val_<task_name><metric_key_suffix>"
    # e.g. task_name="regression_1", metric_key_suffix="_loss" -> "val_regression_1_loss"
    full_metric_name = f"val_{task_name}{metric_key_suffix}"
    if not metric_key_suffix.startswith("_"):  # ensure underscore if user forgets
        full_metric_name = f"val_{task_name}_{metric_key_suffix}"

    print(f"Analyzing logs in: {log_series_dir}")
    print(f"Target task: {task_name}, Target metric: {full_metric_name}")

    for run_dir in sorted(log_series_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        ratio = extract_ratio_from_dirname(run_dir.name)
        if ratio is None:
            print(f"Skipping directory (cannot parse ratio): {run_dir.name}")
            continue

        # Find the metrics.csv file, potentially within a 'version_X' subdirectory
        metrics_csv_path = None
        version_dirs = sorted(list(run_dir.glob("version_*")))
        if version_dirs:  # Standard Lightning structure
            # Pick the latest version directory if multiple exist
            metrics_csv_path = version_dirs[-1] / "metrics.csv"
        else:  # Check if metrics.csv is directly in run_dir (less common for CSVLogger)
            if (run_dir / "metrics.csv").exists():
                metrics_csv_path = run_dir / "metrics.csv"

        if not metrics_csv_path or not metrics_csv_path.exists():
            print(f"Warning: metrics.csv not found in {run_dir} or its version subdirectories. Skipping.")
            continue

        try:
            df = pd.read_csv(metrics_csv_path)
        except Exception as e:
            print(f"Error reading {metrics_csv_path}: {e}. Skipping.")
            continue

        if full_metric_name not in df.columns:
            print(
                f"Warning: Metric '{full_metric_name}' not found in {metrics_csv_path}. Available: {df.columns.tolist()}. Skipping run."
            )
            continue

        # Get the metric value.
        # Option 1: Last logged value for the metric (dropna to get last valid numeric value)
        # Option 2: Best value (min or max depending on 'lower_is_better')

        valid_metrics = df[[full_metric_name, "epoch"]].dropna(subset=[full_metric_name])
        if valid_metrics.empty:
            print(
                f"Warning: No valid entries found for metric '{full_metric_name}' in {metrics_csv_path}. Skipping run."
            )
            continue

        if lower_is_better:
            best_row = valid_metrics.loc[valid_metrics[full_metric_name].idxmin()]
        else:
            best_row = valid_metrics.loc[valid_metrics[full_metric_name].idxmax()]

        metric_value = best_row[full_metric_name]
        epoch_at_best = int(best_row["epoch"])

        # last_epoch_df = df[df["epoch"] == df["epoch"].max()]
        # metric_value_series = last_epoch_df[full_metric_name].dropna()
        # if metric_value_series.empty:
        #     print(f"Warning: Metric '{full_metric_name}' is all NaN at last epoch in {metrics_csv_path}. Trying overall last valid.")
        #     metric_value_series = df[full_metric_name].dropna()
        #     if metric_value_series.empty:
        #         print(f"Warning: Metric '{full_metric_name}' has no valid values in {metrics_csv_path}. Skipping run.")
        #         continue
        # metric_value = metric_value_series.iloc[-1]
        # last_epoch = df["epoch"].max()

        results.append((ratio, metric_value, epoch_at_best))
        print(
            f"  Run: {run_dir.name}, Ratio: {ratio:.3f}, Best {full_metric_name}: {metric_value:.4f} (at epoch {epoch_at_best})"
        )

    if not results:
        print("No results found to plot.")
        return

    results.sort(key=lambda x: x[0])  # Sort by ratio
    ratios = np.array([r[0] for r in results])
    metric_values = np.array([r[1] for r in results])
    epochs = np.array([r[2] for r in results])

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot metric vs ratio
    color = "tab:red"
    ax1.set_xlabel("Data Ratio for Task")
    ax1.set_ylabel(f"Best {full_metric_name}", color=color)
    ax1.plot(ratios, metric_values, marker="o", linestyle="-", color=color, label=f"Best {full_metric_name}")
    ax1.tick_params(axis="y", labelcolor=color)

    # Add epoch numbers as text annotations
    for i, txt in enumerate(epochs):
        ax1.annotate(
            f"e:{txt}",
            (ratios[i], metric_values[i]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )

    # Attempt to fit power law if there are enough data points
    fit_label = ""
    if len(ratios) > 2:
        try:
            # Provide initial guesses for parameters a, b
            # Initial guess for 'a' could be the first metric value
            # Initial guess for 'b' could be -0.5 (typical for error scaling) or 0.5
            initial_guess_b = -0.5 if lower_is_better else 0.5
            popt, pcov = curve_fit(
                power_law, ratios, metric_values, p0=[metric_values[0], initial_guess_b], maxfev=5000
            )
            a_fit, b_fit = popt
            fit_label = f"Fit: y = {a_fit:.2f} * x^({b_fit:.2f})"
            ax1.plot(ratios, power_law(ratios, *popt), linestyle="--", color="purple", label=fit_label)
            print(f"Power law fit: a={a_fit:.4f}, b={b_fit:.4f}")
        except RuntimeError:
            print("Could not fit power law to the data.")
        except Exception as e:
            print(f"Error during power law fit: {e}")

    ax1.legend(loc="upper right" if lower_is_better else "lower right")
    plt.title(f"{plot_title_prefix}: {task_name} ({full_metric_name})")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plot_filename = f"scaling_law_{task_name}_{metric_key_suffix.strip('_')}.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment logs to plot scaling laws.")
    parser.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="Directory containing the series of experiment run logs (e.g., samples/example_logs/scaling_law_regression_1)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of the task to analyze (e.g., 'regression_1').",
    )
    parser.add_argument(
        "--metric_suffix",
        type=str,
        required=True,
        help="Suffix of the validation metric key to plot (e.g., '_loss', '_mae', '_accuracy'). "
        "The script will look for 'val_<task_name><metric_suffix>'.",
    )
    parser.add_argument(
        "--output_plot_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the generated plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--plot_title", type=str, default="Scaling Law Analysis", help="Custom prefix for the plot title."
    )
    parser.add_argument(
        "--higher_is_better",
        action="store_true",
        help="Set if the metric being analyzed is better when higher (e.g., accuracy). Default is lower is better (e.g., loss).",
    )

    args = parser.parse_args()

    analyze_scaling_laws(
        log_series_dir=args.log_dir,
        task_name=args.task_name,
        metric_key_suffix=args.metric_suffix,
        output_dir=args.output_plot_dir,
        plot_title_prefix=args.plot_title,
        lower_is_better=not args.higher_is_better,
    )

    # Example usage:
    # python samples/helper_tools/scaling_law_analyzer.py \
    #   --log_dir samples/example_logs/scaling_law_regression_1 \
    #   --task_name regression_1 \
    #   --metric_suffix _mae \
    #   --output_plot_dir samples/scaling_law_plots
    #
    # python samples/helper_tools/scaling_law_analyzer.py \
    #   --log_dir samples/example_logs/scaling_law_classification_A \
    #   --task_name classification_A \
    #   --metric_suffix _accuracy \
    #   --higher_is_better \
    #   --output_plot_dir samples/scaling_law_plots
    #
    # To install dependencies if not already present (using uv):
    # uv add pandas matplotlib numpy scipy
