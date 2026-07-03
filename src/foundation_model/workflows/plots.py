# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Pure plotting helpers for the training workflows.

Each function takes plain data and **returns a matplotlib ``Figure``** — it has no knowledge of
the output layout; the caller hands the figure to :meth:`RunRecorder.save_figure`. Migrated from
``scripts/continual_rehearsal_common.py`` (parity / confusion / kr-sequences) and
``continual_rehearsal_full._plot_forgetting`` (refactored to take a task→color map and the set of
classification tasks instead of runner ``self`` state).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display backend needed for file output

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import r2_score  # type: ignore[import-untyped]  # noqa: E402

# --- Shared constants (migrated from continual_rehearsal_common) --------------------------

#: Single blue for every regression parity scatter and KR-prediction line.
SCATTER_COLOR = "#2563EB"
#: Orange highlight for inverse-design "discovered" elements.
DISCOVERED_ELEMENT_COLOR = "#E67E22"
#: Merged material_type label set (canonical index order: class 0 == "AC").
MATERIAL_TYPE_CLASSES: tuple[str, ...] = ("AC", "QC", "others")
#: Confusion-matrix axis order (bottom-left → top-right, minority QC upper-right).
MATERIAL_TYPE_DISPLAY_ORDER: tuple[str, ...] = ("others", "AC", "QC")


def plot_parity(true: np.ndarray, pred: np.ndarray, *, r2: float, title: str) -> Figure:
    """Regression parity scatter (true vs predicted) with ideal line and an R² annotation."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true, pred, s=14, alpha=0.55, color=SCATTER_COLOR, edgecolor="none")
    lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#444444", ls="--", lw=1.2, label="ideal")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.text(
        0.04,
        0.96,
        f"R² = {r2:.3f}\nn = {len(true)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d0d0d0", alpha=0.9),
    )
    ax.legend(loc="lower right")
    return fig


def plot_confusion(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    num_classes: int,
    acc: float,
    title: str,
    special_material_type: bool = False,
) -> Figure:
    """Row-normalised confusion matrix (recall diagonal), annotated with fraction + raw count.

    When ``special_material_type`` is set (the merged 3-class material_type task) axes are
    reordered to ``MATERIAL_TYPE_DISPLAY_ORDER``.
    """
    counts = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            counts[t, p] += 1
    # The special material-type ordering only applies to the merged 3-class encoding; for any
    # other class count fall back to plain integer labels so ticks and labels always match.
    if special_material_type and num_classes == len(MATERIAL_TYPE_DISPLAY_ORDER):
        labels = list(MATERIAL_TYPE_DISPLAY_ORDER)
        perm = [MATERIAL_TYPE_CLASSES.index(lbl) for lbl in labels]
    else:
        labels = [str(i) for i in range(num_classes)]
        perm = list(range(num_classes))
    counts = counts[np.ix_(perm, perm)]
    row_sums = counts.sum(axis=1, keepdims=True)
    row_frac = np.divide(counts, row_sums, out=np.zeros(counts.shape, dtype=float), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(row_frac, cmap="Blues", vmin=0.0, vmax=1.0, origin="lower")
    fig.colorbar(im, ax=ax, label="row-normalized fraction (recall)", fraction=0.046, pad=0.04)
    ax.set_xticks(range(num_classes), labels, rotation=45, ha="right")
    ax.set_yticks(range(num_classes), labels)
    for i in range(num_classes):
        for j in range(num_classes):
            if counts[i, j]:
                ax.text(
                    j,
                    i,
                    f"{row_frac[i, j] * 100:.0f}%\n{counts[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if row_frac[i, j] > 0.5 else "#333333",
                )
    ax.grid(False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.text(
        0.5,
        -0.22,
        f"accuracy = {acc:.3f}  ·  n = {int(counts.sum())}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    return fig


def plot_kr_sequences(
    comps: Sequence[str],
    t_list: Sequence[np.ndarray],
    true_parts: Sequence[np.ndarray],
    pred: np.ndarray,
    *,
    title: str,
) -> Figure | None:
    """Per-composition KR sequence panels (up to 3), each with its own R² annotation.

    Returns ``None`` when there are no compositions to plot (small KR datasets can produce an
    empty test set at a given step).
    """
    k = min(3, len(comps))
    if k == 0:
        return None
    fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 3.7), squeeze=False)
    offset = 0
    line_true = line_pred = None
    for i in range(k):
        ax = axes[0][i]
        n = int(true_parts[i].size)
        t = np.asarray(t_list[i])
        true_i = np.asarray(true_parts[i])
        pred_i = pred[offset : offset + n]
        order = np.argsort(t)
        (line_true,) = ax.plot(t[order], true_i[order], color="#444444", lw=1.8, label="True")
        (line_pred,) = ax.plot(t[order], pred_i[order], color=SCATTER_COLOR, lw=1.6, ls="--", label="Predicted")
        ax.set_xlabel("t")
        if i == 0:
            ax.set_ylabel("Value")
        r2_i = float(r2_score(true_i, pred_i)) if n >= 2 and float(np.var(true_i)) > 0 else float("nan")
        ax.text(
            0.96,
            0.96,
            f"R² = {r2_i:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d0d0d0", alpha=0.9),
        )
        ax.set_title(str(comps[i]), fontsize=9)
        offset += n
    if line_true is not None and line_pred is not None:
        fig.legend(
            [line_true, line_pred],
            ["True", "Predicted"],
            loc="lower left",
            ncol=2,
            bbox_to_anchor=(0.0, 1.10),
            bbox_transform=axes[0][0].transAxes,
        )
    fig.suptitle(title, y=1.24)
    return fig


def plot_forgetting(
    metric_history: Mapping[str, Sequence[tuple[int, float]]],
    *,
    task_colors: Mapping[str, str],
    clf_tasks: frozenset[str] = frozenset(),
    task_display: Mapping[str, str] | None = None,
) -> Figure:
    """Per-task primary metric across continual finetuning steps.

    ``metric_history`` maps task → ``[(step, primary_metric), ...]``. ``clf_tasks`` selects the
    classification marker/linestyle; ``task_display`` optionally maps task → pretty label.
    """
    task_display = task_display or {}
    n_tasks = sum(1 for pts in metric_history.values() if pts)
    fig, ax = plt.subplots(figsize=(14, max(5.5, 0.32 * n_tasks + 3)))
    all_steps: set[int] = set()
    for task_name, points in metric_history.items():
        if not points:
            continue
        steps = [s for s, _ in points]
        vals = [v for _, v in points]
        all_steps.update(steps)
        is_clf = task_name in clf_tasks
        label = task_display.get(task_name, task_name) + (" · accuracy" if is_clf else "")
        ax.plot(
            steps,
            vals,
            marker="s" if is_clf else "o",
            ms=5,
            ls="--" if is_clf else "-",
            color=task_colors.get(task_name, "#888888"),
            label=label,
        )
    if all_steps:
        ax.set_xticks(sorted(all_steps))
    ax.set_xlabel("Continual finetuning step (a new task is added at each step)")
    ax.set_ylabel("Primary metric  ·  R² (regression) / accuracy (classification)")
    ax.set_title("Per-task performance across continual finetuning")
    ncol = 1 if n_tasks <= 20 else 2
    ax.legend(fontsize=8, ncol=ncol, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    return fig
