# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Shared evaluation-dump + plotting helpers used by both continual-rehearsal runners.

:mod:`continual_rehearsal_demo` (educational, single scenario) and
:mod:`continual_rehearsal_full` (formal, three scenarios) previously carried near-identical
copies of these functions as bound methods on their respective ``Runner`` classes. The
duplication caused at least one drift incident (the ``_plot_kr_sequences`` ``NameError``
on empty ``comps`` was fixed in ``full`` first; ``demo`` carried the broken copy for
several PRs). Centralising the pure helpers here prevents future drift.

What's in scope here:

* **Constants** the two runners share (the project's plot palette and the merged
  material_type 3-class ordering).
* **Pure dumpers** — `(composition, true, pred)` parquet + per-task ``<task>_metrics.json``
  emitted at every step. No model / runner state needed.
* **Pure plotters** — parity scatter, confusion matrix, kernel-regression sequences.
  Each takes a per-task ``title`` argument so the runner-specific task display vocabulary
  (``TASK_DISPLAY`` / ``_title()`` / ``_display()``) stays in its home file.

What's NOT in scope here:

* Anything that needs ``Runner`` state (data caches, ``TASK_SPECS``, model parameters).
* The forgetting trajectory plot (uses per-runner ``_task_colors``).
* The inverse-design plotters (different per runner — single-scenario vs eight-path).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import r2_score  # type: ignore[import-untyped]

# --- Shared constants ----------------------------------------------------------------------

#: Single blue used for every regression parity scatter and KR-prediction line — keeps the
#: meaning of "predicted vs ideal" colour-consistent across regression and kernel-regression
#: panels. PR #18 settled on this exact tone.
SCATTER_COLOR = "#2563EB"

#: The merged material_type label set (5 fine classes → 3). The order here is the *canonical*
#: index order (so ``MATERIAL_TYPE_CLASSES[0] == "AC"`` means merged class 0 is AC, etc.).
MATERIAL_TYPE_CLASSES: tuple[str, ...] = ("AC", "QC", "others")

#: Display order for the confusion-matrix axes. Bottom-left → top-right diagonal places the
#: minority QC class in the upper-right corner, mirroring the canonical "others → AC → QC"
#: progression the project standardised on in PR #18.
MATERIAL_TYPE_DISPLAY_ORDER: tuple[str, ...] = ("others", "AC", "QC")


# --- Pure dumpers --------------------------------------------------------------------------


def dump_predictions(
    task_name: str,
    step_dir: Path,
    *,
    comps: list[str],
    true: np.ndarray,
    pred: np.ndarray,
) -> None:
    """Persist ``(composition, true, pred)`` for a regression or classification task as parquet.

    Single row per test sample. The trio is enough for downstream re-plotting (parity scatter
    for regression, confusion matrix for classification) without re-running the model.
    """
    pd.DataFrame({"composition": comps, "true": true, "pred": pred}).to_parquet(step_dir / f"{task_name}_pred.parquet")


def dump_kr_predictions(
    task_name: str,
    step_dir: Path,
    *,
    comps: list[str],
    t_list: list[np.ndarray],
    true_parts: list[np.ndarray],
    pred: np.ndarray,
) -> None:
    """Persist kernel-regression test predictions in long form: one row per ``(composition, t)``.

    The flat ``pred`` array carries every composition's values back-to-back; we re-split it
    using each composition's ``true_parts`` length so the long-form table is fully reconstructible.
    """
    rows: list[dict[str, object]] = []
    offset = 0
    for comp, t_arr, y_true in zip(comps, t_list, true_parts):
        n = int(y_true.size)
        for k in range(n):
            rows.append(
                {
                    "composition": comp,
                    "t": float(t_arr[k]),
                    "true": float(y_true[k]),
                    "pred": float(pred[offset + k]),
                }
            )
        offset += n
    pd.DataFrame(rows).to_parquet(step_dir / f"{task_name}_pred.parquet")


def dump_metrics(task_name: str, step_dir: Path, metric: dict[str, float]) -> None:
    """Drop the per-task metric dict next to the parquet, for quick human / scripted inspection."""
    (step_dir / f"{task_name}_metrics.json").write_text(json.dumps(metric, indent=2), encoding="utf-8")


# --- Pure plotters -------------------------------------------------------------------------


def plot_parity(
    true: np.ndarray,
    pred: np.ndarray,
    task_name: str,
    r2: float,
    step_dir: Path,
    *,
    title: str,
) -> None:
    """Regression parity scatter (true vs predicted) with ideal-line and an R² annotation."""
    fig, ax = plt.subplots(figsize=(5, 5))
    # Uniform colour/alpha for every regression parity scatter — set in PR #18.
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
    fig.savefig(step_dir / f"{task_name}_parity.png")
    plt.close(fig)


def plot_confusion(
    true: np.ndarray,
    pred: np.ndarray,
    task_name: str,
    acc: float,
    step_dir: Path,
    num_classes: int,
    *,
    title: str,
    special_material_type: bool = False,
) -> None:
    """Row-normalised confusion matrix.

    When ``special_material_type`` is set (the merged 3-class material_type task), axes are
    reordered to ``MATERIAL_TYPE_DISPLAY_ORDER`` so the recall diagonal runs bottom-left →
    top-right with the minority QC class in the upper-right corner.
    """
    counts = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            counts[t, p] += 1
    # Display order + bottom-left origin (PR #18 standardisation).
    if special_material_type:
        labels = list(MATERIAL_TYPE_DISPLAY_ORDER[:num_classes])
        perm = [MATERIAL_TYPE_CLASSES.index(lbl) for lbl in labels]
    else:
        labels = [str(i) for i in range(num_classes)]
        perm = list(range(num_classes))
    counts = counts[np.ix_(perm, perm)]
    # Colour by row-normalised fraction (recall) so a dominant class doesn't leave every other
    # row invisible. Annotate each cell with both the fraction and the raw count.
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
    fig.savefig(step_dir / f"{task_name}_confusion.png")
    plt.close(fig)


def plot_kr_sequences(
    comps: list[str],
    t_list: list,  # list of torch.Tensor — kept as Any to avoid importing torch here
    true_parts: list[np.ndarray],
    pred: np.ndarray,
    task_name: str,
    step_dir: Path,
    *,
    title: str,
) -> None:
    """Per-composition KR sequence panels — up to 3 panels, each with its own R² annotation.

    Empty ``comps`` (no test samples for the task at this step — possible on very small KR
    datasets like ``magnetic_susceptibility``) used to silently break here: ``min(3, 0) == 0``
    skipped the loop, then ``fig.legend([line_true, line_pred], …)`` raised ``NameError`` on
    unbound names. Now we short-circuit with a warning and return without writing a PNG.
    """
    k = min(3, len(comps))
    if k == 0:
        logger.warning(f"plot_kr_sequences: no compositions for '{task_name}' — skipping plot.")
        return
    fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 3.7), squeeze=False)
    offset = 0
    line_true = line_pred = None
    for i in range(k):
        ax = axes[0][i]
        n = true_parts[i].size
        t = t_list[i].cpu().numpy()
        true_i = np.asarray(true_parts[i])
        pred_i = pred[offset : offset + n]
        order = np.argsort(t)  # left-to-right curve
        (line_true,) = ax.plot(t[order], true_i[order], color="#444444", lw=1.8, label="True")
        # Same blue as the regression parity scatter — keeps "Predicted" colour consistent
        # across regression / kernel-regression panels.
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
        ax.set_title(comps[i], fontsize=9)
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
    fig.savefig(step_dir / f"{task_name}_sequences.png")
    plt.close(fig)
