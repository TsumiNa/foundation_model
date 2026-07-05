# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for :mod:`foundation_model.workflows.plots` (render without error)."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from foundation_model.workflows import plots


def test_plot_parity_returns_figure() -> None:
    fig = plots.plot_parity(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]), r2=0.9, title="t")
    assert isinstance(fig, Figure)
    plots.plt.close(fig)


def test_plot_confusion_returns_figure() -> None:
    fig = plots.plot_confusion(np.array([0, 1, 2, 1]), np.array([0, 1, 1, 2]), num_classes=3, acc=0.5, title="t")
    assert isinstance(fig, Figure)
    plots.plt.close(fig)


def test_plot_confusion_special_material_type() -> None:
    fig = plots.plot_confusion(
        np.array([0, 1, 2]), np.array([0, 1, 2]), num_classes=3, acc=1.0, title="t", special_material_type=True
    )
    assert isinstance(fig, Figure)
    plots.plt.close(fig)


def test_plot_confusion_special_flag_ignored_for_non_three_classes() -> None:
    # special_material_type must not break when num_classes != 3 (falls back to integer labels).
    fig = plots.plot_confusion(
        np.array([0, 1, 4]), np.array([0, 2, 4]), num_classes=5, acc=0.6, title="t", special_material_type=True
    )
    assert isinstance(fig, Figure)
    plots.plt.close(fig)


def test_plot_kr_sequences_returns_figure() -> None:
    comps = ["Fe2 O3", "Al2 O3"]
    t_list = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])]
    true_parts = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5])]
    pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    fig = plots.plot_kr_sequences(comps, t_list, true_parts, pred, title="t")
    assert isinstance(fig, Figure)
    plots.plt.close(fig)


def test_plot_kr_sequences_empty_returns_none() -> None:
    assert plots.plot_kr_sequences([], [], [], np.array([]), title="t") is None


def test_plot_forgetting_returns_figure() -> None:
    history = {"a": [(1, 0.5), (2, 0.6)], "b": [(2, 0.7)]}
    fig = plots.plot_forgetting(history, task_colors={"a": "#111111", "b": "#222222"}, clf_tasks=frozenset({"b"}))
    assert isinstance(fig, Figure)
    plots.plt.close(fig)


def test_plot_saves_to_file(tmp_path) -> None:
    fig = plots.plot_parity(np.array([1.0, 2.0]), np.array([1.0, 2.0]), r2=1.0, title="t")
    out = tmp_path / "parity.png"
    fig.savefig(out)
    plots.plt.close(fig)
    assert out.exists()
