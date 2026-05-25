# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared dump / plot helpers in :mod:`continual_rehearsal_common`.

The runners are end-to-end-tested via smoke runs; these tests pin the pure-function behaviour
of the helpers they share, including the edge cases that motivated factoring them out.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from foundation_model.scripts.continual_rehearsal_common import (
    MATERIAL_TYPE_CLASSES,
    MATERIAL_TYPE_DISPLAY_ORDER,
    SCATTER_COLOR,
    dump_kr_predictions,
    dump_metrics,
    dump_predictions,
    plot_confusion,
    plot_kr_sequences,
    plot_parity,
)


# --- shared constants ---


def test_scatter_colour_is_the_project_blue():
    """The blue must match the project palette; ``paper_inverse_comparison`` and slide deck
    reference this exact hex. Changing it without coordinating breaks the slide colour story."""
    assert SCATTER_COLOR == "#2563EB"


def test_material_type_canonical_and_display_orders_are_consistent_three_classes():
    assert sorted(MATERIAL_TYPE_CLASSES) == sorted(MATERIAL_TYPE_DISPLAY_ORDER)
    assert len(MATERIAL_TYPE_CLASSES) == 3


# --- dumpers ---


def test_dump_predictions_writes_parquet_with_expected_columns(tmp_path):
    step_dir = tmp_path / "step01_density"
    step_dir.mkdir()
    dump_predictions(
        "density",
        step_dir,
        comps=["Mg1 Cu1", "Al1 Fe1"],
        true=np.array([0.1, 0.2]),
        pred=np.array([0.15, 0.18]),
    )
    out = step_dir / "density_pred.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert list(df.columns) == ["composition", "true", "pred"]
    assert len(df) == 2
    assert df["composition"].tolist() == ["Mg1 Cu1", "Al1 Fe1"]


def test_dump_kr_predictions_long_form_round_trips(tmp_path):
    """KR predictions are stored long-form (one row per (composition, t)); the flat ``pred``
    array is correctly re-split using ``true_parts`` lengths."""
    step_dir = tmp_path / "step08_dos_density"
    step_dir.mkdir()
    comps = ["Mg1 Cu1", "Al1 Fe1"]
    t_list = [np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])]  # different lengths per comp
    true_parts = [np.array([10.0, 11.0]), np.array([20.0, 21.0, 22.0])]
    pred = np.array([10.5, 11.5, 20.5, 21.5, 22.5])  # flat across both comps

    dump_kr_predictions("dos_density", step_dir, comps=comps, t_list=t_list, true_parts=true_parts, pred=pred)
    out = step_dir / "dos_density_pred.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    # 2 + 3 = 5 rows total
    assert len(df) == 5
    # Per-composition slices recover the right pred values from the long-form table.
    mg = df[df["composition"] == "Mg1 Cu1"].sort_values("t")
    assert mg["pred"].tolist() == [10.5, 11.5]
    al = df[df["composition"] == "Al1 Fe1"].sort_values("t")
    assert al["pred"].tolist() == [20.5, 21.5, 22.5]


def test_dump_metrics_writes_indented_json(tmp_path):
    step_dir = tmp_path / "step01_density"
    step_dir.mkdir()
    dump_metrics("density", step_dir, {"r2": 0.95, "mae": 0.1, "samples": 100, "primary": 0.95})
    out = step_dir / "density_metrics.json"
    assert out.exists()
    body = json.loads(out.read_text())
    assert body == {"r2": 0.95, "mae": 0.1, "samples": 100, "primary": 0.95}


# --- plots ---


def test_plot_parity_writes_png(tmp_path):
    step_dir = tmp_path / "step01_density"
    step_dir.mkdir()
    true = np.linspace(0.0, 1.0, 50)
    pred = true + np.random.default_rng(0).normal(0, 0.05, 50)
    plot_parity(true, pred, "density", r2=0.95, step_dir=step_dir, title="Density (normalized)")
    assert (step_dir / "density_parity.png").exists()


def test_plot_confusion_writes_png_for_generic_and_material_type(tmp_path):
    step_dir = tmp_path / "step11_material_type"
    step_dir.mkdir()
    rng = np.random.default_rng(0)
    true = rng.integers(0, 3, size=100)
    pred = rng.integers(0, 3, size=100)

    plot_confusion(
        true,
        pred,
        "material_type",
        acc=0.5,
        step_dir=step_dir,
        num_classes=3,
        title="Material type",
        special_material_type=True,
    )
    assert (step_dir / "material_type_confusion.png").exists()

    plot_confusion(
        true,
        pred,
        "another_clf",
        acc=0.5,
        step_dir=step_dir,
        num_classes=3,
        title="Another classifier",
        special_material_type=False,
    )
    assert (step_dir / "another_clf_confusion.png").exists()


def test_plot_kr_sequences_returns_silently_on_empty_comps(tmp_path):
    """The PR #18 regression that motivated the refactor: empty ``comps`` used to crash with
    ``NameError`` inside ``fig.legend``. Now it returns early without writing anything."""
    step_dir = tmp_path / "step08_dos_density"
    step_dir.mkdir()
    plot_kr_sequences(
        comps=[],
        t_list=[],
        true_parts=[],
        pred=np.array([]),
        task_name="dos_density",
        step_dir=step_dir,
        title="DOS density",
    )
    assert not (step_dir / "dos_density_sequences.png").exists()


def test_plot_kr_sequences_renders_panels_when_data_present(tmp_path):
    """Single-composition smoke: a sequence panel is rendered without raising."""
    import torch

    step_dir = tmp_path / "step08_dos_density"
    step_dir.mkdir()
    t = torch.linspace(0.0, 1.0, 8)
    true_part = np.linspace(0.0, 1.0, 8)
    pred = np.linspace(0.05, 0.95, 8)
    plot_kr_sequences(
        comps=["Mg1 Cu1"],
        t_list=[t],
        true_parts=[true_part],
        pred=pred,
        task_name="dos_density",
        step_dir=step_dir,
        title="DOS density",
    )
    assert (step_dir / "dos_density_sequences.png").exists()
