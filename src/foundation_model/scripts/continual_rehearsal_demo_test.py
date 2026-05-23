# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the configuration / pure helpers in :mod:`continual_rehearsal_demo`.

The runner's training loop is exercised end-to-end by smoke runs (it needs real parquet
data + a GPU/MPS device), so this file targets the *units that don't need either*:

* ``ContinualRehearsalConfig`` validation in ``__post_init__``.
* The element-system seed dedup / explicit-append logic.
* The ``_plot_kr_sequences`` regression (the function used to raise ``NameError`` when
  ``comps`` was empty — see the PR #18 code review).
* The material-type 5→3 class merge map shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foundation_model.scripts.continual_rehearsal_common import plot_kr_sequences
from foundation_model.scripts.continual_rehearsal_demo import (
    DEFAULT_SEQUENCE,
    MATERIAL_TYPE_CLASSES,
    MATERIAL_TYPE_DISPLAY_ORDER,
    QC_CLASSES,
    TASK_SPECS,
    ContinualRehearsalConfig,
    ContinualRehearsalRunner,
    _MATERIAL_TYPE_MERGE,
)


# --- ContinualRehearsalConfig ---------------------------------------------------------------


def _base_kwargs(**overrides):
    """Minimal valid config kwargs — only the fields without sane defaults need to be filled in.

    Paths are dummies; the validators inside ``__post_init__`` don't touch the filesystem.
    """
    defaults = {
        "qc_data_path": Path("/tmp/qc.parquet"),
        "qc_preprocessing_path": None,
        "superconductor_path": Path("/tmp/sc.parquet"),
        "magnetic_path": Path("/tmp/mag.parquet"),
        "phonix_path": Path("/tmp/ph.parquet"),
        "output_dir": Path("/tmp/out"),
        "task_sequence": list(DEFAULT_SEQUENCE),
    }
    defaults.update(overrides)
    return defaults


def test_config_default_post_init_accepts_default_sequence():
    cfg = ContinualRehearsalConfig(**_base_kwargs())
    assert cfg.task_sequence == list(DEFAULT_SEQUENCE)
    # Every task in the default sequence is registered in TASK_SPECS — would otherwise raise.
    assert set(cfg.task_sequence) <= set(TASK_SPECS)


def test_config_rejects_unknown_task():
    with pytest.raises(ValueError, match="Unknown task"):
        ContinualRehearsalConfig(**_base_kwargs(task_sequence=["density", "this_task_does_not_exist"]))


def test_config_rejects_bad_replay_ratio():
    with pytest.raises(ValueError, match="replay_ratio must be in"):
        ContinualRehearsalConfig(**_base_kwargs(replay_ratio=-0.1))
    with pytest.raises(ValueError, match="replay_ratio must be in"):
        ContinualRehearsalConfig(**_base_kwargs(replay_ratio=1.5))


def test_config_rejects_reg_task_target_length_mismatch():
    with pytest.raises(ValueError, match="inverse_reg_tasks and inverse_reg_targets"):
        ContinualRehearsalConfig(
            **_base_kwargs(inverse_reg_tasks=["formation_energy", "klat"], inverse_reg_targets=[-2.0])
        )


def test_config_rejects_unknown_seed_strategy():
    with pytest.raises(ValueError, match="inverse_seed_strategy must be"):
        ContinualRehearsalConfig(**_base_kwargs(inverse_seed_strategy="oracle"))


def test_config_explicit_strategy_requires_compositions():
    with pytest.raises(ValueError, match="requires inverse_seed_compositions"):
        ContinualRehearsalConfig(**_base_kwargs(inverse_seed_strategy="explicit", inverse_seed_compositions=[]))


# --- material-type 5→3 merge map ------------------------------------------------------------


def test_material_type_merge_covers_all_5_classes_and_3_targets():
    # Source labels are 0..4 (5 classes); merged labels are 0..2 (3 classes: AC / QC / others).
    assert set(_MATERIAL_TYPE_MERGE.keys()) == {0, 1, 2, 3, 4}
    assert set(_MATERIAL_TYPE_MERGE.values()) == {0, 1, 2}
    # QC label index must agree with QC_CLASSES.
    assert QC_CLASSES == [_MATERIAL_TYPE_MERGE[1]] == [_MATERIAL_TYPE_MERGE[3]]


def test_material_type_class_names_and_display_order_consistent():
    # 3 merged classes, both lists carry exactly those names.
    assert len(MATERIAL_TYPE_CLASSES) == 3
    assert sorted(MATERIAL_TYPE_CLASSES) == sorted(MATERIAL_TYPE_DISPLAY_ORDER)


# --- element-system dedup (classmethod, no runner state needed) ------------------------------


def test_dedupe_by_element_system_keeps_first_per_set():
    # First occurrence per element-set wins. Mg-Al-Cu appears twice; only the first survives.
    candidates = [
        "Mg12 Cu3 Ni3",  # {Mg, Cu, Ni}
        "Mg2 Cu1 Ni1",  # {Mg, Cu, Ni}  ← duplicate set, dropped
        "Y8.7 Mg34.6 Zn56.8",  # {Y, Mg, Zn}
        "Y1 Mg1 Zn1",  # {Y, Mg, Zn}   ← duplicate set, dropped
        "Au65 Ga20 Gd15",  # {Au, Ga, Gd}
    ]
    out = ContinualRehearsalRunner._dedupe_by_element_system(candidates, n=10)
    assert out == ["Mg12 Cu3 Ni3", "Y8.7 Mg34.6 Zn56.8", "Au65 Ga20 Gd15"]


def test_dedupe_by_element_system_respects_n_cap():
    candidates = [
        "Mg1",  # {Mg}
        "Al1",  # {Al}
        "Cu1",  # {Cu}
        "Ni1",  # {Ni}
    ]
    out = ContinualRehearsalRunner._dedupe_by_element_system(candidates, n=2)
    assert out == ["Mg1", "Al1"]


def test_dedupe_by_element_system_ignores_empty_strings():
    out = ContinualRehearsalRunner._dedupe_by_element_system(["", "Mg1", "  ", "Al1"], n=5)
    assert out == ["Mg1", "Al1"]


def test_element_system_extracts_symbols_ignoring_amounts():
    # Static-method shape: returns a frozenset of element symbols, no stoichiometry leaks through.
    es = ContinualRehearsalRunner._element_system("Au65 Ga20 Gd15")
    assert es == frozenset({"Au", "Ga", "Gd"})
    # Multi-digit / float amounts handled the same way.
    es = ContinualRehearsalRunner._element_system("Mg36.3 Al32 Zn31.7")
    assert es == frozenset({"Mg", "Al", "Zn"})


# --- plot_kr_sequences empty-comps regression (P1 bug from PR #18 code review) -------------
# The function is now in ``continual_rehearsal_common`` (PR #18 refactor); pre-refactor it lived
# as a bound method on each runner and the empty-comps NameError silently shipped on the demo
# side for several PRs. These tests pin the post-refactor behaviour from both call sites.


def test_plot_kr_sequences_handles_empty_comps_without_crashing(tmp_path):
    """Empty ``comps`` used to raise ``NameError: line_true`` from ``fig.legend(...)``. Now it
    logs a warning and returns early; no file is written."""
    out_dir = tmp_path / "step01_density"
    out_dir.mkdir()
    plot_kr_sequences(
        comps=[],
        t_list=[],
        true_parts=[],
        pred=np.array([]),
        task_name="dos_density",
        step_dir=out_dir,
        title="DOS density",
    )
    assert not (out_dir / "dos_density_sequences.png").exists()


def test_plot_kr_sequences_renders_when_comps_nonempty(tmp_path):
    """Smoke: one composition's sequence renders a PNG with no errors."""
    import torch

    out_dir = tmp_path / "step01_density"
    out_dir.mkdir()
    t = torch.linspace(0.0, 1.0, 8)
    true_part = np.linspace(0.0, 1.0, 8)
    pred = np.linspace(0.05, 0.95, 8)
    plot_kr_sequences(
        comps=["Mg1 Cu1"],
        t_list=[t],
        true_parts=[true_part],
        pred=pred,
        task_name="dos_density",
        step_dir=out_dir,
        title="DOS density",
    )
    assert (out_dir / "dos_density_sequences.png").exists()
