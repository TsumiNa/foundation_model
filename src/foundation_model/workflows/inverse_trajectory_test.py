# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure helpers in :mod:`foundation_model.workflows.inverse_trajectory`.

The full ``_emit_trajectory_outputs`` orchestrator needs a real trained checkpoint to exercise;
this file covers the pure functions — seed picker, progress normalisation, and the writer
smoke-tests (static plot + gif + html + svg). Animations are checked only for "file got written";
visual correctness is verified by inspecting the rerun artefacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from foundation_model.workflows.inverse_trajectory import (
    best_seed_by_target_distance,
    normalize_target_trajectories,
    plot_trajectory_animation,
    plot_trajectory_static,
)


# --- best_seed_by_target_distance --------------------------------------------------------------


def test_best_seed_picks_closest_joint_distance_to_targets():
    """Seed 1 is closest to the joint target (QC=1, fe=-2, klat=2); the picker should return 1."""
    qc = np.array([0.20, 0.95, 0.50])  # seed 1 has highest QC
    reg = {
        "formation_energy": np.array([+0.5, -1.9, -1.0]),  # seed 1 hits target -2 best
        "klat": np.array([0.0, 1.8, 1.2]),  # seed 1 hits target 2 best
    }
    reg_targets = {"formation_energy": -2.0, "klat": 2.0}
    assert best_seed_by_target_distance(qc, reg, reg_targets) == 1


def test_best_seed_handles_zero_target_without_div_by_zero():
    """``target == 0`` would naively divide by zero; the picker uses a min-scale guard."""
    qc = np.array([0.9, 0.8])
    reg = {"some_task": np.array([0.1, 0.5])}
    # Should pick seed 0 (closer to target 0).
    assert best_seed_by_target_distance(qc, reg, {"some_task": 0.0}) == 0


def test_best_seed_empty_qc_raises():
    with pytest.raises(ValueError, match="empty qc_final"):
        best_seed_by_target_distance(np.array([]), {}, {})


# --- normalize_target_trajectories -------------------------------------------------------------


def test_normalize_trajectory_maps_baseline_to_zero_and_target_to_one():
    """Per (task, seed): a step's value of (target - baseline) + baseline ⇒ progress = 1."""
    n_steps = 4
    n_seeds = 2
    # One reg target only. Baseline = [0.0, 0.5], target = 2.0.
    reg_targets = {"k": 2.0}
    seed_reg = {"k": np.array([0.0, 0.5])}
    # Per-seed trajectory: linear interpolation from baseline → target across 4 steps.
    traj_k = np.stack(
        [
            np.linspace(0.0, 2.0, n_steps),  # seed 0
            np.linspace(0.5, 2.0, n_steps),  # seed 1
        ],
        axis=1,
    )  # (steps, B)
    # QC trajectory: flat at the seed baseline so it normalises to 0 progress throughout.
    seed_qc = np.array([0.1, 0.2])
    qc_traj = np.tile(seed_qc[None, :], (n_steps, 1))

    progress = normalize_target_trajectories(
        qc_trajectory=qc_traj,
        reg_trajectory={"k": traj_k},
        reg_targets=reg_targets,
        seed_qc=seed_qc,
        seed_reg=seed_reg,
    )
    # k progress: starts at 0, ends at 1 (per-seed normalised then mean over B).
    assert progress["k"].shape == (n_steps,)
    assert progress["k"][0] == pytest.approx(0.0, abs=1e-9)
    assert progress["k"][-1] == pytest.approx(1.0, abs=1e-9)
    # QC stays at baseline ⇒ progress = 0 throughout.
    assert progress["QC"].shape == (n_steps,)
    assert np.allclose(progress["QC"], 0.0)


# --- plot writers ------------------------------------------------------------------------------


def _toy_progress() -> dict[str, np.ndarray]:
    """4-target × 30-step normalised progress, monotone so the picture is interpretable."""
    n = 30
    return {
        "QC": np.clip(np.linspace(0.0, 0.95, n) + 0.02 * np.sin(np.linspace(0, 4 * np.pi, n)), 0, 1.5),
        "formation_energy": np.linspace(0.0, 1.2, n),
        "klat": np.linspace(0.0, 0.8, n),
    }


def _toy_weights(n_steps: int = 30, n_components: int = 12) -> np.ndarray:
    """(steps, n_components) toy weights — start sparse, drift toward a different sparse set."""
    rng = np.random.default_rng(7)
    w = np.zeros((n_steps, n_components), dtype=float)
    # Initial: mass on elements 0..2
    w[0, :3] = [0.5, 0.3, 0.2]
    # Final: mass on elements 4, 6, 7
    end = np.zeros(n_components)
    end[4], end[6], end[7] = 0.5, 0.3, 0.2
    for s in range(n_steps):
        t = s / (n_steps - 1)
        w[s] = (1 - t) * w[0] + t * end + 0.001 * rng.standard_normal(n_components)
        w[s] = np.clip(w[s], 0, None)
        w[s] /= w[s].sum()
    return w


def test_plot_trajectory_static_writes_png(tmp_path):
    out = tmp_path / "static.png"
    plot_trajectory_static(_toy_progress(), out, title="toy trajectory")
    assert out.exists()


def test_plot_trajectory_static_with_seed_composition(tmp_path):
    """``seed_composition`` is rendered as a monospace annotation under the title — verify the
    plot still writes with the kwarg present (visual correctness is by inspection)."""
    out = tmp_path / "static_with_seed.png"
    plot_trajectory_static(
        _toy_progress(), out, title="toy trajectory", seed_composition="Au65 Ga20 Gd15"
    )
    assert out.exists()


def test_plot_trajectory_animation_writes_gif(tmp_path):
    out = tmp_path / "anim.gif"
    plot_trajectory_animation(
        _toy_progress(),
        per_step_weights=_toy_weights(),
        element_symbols=[f"E{i}" for i in range(12)],
        out_paths_by_format={"gif": out},
        title="toy animation",
        max_frames=10,  # keep test fast
    )
    assert out.exists()


def test_plot_trajectory_animation_writes_html(tmp_path):
    out = tmp_path / "anim.html"
    plot_trajectory_animation(
        _toy_progress(),
        per_step_weights=_toy_weights(),
        element_symbols=[f"E{i}" for i in range(12)],
        out_paths_by_format={"html": out},
        title="toy animation",
        max_frames=10,
    )
    assert out.exists()


def test_plot_trajectory_animation_writes_smil_svg(tmp_path):
    out = tmp_path / "anim.svg"
    plot_trajectory_animation(
        _toy_progress(),
        per_step_weights=_toy_weights(),
        element_symbols=[f"E{i}" for i in range(12)],
        out_paths_by_format={"svg": out},
        title="toy animation",
        max_frames=8,
    )
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    # The SMIL animation should contain <animate> tags driving the marker x1/x2 + bar widths.
    assert "<animate" in body
    # The step counter labels every step exactly once.
    assert "step 1/" in body


def test_plot_trajectory_animation_skips_when_steps_dont_match(tmp_path):
    """Mismatched per_step_weights vs progress step count ⇒ warning + no file."""
    out = tmp_path / "should_not_exist.gif"
    progress = _toy_progress()  # 30 steps
    plot_trajectory_animation(
        progress,
        per_step_weights=_toy_weights(n_steps=15),  # mismatch
        element_symbols=[f"E{i}" for i in range(12)],
        out_paths_by_format={"gif": out},
        title="should skip",
        max_frames=10,
    )
    assert not out.exists()
