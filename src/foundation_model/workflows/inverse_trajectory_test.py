# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure helpers in :mod:`foundation_model.workflows.inverse_trajectory`.

The full trajectory orchestrator needs a real trained checkpoint to exercise; this file covers
the pure functions — per-kind progress normalisation and the writer smoke-tests (static plot +
gif + html + svg). Animations are checked only for "file got written"; visual correctness is
verified by inspecting the rerun artefacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from foundation_model.workflows.inverse_trajectory import (
    TargetMeta,
    normalize_target_trajectories,
    plot_trajectory_animation,
    plot_trajectory_static,
    target_progress_matrix,
)


# --- normalize_target_trajectories -------------------------------------------------------------


def _linear_traj(baselines: np.ndarray, finals: np.ndarray, n_steps: int) -> np.ndarray:
    """(steps, B) linear interpolation from per-seed baselines to per-seed finals."""
    t = np.linspace(0.0, 1.0, n_steps)[:, None]
    return baselines[None, :] * (1 - t) + finals[None, :] * t


def test_value_kind_maps_baseline_to_zero_and_target_to_one():
    n_steps, target = 4, 2.0
    baselines = np.array([0.0, 0.5])
    traj = _linear_traj(baselines, np.full(2, target), n_steps)[:, :, None]  # (steps, B, 1)
    metas = [TargetMeta(task="k", kind="value", label="k→2", value=target)]
    progress = normalize_target_trajectories(traj, metas, {"k": baselines})
    assert progress["k→2"].shape == (n_steps,)
    assert progress["k→2"][0] == pytest.approx(0.0, abs=1e-9)
    assert progress["k→2"][-1] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(("class_high", "final_p"), [(True, 1.0), (False, 0.0)])
def test_class_kind_targets_one_or_zero(class_high: bool, final_p: float):
    n_steps = 5
    baselines = np.array([0.4, 0.6])
    traj = _linear_traj(baselines, np.full(2, final_p), n_steps)[:, :, None]
    metas = [TargetMeta(task="mat", kind="class", label="P", class_high=class_high)]
    progress = normalize_target_trajectories(traj, metas, {"mat": baselines})
    assert progress["P"][0] == pytest.approx(0.0, abs=1e-9)
    assert progress["P"][-1] == pytest.approx(1.0, abs=1e-9)


def test_curve_kind_full_rmse_reduction_is_one():
    n_steps = 3
    baselines = np.array([1.0, 2.0])  # baseline RMSE per seed
    traj = _linear_traj(baselines, np.zeros(2), n_steps)[:, :, None]  # RMSE → 0
    metas = [TargetMeta(task="dos", kind="curve", label="dos~curve")]
    progress = normalize_target_trajectories(traj, metas, {"dos": baselines})
    assert progress["dos~curve"][0] == pytest.approx(0.0, abs=1e-9)
    assert progress["dos~curve"][-1] == pytest.approx(1.0, abs=1e-9)


def test_direction_kind_self_normalises_to_final_step():
    n_steps = 6
    baselines = np.array([0.0, 1.0])
    finals = np.array([3.0, -1.0])  # one seed rises, one falls — both normalise 0 → 1
    traj = _linear_traj(baselines, finals, n_steps)[:, :, None]
    metas = [TargetMeta(task="b", kind="direction", label="b↑")]
    progress = normalize_target_trajectories(traj, metas, {"b": baselines})
    assert progress["b↑"][0] == pytest.approx(0.0, abs=1e-9)
    assert progress["b↑"][-1] == pytest.approx(1.0, abs=1e-9)


def test_flat_trajectory_guarded_against_zero_denominator():
    n_steps = 4
    baselines = np.array([0.3, 0.3])
    traj = np.tile(baselines[None, :], (n_steps, 1))[:, :, None]  # flat at baseline
    for kind, value in (("value", 0.3), ("direction", None)):
        metas = [TargetMeta(task="t", kind=kind, label="t", value=value)]
        progress = normalize_target_trajectories(traj, metas, {"t": baselines})
        assert np.allclose(progress["t"], 0.0)


def test_progress_matrix_keeps_per_seed_curves():
    n_steps = 4
    baselines = np.array([0.0, 1.0])
    traj = _linear_traj(baselines, np.array([2.0, 1.0]), n_steps)[:, :, None]  # seed 1 flat
    metas = [TargetMeta(task="a", kind="value", label="a→2", value=2.0)]
    matrix = target_progress_matrix(traj, metas, {"a": baselines})
    assert matrix["a→2"].shape == (n_steps, 2)
    assert matrix["a→2"][-1, 0] == pytest.approx(1.0, abs=1e-9)  # seed 0 reaches the target
    assert np.allclose(matrix["a→2"][:, 1], 0.0)  # seed 1 never moves


# --- plot writers ------------------------------------------------------------------------------


def _toy_progress() -> dict[str, np.ndarray]:
    """4-target × 30-step normalised progress with the new label styles, monotone curves."""
    n = 30
    return {
        "P(mat∈{1,3})↑": np.clip(np.linspace(0.0, 0.95, n) + 0.02 * np.sin(np.linspace(0, 4 * np.pi, n)), 0, 1.5),
        "formation_energy→-2": np.linspace(0.0, 1.2, n),
        "klat↑": np.linspace(0.0, 0.8, n),
        "dos~curve(3pts)": np.linspace(0.0, 0.6, n),
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
    plot_trajectory_static(_toy_progress(), out, title="toy trajectory", seed_composition="Au65 Ga20 Gd15")
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


def test_svg_title_and_legend_are_xml_escaped(tmp_path):
    """User-controlled text (title, target labels) must be XML-escaped in the SMIL SVG."""
    out = tmp_path / "escaped.svg"
    progress = {"fe<&>↓": np.linspace(0.0, 1.0, 30)}
    plot_trajectory_animation(
        progress,
        per_step_weights=_toy_weights(),
        element_symbols=[f"E{i}" for i in range(12)],
        out_paths_by_format={"svg": out},
        title="path <a> & <b>",
        max_frames=8,
    )
    body = out.read_text(encoding="utf-8")
    assert "&lt;a&gt; &amp; &lt;b&gt;" in body
    assert "<title>path <a>" not in body  # the raw, unescaped title must not appear
    assert "fe&lt;&amp;&gt;↓" in body  # legend label escaped too


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
