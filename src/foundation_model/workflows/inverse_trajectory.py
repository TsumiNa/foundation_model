# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Per-step trajectory analytics + plots + animations for inverse-design runs.

Each call to :meth:`FlexibleMultiTaskModel.optimize_latent` /
:meth:`FlexibleMultiTaskModel.optimize_composition` can optionally record:

* ``trajectory_targets``  — shape ``(steps, B, T)``: per-step target channels, one column per
  scenario target in declaration order (regression: ŷ, curve: RMSE-to-curve, classification:
  P(classes)).
* ``trajectory_weights``  — shape ``(steps, B, n_components)``: per-step element weights
  (the optimisation variable for ``optimize_composition``; decoded via ``KMD.inverse`` from the
  per-step AE-decoded ``x`` for ``optimize_latent``).

These are enough to visualise:

1. How fast each target converges relative to the others (static line plot, normalised so all
   targets are on the same y-axis).
2. How the recipe evolves across the optimisation (animated bar chart of the per-step composition
   on the side, frame per step).

This module hosts the pure helpers; :func:`foundation_model.workflows.inverse.run` drives them
(via ``_emit_trajectory``), and they are unit-tested directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape as _xml_escape

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


# --- trajectory normalisation -----------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TargetMeta:
    """The slice of a scenario target this module needs (kept tiny to avoid importing configs).

    ``kind`` is one of ``"value"`` / ``"direction"`` / ``"curve"`` / ``"class"``; ``value`` is the
    regression target for the value kind; ``class_high`` is the class-kind direction; ``label`` is
    the human-readable curve label used as the progress-dict key.
    """

    task: str
    kind: str
    label: str
    value: float | None = None
    class_high: bool = True


def target_progress_matrix(
    trajectory: np.ndarray,
    metas: Sequence[TargetMeta],
    seed_channels: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-seed progress curves for every target: ``{label: (steps, B)}``.

    0 = "at seed baseline", 1 = "goal reached" — per kind:

    - ``value``: ``(y − baseline) / (target − baseline)`` (can overshoot past 1).
    - ``class``: same formula with target 1.0 (``class_high``) or 0.0 (low).
    - ``curve``: ``(baseline_rmse − rmse) / baseline_rmse`` (1 = perfect curve fit).
    - ``direction``: unbounded objective, no fixed goal — self-normalised over the run:
      ``(y − baseline) / (y_final − baseline)`` (0 at the seed, 1 at the final step).
    """
    steps = np.asarray(trajectory, dtype=float)  # (steps, B, T)
    out: dict[str, np.ndarray] = {}
    for j, meta in enumerate(metas):
        traj = steps[:, :, j]  # (steps, B)
        baseline = np.asarray(seed_channels[meta.task], dtype=float)  # (B,)
        if meta.kind == "curve":
            denom = np.where(np.abs(baseline) < 1e-9, 1.0, baseline)
            progress = (baseline[None, :] - traj) / denom[None, :]
        elif meta.kind == "direction":
            denom = traj[-1, :] - baseline
            denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
            progress = (traj - baseline[None, :]) / denom[None, :]
        else:
            if meta.kind == "value":
                if meta.value is None:
                    raise ValueError(f"TargetMeta '{meta.task}': value kind requires a value.")
                target = float(meta.value)
            else:
                target = 1.0 if meta.class_high else 0.0
            denom = target - baseline
            denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
            progress = (traj - baseline[None, :]) / denom[None, :]
        out[meta.label] = progress
    return out


def normalize_target_trajectories(
    trajectory: np.ndarray,
    metas: Sequence[TargetMeta],
    seed_channels: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Mean-over-seeds progress curves: ``{label: (steps,)}``.

    The transform is per-(target, seed) — see :func:`target_progress_matrix` — then averaged over
    the seed cohort so a noisy seed-to-seed baseline doesn't dilute the picture.
    """
    return {label: mat.mean(axis=1) for label, mat in target_progress_matrix(trajectory, metas, seed_channels).items()}


# --- static plot -------------------------------------------------------------------------------


_TARGET_COLORS = ["#2563EB", "#55A868", "#E67E22", "#9467bd", "#C44E52"]  # blue/green/orange/purple/red


def plot_trajectory_static(
    progress: Mapping[str, np.ndarray],
    out_path: Path,
    *,
    title: str,
    seed_composition: str | None = None,
) -> None:
    """Line plot of normalised progress vs step.

    The targets cycle through the project's blue / green / orange palette. The y-axis is
    "progress fraction" (0 = at seed, 1 = at target); a horizontal dashed line at 1.0 marks the
    joint target. The reader gets a one-glance answer to: "do the targets converge together, or
    does the recipe stabilise early and the targets keep moving?" — divergence between the lines
    surfaces immediately.

    When ``seed_composition`` is provided (the per-seed composition string, e.g.
    ``"Au65 Ga20 Gd15"``), it's appended to the figure title under the main title in a monospace
    font — the reader can identify the seed by chemistry rather than by index.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    steps = np.arange(len(next(iter(progress.values()))))

    for i, key in enumerate(progress):
        ax.plot(
            steps,
            progress[key],
            color=_TARGET_COLORS[i % len(_TARGET_COLORS)],
            lw=1.8,
            label=key,
        )

    ax.axhline(1.0, color="#666", ls="--", lw=1.0, alpha=0.7, label="target (progress = 1.0)")
    ax.axhline(0.0, color="#bbb", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("Optimisation step")
    ax.set_ylabel("Progress  (0 = seed, 1 = target)")
    if seed_composition:
        # Two-line layout: bold main title on top + seed composition underneath, with extra
        # ``pad`` so the title doesn't sit flush against the upper axes line. Putting the
        # seed-comp as a text annotation at y=1.02 collided with the title when matplotlib's
        # default title-pad was applied — fix is to render both lines via set_title and a
        # second matching text() at a clearly-distinct y position.
        ax.set_title(title, fontsize=12, fontweight="bold", pad=22)
        ax.text(
            0.5,
            1.005,
            f"seed:  {seed_composition}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            family="monospace",
            color="#444",
        )
    else:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Wrote trajectory static plot to {out_path}")


# --- animation ---------------------------------------------------------------------------------


def _topk_composition_frame(
    weights: np.ndarray, element_symbols: list[str], top_k: int = 10
) -> list[tuple[str, float]]:
    """Top-K elements by weight, sorted descending. Used as one frame of the animation's comp panel."""
    idx = np.argsort(weights)[::-1][:top_k]
    return [(element_symbols[int(i)], float(weights[int(i)])) for i in idx if weights[int(i)] > 1e-4]


def plot_trajectory_animation(
    progress: Mapping[str, np.ndarray],
    per_step_weights: np.ndarray,
    element_symbols: list[str],
    out_paths_by_format: Mapping[str, Path],
    *,
    title: str,
    seed_composition: str | None = None,
    top_k_elements: int = 10,
    fps: int = 15,
    max_frames: int = 120,
) -> None:
    """Targets-vs-step line plot (top panel) + per-step top-K element bar chart (right panel).

    The line plot draws the full curve from step 0; a vertical "current step" marker advances
    one tick per frame. The bar chart on the right re-draws each frame to show the current
    composition's top-K elements (so the viewer can see "what is the recipe right now?" as the
    targets evolve). For long runs (steps > ``max_frames``) we subsample uniformly so the GIF
    stays under a few seconds at fps=15.

    Writers:
      - ``gif``  → ``PillowWriter`` (no external deps; embeddable anywhere).
      - ``html`` → ``HTMLWriter`` (JS-controlled play/pause/scrub; great for inspection).
      - ``svg``  → custom SMIL-animated single-file SVG (browsers play it; PPT cannot embed).
    """
    n_steps = len(next(iter(progress.values())))
    if n_steps == 0:
        logger.warning("plot_trajectory_animation: empty progress arrays — skipping.")
        return
    if per_step_weights.shape[0] != n_steps:
        logger.warning(
            f"plot_trajectory_animation: per_step_weights step count ({per_step_weights.shape[0]}) "
            f"does not match progress step count ({n_steps}); skipping animation."
        )
        return

    # Uniform subsample down to ``max_frames`` so GIFs stay manageable. The line plot still uses
    # the full curve; only the marker / weights frames are subsampled.
    frame_steps = np.linspace(0, n_steps - 1, num=min(n_steps, max_frames)).astype(int)
    frame_steps = np.unique(frame_steps)  # in case of duplicate indices for very small n_steps

    fig = plt.figure(figsize=(12.0, 5.5), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.30)
    ax_line = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    # --- Static line plot in left panel ---
    steps = np.arange(n_steps)
    for i, key in enumerate(progress):
        ax_line.plot(
            steps,
            progress[key],
            color=_TARGET_COLORS[i % len(_TARGET_COLORS)],
            lw=1.8,
            label=key,
        )
    ax_line.axhline(1.0, color="#666", ls="--", lw=1.0, alpha=0.6)
    ax_line.axhline(0.0, color="#bbb", ls=":", lw=0.8, alpha=0.5)
    ax_line.set_xlabel("Optimisation step")
    ax_line.set_ylabel("Progress  (0 = seed, 1 = target)")
    if seed_composition:
        # Two-line title: bold panel title on top + monospace seed-composition underneath. The
        # ``pad=22`` lifts the title clear of the second line; without the pad they overlap
        # because matplotlib's default title baseline sits where the text annotation lands.
        ax_line.set_title(title, fontsize=11, fontweight="bold", pad=22)
        ax_line.text(
            0.5,
            1.005,
            f"seed:  {seed_composition}",
            transform=ax_line.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            family="monospace",
            color="#444",
        )
    else:
        ax_line.set_title(title, fontsize=11, fontweight="bold")
    ax_line.legend(loc="best", fontsize=8, frameon=False)
    ax_line.grid(True, alpha=0.2)
    marker = ax_line.axvline(0, color="#444", lw=1.2, alpha=0.85)

    # --- Bar chart in right panel (redrawn per frame) ---
    ax_bar.set_title("Composition  (top-K by weight)", fontsize=10)
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("weight")

    def _draw_bar(step_idx: int) -> None:
        ax_bar.clear()
        frame = _topk_composition_frame(per_step_weights[step_idx], element_symbols, top_k=top_k_elements)
        if not frame:
            ax_bar.text(0.5, 0.5, "(no elements above threshold)", ha="center", va="center", transform=ax_bar.transAxes)
        else:
            symbols, weights = zip(*frame)
            y_pos = np.arange(len(symbols))
            ax_bar.barh(y_pos, weights, color="#2563EB", alpha=0.75, edgecolor="#222", linewidth=0.5)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(symbols, fontsize=9)
            ax_bar.invert_yaxis()  # largest on top
        ax_bar.set_xlim(0, max(0.5, float(per_step_weights[step_idx].max()) * 1.1))
        ax_bar.set_xlabel("weight")
        ax_bar.set_title(f"Composition  (step {step_idx + 1}/{n_steps})", fontsize=10)
        ax_bar.grid(True, axis="x", alpha=0.2)

    def _init() -> Iterable[Any]:
        _draw_bar(int(frame_steps[0]))
        marker.set_xdata([int(frame_steps[0])])
        return (marker,)

    def _update(frame_idx: int) -> Iterable[Any]:
        step_idx = int(frame_steps[frame_idx])
        _draw_bar(step_idx)
        marker.set_xdata([step_idx])
        return (marker,)

    # Only build the matplotlib FuncAnimation when at least one matplotlib-native format
    # (gif / html) is requested. For svg-only output we render a handwritten SMIL SVG without
    # touching the animation object — building it anyway would emit a "Animation was deleted
    # without rendering anything" UserWarning on test runs.
    needs_mpl_anim = any(fmt in ("gif", "html") for fmt in out_paths_by_format)
    anim = (
        manimation.FuncAnimation(
            fig,
            _update,
            frames=len(frame_steps),
            init_func=_init,
            interval=1000 // fps,
            blit=False,  # the bar chart redraw isn't blittable cleanly
        )
        if needs_mpl_anim
        else None
    )

    for fmt, out_path in out_paths_by_format.items():
        try:
            if fmt == "gif":
                assert anim is not None  # gif ⇒ needs_mpl_anim ⇒ anim built
                anim.save(str(out_path), writer=manimation.PillowWriter(fps=fps))
            elif fmt == "html":
                assert anim is not None  # html ⇒ needs_mpl_anim ⇒ anim built
                # ``to_jshtml`` returns a single self-contained HTML string with frames embedded
                # as base64 PNGs. The ``HTMLWriter`` alternative drops a separate ``*_frames/``
                # folder of 120+ PNGs alongside, which clutters the output dir and makes the
                # artefact non-portable. The base64 version is bigger per-file (~3 MB vs the
                # multi-file's ~10 MB total) but is one self-contained file.
                out_path.write_text(anim.to_jshtml(fps=fps), encoding="utf-8")
            elif fmt == "svg":
                _save_smil_svg(progress, per_step_weights, element_symbols, frame_steps, out_path, title=title, fps=fps)
            else:
                logger.warning(f"plot_trajectory_animation: unknown format {fmt!r} — skipping.")
                continue
            logger.info(f"Wrote trajectory animation ({fmt}) to {out_path}")
        except Exception as exc:  # pragma: no cover  (writer-specific failure modes)
            logger.warning(f"plot_trajectory_animation: failed to write {fmt} → {out_path}: {exc}")

    plt.close(fig)


# --- SMIL SVG writer ---------------------------------------------------------------------------


def _save_smil_svg(
    progress: Mapping[str, np.ndarray],
    per_step_weights: np.ndarray,
    element_symbols: list[str],
    frame_steps: np.ndarray,
    out_path: Path,
    *,
    title: str,
    fps: int,
    top_k_elements: int = 10,
) -> None:
    """Single-file SMIL-animated SVG.

    matplotlib doesn't have a native SVG-animation writer; rather than render N PNGs and ship a
    multi-frame SVG (would defeat the "one file" goal), we emit a compact handwritten SVG with
    the static line plot as a vector overlay + ``<animate>`` tags for the per-step marker and
    per-element bar widths. Plays in any modern browser (Firefox / Chrome / Safari); PowerPoint
    and Keynote cannot embed it directly — for those use the GIF.
    """
    n_steps = len(next(iter(progress.values())))
    duration_s = max(1.0, len(frame_steps) / fps)
    # Coordinate system: 800 × 400 viewBox, line plot in [40, 480] × [40, 360], bar plot in
    # [520, 780] × [40, 360]. Bars are horizontal, top-K elements, redrawn via <animate>.

    # ---- header ----
    # Escape any user-controlled text (the run's path name) before embedding it in XML so a
    # title containing &/</> can't produce invalid SVG or an injection vector.
    safe_title = _xml_escape(title)
    parts: list[str] = []
    parts.append(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 420" '
        'width="800" height="420" font-family="system-ui, sans-serif" font-size="11">'
    )
    parts.append(f"<title>{safe_title}</title>")
    parts.append(f'<text x="400" y="18" text-anchor="middle" font-size="13" font-weight="bold">{safe_title}</text>')

    # ---- line plot (static) ----
    parts.append('<rect x="40" y="40" width="440" height="320" fill="white" stroke="#888" />')
    parts.append('<text x="260" y="395" text-anchor="middle">Optimisation step</text>')
    parts.append(
        '<text x="20" y="200" text-anchor="middle" transform="rotate(-90 20 200)">'
        "Progress  (0 = seed, 1 = target)</text>"
    )

    # Compute y-range across all curves so 0 and 1 are at fixed pixels.
    all_vals = np.concatenate([np.asarray(v) for v in progress.values()])
    y_min, y_max = float(min(all_vals.min(), 0.0)), float(max(all_vals.max(), 1.0))
    y_pad = (y_max - y_min) * 0.05
    y_min -= y_pad
    y_max += y_pad

    def _to_x(step_idx: int) -> float:
        return 40 + (step_idx / max(n_steps - 1, 1)) * 440

    def _to_y(val: float) -> float:
        return 360 - (val - y_min) / (y_max - y_min) * 320

    # Static gridlines + axis labels at 0 / 1.
    y0, y1 = _to_y(0.0), _to_y(1.0)
    parts.append(f'<line x1="40" x2="480" y1="{y0}" y2="{y0}" stroke="#bbb" stroke-dasharray="2,3" />')
    parts.append(f'<line x1="40" x2="480" y1="{y1}" y2="{y1}" stroke="#666" stroke-dasharray="4,3" />')
    parts.append(f'<text x="34" y="{y0 + 4}" text-anchor="end" fill="#888">0</text>')
    parts.append(f'<text x="34" y="{y1 + 4}" text-anchor="end" fill="#666">1</text>')

    color_map = {key: _TARGET_COLORS[i % len(_TARGET_COLORS)] for i, key in enumerate(progress)}

    # Polyline per target. Legend labels derive from user task names — escape them for XML.
    legend_y = 50
    for key, vals in progress.items():
        pts = " ".join(f"{_to_x(s):.1f},{_to_y(float(v)):.1f}" for s, v in enumerate(vals))
        color = color_map[key]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" />')
        parts.append(f'<circle cx="455" cy="{legend_y}" r="4" fill="{color}" />')
        parts.append(f'<text x="462" y="{legend_y + 4}" fill="#222">{_xml_escape(key)}</text>')
        legend_y += 14

    # ---- animated step marker (vertical line in line plot) ----
    x_values_str = ";".join(f"{_to_x(int(s)):.1f}" for s in frame_steps)
    parts.append(
        f'<line y1="40" y2="360" stroke="#444" stroke-width="1.2" opacity="0.85">'
        f'  <animate attributeName="x1" values="{x_values_str}" dur="{duration_s:.2f}s" repeatCount="indefinite" />'
        f'  <animate attributeName="x2" values="{x_values_str}" dur="{duration_s:.2f}s" repeatCount="indefinite" />'
        f"</line>"
    )

    # ---- bar chart (top-K, animated per element) ----
    parts.append('<rect x="520" y="40" width="260" height="320" fill="white" stroke="#888" />')
    parts.append(
        '<text x="650" y="32" text-anchor="middle" font-weight="bold">Composition (top-K, step animated)</text>'
    )

    # Use the union of top-K-per-frame elements across all frames so each bar is one stable row.
    seen_idx: list[int] = []
    for s in frame_steps:
        top = np.argsort(per_step_weights[int(s)])[::-1][:top_k_elements]
        for idx in top:
            if int(idx) not in seen_idx:
                seen_idx.append(int(idx))
    # Cap at 2× top_k to keep the SVG tidy.
    seen_idx = seen_idx[: 2 * top_k_elements]
    n_rows = len(seen_idx)
    bar_y_top = 50
    bar_height = min(20.0, 300.0 / max(n_rows, 1))
    bar_x_left = 560
    bar_max_w = 200

    # Per-bar animation values (weight per frame, scaled).
    for row_i, elem_idx in enumerate(seen_idx):
        y_row = bar_y_top + row_i * bar_height
        widths = [per_step_weights[int(s), elem_idx] for s in frame_steps]
        w_str = ";".join(f"{max(0.0, float(w)) * bar_max_w:.1f}" for w in widths)
        parts.append(
            f'<text x="{bar_x_left - 4}" y="{y_row + bar_height - 4:.1f}" text-anchor="end">'
            f"{_xml_escape(element_symbols[elem_idx])}</text>"
        )
        parts.append(
            f'<rect x="{bar_x_left}" y="{y_row:.1f}" height="{bar_height - 2:.1f}" fill="#2563EB" opacity="0.75">'
            f'  <animate attributeName="width" values="{w_str}" dur="{duration_s:.2f}s" repeatCount="indefinite" />'
            f"</rect>"
        )

    # Step counter at the bottom.
    step_label_values = ";".join(f"step {int(s) + 1}/{n_steps}" for s in frame_steps)
    parts.append(
        f'<text x="650" y="395" text-anchor="middle" fill="#444">'
        f"  step 1/{n_steps}"
        f'  <animate attributeName="textContent" values="{step_label_values}" dur="{duration_s:.2f}s" repeatCount="indefinite" />'
        f"</text>"
    )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
