# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Figures and markdown for a finished scenario.

The leaf on the output side: everything else feeds it and it calls nothing back. Comparison plots
across paths, objective-versus-target scatters, the seed→optimised composition map with its
hand-laid-out rows, element-frequency bars, and the per-scenario and root markdown summaries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea  # noqa: E402

from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS, formula_to_composition  # noqa: E402

from ..plots import DISCOVERED_ELEMENT_COLOR, SCATTER_COLOR  # noqa: E402
from ..recording import RunRecorder  # noqa: E402
from .config import InverseConfig, ScenarioConfig, target_label, TargetKind, TargetSpec  # noqa: E402
from .seeds import element_system  # noqa: E402


def _method_color(method: str) -> str:
    return "#55A868" if method == "latent" else SCATTER_COLOR


def _target_ref_line(spec: TargetSpec) -> float | None:
    """The dashed goal line for a target's channel panel (None = no fixed goal)."""
    if spec.kind is TargetKind.VALUE:
        return float(spec.value)  # type: ignore[arg-type]
    if spec.kind is TargetKind.CLASS:
        return 1.0 if spec.direction == "high" else 0.0
    if spec.kind is TargetKind.CURVE:
        return 0.0  # channel is RMSE-to-curve
    return None  # direction: unbounded


def plot_comparison(results: list[dict[str, Any]], scenario: ScenarioConfig, rec: RunRecorder, rel: str) -> None:
    specs = scenario.targets
    panels = ["objective", *[s.task for s in specs]]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 5.0), squeeze=False)
    labels = [r["path"] for r in results]
    colors = [_method_color(r["method"]) for r in results]
    x = np.arange(len(results))
    for ax, panel in zip(axes[0], panels):
        if panel == "objective":
            means = [float(np.mean(r["objective_after_decode"])) for r in results]
            stds = [float(np.std(r["objective_after_decode"])) for r in results]
            ax.set_title("objective score  (lower = better)")
        else:
            spec = next(s for s in specs if s.task == panel)
            means = [float(np.mean(r["channels_after_decode"][panel])) for r in results]
            stds = [float(np.std(r["channels_after_decode"][panel])) for r in results]
            ref = _target_ref_line(spec)
            if ref is not None:
                ax.axhline(ref, color="#C44E52", ls="--", lw=1.0)
            ax.set_title(target_label(spec))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=3)
        ax.set_xticks(x, labels, rotation=75, ha="right", fontsize=7)
    fig.suptitle("Inverse-design paths — achieved objectives", y=1.02)
    rec.save_figure(rel, fig)
    plt.close(fig)


def plot_objective_vs_targets(
    results: list[dict[str, Any]],
    scenario: ScenarioConfig,
    seed_channels: dict[str, np.ndarray],
    seed_objective: np.ndarray,
    rec: RunRecorder,
    rel: str,
) -> None:
    specs = scenario.targets
    fig, axes = plt.subplots(1, len(specs), figsize=(5.2 * len(specs), 5.0), squeeze=False)
    for ax, spec in zip(axes[0], specs):
        task = spec.task
        ax.scatter(
            seed_objective,
            seed_channels[task],
            marker="*",
            s=70,
            color=DISCOVERED_ELEMENT_COLOR,
            label="seed",
            zorder=1,
        )
        for r in results:
            ax.scatter(
                r["objective_after_decode"],
                r["channels_after_decode"][task],
                s=18,
                alpha=0.6,
                color=_method_color(r["method"]),
                zorder=2,
            )
        ref = _target_ref_line(spec)
        if ref is not None:
            ax.axhline(ref, color="#C44E52", ls="--", lw=1.0)
        ax.set_xlabel("objective score  (lower = better)")
        ax.set_ylabel(target_label(spec))
    fig.suptitle("Objective score vs per-target channels", y=1.02)
    rec.save_figure(rel, fig)
    plt.close(fig)


#: Font size for the composition text in the seed→optimised map, tuned against ``_MAP_ROW_HEIGHT``
#: so rows stay compact without the formulas colliding.
_MAP_FONT = 13


_MAP_ROW_HEIGHT = 0.34  # data-unit row height; the figure height scales with n_rows × this


#: Short display names for the tasks this project optimises most often, so a row like
#: ``Δformation_energy=-1.36`` cannot push the right edge of the figure into the colour bar.
#: Anything not listed falls back to :func:`_short_task`.
_TASK_SHORT: dict[str, str] = {
    "formation_energy": "FE",
    "magnetization": "mag",
    "magnetic_moment": "mm",
    "quasicrystal": "QC",
    "is_quasicrystal": "QC",
}


def _short_task(task: str) -> str:
    """Compact per-row label for a task name (initials for multi-word names, else a prefix)."""
    if task in _TASK_SHORT:
        return _TASK_SHORT[task]
    parts = [p for p in task.split("_") if p]
    if len(parts) > 1:
        return "".join(p[0] for p in parts).upper()
    return task if len(task) <= 6 else task[:4]


def _target_direction_arrow(spec: TargetSpec) -> str:
    """Which way this target wants its channel to move (empty when there is no single way).

    Rendered next to every delta so the reader can check a delta's sign against the goal without
    scrolling back to the header. ``VALUE`` targets get no arrow — their goal is a point, not a
    direction, and it is spelled out in the legend line under the title instead.
    """
    if spec.kind is TargetKind.CURVE:
        return "↓"  # channel is RMSE-to-curve: always drive to 0
    if spec.kind is TargetKind.VALUE:
        return ""
    return "↑" if spec.direction == "high" else "↓"


def _format_channel(spec: TargetSpec, value: float) -> str:
    """A target's channel value — class probabilities as percent, everything else as a number."""
    return f"{value * 100:.1f}%" if spec.kind is TargetKind.CLASS else f"{value:.2f}"


def _format_channel_delta(spec: TargetSpec, delta: float) -> str:
    """Signed change from the seed's channel — percentage points for class probabilities."""
    return f"{delta * 100:+.1f}pp" if spec.kind is TargetKind.CLASS else f"{delta:+.2f}"


def _top_fractions(weights: np.ndarray, *, top_k: int | None, eps: float = 1e-3) -> dict[str, float]:
    """``{element: fraction}`` for the largest ``top_k`` entries, renormalised to sum 1.

    ``top_k=6`` mirrors :func:`format_weights`, so the map plots exactly the elements that
    ``decoded_composition`` prints; the renormalisation is what makes a truncated row still read
    as percentages of 100. ``top_k=None`` keeps every element above ``eps`` (used for seeds,
    which are short and should be shown whole).
    """
    row = np.asarray(weights, dtype=float)
    order = np.argsort(row)[::-1]
    if top_k is not None:
        order = order[:top_k]
    picked = {DEFAULT_ELEMENTS[int(i)]: float(row[int(i)]) for i in order if row[int(i)] > eps}
    total = sum(picked.values())
    return {el: v / total for el, v in picked.items()} if total > 0 else picked


#: Colour ramp for element symbols in the seed→optimised map, keyed on how many of a path's rows
#: contain that element. ``inferno`` truncated to ``[0.05, 0.80]``: dark purple for "reached for
#: once" up to orange for "reached for everywhere". The full ramp's ends — near-black and pale
#: yellow — are both illegible as text on white, so the map uses the readable middle.
_MAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "inverse_map", matplotlib.colormaps["inferno"](np.linspace(0.05, 0.80, 256))
)


#: Advance width of DejaVu Sans Mono (matplotlib's default monospace face) in em. Every glyph in
#: these rows is monospaced, so a row's width is exactly ``n_chars × advance × fontsize`` — that
#: lets :func:`plot_seed_to_optimized` size its columns from the text *before* drawing, instead
#: of hard-coding fractions that break as soon as a scenario adds a target.
_MONO_ADVANCE_EM = 0.602


_ROW_PART_SEP_PT = 2.0  # HPacker `sep` between the pieces of a row


def _row_parts(
    comp: Mapping[str, float], suffix: str, element_colors: Mapping[str, Any] | None
) -> list[tuple[str, float, Any, bool]]:
    """One composition row as ``(text, fontsize, colour, bold)`` pieces, largest fraction first.

    ``element_colors`` colours the element symbols (optimised side); ``None`` keeps the row
    monochrome (seed side), so the colour gradient reads purely as "what the optimiser reached
    for". Returning a description rather than drawing lets the caller measure the row first.
    """
    parts: list[tuple[str, float, Any, bool]] = []
    for el, frac in sorted(comp.items(), key=lambda kv: -kv[1]):
        color = element_colors.get(el, "#aaaaaa") if element_colors is not None else "#111"
        parts.append((el, _MAP_FONT, color, True))
        parts.append((f"{frac * 100:.1f} ", _MAP_FONT, "#111", False))
    parts.append((suffix, _MAP_FONT - 2, "#555", False))
    return parts


def _row_width_pt(parts: Sequence[tuple[str, float, Any, bool]]) -> float:
    """Width of a rendered row in points (exact for a monospaced face — see ``_MONO_ADVANCE_EM``)."""
    text = sum(len(t) * _MONO_ADVANCE_EM * size for t, size, _, _ in parts)
    return text + _ROW_PART_SEP_PT * max(len(parts) - 1, 0)


def _render_row(ax: Any, x_axes_frac: float, y_data: float, parts: Sequence[tuple[str, float, Any, bool]]) -> None:
    """Draw a row built by :func:`_row_parts`, left edge anchored at ``x_axes_frac``.

    Assembled from :class:`~matplotlib.offsetbox.TextArea` pieces rather than one string so each
    element symbol can carry its own colour while the whole row stays baseline-aligned.
    """
    if not parts:
        return
    boxes: list[Any] = [
        TextArea(
            text,
            textprops=dict(color=color, fontsize=size, family="monospace", fontweight="bold" if bold else "normal"),
        )
        for text, size, color, bold in parts
    ]
    ax.add_artist(
        AnnotationBbox(
            HPacker(children=boxes, align="baseline", pad=0, sep=_ROW_PART_SEP_PT),
            (x_axes_frac, y_data),
            xycoords=("axes fraction", "data"),
            frameon=False,
            box_alignment=(0, 0.5),
            pad=0,
        )
    )


def plot_seed_to_optimized(
    seeds: list[str],
    result: dict[str, Any],
    specs: Sequence[TargetSpec],
    seed_channels: Mapping[str, np.ndarray],
    seed_objective: np.ndarray,
    rec: RunRecorder,
    rel: str,
) -> None:
    """Per-seed 1:1 view — left column the seed, right column what the optimiser made of it.

    Both sides are normalised to fractions and printed as percent, so the numbers match the
    ``"Au65 Ga20 Gd15"`` convention seeds are written in.

    * **Seed side** — monochrome formula + ``(obj=…, <task>=<channel>, …)``.
    * **Optimised side** — element symbols coloured by how many rows of *this path* contain them
      (near-black = this path reached for it once, bright = it reached for it everywhere), then
      ``(obj=…, <task>=<channel>[<Δ vs seed>]<goal arrow>, …)``. The absolute value says where
      the candidate landed, the delta says how far this seed moved, and the arrow says which way
      the target wanted it to go.
    * **Colour bar** carries the appearance-count scale, complementing the pooled
      ``element_frequency_heatmap.png`` with per-seed detail.

    Column positions and the figure width are measured from the text, so the layout survives a
    scenario adding targets (which lengthens every parenthetical) without columns colliding.
    """
    decoded_weights = np.asarray(result["optimized_weights"], dtype=float)
    opt_obj = np.asarray(result["objective_after_decode"], dtype=float)
    n = min(len(seeds), len(decoded_weights))
    if n == 0:
        logger.warning(f"seed→optimised map for '{result['path']}': no rows to draw — skipping.")
        return
    if len(seeds) != len(decoded_weights):
        logger.warning(
            f"seed→optimised map for '{result['path']}': {len(seeds)} seeds vs "
            f"{len(decoded_weights)} optimised rows — plotting the first {n}."
        )

    seed_comps = [_top_fractions(formula_to_composition(s), top_k=None) for s in seeds[:n]]
    opt_comps = [_top_fractions(w, top_k=6) for w in decoded_weights[:n]]

    # Appearance count over this path's optimised pool — drives the colours and the colour bar.
    counts: Counter = Counter(el for comp in opt_comps for el in comp)
    cmap = _MAP_CMAP
    norm = mcolors.Normalize(vmin=0, vmax=n)
    element_colors = {el: cmap(norm(c)) for el, c in counts.items()}

    channels = result["channels_after_decode"]
    seed_rows, opt_rows = [], []
    for i in range(n):
        seed_suffix = " (obj={:.2f}, {})".format(
            seed_objective[i],
            ", ".join(f"{_short_task(s.task)}={_format_channel(s, float(seed_channels[s.task][i]))}" for s in specs),
        )
        opt_suffix = " (obj={:.2f}, {})".format(
            opt_obj[i],
            ", ".join(
                f"{_short_task(s.task)}={_format_channel(s, float(channels[s.task][i]))}"
                f"[{_format_channel_delta(s, float(channels[s.task][i] - seed_channels[s.task][i]))}]"
                f"{_target_direction_arrow(s)}"
                for s in specs
            ),
        )
        seed_rows.append(_row_parts(seed_comps[i], seed_suffix, None))
        opt_rows.append(_row_parts(opt_comps[i], opt_suffix, element_colors))

    # Lay the two columns out from the widest row on each side, with the arrow in the gutter.
    header_left, header_right = (
        "Seed (fraction × 100 · obj · channels)",
        "Optimised composition (fraction × 100 · obj · channel[Δ vs seed]goal)",
    )
    gutter_pt = 3 * _MAP_FONT
    left_pt = max([_row_width_pt(p) for p in seed_rows] + [len(header_left) * _MONO_ADVANCE_EM * _MAP_FONT])
    right_pt = max([_row_width_pt(p) for p in opt_rows] + [len(header_right) * _MONO_ADVANCE_EM * _MAP_FONT])
    axes_pt = left_pt + gutter_pt + right_pt
    opt_x, arrow_x = (left_pt + gutter_pt) / axes_pt, (left_pt + gutter_pt / 2) / axes_pt

    # Axes placed by hand: the text column must be exactly `axes_pt` wide or the measured column
    # split stops matching what is drawn (default subplot margins would eat ~20% of it).
    text_in, gap_in, cbar_in, title_in = axes_pt / 72.0, 0.25, 0.3, 0.8
    fig_w = text_in + gap_in + cbar_in
    fig_h = max(3.2, _MAP_ROW_HEIGHT * n + 1.4) + title_in  # floor keeps the colour-bar label legible
    body = (fig_h - title_in) / fig_h  # rows + colour bar live below the two-line title
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes((0.0, 0.0, text_in / fig_w, body))
    ax_cbar = fig.add_axes(((text_in + gap_in) / fig_w, 0.04, cbar_in / fig_w, body - 0.08))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.9, n - 0.3)
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.text(0.0, -0.75, header_left, fontsize=_MAP_FONT, fontweight="bold", va="bottom")
    ax.text(opt_x, -0.75, header_right, fontsize=_MAP_FONT, fontweight="bold", va="bottom")
    for i in range(n):
        _render_row(ax, 0.0, i, seed_rows[i])
        ax.text(arrow_x, i, "→", fontsize=_MAP_FONT + 2, color="#888", ha="center", va="center")
        _render_row(ax, opt_x, i, opt_rows[i])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar)
    cb.set_label(f"Element appearance count\nin optimised pool (out of {n})", fontsize=_MAP_FONT - 2)
    cb.ax.tick_params(labelsize=_MAP_FONT - 3)

    legend = " · ".join(f"{_short_task(s.task)} = {target_label(s)}" for s in specs)
    fig.suptitle(
        f"Seed → optimised composition · {result['path']}\nobj = objective score (lower = better) · {legend}",
        fontsize=_MAP_FONT + 1,
        y=0.999,
    )
    rec.save_figure(rel, fig)
    plt.close(fig)


def plot_element_frequency(results: list[dict[str, Any]], seeds: list[str], rec: RunRecorder, rel: str) -> None:
    seed_elements = set().union(*[element_system(s) for s in seeds]) if seeds else set()
    counts: dict[str, dict[str, int]] = {}
    all_elems: dict[str, int] = {}
    for r in results:
        c: dict[str, int] = {}
        for formula in r["decoded_composition"]:
            for el in element_system(formula):
                c[el] = c.get(el, 0) + 1
                all_elems[el] = all_elems.get(el, 0) + 1
        counts[r["path"]] = c
    top = [el for el, _ in sorted(all_elems.items(), key=lambda kv: -kv[1])[:25]]
    matrix = np.array([[counts[r["path"]].get(el, 0) for el in top] for r in results], dtype=float)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(top)), max(4.0, 0.4 * len(results))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(top)), top, rotation=90, fontsize=7)
    for tick, el in zip(ax.get_xticklabels(), top):
        if el not in seed_elements:  # discovered elements (not in any seed) highlighted
            tick.set_color(DISCOVERED_ELEMENT_COLOR)
    ax.set_yticks(range(len(results)), [r["path"] for r in results], fontsize=7)
    fig.colorbar(im, ax=ax, label="occurrences", fraction=0.03)
    ax.set_title("Element frequency across paths (orange = discovered)")
    rec.save_figure(rel, fig)
    plt.close(fig)


def write_scenario_md(sc_dir: Path, scenario: ScenarioConfig, summary: list[dict[str, Any]]) -> None:
    lines = [f"# Inverse design — {scenario.name}", "", "Targets:"]
    lines.extend(f"- {target_label(t)}  (weight {t.weight:g})" for t in scenario.targets)
    lines.append("")
    lines.append("| path | method | objective (mean±std, lower = better) | elapsed(s) |")
    lines.append("|---|---|---|---|")
    for row in summary:
        lines.append(
            f"| {row['path']} | {row['method']} | {row['objective_mean']}±{row['objective_std']} | {row['elapsed_s']} |"
        )
    (sc_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_root_summary(root: Path, all_summary: Mapping[str, Any], cfg: InverseConfig) -> None:
    lines = ["# Inverse design — all scenarios", "", f"Checkpoint: `{cfg.checkpoint}`", ""]
    for name, summary in all_summary.items():
        best = min(summary, key=lambda row: row["objective_mean"]) if summary else None
        best_str = f"best path by objective: **{best['path']}** ({best['objective_mean']})" if best else "(no paths)"
        lines.append(f"- **{name}** — {best_str}")
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
