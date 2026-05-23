# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Side-by-side overview figure for the two inverse-design algorithms.

Renders a 21:9 white-background diagram that compares ``optimize_latent`` (latent-space
gradient descent) and ``optimize_composition`` (differentiable KMD) on three axes:

  * a flow diagram (top) showing where the optimisation variable lives and what the
    forward path through the model looks like;
  * the loss decomposition (middle), highlighting the term that *differs* between the
    two methods in red;
  * a compact parameter table (bottom) listing the user-facing knobs each algorithm
    exposes, with one-line meanings.

Run it as::

    uv run python docs/figures/inverse_design_algorithms_overview.py

Output is written next to this script as ``inverse_design_algorithms_overview.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- styling constants -----------------------------------------------------------------------

LATENT_COLOR = "#55A868"  # green — matches the comparison/scatter figures' latent group
COMP_COLOR = "#2563EB"  # blue   — matches the comparison/scatter figures' composition group
ACCENT_RED = "#C44E52"  # used to flag the loss-term that differs between the two methods
TEXT_DARK = "#111"
TEXT_MUTED = "#444"
DIVIDER_GRAY = "#BBB"

BOX_KW = dict(boxstyle="round,pad=0.010,rounding_size=0.012", linewidth=1.4)
ARROW_KW = dict(
    arrowstyle="-|>",
    mutation_scale=14,
    linewidth=1.4,
    color="#555",
    shrinkA=2,
    shrinkB=2,
)

# Column extents — both columns share the same vertical section budget, defined once here
# so the two columns line up exactly. Sections (top → bottom): title strip, flow diagram,
# loss formula, parameter table, figure caption.
X_LEFT_COL = (0.04, 0.48)
X_RIGHT_COL = (0.52, 0.96)

Y_TITLE_TOP, Y_TITLE_BOT = 0.99, 0.92  # title strip
Y_FLOW_HEADER = 0.895
Y_FLOW_TOP = 0.815  # row 1 — top of forward flow
Y_FLOW_MID = 0.705  # row 2 — middle (round-trip on latent, KMD transform on comp)
Y_FLOW_BOT = 0.595  # row 3 — task heads
Y_FLOW_CAPTION = 0.530  # "Adam updates" caption

Y_LOSS_HEADER = 0.475
Y_LOSS_LINE_0 = 0.420
Y_LOSS_LINE_1 = 0.378
Y_LOSS_LINE_2 = 0.336

Y_PARAMS_HEADER = 0.292
Y_PARAMS_TOP = 0.265  # top edge of the param-table rounded rect
PARAMS_HEIGHT = 0.225  # rect height; rows are auto-spaced inside


# --- low-level helpers ----------------------------------------------------------------------


def _box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    fc: str = "white",
    ec: str = "#444",
    fontsize: int = 11,
    fontweight: str = "normal",
    text_color: str = TEXT_DARK,
) -> tuple[float, float]:
    """Draw a rounded box centred at (cx, cy). Returns the centre for arrow chaining."""
    cx, cy = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, facecolor=fc, edgecolor=ec, **BOX_KW))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight, color=text_color)
    return cx, cy


def _arrow(ax, src: tuple[float, float], dst: tuple[float, float], **overrides) -> None:
    kw = {**ARROW_KW, **overrides}
    ax.add_patch(FancyArrowPatch(src, dst, **kw))


def _section_header(ax, x_centre: float, y: float, text: str, color: str) -> None:
    ax.text(x_centre, y, text, ha="center", va="center", fontsize=13, fontweight="bold", color=color)


def _draw_title_strip(ax, col_x: tuple[float, float], main: str, sub: str, color: str) -> None:
    """Two-line title: bold main on top, monospace API signature underneath. Keeps the strip
    short enough that the column boundary never clips the text."""
    x_left, x_right = col_x
    x_mid = (x_left + x_right) / 2
    ax.add_patch(
        mpatches.Rectangle(
            (x_left, Y_TITLE_BOT),
            x_right - x_left,
            Y_TITLE_TOP - Y_TITLE_BOT,
            facecolor=color,
            edgecolor="none",
            alpha=0.92,
        )
    )
    ax.text(x_mid, 0.972, main, ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(x_mid, 0.935, sub, ha="center", va="center", fontsize=10, color="white", fontfamily="monospace")


# --- column renderers ------------------------------------------------------------------------


def _draw_latent_column(ax) -> None:
    """Left column: ``optimize_latent`` in latent-space mode (the post-PR-#18 default).

    Optimisation variable is the latent ``h``. The visual story is the AE round-trip:
    ``h ↔ h' = tanh(E(D(h)))`` — when ``α = 0`` it's unconstrained (and decoded-x̂ drifts off
    manifold), when ``α = 1`` h is locked to h' (over-constrained); the user knob ``ae_align_scale``
    interpolates. We pull this to the side of the h box rather than spending a separate row on
    it, since the round-trip is a *loss term* more than a forward step.
    """
    x_left, x_right = X_LEFT_COL
    x_mid = (x_left + x_right) / 2

    _draw_title_strip(ax, X_LEFT_COL, "Latent-space optimisation", 'optimize_latent(optimize_space="latent")', LATENT_COLOR)

    # ============================ FLOW DIAGRAM ============================
    _section_header(ax, x_mid, Y_FLOW_HEADER, "Flow", LATENT_COLOR)

    box_w, box_h = 0.115, 0.055

    # Row 1: Seed x → Encoder → h  (highlighted as the optimisation variable).
    p_seed = _box(ax, (x_left + 0.06, Y_FLOW_TOP), (box_w, box_h), "Seed x", fc="#F2F2F2", ec="#888")
    p_enc1 = _box(ax, (x_mid - 0.005, Y_FLOW_TOP), (box_w + 0.03, box_h), "Encoder + tanh", fc="white", ec=LATENT_COLOR)
    p_h = _box(
        ax,
        (x_right - 0.06, Y_FLOW_TOP),
        (box_w, box_h + 0.012),
        "latent h\n(optimise this)",
        fc=LATENT_COLOR,
        ec=LATENT_COLOR,
        text_color="white",
        fontweight="bold",
        fontsize=10,
    )
    _arrow(ax, (p_seed[0] + box_w / 2, Y_FLOW_TOP), (p_enc1[0] - (box_w + 0.03) / 2, Y_FLOW_TOP))
    _arrow(ax, (p_enc1[0] + (box_w + 0.03) / 2, Y_FLOW_TOP), (p_h[0] - box_w / 2, Y_FLOW_TOP))

    # Row 2: AE round-trip detour — h → D → x̂ → E → tanh → h'. Compact: one combined box.
    p_round = _box(
        ax,
        (x_mid, Y_FLOW_MID),
        (0.34, box_h + 0.005),
        "AE round-trip:    D(·)  →  x̂  →  E(·)  →  tanh   ⇒   h'",
        fc="white",
        ec=LATENT_COLOR,
        fontsize=10,
    )
    # h → round-trip box (drops from row 1 to row 2 on the right side)
    _arrow(ax, (p_h[0] - 0.005, Y_FLOW_TOP - (box_h + 0.012) / 2), (p_round[0] + 0.17 - 0.01, Y_FLOW_MID + box_h / 2))
    # Return arrow back up to h, labelled with the alignment loss — this is the key idea.
    _arrow(
        ax,
        (p_round[0] - 0.17 + 0.01, Y_FLOW_MID + box_h / 2),
        (p_h[0] - box_w / 2 - 0.21, Y_FLOW_TOP - (box_h + 0.012) / 2),
        color=ACCENT_RED,
        linewidth=1.5,
    )
    ax.text(
        x_mid - 0.13,
        (Y_FLOW_TOP + Y_FLOW_MID) / 2 - 0.005,
        "α · ‖ h − h' ‖²\n(AE-alignment penalty)",
        ha="center",
        va="center",
        fontsize=10,
        color=ACCENT_RED,
        fontweight="bold",
    )

    # Row 3: h → task heads.
    p_heads = _box(
        ax,
        (x_mid, Y_FLOW_BOT),
        (0.34, box_h),
        "Task heads   (regression  +  P(quasicrystal))",
        fc="white",
        ec=LATENT_COLOR,
    )
    _arrow(ax, (p_h[0], Y_FLOW_TOP - (box_h + 0.012) / 2), (p_heads[0] + 0.16, Y_FLOW_BOT + box_h / 2))
    ax.text(
        x_mid,
        Y_FLOW_CAPTION,
        "Adam updates h  ←  ∇_h L",
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT_MUTED,
        style="italic",
    )

    # ============================ LOSS ============================
    _section_header(ax, x_mid, Y_LOSS_HEADER, "Loss", LATENT_COLOR)
    ax.text(
        x_left + 0.01,
        Y_LOSS_LINE_0,
        r"L  =  $\sum_t \lambda_t \,\| \hat y_t - \mathrm{target}_t \|^2$",
        ha="left",
        va="center",
        fontsize=13,
        color=TEXT_DARK,
    )
    ax.text(
        x_left + 0.01,
        Y_LOSS_LINE_1,
        r"        $+\; w_{\mathrm{cls}} \cdot \left( -\log P(c = \mathrm{QC}) \right)$",
        ha="left",
        va="center",
        fontsize=13,
        color=TEXT_DARK,
    )
    ax.text(
        x_left + 0.01,
        Y_LOSS_LINE_2,
        r"        $+\; \alpha \cdot \| h - \mathrm{tanh}(E(D(h))) \|^2$    ← differs from composition",
        ha="left",
        va="center",
        fontsize=13,
        color=ACCENT_RED,
    )

    # ============================ PARAMETERS ============================
    _section_header(ax, x_mid, Y_PARAMS_HEADER, "Key tunable parameters", LATENT_COLOR)
    params: list[tuple[str, str]] = [
        ("ae_align_scale  α ∈ [0, 1]", "pull h toward the AE manifold (0 = unconstrained, 1 = strict). Sweet spot ≈ 0.5."),
        ("class_target_weight  w_cls", "relative weight on P(QC) vs the regression targets."),
        ("steps,  lr", "Adam optimisation budget (default 200 steps, lr 0.1)."),
        ("num_restarts,  perturbation_std", "independent restarts with Gaussian jitter on the seed."),
    ]
    _draw_param_table(
        ax,
        x_left + 0.005,
        Y_PARAMS_TOP,
        x_right - x_left - 0.01,
        PARAMS_HEIGHT,
        params,
        accent=LATENT_COLOR,
    )


def _draw_composition_column(ax) -> None:
    """Right column: ``optimize_composition`` — the differentiable-KMD path.

    Optimisation variable is the simplex of element weights ``w`` (parameterised through
    softmax logits ``θ``). No AE round-trip: ``w`` is the recipe you would report.
    """
    x_left, x_right = X_RIGHT_COL
    x_mid = (x_left + x_right) / 2

    _draw_title_strip(ax, X_RIGHT_COL, "Differentiable KMD (composition)", "optimize_composition(...)", COMP_COLOR)

    # ============================ FLOW DIAGRAM ============================
    _section_header(ax, x_mid, Y_FLOW_HEADER, "Flow", COMP_COLOR)

    box_w, box_h = 0.115, 0.055

    # Row 1: logits θ → softmax → w
    p_theta = _box(
        ax,
        (x_left + 0.06, Y_FLOW_TOP),
        (box_w, box_h + 0.012),
        "logits θ\n(optimise this)",
        fc=COMP_COLOR,
        ec=COMP_COLOR,
        text_color="white",
        fontweight="bold",
        fontsize=10,
    )
    p_softmax = _box(ax, (x_mid - 0.005, Y_FLOW_TOP), (box_w + 0.01, box_h), "softmax", fc="white", ec=COMP_COLOR)
    p_w = _box(
        ax,
        (x_right - 0.06, Y_FLOW_TOP),
        (box_w + 0.01, box_h + 0.012),
        "w  (simplex)\nelement recipe",
        fc="white",
        ec=COMP_COLOR,
        fontsize=10,
    )
    _arrow(ax, (p_theta[0] + box_w / 2, Y_FLOW_TOP), (p_softmax[0] - (box_w + 0.01) / 2, Y_FLOW_TOP))
    _arrow(ax, (p_softmax[0] + (box_w + 0.01) / 2, Y_FLOW_TOP), (p_w[0] - (box_w + 0.01) / 2, Y_FLOW_TOP))

    # Row 2: x = w · K (KMD transform) → Encoder + tanh
    p_kmd = _box(
        ax,
        (x_mid + 0.09, Y_FLOW_MID),
        (box_w + 0.03, box_h + 0.005),
        "x  =  w · K\n(KMD transform)",
        fc="white",
        ec=COMP_COLOR,
        fontsize=10,
    )
    p_enc = _box(
        ax,
        (x_mid - 0.09, Y_FLOW_MID),
        (box_w + 0.03, box_h),
        "Encoder + tanh",
        fc="white",
        ec=COMP_COLOR,
    )
    _arrow(ax, (p_w[0], Y_FLOW_TOP - (box_h + 0.012) / 2), (p_kmd[0], Y_FLOW_MID + (box_h + 0.005) / 2))
    _arrow(ax, (p_kmd[0] - (box_w + 0.03) / 2, Y_FLOW_MID), (p_enc[0] + (box_w + 0.03) / 2, Y_FLOW_MID))

    # Side annotation: w *is* the answer.
    ax.text(
        x_left + 0.07,
        (Y_FLOW_TOP + Y_FLOW_MID) / 2,
        "w  is the reported recipe\n(no AE round-trip needed)",
        ha="center",
        va="center",
        fontsize=10,
        color=COMP_COLOR,
        style="italic",
    )

    # Row 3: heads
    p_heads = _box(
        ax,
        (x_mid, Y_FLOW_BOT),
        (0.34, box_h),
        "Task heads   (regression  +  P(quasicrystal))",
        fc="white",
        ec=COMP_COLOR,
    )
    _arrow(ax, (p_enc[0], Y_FLOW_MID - box_h / 2), (p_heads[0] - 0.05, Y_FLOW_BOT + box_h / 2))
    ax.text(
        x_mid,
        Y_FLOW_CAPTION,
        "Adam updates θ  ←  ∇_θ L     ( w = softmax(θ) )",
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT_MUTED,
        style="italic",
    )

    # ============================ LOSS ============================
    _section_header(ax, x_mid, Y_LOSS_HEADER, "Loss", COMP_COLOR)
    ax.text(
        x_left + 0.01,
        Y_LOSS_LINE_0,
        r"L  =  $\sum_t \lambda_t \,\| \hat y_t - \mathrm{target}_t \|^2$",
        ha="left",
        va="center",
        fontsize=13,
        color=TEXT_DARK,
    )
    ax.text(
        x_left + 0.01,
        Y_LOSS_LINE_1,
        r"        $+\; w_{\mathrm{cls}} \cdot \left( -\log P(c = \mathrm{QC}) \right)$",
        ha="left",
        va="center",
        fontsize=13,
        color=TEXT_DARK,
    )
    ax.text(
        x_left + 0.01,
        Y_LOSS_LINE_2,
        r"        $+\; (1 - d) \cdot H(w),\;\; H(w) = -\sum_i w_i \log w_i$    ← differs from latent",
        ha="left",
        va="center",
        fontsize=13,
        color=ACCENT_RED,
    )

    # ============================ PARAMETERS ============================
    _section_header(ax, x_mid, Y_PARAMS_HEADER, "Key tunable parameters", COMP_COLOR)
    params: list[tuple[str, str]] = [
        ("diversity_scale  d ∈ [0, 1]", "per-output element diversity (1 = no penalty, 0 = peaky few-element)."),
        ("class_target_weight  w_cls", "relative weight on P(QC) vs the regression targets."),
        ("seed_blend  ∈ [0, 1]", "keep seed prior vs mix uniform (0.95 lets new elements enter)."),
        ("allowed_elements", "element whitelist (e.g. ALLOY_PALETTE); disallowed forced to w = 0."),
        ("element_step_scale", "per-element gradient scaling; 0 = hard-lock to seed value."),
        ("steps,  lr", "Adam budget over the logits (default 300 steps, lr 0.05)."),
    ]
    _draw_param_table(
        ax,
        x_left + 0.005,
        Y_PARAMS_TOP,
        x_right - x_left - 0.01,
        PARAMS_HEIGHT,
        params,
        accent=COMP_COLOR,
    )


def _draw_param_table(
    ax,
    x0: float,
    y_top: float,
    w: float,
    h: float,
    params: list[tuple[str, str]],
    *,
    accent: str,
) -> None:
    """Compact two-row-per-param list: bold accent-coloured name on top, dim meaning below.

    Side-by-side layout (name | meaning) ran the meanings off the column edge for the longer
    descriptions — stacking gives each meaning the full column width so we don't have to truncate.
    The rectangle gives the section a visual boundary so the column scans as one block.
    """
    n = len(params)
    if n == 0:
        return
    ax.add_patch(
        FancyBboxPatch(
            (x0, y_top - h),
            w,
            h,
            boxstyle="round,pad=0.005,rounding_size=0.010",
            linewidth=0.8,
            facecolor="#FBFBFD",
            edgecolor="#DDD",
        )
    )
    inner_x = x0 + 0.012

    row_h = h / max(n, 1)
    name_offset = row_h * 0.28  # name sits above row centre
    meaning_offset = -row_h * 0.22  # meaning sits below row centre
    for i, (name, meaning) in enumerate(params):
        y_centre = y_top - (i + 0.5) * row_h
        ax.text(
            inner_x,
            y_centre + name_offset,
            name,
            ha="left",
            va="center",
            fontsize=11,
            color=accent,
            fontfamily="monospace",
            fontweight="bold",
        )
        ax.text(
            inner_x,
            y_centre + meaning_offset,
            meaning,
            ha="left",
            va="center",
            fontsize=10.5,
            color=TEXT_DARK,
        )


# --- top-level renderer ----------------------------------------------------------------------


def render(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(21, 9), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Vertical divider between the two columns.
    ax.plot([0.50, 0.50], [0.04, 0.91], color=DIVIDER_GRAY, linewidth=1.0, linestyle=(0, (4, 4)))

    # Figure-level caption — anchors the diagram in one sentence so a reader who only glances
    # at the bottom can still extract the main message.
    ax.text(
        0.5,
        0.022,
        "Both methods share the regression-MSE + (−log P(QC)) backbone; the third loss term "
        "— and the optimisation variable — is what differs.",
        ha="center",
        va="center",
        fontsize=11,
        color=TEXT_MUTED,
        style="italic",
    )

    _draw_latent_column(ax)
    _draw_composition_column(ax)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    render(here / "inverse_design_algorithms_overview.png")
