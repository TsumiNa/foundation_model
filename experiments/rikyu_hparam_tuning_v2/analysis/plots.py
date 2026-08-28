#!/usr/bin/env python3
"""Draw the v2 figures FROM THE SUMMARY JSON ONLY.

This constraint is not stylistic. Raw run output lives on RIKYU and never enters git; the summary
JSONs do. So whoever picks this up next can redraw every figure from what git actually carries,
and a figure that needed the raw runs to exist would be a figure they could not reproduce
(PLAN §9.3).

    python analysis/plots.py stage0 --summary summary/stage0.json -o analysis/
    python analysis/plots.py stage_a --summary summary/stage_a.json -o analysis/
    python analysis/plots.py finals  --summary summary/finals_a.json -o analysis/
    python analysis/plots.py stage_c --summary summary/stage_c.json -o analysis/

PALETTE — validated, not chosen by eye
--------------------------------------
The categorical order below passed all six checks of the dataviz validator at 4 slots (light
surface): lightness band, chroma floor, CVD separation (worst adjacent pair ΔE 10.3 under
deuteranopia), normal-vision floor, and >= 3:1 contrast against the surface.

Two results from that check are load-bearing and should not be quietly undone:

  * v1's five-colour order put ORANGE #EE7733 next to GREEN #009E73, which separate by only
    ΔE 7.4 under protanopia — inside the 6-8 band that is legal ONLY alongside a second encoding.
    v1's figures were fine because they also varied the marker. v2 uses PURPLE as the fourth hue
    instead, which clears the band outright, so the figures do not depend on that rescue.
  * MUTED #6b7280 FAILS the chroma floor as a series colour — it reads as gray. It is kept for
    text, annotation and gridlines, and must never be promoted to a series.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

# Categorical hues, in FIXED order — assigned by entity, never cycled, never reordered by rank.
SERIES = ["#0077BB", "#CC3311", "#009E73", "#AA3377"]
# Ink and furniture. Not series colours (see the chroma-floor note above).
MUTED, GRID, TEXT = "#6b7280", "#e5e7eb", "#374151"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# An axis with at most this many distinct values is drawn at those values; more than this and it
# is binned. The grid axes sit below it, the random-search axes far above.
PER_VALUE_MAX = 10
MARGINAL_BINS = 8


def _bin_axis(buckets: dict[float, list[float]], logx: bool, n_bins: int):
    """Aggregate a continuously-sampled axis into ``n_bins``, returning (centres, score groups).

    Bins are equal-width in log space for the learning-rate axes, because that is how they were
    sampled and how they matter — the distance from 1e-3 to 2e-3 is the same kind of step as
    1e-2 to 2e-2. Empty bins are dropped rather than plotted as gaps at zero.
    """
    xs = sorted(buckets)
    lo, hi = xs[0], xs[-1]
    if logx and lo > 0:
        edges = [10 ** (math.log10(lo) + i * (math.log10(hi) - math.log10(lo)) / n_bins)
                 for i in range(n_bins + 1)]
    else:
        edges = [lo + i * (hi - lo) / n_bins for i in range(n_bins + 1)]
    centres, groups = [], []
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        members = [v for v in xs if (left <= v < right or (i == n_bins - 1 and v == hi))]
        if not members:
            continue
        scores = [s for v in members for s in buckets[v]]
        centres.append(math.sqrt(left * right) if (logx and left > 0) else (left + right) / 2)
        groups.append(scores)
    return centres, groups


def tidy(ax, *, grid_axis="y"):
    """Recessive furniture: the data should be the darkest thing on the page."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _band_span(ax, band: dict, label: str = "seed band (single-run range)"):
    """The noise band, drawn once, so every margin in the figure is read against it."""
    half = (band.get("range") or 0) / 2
    if half <= 0:
        return
    ax.axhspan(-half, half, color=MUTED, alpha=0.10, zorder=0, label=label)


def plot_stage0(summary: dict, out: Path) -> Path:
    """Per-seed scores for each anchor arm.

    A dot per seed rather than a bar of the mean: with nine seeds the SPREAD is the point of the
    figure — it is what every later margin gets measured against — and a bar chart would hide
    exactly that.
    """
    arms = summary["arms"]
    labels = list(arms)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    _band_span(ax, arms["s0_base"]["score"])

    for i, name in enumerate(labels):
        arm = arms[name]
        values = list(arm["per_seed"].values())
        colour = SERIES[i % len(SERIES)]
        # A deterministic fan so coincident seeds stay countable. Two choices matter here:
        # the offset is fixed rather than random, so the figure is reproducible from the JSON;
        # and it follows SEED ORDER, not value rank. Fanning by rank would put a monotone slope
        # on x and invite reading a trend where the x axis carries no quantity at all.
        offs = [(j - (len(values) - 1) / 2) * 0.012 for j in range(len(values))]
        ax.scatter([i + o for o in offs], values, s=38, color=colour, alpha=0.75,
                   edgecolor="white", linewidth=1.0, zorder=3)
        mean = arm["score"]["mean"]
        ax.hlines(mean, i - 0.16, i + 0.16, color=colour, linewidth=2.4, zorder=4)
        # Label above the mean rule, not beside it: beside runs off the axes for the rightmost
        # arm, and widening the axes to fit it wastes the plot area on whitespace.
        ax.annotate(f"{mean:+.2%}", (i, mean), xytext=(0, 7), textcoords="offset points",
                    fontsize=9.5, color=TEXT, va="bottom", ha="center", zorder=5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{n}\n(n={arms[n]['n_seeds']})" for n in labels])
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylabel("mean relative improvement vs the untuned anchor")
    # The annotations are percentages, so the axis must be too — one figure, one unit.
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1%}"))
    ax.axhline(0, color=MUTED, linewidth=1.0)
    ax.set_title("Stage 0 — the anchor, and the seed spread every later margin is read against",
                 fontsize=10, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    tidy(ax)
    fig.tight_layout()
    path = out / "stage0_anchor.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_stage_a(summary: dict, out: Path) -> list[Path]:
    """Two figures: the marginal effect of each searched axis, and the ranked short list."""
    ranking = summary["ranking"]
    band = summary["seed_band_from_stage0"]
    paths = []

    # --- marginals: one small multiple per axis -------------------------------------------
    axes_keys = [
        ("training__encoder_lr", "encoder_lr", True),
        ("training__scheduler__min_lr", "min_lr", True),
        ("training__scheduler__patience", "scheduler patience", False),
        ("model__latent_dim", "latent_dim", False),
    ]
    present = [(k, lbl, log) for k, lbl, log in axes_keys
               if len({r["point"].get(k) for r in ranking if k in r["point"]}) > 1]
    if present:
        fig, axs = plt.subplots(1, len(present), figsize=(3.3 * len(present), 3.4), sharey=True)
        axs = [axs] if len(present) == 1 else list(axs)
        for ax, (key, label, logx) in zip(axs, present):
            buckets: dict[float, list[float]] = {}
            for r in ranking:
                v = r["point"].get(key)
                if v is not None:
                    buckets.setdefault(v, []).append(r["score_mean"])
            xs = sorted(buckets)

            # A grid axis has a handful of values; a random-search axis has one per sample. Both
            # feed this plot, and drawing them the same way makes the second unreadable: 206
            # distinct encoder_lr values become 206 overlapping tick labels and a sawtooth line
            # whose every point averages one or two configs. So few values are drawn as they were
            # searched, and many values are BINNED — which is also what makes the aggregate mean
            # mean anything.
            if len(xs) <= PER_VALUE_MAX:
                centres = xs
                groups = [buckets[x] for x in xs]
                ax.set_xticks(xs)
                ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=8)
                ax.minorticks_off()
            else:
                centres, groups = _bin_axis(buckets, logx, MARGINAL_BINS)

            means = [sum(g) / len(g) for g in groups]
            best = [max(g) for g in groups]
            ax.plot(centres, means, "-o", color=SERIES[0], linewidth=2, markersize=6,
                    label="mean over the other axes", zorder=3)
            ax.plot(centres, best, "--^", color=SERIES[1], linewidth=1.6, markersize=6,
                    label="best at this value", zorder=3)
            if logx:
                ax.set_xscale("log")
            # A rug of the actual samples, so binning never hides where the search really looked.
            ax.plot(xs, [ax.get_ylim()[0]] * len(xs), "|", color=MUTED, alpha=0.35,
                    markersize=4, clip_on=False, zorder=1)
            ax.set_xlabel(label)
            tidy(ax)
        axs[0].set_ylabel("mean relative improvement")
        # One decimal, not zero: the whole spread here is a couple of percent, and rounding to
        # whole percent labels four distinct gridlines identically.
        axs[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1%}"))
        # One legend for the whole row — repeating it per panel is noise.
        axs[0].legend(loc="best", frameon=False, fontsize=8)
        fig.suptitle("Stage A' — marginal effect of each searched axis "
                     "(mean marginalises the others; 'best' shows the reachable frontier)",
                     fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p = out / "stage_a_marginals.png"
        fig.savefig(p, dpi=170)
        plt.close(fig)
        paths.append(p)

    # --- ranked short list with error bars ------------------------------------------------
    top = ranking[: min(15, len(ranking))]
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(top) + 1.8))
    ys = range(len(top))
    tied = set(summary.get("leader_ties", {}).get("statistically_tied_with_leader", []))
    leader = summary.get("leader_ties", {}).get("leader")
    for y, r in zip(ys, top):
        # Colour by STATUS (separated from the leader or not), which is the figure's actual
        # question. Rank is already encoded by position, so colouring by rank would spend a
        # channel restating the y-axis.
        is_tied = r["config"] in tied or r["config"] == leader
        colour = SERIES[2] if is_tied else SERIES[0]
        ax.errorbar(r["score_mean"], y, xerr=2 * r["score_sem"], fmt="o", color=colour,
                    markersize=6, capsize=3, linewidth=1.6, zorder=3)
    half = (band.get("range") or 0) / 2
    if half > 0:
        ax.axvspan(-half, half, color=MUTED, alpha=0.10, zorder=0,
                   label="seed band (single-run range)")
    ax.axvline(0, color=MUTED, linewidth=1.0)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["config"] for r in top], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("mean relative improvement vs the untuned anchor  (bars = ±2 SE)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1%}"))
    n_tied = len(tied)
    # Two short lines rather than one long one: the single-line version ran past the axes and lost
    # its config count, which is the half that carries the finding.
    ax.set_title(
        f"Stage A' ranking — {n_tied} config{'s' if n_tied != 1 else ''} not separated "
        f"from the leader\nat this seed count (green)",
        fontsize=9.5, loc="left")
    # Outside the axes: at the default lower-right it sits on top of the last row's error bar.
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.10 - 0.4 / len(top)),
              frameon=False, fontsize=8)
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    p = out / "stage_a_ranking.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    paths.append(p)
    return paths


def plot_finals(summary: dict, out: Path) -> Path:
    """Finals arms with their seed distributions — the figure that answers 'is the order real?'"""
    arms = summary["arms"]
    order = summary["ranking"]
    fig, ax = plt.subplots(figsize=(max(6.4, 1.5 * len(order) + 2), 4.2))
    for i, name in enumerate(order):
        arm = arms[name]
        values = list(arm["per_seed"].values())
        colour = SERIES[i % len(SERIES)]
        ax.scatter([i] * len(values), values, s=22, color=colour, alpha=0.45,
                   edgecolor="none", zorder=2)
        s = arm["score"]
        ax.errorbar(i, s["mean"], yerr=2 * s["sem"], fmt="_", color=colour,
                    markersize=26, capsize=5, linewidth=2.4, zorder=4)
    ax.axhline(0, color=MUTED, linewidth=1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{n}\n(n={arms[n]['n_seeds']})" for n in order], fontsize=7.5)
    ax.set_ylabel("mean relative improvement vs the untuned anchor")
    resolved = summary.get("ranking_is_fully_resolved")
    ax.set_title(
        "Finals — dots are seeds, bars are ±2 SE of the mean.  "
        + ("Ordering is resolved." if resolved
           else "Leader NOT separated from all rivals: the ordering is not supported."),
        fontsize=10, loc="left")
    tidy(ax)
    fig.tight_layout()
    path = out / "finals.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_a4(summary: dict, out: Path) -> Path:
    """Schedule on/off against encoder_lr — the paired comparison, not two separate curves.

    The question a4 asks is whether the LR schedule earns its place, and the honest way to show
    that is AT each learning rate: a schedule that helps at 5e-3 and does nothing at 2e-4 says the
    schedule's job is rescuing a too-high start, which is a different claim from "the schedule
    helps". Two lines plus the per-LR difference underneath, so both readings are available.
    """
    cells = summary["cells"]
    paired = summary["paired_by_lr"]
    lrs = [p["encoder_lr"] for p in paired]
    floor = min((p["encoder_lr"] for p in paired if not p["below_previous_floor"]), default=None)

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(7.2, 6.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    for i, (arm, label) in enumerate((("sched", "scheduler on"), ("flat", "constant LR"))):
        xs, ys, errs = [], [], []
        for lr in lrs:
            c = cells[arm].get(f"{lr:g}")
            if c:
                xs.append(lr)
                ys.append(c["mean"])
                errs.append(2 * c["sem"])
        ax.errorbar(xs, ys, yerr=errs, fmt="-o" if i == 0 else "--s", color=SERIES[i],
                    linewidth=2, markersize=6, capsize=3, label=label, zorder=3)

    # Everything left of this line is ground stage A' never searched.
    if floor is not None:
        for a in (ax, axd):
            a.axvline(floor, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
        ax.annotate("stage A' floor\n(left of here was never searched)", (floor, ax.get_ylim()[1]),
                    xytext=(-6, -8), textcoords="offset points", fontsize=8, color=MUTED,
                    ha="right", va="top")

    ax.axhline(0, color=MUTED, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_ylabel("mean relative improvement\nvs the untuned anchor")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1%}"))
    resolved = [p for p in paired if p["separated"]]
    ax.set_title(
        "a4 — does the LR schedule earn its place?  bars are ±2 SE.\n"
        f"{len(resolved)} of {len(paired)} learning rates separate the two arms",
        fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)
    tidy(ax)

    # The paired difference, which is what the verdict is actually read from.
    deltas = [p["delta_schedule_minus_flat"] for p in paired]
    errs = [2 * (p["se_of_difference"] or 0.0) for p in paired]
    colours = [SERIES[2] if p["separated"] else MUTED for p in paired]
    for lr, d, e, c in zip(lrs, deltas, errs, colours):
        axd.errorbar(lr, d, yerr=e, fmt="o", color=c, markersize=6, capsize=3, linewidth=1.6, zorder=3)
    axd.axhline(0, color=MUTED, linewidth=1.0)
    axd.set_xscale("log")
    axd.set_xlabel("encoder_lr")
    axd.set_ylabel("schedule − flat")
    axd.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1%}"))
    axd.set_title("green = the seeds separate the two arms at this LR; grey = they do not",
                  fontsize=8.5, loc="left", color=MUTED)
    tidy(axd)

    fig.tight_layout()
    path = out / "a4_schedule_vs_flat.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_stage_c(summary: dict, out: Path) -> Path:
    """Deficit to the single-task ceiling by task size, across arms.

    Deficit rather than raw R2, and split by size group, because that is the frame the replay
    campaign reported in (REPORT_20260809) — the same axis lets a reader put this campaign's
    arms next to that one's without rescaling anything in their head.
    """
    arms = summary["arms"]
    groups = ["big", "mid", "small"]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(arms) + 3), 4.2))
    width = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        vals = [arm["deficit"].get(g) or 0.0 for g in groups]
        xs = [g_i + i * width - 0.4 + width / 2 for g_i in range(len(groups))]
        ax.bar(xs, vals, width=width * 0.9, color=SERIES[i % len(SERIES)],
               label=arm["label"], zorder=3, edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["big  >=20k (6)", "mid  3k-8.1k (14)", "small  <=1.2k (2)"])
    ax.set_ylabel("deficit to the single-task ceiling  (lower is better)")
    ax.set_title("Stage C' — deficit by task size. One seed per arm, so small gaps are unresolved.",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    tidy(ax)
    fig.tight_layout()
    path = out / "stage_c_deficit.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_ceilings(summary: dict, out: Path) -> Path:
    """How far the inherited H200 "ceilings" sit from a same-régime measurement of the same thing.

    Drawn as a per-task offset rather than two overlaid ceiling curves, because the claim being
    made is about the SIZE and the NON-CONSTANCY of the error. Two curves invite the reader to
    eyeball a gap that is 0.10 in one task and −0.02 in another and call it "a small shift"; one
    signed bar per task, sorted, makes the spread the subject.

    Sorted by offset, not by N or alphabetically: the ordering is the finding.
    """
    from common import CEILING, CEILING_SAME_REGIME, N_TRAIN, size_group

    rows = [(t_, CEILING_SAME_REGIME[t_] - CEILING[t_], size_group(t_))
            for t_ in CEILING_SAME_REGIME
            if t_ in CEILING and t_ != "material_type"]  # accuracy vs macro-F1: not a difference
    rows.sort(key=lambda r: r[1])
    labels = [f"{t_}  ({N_TRAIN.get(t_, 0):,})" for t_, _, _ in rows]
    colours = {"big": SERIES[0], "mid": SERIES[1], "small": SERIES[2]}

    fig, ax = plt.subplots(figsize=(7.6, 0.30 * len(rows) + 1.9))
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color=[colours[r[2]] for r in rows],
            height=0.7, zorder=3, edgecolor="white", linewidth=1.0)
    ax.axvline(0, color=MUTED, linewidth=1.0, zorder=4)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("same-régime ceiling − inherited H200 ceiling   (positive: the old one was too low)")
    ax.set_title("The inherited ceilings understate 17 of 23 tasks, and not by a constant",
                 fontsize=10, loc="left")
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=7, color=c, label=g)
               for g, c in colours.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right", title="task size",
              title_fontsize=8)
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    path = out / "ceiling_frame_offset.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_transfer(summary: dict, out: Path) -> Path:
    """Per-task transfer as a RELATIVE percentage, ordered by training-set size.

    Percentage rather than raw R2 delta because +0.045 does not tell a reader whether that is a
    lot; +6.9% does. The relative view (delta / single-task R2) is used rather than the error-
    reduction view (delta / residual), which is more meaningful where a task has headroom and
    explodes where it does not — formation_energy's residual is 0.0053, so its -0.0036 reads as
    -68% and would dominate the figure with a task whose change is practically nil.

    Which is exactly why the ABSOLUTE delta is printed on every bar. A percentage hides magnitude,
    and the campaign's practical threshold is stated in absolute R2 (0.01); showing only the
    percentage would let a large-looking bar stand for a change nobody would act on.

    Ordered by training-set size because the claim is about the RELATIONSHIP between transfer and
    data size; sorting by effect size would let that be read off the sort instead of the data.

    Fill = the difference is BOTH resolved and >= 0.01 absolute. Hollow = it fails one of those,
    and the legend says which, so "statistically real but negligible" cannot be read as a result.
    """
    rows = [r for r in summary["per_task"] if "transfer" in r]
    rows.sort(key=lambda r: -(r["n_train"] or 0))
    ys = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8.0, 0.62 * len(rows) + 2.1))
    for y, r in zip(ys, rows):
        rel = r.get("relative_pct")
        if rel is None:
            continue
        base = abs(r["single_task_r2"]) or 1.0
        se2_rel = 2 * (r["se_of_difference"] or 0.0) / base * 100.0
        matters = r.get("matters", r["separated"])
        colour = SERIES[0] if r["transfer"] > 0 else SERIES[1]
        style = (dict(color=colour, edgecolor="white")
                 if matters else dict(color="white", edgecolor=colour, hatch="///"))
        ax.barh(y, rel, height=0.55, zorder=3, linewidth=1.0, **style)
        ax.errorbar(rel, y, xerr=se2_rel, fmt="none", ecolor=TEXT, elinewidth=1.2,
                    capsize=3, zorder=5)
        # The absolute delta, so the percentage can never stand alone. Anchored past the OUTER
        # end of the error bar, not the bar end — the whisker overruns short bars and the label
        # would land on top of its own cap.
        outer = rel + (se2_rel if rel >= 0 else -se2_rel)
        ax.annotate(f"{r['transfer']:+.4f}", (outer, y), textcoords="offset points",
                    xytext=(9 if rel >= 0 else -9, 0), va="center",
                    ha="left" if rel >= 0 else "right", fontsize=8, color=MUTED)
    ax.axvline(0, color=MUTED, linewidth=1.0, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['task']}\n{r['n_train']:,} labels" for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.margins(x=0.20)
    ax.set_xlabel("relative change in R²  (%)   ·   bars: ±2 SE   ·   grey number: absolute ΔR²")
    ax.set_title("Transfer at the adopted configuration — the smallest tasks gain, the largest pay",
                 fontsize=10, loc="left")
    key = lambda fc, ec, lb: plt.Line2D(  # noqa: E731
        [], [], marker="s", linestyle="", markersize=8, color="white",
        markerfacecolor=fc, markeredgecolor=ec, label=lb)
    ax.legend(handles=[key(SERIES[0], SERIES[0], "multi-task better"),
                       key(SERIES[1], SERIES[1], "single-task better"),
                       key("white", MUTED, "within noise, or |ΔR²| < 0.01")],
              frameon=False, fontsize=8, loc="lower right")
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    path = out / "transfer_adopted.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_finals_sigma(summary: dict, out: Path) -> Path:
    """Each finalist arm's run-to-run σ against its 25-seed mean.

    A scatter because the claim is a RELATIONSHIP between two measured quantities, and because
    with ten arms every point can be shown rather than summarised. The regression line is left
    off deliberately: ten points do not earn a fitted line, and drawing one would invite reading
    a slope off a sample this small. The correlation is stated in the title instead, where it
    carries its own n.

    Labels are English like every other figure here — matplotlib's DejaVu Sans has no CJK
    glyphs, so Chinese would render as boxes; the Chinese prose lives in the report and deck.

    Two arms are labelled and the rest are not. Labelling all ten would collide and would also
    flatten the figure's argument, which is about those two: the untuned control, and the
    configuration that won the 5-seed grid and finished last here.
    """
    arms = summary["arms"]
    control = next((k for k in arms if k.endswith("_base")), None)
    leader = summary.get("leader")
    # The 5-seed grid leader is the arm the finals demoted furthest; identified by lowest mean.
    fell = min(arms, key=lambda k: arms[k]["score"]["mean"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for name, arm in arms.items():
        s = arm["score"]
        highlight = name in (control, leader, fell)
        ax.scatter(s["sigma"] * 100, s["mean"] * 100, s=90 if highlight else 55,
                   color=(SERIES[1] if name == control else
                          SERIES[0] if name == leader else
                          SERIES[3] if name == fell else MUTED),
                   zorder=4, edgecolor="white", linewidth=1.4,
                   alpha=1.0 if highlight else 0.75)
    for name, label, dx, dy in ((leader, "adopted", 0.12, 0.10),
                                (control, "untuned control", 0.12, 0.06),
                                (fell, "led the 5-seed grid", -0.10, 0.12)):
        if name not in arms:
            continue
        s = arms[name]["score"]
        ax.annotate(label, (s["sigma"] * 100, s["mean"] * 100),
                    textcoords="offset points", xytext=(dx * 60, dy * 60),
                    fontsize=9, color=TEXT,
                    ha="left" if dx > 0 else "right")
    n = len(arms)
    xs = [a["score"]["sigma"] for a in arms.values()]
    ys = [a["score"]["mean"] for a in arms.values()]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else float("nan")
    ax.set_xlabel("run-to-run σ  (%)")
    ax.set_ylabel("mean score over 25 seeds  (%)")
    ax.set_title(f"Good configurations are good partly by being stable  —  r = {r:+.3f}, n = {n} arms",
                 fontsize=10, loc="left")
    tidy(ax, grid_axis="both")
    fig.tight_layout()
    path = out / "finals_sigma_vs_mean.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure",
                    choices=["stage0", "stage_a", "finals", "stage_c", "a4", "ceilings", "transfer",
                             "finals_sigma"])
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, default=Path("."))
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    drawn = {
        "stage0": plot_stage0,
        "stage_a": plot_stage_a,
        "finals": plot_finals,
        "stage_c": plot_stage_c,
        "a4": plot_a4,
        "ceilings": plot_ceilings,
        "transfer": plot_transfer,
        "finals_sigma": plot_finals_sigma,
    }[args.figure](summary, args.out)
    for p in (drawn if isinstance(drawn, list) else [drawn]):
        print(p)


if __name__ == "__main__":
    main()
