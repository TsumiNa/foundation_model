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
            means = [sum(buckets[x]) / len(buckets[x]) for x in xs]
            best = [max(buckets[x]) for x in xs]
            ax.plot(xs, means, "-o", color=SERIES[0], linewidth=2, markersize=6,
                    label="mean over the other axes", zorder=3)
            ax.plot(xs, best, "--^", color=SERIES[1], linewidth=1.6, markersize=6,
                    label="best at this value", zorder=3)
            if logx:
                ax.set_xscale("log")
            # Ticks ONLY at the values actually searched. Matplotlib's default ticker invents a
            # continuum — it will happily label 150/200/250/300/350 on an axis whose only sampled
            # values are 128 and 384 — which reads as a swept range rather than two options.
            ax.set_xticks(xs)
            # Rotated because eight log-spaced values collide horizontally — unrotated they
            # render as "0.0020.003" and the axis becomes unreadable exactly where the campaign's
            # dominant knob is.
            ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7.5,
                               rotation=45 if len(xs) > 4 else 0,
                               ha="right" if len(xs) > 4 else "center")
            ax.minorticks_off()
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
    ax.set_title(
        f"Stage A' ranking — green = not separated from the leader at this seed count"
        f" ({n_tied} config{'s' if n_tied != 1 else ''})",
        fontsize=10, loc="left")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=["stage0", "stage_a", "finals", "stage_c"])
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
    }[args.figure](summary, args.out)
    for p in (drawn if isinstance(drawn, list) else [drawn]):
        print(p)


if __name__ == "__main__":
    main()
