#!/usr/bin/env python3
"""Headline comparison: all-task mean final R² vs replay n — frozen subset vs per-epoch resampling.

One point per (mode, n): the mean over the 24 tasks of the final primary metric after all
24 steps. This is the "R² 0.371 (n=100) → 0.600 (n=2500)" curve from the step sweep, with the
epoch-resampling curve on top — the effective-replay multiplier is read off horizontally
(which frozen-n a given epoch-n matches).

Outputs: mean_final_compare.png + the table on stdout.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES_EPOCH = HERE.parent / "results"
RES_STEP = HERE.parent.parent / "rikyu_replay_sweep" / "results"
OUT = HERE / "mean_final_compare.png"

COUNTS = [100, 200, 500, 1000, 1500, 2000, 2500]
BLUE, RED, MUTED, GRID = "#0077BB", "#CC3311", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED})


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean_final(path: Path):
    """Mean final primary over the 23 R² tasks (material_type/accuracy excluded — the
    REPORT_20260706 convention, which puts the step curve at 0.371@n100 → 0.600@n2500)."""
    rows = list(csv.DictReader(open(path)))
    mx = max(int(r["step"]) for r in rows)
    vals = [fnum(r["primary"]) for r in rows if int(r["step"]) == mx and r["task"] != "material_type"]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


MODES = (
    ("step", RES_STEP, "", BLUE, "o", 'resample="step", patience 8 — frozen subset (rikyu GB200)'),
    ("epoch", RES_EPOCH, "_epoch", RED, "D", 'resample="epoch", patience 8 (ism A100)'),
    ("p24", RES_EPOCH, "_epoch_p24", "#009E73", "s", 'resample="epoch", patience 24 ⇒ full 100 epochs (R-CCS H200)'),
    ("m150", RES_EPOCH, "_epoch_m150", "#EE7733", "^", 'resample="epoch", patience 24, max_epochs 150 (R-CCS H200)'),
)

curves = {}
for mode, res_dir, suffix, *_ in MODES:
    xs, ys = [], []
    for n in COUNTS:
        p = res_dir / f"mt_n{n}{suffix}.csv"
        if p.exists():
            m, cnt = mean_final(p)
            if m is not None:
                xs.append(n)
                ys.append(m)
    curves[mode] = (xs, ys)

hdr = " ".join(f"{m:>8}" for m, *_ in MODES)
print(f"{'n':>6} {hdr}")
dicts = {m: dict(zip(*curves[m])) for m, *_ in MODES}
for n in COUNTS:
    row = " ".join("       -" if dicts[m].get(n) is None else f"{dicts[m][n]:>8.3f}" for m, *_ in MODES)
    print(f"{n:>6} {row}")

OFFSETS = {"step": -14, "epoch": 9, "p24": 9, "m150": -14}
fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=150)
for mode, _, _, color, marker, label in MODES:
    xs, ys = curves[mode]
    if xs:
        ax.plot(xs, ys, color=color, marker=marker, ms=7, lw=2, mec="white", label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, OFFSETS[mode]), fontsize=7.5, color=color, ha="center")
ax.set_xscale("log")
ax.set_xticks(COUNTS)
ax.set_xticklabels([str(n) for n in COUNTS])
ax.minorticks_off()
ax.set_xlabel("replay labels per old task per step, n (log scale)")
ax.set_ylabel("mean final test R² over 23 R² tasks (material_type excluded)")
ax.set_title("Replay retention: frozen subset vs per-epoch resampling", fontsize=13)
ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
ax.legend(frameon=False, loc="lower right", fontsize=9.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}")
