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


curves = {}
for mode, res_dir, suffix in (("step", RES_STEP, ""), ("epoch", RES_EPOCH, "_epoch")):
    xs, ys = [], []
    for n in COUNTS:
        p = res_dir / f"mt_n{n}{suffix}.csv"
        if p.exists():
            m, cnt = mean_final(p)
            if m is not None:
                xs.append(n)
                ys.append(m)
    curves[mode] = (xs, ys)

print(f"{'n':>6} {'step':>8} {'epoch':>8} {'delta':>8}")
step_d = dict(zip(*curves["step"]))
epoch_d = dict(zip(*curves["epoch"]))
for n in COUNTS:
    s, e = step_d.get(n), epoch_d.get(n)
    s_txt = "-" if s is None else f"{s:.3f}"
    e_txt = "-" if e is None else f"{e:.3f}"
    d_txt = "-" if None in (s, e) else f"{e - s:+.3f}"
    print(f"{n:>6} {s_txt:>8} {e_txt:>8} {d_txt:>8}")

fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=150)
for mode, color, marker, label in (
    ("step", BLUE, "o", 'resample="step" — frozen subset (rikyu GB200)'),
    ("epoch", RED, "D", 'resample="epoch" — redrawn every epoch (ism A100)'),
):
    xs, ys = curves[mode]
    if xs:
        ax.plot(xs, ys, color=color, marker=marker, ms=7, lw=2, mec="white", label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 9 if mode == "epoch" else -14), fontsize=8, color=color, ha="center")
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
