#!/usr/bin/env python3
"""Per-task replay saturation: final performance vs replay amount, one panel per task.

For every task and every completed run, x = the number of labels that task gets per replay step
under that run's setting (fraction runs: fraction x n_train of the task; fixed-count runs:
min(count, n_train) — clamped), y = the task's final primary metric (test R2; accuracy for
material_type) after all 24 steps. Each panel also shows the task's own ceiling (its at-intro
primary, averaged over runs — the level before any forgetting) and, where enough points exist,
a saturation fit gap(x) = g0 / (1 + x/k) with the estimated x for 90% gap recovery.

Outputs per_task_saturation.png + a per-task fit table on stdout.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE / "per_task_saturation.png"

# n_valid_train per task (train-split valid labels), extracted from the run logs.
N_TRAIN = {
    "density": 23678, "efermi": 23668, "final_energy": 23678, "total_magnetization": 23678,
    "volume": 23678, "dielectric_total": 3124, "dielectric_ionic": 3124, "dielectric_electronic": 3124,
    "magnetization": 1160, "curie": 6272, "neel": 3466, "kp": 3875,
    "magnetic_susceptibility": 58, "zt": 3445, "power_factor": 3638, "thermal_conductivity": 4272,
    "electrical_resistivity": 5051, "dos_density": 7009, "seebeck": 8072,
    "formation_energy": 23180, "magnetic_moment": 851, "tc": 7207, "klat": 3863, "material_type": 33556,
}  # fmt: skip
ORDER = list(N_TRAIN)

RUNS = {
    "base0p05": ("frac", 0.05), "0p10": ("frac", 0.10), "0p15": ("frac", 0.15), "0p20": ("frac", 0.20),
    "n100": ("count", 100), "n200": ("count", 200), "n500": ("count", 500), "n1000": ("count", 1000),
    "n1500": ("count", 1500), "n2000": ("count", 2000), "n2500": ("count", 2500),
}  # fmt: skip
# The 0.05 baseline mirrors the reference config: per_task = 0.10 on the fixed tail.
BASE0P05_TAIL_OVERRIDE = {"formation_energy", "magnetic_moment", "tc", "klat", "material_type"}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path: Path):
    rows = list(csv.DictReader(open(path)))
    mx = max(int(r["step"]) for r in rows)
    final = {r["task"]: fnum(r["primary"]) for r in rows if int(r["step"]) == mx}
    intro = {r["task"]: fnum(r["primary"]) for r in rows if r["new_task"] == r["task"]}
    return final, intro


def replay_x(task: str, tag: str, kind: str, val: float) -> float:
    """Labels this task gets per replay step under this run's setting."""
    n = N_TRAIN[task]
    if kind == "frac":
        f = 0.10 if (tag == "base0p05" and task in BASE0P05_TAIL_OVERRIDE) else val
        return f * n
    return float(min(val, n))


def fit_gap(xs: np.ndarray, gaps: np.ndarray):
    """gap(x) = g0 / (1 + x/k) via linear regression on 1/gap.

    Returns (params, reason): params = (g0, k) or None; reason explains a skipped fit —
    the forgetting-gap model only applies when the final metric sits BELOW the task's own
    at-intro ceiling and the gap decays with x.
    """
    above = (gaps < -1e-3).sum()
    good = gaps > 1e-3
    if above >= 3:
        return None, f"no fit: {above}/{len(gaps)} runs END ABOVE the ceiling (backward transfer)"
    if good.sum() < 4:
        return None, "no fit: <4 runs below the ceiling"
    if len(set(np.round(xs[good], 6))) < 3:
        return None, "no fit: x collapsed by count clamping"
    A = np.vstack([np.ones(good.sum()), xs[good]]).T
    coef, *_ = np.linalg.lstsq(A, 1.0 / gaps[good], rcond=None)
    a, b = coef
    if a <= 0 or b <= 0:
        return None, "no fit: gap does not decay with x"
    return (1.0 / a, a / b), ""  # (g0, k)


data = {}
for tag, (kind, val) in RUNS.items():
    p = RES / f"mt_{tag}.csv"
    if p.exists():
        final, intro = load(p)
        data[tag] = dict(kind=kind, val=val, final=final, intro=intro)

BLUE, ORANGE, MUTED, GRID = "#0077BB", "#EE7733", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED})
SINGLE_DIR = HERE / "per_task_saturation"
SINGLE_DIR.mkdir(exist_ok=True)

LEGEND_NOTE = (
    "blue ○ fixed-count runs · orange □ fraction runs · dashed = the task's own at-intro ceiling\n"
    "dotted vertical = the task's full train size · fit: gap = g0/(1+x/k)"
)


def draw_panel(ax, task: str, *, annotate_size: float = 7.5) -> tuple[str, str]:
    """Draw one task's saturation panel onto ax; return (fit_txt, printed table row)."""
    xs, ys, kinds, ceils = [], [], [], []
    for tag, d in data.items():
        y = d["final"].get(task)
        c = d["intro"].get(task)
        if y is None:
            continue
        xs.append(replay_x(task, tag, d["kind"], d["val"]))
        ys.append(y)
        kinds.append(d["kind"])
        if c is not None:
            ceils.append(c)
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    ceiling = float(np.mean(ceils))
    n_train = N_TRAIN[task]

    fit, reason = fit_gap(xs_arr, ceiling - ys_arr)
    if fit is not None:
        g0, k = fit
        n90 = 9 * k
        grid = np.logspace(np.log10(max(min(xs), 3) * 0.7), np.log10(n_train), 200)
        ax.plot(grid, ceiling - g0 / (1 + grid / k), color="#334155", lw=1.6, zorder=2)
        fit_txt = f"knee≈{k:,.0f}   90%≈{n90:,.0f}"
        row = f"{task:24} {n_train:>7} {ceiling:>7.3f} {min(ys):>7.3f}..{max(ys):<7.3f} {k:>8,.0f} {n90:>8,.0f} {n90 / n_train:>10.1f}x"
    else:
        fit_txt = reason
        row = f"{task:24} {n_train:>7} {ceiling:>7.3f} {min(ys):>7.3f}..{max(ys):<7.3f} {reason}"

    for kind, marker, color in (("count", "o", BLUE), ("frac", "s", ORANGE)):
        sel = [i for i, kk in enumerate(kinds) if kk == kind]
        ax.scatter(xs_arr[sel], ys_arr[sel], s=34, marker=marker, color=color, edgecolor="white", lw=0.6, zorder=3)
    ax.axhline(ceiling, color=MUTED, ls="--", lw=1.1, zorder=1)
    ax.axvline(n_train, color=GRID, lw=1.0, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set_title(f"{task}  (n_train={n_train:,})", fontsize=9 if annotate_size < 9 else 12)
    ax.text(0.02, 0.03, fit_txt, transform=ax.transAxes, fontsize=annotate_size, color="#334155", va="bottom")
    ax.grid(True, which="both", color=GRID, lw=0.5, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=annotate_size)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fit_txt, row


# --- combined grid + per-task singles -------------------------------------------------------
fig, axes = plt.subplots(4, 6, figsize=(24, 14.5), dpi=140)
print(f"{'task':24} {'n_train':>7} {'ceiling':>7} {'final(min..max)':>16} {'knee k':>8} {'n_90%':>8} {'90%/n_train':>11}")
print("-" * 110)
for ax, task in zip(axes.flat, ORDER):
    _, row = draw_panel(ax, task)
    print(row)

fig.suptitle(f"Per-task replay saturation — final metric vs labels replayed per step\n{LEGEND_NOTE}", fontsize=13, y=0.995)
fig.supxlabel("labels replayed per step for this task (log scale)", fontsize=11)
fig.supylabel("final primary metric after all 24 steps (test R²; accuracy for material_type)", fontsize=11)
fig.tight_layout(rect=(0.01, 0.01, 1, 0.96))
fig.savefig(OUT, bbox_inches="tight")
print(f"\nsaved {OUT}")

for task in ORDER:
    fig1, ax1 = plt.subplots(figsize=(7.2, 5.2), dpi=150)
    draw_panel(ax1, task, annotate_size=9.5)
    ax1.set_xlabel("labels replayed per step for this task (log scale)")
    ax1.set_ylabel("final primary metric (test)")
    fig1.text(0.01, 0.005, LEGEND_NOTE.replace("\n", " · "), fontsize=6.5, color=MUTED)
    fig1.tight_layout(rect=(0, 0.02, 1, 1))
    fig1.savefig(SINGLE_DIR / f"{task}.png", bbox_inches="tight")
    plt.close(fig1)
print(f"saved 24 per-task figures to {SINGLE_DIR}/")
