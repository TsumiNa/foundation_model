#!/usr/bin/env python3
"""Ratio-family (amount = fraction of each old task's labels) vs fixed-count replay under the
m150 recipe, plus the no-replay / joint-retrain baseline family.

Figures (all skip-tolerant while runs are still landing):
  ratio_cost_view.png   mean final R² (23 tasks) vs TOTAL labels replayed per step —
                        do ratio and fixed-n land on the same cost curve?
  baseline_family.png   (a) per-step BOXPLOTS of test R² over the learned tasks, no-replay vs
                        n1000-m150, symlog below −1 (the full distribution, not a threshold
                        count); (b) joint-retrain recovery vs epoch cap vs the replay-arm band.
(The per-size-group deficit view lives in replay_requirement_vs_size.py — every arm one axis.)

Metric convention: mean/median final test R² over the 23 R² tasks (material_type excluded).
Single-task ceiling & at-intro reference as in replay_requirement_vs_size.py.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES_E = HERE.parent / "results"
RES_S = HERE.parent.parent / "rikyu_replay_sweep" / "results"

BLUE, ORANGE, PURPLE, MUTED, GRID = "#0077BB", "#EE7733", "#AA3377", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED})

# Train-label counts for the 23 replayable tasks (material_type is introduced last, never
# replayed, and excluded from the metric; magnetic_susceptibility IS a replay cost but is
# excluded from deficit groups — degenerate single-task baseline).
N_TRAIN = {
    "density": 23678, "efermi": 23668, "final_energy": 23678, "total_magnetization": 23678,
    "volume": 23678, "dielectric_total": 3124, "dielectric_ionic": 3124, "dielectric_electronic": 3124,
    "magnetization": 1160, "curie": 6272, "neel": 3466, "kp": 3875,
    "zt": 3445, "power_factor": 3638, "thermal_conductivity": 4272,
    "electrical_resistivity": 5051, "dos_density": 7009, "seebeck": 8072,
    "formation_energy": 23180, "magnetic_moment": 851, "tc": 7207, "klat": 3863,
    "magnetic_susceptibility": 58,
}  # fmt: skip
DEFICIT_TASKS = {t: n for t, n in N_TRAIN.items() if t != "magnetic_susceptibility"}
GROUPS = (
    ("big tasks — ≥20k labels (6)", [t for t, n in DEFICIT_TASKS.items() if n >= 20000]),
    ("mid tasks — 3k–8.1k labels (14)", [t for t, n in DEFICIT_TASKS.items() if 3000 <= n < 20000]),
    ("small tasks — ≤1.2k labels (2)", [t for t, n in DEFICIT_TASKS.items() if n < 3000]),
)
COUNTS = [100, 200, 500, 1000, 1500, 2000, 2500]
RATIOS = [(0.10, "0p10"), (0.20, "0p20"), (0.30, "0p30"), (0.50, "0p50")]
JOINT_CAPS = [150, 200, 250, 300]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_final(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    mx = max(int(r["step"]) for r in rows)
    return {r["task"]: fnum(r["primary"]) for r in rows if int(r["step"]) == mx}


def mean23(final: dict[str, float]) -> float:
    vals = [v for t, v in final.items() if t != "material_type" and v is not None]
    return float(np.mean(vals))


def single_task_baseline() -> dict[str, float]:
    best: dict[str, float] = {}
    for r in csv.DictReader(open(RES_S / "warm_restart.csv")):
        v = fnum(r["primary"])
        if v is not None:
            best[r["task"]] = v if r["task"] not in best else max(best[r["task"]], v)
    return best


SINGLE = single_task_baseline()

fixed = {n: load_final(RES_E / f"mt_n{n}_epoch_m150.csv") for n in COUNTS}
ratio = {r: load_final(RES_E / f"mt_{tag}_epoch_m150.csv") for r, tag in RATIOS}
noreplay_rows = list(csv.DictReader(open(RES_E / "mt_noreplay.csv")))
joint = {}
for cap in JOINT_CAPS:
    p = RES_E / f"joint_retrain_m{cap}.json"
    if p.exists():
        s = json.loads(p.read_text())
        joint[cap] = {"after": {t: m["primary"] for t, m in s["metrics_after"].items()}, "epochs_run": s["epochs_run"]}

# at-intro reference: mean intro value per task over the m150 fixed+ratio runs (near-identical
# across runs by design)
intro_acc: dict[str, list[float]] = {t: [] for t in DEFICIT_TASKS}
for path in list(RES_E.glob("mt_n*_epoch_m150.csv")) + list(RES_E.glob("mt_0p*_epoch_m150.csv")):
    for r in csv.DictReader(open(path)):
        if r["new_task"] == r["task"] and r["task"] in intro_acc and fnum(r["primary"]) is not None:
            intro_acc[r["task"]].append(float(r["primary"]))
INTRO_REF = {t: float(np.mean(v)) for t, v in intro_acc.items() if v}

# ---------------------------------------------------------------- fig 1: cost view
fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
xs_f = [sum(min(n, N) for N in N_TRAIN.values()) for n in COUNTS if fixed[n]]
ys_f = [mean23(fixed[n]) for n in COUNTS if fixed[n]]
lab_f = [f"n{n}" for n in COUNTS if fixed[n]]
xs_r = [r * sum(N_TRAIN.values()) for r, _ in RATIOS if ratio[r]]
ys_r = [mean23(ratio[r]) for r, _ in RATIOS if ratio[r]]
lab_r = [f"r={r:g}" for r, _ in RATIOS if ratio[r]]
ax.plot(xs_f, ys_f, color=ORANGE, lw=2.2, marker="^", ms=6.5, mec="white", label="fixed-count (epoch-m150)", zorder=3)
ax.plot(xs_r, ys_r, color=PURPLE, lw=2.2, marker="o", ms=6.5, mec="white", label="ratio (epoch-m150)", zorder=3)
for x, y, s in zip(xs_f, ys_f, lab_f):
    off = (10, -3) if s == "n100" else (0, -13)
    ax.annotate(s, (x, y), textcoords="offset points", xytext=off, fontsize=7.5, color=MUTED,
                ha="left" if s == "n100" else "center")
for x, y, s in zip(xs_r, ys_r, lab_r):
    ax.annotate(s, (x, y), textcoords="offset points", xytext=(0, 7), fontsize=7.5, color=PURPLE, ha="center")
if 300 in joint:
    yj = float(np.mean([v for t, v in joint[300]["after"].items() if t != "material_type"]))
    ax.axhline(yj, color=MUTED, ls="--", lw=1.4)
    ax.text(0.02, yj + 0.0015, f"no replay → joint retrain at the end, CONVERGED ({yj:.3f})",
            transform=ax.get_yaxis_transform(), fontsize=8.5, color=MUTED)
ax.axhline(0.600, color=BLUE, ls=":", lw=1.4)
ax.text(0.02, 0.600 + 0.0015, "best frozen-subset arm (step-p8 n2500, 0.600)",
        transform=ax.get_yaxis_transform(), fontsize=8.5, color=BLUE)
ax.text(0.98, 0.578, "no replay, no retrain: mean R² = −33 (off scale)",
        transform=ax.get_yaxis_transform(), fontsize=8.5, color=MUTED, style="italic", ha="right")
ax.set_xscale("log")
ax.set_ylim(0.574, max(max(ys_f), max(ys_r)) + 0.008)
ax.set_xlabel("total labels replayed per step (sum over the 23 old tasks, log)")
ax.set_ylabel("mean final R² (23 tasks)")
ax.set_title("Same recipe, same cost axis: ratio and fixed-count replay land on one curve", fontsize=12)
ax.grid(True, which="major", color=GRID, lw=0.5, zorder=1)
ax.legend(loc="upper left", frameon=False, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "ratio_cost_view.png", bbox_inches="tight")
print("saved ratio_cost_view.png")


# ------------------------------------------------------------ fig 3: baseline family
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=150)

def step_dists(rows):
    """Per step: the test R² of every R²-task learned so far (full distribution, no threshold)."""
    steps = sorted({int(r["step"]) for r in rows})
    out = []
    for s in steps:
        vals = [fnum(r["primary"]) for r in rows if int(r["step"]) == s and r["task"] != "material_type"]
        out.append((s, [v for v in vals if v is not None]))
    return out


def styled_box(data, positions, face, edge):
    ax1.boxplot(
        data, positions=positions, widths=0.34, patch_artist=True, manage_ticks=False,
        boxprops={"facecolor": face, "edgecolor": edge, "lw": 1.0},
        whiskerprops={"color": edge, "lw": 1.0}, capprops={"color": edge, "lw": 1.0},
        medianprops={"color": edge, "lw": 1.6},
        flierprops={"marker": ".", "ms": 3, "mfc": edge, "mec": edge, "alpha": 0.6},
    )


nr = step_dists(noreplay_rows)
styled_box([v for _, v in nr], [s - 0.21 for s, _ in nr], "#d7d9dd", "#555555")
med_nr = float(np.median(nr[-1][1]))
n1000 = RES_E / "mt_n1000_epoch_m150.csv"
med_rp = float("nan")
if n1000.exists():
    rp = step_dists(list(csv.DictReader(open(n1000))))
    styled_box([v for _, v in rp], [s + 0.21 for s, _ in rp], "#fbdcc4", ORANGE)
    med_rp = float(np.median(rp[-1][1]))
ax1.set_yscale("symlog", linthresh=1)
ax1.set_ylim(-4000, 1.9)
ax1.set_yticks([1, 0, -1, -10, -100, -1000])
ax1.set_yticklabels(["1", "0", "−1", "−10", "−100", "−1000"])
ax1.axhline(0, color=MUTED, lw=1.2, zorder=1)
ax1.text(24.5, 0.06, "R² = 0 — constant-mean predictor", fontsize=8.5, color=MUTED, ha="right")
ax1.set_xlim(0.2, 24.8)
ax1.set_xticks([1, 4, 8, 12, 16, 20, 24])
ax1.set_xlabel("training step (task introductions)")
ax1.set_ylabel("test R² per learned task (symlog below −1)")
ax1.set_title(f"Per-step distribution of test R² over the learned tasks\n"
              f"(step-24 medians: no replay {med_nr:.0f} vs replay {med_rp:.2f}; axis is linear in [−1, 1])",
              fontsize=10.5)
from matplotlib.patches import Patch

ax1.legend(handles=[Patch(facecolor="#d7d9dd", edgecolor="#555555", label="no replay"),
                    Patch(facecolor="#fbdcc4", edgecolor=ORANGE, label="replay n1000 (epoch-m150)")],
           loc="lower left", frameon=False, fontsize=9)
ax1.grid(True, which="major", color=GRID, lw=0.5, zorder=0)

caps = [c for c in JOINT_CAPS if c in joint]
ys = [float(np.mean([v for t, v in joint[c]["after"].items() if t != "material_type"])) for c in caps]
ax2.plot(caps, ys, color="#555555", lw=2.2, marker="D", ms=7, mec="white", zorder=3)
for c, y in zip(caps, ys):
    er = joint[c]["epochs_run"]
    tag = f"{y:.3f}" + (f"\n(stopped @{er})" if er < c else "")
    ax2.annotate(tag, (c, y), textcoords="offset points", xytext=(0, 9), fontsize=8, ha="center", color="#374151")
band_lo, band_hi = 0.639, 0.663
ax2.axhspan(band_lo, band_hi, color=ORANGE, alpha=0.15, zorder=0)
ax2.annotate("continual replay arms (m150 recipe), n100…n2500 & r0.1…0.5",
             (min(caps) - 2, band_hi - 0.003), fontsize=8.5, color=ORANGE, va="top")
ax2.set_xticks(JOINT_CAPS)
ax2.set_xlabel("joint-retrain epoch cap")
ax2.set_ylabel("mean final R² (23 tasks)")
ax2.set_title("…and one full-data joint retrain at the end converges BELOW\nevery continual-replay arm (early stop at 214 epochs)",
              fontsize=10.5)
ax2.grid(True, which="major", color=GRID, lw=0.5, zorder=1)
for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("Baseline family: replay removed, rehearsal deferred to the end", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(HERE / "baseline_family.png", bbox_inches="tight")
print("saved baseline_family.png")

# ------------------------------------------------------------------- stdout tables
print("\nmean final R² (23 tasks)")
for n in COUNTS:
    if fixed[n]:
        print(f"  fixed n{n:<5} {mean23(fixed[n]):.3f}")
for r, tag in RATIOS:
    if ratio[r]:
        print(f"  ratio {r:<7g} {mean23(ratio[r]):.3f}")
for c in caps:
    y = float(np.mean([v for t, v in joint[c]["after"].items() if t != "material_type"]))
    print(f"  joint m{c:<5} {y:.3f}  (epochs_run {joint[c]['epochs_run']})")
print(f"  noreplay     {mean23({r['task']: fnum(r['primary']) for r in noreplay_rows if int(r['step']) == 24}):.3f}")

hyb_final = load_final(RES_E / "mt_hybrid_r03_f1500.csv")
if hyb_final:
    print(f"  hybrid       {mean23(hyb_final):.3f}  (max(1500, 0.3N))")
hj = RES_E / "hybrid_joint_retrain.json"
if hj.exists():
    hja = json.loads(hj.read_text())
    m = float(np.mean([v["primary"] for t, v in hja["metrics_after"].items() if t != "material_type"]))
    print(f"  hybrid+joint {m:.3f}  (epochs_run {hja['epochs_run']})")

print("\ndeficit to single-task ceiling (group means)")
hdr = [f"n{n}" for n in COUNTS if fixed[n]] + [f"r{r:g}" for r, _ in RATIOS if ratio[r]] + ["joint300"]
if hyb_final:
    hdr += ["hybrid"]
if hj.exists():
    hdr += ["hyb+jnt"]
print(f"  {'group':16}" + "".join(f"{h:>9}" for h in hdr))
for glabel, tasks in GROUPS:
    cells = []
    for n in COUNTS:
        if fixed[n]:
            cells.append(float(np.mean([SINGLE[t] - fixed[n][t] for t in tasks])))
    for r, _ in RATIOS:
        if ratio[r]:
            cells.append(float(np.mean([SINGLE[t] - ratio[r][t] for t in tasks])))
    if 300 in joint:
        cells.append(float(np.mean([SINGLE[t] - joint[300]["after"][t] for t in tasks])))
    if hyb_final:
        cells.append(float(np.mean([SINGLE[t] - hyb_final[t] for t in tasks])))
    if hj.exists():
        cells.append(float(np.mean([SINGLE[t] - hja["metrics_after"][t]["primary"] for t in tasks])))
    print(f"  {glabel.split(' —')[0]:16}" + "".join(f"{c:9.3f}" for c in cells))
