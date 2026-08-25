#!/usr/bin/env python3
"""Distance to the single-task ceiling across EVERY arm, by task size — one shared frame.

One panel per task-size group; x = labels actually replayed per old task (group mean of
min(resolved_amount, N_task), log scale) so fixed-count, ratio, hybrid and the end-of-run
joint retrain all land on one comparable axis; y = mean deficit to the task's TRUE ceiling
(single-task baseline from the L2 warm-restart control), i.e. single − final.

Reference lines (at-intro depends on the arm's patience — introductions early-stop lower
at patience 8):
  y = 0          — the single-task ceiling
  grey dashed    — at-intro cost of the FULL-BUDGET arms (p24/m150 family); anchors the
                   green zone (= run ends ABOVE its own introduction level)
  light dotted   — at-intro cost of the patience-8 arms (step-p8 / epoch-p8)

Excluded: material_type (accuracy metric), magnetic_susceptibility (degenerate single-task
baseline — 58 labels, see the L2 notes).

Outputs: replay_requirement_vs_size.png + a group × arm table on stdout.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES_EPOCH = HERE.parent / "results"
RES_STEP = HERE.parent.parent / "rikyu_replay_sweep" / "results"
OUT = HERE / "replay_requirement_vs_size.png"

N_TRAIN = {
    "density": 23678, "efermi": 23668, "final_energy": 23678, "total_magnetization": 23678,
    "volume": 23678, "dielectric_total": 3124, "dielectric_ionic": 3124, "dielectric_electronic": 3124,
    "magnetization": 1160, "curie": 6272, "neel": 3466, "kp": 3875,
    "zt": 3445, "power_factor": 3638, "thermal_conductivity": 4272,
    "electrical_resistivity": 5051, "dos_density": 7009, "seebeck": 8072,
    "formation_energy": 23180, "magnetic_moment": 851, "tc": 7207, "klat": 3863,
}  # fmt: skip  (material_type and magnetic_susceptibility excluded — see docstring)
COUNTS = [100, 200, 500, 1000, 1500, 2000, 2500]
RATIOS = [(0.10, "0p10"), (0.20, "0p20"), (0.30, "0p30"), (0.50, "0p50")]
BLUE, RED, GREEN, ORANGE, PURPLE = "#0077BB", "#CC3311", "#009E73", "#EE7733", "#AA3377"
INKC, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED})

GROUPS = (
    ("big tasks — ≥20k labels (6)", [t for t, n in N_TRAIN.items() if n >= 20000]),
    ("mid tasks — 3k–8.1k labels (14)", [t for t, n in N_TRAIN.items() if 3000 <= n < 20000]),
    ("small tasks — ≤1.2k labels (2)", [t for t, n in N_TRAIN.items() if n < 3000]),
)
# (label, results dir, filename pattern over COUNTS, color, marker, patience family)
ARMS = (
    ("step-p8", RES_STEP, "mt_n{n}.csv", BLUE, "o", "p8"),
    ("epoch-p8", RES_EPOCH, "mt_n{n}_epoch.csv", RED, "D", "p8"),
    ("epoch-p24", RES_EPOCH, "mt_n{n}_epoch_p24.csv", GREEN, "s", "full"),
    ("epoch-m150", RES_EPOCH, "mt_n{n}_epoch_m150.csv", ORANGE, "^", "full"),
)


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


def single_task_baseline() -> dict[str, float]:
    best: dict[str, float] = {}
    for r in csv.DictReader(open(RES_STEP / "warm_restart.csv")):
        v = fnum(r["primary"])
        if v is not None:
            t = r["task"]
            best[t] = v if t not in best else max(best[t], v)
    return best


SINGLE = single_task_baseline()

finals: dict[str, dict[int, dict[str, float]]] = {}
intro_acc: dict[str, dict[str, list[float]]] = {"p8": {t: [] for t in N_TRAIN}, "full": {t: [] for t in N_TRAIN}}
for arm, res_dir, pat, _, _, fam in ARMS:
    finals[arm] = {}
    for n in COUNTS:
        p = res_dir / pat.format(n=n)
        if not p.exists():
            continue
        final, intro = load(p)
        finals[arm][n] = final
        for t in N_TRAIN:
            if intro.get(t) is not None:
                intro_acc[fam][t].append(intro[t])

ratio_finals: dict[float, dict[str, float]] = {}
for r, tag in RATIOS:
    p = RES_EPOCH / f"mt_{tag}_epoch_m150.csv"
    if p.exists():
        final, intro = load(p)
        ratio_finals[r] = final
        for t in N_TRAIN:
            if intro.get(t) is not None:
                intro_acc["full"][t].append(intro[t])

hybrid_final = None
p = RES_EPOCH / "mt_hybrid_r03_f1500.csv"
if p.exists():
    hybrid_final, intro = load(p)
    for t in N_TRAIN:
        if intro.get(t) is not None:
            intro_acc["full"][t].append(intro[t])

joint_final = None
p = RES_EPOCH / "joint_retrain_m300.json"
if p.exists():
    joint_final = {t: m["primary"] for t, m in json.loads(p.read_text())["metrics_after"].items()}

INTRO = {fam: {t: float(np.mean(v)) for t, v in accs.items() if v} for fam, accs in intro_acc.items()}

fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.0), dpi=150, sharey=True)
print(f"{'group':34} {'series':>12} " + "deficit values")
for ax, (glabel, tasks) in zip(axes, GROUPS):
    cost_full = float(np.mean([SINGLE[t] - INTRO["full"][t] for t in tasks]))
    cost_p8 = float(np.mean([SINGLE[t] - INTRO["p8"][t] for t in tasks]))
    ax.axhline(0, color=MUTED, lw=1.6, zorder=2)
    ax.axhline(cost_full, color=MUTED, ls="--", lw=1.4, zorder=2)
    ax.axhline(cost_p8, color="#b0b6bf", ls=":", lw=1.4, zorder=2)
    ax.axhspan(0, cost_full, color="#f0fdf4", zorder=0)

    def gmean_labels(resolver):
        return float(np.mean([min(resolver(t), N_TRAIN[t]) for t in tasks]))

    for arm, _, _, color, marker, fam in ARMS:
        xs, ys = [], []
        for n in COUNTS:
            f = finals[arm].get(n)
            if not f:
                continue
            vals = [SINGLE[t] - f[t] for t in tasks if f.get(t) is not None]
            if vals:
                xs.append(gmean_labels(lambda t, n=n: n))
                ys.append(float(np.mean(vals)))
        lw, alpha = (1.6, 0.75) if fam == "p8" else (2.2, 1.0)
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, marker=marker, ms=5, mec="white", label=arm, zorder=3)
        print(f"{glabel:34} {arm:>12} " + " ".join(f"{y:6.3f}" for y in ys))
    xs, ys = [], []
    for r, _ in RATIOS:
        f = ratio_finals.get(r)
        if f:
            xs.append(gmean_labels(lambda t, r=r: r * N_TRAIN[t]))
            ys.append(float(np.mean([SINGLE[t] - f[t] for t in tasks if f.get(t) is not None])))
    ax.plot(xs, ys, color=PURPLE, lw=2.2, marker="v", ms=5.5, mec="white", label="ratio-m150 (r 0.1–0.5)", zorder=3)
    print(f"{glabel:34} {'ratio-m150':>12} " + " ".join(f"{y:6.3f}" for y in ys))
    if hybrid_final:
        x = gmean_labels(lambda t: max(1500, 0.3 * N_TRAIN[t]))
        y = float(np.mean([SINGLE[t] - hybrid_final[t] for t in tasks]))
        ax.plot([x], [y], marker="*", ms=15, color=INKC, mec="white", ls="none",
                label="hybrid max(1500, 0.3·N)", zorder=4)
        print(f"{glabel:34} {'hybrid':>12} {y:6.3f}")
    if joint_final:
        x = gmean_labels(lambda t: N_TRAIN[t])
        y = float(np.mean([SINGLE[t] - joint_final[t] for t in tasks]))
        ax.plot([x], [y], marker="D", ms=7, color=MUTED, mec="white", ls="none",
                label="no-replay → joint retrain @ end", zorder=3)
        print(f"{glabel:34} {'joint@end':>12} {y:6.3f}")
    ax.set_xscale("log")
    ax.set_title(glabel, fontsize=11)
    ax.minorticks_off()
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("deficit to the single-task ceiling:  single − final")
axes[0].text(0.02, 0.003, "single-task ceiling", transform=axes[0].get_yaxis_transform(),
             fontsize=8.5, color=MUTED)
cost_full0 = float(np.mean([SINGLE[t] - INTRO["full"][t] for t in GROUPS[0][1]]))
cost_p80 = float(np.mean([SINGLE[t] - INTRO["p8"][t] for t in GROUPS[0][1]]))
axes[0].text(0.98, cost_full0 + 0.004, "at-intro — p24/150 arms (green-zone top)",
             transform=axes[0].get_yaxis_transform(), fontsize=8.5, color=MUTED, ha="right")
axes[0].text(0.98, cost_p80 + 0.004, "at-intro — p8 arms (intros early-stop lower)",
             transform=axes[0].get_yaxis_transform(), fontsize=8.5, color="#9aa1ab", ha="right")
axes[2].annotate("amount ≥ N ⇒ 100% coverage:\nall such settings stack here",
                 (1006, 0.16), xytext=(400, 0.40), fontsize=8.5, color=MUTED,
                 arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.0})
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 0.99))
fig.suptitle("How close does replay get to the single-task ceiling? — every arm on one axis, by task size",
             fontsize=13, y=1.05)
fig.supxlabel("labels actually replayed per old task per step (group mean of min(amount, N), log)", fontsize=11)
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
fig.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}")
