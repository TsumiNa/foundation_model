#!/usr/bin/env python3
"""Transfer gain against target-task data volume — results/optimized/raw_v4/transfer_scaling*

From the low-data learning curves (summary/lowdata.json: 5 / 10 / 25 / 50 % of each task's training
labels, three subsample seeds, alone = trained from scratch, warm = fine-tuned from a 24-task
pretrained encoder that never saw the task; the 100 % point is the full-data run pair) pick the tasks
whose warm-start gain at 5–10 % of the labels is clearest, and draw the gain fading as the labels grow.
Training-row counts per fraction come from the low-data manifest (summary/lowdata_manifest.json, seed 0).

Outputs: transfer_scaling_6tasks.png (curves cut at the point where the gain is gone),
transfer_scaling_6tasks_full.png (all five points), transfer_scaling.json / .csv (every number).

    uv run python analysis/transfer_scaling_figure.py -o results/optimized/raw_v4
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deck_figures import BLUE, INK, MUT, TEAL, load  # noqa: E402  (also sets the presentation rcParams)

# ranked by the paired gain at 5 % and 10 % of the labels against its 2×SE (analysis printed by lowdata.py / this script)
TASKS = ["universal_anisotropy", "shear_modulus", "vbm", "refractive_index", "bulk_modulus", "cbm"]
FRACS = [5, 10, 25, 50, 100]
GONE = 0.01  # the gain counts as gone when the paired difference drops below this


def stats(pt, f):
    al = [x for x in pt["alone"] if x is not None]; wm = [x for x in pt["warm"] if x is not None]
    if f == 100:
        diff = st.fmean(wm) - st.fmean(al); se = (st.pvariance(al) / len(al) + st.pvariance(wm) / len(wm)) ** 0.5
    else:
        pairs = [w - a for a, w in zip(pt["alone"], pt["warm"]) if a is not None and w is not None]
        diff = st.fmean(pairs); se = st.stdev(pairs) / len(pairs) ** 0.5
    return dict(alone_mean=st.fmean(al), alone_sd=st.stdev(al) if len(al) > 1 else 0.0, n_alone=len(al),
                warm_mean=st.fmean(wm), warm_sd=st.stdev(wm) if len(wm) > 1 else 0.0, n_warm=len(wm),
                gain=diff, gain_2se=2 * se, verdict="better" if diff - 2 * se > GONE else ("worse" if diff + 2 * se < -GONE else "unresolved"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, required=True); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    low = load("lowdata.json")["tasks"]; man = load("lowdata_manifest.json")
    kept = {}
    for f in man["files"]:
        if f["seed"] != 0:
            continue
        for t, v in f["kept"].items():
            kept.setdefault(t, {})[int(round(f["fraction"] * 100))] = v["train_kept"]; kept[t][100] = v["train_before"]
    table = {}
    for t in TASKS:
        pts = []
        for f in FRACS:
            s = stats(low[t]["points"][str(f)], f); s.update(fraction_pct=f, train_rows=kept[t][f]); pts.append(s)
        peak = max(range(len(pts)), key=lambda i: pts[i]["gain"]); shown = []
        for i, s in enumerate(pts):  # keep every point up to and including the first one after the peak where the gain is gone
            shown.append(s["fraction_pct"])
            if i > peak and s["gain"] <= GONE:
                break
        if pts[-1]["verdict"] == "worse":  # a gain that turns into a loss at full data is part of the story: show the whole curve
            shown = [s["fraction_pct"] for s in pts]
        table[t] = dict(metric=low[t]["metric"], points=pts, shown_fractions=shown)
    (a.out / "transfer_scaling.json").write_text(json.dumps({"tasks": table, "rule": f"curve cut after the first fraction past the peak gain whose paired gain ≤ {GONE}; shown in full when the full-data verdict is worse", "arms": {
        "alone": "single-task model trained from scratch on the reduced labels (probe6_mp2026_lowdata.toml; stL_<task>_f<pct>_s<seed>)",
        "warm": "fine-tune from a 24-task pretrained encoder that never saw the task, encoder and fresh head trained (ft_lowdata_mp2026.toml; ftL_<task>_f<pct>_s<seed>)",
        "100": "full-data runs: 5 single-task seeds (baselines_mp2026.json) and the warm-start runs of added_transfer.json"}}, indent=1) + "\n")
    with open(a.out / "transfer_scaling.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["task", "metric", "fraction_pct", "train_rows", "alone_mean", "alone_sd", "n_alone", "warm_mean", "warm_sd", "n_warm", "gain", "gain_2se", "verdict", "shown_in_cut_figure"])
        for t, d in table.items():
            for s in d["points"]:
                w.writerow([t, d["metric"], s["fraction_pct"], s["train_rows"], f"{s['alone_mean']:.4f}", f"{s['alone_sd']:.4f}", s["n_alone"], f"{s['warm_mean']:.4f}", f"{s['warm_sd']:.4f}", s["n_warm"], f"{s['gain']:+.4f}", f"{s['gain_2se']:.4f}", s["verdict"], s["fraction_pct"] in d["shown_fractions"]])
    for cut, name in ((True, "transfer_scaling_6tasks.png"), (False, "transfer_scaling_6tasks_full.png")):
        fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6)); axes = axes.ravel()
        for ax, t in zip(axes, TASKS):
            d = table[t]; pts = [s for s in d["points"] if (not cut) or s["fraction_pct"] in d["shown_fractions"]]
            x = [s["train_rows"] for s in pts]
            for key, col, lab in (("alone", BLUE, "trained alone"), ("warm", TEAL, "warm-start from the 24-task encoder")):
                ax.errorbar(x, [s[f"{key}_mean"] for s in pts], yerr=[s[f"{key}_sd"] for s in pts], color=col, lw=2.2, marker="o", ms=7, capsize=3, label=lab)
            ax.fill_between(x, [s["alone_mean"] for s in pts], [s["warm_mean"] for s in pts], color=TEAL, alpha=0.12, lw=0)
            ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:,}" for v in x], fontsize=12)
            ax.minorticks_off()
            top = ax.secondary_xaxis("top"); top.set_xticks(x); top.set_xticklabels([f"{s['fraction_pct']} %" for s in pts], fontsize=12, color=MUT); top.tick_params(length=0)
            top.xaxis.set_minor_locator(NullLocator()); top.xaxis.set_minor_formatter(NullFormatter())
            for i, s in enumerate(pts):
                ha = "left" if i == 0 else ("right" if i == len(pts) - 1 else "center")
                ax.annotate(f"{s['gain']:+.3f}", (s["train_rows"], max(s["alone_mean"] + s["alone_sd"], s["warm_mean"] + s["warm_sd"])), xytext=(0, 6), textcoords="offset points",
                            ha=ha, fontsize=11.5, color=(TEAL if s["verdict"] == "better" else MUT))
            ax.set_title(t.replace("_", " "), color=INK, fontsize=17, pad=26)
            ax.set_ylabel("R²" if d["metric"] == "r2" else "macro-F1"); ax.set_xlabel("training rows with a label")
            lo = min(min(s["alone_mean"] - s["alone_sd"], s["warm_mean"] - s["warm_sd"]) for s in pts); hi = max(max(s["alone_mean"] + s["alone_sd"], s["warm_mean"] + s["warm_sd"]) for s in pts)
            pad = 0.18 * (hi - lo) + 0.01; ax.set_ylim(lo - pad, hi + 1.6 * pad); ax.grid(alpha=0.25)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle("Top axis: share of the task's training labels · number above each point: warm-start minus alone (teal = separated at 2×SE)", fontsize=14, color=MUT, y=0.995)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97)); fig.savefig(a.out / name, dpi=170); plt.close(fig)
    for t, d in table.items():
        print(f"{t:22s} " + "  ".join(f"{s['fraction_pct']:>3d}%: {s['gain']:+.3f}±{s['gain_2se']:.3f} {s['verdict'][:5]:5s} n={s['train_rows']:>6,}" for s in d["points"]) + f"   shown: {d['shown_fractions']}")
    print(a.out)


if __name__ == "__main__":
    main()
