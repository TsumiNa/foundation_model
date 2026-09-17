#!/usr/bin/env python3
"""Transfer gain against the number of labelled training rows, fixed counts — results/optimized/raw_v5/*

From summary/lowdata_n.json (100 / 300 / 1,000 / 3,000 / 10,000 labelled rows for every task, three
subsample seeds, alone = trained from scratch, warm = fine-tuned from a 24-task pretrained encoder that
never saw the task; the full-data point from the existing runs). Every task sits on the same x axis,
so the curves are comparable; the top axis of each panel gives the share of that task's labels.

Outputs: transfer_scaling_n_all.png (every task, one panel each), transfer_scaling_n_top.png (the tasks
with the clearest gain at 100–300 rows, at most six, each curve cut after the gain is gone),
transfer_scaling_n.json / .csv (every number; gain = warm − alone paired by subsample seed, with the
95 % confidence interval of the mean gain from the three seeds, and a paired t-test p-value).

    uv run python analysis/transfer_scaling_n_figure.py -o results/optimized/raw_v5 [--top 6]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deck_figures import BLUE, INK, MUT, TEAL, load  # noqa: E402

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}  # two-sided 95 % t quantiles by degrees of freedom
GONE = 0.01


def t_p_value(t, df):
    """Two-sided p-value of Student's t with df degrees of freedom (df = 2 for three seeds)."""
    if df == 2:
        return 1 - abs(t) / math.sqrt(t * t + 2)
    if df == 1:
        return 1 - 2 / math.pi * math.atan(abs(t))
    # normal approximation for larger df (only the full-data point, 5 seeds)
    return math.erfc(abs(t) / math.sqrt(2))


def stats(pt, full):
    al = [x for x in pt["alone"] if x is not None]; wm = [x for x in pt["warm"] if x is not None]
    if full:
        diff = st.fmean(wm) - st.fmean(al); se = (st.pvariance(al) / len(al) + st.pvariance(wm) / len(wm)) ** 0.5; df = len(al) + len(wm) - 2
    else:
        pairs = [w - a for a, w in zip(pt["alone"], pt["warm"]) if a is not None and w is not None]
        diff = st.fmean(pairs); se = (st.stdev(pairs) / len(pairs) ** 0.5) if len(pairs) > 1 else float("nan"); df = len(pairs) - 1
    ci = T95.get(df, 1.96) * se if se == se else float("nan")
    p = t_p_value(diff / se, df) if se and se == se and se > 0 else float("nan")
    verdict = "significant gain" if (diff - ci > 0 and diff > GONE) else ("significant loss" if (diff + ci < 0 and diff < -GONE) else ("gain, not significant" if diff > GONE else ("loss, not significant" if diff < -GONE else "no difference")))
    return dict(alone_mean=st.fmean(al), alone_sd=st.stdev(al) if len(al) > 1 else 0.0, n_alone=len(al), warm_mean=st.fmean(wm), warm_sd=st.stdev(wm) if len(wm) > 1 else 0.0, n_warm=len(wm),
                gain=diff, gain_ci95=ci, p_value=p, verdict=verdict)


def panel(ax, t, pts, metric, fs=11):
    x = [s["rows"] for s in pts]
    for key, col, lab in (("alone", BLUE, "trained from scratch"), ("warm", TEAL, "fine-tuned from the pretrained encoder")):
        ax.errorbar(x, [s[f"{key}_mean"] for s in pts], yerr=[s[f"{key}_sd"] for s in pts], color=col, lw=2.2, marker="o", ms=7, capsize=3, label=lab)
    ax.fill_between(x, [s["alone_mean"] for s in pts], [s["warm_mean"] for s in pts], color=TEAL, alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:,.0f}" for v in x], fontsize=11); ax.minorticks_off()
    top = ax.secondary_xaxis("top"); top.set_xticks(x); top.set_xticklabels([f"{100 * s['share']:.0f} %" if s["share"] >= 0.095 else f"{100 * s['share']:.1f} %" for s in pts], fontsize=11, color=MUT); top.tick_params(length=0)
    top.xaxis.set_minor_locator(NullLocator()); top.xaxis.set_minor_formatter(NullFormatter())
    for i, s in enumerate(pts):
        ha = "left" if i == 0 else ("right" if i == len(pts) - 1 else "center")
        ax.annotate(f"{s['gain']:+.3f}", (s["rows"], max(s["alone_mean"] + s["alone_sd"], s["warm_mean"] + s["warm_sd"])), xytext=(0, 6), textcoords="offset points", ha=ha, fontsize=fs,
                    color=(TEAL if s["verdict"] == "significant gain" else MUT))
    ax.set_title(t.replace("_", " "), color=INK, fontsize=16, pad=24); ax.set_ylabel("R²" if metric == "r2" else "macro-F1"); ax.set_xlabel("labelled training rows")
    lo = min(min(s["alone_mean"] - s["alone_sd"], s["warm_mean"] - s["warm_sd"]) for s in pts); hi = max(max(s["alone_mean"] + s["alone_sd"], s["warm_mean"] + s["warm_sd"]) for s in pts)
    pad = 0.18 * (hi - lo) + 0.01; ax.set_ylim(lo - pad, hi + 1.6 * pad); ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, required=True); ap.add_argument("--top", type=int, default=6); ap.add_argument("--exclude", default="material_type")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    low = load("lowdata_n.json")["tasks"]
    table = {}
    for t, d in low.items():
        pts = []
        for key, pt in d["points"].items():
            if not [x for x in pt["alone"] if x is not None] or not [x for x in pt["warm"] if x is not None]:
                continue
            s = stats(pt, key == "full"); s.update(point=key, rows=pt["rows"], share=pt["share"]); pts.append(s)
        pts.sort(key=lambda s: s["rows"])
        table[t] = dict(metric=d["metric"], train_rows=d["train_rows"], points=pts)
    (a.out / "transfer_scaling_n.json").write_text(json.dumps(table, indent=1) + "\n")
    with open(a.out / "transfer_scaling_n.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["task", "metric", "point", "labelled_rows", "share_of_task_labels", "from_scratch_mean", "from_scratch_sd", "n_seeds_scratch", "fine_tuned_mean", "fine_tuned_sd", "n_seeds_fine_tuned", "gain", "gain_ci95_halfwidth", "p_value_paired_t", "verdict"])
        for t, d in table.items():
            for s in d["points"]:
                w.writerow([t, d["metric"], s["point"], f"{s['rows']:.0f}", f"{s['share']:.4f}", f"{s['alone_mean']:.4f}", f"{s['alone_sd']:.4f}", s["n_alone"], f"{s['warm_mean']:.4f}", f"{s['warm_sd']:.4f}", s["n_warm"], f"{s['gain']:+.4f}", f"{s['gain_ci95']:.4f}", f"{s['p_value']:.3f}", s["verdict"]])
    # every task
    tasks = list(table); ncol = 6; nrow = (len(tasks) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.8 * nrow)); axes = axes.ravel()
    for ax, t in zip(axes, tasks):
        panel(ax, t, table[t]["points"], table[t]["metric"], fs=9)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Top axis: share of the task's labelled training rows · number above each point: fine-tuned minus from scratch (teal = significant at 95 %)", fontsize=14, color=MUT, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98)); fig.savefig(a.out / "transfer_scaling_n_all.png", dpi=150); plt.close(fig)
    # the clearest low-data gains
    def early_gain(t):
        """Largest gain at 100 or 300 rows among points where both models work (metric ≥ 0.2), so a gain over a failed from-scratch model does not count."""
        pts = [s for s in table[t]["points"] if s["point"] in ("100", "300") and s["alone_mean"] >= 0.2 and s["warm_mean"] >= 0.2]
        return max((s["gain"] for s in pts), default=-1.0)
    cand = [t for t in tasks if t not in a.exclude.split(",") and early_gain(t) > GONE]
    cand.sort(key=early_gain, reverse=True); top = cand[:a.top]
    if top:
        ncol = 3; nrow = (len(top) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(16.5, 4.8 * nrow), squeeze=False); axes = axes.ravel()
        for ax, t in zip(axes, top):
            pts = table[t]["points"]; peak = max(range(len(pts)), key=lambda i: pts[i]["gain"]); shown = []; gone = 0
            for i, s in enumerate(pts):  # keep the curve through the point where the gain is gone and one more, so the flat part is visible
                shown.append(s)
                gone += int(i > peak and s["gain"] <= GONE)
                if gone == 2:
                    break
            if any(s["verdict"] == "significant loss" for s in pts[peak + 1:]):  # a gain that turns into a loss is part of the story
                shown = pts
            panel(ax, t, shown, table[t]["metric"])
        for ax in axes[len(top):]:
            ax.axis("off")
        h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle("Top axis: share of the task's labelled training rows · number above each point: fine-tuned minus from scratch (teal = significant at 95 %)", fontsize=14, color=MUT, y=0.995)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97)); fig.savefig(a.out / "transfer_scaling_n_top.png", dpi=170); plt.close(fig)
    # summary table: how many tasks gain / lose at each label count
    summary = []
    for pt in ("100", "300", "1000", "3000", "10000", "full"):
        rows = [(t, s) for t, d in table.items() for s in d["points"] if s["point"] == pt]
        g = [t for t, s in rows if s["gain"] > GONE]; gs = [t for t, s in rows if s["verdict"] == "significant gain"]
        l = [t for t, s in rows if s["gain"] < -GONE]; ls = [t for t, s in rows if s["verdict"] == "significant loss"]
        summary.append(dict(labelled_rows=pt, tasks_measured=len(rows), improved=len(g), improved_significant=len(gs), worse=len(l), worse_significant=len(ls),
                            improved_tasks=g, improved_significant_tasks=gs, worse_tasks=l, worse_significant_tasks=ls))
    (a.out / "transfer_scaling_n_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    md = ["| labelled training rows | tasks measured | fine-tuning better (gain > 0.01) | of which significant (95 %) | fine-tuning worse (loss > 0.01) | of which significant (95 %) |", "|---|---|---|---|---|---|"]
    for r in summary:
        md.append(f"| {('full data' if r['labelled_rows'] == 'full' else format(int(r['labelled_rows']), ','))} | {r['tasks_measured']} | {r['improved']} | {r['improved_significant']}" + (f" ({', '.join(x.replace('_', ' ') for x in r['improved_significant_tasks'])})" if r['improved_significant_tasks'] else "") + f" | {r['worse']} | {r['worse_significant']}" + (f" ({', '.join(x.replace('_', ' ') for x in r['worse_significant_tasks'])})" if r['worse_significant_tasks'] else "") + " |")
    (a.out / "transfer_scaling_n_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("top:", top)
    for t, d in table.items():
        print(f"{t:26s} " + "  ".join(f"{s['point']:>5s}: {s['gain']:+.3f}±{s['gain_ci95']:.3f} p={s['p_value']:.2f} ({100 * s['share']:.1f} %)" for s in d["points"]))
    print(a.out)


if __name__ == "__main__":
    main()
