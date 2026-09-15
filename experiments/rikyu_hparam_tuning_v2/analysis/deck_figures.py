#!/usr/bin/env python3
"""Figures for the 2026-09-15 deck — results/deck_20260915/*.png  (presentation-sized fonts)

Every number comes from the campaign's summary JSONs and the per-run prediction files pulled from
RIKYU; nothing is typed in here. Inputs:

  summary/baselines_mp2026.json          relabelled + added MP tasks on the 2026-09-11 dataset (5 seeds, scatter samples)
  summary/ceilings_adopted_v2.json       the tasks whose labels did not change (5 seeds, same rows)
  summary/classification_weights.json, material_type_weights.json, space_group_confirm.json,
  summary/space_group_perclass.json, space_group_classes.json     the class_weights = "none" runs
  summary/material_type_warmstart_none.json                       material_type transfer, class weights off in every arm
  summary/position_runs.json                                      material_type score by position (continual arm)
  summary/task_inventory_20260911.json, ft.json                   rows per task
  <preds>/stapred/<task>.parquet         stage_single seed-2025 predictions of the unchanged tasks
  <preds>/mtpred/stMn_2025.parquet, clfpred/stC_<task>_2025.parquet, sgpred/stW.parquet   "none" predictions

    uv run python analysis/deck_figures.py --preds <scratch dir> -o results/deck_20260915
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
S = HERE / "summary"
INK, MUT, BLUE, TEAL, PURPLE, ORANGE, GREY = "#1F2937", "#6B7280", "#0077BB", "#0E6E78", "#5B3F8C", "#A85A1A", "#7A858B"
GROUP_COLOURS = {"Materials Project": "#0077BB", "thermoelectric (starry)": "#EE7733", "NEMAD magnetic": "#009988",
                 "NEMAD superconductor": "#CC3311", "phonix-db": "#33BBEE", "quasicrystal (qa/starry)": "#EE3377"}
CLASSIFICATION = {"material_type": ["DAC", "DQC", "IAC", "IQC", "others"], "magnetic_ordering": ["AFM", "FM", "FiM", "NM"],
                  "is_metal": ["no", "yes"], "is_gap_direct": ["no", "yes"]}
ADDED = {"band_gap", "density_atomic", "magnetization_per_volume", "magnetization_per_fu", "reaction_energy", "cbm", "vbm", "bulk_modulus",
         "shear_modulus", "poisson_ratio", "universal_anisotropy", "refractive_index", "piezoelectric_max", "magnetic_ordering", "is_metal",
         "is_gap_direct", "space_group"}
plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
                     "axes.edgecolor": "#9CA3AF", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 200})


def load(name):
    return json.loads((S / name).read_text())


def source_of(task: str) -> str:
    if task in {"seebeck", "power_factor", "thermal_conductivity", "electrical_resistivity", "zt", "dos_density", "magnetic_susceptibility"}:
        return "thermoelectric (starry)"
    if task in {"magnetization", "curie", "neel", "magnetic_moment"}:
        return "NEMAD magnetic"
    if task == "tc":
        return "NEMAD superconductor"
    if task in {"kp", "klat"}:
        return "phonix-db"
    if task == "material_type":
        return "quasicrystal (qa/starry)"
    return "Materials Project"


def performance_table():
    base = {r["task"]: r for r in load("baselines_mp2026.json")["per_task"]}
    ceil = load("ceilings_adopted_v2.json")
    inv = {t["name"]: t for t in load("task_inventory_20260911.json")}
    ntr = {r["task"]: r["n_train"] for r in load("ft.json")["per_task"]}
    clf = load("classification_weights.json")["tasks"]
    mt = load("material_type_weights.json")["arms"]["none"]["runs"]
    sg = load("space_group_confirm.json")["arms"]["plain_kmd"]["runs"]
    rows = []
    for name, t in inv.items():
        n = t["n"] or {}
        n_train = n.get("train") or ntr.get(name); n_test = n.get("test"); kind = t["kind"]
        group = "added" if name in ADDED else "existing"
        if kind == "classification":
            if name == "material_type":
                vals = [r["macro_f1"] for r in mt]; acc = [r["accuracy"] for r in mt]
            elif name == "space_group":
                vals = [r["macro_f1"] for r in sg]; acc = [r["accuracy"] for r in sg]
            else:
                rs = clf[name]["arms"]["none"]["runs"]; vals = [r["macro_f1"] for r in rs]; acc = [r["accuracy"] for r in rs]
            rows.append(dict(task=name, kind=kind, group=group, source=source_of(name), metric="macro-F1", mean=st.fmean(vals), sd=st.stdev(vals), n=len(vals),
                             accuracy=st.fmean(acc), n_train=n_train, n_test=n_test, status="class weights off, 2026-09-11 dataset"))
        elif name in base and base[name].get("n_seeds"):
            b = base[name]
            rows.append(dict(task=name, kind=kind, group=group, source=source_of(name), metric="R²", mean=b["r2"]["mean"], sd=b["r2"]["sd"], n=b["n_seeds"], mae=b["mae"]["mean"],
                             n_train=n_train, n_test=b["n_test"], status=("relabelled" if b["group"] == "updated" else "added") + ", 2026-09-11 dataset"))
        elif name in ceil:
            c = ceil[name]
            rows.append(dict(task=name, kind=kind, group=group, source=source_of(name), metric="R²", mean=c["mean"], sd=c["sd"], n=c["n"], n_train=n_train, n_test=n_test,
                             status="labels unchanged; same rows, measured 2026-08"))
    return rows


def fig_overview(rows, out):
    reg = sorted([r for r in rows if r["metric"] == "R²"], key=lambda r: r["mean"])
    clf = sorted([r for r in rows if r["metric"] == "macro-F1"], key=lambda r: r["mean"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [3.4, 1]})
    for ax, data, xl in ((axes[0], reg, "single-task R² (test split, mean ± sd, 5 seeds)"), (axes[1], clf, "macro-F1 (5 seeds)")):
        y = np.arange(len(data))
        ax.barh(y, [r["mean"] for r in data], xerr=[r["sd"] for r in data], color=[GROUP_COLOURS[r["source"]] for r in data], height=0.72, error_kw={"lw": 1.2, "ecolor": INK})
        ax.set_yticks(y); ax.set_yticklabels([f"{r['task'].replace('_', ' ')}  ({r['n_train']:,})" for r in data], fontsize=12.5)
        ax.set_xlim(0, 1.02); ax.set_xlabel(xl); ax.grid(axis="x", color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
        for yi, r in zip(y, data):
            ax.text(min(r["mean"] + 0.012, 0.93), yi, f"{r['mean']:.2f}", va="center", fontsize=11.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOURS.values()]
    axes[0].legend(handles, list(GROUP_COLOURS), loc="lower right", fontsize=12.5, frameon=False, title="data source (n = training rows)", title_fontsize=12.5)
    fig.tight_layout(); fig.savefig(out / "overview_bar.png"); plt.close(fig)


def fig_r2_vs_n(rows, out):
    data = [r for r in rows if r["metric"] == "R²" and r["n_train"]]
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xscale("log"); ax.set_xlim(45, 70000); ax.set_ylim(0, 1.05)
    for r in data:
        ax.scatter(r["n_train"], r["mean"], s=110, color=GROUP_COLOURS[r["source"]], edgecolor="white", zorder=3)
    placed = []
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    for r in sorted(data, key=lambda r: (-r["n_train"], -r["mean"])):
        for dx, dy in ((8, 5), (8, -14), (8, 16), (-8, 5), (-8, -14), (8, -26), (8, 26), (-8, 16), (-8, -26)):
            t = ax.annotate(r["task"].replace("_", " "), (r["n_train"], r["mean"]), xytext=(dx, dy), textcoords="offset points", fontsize=12.5, color=INK,
                            ha="left" if dx > 0 else "right", va="center")
            bb = t.get_window_extent(renderer=rend).expanded(1.05, 1.15)
            if not any(bb.overlaps(b) for b in placed):
                placed.append(bb); break
            t.remove()
        else:
            ax.annotate(r["task"].replace("_", " "), (r["n_train"], r["mean"]), xytext=(8, 34), textcoords="offset points", fontsize=12.5, color=INK, arrowprops={"arrowstyle": "-", "color": "#9CA3AF", "lw": 0.8})
    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=c) for c in GROUP_COLOURS.values()]
    ax.legend(handles, list(GROUP_COLOURS), loc="lower left", frameon=False, fontsize=13)
    ax.set_xlabel("training rows (log scale)"); ax.set_ylabel("single-task R²"); ax.grid(color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "r2_vs_n.png"); plt.close(fig)


def scatter_panels(rows, preds: Path, out):
    """2 × 2 panels per figure; existing tasks and added tasks in separate figure series, one colour."""
    base = {r["task"]: r for r in load("baselines_mp2026.json")["per_task"]}
    panels = {"existing": [], "added": []}
    for r in rows:
        if r["metric"] != "R²":
            continue
        t = r["task"]
        if t in base and "scatter" in base[t]:
            sc = base[t]["scatter"]; panels[r["group"]].append((t, np.array(sc["true"]), np.array(sc["pred"]), r, "2,000 test rows · seed 2025"))
        elif (preds / "stapred" / f"{t}.parquet").exists():
            df = pd.read_parquet(preds / "stapred" / f"{t}.parquet")
            rng = np.random.default_rng(0); idx = rng.choice(len(df), min(2000, len(df)), replace=False)
            note = f"{min(2000, len(df)):,} of {len(df):,} test " + ("curve points · seed 2025" if "t" in df.columns else "rows · seed 2025")
            panels[r["group"]].append((t, df["true"].to_numpy()[idx], df["pred"].to_numpy()[idx], r, note))
    counts = {}
    for group, ps in panels.items():
        ps.sort(key=lambda p: -p[3]["mean"])
        per = 4; n_fig = (len(ps) + per - 1) // per; counts[group] = n_fig
        for k in range(n_fig):
            chunk = ps[k * per:(k + 1) * per]
            fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5)); axes = axes.ravel()
            for ax, (t, tr, pr, r, note) in zip(axes, chunk):
                lo, hi = np.percentile(np.r_[tr, pr], [0.5, 99.5]); pad = 0.05 * (hi - lo)
                ax.scatter(pr, tr, s=12, alpha=0.45, color=BLUE, edgecolor="none")
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUT, lw=1.2, ls="--")
                ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal")
                ax.set_title(f"{t.replace('_', ' ')}    R² = {r['mean']:.3f} ± {r['sd']:.3f}", fontsize=17, color=INK)
                ax.set_xlabel("predicted (normalised)", fontsize=14); ax.set_ylabel("observed (normalised)", fontsize=14)
                ax.text(0.03, 0.96, note, transform=ax.transAxes, fontsize=11, color=MUT, va="top")
            for ax in axes[len(chunk):]:
                ax.axis("off")
            fig.tight_layout(); fig.savefig(out / f"scatter_{group}_{k + 1}.png"); plt.close(fig)
    return counts


def confusion_from_pred(df, k):
    t, p = df["true"].to_numpy(int), df["pred"].to_numpy(int)
    return np.array([[int(((t == i) & (p == j)).sum()) for j in range(k)] for i in range(k)])


def draw_cm(ax, cm, labels, title, cell_fs=None):
    rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=13, rotation=45 if len(labels) > 5 else 0, ha="right" if len(labels) > 5 else "center"); ax.set_yticklabels(labels, fontsize=13)
    fs = cell_fs or (11 if len(labels) > 6 else 14)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{rn[i, j] * 100:.0f}%\n{cm[i, j]:,}", ha="center", va="center", fontsize=fs, color="white" if rn[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=16, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_confusions(rows, preds: Path, out):
    stats = {r["task"]: r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, task in zip(axes, ("magnetic_ordering", "is_metal", "is_gap_direct")):
        df = pd.read_parquet(preds / "clfpred" / f"stC_{task}_2025.parquet"); labels = CLASSIFICATION[task]
        r = stats[task]; draw_cm(ax, confusion_from_pred(df, len(labels)), labels, f"{task.replace('_', ' ')}\nmacro-F1 {r['mean']:.3f} · accuracy {r['accuracy']:.3f}")
    fig.tight_layout(); fig.savefig(out / "confusion_clf.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    df = pd.read_parquet(preds / "mtpred" / "stMn_2025.parquet"); r = stats["material_type"]
    draw_cm(ax, confusion_from_pred(df, 5), CLASSIFICATION["material_type"], f"material_type, trained alone (seed 2025)\nmacro-F1 {r['mean']:.3f} · accuracy {r['accuracy']:.3f}")
    fig.tight_layout(); fig.savefig(out / "confusion_material_type.png"); plt.close(fig)


def fig_space_group(rows, preds: Path, out):
    sg = load("space_group_classes.json"); classes = sg["classes"]; counts = sg["counts"]
    df = pd.read_parquet(preds / "sgpred" / "stW.parquet"); t, p = df["true"].to_numpy(int), df["pred"].to_numpy(int)
    order = sorted(range(len(classes)), key=lambda c: -counts[classes[c]])[:12]
    fold = {c: i for i, c in enumerate(order)}
    tf = np.array([fold.get(x, 12) for x in t]); pf = np.array([fold.get(x, 12) for x in p])
    cm = np.array([[int(((tf == i) & (pf == j)).sum()) for j in range(13)] for i in range(13)])
    fig, ax = plt.subplots(figsize=(11.5, 10.5))
    draw_cm(ax, cm, [classes[c] for c in order] + ["other (139)"], f"space group, 151 classes — top-1 accuracy {(t == p).mean():.3f} (seed 2025)", cell_fs=11.5)
    fig.tight_layout(); fig.savefig(out / "confusion_sg.png"); plt.close(fig)
    per = load("space_group_perclass.json")["arms"]["plain_kmd"]["per_class"]
    fig, ax = plt.subplots(figsize=(14, 7.5))
    names = list(per); xs = [per[n]["n_all"] for n in names]
    ys = [(2 * per[n]["precision"] * per[n]["recall"] / (per[n]["precision"] + per[n]["recall"]) if per[n]["precision"] and per[n]["recall"] else 0.0) for n in names]
    ax.scatter(xs, ys, s=60, color=PURPLE, alpha=0.75, edgecolor="none")
    for n, x, y in zip(names, xs, ys):
        if x >= 600 or (x >= 150 and y > 0.55):
            ax.annotate(n, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=13)
    ax.set_xscale("log"); ax.set_xlabel("rows in the space group (log scale)"); ax.set_ylabel("F1 of the group (seed 2025)"); ax.set_ylim(-0.02, 1.02)
    ax.grid(color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "sg_f1_vs_size.png"); plt.close(fig)


def _median_curve(runs, col):
    """Median (and IQR) over runs of a per-epoch column (1 = train loss, 2 = val loss), up to the shortest run."""
    L = min(len(r["curve"]) for r in runs)
    med, lo, hi = [], [], []
    for e in range(L):
        vals = [r["curve"][e][col] for r in runs if r["curve"][e][col] is not None]
        if not vals:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        med.append(float(np.median(vals))); lo.append(float(np.percentile(vals, 25))); hi.append(float(np.percentile(vals, 75)))
    return np.arange(L), np.array(med), np.array(lo), np.array(hi)


def fig_material_type(out):
    d = load("material_type_warmstart_none.json"); arms = d["arms"]
    alone = [r["macro_f1"] for r in arms["alone"]]; seen = [r["after"]["macro_f1"] for r in arms["warm_seen"]]; unseen = [r["after"]["macro_f1"] for r in arms["warm_unseen"]]
    seen_b = [r["before"]["macro_f1"] for r in arms["warm_seen"]]; unseen_b = [r["before"]["macro_f1"] for r in arms["warm_unseen"]]
    cols = {"alone": GREY, "seen": TEAL, "unseen": PURPLE}
    # 1 strip plot
    fig, ax = plt.subplots(figsize=(14, 7.5))
    for i, (vals, c) in enumerate(((alone, cols["alone"]), (seen, cols["seen"]), (unseen, cols["unseen"]))):
        rng = np.random.default_rng(i); x = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=140, color=c, alpha=0.85, edgecolor="white", zorder=3)
        m = st.fmean(vals); se = st.stdev(vals) / np.sqrt(len(vals))
        ax.hlines(m, i - 0.3, i + 0.3, color=c, lw=4.5, zorder=4); ax.errorbar(i + 0.36, m, yerr=2 * se, color=c, capsize=6, lw=2)
        ax.text(i + 0.42, m, f"{m:.3f}\n±{2 * se:.3f} (2×SE)", va="center", fontsize=15, color=INK)
    ax.axhline(st.fmean(alone), color=cols["alone"], ls="--", lw=1.2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([f"trained alone\n({len(alone)} seeds)", f"warm-start, encoder\nSAW material_type ({len(seen)} orderings)", f"warm-start, encoder\nNEVER saw it ({len(unseen)} orderings)"], fontsize=15)
    ax.set_ylabel(f"macro-F1 on the {d['test_rows']:,} test rows"); ax.set_xlim(-0.5, 2.95); ax.grid(axis="y", color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "mt_warmstart_strip.png"); plt.close(fig)
    # 2 before → after fine-tuning
    fig, ax = plt.subplots(figsize=(14, 7.2))
    for j, (b, a) in enumerate(zip(seen_b, seen)):
        ax.plot([0, 1], [b, a], color=cols["seen"], alpha=0.75, lw=2, marker="o", ms=8, label="encoder saw material_type (head from the pretraining step)" if j == 0 else None)
    for j, (b, a) in enumerate(zip(unseen_b, unseen)):
        ax.plot([0, 1], [b, a], color=cols["unseen"], alpha=0.9, lw=2, marker="s", ms=8, label="encoder never saw it (fresh head)" if j == 0 else None)
    ax.axhline(st.fmean(alone), color=cols["alone"], ls="--", lw=1.5, label=f"trained alone, mean {st.fmean(alone):.3f}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["before fine-tuning\n(pretrained encoder + head as loaded)", "after warm-start fine-tuning"], fontsize=15)
    ax.set_ylabel("macro-F1"); ax.set_ylim(-0.03, 0.95); ax.legend(loc="center left", frameon=False); ax.grid(axis="y", color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "mt_before_after.png"); plt.close(fig)
    # 3 position: box per position + median line (continual arm)
    pr = load("position_runs.json")["material_type"]
    pos = np.array([r["pos"] for r in pr]); at = np.array([r["at_train"] for r in pr], dtype=float)
    fig, ax = plt.subplots(figsize=(15, 7.2))
    data = [at[pos == p][~np.isnan(at[pos == p])] for p in range(1, 25)]
    ax.boxplot(data, positions=range(1, 25), widths=0.6, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#D5E8EA", "edgecolor": TEAL, "lw": 1.2}, medianprops={"color": TEAL, "lw": 2}, whiskerprops={"color": TEAL, "lw": 1}, capprops={"color": TEAL, "lw": 1})
    med = [float(np.median(v)) for v in data]
    ax.plot(range(1, 25), med, color=TEAL, lw=3, marker="o", ms=7, zorder=5, label="median of the 10 runs at that position")
    wa = load("material_type_weights.json")["arms"]["balanced"]["runs"]; wal = [r["macro_f1"] for r in wa]
    ax.axhspan(min(wal), max(wal), color=GREY, alpha=0.18); ax.axhline(st.fmean(wal), color=GREY, ls="--", lw=1.5, label=f"trained alone under the same (weighted) loss: mean {st.fmean(wal):.3f}, range of 5 seeds")
    ax.set_xlabel("number of tasks pretrained before material_type (its position in the 24-task sequence)"); ax.set_ylabel("macro-F1 at its own step")
    ax.set_xticks(range(1, 25)); ax.legend(loc="lower right", frameon=False); ax.grid(axis="y", color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "mt_position.png"); plt.close(fig)
    # 4 training and validation loss: alone vs warm-start (seen / never seen), median over runs with IQR
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    for ax, col, title in ((axes[0], 1, "training loss (material_type cross-entropy, per epoch)"), (axes[1], 2, "validation loss")):
        for tag, c, lab in (("alone", cols["alone"], f"trained alone ({len(alone)} seeds)"), ("warm_seen", cols["seen"], f"warm-start, encoder saw it ({len(seen)})"), ("warm_unseen", cols["unseen"], f"warm-start, never saw it ({len(unseen)})")):
            runs = [r for r in arms[tag] if r["curve"]]
            if not runs:
                continue
            x, m, lo, hi = _median_curve(runs, col)
            ax.plot(x, m, color=c, lw=2.5, label=lab); ax.fill_between(x, lo, hi, color=c, alpha=0.15)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_title(title, fontsize=16, color=INK); ax.grid(color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("loss (log scale)"); axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "mt_losses.png"); plt.close(fig)
    # numbers for the slides
    def summ(vals):
        return {"mean": st.fmean(vals), "sd": st.stdev(vals), "n": len(vals), "values": vals}
    loss_end = {}
    for tag in ("alone", "warm_seen", "warm_unseen"):
        runs = [r for r in arms[tag] if r["curve"]]
        tr = [next((c[1] for c in reversed(r["curve"]) if c[1] is not None), None) for r in runs]
        va = [next((c[2] for c in reversed(r["curve"]) if c[2] is not None), None) for r in runs]
        vmin = [min(c[2] for c in r["curve"] if c[2] is not None) for r in runs]
        v0 = [next((c[2] for c in r["curve"] if c[2] is not None), None) for r in runs]
        ep = [len(r["curve"]) for r in runs]
        loss_end[tag] = {"train_last": float(np.median([v for v in tr if v is not None])), "val_last": float(np.median([v for v in va if v is not None])),
                         "val_min": float(np.median(vmin)), "val_first": float(np.median([v for v in v0 if v is not None])), "epochs": float(np.median(ep))}
    return {"alone": summ(alone), "seen": summ(seen), "unseen": summ(unseen), "seen_before": seen_b, "unseen_before": unseen_b,
            "test_rows": d["test_rows"], "losses": loss_end}


def fig_added_transfer(out):
    d = load("added_transfer.json"); rows = [r for r in d["per_task"] if r.get("warm")]
    rows.sort(key=lambda r: r["delta"])
    fig, ax = plt.subplots(figsize=(15, 8.8))
    y = np.arange(len(rows))
    for yi, r in zip(y, rows):
        c = TEAL if r["verdict"] == "better" else ORANGE if r["verdict"] == "worse" else GREY
        ax.barh(yi, r["delta"], color=c, height=0.66, alpha=0.9); ax.errorbar(r["delta"], yi, xerr=2 * r["se_of_difference"], color=INK, capsize=4, lw=1.4)
    lo = min(r["delta"] - 2 * r["se_of_difference"] for r in rows); hi = max(r["delta"] + 2 * r["se_of_difference"] for r in rows)
    span = hi - lo; ax.set_xlim(lo - 0.06 * span, hi + 0.95 * span)
    xt = hi + 0.08 * span
    for yi, r in zip(y, rows):
        rel = f"{r['relative_pct']:+.1f} %" if abs(r["alone"]["mean"]) >= 0.1 else "n/a (alone ≈ 0)"
        ax.text(xt, yi, f"{r['alone']['mean']:.3f} → {r['warm']['mean']:.3f}   {rel}   {r['verdict']}", va="center", fontsize=13, color=INK, family="monospace")
    ax.text(xt, len(rows) - 0.25, "alone → warm-start   relative   verdict", fontsize=12, color=MUT, family="monospace")
    ax.axvline(0, color=INK, lw=1.2); ax.set_yticks(y); ax.set_yticklabels([f"{r['task'].replace('_', ' ')}  ({'macro-F1' if r['kind'] == 'classification' else 'R²'})" for r in rows], fontsize=13.5)
    ax.set_xlabel("warm-start − alone (R² or macro-F1; whisker = 2×SE of the difference)"); ax.grid(axis="x", color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (TEAL, GREY, ORANGE)]
    ax.legend(handles, [f"better ({d['counts']['better']})", f"unresolved ({d['counts']['unresolved']})", f"worse ({d['counts']['worse']})"], loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(out / "added_transfer.png"); plt.close(fig)


def fig_lowdata(out):
    """Learning curves: material_type in detail (3-class F1 from the prediction files) and the 17 added tasks as small multiples."""
    mt = load("lowdata_material_type.json")
    order = ["f10", "f25", "f50", "f100"]; xs = [10, 25, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    for ax, key, ttl in ((axes[0], "f1_3", "3-class macro-F1 (QC / AC / others) — the 2021 paper's task"), (axes[1], "f1_5", "5-class macro-F1 (swings on the 1-row DAC and 3-row DQC test classes)")):
        for arm, c, lab in (("alone", GREY, "trained alone"), ("warm", TEAL, "warm-start from a 23-task encoder that never saw material_type")):
            m = [st.fmean(r[key] for r in mt[f][arm]) for f in order]; sd = [st.stdev(r[key] for r in mt[f][arm]) if len(mt[f][arm]) > 1 else 0 for f in order]
            ax.errorbar(xs, m, yerr=sd, color=c, lw=2.5, marker="o", ms=9, capsize=5, label=lab)
            for x, f in zip(xs, order):
                for r in mt[f][arm]:
                    ax.scatter(x, r[key], s=30, color=c, alpha=0.5, zorder=3)
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x} %" for x in xs]); ax.set_xlabel("share of material_type training labels kept (test split unchanged)")
        ax.set_title(ttl, fontsize=15, color=INK); ax.grid(color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("macro-F1 on the 7,357 test rows"); axes[0].legend(frameon=False, loc="lower right", fontsize=12.5)
    fig.tight_layout(); fig.savefig(out / "lowdata_material_type.png"); plt.close(fig)
    # 17 added tasks: small multiples of alone vs warm across 5 / 10 / 25 / 50 / 100 %
    d = load("lowdata.json")["tasks"]
    tasks = [t for t in d if t != "material_type"]
    fig, axes = plt.subplots(3, 6, figsize=(20, 10.5)); axes = axes.ravel()
    for ax, t in zip(axes, tasks):
        pts = d[t]["points"]; fr = sorted(int(k) for k in pts)
        for arm, c in (("alone", GREY), ("warm", TEAL)):
            m, sd, xx = [], [], []
            for f in fr:
                v = [x for x in pts[str(f)][arm] if x is not None]
                if v:
                    xx.append(f); m.append(st.fmean(v)); sd.append(st.stdev(v) if len(v) > 1 else 0)
            ax.errorbar(xx, m, yerr=sd, color=c, lw=2, marker="o", ms=5, capsize=3, label="alone" if arm == "alone" else "warm-start")
        ax.set_xscale("log"); ax.set_xticks([5, 10, 25, 50, 100]); ax.set_xticklabels(["5", "10", "25", "50", "100"], fontsize=10)
        ax.set_title(f"{t.replace('_', ' ')}  ({'macro-F1' if d[t]['metric'] == 'macro_f1' else 'R²'})", fontsize=12.5, color=INK); ax.tick_params(axis="y", labelsize=10)
        ax.grid(color="#E5E7EB", lw=0.6); ax.set_axisbelow(True)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=10, loc="lower right")
    fig.text(0.5, 0.005, "share of the task's training labels kept (%), test split unchanged; mean ± sd over 3 subsample seeds (5 seeds / 5 encoders at 100 %)", ha="center", fontsize=13, color=MUT)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out / "lowdata_added.png"); plt.close(fig)
    # paired deltas at each fraction, all 17 tasks: share of tasks where warm > alone
    rows = []
    for t in tasks:
        pts = d[t]["points"]
        for f in sorted(int(k) for k in pts):
            al = [x for x in pts[str(f)]["alone"] if x is not None]; wm = [x for x in pts[str(f)]["warm"] if x is not None]
            if al and wm:
                rows.append((t, f, st.fmean(wm) - st.fmean(al), (st.stdev(wm) ** 2 / len(wm) + st.stdev(al) ** 2 / len(al)) ** 0.5 if len(wm) > 1 and len(al) > 1 else None))
    return rows


def fig_kmd_scale(out):
    d = {r["task"]: r for r in load("descriptor.json")["per_task"]}
    sgc = load("space_group_confirm.json")["arms"]
    base = {r["task"]: r for r in load("baselines_mp2026.json")["per_task"]}
    items = [("volume (cell)", d["volume"]["kmd"]["mean"], d["volume"]["classic"]["mean"], "R²"),
             ("volume per atom", base["density_atomic"]["r2"]["mean"], None, "R²"),
             ("total magnetisation (cell)", base["total_magnetization"]["r2"]["mean"], None, "R²"),
             ("space group (151 classes)", st.fmean(r["accuracy"] for r in sgc["plain_kmd"]["runs"]), st.fmean(r["accuracy"] for r in sgc["plain_classic"]["runs"]), "top-1")]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    y = np.arange(len(items))
    ax.barh(y - 0.19, [i[1] for i in items], height=0.36, color=GREY, label="KMD (atomic fractions, invertible)")
    ax.barh(y + 0.19, [i[2] if i[2] is not None else 0 for i in items], height=0.36, color=TEAL, label="XenonPy classic (carries the atom count)")
    for yi, it in zip(y, items):
        ax.text(it[1] + 0.012, yi - 0.19, f"{it[1]:.3f} {it[3]}", va="center", fontsize=14)
        if it[2] is not None:
            ax.text(it[2] + 0.012, yi + 0.19, f"{it[2]:.3f} {it[3]}", va="center", fontsize=14)
        else:
            ax.text(0.012, yi + 0.19, "not measured with XenonPy", va="center", fontsize=12.5, color=MUT)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items]); ax.set_xlim(0, 1.15); ax.invert_yaxis()
    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out / "kmd_scale.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True); ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    for old in a.out.glob("scatter_*.png"):
        old.unlink()
    rows = performance_table()
    (a.out / "performance_table.json").write_text(json.dumps(rows, indent=1))
    fig_overview(rows, a.out); fig_r2_vs_n(rows, a.out)
    counts = scatter_panels(rows, a.preds, a.out)
    fig_confusions(rows, a.preds, a.out); fig_space_group(rows, a.preds, a.out)
    mt = fig_material_type(a.out); (a.out / "material_type_numbers.json").write_text(json.dumps(mt, indent=1))
    fig_kmd_scale(a.out)
    if (S / "added_transfer.json").exists():
        fig_added_transfer(a.out)
    if (S / "lowdata.json").exists():
        rows = fig_lowdata(a.out); (a.out / "lowdata_deltas.json").write_text(json.dumps(rows))
    print(f"{len(rows)} tasks; scatter figures {counts}; figures in {a.out}")


if __name__ == "__main__":
    main()
