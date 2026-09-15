#!/usr/bin/env python3
"""Figures for the 2026-09-15 deck — results/deck_20260915/*.png

Every number comes from the campaign's summary JSONs and the per-run prediction files pulled from
RIKYU; nothing is typed in here. Inputs:

  summary/baselines_mp2026.json          relabelled + added MP tasks on the 2026-09-11 dataset (5 seeds, scatter samples)
  summary/ceilings_adopted_v2.json       the tasks whose labels did not change (5 seeds, same rows)
  summary/classification_weights.json, material_type_weights.json, space_group_confirm.json,
  summary/space_group_perclass.json, space_group_classes.json     the class_weights = "none" runs
  summary/material_type_warmstart_runs.json, position_runs.json  material_type transfer
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
INK, MUT = "#1F2937", "#6B7280"
GROUP_COLOURS = {"Materials Project": "#0077BB", "thermoelectric (starry)": "#EE7733", "NEMAD magnetic": "#009988",
                 "NEMAD superconductor": "#CC3311", "phonix-db": "#33BBEE", "quasicrystal (qa/starry)": "#EE3377"}
CLASSIFICATION = {"material_type": ["DAC", "DQC", "IAC", "IQC", "others"], "magnetic_ordering": ["AFM", "FM", "FiM", "NM"],
                  "is_metal": ["no", "yes"], "is_gap_direct": ["no", "yes"]}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#9CA3AF", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
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
    """One row per task: primary metric (mean, sd, n), kind, status, n_train/test."""
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
        n_train = n.get("train") or ntr.get(name)
        n_test = n.get("test")
        kind = t["kind"]
        if kind == "classification":
            if name == "material_type":
                vals = [r["macro_f1"] for r in mt]; acc = [r["accuracy"] for r in mt]
            elif name == "space_group":
                vals = [r["macro_f1"] for r in sg]; acc = [r["accuracy"] for r in sg]
            else:
                rs = clf[name]["arms"]["none"]["runs"]; vals = [r["macro_f1"] for r in rs]; acc = [r["accuracy"] for r in rs]
            rows.append(dict(task=name, kind=kind, source=source_of(name), metric="macro-F1", mean=st.fmean(vals), sd=st.stdev(vals), n=len(vals),
                             accuracy=st.fmean(acc), n_train=n_train, n_test=n_test, status="classification, class_weights = none (2026-09-11)"))
        elif name in base and base[name].get("n_seeds"):
            b = base[name]
            rows.append(dict(task=name, kind=kind, source=source_of(name), metric="R²", mean=b["r2"]["mean"], sd=b["r2"]["sd"], n=b["n_seeds"], mae=b["mae"]["mean"],
                             n_train=n_train, n_test=b["n_test"], status=("relabelled" if b["group"] == "updated" else "added") + " (2026-09-11 dataset)"))
        elif name in ceil:
            c = ceil[name]
            rows.append(dict(task=name, kind=kind, source=source_of(name), metric="R²", mean=c["mean"], sd=c["sd"], n=c["n"], n_train=n_train, n_test=n_test,
                             status="unchanged labels (same rows; measured 2026-08)"))
    return rows


def fig_overview(rows, out):
    reg = sorted([r for r in rows if r["metric"] == "R²"], key=lambda r: r["mean"])
    clf = sorted([r for r in rows if r["metric"] == "macro-F1"], key=lambda r: r["mean"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.2), gridspec_kw={"width_ratios": [3.2, 1]})
    for ax, data, xl in ((axes[0], reg, "single-task R² (test split, mean ± sd over 5 seeds)"), (axes[1], clf, "macro-F1 (5 seeds)")):
        y = np.arange(len(data))
        ax.barh(y, [r["mean"] for r in data], xerr=[r["sd"] for r in data], color=[GROUP_COLOURS[r["source"]] for r in data], height=0.7, error_kw={"lw": 1, "ecolor": INK})
        ax.set_yticks(y); ax.set_yticklabels([f"{r['task'].replace('_', ' ')}  (n={r['n_train']:,})" for r in data], fontsize=9)
        ax.set_xlim(0, 1.02); ax.set_xlabel(xl); ax.grid(axis="x", color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
        for yi, r in zip(y, data):
            ax.text(min(r["mean"] + 0.012, 0.96), yi, f"{r['mean']:.3f}", va="center", fontsize=8.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOURS.values()]
    axes[0].legend(handles, list(GROUP_COLOURS), loc="lower right", fontsize=9, frameon=False, title="data source", title_fontsize=9)
    fig.suptitle("Single-task performance of every task on the 2026-09-11 dataset (KMD descriptor, stage_single recipe)", fontsize=13, color=INK)
    fig.tight_layout(); fig.savefig(out / "overview_bar.png"); plt.close(fig)


def fig_r2_vs_n(rows, out):
    data = [r for r in rows if r["metric"] == "R²" and r["n_train"]]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xscale("log"); ax.set_xlim(45, 60000); ax.set_ylim(0, 1.04)
    for r in data:
        ax.scatter(r["n_train"], r["mean"], s=60, color=GROUP_COLOURS[r["source"]], edgecolor="white", zorder=3)
    # greedy label placement: try a few offsets, keep the first whose box does not overlap a placed label
    placed = []
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    for r in sorted(data, key=lambda r: (-r["n_train"], -r["mean"])):
        for dx, dy in ((6, 4), (6, -11), (6, 12), (-6, 4), (-6, -11), (6, -20), (6, 20), (-6, 12), (-6, -20)):
            t = ax.annotate(r["task"].replace("_", " "), (r["n_train"], r["mean"]), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK,
                            ha="left" if dx > 0 else "right", va="center")
            bb = t.get_window_extent(renderer=rend).expanded(1.05, 1.15)
            if not any(bb.overlaps(b) for b in placed):
                placed.append(bb); break
            t.remove()
        else:
            ax.annotate(r["task"].replace("_", " "), (r["n_train"], r["mean"]), xytext=(6, 28), textcoords="offset points", fontsize=8, color=INK, arrowprops={"arrowstyle": "-", "color": "#9CA3AF", "lw": 0.6})
    ax.set_xlabel("training rows (log scale)"); ax.set_ylabel("single-task R²")
    ax.grid(color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Single-task R² against training rows — data volume is not what separates the tasks", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(out / "r2_vs_n.png"); plt.close(fig)


def scatter_panels(rows, preds: Path, out):
    base = {r["task"]: r for r in load("baselines_mp2026.json")["per_task"]}
    panels = []
    for r in rows:
        if r["metric"] != "R²":
            continue
        t = r["task"]
        if t in base and "scatter" in base[t]:
            sc = base[t]["scatter"]; panels.append((t, np.array(sc["true"]), np.array(sc["pred"]), r, "2,000 test rows, seed 2025"))
        elif (preds / "stapred" / f"{t}.parquet").exists():
            df = pd.read_parquet(preds / "stapred" / f"{t}.parquet")
            rng = np.random.default_rng(0); idx = rng.choice(len(df), min(2000, len(df)), replace=False)
            note = f"{min(2000, len(df)):,} of {len(df):,} test " + ("points (curve samples), seed 2025" if "t" in df.columns else "rows, seed 2025")
            panels.append((t, df["true"].to_numpy()[idx], df["pred"].to_numpy()[idx], r, note))
    panels.sort(key=lambda p: -p[3]["mean"])
    per = 6; n_fig = (len(panels) + per - 1) // per
    for k in range(n_fig):
        chunk = panels[k * per:(k + 1) * per]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.6)); axes = axes.ravel()
        for ax, (t, tr, pr, r, note) in zip(axes, chunk):
            lo, hi = np.percentile(np.r_[tr, pr], [0.5, 99.5]); pad = 0.05 * (hi - lo)
            ax.scatter(pr, tr, s=7, alpha=0.45, color=GROUP_COLOURS[r["source"]], edgecolor="none")
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUT, lw=1, ls="--")
            ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal")
            ax.set_title(f"{t.replace('_', ' ')}   R² {r['mean']:.3f} ± {r['sd']:.3f}", fontsize=11, color=INK)
            ax.set_xlabel("predicted (normalised)", fontsize=9); ax.set_ylabel("observed (normalised)", fontsize=9)
            ax.text(0.03, 0.96, note, transform=ax.transAxes, fontsize=7.5, color=MUT, va="top")
        for ax in axes[len(chunk):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(out / f"scatter_{k + 1}.png"); plt.close(fig)
    return n_fig, [p[0] for p in panels]


def confusion_from_pred(df, k):
    t, p = df["true"].to_numpy(int), df["pred"].to_numpy(int)
    return np.array([[int(((t == i) & (p == j)).sum()) for j in range(k)] for i in range(k)])


def draw_cm(ax, cm, labels, title):
    rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right"); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{rn[i, j] * 100:.0f}%\n{cm[i, j]:,}", ha="center", va="center", fontsize=7.5 if len(labels) > 6 else 9, color="white" if rn[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=11, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_confusions(rows, preds: Path, out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    stats = {r["task"]: r for r in rows}
    for ax, (task, f) in zip(axes, (("material_type", preds / "mtpred" / "stMn_2025.parquet"), ("magnetic_ordering", preds / "clfpred" / "stC_magnetic_ordering_2025.parquet"),
                                    ("is_metal", preds / "clfpred" / "stC_is_metal_2025.parquet"), ("is_gap_direct", preds / "clfpred" / "stC_is_gap_direct_2025.parquet"))):
        df = pd.read_parquet(f); labels = CLASSIFICATION[task]; cm = confusion_from_pred(df, len(labels))
        r = stats[task]; draw_cm(ax, cm, labels, f"{task.replace('_', ' ')}\nmacro-F1 {r['mean']:.3f} · acc {r['accuracy']:.3f}")
    fig.suptitle("Classification heads, unweighted cross-entropy — confusion on the test split (seed 2025; cell = % of the true row, rows)", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(out / "confusion_clf.png"); plt.close(fig)


def fig_space_group(rows, preds: Path, out):
    sg = load("space_group_classes.json"); classes = sg["classes"]; counts = sg["counts"]
    df = pd.read_parquet(preds / "sgpred" / "stW.parquet"); t, p = df["true"].to_numpy(int), df["pred"].to_numpy(int)
    order = sorted(range(len(classes)), key=lambda c: -counts[classes[c]])[:12]
    fold = {c: i for i, c in enumerate(order)}
    tf = np.array([fold.get(x, 12) for x in t]); pf = np.array([fold.get(x, 12) for x in p])
    cm = np.array([[int(((tf == i) & (pf == j)).sum()) for j in range(13)] for i in range(13)])
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    draw_cm(ax, cm, [classes[c] for c in order] + ["other (139)"], f"space group, 151 classes — top-1 {(t == p).mean():.3f} (seed 2025); the twelve largest groups, the rest folded")
    fig.tight_layout(); fig.savefig(out / "confusion_sg.png"); plt.close(fig)
    # per-class F1 against class size
    per = load("space_group_perclass.json")["arms"]["plain_kmd"]["per_class"]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    xs = [v["n_all"] for v in per.values()]; ys = [(2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"]) if v["precision"] and v["recall"] else 0.0) for v in per.values()]
    ax.scatter(xs, ys, s=28, color="#5B3F8C", alpha=0.75, edgecolor="none")
    for name, v in per.items():
        if v["n_all"] >= 600 or (v["n_all"] >= 150 and ys[list(per).index(name)] > 0.55):
            ax.annotate(name, (v["n_all"], ys[list(per).index(name)]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("rows in the space group (log scale)"); ax.set_ylabel("F1 of the group (seed 2025)"); ax.set_ylim(-0.02, 1.02)
    ax.grid(color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Space group: per-class F1 against class size — groups with hundreds of examples are learned, groups with tens mostly are not", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(out / "sg_f1_vs_size.png"); plt.close(fig)


def fig_material_type(out):
    d = load("material_type_warmstart_runs.json"); arms = d["arms"]
    alone = [r["macro_f1"] for r in arms["alone"]]; seen = [r["after"]["macro_f1"] for r in arms["warm_seen"]]; unseen = [r["after"]["macro_f1"] for r in arms["warm_unseen"]]
    seen_b = [r["before"]["macro_f1"] for r in arms["warm_seen"]]; unseen_b = [r["before"]["macro_f1"] for r in arms["warm_unseen"]]
    cols = {"alone": "#7A858B", "seen": "#0E6E78", "unseen": "#5B3F8C"}
    # 1 strip plot
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, (name, vals, c) in enumerate((("trained alone\n(5 seeds)", alone, cols["alone"]), ("warm-start from an encoder\nthat SAW material_type\n(10 orderings)", seen, cols["seen"]),
                                          ("warm-start from an encoder\nthat NEVER saw it\n(3 orderings, fresh head)", unseen, cols["unseen"]))):
        rng = np.random.default_rng(i); x = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=70, color=c, alpha=0.85, edgecolor="white", zorder=3)
        m = st.fmean(vals); se = st.stdev(vals) / np.sqrt(len(vals))
        ax.hlines(m, i - 0.28, i + 0.28, color=c, lw=3.5, zorder=4); ax.errorbar(i + 0.33, m, yerr=2 * se, color=c, capsize=4, lw=1.5)
        ax.text(i + 0.38, m, f"{m:.3f}\n±{2 * se:.3f} (2×SE)", va="center", fontsize=9.5, color=INK)
    ax.axhline(st.fmean(alone), color=cols["alone"], ls="--", lw=1)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["trained alone\n(5 seeds)", "warm-start, encoder saw material_type\n(10 orderings)", "warm-start, encoder never saw it\n(3 orderings, fresh head)"], fontsize=10)
    ax.set_ylabel("macro-F1 on the 7,354 test rows"); ax.set_xlim(-0.5, 2.9); ax.grid(axis="y", color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("material_type: warm-start fine-tuning (encoder + head trained) against training alone — every dot is one run", fontsize=11.5, color=INK)
    fig.tight_layout(); fig.savefig(out / "mt_warmstart_strip.png"); plt.close(fig)
    # 2 before → after fine-tuning
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for j, (b, a) in enumerate(zip(seen_b, seen)):
        ax.plot([0, 1], [b, a], color=cols["seen"], alpha=0.7, lw=1.5, marker="o", ms=5, label="encoder saw material_type (head from the pretraining step)" if j == 0 else None)
    for j, (b, a) in enumerate(zip(unseen_b, unseen)):
        ax.plot([0, 1], [b, a], color=cols["unseen"], alpha=0.85, lw=1.5, marker="s", ms=5, label="encoder never saw it (fresh head)" if j == 0 else None)
    ax.axhline(st.fmean(alone), color=cols["alone"], ls="--", lw=1.2, label=f"trained alone, mean {st.fmean(alone):.3f}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["before fine-tuning\n(pretrained encoder + head as loaded)", "after warm-start fine-tuning\n(≤ 150 epochs, early stopping)"])
    ax.set_ylabel("macro-F1"); ax.set_ylim(-0.03, 0.8); ax.legend(loc="center left", fontsize=9, frameon=False); ax.grid(axis="y", color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("What the fine-tune does: a fresh head starts at 0.01 and ends where the pretrained head ends", fontsize=11.5, color=INK)
    fig.tight_layout(); fig.savefig(out / "mt_before_after.png"); plt.close(fig)
    # 3 position curve (continual arm): score at the step where material_type was trained, by position
    pr = load("position_runs.json")["material_type"]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    pos = np.array([r["pos"] for r in pr]); at = np.array([r["at_train"] for r in pr], dtype=float); end = np.array([r["at_end"] for r in pr], dtype=float)
    ax.scatter(pos, at, s=22, color=cols["seen"], alpha=0.55, label="at its own step (240 runs)")
    ax.scatter(pos, end, s=22, color="#A85A1A", alpha=0.45, marker="s", label="after the 24-task sequence ended")
    for arr, c in ((at, cols["seen"]), (end, "#A85A1A")):
        med = [np.nanmedian(arr[pos == p]) for p in range(1, 25)]; ax.plot(range(1, 25), med, color=c, lw=2)
    ax.axhspan(min(alone), max(alone), color=cols["alone"], alpha=0.15); ax.axhline(st.fmean(alone), color=cols["alone"], ls="--", lw=1, label="trained alone (mean, range of 5 seeds)")
    ax.set_xlabel("position of material_type in the 24-task pretraining sequence (number of tasks before it + 1)"); ax.set_ylabel("macro-F1"); ax.set_xticks(range(1, 25))
    ax.legend(fontsize=9, frameon=False, loc="lower right"); ax.grid(color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Continual pretraining arm: material_type improves the MORE tasks precede it — the only task that scales with pretraining breadth", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(out / "mt_position.png"); plt.close(fig)
    return dict(alone=alone, seen=seen, unseen=unseen, seen_before=seen_b, unseen_before=unseen_b, test_rows=d["test_rows"])


def fig_kmd_scale(out):
    d = {r["task"]: r for r in load("descriptor.json")["per_task"]}
    sgc = load("space_group_confirm.json")["arms"]
    base = {r["task"]: r for r in load("baselines_mp2026.json")["per_task"]}
    items = [("volume (cell)", d["volume"]["kmd"]["mean"], d["volume"]["classic"]["mean"], "R²"),
             ("volume per atom (density_atomic)", base["density_atomic"]["r2"]["mean"], None, "R²"),
             ("total magnetisation (cell)", base["total_magnetization"]["r2"]["mean"], None, "R²"),
             ("space group (151 classes)", st.fmean(r["accuracy"] for r in sgc["plain_kmd"]["runs"]), st.fmean(r["accuracy"] for r in sgc["plain_classic"]["runs"]), "top-1")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(items))
    ax.barh(y - 0.18, [i[1] for i in items], height=0.34, color="#7A858B", label="KMD (atomic fractions, invertible)")
    ax.barh(y + 0.18, [i[2] if i[2] is not None else 0 for i in items], height=0.34, color="#0E6E78", label="XenonPy classic (carries the atom count)")
    for yi, it in zip(y, items):
        ax.text(it[1] + 0.01, yi - 0.18, f"{it[1]:.3f} {it[3]}", va="center", fontsize=9)
        if it[2] is not None:
            ax.text(it[2] + 0.01, yi + 0.18, f"{it[2]:.3f} {it[3]}", va="center", fontsize=9)
        else:
            ax.text(0.01, yi + 0.18, "not measured with XenonPy", va="center", fontsize=8.5, color=MUT)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items]); ax.set_xlim(0, 1.12); ax.invert_yaxis()
    fig.legend(loc="lower center", ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.55, 0.0))
    ax.set_title("Labels that depend on the size of the cell: KMD against a descriptor that carries the atom count", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "kmd_scale.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True); ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = performance_table()
    (a.out / "performance_table.json").write_text(json.dumps(rows, indent=1))
    fig_overview(rows, a.out); fig_r2_vs_n(rows, a.out)
    n_fig, tasks = scatter_panels(rows, a.preds, a.out)
    fig_confusions(rows, a.preds, a.out); fig_space_group(rows, a.preds, a.out)
    mt = fig_material_type(a.out); (a.out / "material_type_numbers.json").write_text(json.dumps(mt, indent=1))
    fig_kmd_scale(a.out)
    print(f"{len(rows)} tasks in the table; {n_fig} scatter figures over {len(tasks)} regression panels; figures in {a.out}")


if __name__ == "__main__":
    main()
