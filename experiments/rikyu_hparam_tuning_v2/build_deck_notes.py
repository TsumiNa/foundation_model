#!/usr/bin/env python3
"""Per-slide design brief + raw data for results/DECK_20260915.pptx — results/DECK_20260915_NOTES.md

Reads the built deck (titles, subtitles, bullet text, tables) so the brief matches the slides exactly,
adds a design note per slide (purpose, key message, layout, speaker cue) and the raw data behind every
figure from the same summary JSONs the figures were drawn from.

    uv run --with python-pptx python experiments/rikyu_hparam_tuning_v2/build_deck_notes.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
from pptx import Presentation

HERE = Path(__file__).resolve().parent
S = HERE / "summary"
FIG = HERE / "results" / "deck_20260915"
DECK = HERE / "results" / "DECK_20260915.pptx"
OUT = HERE / "results" / "DECK_20260915_NOTES.md"


def load(name):
    return json.loads((S / name).read_text())


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c).replace("\n", " ").replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ design notes, matched by title prefix (first match wins)
DESIGN = [
    ("Composition-only foundation model", dict(purpose="Title slide.", message="A composition-only materials foundation model: what it is trained on, whether it transfers, what data comes next, how it will be released.", layout="Two-line title, one-line status subtitle, four-part agenda line.", cue="State the four parts in one breath; the deck is long, the audience should know they can skip.")),
    ("Part 1 —", dict(purpose="Section divider.", message="Part 1: the task inventory and how each task performs alone.", layout="Full-bleed teal background, white title and one-line subtitle.", cue="—")),
    ("The 2026-09-11 dataset", dict(purpose="Where the rows come from.", message="Five sources, ~49k compositions, 41 task columns; the model only ever sees the composition string.", layout="Six-row source table (source · compositions · what it carries · tasks) + two grey footnote lines.", cue="Stress that every task is a column of the same composition table; missing values are allowed.")),
    ("Tasks 1/5", dict(purpose="Task inventory, Materials Project energies and structure.", message="What each label means, its unit, its kind and how many rows train/test it.", layout="9-row table: task · what it is · unit · kind · train rows · test rows.", cue="Do not read the table; point at final_energy/formation_energy (largest) and volume (the KMD-limited one).")),
    ("Tasks 2/5", dict(purpose="Task inventory, Materials Project electronic and magnetic.", message="Same structure.", layout="9-row table.", cue="Note the four classification heads (is_metal, is_gap_direct, magnetic_ordering) and the per-cell vs intensive magnetisation columns.")),
    ("Tasks 3/5", dict(purpose="Task inventory, Materials Project dielectric, elastic, symmetry.", message="Same structure.", layout="8-row table.", cue="Space group is a 151-class head; its own slide explains the count.")),
    ("Tasks 4/5", dict(purpose="Task inventory, thermoelectric curve tasks.", message="Seven experimental curve tasks: one curve per composition, learned as kernel regression over a coordinate t (temperature or energy).", layout="7-row table.", cue="These are the tasks where the model predicts a whole curve, not a scalar.")),
    ("Tasks 5/5", dict(purpose="Task inventory, NEMAD, phonon transport, quasicrystals.", message="Text-mined experimental magnetic and superconductor data, first-principles phonon transport, and the quasicrystal classification (material_type).", layout="8-row table.", cue="material_type is the transfer example of Part 2: 5 classes, 99 % “others”.")),
    ("Single-task performance, all 41 tasks", dict(purpose="One-glance overview of every task's baseline.", message="Most tasks are at R² 0.7–1.0 alone; the weak ones are either cell-scale labels or low-signal / noisy experimental labels, not the small ones.", layout="Figure: horizontal bars (R² for regression / curve tasks, macro-F1 for classification), colour = data source, whiskers = sd over 5 seeds, training rows in brackets.", cue="Point at the bottom of the chart: piezoelectric, magnetic susceptibility, electrical resistivity, poisson ratio, reaction energy, universal anisotropy — the tasks to treat with care.")),
    ("Single-task R² against training rows", dict(purpose="Show that data volume does not order the tasks.", message="Tasks with 20k rows span R² 0.3–1.0; tasks with 1k rows can reach 0.8. What limits a task is its label, not its row count.", layout="Figure: scatter, x = training rows (log), y = R², colour = source, every point labelled.", cue="This is the argument for the descriptor slide later: volume sits at 0.6 with 23k rows.")),
    ("Single-task metrics 1/6", dict(purpose="Numbers behind the overview, part 1 (regression, ranked by R²).", message="Exact R² and MAE per task with seed spread.", layout="8-row table: task · metric · mean ± sd · MAE · test rows · measured on.", cue="Reference slide; skip in a talk.")),
    ("Single-task metrics 2/6", dict(purpose="Numbers behind the overview, part 2.", message="—", layout="8-row table.", cue="Reference slide.")),
    ("Single-task metrics 3/6", dict(purpose="Numbers behind the overview, part 3.", message="—", layout="8-row table.", cue="Reference slide.")),
    ("Single-task metrics 4/6", dict(purpose="Numbers behind the overview, part 4.", message="—", layout="8-row table.", cue="Reference slide.")),
    ("Single-task metrics 5/6", dict(purpose="Numbers behind the overview, part 5 (the weakest regression tasks).", message="—", layout="4-row table.", cue="Reference slide.")),
    ("Single-task metrics 6/6", dict(purpose="Numbers behind the overview, classification heads.", message="macro-F1 and accuracy for the five classification heads.", layout="5-row table.", cue="Reference slide.")),
    ("Observed vs predicted — existing tasks", dict(purpose="Scatter plots of the tasks that were in the model before 2026-09-11.", message="Where the R² comes from: tight diagonals for energies and density, fans for the magnetic and curve tasks.", layout="2 × 2 panels, one seed (2025), 2,000 test points per panel, dashed y = x, title carries R² ± sd; single colour.", cue="Pick one panel per slide to comment on; the rest are for the reader.")),
    ("Observed vs predicted — tasks added", dict(purpose="Scatter plots of the 13 regression tasks added on 2026-09-11.", message="Band gap, CBM/VBM, bulk modulus, refractive index are solid new tasks; poisson ratio, universal anisotropy, reaction energy, piezoelectric carry little composition signal.", layout="2 × 2 panels as before.", cue="Say explicitly which added tasks earn a place in the multi-task set and which do not.")),
    ("Added classification heads: confusion matrices", dict(purpose="Show the three small added classification heads.", message="is_metal is nearly solved (0.93); magnetic ordering separates non-magnetic from magnetic but confuses FM / FiM / AFM; direct-gap detection is weak (0.63).", layout="Three row-normalised confusion matrices, cell = % of the true row + count.", cue="From composition alone the model knows which elements carry moments, not how they order.")),
    ("Space group: why 151 classes", dict(purpose="Pre-empt the question “why not 230”.", message="230 groups exist; 213 occur in the stable set; 151 have ≥ 10 rows and a row in both train and test; the other 62 groups (274 rows, 0.8 %) are treated as missing labels — a data-availability count, not a physics choice.", layout="Five bullets, no figure.", cue="Say the number 274 rows out loud; it answers the objection.")),
    ("Space group: confusion over the twelve largest groups", dict(purpose="What the space-group head gets right and wrong.", message="Cubic and hexagonal groups (Fm-3m, I4/mmm, P-62m, Pm-3m) are recognised; low-symmetry monoclinic and orthorhombic groups (P2₁/c, C2/c, Pnma) only partly.", layout="13 × 13 confusion matrix, top-12 groups + “other (139)”, cell = % of the true row + count.", cue="Top-1 accuracy 0.465 over 151 classes; the comparison with the ShotgunCSP classifier (0.60, different descriptor and network) is on the space-group page.")),
    ("Space group: per-class F1 against class size", dict(purpose="Show that the tail sets the ceiling.", message="Groups with hundreds of examples are learned; groups with tens mostly are not. The imbalance is the physics of the crystal world.", layout="Scatter: x = rows in the group (log), y = F1 of the group; largest groups labelled.", cue="This is why macro-F1 is low (0.31) while top-1 is 0.47.")),
    ("Where the descriptor limits the label", dict(purpose="State the one known limitation of the input representation.", message="KMD works on atomic fractions, so it cannot see cell size; volume (0.62 vs 0.997 with a descriptor that carries the atom count), total magnetisation and space group are capped by that. Decision: keep KMD for its invertibility, use the intensive forms.", layout="Grouped horizontal bars (KMD vs XenonPy classic) for four labels + one grey decision line.", cue="Volume per atom at 0.979 is the practical answer.")),
    ("Part 2 —", dict(purpose="Section divider.", message="Part 2: does a pretrained encoder transfer? material_type as the worked example.", layout="Teal divider.", cue="—")),
    ("material_type: the task, trained alone", dict(purpose="Introduce the task before the transfer result.", message="Five classes, 99 % “others”; the rare classes are approximant crystals (DAC, IAC) and quasicrystals (DQC, IQC). Trained alone the head finds the rare classes but files many ordinary materials as rare.", layout="One 5 × 5 confusion matrix (seed 2025), cell = % of the true row + count.", cue="Point at the “others” row leaking into IAC / IQC — that is what transfer will fix.")),
    ("material_type transfer: the setup", dict(purpose="Define the three arms.", message="Alone (5 seeds) vs warm-start from a 24-task encoder that saw material_type (10 orderings) vs warm-start from a 23-task encoder that never saw it (3 orderings); encoder + head trained in the fine-tune; same test rows.", layout="3-row arm table + three grey lines of protocol.", cue="The third arm is the interesting one: it asks whether exposure during pretraining matters.")),
    ("material_type transfer: the numbers", dict(purpose="The headline transfer result.", message="Warm-start beats training alone by 22 % macro-F1 (0.571 → 0.696), and an encoder that never saw the task does the same (0.694). The gain is the 23-task representation, not exposure.", layout="3-row numbers table (mean ± sd, n, Δ vs alone, 2×SE, verdict) + three bullets.", cue="Consequence for the pipeline: pretrain once, warm-start each new task; skip the continual step.")),
    ("material_type: every run of the three arms", dict(purpose="Show the spread behind the means.", message="Every warm-started run, seen or never-seen, sits above every alone run.", layout="Strip plot: one dot per run, bar = mean, whisker = 2×SE, dashed = alone mean.", cue="No overlap between the alone cloud and either warm-start cloud.")),
    ("material_type: before and after the fine-tune", dict(purpose="Show what the fine-tune does to each starting point.", message="The seen encoder starts at 0.51–0.74 (its pretrained head) and ends at 0.63–0.74; the never-seen encoder starts at 0.01 (random head) and ends in the same band.", layout="Paired lines before → after per run, dashed alone mean.", cue="The fresh-head arm is the proof that the encoder, not the head, carries the gain.")),
    ("material_type: training and validation loss, alone vs warm-start", dict(purpose="Convergence behaviour.", message="Warm-started runs start lower, fit the training set further and stop at half the epochs.", layout="Two panels (training loss, validation loss) on log scale, median over runs with inter-quartile band, three arms.", cue="Read together with the next slide; the loss alone does not show the macro-F1 gain, the per-class figure does.")),
    ("material_type: where the gain comes from", dict(purpose="Decompose the gain by class.", message="Recall of the rare classes is unchanged; their precision roughly doubles — the pretrained encoder stops ordinary materials being mistaken for approximants and quasicrystals.", layout="Two grouped bar panels (recall, precision) × 5 classes × 3 arms, values printed.", cue="IAC precision 0.39 → 0.54, IQC 0.51 → 0.62; 75 vs 42 rows a run filed as IAC for 24 real ones.")),
    ("Why material_type gains from the pretrained encoder", dict(purpose="Explanation and a minimal theoretical frame.", message="Fewer false alarms, faster convergence; 23 regression tasks teach what an ordinary composition looks like, so the head only draws a tight boundary around the 99 % class; the shared-representation bound says the representation term is already paid by pretraining.", layout="Five bullets.", cue="Keep the theory to one sentence in the talk.")),
    ("material_type against pretraining breadth", dict(purpose="The scaling view: does more pretraining help?", message="In the 240-run continual stage, the later material_type appears in the sequence (the more tasks pretrained before it), the better its score — the only task with that behaviour.", layout="Box per position (runs at that position) + median line + alone mean/range band.", cue="Positions 1–2 sit on the alone band; from position 3 on the median is above it.")),
    ("Part 3 —", dict(purpose="Section divider.", message="Part 3: the data plan.", layout="Teal divider.", cue="—")),
    ("Data plan — catalysts, high-entropy alloys, MOFs", dict(purpose="Candidate datasets, ranked, with a status.", message="Three IMPORT NOW sets (Catalysis-Hub alloy adsorption energies, OCM high-throughput curves, QMOF band gaps) plus the KKR-CPA quaternary HEA set as the HEA priority; the rest are still being evaluated.", layout="One table with three teal section rows; columns dataset · size · labels / consistency · fit · status.", cue="Sizes are as published; nothing has been downloaded; the split IMPORT NOW / EVALUATING is a proposal.")),
    ("Data plan — the two decisions", dict(purpose="What must be decided before importing.", message="Catalyst supports/promoters: in the composition string or held fixed; MOFs: the organic part dominates atomic fractions, an auxiliary node label may be needed; HEAs fit directly.", layout="Four bullets.", cue="Ask the room for the two decisions.")),
    ("Part 4 —", dict(purpose="Section divider.", message="Part 4: release and LLM access.", layout="Teal divider.", cue="—")),
    ("Roadmap", dict(purpose="The release plan as a flow.", message="Model library → online prediction API / local download; web app for preview → upload own data → AI-assisted fine-tune → inspect → download or host; the package ships preset skills an AI agent uses locally or via API.", layout="Three lanes of rounded boxes with arrows (editable shapes): Model base / Web app / Package + AI.", cue="Proposal; nothing is released yet.")),
    ("Status and sources", dict(purpose="Provenance.", message="Every number traces to a run on RIKYU and a summary file; evidence pages exist for the three investigations.", layout="Five bullets.", cue="Backup slide.")),
]


def design_for(title):
    for prefix, d in DESIGN:
        if title.startswith(prefix):
            return d
    return dict(purpose="—", message="—", layout="—", cue="—")


# ------------------------------------------------------------------ raw data per figure
def perf_rows():
    rows = json.loads((FIG / "performance_table.json").read_text())
    return sorted(rows, key=lambda r: -r["mean"])


def data_overview():
    rows = perf_rows()
    return "Every task: primary metric (mean ± sd over 5 seeds), training rows, source, group.\n\n" + md_table(
        ["task", "kind", "metric", "mean", "sd", "training rows", "test rows", "source", "group", "measured on"],
        [[r["task"], r["kind"], r["metric"], f"{r['mean']:.4f}", f"{r['sd']:.4f}", r["n_train"], r["n_test"] or "—", r["source"], r["group"], r["status"]] for r in rows])


def data_scatter(kind, k):
    base = {r["task"]: r for r in load("baselines_mp2026.json")["per_task"]}
    rows = [r for r in perf_rows() if r["metric"] == "R²" and r["group"] == kind][(k - 1) * 4:k * 4]
    return ("The four panels of this slide (ranked by R² across the series): task, R² mean ± sd (5 seeds), MAE, sample shown. The 2,000 (true, pred) points per panel are in "
            "summary/baselines_mp2026.json (`scatter` field, seed 2025) for the Materials Project tasks and in the seed-2025 prediction "
            "parquet of the stage_single run for the others.\n\n" + md_table(
                ["task", "R² mean", "sd", "MAE", "points shown"],
                [[r["task"], f"{r['mean']:.4f}", f"{r['sd']:.4f}", f"{r.get('mae', float('nan')):.4f}" if r.get("mae") else "—", "2,000 test rows" if r["task"] in base else "2,000 sampled test points"] for r in rows]))


def data_confusions():
    import pandas as pd
    S_ = Path("/private/tmp/claude-501/-Users-liuchang-projects-foundation-model/3c192fcf-9115-4fcd-85b0-49a8b332fcfb/scratchpad")
    out = []
    for task, labels, f in (("magnetic_ordering", ["AFM", "FM", "FiM", "NM"], S_ / "clfpred" / "stC_magnetic_ordering_2025.parquet"),
                            ("is_metal", ["no", "yes"], S_ / "clfpred" / "stC_is_metal_2025.parquet"), ("is_gap_direct", ["no", "yes"], S_ / "clfpred" / "stC_is_gap_direct_2025.parquet")):
        if not f.exists():
            continue
        df = pd.read_parquet(f); t, p = df["true"].to_numpy(int), df["pred"].to_numpy(int)
        cm = [[int(((t == i) & (p == j)).sum()) for j in range(len(labels))] for i in range(len(labels))]
        out.append(f"**{task}** (rows = true, columns = predicted, seed 2025):\n\n" + md_table(["true \\ pred"] + labels, [[labels[i]] + cm[i] for i in range(len(labels))]))
    return "\n\n".join(out)


def data_mt_confusion():
    import pandas as pd
    f = Path("/private/tmp/claude-501/-Users-liuchang-projects-foundation-model/3c192fcf-9115-4fcd-85b0-49a8b332fcfb/scratchpad/mtpred/stMb_2025.parquet")
    labels = ["DAC", "DQC", "IAC", "IQC", "others"]
    if not f.exists():
        return "(prediction file not on this machine)"
    df = pd.read_parquet(f); t, p = df["true"].to_numpy(int), df["pred"].to_numpy(int)
    cm = [[int(((t == i) & (p == j)).sum()) for j in range(5)] for i in range(5)]
    return "material_type trained alone, seed 2025, 7,357 test rows (rows = true, columns = predicted):\n\n" + md_table(["true \\ pred"] + labels, [[labels[i]] + cm[i] for i in range(5)])


def data_space_group():
    import pandas as pd
    sg = load("space_group_classes.json"); classes = sg["classes"]; counts = sg["counts"]
    f = Path("/private/tmp/claude-501/-Users-liuchang-projects-foundation-model/3c192fcf-9115-4fcd-85b0-49a8b332fcfb/scratchpad/sgpred/stW.parquet")
    order = sorted(range(len(classes)), key=lambda c: -counts[classes[c]])[:12]
    txt = "Twelve largest groups and their row counts: " + ", ".join(f"{classes[c]} ({counts[classes[c]]:,})" for c in order) + ".\n\n"
    if f.exists():
        df = pd.read_parquet(f); t, p = df["true"].to_numpy(int), df["pred"].to_numpy(int)
        fold = {c: i for i, c in enumerate(order)}; tf = np.array([fold.get(x, 12) for x in t]); pf = np.array([fold.get(x, 12) for x in p])
        cm = [[int(((tf == i) & (pf == j)).sum()) for j in range(13)] for i in range(13)]
        labels = [classes[c] for c in order] + ["other"]
        txt += "Confusion (seed 2025, rows = true, columns = predicted):\n\n" + md_table(["true \\ pred"] + labels, [[labels[i]] + cm[i] for i in range(13)])
    per = load("space_group_perclass.json")["arms"]["plain_kmd"]["per_class"]
    rows = sorted(((n, v["n_all"], v["recall"], v["precision"]) for n, v in per.items()), key=lambda x: -x[1])
    txt += "\n\nPer-class recall / precision (seed 2025), all 151 groups by size:\n\n" + md_table(["group", "rows", "recall", "precision"], [[n, c, f"{r:.2f}", f"{p:.2f}" if p is not None else "—"] for n, c, r, p in rows])
    return txt


def data_kmd():
    d = {r["task"]: r for r in load("descriptor.json")["per_task"]}; sgc = load("space_group_confirm.json")["arms"]; base = {r["task"]: r for r in load("baselines_mp2026.json")["per_task"]}
    return md_table(["label", "KMD", "XenonPy classic (290 columns, carries the atom count)", "metric"], [
        ["volume (cell)", f"{d['volume']['kmd']['mean']:.3f}", f"{d['volume']['classic']['mean']:.3f}", "R², 5 seeds"],
        ["volume per atom (density_atomic)", f"{base['density_atomic']['r2']['mean']:.3f}", "not measured", "R²"],
        ["total magnetisation (cell)", f"{base['total_magnetization']['r2']['mean']:.3f}", "not measured", "R²"],
        ["space group (151 classes)", f"{st.fmean(r['accuracy'] for r in sgc['plain_kmd']['runs']):.3f}", f"{st.fmean(r['accuracy'] for r in sgc['plain_classic']['runs']):.3f}", "top-1 accuracy, 5 seeds"]])


def data_mt_runs():
    d = load("material_type_warmstart_weighted.json"); a = d["arms"]
    rows = [[f"seed {r['run'].rsplit('_s', 1)[1]}", "alone", "—", f"{r['macro_f1']:.4f}", f"{r['accuracy']:.4f}", len(r["curve"])] for r in a["alone"]]
    rows += [[f"24-task encoder, ordering {int(r['run'].rsplit('_o', 1)[1]) + 1}", "warm-start, encoder saw material_type", f"{r['before']['macro_f1']:.4f}", f"{r['after']['macro_f1']:.4f}", f"{r['after']['accuracy']:.4f}", r["epochs_run"]] for r in a["warm_seen"]]
    rows += [[f"23-task encoder, ordering {int(r['run'].rsplit('_o', 1)[1]) + 1}", "warm-start, encoder never saw it", f"{r['before']['macro_f1']:.4f}", f"{r['after']['macro_f1']:.4f}", f"{r['after']['accuracy']:.4f}", r["epochs_run"]] for r in a["warm_unseen"]]
    return f"Every run (test rows = {d['test_rows']:,}):\n\n" + md_table(["run", "arm", "macro-F1 before fine-tune", "macro-F1 after / final", "accuracy", "epochs"], rows)


def data_mt_losses():
    d = load("material_type_warmstart_weighted.json"); a = d["arms"]
    out = []
    for tag, lab in (("alone", "alone"), ("warm_seen", "warm-start, seen"), ("warm_unseen", "warm-start, never seen")):
        runs = [r for r in a[tag] if r["curve"]]; L = min(len(r["curve"]) for r in runs)
        rows = []
        for e in range(0, L, max(1, L // 12)):
            tr = [r["curve"][e][1] for r in runs if r["curve"][e][1] is not None]; va = [r["curve"][e][2] for r in runs if r["curve"][e][2] is not None]
            rows.append([e, f"{np.median(tr):.4f}" if tr else "—", f"{np.median(va):.4f}" if va else "—"])
        out.append(f"**{lab}** (median over {len(runs)} runs; shortest run {L} epochs):\n\n" + md_table(["epoch", "training loss", "validation loss"], rows))
    return "\n\n".join(out)


def data_mt_perclass():
    d = load("material_type_weighted_perclass.json")["arms"]; names = ["DAC", "DQC", "IAC", "IQC", "others"]
    rows = []
    for arm, lab in (("alone", "alone (5)"), ("seen", "warm-start, seen (10)"), ("unseen", "warm-start, never seen (3)")):
        for n in names:
            rows.append([lab, n, d[arm][0]["per_class"][n]["n_test"], f"{st.fmean(r['per_class'][n]['recall'] for r in d[arm]):.3f}", f"{st.fmean(r['per_class'][n]['precision'] for r in d[arm]):.3f}", f"{st.fmean(r['per_class'][n]['n_pred'] for r in d[arm]):.1f}"])
    return md_table(["arm", "class", "test rows", "recall (mean)", "precision (mean)", "predicted rows per run (mean)"], rows)


def data_mt_position():
    pr = load("position_runs.json")["material_type"]; pos = np.array([r["pos"] for r in pr]); at = np.array([r["at_train"] for r in pr], dtype=float)
    rows = []
    for p in range(1, 25):
        v = at[pos == p]; v = v[~np.isnan(v)]
        rows.append([p, len(v), f"{np.median(v):.4f}", f"{np.percentile(v, 25):.4f}", f"{np.percentile(v, 75):.4f}", f"{v.min():.4f}", f"{v.max():.4f}"])
    wa = load("material_type_weights.json")["arms"]["balanced"]["runs"]; wal = [r["macro_f1"] for r in wa]
    return f"Alone: mean {st.fmean(wal):.4f}, range {min(wal):.4f}–{max(wal):.4f} (5 seeds). Per position in the 240-run continual stage (the number of orderings that put material_type at that position varies), macro-F1 measured at material_type's own step:\n\n" + md_table(["position", "runs", "median", "Q1", "Q3", "min", "max"], rows)


FIGURE_DATA = [
    ("Single-task performance, all 41 tasks", data_overview),
    ("Single-task R² against training rows", lambda: "Same table as the overview figure (task, R², training rows)."),
    ("Added classification heads: confusion matrices", data_confusions),
    ("Space group: confusion over the twelve largest groups", data_space_group),
    ("Space group: per-class F1 against class size", lambda: "Same per-class table as the confusion slide (F1 = harmonic mean of recall and precision; x = rows in the group)."),
    ("Where the descriptor limits the label", data_kmd),
    ("material_type: the task, trained alone", data_mt_confusion),
    ("material_type: every run of the three arms", data_mt_runs),
    ("material_type: before and after the fine-tune", lambda: "Same per-run table as the strip-plot slide (columns “before fine-tune” and “after”)."),
    ("material_type: training and validation loss", data_mt_losses),
    ("material_type: where the gain comes from", data_mt_perclass),
    ("material_type against pretraining breadth", data_mt_position),
]


FIGFILE = [("Single-task performance", "overview_bar.png"), ("Single-task R² against", "r2_vs_n.png"), ("Added classification heads", "confusion_clf.png"),
           ("Space group: confusion", "confusion_sg.png"), ("Space group: per-class", "sg_f1_vs_size.png"), ("Where the descriptor", "kmd_scale.png"),
           ("material_type: the task", "confusion_material_type.png"), ("material_type: every run", "mt_warmstart_strip.png"), ("material_type: before and after", "mt_before_after.png"),
           ("material_type: training and validation", "mt_losses.png"), ("material_type: where the gain", "mt_perclass.png"), ("material_type against", "mt_position.png")]


def figure_file(title):
    import re
    m = re.search(r"(existing tasks|tasks added 2026-09-11) \((\d+)/", title)
    if m:
        return f"scatter_{'existing' if m.group(1).startswith('existing') else 'added'}_{m.group(2)}.png"
    for prefix, f in FIGFILE:
        if title.startswith(prefix):
            return f
    return "image.png"


def figure_data_for(title):
    import re
    m = re.search(r"(existing tasks|tasks added 2026-09-11) \((\d+)/", title)
    if m:
        return data_scatter("existing" if m.group(1).startswith("existing") else "added", int(m.group(2)))
    for prefix, fn in FIGURE_DATA:
        if title.startswith(prefix):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                return f"(could not render raw data: {e})"
    return None


# ------------------------------------------------------------------ walk the deck
def main():
    prs = Presentation(str(DECK))
    lines = [f"# Design brief and raw data — DECK_20260915.pptx ({len(prs.slides)} slides)", "",
             "Generated from the built deck and the same summary files its figures were drawn from. For every slide: title and subtitle as on the slide, "
             "the design note (purpose · key message · layout · speaker cue), the slide's text and tables verbatim, and — for figure slides — the numbers behind the figure.",
             "", "Conventions used throughout the deck: ± is the standard deviation over seeds (or orderings); 2×SE is twice the standard error of a difference between two arms; "
             "“better / worse” needs separation at 2×SE and an absolute difference ≥ 0.01, otherwise “unresolved”. Regression labels are trained on their normalised form, so R² and MAE are on that scale. "
             "Figure labels stay in English. Dataset date: 2026-09-11.", ""]
    for i, s in enumerate(prs.slides, 1):
        texts = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        title = texts[0].text_frame.text.split("\n")[0] if texts else f"slide {i}"
        sub = texts[1].text_frame.text if len(texts) > 1 and s.shapes[0].shape_type != 1 and len(texts[1].text_frame.text) < 400 else ""
        d = design_for(title)
        lines += [f"---", f"## Slide {i} — {title}", ""]
        if sub and sub != title:
            lines += [f"*Subtitle:* {sub}", ""]
        lines += ["**Design note**", f"- Purpose: {d['purpose']}", f"- Key message: {d['message']}", f"- Layout: {d['layout']}", f"- Speaker cue: {d['cue']}", ""]
        body = []
        for sh in texts[2:] if sub else texts[1:]:
            t = sh.text_frame.text.strip()
            if t and t != title:
                body.append(t)
        pics = [sh for sh in s.shapes if sh.shape_type == 13]
        tables = [sh for sh in s.shapes if sh.has_table]
        autoshapes = [sh for sh in s.shapes if sh.shape_type == 1 and sh.has_text_frame and sh.text_frame.text.strip() and sh not in texts[:2]]
        if title.startswith("Roadmap"):
            body = []
            lines += ["**Flowchart (three lanes, left to right, boxes joined by arrows)**", "",
                      "- Lane 1 · Model base: Pretrained model library (foundation base: encoder + task heads) → Online prediction service (API) · Download, run locally.",
                      "- Lane 2 · Web app: Model preview (browse tasks, try compositions) → Upload your own data → AI-assisted fine-tuning → Inspect and preview the fine-tuned model online → Download the model · Host it online as a prediction service.",
                      "- Lane 3 · Package + AI: foundation-model package with preset skills (predict, fine-tune, inverse design) → AI agent / LLM (uses the skills) → Local model service · Online service via API.",
                      "- Footer line: “Skills work the same locally and through the API; the LLM plans, the model computes.”", ""]
        if body:
            lines += ["**Text on the slide**", ""]
            for b in body:
                for ln in b.split("\n"):
                    if ln.strip():
                        lines.append(f"- {ln.strip()}" if not ln.strip().startswith(("•", "-")) else ln.strip())
            lines.append("")
        for tbl in tables:
            rows = [[c.text for c in r.cells] for r in tbl.table.rows]
            lines += ["**Table on the slide**", "", md_table(rows[0], rows[1:]), ""]
        if pics:
            lines += [f"**Figure**: {figure_file(title)} (PNG in results/deck_20260915/)", ""]
            raw = figure_data_for(title)
            if raw:
                lines += ["**Raw data behind the figure**", "", raw, ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{OUT}  ({len(lines)} lines, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
