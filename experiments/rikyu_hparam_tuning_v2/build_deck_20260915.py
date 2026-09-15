#!/usr/bin/env python3
"""Build results/DECK_20260915.pptx — tasks & single-task performance, material_type transfer,
the data plan, and the release / LLM roadmap.

Numbers come from results/deck_20260915/performance_table.json and material_type_numbers.json
(written by analysis/deck_figures.py) and from the summary JSONs; figures from results/deck_20260915.
One idea per slide, presentation-sized fonts; the reader deletes what is not needed.

    uv run --with python-pptx --with pillow python experiments/rikyu_hparam_tuning_v2/build_deck_20260915.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from lxml import etree
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from build_report_pptx import BLANK, GREEN, HEADGREY, INK, MUT, RED, WHITE, new, pic_slide, prs, txt  # noqa: F401
from pptx.dml.color import RGBColor

HERE = Path(__file__).resolve().parent
S = HERE / "summary"
FIG = HERE / "results" / "deck_20260915"
DATE = "2026-09-15"
PERF = json.loads((FIG / "performance_table.json").read_text())
MT = json.loads((FIG / "material_type_numbers.json").read_text())
INV = {t["name"]: t for t in json.loads((S / "task_inventory_20260911.json").read_text())}
TEAL = RGBColor(0x0E, 0x6E, 0x78); PURPLE = RGBColor(0x5B, 0x3F, 0x8C); PALE = RGBColor(0xE8, 0xEE, 0xF2); PALE2 = RGBColor(0xD5, 0xE8, 0xEA)

# ------------------------------------------------------------------ task descriptions
DESC = {
    "final_energy": ("Materials Project", "DFT total energy per atom (GGA / GGA+U scheme)", "eV / atom"),
    "formation_energy": ("Materials Project", "Formation energy per atom relative to the elements", "eV / atom"),
    "reaction_energy": ("Materials Project", "Equilibrium reaction energy: margin to the nearest competing phases", "eV / atom"),
    "volume": ("Materials Project", "Volume of the cell as stored (KMD-limited)", "Å³"),
    "density": ("Materials Project", "Mass density of the crystal", "g / cm³"),
    "density_atomic": ("Materials Project", "Volume per atom (intensive form of volume)", "Å³ / atom"),
    "efermi": ("Materials Project", "Fermi energy", "eV"),
    "band_gap": ("Materials Project", "Electronic band gap (GGA band structure)", "eV"),
    "cbm": ("Materials Project", "Conduction-band minimum (non-metals only)", "eV"),
    "vbm": ("Materials Project", "Valence-band maximum (non-metals only)", "eV"),
    "is_metal": ("Materials Project", "Metal or not (gap = 0), 2 classes", "—"),
    "is_gap_direct": ("Materials Project", "Direct or indirect gap (non-metals), 2 classes", "—"),
    "total_magnetization": ("Materials Project", "Net magnetic moment of the cell (KMD-limited)", "μB / cell"),
    "magnetization_per_volume": ("Materials Project", "Net magnetic moment per unit volume", "μB / Å³"),
    "magnetization_per_fu": ("Materials Project", "Net magnetic moment per formula unit", "μB / f.u."),
    "magnetic_ordering": ("Materials Project", "Ground-state ordering NM / FM / FiM / AFM, 4 classes", "—"),
    "dielectric_total": ("Materials Project", "Total dielectric constant (DFPT)", "—"),
    "dielectric_ionic": ("Materials Project", "Ionic part of the dielectric constant", "—"),
    "dielectric_electronic": ("Materials Project", "Electronic part of the dielectric constant", "—"),
    "refractive_index": ("Materials Project", "Refractive index (√ electronic dielectric constant)", "—"),
    "bulk_modulus": ("Materials Project", "Bulk modulus, Voigt–Reuss–Hill average", "GPa"),
    "shear_modulus": ("Materials Project", "Shear modulus, Voigt–Reuss–Hill average", "GPa"),
    "poisson_ratio": ("Materials Project", "Poisson ratio from the elastic tensor", "—"),
    "universal_anisotropy": ("Materials Project", "Universal elastic anisotropy index", "—"),
    "piezoelectric_max": ("Materials Project", "Largest piezoelectric tensor component (DFPT)", "C / m²"),
    "space_group": ("Materials Project", "Space group of the relaxed structure, 151 classes (see the space-group slide)", "—"),
    "seebeck": ("thermoelectric (starry)", "Seebeck coefficient S(T), curve over temperature", "V / K"),
    "electrical_resistivity": ("thermoelectric (starry)", "Electrical resistivity ρ(T), curve over temperature", "Ω·m"),
    "thermal_conductivity": ("thermoelectric (starry)", "Thermal conductivity κ(T), curve over temperature", "W / (m·K)"),
    "power_factor": ("thermoelectric (starry)", "Power factor S²σ(T), curve over temperature", "W / (m·K²)"),
    "zt": ("thermoelectric (starry)", "Figure of merit ZT(T), curve over temperature", "—"),
    "magnetic_susceptibility": ("thermoelectric (starry)", "Magnetic susceptibility χ(T), curve over temperature (58 compositions)", "A·m² / mol"),
    "dos_density": ("thermoelectric (starry)", "Electronic density of states, curve over energy", "states / eV"),
    "magnetization": ("NEMAD magnetic", "Magnetization (experimental, text-mined)", "A·m² / kg"),
    "magnetic_moment": ("NEMAD magnetic", "Magnetic moment per formula unit (experimental)", "μB / f.u."),
    "curie": ("NEMAD magnetic", "Curie temperature", "K"),
    "neel": ("NEMAD magnetic", "Néel temperature", "K"),
    "tc": ("NEMAD superconductor", "Superconducting transition temperature", "K"),
    "kp": ("phonix-db", "Particle-like (Peierls) part of the lattice thermal conductivity κ_p", "W / (m·K)"),
    "klat": ("phonix-db", "Lattice thermal conductivity κ_lat", "W / (m·K)"),
    "material_type": ("quasicrystal (qa/starry)", "DAC / DQC / IAC / IQC / others — approximants, quasicrystals, everything else (5 classes, 99 % “others”)", "—"),
}
KIND_LABEL = {"regression": "regression", "kernel_regression": "curve", "classification": "classification"}
ADDED = {"band_gap", "density_atomic", "magnetization_per_volume", "magnetization_per_fu", "reaction_energy", "cbm", "vbm", "bulk_modulus",
         "shear_modulus", "poisson_ratio", "universal_anisotropy", "refractive_index", "piezoelectric_max", "magnetic_ordering", "is_metal",
         "is_gap_direct", "space_group"}


SHORT_STATUS = {"labels unchanged; same rows, measured 2026-08": "unchanged labels (2026-08 runs)", "relabelled, 2026-09-11 dataset": "relabelled (2026-09-11)",
                "added, 2026-09-11 dataset": "added (2026-09-11)", "class weights off, 2026-09-11 dataset": "class weights off (2026-09-11)"}


def perf(task):
    return next(r for r in PERF if r["task"] == task)


def fmt_n(v):
    return "—" if v is None else f"{v:,}"


# ------------------------------------------------------------------ helpers (bigger type than the old deck)
def table(slide, x, y, w, headers, rows, col_w=None, size=13, head_size=13, section_rows=()):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(0.42 * (len(rows) + 1)))
    tbl = shape.table
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Inches(cw)
    for j, head in enumerate(headers):
        cell = tbl.cell(0, j); cell.text = str(head); para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(head_size); para.font.bold = True; para.font.color.rgb = WHITE
        cell.fill.solid(); cell.fill.fore_color.rgb = HEADGREY
    for i, row in enumerate(rows, start=1):
        is_section = (i - 1) in section_rows
        for j, value in enumerate(row):
            cell = tbl.cell(i, j); cell.text = str(value); para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(size); para.font.color.rgb = INK
            if is_section:
                para.font.bold = True; para.font.color.rgb = WHITE; cell.fill.solid(); cell.fill.fore_color.rgb = TEAL
            elif j:
                para.alignment = PP_ALIGN.LEFT
        if is_section and len(row) > 1:
            tbl.cell(i, 0).merge(tbl.cell(i, len(row) - 1))
    return tbl


def divider(title, sub):
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = TEAL; bg.line.fill.background()
    txt(s, 0.9, 2.6, 11.5, 1.4, [title], size=40, bold=True, color=WHITE)
    txt(s, 0.9, 4.0, 11.5, 1.2, [sub], size=20, color=WHITE)
    return s


def bullets(s, lines, y=1.4, size=18, h=5.6):
    txt(s, 0.6, y, 12.1, h, lines, size=size)


# ------------------------------------------------------------------ slides
def slide_title():
    s = prs.slides.add_slide(BLANK)
    txt(s, 0.7, 2.1, 12, 1.4, ["Composition-only foundation model:", "tasks, transfer, data plan, release"], size=34, bold=True)
    txt(s, 0.7, 4.0, 12, 1.5, [f"Status deck, {DATE} — dataset 2026-09-11, KMD descriptor",
                               "1 · Tasks and single-task performance   2 · material_type transfer   3 · Data plan   4 · Release and LLM access"], size=16, color=MUT)


def slide_dataset():
    s = new("The 2026-09-11 dataset", "One row per composition; every task is a column, missing where the source has no value")
    rows = [
        ["Materials Project — stable entries, GGA / GGA+U only", "33,829", "energies, structure, electronic, magnetic, elastic, dielectric, space group", "26"],
        ["Thermoelectric curves (starry)", "14,838", "S, ρ, κ, PF, ZT, χ over temperature; DOS over energy", "7"],
        ["Quasicrystal set (qa / starry)", "49,034 rows labelled", "material type: DAC / DQC / IAC / IQC / others", "1"],
        ["NEMAD magnetic (text-mined, experimental)", f"{fmt_n(perf('curie')['n_train'])} (Curie)", "magnetization, moment, Curie, Néel", "4"],
        ["NEMAD superconductor", fmt_n(perf("tc")["n_train"]), "transition temperature", "1"],
        ["phonix-db (phonon transport)", fmt_n(perf("klat")["n_train"]), "κ_p, κ_lat", "2"],
    ]
    table(s, 0.5, 1.4, 12.3, ["source", "compositions", "what it carries", "tasks"], rows, col_w=[4.3, 2.0, 5.0, 1.0], size=13)
    txt(s, 0.5, 4.8, 12.3, 2.3, [
        "Input to the model: the composition string only → KMD descriptor (464 columns, invertible).",
        "Materials Project part rebuilt 2026-09-11 on one level of theory per column family; it fixed final_energy (R² 0.77 → 0.999) and added 17 properties.",
    ], size=14, color=MUT)


def slide_inventory(title, tasks, sub):
    s = new(title, sub)
    rows = []
    for t in tasks:
        src, what, unit = DESC[t]; inv = INV[t]; n = inv["n"] or {}
        rows.append([t.replace("_", " "), what, unit, KIND_LABEL[inv["kind"]], fmt_n(n.get("train") or perf(t)["n_train"]), fmt_n(n.get("test") or perf(t)["n_test"])])
    table(s, 0.4, 1.35, 12.5, ["task", "what it is", "unit", "kind", "train", "test"], rows, col_w=[2.3, 5.9, 1.3, 1.3, 0.85, 0.85], size=12.5, head_size=12.5)


def slide_perf_table(title, tasks, sub):
    s = new(title, sub)
    rows = []
    for t in tasks:
        r = perf(t)
        if r["metric"] == "R²":
            rows.append([t.replace("_", " "), "R²", f"{r['mean']:.3f} ± {r['sd']:.3f}", f"{r['mae']:.3f}" if r.get("mae") else "—", fmt_n(r["n_test"]), SHORT_STATUS.get(r["status"], r["status"])])
        else:
            rows.append([t.replace("_", " "), "macro-F1", f"{r['mean']:.3f} ± {r['sd']:.3f}", f"acc {r['accuracy']:.3f}", fmt_n(r["n_test"]), SHORT_STATUS.get(r["status"], r["status"])])
    table(s, 0.4, 1.35, 12.5, ["task", "metric", "mean ± sd (5 seeds)", "MAE / accuracy", "test rows", "measured on"], rows, col_w=[2.6, 1.1, 2.3, 1.7, 1.1, 3.7], size=12.5, head_size=12.5)


def slide_space_group_why():
    s = new("Space group: why 151 classes and not 230", "The count is data availability, not a physics choice")
    bullets(s, [
        "• 230 space groups exist. The stable Materials Project set contains 213 of them; 17 never occur among stable entries.",
        "• A class needs enough examples to be learned and to be evaluated. We keep every group with ≥ 10 rows and at least one row in both the train and the test split: 151 groups.",
        "• The other 62 groups hold 274 rows together (0.8 % of the rows). Their rows are not dropped: the space-group label is set to missing for them, exactly as any other task treats a missing value, and they still train every other head.",
        "• The distribution is the physics of the crystal world: Fm-3m alone is 10 % of the rows, the 12 largest groups are half of them, and the tail is long. Macro-F1 is bounded by that tail; top-1 accuracy is the number to watch.",
        "• Single-task result (5 seeds): top-1 accuracy 0.465, macro-F1 0.313. The ShotgunCSP classifier reaches 0.60 with a descriptor that carries the atom count and a wider network — see the space-group investigation page.",
    ], size=17)


def slide_clf_change():
    s = new("Classification heads: class weights switched off", "Inverse-frequency weights were on for every classification head until 2026-09-11; every number in this deck is without them")
    cw = json.loads((S / "classification_weights.json").read_text())["tasks"]
    mt = json.loads((S / "material_type_weights.json").read_text())["arms"]
    sg = json.loads((S / "space_group_confirm.json").read_text())["arms"]
    def m(rs, k): return st.fmean(r[k] for r in rs)
    rows = [["material_type (5 classes, 99 % majority)", f"{m(mt['balanced']['runs'], 'macro_f1'):.3f}", f"{m(mt['none']['runs'], 'macro_f1'):.3f}", f"{m(mt['balanced']['runs'], 'accuracy'):.3f} → {m(mt['none']['runs'], 'accuracy'):.3f}", "rare-class precision doubles"],
            ["space_group (151 classes)", f"{m(sg['balanced_kmd']['runs'], 'macro_f1'):.3f}", f"{m(sg['plain_kmd']['runs'], 'macro_f1'):.3f}", f"{m(sg['balanced_kmd']['runs'], 'accuracy'):.3f} → {m(sg['plain_kmd']['runs'], 'accuracy'):.3f}", "large groups become learnable"],
            ["magnetic_ordering (4 classes)", f"{m(cw['magnetic_ordering']['arms']['balanced']['runs'], 'macro_f1'):.3f}", f"{m(cw['magnetic_ordering']['arms']['none']['runs'], 'macro_f1'):.3f}", f"{m(cw['magnetic_ordering']['arms']['balanced']['runs'], 'accuracy'):.3f} → {m(cw['magnetic_ordering']['arms']['none']['runs'], 'accuracy'):.3f}", "AFM recall 0.56 → 0.14 (118 rows)"],
            ["is_metal (2 balanced classes)", f"{m(cw['is_metal']['arms']['balanced']['runs'], 'macro_f1'):.3f}", f"{m(cw['is_metal']['arms']['none']['runs'], 'macro_f1'):.3f}", "unchanged", "weights ≈ 1"],
            ["is_gap_direct (13 % direct)", f"{m(cw['is_gap_direct']['arms']['balanced']['runs'], 'macro_f1'):.3f}", f"{m(cw['is_gap_direct']['arms']['none']['runs'], 'macro_f1'):.3f}", f"{m(cw['is_gap_direct']['arms']['balanced']['runs'], 'accuracy'):.3f} → {m(cw['is_gap_direct']['arms']['none']['runs'], 'accuracy'):.3f}", "direct-gap recall 0.79 → 0.25"]]
    table(s, 0.5, 1.4, 12.3, ["head", "macro-F1, weights on", "macro-F1, off", "accuracy on → off", "what moves"], rows, col_w=[3.6, 1.9, 1.7, 2.2, 2.9], size=13)
    txt(s, 0.5, 4.6, 12.3, 2.2, ["Five seeds per arm, same recipe, same rows. The weights help no head on macro-F1 or accuracy; they only buy minority recall where the minority is a real class.",
                                 "Decision (2026-09-11): class_weights = \"none\" on every classification head (knob released as PR #57, version 0.4.1)."], size=14, color=MUT)


def slide_kmd():
    s = pic_slide("Where the descriptor limits the label", "KMD works on atomic fractions: Fe₂O₃ and Fe₄O₆ are the same input, so labels that scale with the cell stop at what composition implies", FIG / "kmd_scale.png", top=1.35, bottom=1.35)
    txt(s, 0.5, 6.15, 12.3, 1.2, ["Decision (2026-09-11): keep KMD — its invertibility is what inverse design needs. volume, total magnetisation and space group carry this caveat; the intensive forms (volume per atom 0.979, magnetisation per volume) are the ones to use."], size=14, color=MUT)


# ---- Part 2
def slide_mt_setup():
    s = new("material_type transfer: the setup", "Warm-start fine-tuning with the encoder trained")
    table(s, 0.5, 1.4, 12.3, ["arm", "encoder comes from", "head", "n"], [
        ["trained alone", "random init", "fresh", "5 seeds"],
        ["warm-start, encoder SAW material_type", "23 tasks + material_type pretrained continually (material_type last, with replay)", "the head trained at that step", "10 orderings"],
        ["warm-start, encoder NEVER saw it", "the same orderings stopped one step earlier (23 tasks, material_type dropped)", "fresh", "3 orderings"],
    ], col_w=[3.4, 5.6, 2.3, 1.0], size=13)
    txt(s, 0.5, 3.9, 12.3, 3.2, [
        f"Fine-tune: fm finetune, encoder + head trained, replay off, early stopping on material_type's own validation loss (patience 24, cap 150), last-epoch weights.",
        f"Metric: macro-F1 over the five classes on the same {MT['test_rows']:,} test rows for every arm; the same training recipe in every arm.",
        "The question the third arm answers: does the encoder need to have met the task during pretraining, or is the 23-task representation what transfers?",
    ], size=14, color=MUT)


def slide_mt_numbers():
    a, se, un = MT["alone"]["values"], MT["seen"]["values"], MT["unseen"]["values"]
    def row(name, v, base=None):
        m = st.fmean(v); sd = st.stdev(v); r = [name, f"{m:.3f} ± {sd:.3f}", str(len(v))]
        if base is not None:
            bm = st.fmean(base); d = m - bm; se2 = 2 * ((sd ** 2 / len(v)) + (st.stdev(base) ** 2 / len(base))) ** 0.5
            r += [f"{d:+.3f} ({d / bm * 100:+.1f} %)", f"{se2:.3f}", "separated" if abs(d) > se2 and abs(d) >= 0.01 else "unresolved"]
        else:
            r += ["—", "—", "reference"]
        return r
    d = st.fmean(se) - st.fmean(un); se2 = 2 * ((st.stdev(se) ** 2 / len(se)) + (st.stdev(un) ** 2 / len(un))) ** 0.5
    s = new("material_type transfer: the numbers", "Warm-start beats training alone by 22 %, whether or not the encoder ever saw the task")
    table(s, 0.5, 1.4, 12.3, ["arm", "macro-F1 (mean ± sd)", "n", "vs alone", "2×SE", "verdict"],
          [row("trained alone", a), row("warm-start, encoder saw material_type", se, a), row("warm-start, encoder never saw it", un, a)], col_w=[4.0, 2.4, 0.6, 2.2, 1.2, 1.9], size=13)
    bullets(s, [
        f"Seen vs never-seen: {st.fmean(se):.3f} vs {st.fmean(un):.3f}, difference {d:+.3f} against 2×SE {se2:.3f} — indistinguishable. The +22 % is the 23-task representation, not the single exposure at the last pretraining step.",
        "Consequence: for a new task, pretrain on the existing tasks and warm-start fine-tune — the continual step with replay for the new task adds nothing the fine-tune does not recover, and costs the most.",
        "Extending the never-seen arm from 3 to 5 orderings: two more 23-step pretraining runs (~10–15 GPU-hours each, about 1.5 days wall-clock in parallel) plus minutes of fine-tuning.",
    ], y=3.6, size=15, h=3.3)



def _cv_table():
    d = json.loads((S / "material_type_cv.json").read_text())["runs"]
    by = {}
    for r in d:
        by.setdefault((r["protocol"], r["descriptor"], r["model"]), []).append(r)
    def f1c(rs, c):
        return st.fmean((2 * r["per_class_3"][c]["precision"] * r["per_class_3"][c]["recall"] / (r["per_class_3"][c]["precision"] + r["per_class_3"][c]["recall"]) if r["per_class_3"][c]["precision"] and r["per_class_3"][c]["recall"] else 0.0) for r in rs)
    rows = [["2021 paper (Adv. Mater.): random 80/20 × 100, XenonPy 232, random forest, 80 QC / 78 AC / 10,090 others", "0.650", "0.658", "≈ 0.77", "—"]]
    lab = {"dataset": "the dataset's own split (pipeline)", "random": "5-fold random CV over rows", "system": "5-fold CV grouped by element set (new systems)"}
    for p in ("dataset", "random", "system"):
        for desc, mdl in (("classic", "rf"), ("classic", "nn"), ("kmd", "rf"), ("kmd", "nn")):
            rs = by.get((p, desc, mdl))
            if not rs:
                continue
            f3 = [r["macro_f1_3class"] for r in rs]; f5 = [r["macro_f1"] for r in rs]
            sd = (lambda v: f" ± {st.stdev(v):.3f}" if len(v) > 1 else "")
            rows.append([f"{lab[p]} — {'XenonPy classic' if desc == 'classic' else 'KMD'} + {'random forest' if mdl == 'rf' else 'pipeline network'}",
                         f"{f1c(rs, 'QC'):.3f}", f"{f1c(rs, 'AC'):.3f}", f"{st.fmean(f3):.3f}{sd(f3)}", f"{st.fmean(f5):.3f}{sd(f5)}"])
    return rows


def slide_mt_cv():
    s = new("material_type, evaluated properly", "The 0.83 on the dataset's split is a leaky split, not a better model: 75 % of the minority test rows have a training row within L1 0.05 in atomic fractions")
    table(s, 0.4, 1.35, 12.5, ["protocol — descriptor + model", "QC F1", "AC F1", "3-class macro-F1", "5-class macro-F1"], _cv_table(), col_w=[6.6, 1.2, 1.2, 1.9, 1.6], size=11.5, head_size=12)
    txt(s, 0.4, 6.35, 12.5, 1.1, ["3-class = the 2021 paper's task (QC = DQC + IQC, AC = DAC + IAC, others). The 5-class macro-F1 swings on the 1-row DAC and 3-row DQC test classes and should not be quoted. Grouped by element set — the case of a new system — the numbers land where the 2021 paper's did; XenonPy classic beats KMD on this task under every protocol."], size=12.5, color=MUT)


def slide_mt_transfer_cv():
    d = json.loads((S / "material_type_transfer_cv.json").read_text())["runs"]
    def agg(p, arm, key):
        v = [r[key] for r in d if r["protocol"] == p and r["arm"] == arm]; return st.fmean(v), st.stdev(v), len(v)
    rows = []
    for p, lab in (("random", "5-fold random CV over rows"), ("system", "5-fold CV grouped by element set (new systems)")):
        for arm, alab in (("alone", "trained alone (3 seeds × 5 folds)"), ("warm", "warm-start from a never-seen 23-task encoder (3 encoders × 5 folds)")):
            m3, s3, n = agg(p, arm, "macro_f1_3class"); m5, s5, _ = agg(p, arm, "macro_f1"); ep, _, _ = agg(p, arm, "epochs")
            rows.append([lab if arm == "alone" else "", alab, f"{m3:.3f} ± {s3:.3f}", f"{m5:.3f} ± {s5:.3f}", f"{ep:.0f}"])
    diffs = {}
    for p in ("random", "system"):
        dd = []
        for k in range(5):
            al = [r["macro_f1_3class"] for r in d if r["protocol"] == p and r["fold"] == k and r["arm"] == "alone"]; wm = [r["macro_f1_3class"] for r in d if r["protocol"] == p and r["fold"] == k and r["arm"] == "warm"]
            dd.append(st.fmean(wm) - st.fmean(al))
        diffs[p] = (st.fmean(dd), 2 * st.stdev(dd) / len(dd) ** 0.5)
    s = new("Warm-start vs alone under cross-validation", "KMD, pipeline network, class weights off; the never-seen encoders are the stage_xu checkpoints")
    table(s, 0.5, 1.4, 12.3, ["protocol", "arm", "3-class macro-F1", "5-class macro-F1", "epochs"], rows, col_w=[3.9, 4.6, 1.5, 1.5, 0.8], size=12.5)
    bullets(s, [
        f"Paired by fold, warm-start − alone (3-class F1): random split {diffs['random'][0]:+.3f} (2×SE {diffs['random'][1]:.3f}); grouped by system {diffs['system'][0]:+.3f} (2×SE {diffs['system'][1]:.3f}) — neither is a resolved gain.",
        "The pretrained encoder converges in fewer epochs and lifts the noisy 5-class number through the tiny DAC / DQC classes; on new systems it does not generalise better than training from scratch.",
        "Conclusion for material_type: 34,000 rows and a 3-class problem the descriptor already separates — the task does not need the encoder. The transfer case has to be made on tasks with few rows or on tasks the encoder never saw.",
    ], y=4.1, size=14, h=3.0)


def slide_mt_why():
    L = MT["losses"]; pc = json.loads((S / "material_type_weighted_perclass.json").read_text())["arms"]
    p = lambda arm, c, k: st.fmean(r["per_class"][c][k] for r in pc[arm])
    s = new("Why material_type gains from the pretrained encoder", "Reading the per-class figure and the loss curves")
    bullets(s, [
        f"• Recall does not move: the rare classes are found equally well by both arms (IAC {p('alone', 'IAC', 'recall'):.2f} → {p('seen', 'IAC', 'recall'):.2f}, IQC {p('alone', 'IQC', 'recall'):.2f} → {p('seen', 'IQC', 'recall'):.2f}). Precision does: IAC {p('alone', 'IAC', 'precision'):.2f} → {p('seen', 'IAC', 'precision'):.2f}, IQC {p('alone', 'IQC', 'precision'):.2f} → {p('seen', 'IQC', 'precision'):.2f}, DQC {p('alone', 'DQC', 'precision'):.2f} → {p('seen', 'DQC', 'precision'):.2f}.",
        f"• So the gain is fewer false alarms: the model trained alone files {p('alone', 'IAC', 'n_pred'):.0f} test rows a run as IAC for 24 real ones, the warm-started one {p('seen', 'IAC', 'n_pred'):.0f}; ordinary materials stop being mistaken for approximants and quasicrystals. macro-F1 rewards exactly that.",
        f"• The loss curves say the same: the warm-started runs start lower (validation loss at epoch 1: {L['warm_seen']['val_first']:.2f} vs {L['alone']['val_first']:.2f}), fit the training set further (last training loss {L['warm_seen']['train_last']:.3f} vs {L['alone']['train_last']:.3f}) and stop after {L['warm_seen']['epochs']:.0f} epochs instead of {L['alone']['epochs']:.0f}.",
        "• Interpretation: 23 property-regression tasks teach the encoder what an ordinary composition looks like; the head then only has to draw a tight boundary around the 99 % class. A model trained alone must learn that representation from 367 positive examples, and inflates the rare-class regions instead.",
        "• Minimal theory (shared-representation bound, Tripuraneni · Jordan · Jin 2020): excess risk ≈ C(representation) / (n·T) + C(head) / n. With T = 23 tasks and ~23,000 rows each, the first term is already paid; the fine-tune only pays the head term. Training alone pays both from one task's labels.",
    ], size=15.5)


def slide_mt_compare():
    mtw = json.loads((S / "material_type_warmstart_runs.json").read_text())["arms"]
    ws = [r["after"]["macro_f1"] for r in mtw["warm_seen"]]; wu = [r["after"]["macro_f1"] for r in mtw["warm_unseen"]]; wa = [r["macro_f1"] for r in mtw["alone"]]
    s = new("The same experiment under the two losses", "Why the earlier reports said +22 % and this deck says a wash")
    table(s, 0.5, 1.4, 12.3, ["arm", "class weights on (earlier reports)", "class weights off (this deck)"], [
        ["trained alone (5 seeds)", f"{st.fmean(wa):.3f} ± {st.stdev(wa):.3f}", f"{MT['alone']['mean']:.3f} ± {MT['alone']['sd']:.3f}"],
        ["warm-start, encoder saw it (10)", f"{st.fmean(ws):.3f} ± {st.stdev(ws):.3f}  (+{(st.fmean(ws) / st.fmean(wa) - 1) * 100:.0f} %)", f"{MT['seen']['mean']:.3f} ± {MT['seen']['sd']:.3f}  ({(MT['seen']['mean'] / MT['alone']['mean'] - 1) * 100:+.1f} %)"],
        ["warm-start, never saw it (3)", f"{st.fmean(wu):.3f} ± {st.stdev(wu):.3f}  (+{(st.fmean(wu) / st.fmean(wa) - 1) * 100:.0f} %)", f"{MT['unseen']['mean']:.3f} ± {MT['unseen']['sd']:.3f}  ({(MT['unseen']['mean'] / MT['alone']['mean'] - 1) * 100:+.1f} %)"],
    ], col_w=[4.0, 4.15, 4.15], size=14)
    bullets(s, ["The weighted loss made training alone worse (0.571), not transfer better: every arm improves once the weights are off, and the from-scratch arm improves most.",
                "Same encoders, same test rows, same fine-tune recipe in both columns; only the loss weighting of the material_type head differs."], y=3.7, size=15, h=2.5)


def slide_added_transfer_table():
    d = json.loads((S / "added_transfer.json").read_text()); rows = []
    for r in sorted([r for r in d["per_task"] if r.get("warm")], key=lambda r: -(r["relative_pct"] or 0)):
        rows.append([r["task"].replace("_", " "), "macro-F1" if r["kind"] == "classification" else "R²", f"{r['alone']['mean']:.3f} ± {r['alone']['sd']:.3f}", f"{r['warm']['mean']:.3f} ± {r['warm']['sd']:.3f}",
                     f"{r['relative_pct']:+.1f} %", f"{2 * r['se_of_difference']:.3f}", r["verdict"], f"{st.fmean(r['epochs']):.0f}" if r["epochs"] else "—"])
    c = d["counts"]
    s = new("Transfer to the 17 added Materials Project tasks — the numbers", f"Warm-start from five 24-task library encoders that never saw the task vs training alone (5 seeds): {c['better']} better / {c['worse']} worse / {c['unresolved']} unresolved")
    table(s, 0.4, 1.35, 12.5, ["task", "metric", "alone (5 seeds)", "warm-start (5 encoders)", "Δ rel.", "2×SE", "verdict", "epochs"], rows, col_w=[2.7, 1.1, 1.9, 2.1, 1.1, 0.9, 1.4, 0.9], size=11, head_size=11.5)


def slide_added_transfer_setup():
    s = new("Transfer to the 17 added Materials Project tasks — the setup", "The library's real use case: a property the encoder has never seen")
    bullets(s, [
        "• Encoders: five 24-task library checkpoints (five random pretraining orderings, the 2026-05-15 labels) — none of the 17 added tasks was in that pretraining.",
        "• Fine-tune: fm finetune with a fresh head (add_new_tasks), encoder + head trained, unweighted cross-entropy for the classification heads, early stopping on the task's own validation loss, cap 150 epochs — the same recipe and rows as the from-scratch baselines on the 2026-09-11 dataset.",
        "• Comparison: mean over 5 encoders vs mean over 5 seeds trained alone; 2×SE of the difference from both arms; better / worse needs separation at 2×SE and |Δ| ≥ 0.01.",
        "• What it tests: whether a representation learned from 24 other properties transfers to new properties of the same compositions — the case a released model library is for.",
    ], size=16)


def slide_lowdata_material_type():
    mt = json.loads((S / "lowdata_material_type.json").read_text())
    rows = []
    for f, lab in (("f10", "10 % (≈ 3,400 rows; 16 QC, 9 AC)"), ("f25", "25 % (≈ 8,600 rows; 40 QC, 24 AC)"), ("f50", "50 % (≈ 17,000 rows; 80 QC, 48 AC)"), ("f100", "100 % (34,322 rows; 162 QC, 94 AC)")):
        a = mt[f]["alone"]; w = mt[f]["warm"]; m = lambda rs, k: st.fmean(r[k] for r in rs); sd = lambda rs, k: st.stdev(r[k] for r in rs) if len(rs) > 1 else 0
        rows.append([lab, f"{m(a, 'f1_3'):.3f} ± {sd(a, 'f1_3'):.3f} ({len(a)})", f"{m(w, 'f1_3'):.3f} ± {sd(w, 'f1_3'):.3f} ({len(w)})", f"{m(w, 'f1_3') - m(a, 'f1_3'):+.3f}", f"{m(a, 'f1_5'):.3f} → {m(w, 'f1_5'):.3f}", f"{m(a, 'QC_recall'):.2f} → {m(w, 'QC_recall'):.2f}"])
    s = new("material_type at low data: warm-start vs alone", "Training labels kept at 10 / 25 / 50 / 100 % (test split unchanged, 7,357 rows); class weights off; the encoders never saw material_type")
    table(s, 0.5, 1.4, 12.3, ["training labels kept", "alone, 3-class F1", "warm-start, 3-class F1", "Δ", "5-class F1 alone → warm", "QC recall alone → warm"], rows, col_w=[3.4, 2.1, 2.3, 0.8, 2.0, 1.7], size=12.5)
    bullets(s, [
        "• At 10 % of the labels the two arms are level on the mean, but the warm-started runs are far less scattered (sd 0.06 vs 0.15): the pretrained encoder makes a 16-QC / 9-AC training set trainable at all.",
        "• At 25 % warm-start is ahead by 0.02–0.03 (3-class) and by 0.1 on the noisy 5-class number; from 50 % up the arms are level, and at 100 % alone is ahead by 0.04 (n = 5 vs 3).",
        "• So the encoder's contribution to material_type is a low-data effect on stability and rare-class recall, and it fades once a few hundred QC / AC rows are available — the pattern the representation bound predicts.",
    ], y=4.0, size=14.5, h=3.2)


def slide_lowdata_added():
    d = json.loads((FIG / "lowdata_deltas.json").read_text())
    by = {}
    for t, f, delta, se in d:
        by.setdefault(f, []).append((t, delta, se))
    rows = []
    for f in sorted(by):
        items = by[f]; better = [t for t, dd, se in items if se is not None and dd > 2 * se and dd >= 0.01]; worse = [t for t, dd, se in items if se is not None and dd < -2 * se and dd <= -0.01]
        rows.append([f"{f} %", str(len(items)), str(sum(1 for _, dd, _ in items if dd > 0)), str(len(better)), str(len(worse)), ", ".join(t.replace("_", " ") for t in better) or "—"])
    s = new("The 17 added tasks at low data — where warm-start wins", "Per fraction of training labels: how many tasks warm-start beats training alone, and how many are separated at 2×SE with |Δ| ≥ 0.01")
    table(s, 0.5, 1.4, 12.3, ["labels kept", "tasks", "warm > alone (any margin)", "better (separated)", "worse (separated)", "separated gains"], rows, col_w=[1.4, 0.9, 2.3, 1.9, 1.9, 3.9], size=12.5)
    bullets(s, ["Read with the figure on the previous slide: the gains concentrate at 5–10 % of the labels and on the tasks with a clear composition signal; at 100 % the count is 1 better / 5 worse / 11 unresolved.",
                "Three subsample seeds per point (5 seeds / 5 encoders at 100 %); the same test rows throughout."], y=4.2, size=14, h=2.5)


# ---- Part 3
def slide_data_plan():
    s = new("Data plan — catalysts, high-entropy alloys, MOFs", "2–3 sets per family, ranked by fit for a composition-only model, adoption, and diversity; status = proposal")
    rows = [
        ["CATALYSTS", "", "", "", ""],
        ["Catalysis-Hub alloy adsorption set (Mamun et al. 2019)", "≈ 2,000 alloys × 11 adsorbates", "adsorption energies; one functional", "high — 11 regression tasks from composition", "IMPORT NOW"],
        ["OCM high-throughput set (Nguyen et al. 2020)", "300 catalysts, 12,708 points", "C₂ yield, conversion, selectivity vs T; one rig", "high — temperature curves; support convention needed", "IMPORT NOW"],
        ["OCx24 (Meta 2024)", "572 samples", "HER / CO₂RR voltage, Faradaic efficiency (exp.)", "medium — small; FE vs current as a curve", "EVALUATING"],
        ["HIGH-ENTROPY ALLOYS", "", "", "", ""],
        ["KKR-CPA equiatomic quaternary HEAs (Fukushima et al. 2022, PRMaterials; NIMS MDR / Zenodo)", "147,630 alloys, 38 elements", "total energy, magnetization, Curie T, residual resistivity; one method", "high — composition is the whole input; one level of theory", "IMPORT NOW (priority)"],
        ["MPEA mechanical compilation (Borg et al., Sci. Data 2020)", "630 alloys", "hardness, yield strength, elongation, phases (exp.)", "medium — processing state to fix", "EVALUATING"],
        ["HEA / CCA mechanical (Gorsse et al., Data in Brief 2018)", "≈ 370 alloys", "tensile properties, hardness (exp.)", "medium — small; overlaps Borg 2020", "EVALUATING"],
        ["MOFs", "", "", "", ""],
        ["QMOF (Rosen et al., CC BY 4.0)", "20,375 MOFs", "PBE band gap, energies; one workflow", "high — new chemical space for band_gap", "IMPORT NOW"],
        ["MOFSimplify (Kulik group, MIT)", "≈ 3,000 + ≈ 2,000", "decomposition T, solvent-removal stability (exp.)", "medium — composition via CSD refcode", "EVALUATING"],
        ["CoRE MOF 2025 (Zenodo)", "43,439 structures", "density, pore metrics (computed)", "medium-low — pore labels are topology", "EVALUATING"],
    ]
    tbl = table(s, 0.3, 1.2, 12.75, ["dataset", "size", "labels / consistency", "fit for a composition-only model", "status"], rows,
                col_w=[3.9, 2.1, 3.0, 2.6, 1.15], size=10.5, head_size=11, section_rows=(0, 4, 8))
    for i in (1, 5, 9):
        tbl.rows[i].height = Inches(0.3)


def slide_data_notes():
    s = new("Data plan — the two decisions before importing", "Both follow from the input being a composition string")
    bullets(s, [
        "• Catalysts: supports and promoters (SiO₂, Al₂O₃, alkali dopants) — fold them into the composition string, or keep only the active components and hold the support fixed. The choice sets what the descriptor sees; the OCM set needs it before import.",
        "• MOFs: the organic part dominates the atomic fractions, so MOFs look alike to KMD. An auxiliary label (metal node type) or a per-node composition view may be needed; QMOF is the test case.",
        "• HEAs: the KKR-CPA set is disordered and equiatomic — composition is the entire specification, which is exactly the input this model takes; it also brings magnetization and Curie temperature computed on one footing, comparable with our NEMAD tasks only as separate columns (different provenance, never pooled).",
        "• Every import follows the 2026-09-11 standard: loader from a notebook, versioned parquet, provenance with the value, four validation gates, single-task baseline before the task joins the multi-task set. Nothing here has been downloaded yet; sizes are as published.",
    ], size=16)


# ---- Part 4: flowchart
def _box(s, x, y, w, h, text, fill=PALE, size=13, bold=False, colour=INK):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill; shp.line.color.rgb = RGBColor(0x9C, 0xA3, 0xAF); shp.line.width = Pt(1)
    tf = shp.text_frame; tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = line; p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(size); p.font.bold = bold if i == 0 else False; p.font.color.rgb = colour
    return shp


def _arrow(s, x1, y1, x2, y2):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor(0x4B, 0x55, 0x63); c.line.width = Pt(2)
    ln = c.line._get_or_add_ln()
    ln.append(etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", type="triangle", w="med", len="med"))
    return c


def slide_roadmap():
    s = new("Roadmap: from the pretrained model library to hosted and local services", "Proposal — what a user can do with the released models")
    W = RGBColor(0xFF, 0xFF, 0xFF)
    # lane labels
    for y, lab in ((1.35, "Model base"), (3.55, "Web app"), (5.75, "Package + AI")):
        txt(s, 0.3, y, 1.6, 0.4, [lab], size=12, color=MUT, bold=True)
    # lane 1: library -> online service / download
    lib = _box(s, 0.4, 1.75, 2.9, 1.3, "Pretrained model library\n(foundation base: encoder + task heads)", fill=TEAL, size=13, bold=True, colour=W)
    _box(s, 4.3, 1.55, 2.9, 0.8, "Online prediction service\n(API)", size=12)
    _box(s, 4.3, 2.55, 2.9, 0.8, "Download, run locally", size=12)
    _arrow(s, 3.3, 2.2, 4.3, 1.95); _arrow(s, 3.3, 2.5, 4.3, 2.95)
    # lane 2: web app chain
    _box(s, 0.4, 3.95, 2.9, 1.1, "Web app: model preview\n(browse tasks, try compositions)", size=12)
    _box(s, 3.7, 3.95, 2.9, 1.1, "Upload your own data\n→ AI-assisted fine-tuning", size=12)
    _box(s, 7.0, 3.95, 2.6, 1.1, "Inspect and preview\nthe fine-tuned model online", size=12)
    _box(s, 10.0, 3.6, 2.9, 0.75, "Download the model", size=12)
    _box(s, 10.0, 4.55, 2.9, 0.75, "Host it online as a\nprediction service", size=12)
    _arrow(s, 1.85, 3.05, 1.85, 3.95)  # library -> web app
    _arrow(s, 3.3, 4.5, 3.7, 4.5); _arrow(s, 6.6, 4.5, 7.0, 4.5); _arrow(s, 9.6, 4.35, 10.0, 3.98); _arrow(s, 9.6, 4.65, 10.0, 4.92)
    # lane 3: package + AI
    _box(s, 0.4, 6.05, 3.6, 1.1, "foundation-model package\nwith preset skills (predict, fine-tune, inverse design)", size=12)
    _box(s, 4.6, 6.05, 2.6, 1.1, "AI agent / LLM\n(uses the skills)", fill=PALE2, size=12)
    _box(s, 7.8, 5.85, 2.4, 0.7, "Local model service", size=12)
    _box(s, 7.8, 6.65, 2.4, 0.7, "Online service via API", size=12)
    _arrow(s, 1.85, 5.05, 1.85, 6.05)  # web app -> package (same models)
    _arrow(s, 4.0, 6.6, 4.6, 6.6); _arrow(s, 7.2, 6.45, 7.8, 6.2); _arrow(s, 7.2, 6.75, 7.8, 7.0)
    _arrow(s, 5.75, 2.35, 5.9, 6.05)  # online service <-> agent (dashed would be nicer; keep simple)
    txt(s, 10.4, 6.0, 2.7, 1.3, ["Skills work the same locally and through the API; the LLM plans, the model computes."], size=11, color=MUT)


def slide_status():
    s = new("Status and sources", f"Built {DATE}; every number traces to a run on RIKYU")
    bullets(s, [
        "• Baselines: stage_single (unchanged tasks, 2026-08) and stage_single_mp2026 (relabelled + added tasks) — summary/baselines_mp2026.json, ceilings_adopted_v2.json, classification_weights.json, material_type_weights.json, space_group_confirm.json.",
        "• Transfer: stage_xfer (240 continual runs, position curve), stage_ft ftf / ftfu (warm-start, 13 runs), stage_xu (never-seen encoders) — summary/material_type_warmstart_runs.json, position_runs.json.",
        "• Dataset: data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet (reported as 2026-09-11) with its CHANGES note and rebuild script; NEMAD and phonix-db parquets as before.",
        "• Evidence pages: transferability (four ways), two easy tasks that would not train, space group 0.24 vs 0.60. HANDOFF.md experiments 1–15.",
        "• Open: never-seen arm to 5 orderings; phase-B re-run on the 0.4.x image; the three IMPORT NOW datasets; HEA loader for the KKR-CPA set.",
    ], size=14)


def main():
    slide_title()
    # ---- Part 1
    divider("Part 1 — Tasks and single-task performance", "41 tasks over five data sources; the same recipe, descriptor and five seeds for every task")
    slide_dataset()
    mp = [t for t in DESC if DESC[t][0] == "Materials Project"]
    slide_inventory("Tasks 1/5 — Materials Project: energies and structure", mp[:9], "DFT, GGA / GGA+U; rebuilt 2026-09-11")
    slide_inventory("Tasks 2/5 — Materials Project: electronic and magnetic", mp[9:18], "DFT, GGA / GGA+U; rebuilt 2026-09-11")
    slide_inventory("Tasks 3/5 — Materials Project: dielectric, elastic, symmetry", mp[18:], "DFT, GGA / GGA+U; rebuilt 2026-09-11")
    curves = [t for t in DESC if DESC[t][0] == "thermoelectric (starry)"]
    rest = [t for t in DESC if DESC[t][0] not in ("Materials Project", "thermoelectric (starry)")]
    slide_inventory("Tasks 4/5 — thermoelectric curves (starry)", curves, "Experimental curves; one curve per composition, learned as kernel regression over the coordinate t")
    slide_inventory("Tasks 5/5 — NEMAD, phonon transport, quasicrystals", rest, "Text-mined experimental databases, first-principles phonon transport, the quasicrystal classification")
    pic_slide("Single-task performance, all 41 tasks", "Bars = mean over 5 seeds, whiskers = sd; the number in brackets is the training rows", FIG / "overview_bar.png")
    pic_slide("Single-task R² against training rows", "Data volume alone does not order the tasks: the weakest are cell-scale labels and noisy experimental ones, not the smallest", FIG / "r2_vs_n.png")
    reg = [r["task"] for r in sorted(PERF, key=lambda r: -r["mean"]) if r["metric"] == "R²"]
    chunks = [reg[i:i + 8] for i in range(0, len(reg), 8)]
    for i, ch in enumerate(chunks, 1):
        slide_perf_table(f"Single-task metrics {i}/{len(chunks) + 1} — regression and curve tasks, ranked by R²", ch, "R² and MAE on the normalised scale, test split, mean ± sd over 5 seeds")
    slide_perf_table(f"Single-task metrics {len(chunks) + 1}/{len(chunks) + 1} — classification heads", [r["task"] for r in PERF if r["metric"] == "macro-F1"], "macro-F1 and accuracy, 5 seeds")
    n_ex = len(list(FIG.glob("scatter_existing_*.png"))); n_ad = len(list(FIG.glob("scatter_added_*.png")))
    for k in range(1, n_ex + 1):
        pic_slide(f"Observed vs predicted — existing tasks ({k}/{n_ex})", "One seed (2025), test split; dashed = y = x; curve tasks show sampled (t, value) points", FIG / f"scatter_existing_{k}.png")
    for k in range(1, n_ad + 1):
        pic_slide(f"Observed vs predicted — tasks added 2026-09-11 ({k}/{n_ad})", "One seed (2025), test split; dashed = y = x", FIG / f"scatter_added_{k}.png")
    pic_slide("Added classification heads: confusion matrices", "Seed 2025; cell = share of the true row and the row count", FIG / "confusion_clf.png")
    slide_space_group_why()
    pic_slide("Space group: confusion over the twelve largest groups", "151 classes; the remaining 139 folded into “other”; cubic and hexagonal groups are recognised, monoclinic ones partly", FIG / "confusion_sg.png")
    pic_slide("Space group: per-class F1 against class size", "Groups with hundreds of examples are learned; groups with tens mostly are not", FIG / "sg_f1_vs_size.png")
    slide_kmd()
    # ---- Part 2
    divider("Part 2 — material_type transfer", "Warm-start fine-tuning with the encoder trained; with and without the target in pretraining")
    pic_slide("material_type: the task, trained alone", "Five classes, 99 % “others”; the rare classes are approximant crystals (DAC, IAC) and quasicrystals (DQC, IQC)", FIG / "confusion_material_type.png")
    slide_mt_setup(); slide_mt_numbers()
    pic_slide("material_type: every run of the three arms", "Warm-start fine-tuning with the encoder trained; the dashed line is the alone mean", FIG / "mt_warmstart_strip.png")
    pic_slide("material_type: before and after the fine-tune", "The never-seen encoder starts from a random head and ends within the spread of the seen one", FIG / "mt_before_after.png")
    pic_slide("material_type: training and validation loss, alone vs warm-start", "Median over runs with the inter-quartile band; the warm-started runs start lower and stop earlier at the same floor", FIG / "mt_losses.png")
    pic_slide("material_type: where the gain comes from", "Per-class recall and precision on the 7,354 test rows, mean over runs; the rare classes are approximants (DAC, IAC) and quasicrystals (DQC, IQC)", FIG / "mt_perclass.png")
    slide_mt_why()
    pic_slide("material_type against pretraining breadth (continual arm)", "From the 240-run transfer stage: the later material_type appears in the sequence, the better — the opposite of every regression task", FIG / "mt_position.png")
    # ---- Part 3
    divider("Part 3 — Data plan", "Catalysts, high-entropy alloys, MOFs: what is selected, what is still being evaluated")
    slide_data_plan(); slide_data_notes()
    # ---- Part 4
    divider("Part 4 — Release and LLM access", "The pretrained library as a base for hosted and local services (proposal)")
    slide_roadmap(); slide_status()
    out = HERE / "results" / f"DECK_{DATE.replace('-', '')}.pptx"
    prs.save(str(out)); print(f"{out}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
