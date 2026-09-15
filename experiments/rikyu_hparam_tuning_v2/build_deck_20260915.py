#!/usr/bin/env python3
"""Build results/DECK_20260915.pptx — tasks & single-task performance, material_type transfer,
the data plan, and the release / LLM proposal.

Numbers come from results/deck_20260915/performance_table.json and material_type_numbers.json
(written by analysis/deck_figures.py) and from the summary JSONs; figures from results/deck_20260915.
One idea per slide; the reader deletes what is not needed.

    uv run --with python-pptx --with pillow python experiments/rikyu_hparam_tuning_v2/build_deck_20260915.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from pptx.util import Inches, Pt

from build_report_pptx import BLANK, GREEN, INK, MUT, RED, WHITE, new, pic_slide, prs, table, txt  # noqa: F401

HERE = Path(__file__).resolve().parent
S = HERE / "summary"
FIG = HERE / "results" / "deck_20260915"
DATE = "2026-09-15"
PERF = json.loads((FIG / "performance_table.json").read_text())
MT = json.loads((FIG / "material_type_numbers.json").read_text())
INV = {t["name"]: t for t in json.loads((S / "task_inventory_20260911.json").read_text())}

# ------------------------------------------------------------------ task descriptions
# (source, what it is, unit of the raw label; every label is trained on its normalised form)
DESC = {
    # Materials Project — DFT (GGA / GGA+U) properties of stable inorganic crystals, rebuilt 2026-09-11
    "final_energy": ("Materials Project", "DFT total energy per atom (GGA / GGA+U scheme)", "eV / atom"),
    "formation_energy": ("Materials Project", "Formation energy per atom relative to the elements", "eV / atom"),
    "reaction_energy": ("Materials Project", "Equilibrium reaction energy: thermodynamic margin to the nearest competing phases", "eV / atom"),
    "volume": ("Materials Project", "Volume of the cell as stored (KMD-limited: the descriptor cannot see the cell size)", "Å³"),
    "density": ("Materials Project", "Mass density of the crystal", "g / cm³"),
    "density_atomic": ("Materials Project", "Volume per atom (the intensive form of volume)", "Å³ / atom"),
    "efermi": ("Materials Project", "Fermi energy", "eV"),
    "band_gap": ("Materials Project", "Electronic band gap from the GGA band structure", "eV"),
    "cbm": ("Materials Project", "Conduction-band minimum (non-metals only)", "eV"),
    "vbm": ("Materials Project", "Valence-band maximum (non-metals only)", "eV"),
    "is_metal": ("Materials Project", "Metal or not (gap = 0) — 2 classes", "—"),
    "is_gap_direct": ("Materials Project", "Direct or indirect gap (non-metals) — 2 classes", "—"),
    "total_magnetization": ("Materials Project", "Net magnetic moment of the cell (KMD-limited, per-cell quantity)", "μB / cell"),
    "magnetization_per_volume": ("Materials Project", "Net magnetic moment per unit volume", "μB / Å³"),
    "magnetization_per_fu": ("Materials Project", "Net magnetic moment per formula unit", "μB / f.u."),
    "magnetic_ordering": ("Materials Project", "Magnetic ordering of the ground state: NM / FM / FiM / AFM — 4 classes", "—"),
    "dielectric_total": ("Materials Project", "Total dielectric constant (DFPT)", "—"),
    "dielectric_ionic": ("Materials Project", "Ionic contribution to the dielectric constant", "—"),
    "dielectric_electronic": ("Materials Project", "Electronic contribution to the dielectric constant", "—"),
    "refractive_index": ("Materials Project", "Refractive index (√ of the electronic dielectric constant)", "—"),
    "bulk_modulus": ("Materials Project", "Bulk modulus, Voigt–Reuss–Hill average of the elastic tensor", "GPa"),
    "shear_modulus": ("Materials Project", "Shear modulus, Voigt–Reuss–Hill average", "GPa"),
    "poisson_ratio": ("Materials Project", "Poisson ratio from the elastic tensor", "—"),
    "universal_anisotropy": ("Materials Project", "Universal elastic anisotropy index", "—"),
    "piezoelectric_max": ("Materials Project", "Largest component of the piezoelectric tensor (DFPT)", "C / m²"),
    "space_group": ("Materials Project", "Space group of the relaxed structure — 151 classes with ≥ 10 rows", "—"),
    # thermoelectric curves (starry): one curve per composition, kernel regression over a coordinate t
    "seebeck": ("thermoelectric (starry)", "Seebeck coefficient S(T) — curve over temperature", "V / K"),
    "electrical_resistivity": ("thermoelectric (starry)", "Electrical resistivity ρ(T) — curve over temperature", "Ω·m"),
    "thermal_conductivity": ("thermoelectric (starry)", "Thermal conductivity κ(T) — curve over temperature", "W / (m·K)"),
    "power_factor": ("thermoelectric (starry)", "Power factor S²σ(T) — curve over temperature", "W / (m·K²)"),
    "zt": ("thermoelectric (starry)", "Thermoelectric figure of merit ZT(T) — curve over temperature", "—"),
    "magnetic_susceptibility": ("thermoelectric (starry)", "Magnetic susceptibility χ(T) — curve over temperature (58 compositions)", "A·m² / mol"),
    "dos_density": ("thermoelectric (starry)", "Electronic density of states — curve over energy", "states / eV"),
    # NEMAD (text-mined experimental magnetic / superconductor databases)
    "magnetization": ("NEMAD magnetic", "Magnetization (experimental)", "A·m² / kg"),
    "magnetic_moment": ("NEMAD magnetic", "Magnetic moment per formula unit (experimental)", "μB / f.u."),
    "curie": ("NEMAD magnetic", "Curie temperature", "K"),
    "neel": ("NEMAD magnetic", "Néel temperature", "K"),
    "tc": ("NEMAD superconductor", "Superconducting transition temperature", "K"),
    # phonix-db (first-principles phonon transport)
    "kp": ("phonix-db", "Particle-like (Peierls) part of the lattice thermal conductivity, κ_p", "W / (m·K)"),
    "klat": ("phonix-db", "Lattice thermal conductivity, κ_lat", "W / (m·K)"),
    # quasicrystal classification
    "material_type": ("quasicrystal (qa/starry)", "DAC / DQC / IAC / IQC / others — approximant crystals, quasicrystals, everything else (5 classes, 99 % “others”)", "—"),
}
KIND_LABEL = {"regression": "regression", "kernel_regression": "curve (kernel regression over t)", "classification": "classification"}


def perf(task):
    return next(r for r in PERF if r["task"] == task)


def fmt_n(v):
    return "—" if v is None else f"{v:,}"


# ------------------------------------------------------------------ slides
def slide_title():
    s = prs.slides.add_slide(BLANK)
    txt(s, 0.7, 2.2, 12, 1.2, ["Composition-only foundation model: tasks, transfer, data plan, release"], size=32, bold=True)
    txt(s, 0.7, 3.5, 12, 1.5, [f"Status deck, {DATE} — dataset 2026-09-11, KMD descriptor, stage_single recipe, class weights off",
                               "1 · What the model is trained on and how each task performs alone",
                               "2 · material_type transfer: warm-start fine-tuning, with and without the target in pretraining",
                               "3 · Data plan: catalysts, high-entropy alloys, MOFs",
                               "4 · Pretrained-model release and LLM access (proposal)"], size=14, color=MUT)


def slide_agenda():
    s = new("How to read this deck", "One idea per slide; delete freely")
    txt(s, 0.6, 1.4, 12, 5, [
        "Part 1 — 41 tasks over five data sources. Every task has a single-task baseline: the same recipe, KMD descriptor, five seeds, early stopping, test split.",
        "   Tasks whose labels changed or were added in the 2026-09-11 rebuild were re-measured on it; tasks whose rows and labels did not change keep their measured baselines.",
        "   Classification heads are quoted with class weights OFF (the 2026-09-11 decision); the change against the old weighted numbers is on its own slide.",
        "Part 2 — material_type transfer. Only warm-start fine-tuning with the encoder trained is shown; the key comparison is an encoder that saw the task during pretraining vs one that never did.",
        "Part 3 — the datasets surveyed for catalysts, high-entropy alloys and MOFs, split into import-now and still-evaluating; HEA has not been surveyed yet.",
        "Part 4 — release plan and LLM access. This part is a proposal, not a measurement.",
        "",
        "Conventions: ± is the sd over seeds; 2×SE is twice the standard error of a difference; every regression label is trained on its normalised form, so R² and MAE are on that scale.",
    ], size=13)


def slide_dataset():
    s = new("The 2026-09-11 dataset", "One row per composition; every task is a column, missing where the source has no value")
    rows = [
        ["Materials Project (stable entries, GGA / GGA+U only)", "33,829", "energies, structure, electronic, magnetic, elastic, dielectric, space group", "26 tasks"],
        ["Thermoelectric curves (starry)", "14,838", "S, ρ, κ, PF, ZT, χ over temperature; DOS over energy", "7 curve tasks"],
        ["Quasicrystal set (qa / starry)", "49,034 (all rows carry a label)", "material type: DAC / DQC / IAC / IQC / others", "1 task"],
        ["NEMAD magnetic (text-mined, experimental)", f"{fmt_n(perf('curie')['n_train'])} train (Curie)", "magnetization, moment, Curie, Néel", "4 tasks"],
        ["NEMAD superconductor", f"{fmt_n(perf('tc')['n_train'])} train", "transition temperature", "1 task"],
        ["phonix-db (phonon transport)", f"{fmt_n(perf('klat')['n_train'])} train", "κ_p, κ_lat", "2 tasks"],
    ]
    table(s, 0.5, 1.4, 12.3, ["source", "compositions", "what it carries", "tasks"], rows, col_w=[4.0, 2.4, 4.6, 1.3], size=11)
    txt(s, 0.5, 4.6, 12.3, 2.4, [
        "Rebuild of the Materials Project part (2026-09-11): one level of theory per column family (thermo GGA_GGA+U; structure and magnetism from the GGA-family task; electronic values only where MP's origin is GGA-family),",
        "per-cell quantities rescaled to the dataset's cell, provenance recorded, versioned files never overwritten. It fixed final_energy (R² 0.77 → 0.999) and added 17 properties.",
        "Input to the model: the composition string only, turned into the KMD descriptor (464 columns, invertible — the property the inverse-design path needs).",
    ], size=12, color=MUT)


def slide_inventory(title, tasks, sub):
    s = new(title, sub)
    rows = []
    for t in tasks:
        src, what, unit = DESC[t]; inv = INV[t]; n = inv["n"] or {}
        rows.append([t.replace("_", " "), what, unit, KIND_LABEL[inv["kind"]], fmt_n(n.get("train") or perf(t)["n_train"]), fmt_n(n.get("test") or perf(t)["n_test"])])
    table(s, 0.4, 1.35, 12.5, ["task", "what it is", "unit", "kind", "train rows", "test rows"], rows, col_w=[2.2, 5.6, 1.2, 1.7, 0.9, 0.9], size=9.5, head_size=10)


def slide_perf_table(title, tasks, sub):
    s = new(title, sub)
    rows = []
    for t in tasks:
        r = perf(t)
        if r["metric"] == "R²":
            rows.append([t.replace("_", " "), "R²", f"{r['mean']:.4f} ± {r['sd']:.4f}", f"{r.get('mae', float('nan')):.4f}" if r.get("mae") else "—", fmt_n(r["n_test"]), r["status"]])
        else:
            rows.append([t.replace("_", " "), "macro-F1", f"{r['mean']:.4f} ± {r['sd']:.4f}", f"acc {r['accuracy']:.4f}", fmt_n(r["n_test"]), r["status"]])
    table(s, 0.4, 1.35, 12.5, ["task", "metric", "mean ± sd (5 seeds)", "MAE / accuracy", "test rows", "measured on"], rows, col_w=[2.3, 1.0, 2.2, 1.6, 1.0, 4.4], size=9.5, head_size=10)


def slide_clf_change():
    s = new("Classification heads: class weights off", "Inverse-frequency weights were on for every classification head until 2026-09-11; switching them off helps every head on the aggregate metric")
    cw = json.loads((S / "classification_weights.json").read_text())["tasks"]
    mt = json.loads((S / "material_type_weights.json").read_text())["arms"]
    sg = json.loads((S / "space_group_confirm.json").read_text())["arms"]
    def m(rs, k): return st.fmean(r[k] for r in rs)
    rows = [["material_type (5 classes, 99 % majority)", f"{m(mt['balanced']['runs'], 'macro_f1'):.3f}", f"{m(mt['none']['runs'], 'macro_f1'):.3f}", f"{m(mt['balanced']['runs'], 'accuracy'):.4f} → {m(mt['none']['runs'], 'accuracy'):.4f}", "rare-class recall unchanged, precision doubles"],
            ["space_group (151 classes)", f"{m(sg['balanced_kmd']['runs'], 'macro_f1'):.3f}", f"{m(sg['plain_kmd']['runs'], 'macro_f1'):.3f}", f"{m(sg['balanced_kmd']['runs'], 'accuracy'):.3f} → {m(sg['plain_kmd']['runs'], 'accuracy'):.3f}", "the large groups become learnable"],
            ["magnetic_ordering (4 classes)", f"{m(cw['magnetic_ordering']['arms']['balanced']['runs'], 'macro_f1'):.3f}", f"{m(cw['magnetic_ordering']['arms']['none']['runs'], 'macro_f1'):.3f}", f"{m(cw['magnetic_ordering']['arms']['balanced']['runs'], 'accuracy'):.3f} → {m(cw['magnetic_ordering']['arms']['none']['runs'], 'accuracy'):.3f}", "AFM recall 0.56 → 0.14 (118 test rows)"],
            ["is_metal (2 balanced classes)", f"{m(cw['is_metal']['arms']['balanced']['runs'], 'macro_f1'):.3f}", f"{m(cw['is_metal']['arms']['none']['runs'], 'macro_f1'):.3f}", "unchanged", "weights ≈ 1"],
            ["is_gap_direct (13 % direct)", f"{m(cw['is_gap_direct']['arms']['balanced']['runs'], 'macro_f1'):.3f}", f"{m(cw['is_gap_direct']['arms']['none']['runs'], 'macro_f1'):.3f}", f"{m(cw['is_gap_direct']['arms']['balanced']['runs'], 'accuracy'):.3f} → {m(cw['is_gap_direct']['arms']['none']['runs'], 'accuracy'):.3f}", "direct-gap recall 0.79 → 0.25; threshold at inference if needed"]]
    table(s, 0.5, 1.4, 12.3, ["head", "macro-F1 with weights", "macro-F1 without", "accuracy", "what moves"], rows, col_w=[3.3, 1.7, 1.7, 2.0, 3.6], size=11)
    txt(s, 0.5, 4.3, 12.3, 2.5, ["Five seeds per arm, same recipe, same rows. Decision (2026-09-11): class_weights = \"none\" on every classification head; the knob is PR #57 (version 0.4.1).",
                                 "Every material_type transfer number in Part 2 was measured with the weights ON in every arm (baseline 0.571); against the new 0.834 baseline the transfer gain is unmeasured."], size=12, color=MUT)


def slide_kmd():
    s = pic_slide("Where the descriptor limits the label", "KMD works on atomic fractions, so Fe₂O₃ and Fe₄O₆ are the same input; labels that scale with the cell cannot be learned beyond what composition implies", FIG / "kmd_scale.png", top=1.35, bottom=1.3)
    txt(s, 0.5, 6.2, 12.3, 1.2, ["Decision (2026-09-11): keep KMD — its invertibility is what inverse design needs. volume, total magnetisation and space group are reported with this caveat; the intensive forms (volume per atom 0.979, magnetisation per volume) are the ones to use."], size=12, color=MUT)


def slide_material_type_setup():
    s = new("material_type transfer: the setup", "Only warm-start fine-tuning with the encoder trained is shown")
    table(s, 0.5, 1.4, 12.3, ["arm", "encoder comes from", "head", "what trains", "n"], [
        ["trained alone", "random init", "fresh", "encoder + head on material_type only", "5 seeds"],
        ["warm-start, encoder SAW material_type", "23 tasks + material_type pretrained continually (material_type last, with replay)", "the head trained at that step", "encoder + head, fm finetune, replay off, early stopping on material_type's own loss", "10 orderings"],
        ["warm-start, encoder NEVER saw it", "the same orderings stopped one step earlier (23 tasks, material_type dropped)", "fresh", "encoder + head, same fine-tune", "3 orderings"],
    ], col_w=[2.9, 4.2, 1.8, 2.6, 0.8], size=10.5)
    txt(s, 0.5, 3.8, 12.3, 3, [
        f"Metric: macro-F1 over the five classes on the same {MT['test_rows']:,} test rows for every arm. Fine-tune cap 150 epochs, patience 24, last-epoch weights, encoder lr 2e-3, head lr 5e-3.",
        "The question the third arm answers: does the encoder need to have met the task during pretraining, or is the 23-task representation what transfers?",
        "Caveat: all three arms used the inverse-frequency class weights that were the default at the time (alone = 0.571). Without them the alone baseline is 0.834; the transfer has to be re-measured on that footing.",
    ], size=12, color=MUT)


def slide_material_type_numbers():
    s = new("material_type transfer: the numbers", "Warm-start beats training alone by 22 %, whether or not the encoder ever saw the task")
    a, se, un = MT["alone"], MT["seen"], MT["unseen"]
    def row(name, v, base=None):
        m = st.fmean(v); sd = st.stdev(v); r = [name, f"{m:.4f} ± {sd:.4f}", str(len(v))]
        if base is not None:
            bm = st.fmean(base); d = m - bm; se2 = 2 * ((sd ** 2 / len(v)) + (st.stdev(base) ** 2 / len(base))) ** 0.5
            r += [f"{d:+.4f} ({d / bm * 100:+.1f} %)", f"{se2:.4f}", "separated" if abs(d) > se2 else "unresolved"]
        else:
            r += ["—", "—", "reference"]
        return r
    table(s, 0.5, 1.4, 12.3, ["arm", "macro-F1 (mean ± sd)", "n", "vs alone", "2×SE", "verdict"],
          [row("trained alone", a), row("warm-start, encoder saw material_type", se, a), row("warm-start, encoder never saw it", un, a)], col_w=[3.8, 2.2, 0.6, 2.0, 1.2, 1.5], size=11)
    d = st.fmean(se) - st.fmean(un); se2 = 2 * ((st.stdev(se) ** 2 / len(se)) + (st.stdev(un) ** 2 / len(un))) ** 0.5
    txt(s, 0.5, 3.3, 12.3, 3.5, [
        f"Seen vs never-seen: {st.fmean(se):.4f} vs {st.fmean(un):.4f}, difference {d:+.4f} against 2×SE {se2:.4f} — indistinguishable.",
        "So the +22 % is the 23-task representation, not the single exposure at the last pretraining step. Across all 23 tasks the same comparison is 2 better / 2 worse / 19 unresolved.",
        "Consequence for the next phase: pretrain once on the existing tasks, then warm-start fine-tune each new task; the continual replay step for the new task is the most expensive part of the pipeline and adds nothing the fine-tune does not recover.",
        "Read the never-seen arm generously: n = 3 orderings against n = 10.",
    ], size=12)


def slide_data_plan_intro():
    s = new("Data plan: what the model needs next", "Diversity of chemical space and of label type, with one level of theory or one instrument per column")
    txt(s, 0.6, 1.4, 12, 5.5, [
        "What the current data is: inorganic crystals (Materials Project, quasicrystals), thermoelectric and magnetic measurements, phonon transport. Every label is a scalar per composition or a curve over one coordinate.",
        "What is missing: metal–organic frameworks (organic–inorganic hybrids, a different composition distribution), catalysts (adsorption / reaction energies and experimental activity; often curves over temperature or current), high-entropy alloys (multi-principal-element metals, mechanical and phase labels).",
        "Rules carried over from the 2026-09-11 rebuild: one source = one level of theory or one instrument per column; no pooling of heterogeneous literature values into one column; provenance stored with the value; versioned files.",
        "Two design questions to settle before importing catalysts and MOFs: (a) supports and promoters — fold them into the composition string or hold them fixed; (b) MOFs — the organic part dominates the atomic fractions, so MOFs look alike to KMD; an auxiliary label (metal node) may be needed.",
        "Status legend on the next slides: IMPORT NOW = selected, loader to be built; EVALUATING = fits the input format but a design question or the licence / size is open; NOT PLANNED = does not fit a composition-only model.",
    ], size=13)


def slide_data_mof():
    s = new("Data plan — MOFs", "Surveyed 2026-09; sizes as published by the sources")
    rows = [
        ["QMOF (Rosen et al., figshare, CC BY 4.0)", "20,375 experimentally synthesised MOFs", "PBE band gap (all), HSE06 gap (subset), energies", "one DFT workflow", "IMPORT NOW — same task family as band_gap, new chemical space"],
        ["MOFSimplify (Kulik group, MIT licence)", "≈ 3,000 decomposition temperatures (TGA) + ≈ 2,000 solvent-removal stability labels", "experimental", "literature-mined, conditions vary", "EVALUATING — small but experimental; composition from CSD refcode"],
        ["CoRE MOF 2025 (Zenodo)", "43,439", "crystal density, pore metrics, surface area (computed)", "one workflow", "EVALUATING — density usable; pore labels are topology-driven"],
        ["hMOF / MOFX-DB", "137k hypothetical MOFs", "GCMC gas uptake", "one workflow", "NOT PLANNED — Zn/Cu nodes and near-identical linkers, compositions indistinguishable"],
        ["ODAC23 (Meta)", "8.4k MOFs, 176k adsorption energies", "CO₂ / H₂O adsorption energy (DFT)", "one workflow", "NOT PLANNED — labels depend on the adsorption site"],
    ]
    table(s, 0.4, 1.35, 12.5, ["dataset", "size", "labels", "consistency", "status"], rows, col_w=[2.7, 2.4, 2.6, 1.6, 3.2], size=9.5, head_size=10)


def slide_data_catalyst():
    s = new("Data plan — catalysts", "Surveyed 2026-09; the three IMPORT NOW sets bring new chemical space, a new label type and experimental curves")
    rows = [
        ["Catalysis-Hub bimetallic alloy set (Mamun et al. 2019)", "≈ 2,000 alloy surfaces × 11 adsorbates ≈ 37k energies", "adsorption energies of H, C, N, O, S, OH, CH, CH₂, CH₃, NH, SH (BEEF-vdW)", "one publication, one functional", "IMPORT NOW — 11 regression tasks from surface composition"],
        ["OCM high-throughput set (Nguyen et al., ACS Catal. 2020)", "300 quaternary catalysts × conditions = 12,708 points", "C₂ yield, CH₄ conversion, selectivity vs temperature", "one rig, one operator", "IMPORT NOW — temperature curves map onto the kernel-regression tasks; support handling to decide"],
        ["OCx24 (Meta, 2024, CC BY 4.0)", "572 samples, 441 electrodes", "HER / CO₂RR voltage and Faradaic efficiency at several current densities (experimental, XRF compositions)", "one pipeline", "EVALUATING — small; FE vs current density as a curve task"],
        ["TheMeCat (Sci. Data 2025, Zenodo)", "literature compilation, size to confirm", "CO₂ → methanol conversion / selectivity", "conditions vary across papers", "EVALUATING — T, P, GHSV must be handled; we have one curve coordinate"],
        ["Catalysis-Hub full database", "> 100k reaction energies", "reaction / activation energies", "MIXED functionals across publications", "NOT PLANNED as one column — usable only per publication"],
        ["OC20 / OC22", "millions of structures", "structure-level adsorption energies", "one workflow", "NOT PLANNED — aggregating to composition discards the signal"],
    ]
    table(s, 0.4, 1.35, 12.5, ["dataset", "size", "labels", "consistency", "status"], rows, col_w=[2.8, 2.3, 3.0, 1.5, 2.9], size=9, head_size=10)


def slide_data_hea():
    s = new("Data plan — high-entropy alloys", "NOT YET SURVEYED — the rows below are well-known public sets to start from, sizes not yet verified by us")
    rows = [
        ["Borg et al., Sci. Data 7, 430 (2020) — multi-principal-element alloys", "≈ 1,500 alloys (literature compilation)", "hardness, yield strength, elongation, observed phases", "experimental, conditions vary", "TO EVALUATE — candidate for mechanical-property regression + phase classification"],
        ["Gorsse et al., Data in Brief 21, 2664 (2018)", "≈ 370 HEAs / CCAs", "mechanical properties (tensile, hardness) with processing", "experimental", "TO EVALUATE — small; processing conditions are not in our input"],
        ["HEA phase-formation compilations (e.g. FCC / BCC / intermetallic labels)", "hundreds to low thousands", "single-phase vs multi-phase, crystal structure", "literature-mined", "TO EVALUATE — natural classification task; sources overlap, deduplication needed"],
    ]
    table(s, 0.4, 1.35, 12.5, ["dataset", "size", "labels", "consistency", "status"], rows, col_w=[3.4, 2.2, 2.8, 1.6, 2.5], size=9.5, head_size=10)
    txt(s, 0.5, 4.2, 12.3, 2.6, ["Why HEAs fit: compositions are the whole story (near-equimolar, 4–7 elements), so a composition-only model is the natural learner, and the KMD descriptor is invariant to how the formula is written.",
                                 "Why care is needed: the literature values depend on processing (as-cast / annealed / temperature), which is not in the input; the import has to pick one processing state per label or add it as a coordinate.",
                                 "Next step: run the same survey as for MOFs and catalysts (size, labels, composition availability, consistency, licence) and decide import-now vs evaluating."], size=12, color=MUT)


def slide_data_sequence():
    s = new("Data plan — order of work", "Each import follows the 2026-09-11 standard: loader from a notebook, versioned parquet, four validation gates, single-task baseline before it joins the multi-task set")
    table(s, 0.5, 1.4, 12.3, ["step", "what", "gate"], [
        ["1", "QMOF band gap (20k MOFs) — loader, parquet, baseline; compare with the Materials Project band-gap task", "R² of the single-task baseline; overlap check against MP compositions"],
        ["2", "Catalysis-Hub alloy adsorption energies — 11 tasks, ≈ 2,000 compositions", "surface composition convention fixed (bulk composition, facet held constant)"],
        ["3", "OCM high-throughput curves — C₂ yield vs temperature as a kernel-regression task", "support / promoter convention decided; one rig only"],
        ["4", "HEA survey → one mechanical-property set + one phase-label set", "processing state fixed or added as coordinate"],
        ["5", "Evaluating tier (OCx24, MOFSimplify, CoRE density, TheMeCat) as small experimental complements", "only if step 1–3 baselines show the chemical space is learnable"],
    ], col_w=[0.6, 7.6, 4.1], size=10.5)
    txt(s, 0.5, 4.5, 12.3, 2, ["None of these has been downloaded yet; sizes and labels are as published by the sources. Selection between IMPORT NOW and EVALUATING is a proposal for discussion."], size=12, color=MUT)


def slide_release():
    s = new("Pretrained-model release plan (proposal)", "What to release, in what order, and what each stage requires")
    table(s, 0.5, 1.4, 12.3, ["stage", "what is released", "to whom", "prerequisite"], [
        ["0 — now", "model library on RIKYU: 240 encoders + manifest (task, ordering, seed, every score); fm CLI (pretrain / finetune / predict / inverse)", "the group", "—"],
        ["1 — after the re-run", "one pretrained encoder on the 2026-09-11 dataset + warm-start recipe + single-task baselines; dataset card and the data standard", "collaborators", "phase-B re-run on the 0.4.x image (tasks fixed, class weights off)"],
        ["2 — public", "weights, KMD descriptor code, config schema, evaluation script that reproduces the baseline table; the 2026-09-11 dataset where licences allow", "public (GitHub + model hub)", "licence audit per source (MP, NEMAD, starry, phonix-db); a versioned release (0.5)"],
        ["3 — service", "prediction and inverse-design endpoints backed by the released model", "collaborators, then public", "stage 2 + the LLM interface on the next slide"],
    ], col_w=[1.6, 5.6, 2.2, 2.9], size=10.5)
    txt(s, 0.5, 4.7, 12.3, 2, ["What a user gets: composition in → 41 properties out with a per-task uncertainty (seed spread), or a target property in → candidate compositions out (the invertible KMD path).",
                               "What stays internal until stage 2: the raw source dumps and the RIKYU run trees. Everything on this slide is a proposal for discussion; nothing has been released."], size=12, color=MUT)


def slide_llm():
    s = new("LLM access (proposal)", "Expose the model as tools an LLM can call; the LLM plans, the model computes")
    table(s, 0.5, 1.4, 12.3, ["tool", "input → output", "backed by"], [
        ["describe_tasks", "— → the 41 tasks with meaning, unit, baseline R² / F1 and the KMD caveats", "task catalog + baseline table"],
        ["predict", "composition(s) → property values with seed-spread uncertainty and a validity flag", "fm predict on the released encoder + heads"],
        ["predict_curve", "composition + coordinate grid → S(T), ZT(T), DOS(E) …", "kernel-regression heads"],
        ["inverse_design", "target property window (+ element constraints) → candidate compositions", "the invertible KMD path (fm inverse)"],
        ["explain", "prediction → nearest training compositions and their measured values", "descriptor neighbours on the training set"],
    ], col_w=[1.8, 6.6, 3.9], size=11)
    txt(s, 0.5, 4.3, 12.3, 3, [
        "Interface: an MCP server (the tool protocol Claude and other agents already speak) wrapping the fm CLI; the same tools serve a REST endpoint. The LLM never touches weights — it composes calls and reads results.",
        "Guardrails the tools enforce, not the LLM: composition parsing and element coverage, out-of-distribution flag from descriptor distance, the KMD caveat attached to volume / magnetisation / space-group answers, uncertainty from the seed ensemble.",
        "Example: “find Fe-free compositions with ZT > 1 at 600 K and a band gap under 0.5 eV” → inverse_design → predict_curve on the candidates → explain on the top three.",
        "First milestone: describe_tasks + predict on the released stage-1 encoder, used internally; inverse_design once the KMD inversion is validated on held-out targets.",
    ], size=12)


def slide_status():
    s = new("Status and sources", f"Built {DATE} from the campaign's summary files; every number traces to a run on RIKYU")
    txt(s, 0.6, 1.4, 12, 5.5, [
        "Baselines: stage_single (unchanged tasks, 2026-08) and stage_single_mp2026 (relabelled + added tasks, class-weight arms) — summary/baselines_mp2026.json, ceilings_adopted_v2.json, classification_weights.json, material_type_weights.json, space_group_confirm.json.",
        "Transfer: stage_xfer (240 continual runs), stage_ft warm-start (ftf, 10 orderings), stage_xu → ftfu never-seen (3 orderings) — summary/ft.json, material_type_warmstart_runs.json, position_runs.json.",
        "Dataset: data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet (dated 2026-09-11 in the reports), CHANGES note and rebuild script alongside; NEMAD and phonix-db parquets as before.",
        "Pages with the full evidence: transferability (four ways), two easy tasks that would not train (labels and descriptor), space group 0.24 vs 0.60 (class weights). HANDOFF.md experiments 1–12.",
        "Open items: re-measure material_type transfer against the class-weights-off baseline; phase-B re-run on the 0.4.x image; HEA survey; import of the three IMPORT NOW sets.",
    ], size=12.5)


def main():
    slide_title(); slide_agenda(); slide_dataset()
    mp = [t for t in DESC if DESC[t][0] == "Materials Project"]
    slide_inventory("Tasks 1/3 — Materials Project (energies, structure, electronic)", mp[:13], "DFT, GGA / GGA+U; rebuilt 2026-09-11")
    slide_inventory("Tasks 2/3 — Materials Project (magnetic, dielectric, elastic, symmetry)", mp[13:], "DFT, GGA / GGA+U; rebuilt 2026-09-11")
    curves = [t for t in DESC if DESC[t][0] == "thermoelectric (starry)"]
    rest = [t for t in DESC if DESC[t][0] not in ("Materials Project", "thermoelectric (starry)")]
    slide_inventory("Tasks 3/4 — thermoelectric curves (starry)", curves, "Experimental curves; one curve per composition, learned as kernel regression over the coordinate t")
    slide_inventory("Tasks 4/4 — NEMAD, phonon transport, quasicrystals", rest, "Text-mined experimental databases, first-principles phonon transport, and the quasicrystal classification")
    pic_slide("Single-task performance, all 41 tasks", "Same recipe and descriptor for every task; bars = mean over 5 seeds, whiskers = sd; n = training rows", FIG / "overview_bar.png")
    pic_slide("Single-task R² against training rows", "Data volume alone does not order the tasks: the weakest are the cell-scale labels and the noisiest experimental ones, not the smallest", FIG / "r2_vs_n.png")
    reg = [r["task"] for r in sorted(PERF, key=lambda r: -r["mean"]) if r["metric"] == "R²"]
    chunks = [reg[i:i + 12] for i in range(0, len(reg), 12)]
    for i, ch in enumerate(chunks, 1):
        slide_perf_table(f"Single-task metrics {i}/{len(chunks) + 1} — regression and curve tasks, ranked by R²", ch, "R² and MAE on the normalised scale, test split, mean ± sd over 5 seeds")
    slide_perf_table(f"Single-task metrics {len(chunks) + 1}/{len(chunks) + 1} — classification heads", [r["task"] for r in PERF if r["metric"] == "macro-F1"], "macro-F1 and accuracy, class weights off, 5 seeds")
    n_sc = len(list(FIG.glob("scatter_*.png")))
    for k in range(1, n_sc + 1):
        pic_slide(f"Observed vs predicted {k}/{n_sc}", "One seed (2025), test split; dashed = y = x; curve tasks show sampled (t, value) points", FIG / f"scatter_{k}.png")
    pic_slide("Classification heads: confusion matrices", "Unweighted cross-entropy, seed 2025; cell = share of the true row and the row count", FIG / "confusion_clf.png")
    pic_slide("Space group: confusion over the twelve largest groups", "151 classes; the remaining 139 folded into “other”; cubic and hexagonal groups are recognised, low-symmetry monoclinic ones partly", FIG / "confusion_sg.png")
    pic_slide("Space group: per-class F1 against class size", "The class imbalance is the physics of the crystal world; it sets what macro-F1 can reach", FIG / "sg_f1_vs_size.png")
    slide_clf_change(); slide_kmd()
    slide_material_type_setup(); slide_material_type_numbers()
    pic_slide("material_type: every run of the three arms", "Warm-start fine-tuning with the encoder trained; the dashed line is the alone mean", FIG / "mt_warmstart_strip.png")
    pic_slide("material_type: before and after the fine-tune", "The never-seen encoder starts from a random head and ends where the seen one ends", FIG / "mt_before_after.png")
    pic_slide("material_type: score against pretraining breadth (continual arm)", "From the 240-run transfer stage: the later material_type appears in the sequence, the better — the opposite of every regression task", FIG / "mt_position.png")
    slide_data_plan_intro(); slide_data_mof(); slide_data_catalyst(); slide_data_hea(); slide_data_sequence()
    slide_release(); slide_llm(); slide_status()
    out = HERE / "results" / f"DECK_{DATE.replace('-', '')}.pptx"
    prs.save(str(out)); print(f"{out}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
