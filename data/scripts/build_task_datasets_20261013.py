#!/usr/bin/env python3
"""Collect every composition-keyed task dataset into one folder with readable names — data/fm_task_datasets_20261013/

Copies only: no value is changed, no row is dropped. The mixed qc table
(qc_ac_te_mp_dos_reformat_20260912.pd.parquet) is split by source and task type; every split file keeps the
qc row id, the composition, the qc `split` column and the original column names, so the task index maps
one-to-one onto the training configs. Every other dataset keeps one task file (the `_norm` table used in
training) plus its `_raw` table with the unnormalized values and units.

Outputs
  hypodx_material_type.parquet      material_type, all 49,034 qc compositions (positives from HYPOD-X)
  mp_properties.parquet             Materials Project scalar properties + 4 classification labels (33,829 rows)
  mp_dos_curves.parquet             Materials Project DOS curves (10,321 rows)
  transport_curves.parquet          temperature curves (Seebeck, ZT, PF, κ, ρ, χ) from Starrydata / HYPOD-X
  nemad_magnetic{,_raw}.parquet, nemad_superconductor{,_raw}.parquet, phonix_thermal_conductivity{,_raw}.parquet,
  kkr_cpa_hea{,_raw}.parquet, qmof{,_raw}.parquet, catalysis_hub_adsorption{,_raw}.parquet
  task_index.csv                    one row per task: file, column, kind, counts per split, unit, source

    uv run python data/scripts/build_task_datasets_20261013.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent
OUT = DATA / "fm_task_datasets_20261013"
QC = DATA / "qc_ac_te_mp_dos_reformat_20260912.pd.parquet"

CURVES = ["Seebeck coefficient", "ZT", "Power factor", "Thermal conductivity", "Electrical resistivity", "Magnetic susceptibility"]
MP_REG = {  # task name in the configs -> (qc column stem, unit)
    "final_energy": ("Final energy per atom", "eV/atom"), "formation_energy": ("Formation energy per atom", "eV/atom"),
    "reaction_energy": ("Equilibrium reaction energy per atom", "eV/atom"), "volume": ("Volume", "Å³ (per cell)"),
    "density": ("Density", "g/cm³"), "density_atomic": ("Density atomic", "Å³/atom"), "efermi": ("Efermi", "eV"),
    "band_gap": ("Band gap", "eV"), "cbm": ("CBM", "eV"), "vbm": ("VBM", "eV"),
    "total_magnetization": ("Total magnetization", "μB (per cell)"), "magnetization_per_fu": ("Total magnetization per formula unit", "μB/f.u."),
    "magnetization_per_volume": ("Total magnetization per volume", "μB/Å³"), "dielectric_total": ("Dielectric total", "dimensionless"),
    "dielectric_ionic": ("Dielectric ionic", "dimensionless"), "dielectric_electronic": ("Dielectric electronic", "dimensionless"),
    "refractive_index": ("Refractive index", "dimensionless"), "bulk_modulus": ("Bulk modulus", "GPa"), "shear_modulus": ("Shear modulus", "GPa"),
    "poisson_ratio": ("Poisson ratio", "dimensionless"), "universal_anisotropy": ("Universal anisotropy", "dimensionless"),
    "piezoelectric_max": ("Piezoelectric max", "C/m²"),
}
MP_CLF = {"magnetic_ordering": ("Magnetic ordering", "Magnetic ordering (label)"), "is_metal": ("Is metal", "Is metal (label)"),
          "is_gap_direct": ("Is gap direct", "Is gap direct (label)"), "space_group": ("Space group", "Space group (task label)")}
CURVE_TASKS = {"seebeck": ("Seebeck coefficient", "V/K"), "zt": ("ZT", "dimensionless"), "power_factor": ("Power factor", "W/(m·K²)"),
               "thermal_conductivity": ("Thermal conductivity", "W/(m·K)"), "electrical_resistivity": ("Electrical resistivity", "Ω·m"),
               "magnetic_susceptibility": ("Magnetic susceptibility", "as stored (unit not recorded)")}

SRC = {
    "HYPOD-X": ("experiment (literature)", "Fujita, E. et al. Sci. Data 11, 1211 (2024); figshare 10.6084/m9.figshare.25650705"),
    "Materials Project": ("DFT, GGA/GGA+U (rebuilt 2026-09-12)", "Materials Project, api.materialsproject.org (pulled 2026-09-12)"),
    "Starrydata": ("experiment (published plots)", "Katsura, Y. et al. Sci. Technol. Adv. Mater. Methods 5, 2506976 (2025)"),
    "NEMAD": ("experiment (LLM text-mined)", "Itani, S., Zhang, Y. & Zang, J. Nat. Commun. (2025), doi:10.1038/s41467-025-64458-z"),
    "phonix-db": ("DFT anharmonic phonons (VASP + ALAMODE)", "npj Comput. Mater. 12, 150 (2026), doi:10.1038/s41524-026-02033-w"),
    "KKR-CPA HEA": ("all-electron KKR-CPA", "Fukushima, T., Akai, H., Chikyow, T. & Kino, H. Phys. Rev. Mater. 6, 023802 (2022)"),
    "QMOF": ("DFT, PBE-D3(BJ)", "Rosen, A. S. et al. Matter 4, 1578–1597 (2021)"),
    "Catalysis-Hub": ("DFT, BEEF-vdW", "Mamun, O., Winther, K. T., Boes, J. R. & Bligaard, T. Sci. Data 6, 76 (2019)"),
}
TRAINED_V2 = set(MP_REG) | set(MP_CLF) | {"material_type", "dos_density"} | set(CURVE_TASKS) | {"magnetization", "curie", "neel", "magnetic_moment", "tc", "kp", "klat"}


def counts(series: pd.Series, split: pd.Series) -> dict:
    nn = series.notna()
    return {"n_total": int(nn.sum()), **{f"n_{s}": int((nn & (split == s)).sum()) for s in ("train", "val", "test")}}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists — choose a new date instead of overwriting.")
    OUT.mkdir(parents=True)
    rows = []

    qc = pd.read_parquet(QC)
    qc.index.name = "id"
    qc = qc.reset_index()
    qc["source table"] = qc["id"].str.extract(r"^(mp|starry|qa)-")[0].map({"mp": "Materials Project", "starry": "Starrydata", "qa": "HYPOD-X"})

    # 1. material type: all rows (HYPOD-X positives, MP and Starrydata compositions as "others")
    mt = qc[["id", "composition", "split", "source table", "Material type", "Material type (label)"]]
    mt.to_parquet(OUT / "hypodx_material_type.parquet", index=False)
    rows.append(dict(task="material_type", file="hypodx_material_type.parquet", column="Material type (label)", t_column="", raw_column="Material type",
                     kind="classification", n_classes=int(mt["Material type (label)"].nunique()), unit="class: DAC / DQC / IAC / IQC / others",
                     source="HYPOD-X (+ MP, Starrydata as others)", **counts(mt["Material type (label)"], mt["split"])))

    # 2. Materials Project scalar properties and labels
    mp = qc[qc["id"].str.startswith("mp-")]
    keep = ["id", "MP id", "composition", "split"]
    for stem, _ in MP_REG.values():
        keep += [stem, f"{stem} (normalized)"]
    keep += ["Energy above hull", "Number of magnetic sites"]
    for raw, lab in MP_CLF.values():
        keep += [raw, lab]
    mp = mp[list(dict.fromkeys(keep))]
    mp.to_parquet(OUT / "mp_properties.parquet", index=False)
    for t, (stem, unit) in MP_REG.items():
        rows.append(dict(task=t, file="mp_properties.parquet", column=f"{stem} (normalized)", t_column="", raw_column=stem, kind="regression",
                         n_classes="", unit=unit, source="Materials Project", **counts(mp[f"{stem} (normalized)"], mp["split"])))
    for t, (raw, lab) in MP_CLF.items():
        rows.append(dict(task=t, file="mp_properties.parquet", column=lab, t_column="", raw_column=raw, kind="classification",
                         n_classes=int(mp[lab].nunique()), unit="class", source="Materials Project", **counts(mp[lab], mp["split"])))

    # 3. Materials Project DOS curves
    dos = qc.loc[qc["DOS density (normalized)"].notna(), ["id", "composition", "split", "DOS energy", "DOS density", "DOS density (normalized)"]]
    dos.to_parquet(OUT / "mp_dos_curves.parquet", index=False)
    rows.append(dict(task="dos_density", file="mp_dos_curves.parquet", column="DOS density (normalized)", t_column="DOS energy", raw_column="DOS density",
                     kind="curve", n_classes="", unit="states/eV vs E − E_F (eV)", source="Materials Project", **counts(dos["DOS density (normalized)"], dos["split"])))

    # 4. temperature curves
    ccols = [c for name in CURVES for c in (name, f"{name} (T/K)", f"{name} (normalized)")]
    has = qc[[f"{n} (normalized)" for n in CURVES]].notna().any(axis=1)
    tr = qc.loc[has, ["id", "composition", "split", "source table"] + ccols]
    tr.to_parquet(OUT / "transport_curves.parquet", index=False)
    for t, (name, unit) in CURVE_TASKS.items():
        rows.append(dict(task=t, file="transport_curves.parquet", column=f"{name} (normalized)", t_column=f"{name} (T/K)", raw_column=name, kind="curve",
                         n_classes="", unit=f"{unit} vs T (K)", source="Starrydata (+ HYPOD-X curves on qa- rows)", **counts(tr[f"{name} (normalized)"], tr["split"])))

    # 5. the other datasets: task table + raw table, renamed
    other = [
        ("nemad_magnetic", "NEMAD_magnetic_20260419_norm.parquet", "NEMAD_magnetic_20260419.parquet", "NEMAD",
         {"magnetic_moment": ("Magnetic moment", "μB/f.u."), "magnetization": ("Magnetization", "A·m²/mol"), "curie": ("Curie temperature", "K"), "neel": ("Neel temperature", "K")}),
        ("nemad_superconductor", "NEMAD_superconductor_20260425_norm.parquet", "NEMAD_superconductor_20260425.parquet", "NEMAD", {"tc": ("Transition temperature", "K")}),
        ("phonix_thermal_conductivity", "phonix-db-filtered_20260425_norm.parquet", "phonix-db-filtered_20260425.parquet", "phonix-db",
         {"kp": ("kp", "W/(m·K)"), "klat": ("klat", "W/(m·K)")}),
        ("kkr_cpa_hea", "KKR_CPA_HEA_20261011_norm.parquet", "KKR_CPA_HEA_20261011.parquet", "KKR-CPA HEA",
         {f"hea_{p}_{s}": (f"{n} {s}", u) for s in ("bcc", "fcc") for p, n, u in (("total_moment", "Total magnetic moment", "μB"), ("curie", "Curie temperature", "K"), ("resistivity", "Residual resistivity", "μΩ·cm"))}
         | {"hea_structure_preference": ("Structure preference", "class: bcc / fcc")}),
        ("qmof", "QMOF_20261011_norm.parquet", "QMOF_20261011.parquet", "QMOF",
         {"qmof_band_gap_pbe": ("QMOF band gap PBE", "eV"), "qmof_cbm_pbe": ("QMOF CBM PBE", "eV"), "qmof_vbm_pbe": ("QMOF VBM PBE", "eV"), "qmof_energy_per_atom_pbe": ("QMOF energy per atom PBE", "eV/atom")}),
        ("catalysis_hub_adsorption", "CatalysisHub_MamunHighT2019_20261012_norm.parquet", "CatalysisHub_MamunHighT2019_20261012.parquet", "Catalysis-Hub", {}),
    ]
    for name, norm_f, raw_f, src, tasks in other:
        shutil.copy2(DATA / norm_f, OUT / f"{name}.parquet")
        shutil.copy2(DATA / raw_f, OUT / f"{name}_raw.parquet")
        d = pd.read_parquet(OUT / f"{name}.parquet")
        if name == "catalysis_hub_adsorption":
            tasks = {f"ads_{c.split('*')[0].lower()}": (c.replace(" (normalized)", ""), "eV") for c in d.columns if c.endswith("(normalized)")}
        for t, (stem, unit) in tasks.items():
            col = f"{stem} (label)" if stem == "Structure preference" else f"{stem} (normalized)"
            kind = "classification" if col.endswith("(label)") else "regression"
            rows.append(dict(task=t, file=f"{name}.parquet", column=col, t_column="", raw_column=stem, kind=kind,
                             n_classes=int(d[col].nunique()) if kind == "classification" else "", unit=unit, source=src, **counts(d[col], d["split"])))

    idx = pd.DataFrame(rows)
    first_src = idx["source"].str.split(" ").str[0].replace({"Materials": "Materials Project", "KKR-CPA": "KKR-CPA HEA"})
    idx["origin"] = first_src.map(lambda s: SRC.get(s, ("", ""))[0])
    idx["reference"] = first_src.map(lambda s: SRC.get(s, ("", ""))[1])
    idx["trained_in_campaign_v2"] = idx["task"].isin(TRAINED_V2)
    idx = idx[["task", "kind", "source", "origin", "file", "column", "t_column", "raw_column", "unit", "n_classes", "n_total", "n_train", "n_val", "n_test", "trained_in_campaign_v2", "reference"]]
    idx.to_csv(OUT / "task_index.csv", index=False)
    print(f"{len(idx)} tasks in {OUT}")
    print(idx.groupby(["file", "kind"]).size().to_string())


if __name__ == "__main__":
    main()
