#!/usr/bin/env python3
"""Rebuild the Materials Project part of the qc dataset on ONE level of theory (GGA / GGA+U).

Why. The 2025-04-10 MP export copied `summary.energy_per_atom`, which in today's Materials Project is
the GGA/GGA+U/r2SCAN *mixed* thermo scheme: for ~20% of entries it is an r2SCAN total energy, tens of
eV below the GGA one (Pt -51.5 vs -6.1 eV/atom), so the column mixed two energy references and could
not be learned (R2 0.77). The same mixing sits behind `formation_energy_per_atom`, and the summary's
structure (hence volume, density) and magnetism come from r2SCAN tasks where those exist. This script
rebuilds every MP-derived column from GGA / GGA+U calculations only, keeps the qa-/starry- rows and
every non-MP column untouched, and adds the extra MP properties chosen on 2026-09-12.

Sources (all pulled from api.materialsproject.org on 2026-09-12 into the scratch tables listed below;
legacy `mp-<n>` ids were resolved to the new ids through the API itself):
  mp_thermo_all.parquet       /materials/thermo/  thermo_type = GGA_GGA+U: energy_per_atom,
                              formation_energy_per_atom, energy_above_hull
  mp_thermo_gga_extra.parquet /materials/thermo/  GGA_GGA+U: equilibrium_reaction_energy_per_atom
  mp_core_entries.parquet     /materials/core/    entries per run type: the GGA / GGA_U entry's task id,
                              structure volume, nsites
  mp_gga_tasks.parquet,       /materials/tasks/   output of that GGA / GGA+U task: structure, density,
  mp_ggau_tasks.parquet                           outcar.total_magnetization, per-site magnetization,
                                                  bandgap, efermi, is_gap_direct, is_metal
  mp_summary_all.parquet      /materials/summary/ band_gap, cbm, vbm, efermi, is_gap_direct, is_metal,
                              elastic, dielectric, refractive index, piezoelectric maxima
  mp_origins.parquet          /materials/summary/ origins -> /materials/tasks/ run_type of the task
                              each summary property came from

Policy, per column family:
  energies       thermo GGA_GGA+U only (Final energy per atom, Formation energy per atom,
                 Equilibrium reaction energy per atom); no value if the material has no GGA-family entry
  structure      the GGA / GGA+U entry's own structure: Volume (rescaled to the dataset's cell by the
                 atom-count ratio, same reduced formula required), Density, Density atomic (per atom)
  magnetism      the GGA / GGA+U entry task's OUTCAR: Total magnetization (|mu_B| per dataset cell,
                 rescaled like Volume), per formula unit, per volume; Ordering and Number of magnetic
                 sites from pymatgen's CollinearMagneticStructureAnalyzer on that task's per-site moments
  electronic     summary band_gap / cbm / vbm / efermi / is_gap_direct / is_metal, KEPT only where the
                 summary's electronic_structure origin task is GGA or GGA+U (else NaN)
  elastic, dielectric, piezoelectric
                 summary values, kept only where the origin task is GGA or GGA+U, and only inside
                 physical bounds (see FILTERS) — MP itself flags the rest as unreliable
Normalisation follows data/scripts/process_ac_qe_te_data.ipynb cell 18 exactly: per scalar column
StandardScaler -> PowerTransformer(yeo-johnson) fitted on every non-null value, stored as
`<snake_case>_scaler`; categorical columns get a LabelEncoder. Nothing is overwritten: the outputs are
new files with the 20260912 date.

    uv run python data/scripts/rebuild_mp_gga_20260912.py --scratch <dir with the pulled tables>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.core import Composition, Structure
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "data" / "qc_ac_te_mp_dos_reformat_20260515.pd.parquet"
OLD_PRE = ROOT / "data" / "preprocessing_objects_20250615.pkl.z"
NEW = ROOT / "data" / "qc_ac_te_mp_dos_reformat_20260912.pd.parquet"
NEW_PRE = ROOT / "data" / "preprocessing_objects_20260912.pkl.z"
REPORT = ROOT / "data" / "qc_ac_te_mp_dos_reformat_20260912_CHANGES.md"

GGA = {"GGA", "GGA+U"}
# physical sanity bounds for the summary-only properties (MP's elastic/dielectric docs carry known outliers)
FILTERS = {
    "Bulk modulus": (0.0, 700.0),  # GPa, K_VRH
    "Shear modulus": (0.0, 700.0),  # GPa, G_VRH
    "Poisson ratio": (-1.0, 0.5),
    "Universal anisotropy": (0.0, 50.0),
    "Dielectric total": (0.0, 1000.0),
    "Dielectric ionic": (0.0, 1000.0),
    "Dielectric electronic": (0.0, 1000.0),
    "Refractive index": (0.0, 30.0),
    "Piezoelectric max": (0.0, 20.0),  # C/m^2
}


def snake(col: str) -> str:
    return "_".join(w.lower() for w in col.strip().split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", type=Path, required=True)
    args = ap.parse_args()
    S = args.scratch

    old = pd.read_parquet(OLD)
    new = old.copy()
    mp_rows = [i for i in old.index if i.startswith("mp-")]
    print(f"old dataset {old.shape}; MP rows {len(mp_rows)}")

    # ---- id mapping and GGA_GGA+U energies -------------------------------------------------------
    th = pd.read_parquet(S / "mp_thermo_all.parquet")
    id_map = th.dropna(subset=["legacy_id"]).drop_duplicates("legacy_id").set_index("legacy_id")["new_id"]
    gga = (
        th[th.thermo_type == "GGA_GGA+U"]
        .dropna(subset=["legacy_id"])
        .drop_duplicates("legacy_id")
        .set_index("legacy_id")
    )
    extra = pd.read_parquet(S / "mp_thermo_gga_extra.parquet").drop_duplicates("new_id").set_index("new_id")
    new["MP id"] = pd.Series(id_map).reindex(new.index)
    print(f"legacy ids resolved: {new['MP id'].notna().sum()} / {len(mp_rows)}; GGA_GGA+U energies for {len(gga)}")

    # ---- the GGA / GGA+U entry of every material, with its task output --------------------------
    ent = pd.read_parquet(S / "mp_core_entries.parquet")
    ent = ent[ent.run_type.isin(["GGA", "GGA_U"])].copy()
    ent["pref"] = (ent.run_type == "GGA_U").astype(int)  # MP's mixing uses the +U entry where one exists
    pick = ent.sort_values(["new_id", "pref"], ascending=[True, False]).drop_duplicates("new_id").set_index("new_id")
    tasks = (
        pd.concat(
            [pd.read_parquet(S / "mp_gga_tasks.parquet"), pd.read_parquet(S / "mp_ggau_tasks.parquet")],
            ignore_index=True,
        )
        .drop_duplicates("task_id")
        .set_index("task_id")
    )
    pick = pick.join(tasks, on="task_id", rsuffix="_task")
    print(f"materials with a GGA-family entry: {len(pick)}; with task output: {pick['structure'].notna().sum()}")

    # ---- origins: which run type produced each summary property ----------------------------------
    orig = pd.read_parquet(S / "mp_origins.parquet")
    ok = {}
    for name, grp in orig.groupby("name"):
        ok[name] = set(grp[grp.run_type.isin(GGA)].new_id)
    # elasticity has no origin entry in the summary; MP's elastic workflow is GGA/GGA+U deformations only,
    # so a material qualifies when it carries a GGA-family Deformation calculation
    calc = pd.read_parquet(S / "mp_calc_types.parquet")
    ok["elasticity"] = set(calc[calc.calc_type.isin(["GGA Deformation", "GGA+U Deformation"])].new_id)
    print("origins run types by property:")
    print(orig.groupby(["name", "run_type"]).size().unstack(fill_value=0).to_string())
    summ = pd.read_parquet(S / "mp_summary_all.parquet").drop_duplicates("material_id").set_index("material_id")

    # ---- assemble the MP-derived columns ---------------------------------------------------------
    cols = {}

    def put(col, series):
        cols[col] = series

    nid = new.loc[mp_rows, "MP id"]

    # energies (GGA_GGA+U only)
    e = gga.reindex(mp_rows)
    put("Final energy per atom", e["energy_per_atom"])
    put("Formation energy per atom", e["formation_energy_per_atom"])
    put("Energy above hull", e["energy_above_hull"])
    put(
        "Equilibrium reaction energy per atom",
        extra["equilibrium_reaction_energy_per_atom"].reindex(nid.values).set_axis(mp_rows),
    )

    # structure + magnetism from the GGA-family entry task.
    # The dataset's `composition` is the cell of the SUMMARY structure (r2SCAN where one exists), and the
    # GGA task often used a different cell of the same material (Fe: dataset Fe2, GGA task Fe1). Per-cell
    # quantities are therefore rescaled to the dataset's cell by the atom-count ratio, after checking
    # that the two cells are the same reduced formula; intensive quantities need no rescaling.
    p = pick.reindex(nid.values).set_axis(mp_rows)
    vol_t = p["volume_task"].where(p["volume_task"].notna(), p["volume"])
    n_t = p["nsites_task"].where(p["nsites_task"].notna(), p["nsites"])
    ds_comp = {
        rid: Composition(old.loc[rid, "composition"]) for rid in mp_rows if isinstance(old.loc[rid, "composition"], str)
    }
    n_ds = pd.Series({rid: c.num_atoms for rid, c in ds_comp.items()}).reindex(mp_rows)
    same = pd.Series(False, index=mp_rows)
    for rid, row in p.iterrows():
        if isinstance(row.get("structure"), str) and rid in ds_comp:
            tc = Structure.from_dict(json.loads(row["structure"])).composition
            same[rid] = tc.reduced_formula == ds_comp[rid].reduced_formula
    factor = (n_ds / n_t).where(same)
    print(
        f"GGA task cell vs dataset cell: same reduced formula {int(same.sum())}, different/missing {int((~same).sum())}; "
        f"integer atom ratio {int(((factor - factor.round()).abs() < 1e-6).sum())}"
    )
    orderings, nmag, totmag, dens = {}, {}, {}, {}
    for rid, row in p.iterrows():
        if not isinstance(row.get("structure"), str):
            continue
        st = Structure.from_dict(json.loads(row["structure"]))
        dens[rid] = float(st.density)  # g/cm^3 from the GGA structure, intensive
        mm = json.loads(row["site_magmoms"]) if isinstance(row.get("site_magmoms"), str) else None
        tm = row.get("total_magnetization")
        if mm is not None and len(mm) == len(st):
            st.add_site_property("magmom", [float(x or 0.0) for x in mm])
            an = CollinearMagneticStructureAnalyzer(st)
            orderings[rid] = an.ordering.value
            nmag[rid] = an.number_of_magnetic_sites
        elif tm is not None:
            orderings[rid] = "NM" if abs(float(tm)) < 0.1 else "FM"  # no site moments stored: coarse label
        if tm is not None:
            totmag[rid] = abs(float(tm))
    vol = vol_t * factor
    put("Volume", vol)  # A^3 per dataset cell
    put("Density", pd.Series(dens).reindex(mp_rows))  # g/cm^3 from the GGA structure, intensive
    put("Density atomic", vol_t / n_t)  # A^3 per atom, intensive
    tot_t = pd.Series(totmag).reindex(mp_rows)
    tot = tot_t * factor  # |mu_B| per dataset cell — the old column's meaning
    fu = pd.Series({rid: c.get_reduced_composition_and_factor()[1] for rid, c in ds_comp.items()}).reindex(mp_rows)
    put("Total magnetization", tot)
    put("Total magnetization per formula unit", tot / fu)
    put("Total magnetization per volume", tot_t / vol_t)  # mu_B / A^3, intensive
    put("Magnetic ordering", pd.Series(orderings).reindex(mp_rows))
    put("Number of magnetic sites", pd.Series(nmag).reindex(mp_rows))

    # electronic structure: summary values where the electronic_structure origin is GGA-family
    es_ok = pd.Series([n in ok.get("electronic_structure", set()) for n in nid.values], index=mp_rows)
    sm = summ.reindex(nid.values).set_axis(mp_rows)
    for col, src in (("Band gap", "band_gap"), ("Efermi", "efermi"), ("CBM", "cbm"), ("VBM", "vbm")):
        put(col, pd.to_numeric(sm[src], errors="coerce").where(es_ok))
    put("Is metal", sm["is_metal"].where(es_ok).astype("object"))
    put("Is gap direct", sm["is_gap_direct"].where(es_ok).astype("object"))

    # elastic / dielectric / piezo: GGA-family origin and physical bounds
    def gated(col, src, origin):
        okset = ok.get(origin, set())
        m = pd.Series([n in okset for n in nid.values], index=mp_rows)
        x = pd.to_numeric(sm[src], errors="coerce").where(m)
        lo, hi = FILTERS[col]
        return x.where((x > lo) & (x <= hi))

    put("Bulk modulus", gated("Bulk modulus", "bulk_modulus_vrh", "elasticity"))
    put("Shear modulus", gated("Shear modulus", "shear_modulus_vrh", "elasticity"))
    put("Poisson ratio", gated("Poisson ratio", "homogeneous_poisson", "elasticity"))
    put("Universal anisotropy", gated("Universal anisotropy", "universal_anisotropy", "elasticity"))
    put("Dielectric total", gated("Dielectric total", "e_total", "dielectric"))
    put("Dielectric ionic", gated("Dielectric ionic", "e_ionic", "dielectric"))
    put("Dielectric electronic", gated("Dielectric electronic", "e_electronic", "dielectric"))
    put("Refractive index", gated("Refractive index", "n", "dielectric"))
    put("Piezoelectric max", gated("Piezoelectric max", "e_ij_max", "piezoelectric"))

    # ---- write the columns into the new frame ----------------------------------------------------
    changed = {}
    for col, s in cols.items():
        before = old[col].reindex(mp_rows) if col in old.columns else pd.Series(np.nan, index=mp_rows)
        if col not in new.columns:
            new[col] = np.nan if s.dtype != object else None
        new.loc[mp_rows, col] = s.values
        b, a = pd.to_numeric(before, errors="coerce"), pd.to_numeric(s, errors="coerce")
        changed[col] = {
            "non_null_before": int(before.notna().sum()),
            "non_null_after": int(s.notna().sum()),
            "changed_values": int(((b - a).abs() > 1e-6).sum()) if col in old.columns and s.dtype != object else None,
        }

    # ---- normalisation (notebook cell 18) and encoders ---------------------------------------------
    pre = joblib.load(OLD_PRE)
    numeric = [
        c
        for c in cols
        if c not in ("Is metal", "Is gap direct", "Magnetic ordering", "Number of magnetic sites", "Energy above hull")
    ]
    for col in numeric:
        x = pd.to_numeric(new[col], errors="coerce")
        idx = x.notna()
        print(f"  normalising {col:40s} non-null {int(idx.sum())}")
        if idx.sum() < 10:
            print("    skipped: too few values")
            continue
        pipe = Pipeline(
            [("standardscaler", StandardScaler()), ("powertransformer", PowerTransformer(method="yeo-johnson"))]
        )
        new[f"{col} (normalized)"] = np.nan
        new.loc[idx, f"{col} (normalized)"] = pipe.fit_transform(
            x[idx].to_numpy(dtype=np.float64).reshape(-1, 1)
        ).flatten()
        pre[f"{snake(col)}_scaler"] = pipe
    for col in ("Magnetic ordering",):
        le = LabelEncoder()
        idx = new[col].notna()
        new[f"{col} (label)"] = np.nan
        new.loc[idx, f"{col} (label)"] = le.fit_transform(new.loc[idx, col].astype(str))
        pre[f"{snake(col)}_label_encoder"] = le
    for col in ("Is metal", "Is gap direct"):
        new[f"{col} (label)"] = pd.to_numeric(new[col].map({True: 1, False: 0}), errors="coerce")
    pre["mp_level_of_theory"] = (
        "GGA / GGA+U only (thermo_type GGA_GGA+U; structure, magnetism from the GGA-family entry task; summary properties kept only where their origin task is GGA or GGA+U)"
    )
    pre["rebuilt_from"] = str(OLD.name)

    new.to_parquet(NEW)
    joblib.dump(pre, NEW_PRE)

    # ---- change report ------------------------------------------------------------------------
    lines = [
        f"# {NEW.name}\n",
        f"Rebuilt from `{OLD.name}` on 2026-09-12; MP columns on GGA / GGA+U only. Rows: {len(new)} (unchanged). Columns: {old.shape[1]} -> {new.shape[1]}.\n",
        "| column | non-null before | non-null after | values changed |",
        "|---|---|---|---|",
    ]
    for col, c in changed.items():
        lines.append(
            f"| {col} | {c['non_null_before']} | {c['non_null_after']} | {c['changed_values'] if c['changed_values'] is not None else '-'} |"
        )
    lines.append("\nElectronic-structure origins by run type:\n")
    lines.append("```\n" + orig.groupby(["name", "run_type"]).size().unstack(fill_value=0).to_string() + "\n```\n")
    REPORT.write_text("\n".join(lines))
    print(f"wrote {NEW} {new.shape}, {NEW_PRE}, {REPORT}")
    for col, c in changed.items():
        print(
            f"  {col:40s} non-null {c['non_null_before']:6d} -> {c['non_null_after']:6d}   changed {c['changed_values']}"
        )


if __name__ == "__main__":
    main()
