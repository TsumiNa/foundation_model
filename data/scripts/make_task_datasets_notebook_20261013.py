#!/usr/bin/env python3
"""Write data/fm_task_datasets_20261013/datasets_overview.ipynb (one section per dataset), then execute it.

uv run --with nbformat --with nbconvert --with ipykernel python data/scripts/make_task_datasets_notebook_20261013.py
"""

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

OUT = Path(__file__).resolve().parent.parent / "fm_task_datasets_20261013" / "datasets_overview.ipynb"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Task datasets of the composition foundation model

Every dataset the model trains on is keyed by **chemical composition**: one row per composition, one column per
task, a missing value where the source has no measurement. The model reads only the composition (as a KMD
descriptor of atomic fractions) and predicts each task through its own head.

This folder collects all of them under readable names (built by `data/scripts/build_task_datasets_20261013.py`;
nothing was changed, only copied, split and renamed). Each section below introduces one dataset — where the data
come from, which tasks it carries and what they mean physically — and then shows how to read it and what it looks like.

**Conventions shared by every file**

- `composition`: a formula string such as `Fe2 O3` (pymatgen `Composition.formula`).
- `split`: `train` / `val` / `test`, fixed per dataset (70 / 15 / 15).
- `<property>` (often with the unit in brackets): the raw value. `<property> (normalized)`: the value the model is
  trained on — StandardScaler followed by a Yeo-Johnson transform, fitted per column. `<property> (label)`: an
  integer class label for classification tasks.
- Curve tasks store one curve per row as a list: the values, the x coordinate (temperature in K, or energy in eV),
  and the normalized values.
- `task_index.csv` lists every task with its file, column, kind, counts per split, unit and source.
""")
code("""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path('.').resolve()
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
index = pd.read_csv(HERE / 'task_index.csv')
print(f"{len(index)} tasks in {index['file'].nunique()} files")
index.groupby(['file', 'kind']).size().rename('tasks').to_frame()
""")
code("""
# Data volume per task (training rows), coloured by kind
ix = index.sort_values('n_total')
colors = ix['kind'].map({'regression': '#0f4c5c', 'classification': '#b8652a', 'curve': '#4f8473'})
fig, ax = plt.subplots(figsize=(8, 13))
ax.barh(ix['task'], ix['n_total'], color=colors)
ax.set_xscale('log'); ax.set_xlabel('rows with a value (log scale)')
ax.set_title('Rows per task — regression (teal), classification (copper), curve (green)')
plt.tight_layout()
""")


def section(title, intro, body_code):
    md(title + "\n\n" + intro.strip())
    for c in body_code:
        code(c)


COMMON_HIST = """
def hist_grid(df, cols, ncols=3, log=None, title=None):
    log = log or set()
    n = len(cols); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.9 * nrows), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        v = pd.to_numeric(df[c], errors='coerce').dropna()
        if c in log:
            v = v[v > 0]; ax.hist(np.log10(v), bins=60, color='#0f4c5c'); ax.set_xlabel(f'log10 {c}')
        else:
            ax.hist(v, bins=60, color='#0f4c5c'); ax.set_xlabel(c)
        ax.set_title(f'n = {len(v):,}', fontsize=9)
    for ax in axes.ravel()[n:]: ax.axis('off')
    if title: fig.suptitle(title, y=1.01)
    plt.tight_layout()

def split_table(df):
    return df['split'].value_counts().reindex(['train', 'val', 'test']).to_frame('rows').T
"""
code(COMMON_HIST)

# ---------------------------------------------------------------- qc: HYPOD-X material type
section(
    "## 1. Quasicrystals and approximants — `hypodx_material_type.parquet`",
    """
**Source.** HYPOD-X (Hypermaterials Open Datasets): Fujita, E. et al. *Comprehensive experimental datasets of
quasicrystals and their approximants*, Sci. Data 11, 1211 (2024); data on figshare
(10.6084/m9.figshare.25650705). Compositions of quasicrystals and approximant crystals collected by hand from
the experimental literature.

**Origin of this file.** It is the `material_type` column of our mixed table
`qc_ac_te_mp_dos_reformat_20260912.pd.parquet` (the "qc ac" part of its name), kept with **all 49,034
compositions**: the 367 HYPOD-X compositions (row ids `qa-…`) are the positive classes, and every Materials
Project (`mp-…`) and Starrydata (`starry-…`) composition is labelled `others`. Splitting the rows by source would
remove the negatives, so this task stays whole.

**Task: `material_type`** — 5-class classification.

| class | meaning |
|---|---|
| IQC | icosahedral quasicrystal: long-range order without periodicity, icosahedral (5-fold) symmetry in 3D |
| DQC | decagonal quasicrystal: quasiperiodic in a plane, periodic along the 10-fold axis |
| IAC | icosahedral approximant crystal: periodic crystal whose local atomic clusters match an IQC |
| DAC | decagonal approximant crystal: the periodic counterpart of a DQC |
| others | any other compound (here: every MP and Starrydata composition) |

How it is obtained in general: quasicrystals are identified experimentally by diffraction (sharp Bragg peaks with
forbidden rotational symmetry), approximants by structure refinement showing the same clusters in a periodic cell.
The class is therefore an experimental structural assignment, not a computed quantity. The dataset is extremely
imbalanced (99 % `others`).
""",
    [
        """
mt = pd.read_parquet(HERE / 'hypodx_material_type.parquet')
print(mt.shape); display(split_table(mt)); mt.head()
""",
        """
display(pd.crosstab(mt['Material type'], mt['source table']))
fig, ax = plt.subplots(figsize=(6, 3))
c = mt.loc[mt['Material type'] != 'others', 'Material type'].value_counts()
ax.bar(c.index, c.values, color='#b8652a'); ax.set_ylabel('compositions'); ax.set_title('Positive classes (others = %s rows)' % f"{(mt['Material type'] == 'others').sum():,}")
plt.tight_layout()
""",
        """
# Which elements make quasicrystals? Element frequency among the 367 positive compositions
from pymatgen.core import Composition
pos = mt[mt['Material type'] != 'others']
el = pd.Series([e.symbol for c in pos['composition'] for e in Composition(c).elements]).value_counts().head(15)
fig, ax = plt.subplots(figsize=(7, 2.8)); ax.bar(el.index, el.values, color='#0f4c5c'); ax.set_ylabel('compositions'); ax.set_title('Most frequent elements in quasicrystals / approximants')
plt.tight_layout()
""",
    ],
)

# ---------------------------------------------------------------- MP properties
section(
    "## 2. Materials Project scalar properties — `mp_properties.parquet`",
    """
**Source.** Materials Project (api.materialsproject.org). The `mp-…` rows of our mixed table. The Materials
Project part was **rebuilt on 2026-09-12 on one level of theory** (GGA / GGA+U): the original 2025 export mixed
GGA and r2SCAN energies (e.g. Pt −51.5 vs −6.1 eV/atom for the two schemes), so energies, structures and magnetism
are now taken from GGA-family calculations only, and 17 properties were added (see
`data/qc_ac_te_mp_dos_reformat_20260912_CHANGES.md`). 33,829 compositions of stable or near-stable crystals.

**How the values are obtained.** Every number is a density-functional-theory (DFT) calculation on the relaxed
crystal structure, run by the Materials Project workflows (VASP, PBE-based GGA, +U on transition-metal oxides and
fluorides). No experiment is involved.

**Tasks (22 regression, 4 classification).**

| task | column stem | meaning | how it is computed |
|---|---|---|---|
| final_energy | Final energy per atom | total DFT energy per atom | GGA/GGA+U thermo scheme |
| formation_energy | Formation energy per atom | energy to form the compound from its elements | total energy minus elemental references, with MP's corrections |
| reaction_energy | Equilibrium reaction energy per atom | margin to the competing phases (stability) | convex-hull construction |
| volume | Volume | volume of the stored cell (scales with atoms per cell) | relaxed structure |
| density | Density | mass density | relaxed structure |
| density_atomic | Density atomic | volume per atom (intensive form of volume) | relaxed structure |
| efermi | Efermi | Fermi energy | band structure |
| band_gap | Band gap | energy gap between filled and empty states (0 for metals) | band structure along high-symmetry lines; GGA underestimates gaps |
| cbm / vbm | CBM / VBM | conduction-band minimum / valence-band maximum | band structure |
| total_magnetization | Total magnetization | net magnetic moment per cell | spin-polarized GGA |
| magnetization_per_fu / _per_volume | … per formula unit / per volume | intensive forms of the net moment | derived |
| dielectric_total / ionic / electronic | Dielectric … | response to an electric field; electronic part = ε∞, ionic part from lattice vibrations | DFPT |
| refractive_index | Refractive index | √(electronic dielectric constant) | DFPT |
| bulk_modulus / shear_modulus | … | resistance to compression / shear | elastic tensor from strained cells (Voigt-Reuss-Hill) |
| poisson_ratio | Poisson ratio | lateral vs axial strain | from bulk and shear moduli |
| universal_anisotropy | Universal anisotropy | how direction-dependent the elastic response is (0 = isotropic) | elastic tensor |
| piezoelectric_max | Piezoelectric max | largest piezoelectric coefficient | DFPT |
| magnetic_ordering | Magnetic ordering (label) | NM / FM / FiM / AFM ground-state order | site moments of the GGA calculation |
| is_metal | Is metal (label) | gap = 0 or not | band structure |
| is_gap_direct | Is gap direct (label) | VBM and CBM at the same k-point or not | band structure |
| space_group | Space group (task label) | crystal symmetry (151 groups with ≥ 10 entries) | symmetry analysis of the relaxed structure |
""",
    [
        """
mp = pd.read_parquet(HERE / 'mp_properties.parquet')
print(mp.shape); display(split_table(mp)); mp.head(3)
""",
        """
# Coverage: rows with a value per property
raw = [c for c in mp.columns if not c.endswith(('(normalized)', '(label)')) and c not in ('id', 'MP id', 'composition', 'split')]
mp[raw].notna().sum().sort_values().plot.barh(figsize=(7, 7), color='#0f4c5c', title='Rows with a value'); plt.tight_layout()
""",
        """
hist_grid(mp, ['Formation energy per atom', 'Band gap', 'Density', 'Density atomic', 'Total magnetization per formula unit',
               'Bulk modulus', 'Shear modulus', 'Dielectric total', 'Refractive index'],
          log={'Dielectric total', 'Bulk modulus', 'Shear modulus'}, title='Raw values of selected MP properties')
""",
        """
fig, axes = plt.subplots(1, 3, figsize=(13, 3))
for ax, c in zip(axes, ['Magnetic ordering', 'Is metal', 'Is gap direct']):
    v = mp[c].astype(str).replace('nan', np.nan).dropna().value_counts(); ax.bar(v.index, v.values, color='#b8652a'); ax.set_title(c)
plt.tight_layout()
sg = mp['Space group'].dropna().value_counts()
print(f"space groups present: {sg.size}; the 12 most common hold {sg.head(12).sum() / sg.sum():.0%} of the rows"); sg.head(12).to_frame('rows').T
""",
        """
# Raw vs normalized: the transform the model sees, for the band gap (half the values are exactly 0 — metals)
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].hist(mp['Band gap'].dropna(), bins=80, color='#0f4c5c'); axes[0].set_title('Band gap (eV), raw')
axes[1].hist(mp['Band gap (normalized)'].dropna(), bins=80, color='#4f8473'); axes[1].set_title('Band gap, normalized')
plt.tight_layout()
""",
    ],
)

# ---------------------------------------------------------------- MP DOS
section(
    "## 3. Materials Project density of states — `mp_dos_curves.parquet`",
    """
**Source.** Materials Project, the DOS of metallic compounds (10,321 `mp-…` rows), smoothed and resampled on a
common energy grid by `data/scripts/DOS_smoothing.ipynb`.

**Task: `dos_density`** (curve). The electronic density of states counts how many electron states exist per unit
energy; its value near the Fermi level governs conductivity, heat capacity and magnetic instabilities. Computed
from the DFT band structure on a dense k-point mesh. Each row stores 300 points on a fixed grid of E − E_F from
about −6 to +10 eV (`DOS energy`), the smoothed DOS (`DOS density`, states/eV) and its normalized form.
The model treats it as a curve task: it predicts the DOS value at any requested energy.
""",
    [
        """
dos = pd.read_parquet(HERE / 'mp_dos_curves.parquet')
print(dos.shape); display(split_table(dos))
print('points per curve:', dos['DOS density'].map(len).value_counts().to_dict())
""",
        """
fig, ax = plt.subplots(figsize=(8, 3.2))
for _, r in dos.sample(5, random_state=1).iterrows():
    ax.plot(r['DOS energy'], r['DOS density'], lw=1.2, label=r['composition'])
ax.axvline(0, color='grey', lw=0.8, ls='--'); ax.set_xlabel('E − E_F (eV)'); ax.set_ylabel('DOS (states/eV)'); ax.legend(fontsize=8); ax.set_title('Five DOS curves')
plt.tight_layout()
""",
    ],
)

# ---------------------------------------------------------------- transport curves
section(
    "## 4. Temperature-dependent transport and magnetic curves — `transport_curves.parquet`",
    """
**Source.** Starrydata: Katsura, Y. et al. *Starrydata: from published plots to shared materials data*,
Sci. Technol. Adv. Mater. Methods 5, 2506976 (2025). An open database of curves digitized from figures in
published experimental papers (thermoelectric and magnetic materials). This is the "te" part of the mixed table;
rows with `qa-` ids carry property curves of quasicrystals from HYPOD-X.

**How the values are obtained.** Each curve is a measurement on one physical sample, digitized from a published
plot, then interpolated and smoothed onto 300 points between its first and last temperature
(`data/scripts/process_ac_qe_te_data.ipynb`). Units below are SI as stored (inferred from magnitudes; the magnetic
susceptibility unit is not recorded).

| task | column | meaning | unit |
|---|---|---|---|
| seebeck | Seebeck coefficient | voltage generated per kelvin of temperature difference; sign = carrier type | V/K |
| electrical_resistivity | Electrical resistivity | resistance to current | Ω·m |
| thermal_conductivity | Thermal conductivity | heat flow per temperature gradient (electrons + phonons) | W/(m·K) |
| power_factor | Power factor | S²/ρ, the electrical part of thermoelectric performance | W/(m·K²) |
| zt | ZT | thermoelectric figure of merit S²T/(ρκ) | dimensionless |
| magnetic_susceptibility | Magnetic susceptibility | magnetization per applied field (98 curves only) | not recorded |

Every curve task pairs the values with their temperatures (`<name> (T/K)`); the model predicts the property at any
requested temperature.
""",
    [
        """
tc = pd.read_parquet(HERE / 'transport_curves.parquet')
print(tc.shape); display(split_table(tc))
names = ['Seebeck coefficient', 'ZT', 'Power factor', 'Thermal conductivity', 'Electrical resistivity', 'Magnetic susceptibility']
pd.DataFrame({n: [tc[n].notna().sum(), tc.loc[tc[n].notna(), 'source table'].value_counts().to_dict()] for n in names}, index=['curves', 'by row source']).T
""",
        """
fig, axes = plt.subplots(2, 3, figsize=(13, 6)); rng = np.random.default_rng(0)
for ax, n in zip(axes.ravel(), names):
    sub = tc[tc[n].notna()]
    for i in rng.choice(len(sub), size=min(6, len(sub)), replace=False):
        r = sub.iloc[i]; ax.plot(r[f'{n} (T/K)'], r[n], lw=1)
    ax.set_title(n); ax.set_xlabel('T (K)')
    if n in ('Electrical resistivity', 'Magnetic susceptibility'): ax.set_yscale('log')
plt.tight_layout()
""",
        """
# Temperature coverage: first and last temperature of every Seebeck curve
s = tc[tc['Seebeck coefficient (T/K)'].notna()]['Seebeck coefficient (T/K)']
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(s.map(min), bins=60, alpha=.8, label='curve start', color='#0f4c5c'); ax.hist(s.map(max), bins=60, alpha=.8, label='curve end', color='#b8652a')
ax.set_xlabel('T (K)'); ax.legend(); ax.set_title('Temperature range of the Seebeck curves'); plt.tight_layout()
""",
    ],
)

# ---------------------------------------------------------------- NEMAD magnetic
section(
    "## 5. NEMAD magnetic materials — `nemad_magnetic.parquet` (+ `_raw`)",
    """
**Source.** NEMAD, the Northeast Materials Database: Itani, S., Zhang, Y. & Zang, J. *The northeast materials
database for magnetic materials*, Nat. Commun. (2025), doi:10.1038/s41467-025-64458-z; nemad.org. Built by large
language models (GPT-4o with structured prompts) reading experimental papers from Elsevier and APS journals and
extracting composition, transition temperatures and magnetic properties.

**How the values are obtained.** Experimental values as reported in the papers, extracted automatically, then
cleaned and unit-converted by `data/scripts/nemad_magnetic.py`. Text mining brings some noise (misread units,
mixed sample conditions).

| task | column | meaning | unit |
|---|---|---|---|
| curie | Curie temperature | temperature above which a ferromagnet loses spontaneous magnetization | K |
| neel | Neel temperature | ordering temperature of an antiferromagnet | K |
| magnetization | Magnetization | magnetic moment per mole (saturation / reported value) | A·m²/mol |
| magnetic_moment | Magnetic moment | moment per formula unit | μB/f.u. |
""",
    [
        """
nm = pd.read_parquet(HERE / 'nemad_magnetic.parquet'); nmr = pd.read_parquet(HERE / 'nemad_magnetic_raw.parquet')
print('task table', nm.shape, '| raw table', nmr.shape); display(split_table(nm)); nmr.head()
""",
        """
hist_grid(nmr, ['Curie temperature[K]', 'Neel temperature[K]', 'Magnetization[A·m²/mol]', 'Magnetic moment[μB/f.u.]'], ncols=4,
          log={'Magnetization[A·m²/mol]'}, title='Raw values (NEMAD magnetic)')
""",
    ],
)

# ---------------------------------------------------------------- NEMAD SC
section(
    "## 6. NEMAD superconductors — `nemad_superconductor.parquet` (+ `_raw`)",
    """
**Source.** NEMAD (nemad.org), superconductor part: experimental superconducting transition temperatures extracted
from the literature by the same LLM pipeline. The value per composition is the median of the reported Tc.

**Task: `tc`** — superconducting transition temperature (K): the temperature below which the electrical resistance
vanishes and magnetic flux is expelled. Measured by resistivity, magnetization or specific heat. The raw table also
records the pressure of the measurement (high-pressure superconductors such as hydrides have much higher Tc).
""",
    [
        """
sc = pd.read_parquet(HERE / 'nemad_superconductor.parquet'); scr = pd.read_parquet(HERE / 'nemad_superconductor_raw.parquet')
print('task table', sc.shape, '| raw table', scr.shape); display(split_table(sc)); scr.head()
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(scr['Transition temperature[K]'].dropna(), bins=80, color='#0f4c5c'); axes[0].set_xlabel('Tc (K)'); axes[0].set_title('Transition temperature')
p = scr['Pressure[GPa]'].dropna(); axes[1].hist(p[p > 0], bins=60, color='#b8652a'); axes[1].set_xlabel('pressure (GPa), > 0 only'); axes[1].set_title(f'{(p > 0).sum():,} entries measured under pressure')
plt.tight_layout()
""",
    ],
)

# ---------------------------------------------------------------- phonix
section(
    "## 7. phonix-db lattice thermal conductivity — `phonix_thermal_conductivity.parquet` (+ `_raw`)",
    """
**Source.** Phonix database: *Database and deep-learning scalability of anharmonic phonon properties by automated
brute-force first-principles calculations*, npj Comput. Mater. 12, 150 (2026), doi:10.1038/s41524-026-02033-w;
Hugging Face `phonix-db/phonix-full` (CC BY 4.0). Inorganic, non-metallic, non-magnetic compounds from the
Materials Project and Phonondb.

**How the values are obtained.** First-principles anharmonic phonon calculations (auto-kappa workflow: VASP force
constants, ALAMODE for the Boltzmann transport equation). Filtered here to force-constant fitting errors ≤ 10 % and
κ ≤ 2000 W/(m·K) (`data/scripts/phonix-db.ipynb`).

| task | column | meaning | unit |
|---|---|---|---|
| kp | kp | lattice thermal conductivity from the phonon Boltzmann equation (particle-like phonon transport) | W/(m·K) |
| klat | klat | total lattice thermal conductivity, κ_p plus the coherence (wave-like) contribution κ_c | W/(m·K) |
""",
    [
        """
ph = pd.read_parquet(HERE / 'phonix_thermal_conductivity.parquet'); phr = pd.read_parquet(HERE / 'phonix_thermal_conductivity_raw.parquet')
print('task table', ph.shape, '| raw table', phr.shape); display(split_table(ph)); phr.head()
""",
        """
hist_grid(phr, ['kp[W/mK]', 'klat[W/mK]', 'kc[W/mK]'], log={'kp[W/mK]', 'klat[W/mK]', 'kc[W/mK]'}, title='Thermal conductivity (log10 W/mK)')
""",
    ],
)

# ---------------------------------------------------------------- HEA
section(
    "## 8. KKR-CPA high-entropy alloys — `kkr_cpa_hea.parquet` (+ `_raw`)",
    """
**Source.** Fukushima, T., Akai, H., Chikyow, T. & Kino, H. *Automatic exhaustive calculations of large material
space by Korringa-Kohn-Rostoker coherent potential approximation method applied to equiatomic quaternary high
entropy alloys*, Phys. Rev. Mater. 6, 023802 (2022); data Zenodo 10.5281/zenodo.5606502 (CC BY 4.0).
Every equiatomic quaternary of 38 metals and metalloids (73,165 converged compositions), in both bcc and fcc.

**How the values are obtained.** All-electron KKR-CPA calculations (AkaiKKR on Fugaku): the coherent potential
approximation treats the four elements as randomly mixed on one lattice, so a disordered solid solution is computed
without supercells. Processed by `data/scripts/process_kkr_cpa_hea_20261011.py`. bcc and fcc are separate tasks.

| task | column | meaning | unit |
|---|---|---|---|
| hea_total_moment_{bcc,fcc} | Total magnetic moment | net magnetic moment per atom of the disordered alloy (absolute value) | μB |
| hea_curie_{bcc,fcc} | Curie temperature | magnetic transition temperature from the computed exchange interactions (mean-field) | K |
| hea_resistivity_{bcc,fcc} | Residual resistivity | resistivity at 0 K caused by chemical disorder alone (Kubo-Greenwood) | μΩ·cm |
| hea_structure_preference | Structure preference | the structure with the lower total energy | class: bcc / fcc |

About 70 % of the compositions are non-magnetic (moment and Curie temperature exactly 0). Volume and lattice
constant are inputs of the calculation, not results, and are left out; the raw table keeps the fcc − bcc energy
difference (eV/atom) behind the structure label.
""",
    [
        """
hea = pd.read_parquet(HERE / 'kkr_cpa_hea.parquet'); hear = pd.read_parquet(HERE / 'kkr_cpa_hea_raw.parquet')
print('task table', hea.shape, '| raw table', hear.shape); display(split_table(hea)); hear.head()
""",
        """
hist_grid(hear, ['Total magnetic moment bcc[μB]', 'Curie temperature bcc[K]', 'Residual resistivity bcc[μΩ cm]',
                 'Total magnetic moment fcc[μB]', 'Curie temperature fcc[K]', 'Residual resistivity fcc[μΩ cm]'], title='Raw values (KKR-CPA HEA)')
print('share exactly zero:', {c: f"{(hear[c] == 0).mean():.0%}" for c in ['Total magnetic moment bcc[μB]', 'Curie temperature bcc[K]']})
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
v = hear['Structure preference'].value_counts(); axes[0].bar(v.index, v.values, color='#b8652a'); axes[0].set_title('Structure preference')
e = hear['Energy fcc - bcc[eV/atom]'].dropna(); e = e[e.abs() < 0.68]
axes[1].hist(e, bins=80, color='#0f4c5c'); axes[1].axvline(0, color='grey', ls='--'); axes[1].set_xlabel('E_fcc − E_bcc (eV/atom)'); axes[1].set_title('Energy difference behind the label')
plt.tight_layout()
""",
    ],
)

# ---------------------------------------------------------------- QMOF
section(
    "## 9. QMOF metal–organic frameworks — `qmof.parquet` (+ `_raw`)",
    """
**Source.** Rosen, A. S. et al. *Machine learning the quantum-chemical properties of metal–organic frameworks for
accelerated materials discovery*, Matter 4, 1578–1597 (2021); figshare 10.6084/m9.figshare.13147324, v18
(CC BY 4.0). 20,372 MOFs and coordination polymers, mostly experimental structures from the CSD.

**How the values are obtained.** Periodic DFT: structures relaxed with PBE-D3(BJ), properties from PBE single
points. MOFs that share a reduced formula (1,545 formulas) are reduced to the one with the lowest PBE energy per
atom (`data/scripts/process_qmof_20261011.py`), leaving 18,313 compositions. In a MOF, carbon, hydrogen, nitrogen
and oxygen make up about 95 % of the atoms, so the metal node is a small part of the composition.

| task | column | meaning | unit |
|---|---|---|---|
| qmof_band_gap_pbe | QMOF band gap PBE | electronic band gap (PBE underestimates it) | eV |
| qmof_cbm_pbe / qmof_vbm_pbe | QMOF CBM / VBM PBE | band edges (not referenced to the vacuum level) | eV |
| qmof_energy_per_atom_pbe | QMOF energy per atom PBE | DFT total energy per atom | eV/atom |
""",
    [
        """
qm = pd.read_parquet(HERE / 'qmof.parquet'); qmr = pd.read_parquet(HERE / 'qmof_raw.parquet')
print('task table', qm.shape, '| raw table', qmr.shape); display(split_table(qm)); qmr.head()
""",
        """
hist_grid(qmr, ['QMOF band gap PBE[eV]', 'QMOF CBM PBE[eV]', 'QMOF VBM PBE[eV]', 'QMOF energy per atom PBE[eV/atom]'], ncols=4, title='Raw values (QMOF)')
""",
    ],
)

# ---------------------------------------------------------------- Catalysis-Hub
section(
    "## 10. Catalysis-Hub adsorption energies — `catalysis_hub_adsorption.parquet` (+ `_raw`)",
    """
**Source.** Mamun, O., Winther, K. T., Boes, J. R. & Bligaard, T. *High-throughput calculations of catalytic
properties of bimetallic alloy surfaces*, Sci. Data 6, 76 (2019); Catalysis-Hub publication `MamunHighT2019`.
1,812 surfaces with data: 35 pure metals, 611 AB (L1₀) and 1,166 A₃B (L1₂) alloys.

**How the values are obtained.** DFT (Quantum ESPRESSO, BEEF-vdW) on a 2×2, 3-layer slab cut from the bulk alloy
along (111) for pure metals and A₃B, (101) for AB. An adsorbate is placed on every distinct site and relaxed; the
adsorption energy is ΔE = E(slab + X) − E(slab) − Σ E(gas references), referenced to H₂, H₂O, CH₄, N₂ and H₂S.
The value per (composition, adsorbate) is the most stable site. Negative = strong binding. Processed by
`data/scripts/process_cathub_mamun2019_20261012.py`; a primer on the physics is in
`data/CatalysisHub_MamunHighT2019/catalysis_primer.html`.

**Tasks**: one per adsorbate (`ads_h`, `ads_o`, `ads_s`, `ads_n`, `ads_c` on ~1,360–1,520 surfaces; the fragments
`ads_sh`, `ads_nh`, `ads_ch`, `ads_h2o`, `ads_ch2`, `ads_ch3`, `ads_oh` on 178–279). Adsorption energies of a few
simple atoms are the standard descriptors of catalytic activity (Sabatier principle, scaling relations).
""",
    [
        """
ch = pd.read_parquet(HERE / 'catalysis_hub_adsorption.parquet'); chr_ = pd.read_parquet(HERE / 'catalysis_hub_adsorption_raw.parquet')
print('task table', ch.shape, '| raw table', chr_.shape); display(split_table(ch)); chr_.filter(regex='composition|^(H|O|C)\\\\* ').head()
""",
        """
cols = [c for c in chr_.columns if c.endswith('adsorption energy[eV]')]
hist_grid(chr_, cols, ncols=4, title='Adsorption energies (eV), most stable site')
""",
        """
# Scaling relation: O* against OH* on surfaces that carry both
both = chr_.dropna(subset=['O* adsorption energy[eV]', 'OH* adsorption energy[eV]'])
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.scatter(both['O* adsorption energy[eV]'], both['OH* adsorption energy[eV]'], s=12, color='#0f4c5c')
ax.set_xlabel('O* (eV)'); ax.set_ylabel('OH* (eV)'); ax.set_title(f'Scaling between O* and OH* (n = {len(both)})'); plt.tight_layout()
print('Pearson r =', round(both['O* adsorption energy[eV]'].corr(both['OH* adsorption energy[eV]']), 3))
""",
    ],
)

md("""
## 11. Known issues (data audit, 2026-10-13)

Recorded in `experiments/data_audit_20261013/README.md`; nothing here has been changed in the files.

- **48 rows with symbols outside the 94-element table** (36 Starrydata rows with placeholders such as M, Ln, Q,
  Ze, or Am; 10 NEMAD Cm compounds; 2 NEMAD Am superconductors). The KMD encoder silently drops those symbols.
- **Split conflicts across datasets**: 4,688 compositions are train in one dataset and val/test in another; 265
  inside the qc table. A test composition can be seen in training through another task.
- **Zero-inflated targets**: band gap 49 % zero, HEA magnetic moment and Curie temperature ≈70 % zero.
- Different levels of theory (GGA/GGA+U, PBE, BEEF-vdW, KKR-CPA) — same-named quantities are separate tasks.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print("wrote", OUT, len(cells), "cells")

ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print("executed")
