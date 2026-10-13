# Data audit — 2026-10-13

Seven composition-keyed datasets, 64 tasks (41 trained in campaign v2, 23 new). Polymer datasets are out of scope.
Per-task distribution statistics behind section 3: `uv run python experiments/data_audit_20261013/analysis/label_distribution.py`
-> `results/label_distribution.csv` (untracked). `results/dataset_inventory_vs_sheet.csv` maps the curated datasets onto the shared dataset sheet
(original row numbers kept).

## 1. Inventory

| dataset | file | compositions | tasks | level of theory / source |
|---|---|---|---|---|
| qc main table | qc_ac_te_mp_dos_reformat_20260912.pd.parquet | 49,034 (MP 33,829 · starry 14,838 · qa 367) | 34 (22 regression, 5 classification, 7 curves) | MP: DFT GGA/GGA+U; starry: experiment (literature); qa: experiment |
| NEMAD magnetic | NEMAD_magnetic_20260419_norm.parquet | 20,265 | 4 | experiment, text-mined |
| NEMAD superconductor | NEMAD_superconductor_20260425_norm.parquet | 10,458 | 1 | experiment, text-mined |
| phonix-db | phonix-db-filtered_20260425_norm.parquet | 5,881 | 2 | DFT phonons |
| KKR-CPA HEA | KKR_CPA_HEA_20261011_norm.parquet | 73,165 | 7 | all-electron KKR-CPA |
| QMOF | QMOF_20261011_norm.parquet | 18,313 | 4 | DFT PBE(-D3) |
| Catalysis-Hub | CatalysisHub_MamunHighT2019_20261012_norm.parquet | 1,812 | 12 | DFT BEEF-vdW |

## 2. Findings that need action

1. **Silently mis-encoded inputs (48 rows).** `formula_to_composition` drops symbols outside the 94-element
   table without an error (AmO2 is encoded as pure O, weights summing to 0.667). qc: 36 starry rows with
   placeholder symbols (M, Ln, Q, Ze, Zb, A, E, G, Gi, L) or Am — 29 of them carry a Seebeck curve;
   NEMAD magnetic: 10 Cm compounds; NEMAD superconductor: 2 Am compounds.
2. **The same descriptor input in train and in val/test (2,672 reduced compositions).** The training path
   resolves one global split per composition: `resolve_splits` (src/foundation_model/data/composition_sources.py)
   overlays the per-dataset split columns with precedence test > val > train, on canonical but *non-reduced*
   formulas. So a composition marked test in one dataset and train in another becomes test for every task
   (4,688 compositions carry such conflicting labels; the overlay removes them from training rather than leaking
   them — the cost is fewer training rows and per-dataset split counts that differ from the effective ones).
   What the overlay does not catch is formulas that differ only by a multiple: `C4 O8` and `C1 O2` are different
   keys but the same KMD input (atomic fractions). After the overlay, 2,672 reduced compositions still have one
   spelling in train and another in val/test (e.g. `C4 O8` train, `C1 O2` test), including within the
   Materials Project rows, which use cell formulas.
3. **Zero-inflated regression targets.** Band gap 49 % zero (metals); HEA total moment and Curie temperature
   ≈70 % zero (non-magnetic); MP total magnetization ≈7 % zero. Yeo-Johnson leaves a spike plus a tail.

## 3. Findings to note

- Poisson ratio: 33 negative values (all within −1…0.5); FeSi2Ti −0.966 is likely a failed elastic tensor;
  after scaling these sit near −8σ.
- VBM extremes (−14 eV) are isolated molecules / noble gases (He, Ne, H2) — real DFT values.
- phonix κ: normalized values clipped at +5 (7 rows).
- Catalysis-Hub H2O*: FeCo at 1.87 eV against a median of −0.20 eV (+6σ after scaling); the seven fragment
  tasks have 178–279 rows each.
- Curves with a temperature axis starting below 0 K: 1 resistivity, 3 susceptibility curves.
- Space group: 7 composition groups whose MP entries carry different labels; `Space group (label)` (214 classes,
  all rows) is not the task column — `Space group (task label)` (151 classes, 33,524 rows) is.
- Energies come from four different methods (GGA/GGA+U, PBE, BEEF-vdW, KKR-CPA): same-named quantities are
  separate tasks.
- Size spread: HEA (73k rows) exceeds the MP part of qc (34k); magnetic susceptibility has 98 curves.
