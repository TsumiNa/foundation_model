# Training-data licences (audit 2026-11-04)

Every source behind the 64 composition-keyed tasks in `data/fm_task_datasets_20261013/task_index.csv`. Polymer
data (PoLyInfo, RadonPy) and AGIS are not in that task set and are not audited here; a release that adds them
needs its own row.

"Verified" means the licence was read from the record that hosts the data (the repository API or the site's own
licence statement) on the audit date, not from a third-party listing.

## Summary

| Source | Tasks | Data record | Licence | How verified |
|---|---|---|---|---|
| Materials Project | 27: 22 scalar regression, 1 DOS curve, 4 classification; also the `others` compositions of `material_type` | api.materialsproject.org, pulled 2026-09-12 | CC BY 4.0: every entry whose tag could be read is `BY-C`; none carries the GNoME terms that apply to `BY-NC` entries; 31 rows unresolved | MP API, per-entry `builder_meta.license` (open item 1) |
| HYPOD-X | `material_type` positives; transport curves on `qa-` rows | figshare 10.6084/m9.figshare.25650705 (v4, 2025-01-21) | CC BY 4.0 | figshare API |
| Starrydata2 | 6 transport curves (`seebeck`, `zt`, `power_factor`, `thermal_conductivity`, `electrical_resistivity`, `magnetic_susceptibility`); `others` compositions of `material_type` | figshare 10.6084/m9.figshare.29205863 (release of 2025-05-31, the one used) | CC BY 4.0 | figshare API (open item 2) |
| NEMAD | `magnetic_moment`, `magnetization`, `curie`, `neel`, `tc` | nemad.org (`/download`) | CC BY 4.0 | the download page's schema.org dataset metadata |
| phonix-db | `kp`, `klat` | Hugging Face `phonix-db/phonix-full` | CC BY 4.0 | Hugging Face dataset API (`cardData.license`) |
| KKR-CPA HEA | 6 regression (bcc / fcc) + structure preference | Zenodo 10.5281/zenodo.5606502 | CC BY 4.0 | Zenodo API |
| QMOF | 4 regression | figshare 10.6084/m9.figshare.13147324 (v18, 2025-11-15) | CC BY 4.0 | figshare API |
| Catalysis-Hub (MamunHighT2019) | 12 adsorption energies | catalysis-hub.org, pulled 2026-10-11/12 | CC BY 4.0 ("Except where otherwise noted, content on Catalysis Hub is licensed under a Creative Commons Attribution 4.0 International License") | the site's footer, read from its JavaScript bundle |

Input features shipped with the package:

| Item | Origin | Licence | How verified |
|---|---|---|---|
| `src/foundation_model/utils/element_features.csv` (58 features × 94 elements, H–Pu) | matches XenonPy `elements_completed` (58 properties of 94 elements, H to Pu) | BSD-3-Clause (XenonPy) | XenonPy documentation; the match is by shape and column names, the file carries no provenance line |

## What the licences require of a weights release

CC BY 4.0 permits adaptation and redistribution, including commercial use, provided that:

1. **Attribution** is given to each source: creator, a copyright notice if supplied, a link to the licence, and a link
   to the material. The model card must carry the citation list below.
2. **Changes are indicated**: the card must state that the data were filtered, deduplicated, normalised and used
   to train a model.
3. **No downstream restrictions are placed on the licensed material itself.** This binds the data, not a model
   trained on it. The weights release redistributes no training data (decision 2026-11-04), so gating the
   weights and licensing them non-commercially does not conflict with it. The teaching table that the
   repository does redistribute is under CC BY 4.0 with no added terms (see below).

None of the verified sources is ShareAlike, NoDerivatives or non-commercial, so a non-commercial,
share-alike licence on the weights is compatible with all of them. **This conclusion is provisional on open
item 1:** no Materials Project entry whose tag could be read carries the non-commercial GNoME terms, but 31
rows could not be read.

## Data tracked in this repository

The release ships no training data with the weights. The repository itself tracks two data files:

| File | Contents | Licence |
|---|---|---|
| `src/foundation_model/utils/element_features.csv` | element features, model input, shipped in the package | BSD-3-Clause (XenonPy); full notice in `NOTICE` |
| `notebooks/teaching_inverse_design/data/qc_inverse_design_teaching.parquet` | teaching table: 29,802 compositions × 5 properties, derived from HYPOD-X, Materials Project, Starrydata2, NEMAD and phonix-db | CC BY 4.0; sources, changes and citations in `notebooks/teaching_inverse_design/data/README.md` |

The teaching table is not a sample of the pretraining set in every column. Its `tc` (10,458 rows) and `klat`
(5,881 rows) columns hold every row of the NEMAD superconductor set and of phonix-db, so for those two
properties the training values are public.

## Open items

1. **Materials Project `BY-NC` entries — checked 2026-11-04; none among readable tags, 31 rows unresolved.** The MP API tags every document
   `BY-C` or `BY-NC` (`builder_meta.license`). The `BY-NC` documents are the GNoME batch
   (`batch_id = "gnome_r2scan_statics"`), governed by a separate GNoME Database License and Terms of Use. The
   data scripts did not filter by licence, so every MP id was queried against `materials/core` with
   `license = "All"`:

   | Id set | Queried | Returned | `BY-C` | `BY-NC` | GNoME batch |
   |---|---|---|---|---|---|
   | `mp_properties.parquet`, `MP id` | 33,798 | 33,798 | 33,798 | 0 | 0 |
   | `mp_dos_curves.parquet`, legacy `id` | 10,321 | 10,318 | 10,318 | 0 | 0 |
   | `hypodx_material_type.parquet`, `mp-` ids (`others`) | 33,829 | 33,799 | 33,799 | 0 | 0 |

   The ids that returned nothing come to 31 rows of the qc table. Their legacy `mp-` id no longer resolves in
   the API, and the 2026-09-12 rebuild could not map them to a new id either (`MP id` is null). They carry:

   - MP-derived property values in `mp_properties.parquet` (31 rows);
   - 3 DOS curves;
   - the `others` label of `material_type` (30 rows).

   Their licence tag cannot be read, so absence of `BY-NC` is not shown for them. **Until a tag is recovered,
   these 31 compositions are excluded from the training run for the release.** The next query to try is
   `materials/core` with `deprecated=true`. It has not been run yet: the API key in use was rotated during
   the check.
2. **Starrydata release used — closed 2026-11-04.** The transport curves were built from
   `data/raw/starrydata_dataset_20250531/` (`data/scripts/process_ac_qe_te_data.ipynb`). That is the figshare
   release "20250601_starrydata2", published 2025-05-31 (10.6084/m9.figshare.29205863, file
   `starrydata_dataset_250601-0300.zip`), licensed CC BY 4.0. The release was matched by date. The
   neighbouring releases of 2025-04-30 and 2025-06-30 are CC BY 4.0 as well.
3. **`element_features.csv` provenance.** The BSD-3 notice is now in `NOTICE`. Still to do: add a header
   comment or a sidecar note naming the XenonPy version the table was exported from.
4. **NEMAD.** The licence comes from machine-readable metadata on the download page; there is no separate
   terms page. Re-check before release, and record the download date of the 2026-04-19 / 2026-04-25 exports.

## Citations for the model card

- Materials Project: A. Jain et al., *APL Materials* 1, 011002 (2013). Data from api.materialsproject.org.
- HYPOD-X: E. Fujita et al., *Sci. Data* 11, 1211 (2024); figshare 10.6084/m9.figshare.25650705.
- Starrydata: Y. Katsura et al., *Sci. Technol. Adv. Mater. Methods* 5, 2506976 (2025); figshare
  10.6084/m9.figshare.29205863 (release of 2025-05-31).
- NEMAD: S. Itani, Y. Zhang & J. Zang, *Nat. Commun.* (2025), doi:10.1038/s41467-025-64458-z; nemad.org.
- phonix-db: *npj Comput. Mater.* 12, 150 (2026), doi:10.1038/s41524-026-02033-w; huggingface.co/datasets/phonix-db/phonix-full.
- KKR-CPA HEA: T. Fukushima, H. Akai, T. Chikyow & H. Kino, *Phys. Rev. Mater.* 6, 023802 (2022);
  Zenodo 10.5281/zenodo.5606502.
- QMOF: A. S. Rosen et al., *Matter* 4, 1578–1597 (2021); figshare 10.6084/m9.figshare.13147324.
- Catalysis-Hub: O. Mamun, K. T. Winther, J. R. Boes & T. Bligaard, *Sci. Data* 6, 76 (2019); K. T. Winther et al.,
  *Sci. Data* 6, 75 (2019); catalysis-hub.org.
- Element features: XenonPy (BSD-3-Clause), xenonpy.readthedocs.io.

## Records consulted (2026-11-04)

- figshare API: `api.figshare.com/v2/articles/{25650705, 29205863, 30747185, 13147324}` → `license.name = "CC BY 4.0"`; the Starrydata releases were listed with `POST /v2/articles/search` (`"search_for": "starrydata2"`).
- Zenodo API: `zenodo.org/api/records/5606502` → `metadata.license.id = "cc-by-4.0"`.
- Hugging Face API: `huggingface.co/api/datasets/phonix-db/phonix-full` → `cardData.license = "cc-by-4.0"`.
- nemad.org/download: schema.org JSON-LD, `"license": "https://creativecommons.org/licenses/by/4.0/"`.
- catalysis-hub.org: footer text in `main.0be89be1f15ea5bf3db3.js`.
- Materials Project: `materialsproject/emmet` on GitHub — `emmet-core/emmet/core/base.py` (`license: Literal["BY-C", "BY-NC"]`),
  `emmet-api/.../materials/query_operators.py` (license query defaults to `"BY-C"`); docs.materialsproject.org,
  "Access controlled data" (GNoME batch and its terms). The MP terms page refused automated access (HTTP 403),
  so its wording is not quoted here.
- XenonPy documentation: "Features" (`elements_completed`, 58 properties of 94 elements) and "Copyright" (BSD-3).
