# Training-data licences (audit 2026-11-04)

Every source behind the 64 composition-keyed tasks in `data/fm_task_datasets_20261013/task_index.csv`. Polymer
data (PoLyInfo, RadonPy) and AGIS are not in that task set and are not audited here; a release that adds them
needs its own row.

"Verified" means the licence was read from the record that hosts the data (the repository API or the site's own
licence statement) on the audit date, not from a third-party listing.

## Summary

| Source | Tasks | Data record | Licence | How verified |
|---|---|---|---|---|
| Materials Project | 27 scalar + DOS curve, 4 classification; also the `others` compositions of `material_type` | api.materialsproject.org, pulled 2026-09-12 | CC BY 4.0 for entries tagged `BY-C`; entries tagged `BY-NC` (the GNoME batch) carry the GNoME Database License and Terms of Use | API source code (see Notes) |
| HYPOD-X | `material_type` positives; transport curves on `qa-` rows | figshare 10.6084/m9.figshare.25650705 (v4, 2025-01-21) | CC BY 4.0 | figshare API |
| Starrydata2 | 6 transport curves (`seebeck`, `zt`, `power_factor`, `thermal_conductivity`, `electrical_resistivity`, `magnetic_susceptibility`); `others` compositions of `material_type` | figshare 10.6084/m9.figshare.30747185 (2025-12-01 release) | CC BY 4.0 | figshare API, on the 2025-12-01 release |
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
   trained on it. The release does not redistribute any training data (decision 2026-11-04), so gating the
   weights and licensing them non-commercially does not conflict with it.

None of the verified sources is ShareAlike, NoDerivatives or non-commercial, so a non-commercial,
share-alike licence on the weights is compatible with all of them. The open item below is the only
exception.

## Open items

1. **Materials Project `BY-NC` entries.** The MP API tags every document `BY-C` or `BY-NC`
   (`builder_meta.license`). The `BY-NC` documents are the GNoME batch (`batch_id = "gnome_r2scan_statics"`),
   governed by a separate GNoME Database License and Terms of Use. The API's default query is `BY-C`, and the
   2026-09-12 rebuild takes GGA / GGA+U calculations only while GNoME is an r2SCAN batch. So GNoME entries are
   unlikely to be in the data, but the scripts did not filter by licence and did not record it. To close this
   item: query the material ids of `mp_properties.parquet` and `mp_dos_curves.parquet` with `license = "All"`
   and list any `BY-NC` id. If any exist, either drop them or read the GNoME terms before release.
2. **Starrydata release used.** The CC BY 4.0 reading is from the 2025-12-01 release. The training data were
   built from an earlier export (`starry-` rows of the 2025-06-15 qc table). Confirm the licence of that release.
3. **`element_features.csv` provenance.** Add a header comment or a sidecar note naming the XenonPy version
   it was exported from, so the BSD-3 notice can be reproduced in the package.
4. **NEMAD.** The licence comes from machine-readable metadata on the download page; there is no separate
   terms page. Re-check before release, and record the download date of the 2026-04-19 / 2026-04-25 exports.

## Citations for the model card

- Materials Project: A. Jain et al., *APL Materials* 1, 011002 (2013). Data from api.materialsproject.org.
- HYPOD-X: E. Fujita et al., *Sci. Data* 11, 1211 (2024); figshare 10.6084/m9.figshare.25650705.
- Starrydata: Y. Katsura et al., *Sci. Technol. Adv. Mater. Methods* 5, 2506976 (2025); figshare
  10.6084/m9.figshare.30747185.
- NEMAD: S. Itani, Y. Zhang & J. Zang, *Nat. Commun.* (2025), doi:10.1038/s41467-025-64458-z; nemad.org.
- phonix-db: *npj Comput. Mater.* 12, 150 (2026), doi:10.1038/s41524-026-02033-w; huggingface.co/datasets/phonix-db/phonix-full.
- KKR-CPA HEA: T. Fukushima, H. Akai, T. Chikyow & H. Kino, *Phys. Rev. Mater.* 6, 023802 (2022);
  Zenodo 10.5281/zenodo.5606502.
- QMOF: A. S. Rosen et al., *Matter* 4, 1578–1597 (2021); figshare 10.6084/m9.figshare.13147324.
- Catalysis-Hub: O. Mamun, K. T. Winther, J. R. Boes & T. Bligaard, *Sci. Data* 6, 76 (2019); K. T. Winther et al.,
  *Sci. Data* 6, 75 (2019); catalysis-hub.org.
- Element features: XenonPy (BSD-3-Clause), xenonpy.readthedocs.io.

## Records consulted (2026-11-04)

- figshare API: `api.figshare.com/v2/articles/{25650705, 30747185, 13147324}` → `license.name = "CC BY 4.0"`.
- Zenodo API: `zenodo.org/api/records/5606502` → `metadata.license.id = "cc-by-4.0"`.
- Hugging Face API: `huggingface.co/api/datasets/phonix-db/phonix-full` → `cardData.license = "cc-by-4.0"`.
- nemad.org/download: schema.org JSON-LD, `"license": "https://creativecommons.org/licenses/by/4.0/"`.
- catalysis-hub.org: footer text in `main.0be89be1f15ea5bf3db3.js`.
- Materials Project: `materialsproject/emmet` on GitHub — `emmet-core/emmet/core/base.py` (`license: Literal["BY-C", "BY-NC"]`),
  `emmet-api/.../materials/query_operators.py` (license query defaults to `"BY-C"`); docs.materialsproject.org,
  "Access controlled data" (GNoME batch and its terms). The MP terms page refused automated access (HTTP 403),
  so its wording is not quoted here.
- XenonPy documentation: "Features" (`elements_completed`, 58 properties of 94 elements) and "Copyright" (BSD-3).
