# `qc_inverse_design_teaching.parquet` — sources and licence

The teaching table: 29,802 compositions, one row per composition, built by `../prepare_data.py`.

**Licence:** CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Every source below is CC BY 4.0.
This table is an adaptation of them and is redistributed on the same terms. It is not covered by the
repository's Apache-2.0 code licence.

| Column | Non-null rows | Source | Licence |
|---|---|---|---|
| `material_type`, `material_type_name` | 12,361 | AC / QC labels: HYPOD-X. `others`: compositions from Materials Project and Starrydata2 | CC BY 4.0 |
| `formation_energy` | 8,368 | Materials Project (export of 2025-04-10) | CC BY 4.0 |
| `magnetization` | 1,626 | NEMAD, magnetic set (2026-04-19) | CC BY 4.0 |
| `tc` | 10,458 | NEMAD, superconductor set (2026-04-25) | CC BY 4.0 |
| `klat` | 5,881 | phonix-db (2026-04-25) | CC BY 4.0 |

## Changes made to the source data

The sources were merged on a canonical composition string, an outer join with missing values left as NaN.
The other steps:

- The five material-type labels were collapsed to three: AC, QC and others.
- The `others` class was down-sampled to 12,000 rows.
- `formation_energy` and `material_type` carry the normalisation and split of the project's qc table.
- `magnetization`, `tc` and `klat` were transformed by `log1p`, then a z-score fitted on training rows, then
  clipped.
- Rows without a qc split were given a random 70/15/15 split. Details are in `../prepare_data.py`.

## Citations

- HYPOD-X: E. Fujita et al., *Sci. Data* 11, 1211 (2024); figshare 10.6084/m9.figshare.25650705.
- Materials Project: A. Jain et al., *APL Materials* 1, 011002 (2013); api.materialsproject.org.
- Starrydata: Y. Katsura et al., *Sci. Technol. Adv. Mater. Methods* 5, 2506976 (2025);
  figshare 10.6084/m9.figshare.29205863.
- NEMAD: S. Itani, Y. Zhang & J. Zang, *Nat. Commun.* (2025), doi:10.1038/s41467-025-64458-z; nemad.org.
- phonix-db: *npj Comput. Mater.* 12, 150 (2026), doi:10.1038/s41524-026-02033-w;
  huggingface.co/datasets/phonix-db/phonix-full.

Licence evidence for each source: `docs/licensing/data_sources.md`.
