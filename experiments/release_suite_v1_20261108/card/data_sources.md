## Training data

The weights were trained on the sources below. The training data are **not** released with the
weights; only the weights, their configuration and these notes are. All sources are licensed
CC BY 4.0. Audit: `docs/licensing/data_sources.md` in the code repository.

| Source | Tasks in this release | Record used |
|---|---|---|
| Materials Project | 22 regression, the DOS curve (shape), 4 classification; also the `others` class of `material_type` | api.materialsproject.org, pulled 2026-09-12 (GGA / GGA+U entries) |
| HYPOD-X | `material_type` | figshare 10.6084/m9.figshare.25650705 |
| Starrydata2 | 6 temperature curves: `seebeck`, `zt`, `power_factor`, `thermal_conductivity`, `electrical_resistivity`, `magnetic_susceptibility` | figshare 10.6084/m9.figshare.29205863 (release of 2025-05-31) |
| NEMAD | `magnetic_moment`, `magnetization`, `curie`, `neel`, `tc` | nemad.org, exports of 2026-04-19 / 2026-04-25 |
| phonix-db | `kp`, `klat` | huggingface.co/datasets/phonix-db/phonix-full |

**Changes made to the data.**
- Compositions were reduced to formula keys. Targets were normalised: standard scaling plus
  Yeo-Johnson, `log1p` plus z-score for NEMAD and phonix-db, and per-curve max-abs scaling for the DOS.
  Transport curves were resampled to 300 points per curve, the DOS to 300 points on E − E_F ∈ [−6, 10] eV.
- Splits are grouped by composition, so one composition never appears in two splits.
- 31 Materials Project entries whose licence tag could not be read through the API were excluded.

**Input features.** Element features from XenonPy (BSD-3-Clause), shipped with the code.

## Citations

- Materials Project: A. Jain et al., *APL Materials* 1, 011002 (2013).
- HYPOD-X: E. Fujita et al., *Sci. Data* 11, 1211 (2024).
- Starrydata: Y. Katsura et al., *Sci. Technol. Adv. Mater. Methods* 5, 2506976 (2025).
- NEMAD: S. Itani, Y. Zhang & J. Zang, *Nat. Commun.* (2025), doi:10.1038/s41467-025-64458-z.
- phonix-db: *npj Comput. Mater.* 12, 150 (2026), doi:10.1038/s41524-026-02033-w.
- XenonPy: xenonpy.readthedocs.io.
