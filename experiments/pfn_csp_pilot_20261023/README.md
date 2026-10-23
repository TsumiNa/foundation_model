# PFN × CSP pilot (2026-10-23)

**Question.** Does the composition → structure-type map carry family-level rule variation that a few
same-family samples reveal but global features do not? Plan: "PFN × CSP 预实验计划" (2026-10-23, Chang Liu),
followed section by section; every departure is listed under *Deviations*.

**Hypotheses.** H1: same-family context beats an equal number of other-family samples (C2 > C3).
H1': the gain exceeds a descriptor-space nearest-neighbour context (C2 > C4). H2: a global model
loses accuracy on a held-out family, and k ≤ 10 same-family samples repair most of the loss.

## Layout

This experiment has its own Python environment (`pyproject.toml`, `uv.lock`, `requirements.txt` for the
RIKYU venv); it does not use the `foundation_model` package.

| path | content |
|---|---|
| `scripts/fetch_mp.py` | MP download, `theoretical=False`, anonymous formulas AB and ABC3; key from `MP_API_KEY` only |
| `scripts/csp_labels.py` | site roles (plan 2.2), Wyckoff skeleton L1 and aristotype matching L2 (2.3), top-K |
| `scripts/csp_features.py` | Magpie + role-ordered site descriptors + Goldschmidt t (2.4); P1 / P2 families and the small-family merge (2.5) |
| `scripts/build_dataset.py` | builds `artifacts/data/table_{AB,ABX3}.parquet` and the appendix tables |
| `scripts/engines.py` | TabPFN-3.5, TabICLv2, CatBoost with probabilities mapped to the full label set |
| `scripts/run_experiment.py` | one (class × label × partition × engine × protocol × seed) unit, protocols A and B |
| `scripts/rikyu/` | `make_units.py` → `units.txt` (180 units), `array.sbatch` (16 units per GPU) |
| `analysis/metrics.py` (+ `_test.py`) | Top-1/Top-3/log-loss/macro F1/AUROC/coverage, seed bootstrap CI |
| `analysis/summarize.py` | T2–T4, decision rules (plan 4.2), consistency checks (4.3) → `results/summary.json` |
| `analysis/make_report.py` | `results/report.html` (T1–T4, F1, F2) |
| `configs/references.toml` | reference structures of the L2 aristotype library |

## Run

```bash
uv sync                                                   # local env (macOS)
MP_API_KEY=... uv run python scripts/fetch_mp.py          # MP snapshot -> artifacts/data/
uv run python scripts/build_dataset.py                    # tables + results/appendix/
uv run python scripts/rikyu/make_units.py                 # units.txt
# RIKYU: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; TabPFN weights in ~/.cache/tabpfn
sbatch --account=<project> --export=ALL,PFN_CSP_COMMIT=<commit> --array=0-11 scripts/rikyu/array.sbatch
uv run python analysis/summarize.py && uv run python analysis/make_report.py
uv run --with pytest pytest -q -c pyproject.toml analysis/metrics_test.py
```

## Environment

tabpfn 9.2.0 (checkpoint `tabpfn-v3.5-20260909.safetensors`, the package default), tabicl 2.2.0
(checkpoint `tabicl-classifier-v2-20260212.ckpt`), catboost 1.2.10, matminer 0.10.1, pymatgen 2026.9.24,
spglib 2.8.0, networkx 3.7, torch 2.14.1 (RIKYU: cu130, AArch64). MP database snapshot 2026.09.28, fetched
2026-10-23. Seeds 0–4; Louvain seed 0; bootstrap seed 0, B = 10,000.

TabPFN-3.5 weights are under the TabPFN-3.5 Non-Commercial License (research and internal evaluation
only; outputs likewise); this pilot is such use.

## Smoke test (plan 0, 3.3): time per fit

| where | engine | one process | packed |
|---|---|---|---|
| MacBook (CPU or MPS) | TabPFN-3.5 | 0.65 s | 1.8 s at 8 processes (4.4 fits/s) |
| MacBook (MPS) | TabICLv2 | 0.70 s | — |
| RIKYU GB200 | TabPFN-3.5 | 0.32 s | 1.28 s at 16 processes per GPU (12.5 fits/s per GPU) |
| RIKYU GB200 | TabICLv2 | 0.80 s | — |

Context of 20 rows, 145 features (AB), 13 classes; the cost is dominated by the per-fit preprocessing of the
eight estimators, not by the context size. All fit sets are ≤ 1,139 rows and ≤ 19 classes.

## Execution log

- 2026-10-23 16:31 MP fetch: AB 1,784 entries, ABX3 1,787 entries (`theoretical=False`, not deprecated).
- 16:55 First build stalled on one structure in the aristotype matcher (supercell search on a large cell);
  added a 60 s cap per (structure, reference) comparison, logged in `results/appendix/l2_match_timeouts_AB.csv`.
- 17:10 RIKYU job 207544 submitted, then cancelled within minutes: the 1:1 test rejected peroxide-type
  formulas that pymatgen leaves unreduced (15 entries of six peroxides: Li2O2, Na2O2, K2O2, Rb2O2, Cs2O2, H2O2). Fixed, tables rebuilt
  (AB 1,133 → 1,139 compositions), outputs of 207544 deleted.
- 17:15 RIKYU job 207557 started (12 array tasks × 16 units, commit f29b31a); all 180 units completed,
  longest array task 18 min 47 s, no failed unit. Outputs mirrored to `artifacts/runs/`.
- 17:40 Summary, report (`results/report.html`), published as a private artifact:
  https://claude.ai/artifact/7fnHZtx5At8RYFrnj88K3B
- 17:57 Follow-ups: job 208460 (6 array tasks, 95 units from `scripts/rikyu/units_followup.txt`, commit 0d7b374);
  all completed, longest task 21 min 50 s. `analysis/followups.py` → `results/followups.json`;
  `analysis/make_followup_report.py` → `results/followups.html`, published at
  https://claude.ai/artifact/2qLhYRif6WTeis8b1owEJH

## Deviations from the plan

1. **TabPFN weights**: downloaded from the public Hugging Face repo `Prior-Labs/tabpfn_3_5` into the
   package cache instead of the package's auto-download (which asks for a Prior Labs token). Same file as the
   package default.
2. **Aristotype matching timeout**: a comparison that runs longer than 60 s counts as no match for that
   reference (2 AB compositions, listed in the appendix).
3. **Single-class contexts** (frequent for C4 and for C2 at small k) are answered with probability 1 for that
   class without calling the engine; the engines can only return context classes, so this is their output.
4. **1:1 / 1:1:3 test** uses element-amount ratios rather than pymatgen's reduced formula, which keeps
   peroxide formulas unreduced.
5. **Reference library** (plan 2.3 step 4): one reference per prototype of the 2.2 table; the hexagonal
   perovskites as four polytypes (2H BaNiO3, 4H SrMnO3, 6H BaTiO3, 9R BaMnO3); calcite and aragonite
   separately; distorted perovskites map to the cubic SrTiO3 reference without extra references. IDs in
   `configs/references.toml` and `results/appendix/reference_structures_*.csv`.
6. **Manual review of the L2 map** was done by the agent as a spot check and changed nothing; it still needs a
   crystallographer's review (see `results/appendix/prototype_map_*.csv`).
7. **Plan table 4.2, first protocol-B row** is truncated in the plan (conclusion and route columns lost). It is
   read as: extrapolation fails and k ≤ 10 same-family samples repair ≥ half of it → rule-prior PFN route.
8. **ICSD cross-check**: every `theoretical=False` entry had an ICSD id, so the check removed nothing.
9. **Plan 4.3 C4 rule inside rule A**: where the C4 same-family share is ≥ 0.8 at k = 10 or 20, C2 − C4 is
   unreadable, so rows A1/A2 cannot fire (ABX3 × P1 becomes "uncertain").
10. **Added, not in the plan**: a seed-level bootstrap CI of the pooled repair ratio (C1_A held at its seed
    mean), used only to say how close a ratio is to the 0.5 threshold.

## Follow-ups (requested 2026-10-23 after the pilot; criteria fixed before the runs)

Same data, features, seeds and engines; TabPFN-3.5 is primary, TabICLv2 the engine check. Top-1 is the
metric; differences are paired per seed with the same bootstrap CI as the pilot.

1. **Retrieval (`--protocol R`, AB and ABX3, L2 × P1).** Protocol-A split, all test queries. C4 = the k
   nearest training rows (any family), C5 = k random training rows from all families (control),
   k ∈ {5, 10, 20, 50, 100, 200}; C1 as before.
   - *Retrieval suffices*: at some k ≤ 200, C4 − C1 ≥ −3 pp and C4 − C5 ≥ 5 pp with CI excluding 0.
   - *Retrieval helps, global data still adds*: C4 − C5 ≥ 5 pp (CI excludes 0) but C4 − C1 < −3 pp at every k.
   - *No locality effect*: C4 − C5 < 3 pp at k = 20.
2. **Intermetallics split (`--part P1x`, AB, L1 and L2, protocols A and B).** P1 with the intermetallic family
   split by (block of A)-(block of B); families under 15 merged by the plan 2.5 rule (12 families, largest 202).
   - *Broad repair*: weighted gap ≥ 10 pp and pooled repair ratio at k = 10 ≥ 0.5 (plan rule B1).
   - *Repair was one family*: gap ≥ 10 pp and ratio < 0.5 (B2). Gap < 5 pp: extrapolation works (B3).
3. **Mixed context (`--protocol M`, AB L2 × P1, ABX3 L2 × P1, AB L2 × P1x).** Held-out family F; context =
   all other families + k rows of F (the same k rows as protocol B's C2), k ∈ {5, 10, 20}.
   - *Few own samples help on top of global data*: |F|-weighted M(k = 10) − C1_B ≥ 5 pp, CI excluding 0.
   - *No added value*: M(10) − C1_B < 3 pp. Between 3 and 5 pp, or CI including 0: uncertain.
   - Also reported: mixed repair ratio (M − C1_B) / (C1_A − C1_B) and M − C2 (paired).

### Follow-up outcomes (TabPFN-3.5; TabICLv2 agrees on every verdict)

| follow-up | key numbers | verdict |
|---|---|---|
| 1. retrieval | C4 − C1: AB −0.5 pp [−1.6, +0.5] at k = 100; ABX3 −1.4 pp [−2.7, −0.3] at k = 200; C4 − C5 +9.3 / +3.4 pp | retrieval suffices (from k = 50; TabICLv2 AB from k = 20) |
| 2. intermetallics split (L2) | gap 13.9 pp; pooled repair k = 10 −0.26 [−0.33, −0.20] (TabICLv2 0.12, L1 0.31) | repair was one family (B2) |
| 3. mixed context, k = 10 | M − C1_B: ABX3 +5.6 [+4.8, +6.4]; AB P1x +3.3 [+2.9, +3.7]; AB P1 +25.5 [+24.9, +25.9] | useful / uncertain (+6.2 at k = 20) / useful |

Reading: no evidence for family-specific rules; the global model performs like a lookup of the ~100 nearest
compounds, and a few examples of a new family help on top of global data, strongly only where the family's
chemistry is missing from training. This argues for a generic PFN or kNN-style model with a retrieved context
and against a library of family rules as a prior.

## Outcome (2026-10-23)

**No route is decided ("uncertain", plan 4.2): the two classes trigger different protocol-B rules.**

| main combination (L2 × P1), TabPFN-3.5 | AB | ABX3 |
|---|---|---|
| C2 − C3 at k = 20 (pp, 95% CI) | +37.3 [+34.0, +41.6] | +19.5 [+14.3, +24.7] |
| C2 − C4 at k = 20 | −13.7 [−15.9, −11.5] | −7.3 [−10.3, −4.4] |
| C2 − C1 at k = 20 | −17.1 | −12.2 |
| C4 same-family share at k = 20 | 0.64 | 0.89 (C2 − C4 unreadable) |
| rule A | A2 (locality) | uncertain |
| C1_A − C1_B, weighted by family size | 39.9 pp | 13.1 pp |
| pooled repair ratio, k = 10 | 0.50 [0.48, 0.53] | −0.42 [−0.60, −0.23] |
| rule B → decision | B1 → rule-prior PFN route | B2 → no rule prior |

- TabICLv2 gives the same rules on both main combinations (AB repair 0.59 [0.52, 0.65]).
- Sign checks pass across engines, label schemes and partitions. CatBoost is within 5 pp of TabPFN's C1
  except AB × L2 × P2 (TabPFN 5.0 pp higher).
- The AB B1 call rests on the intermetallic family (C1 drops 78 % → 14 % when held out; ten samples bring it to
  56 %) and turns into B2 with L1 labels (repair 0.45).
- In every combination C4 (nearest neighbours, any family) beats C2 (same family), and no in-context
  condition reaches C1.

Next step candidates (report section 7): retrieval-context test for the locality route; protocol B with the
intermetallics split by chemistry; the mixed global + k-same-family context (outside this plan).
