# v2 running log — deviations, anomalies, and measured facts

Kept as the campaign runs, not reconstructed at the end. PLAN §9.2 requires HANDOFF.md to list
every deviation and every anomaly; recalling them afterwards is how they get lost. HANDOFF.md is
generated from this file when the campaign closes.

---

## Measured facts that replaced planning estimates

| quantity | PLAN's figure | measured | where it came from |
|---|---|---|---|
| probe6 wall clock per run | 0.8h (extrapolated from probe3) | **mean 1.93h**, p90 2.62h, range 0.94–2.80h | 288 stage-A runs, `_timing.tsv` |
| seed sigma | 5.01% (probe3, inherited) | **3.58%** | 9 completed `s0_v1enc` seeds |
| probe stage total | ~1400 GPU-h | **~3460 GPU-h** | 1778 runs × 1.93h |
| campaign total | ~1500 GPU-h | **~3550 GPU-h** | probe + stage C' |

The sigma result is the one the campaign was designed around: probe6 exists partly to average
over more tasks and shrink the seed band (PLAN §7.2), and it did — 3.58% against probe3's 5.01%,
a 28% reduction. That directly changes what the finals can resolve. At 25 seeds two arms separate
at ~1.43%, and v1's unresolved top three were 1.5–1.8% apart, so **25 seeds is the right number
and is not a place to economise** — it is exactly the margin v1 could not buy.

Noise is not evenly sourced. Per-task relative spread across 9 seeds:

| task | metric sd | mean | relative |
|---|---|---|---|
| seebeck (KR) | 0.0475 | 0.6647 | 7.1% |
| magnetic_moment (small) | 0.0489 | 0.6758 | 7.2% |
| zt (KR) | 0.0266 | 0.6896 | 3.9% |
| magnetization (small) | 0.0190 | 0.7863 | 2.4% |
| volume (big) | 0.0172 | 0.5968 | 2.9% |
| formation_energy (big) | 0.0021 | 0.9897 | 0.2% |

The two noisiest tasks are one kernel regression and one small task — the two things probe3 did
not contain. So probe6 both lowered the mean noise AND is where the remaining noise lives.
formation_energy contributes essentially nothing at 0.2%, which is the saturation PLAN §6.3
predicted; it stays in the probe for continuity with v1 but cannot move a ranking.

---

## Deviations from PLAN

**1. probe6 task composition is a choice PLAN left open.** §7.2 specifies "two per size group, two
of them kernel regressions" but not which. Every KR task is mid-sized, so both KR slots must come
from the mid group, which rules out keeping v1's `tc` there. Chosen: volume + formation_energy
(big), seebeck + zt (mid, both KR), magnetization + magnetic_moment (small). Picked for
resolution as well as size — `volume` (ceiling 0.569) carries the big group's ranking because
`formation_energy` is saturated at 0.995. Excluded `electrical_resistivity` (ceiling 0.162) and
`magnetic_susceptibility` (58 labels, degenerate baseline): both add variance without resolution.

**2. A'1 grid densified from 48 to 96 points, and extended past PLAN's range.** PLAN §2 lists
`encoder_lr ∈ {1e-3, 3e-3, 1e-2, 3e-2}`; §7.5 budgets 96 points. The extra points were spent on
`encoder_lr` — 8 values, 1e-3 … 5e-2 — rather than on new axes, because v1 measured it as the
dominant knob (~2/3 of the total gain) and because v1's "0.01 diverges" verdict was a symptom of
having no working annealing, so where the ceiling actually sits is an open question. 5e-3 is
included so the untuned default sits inside the grid as an interior point.

**3. A new submit path instead of reusing v1's.** v1's `fm_array.sbatch` defaults `IMAGE` to the
0.2.1 container and v1's `submit.sh` never exports `IMAGE`. Reusing that would have run the whole
v2 campaign under the per-batch scheduler cadence v2 exists to escape — silently, since the runs
would exit 0 and drop DONE markers. v2's worker requires `IMAGE` and asserts the container's own
reported version against `EXPECT_VERSION`. Verified negatively: 0.2.1 against an expectation of
0.3.2 exits 3, leaves no DONE marker, and therefore re-runs on resubmission rather than being
skipped.

**4. A separate RIKYU checkout (`~/projects/foundation_model_v2`).** v1's Stage C was still
running out of `~/projects/foundation_model` and another session owns that tree. A full copy costs
375 MB against 32 GB free, so isolation was cheaper than coordination.

**5. A'1 and A'1r launched in parallel with stage 0, not strictly after it.** PLAN §1 requires
stages to run in sequence, but that rule is about dependency: B' builds on A's fixed base, C'
builds on A'+B'. Stage 0 fixes nothing — it is a reference measurement — and the A' grids and
seed counts are specified by PLAN, not derived from stage 0. Stage 0's calibration only sizes the
FINALS. By the time A' launched, stage 0 had already shown the mechanism working end to end
(config resolves, replay applies, epochs advance, early stopping fires, image gate holds), which
was the other reason to wait. Saved roughly 45 minutes of serial time.

**6. Probe walltime raised from 3h to 6h.** See the anomaly below; this is the fix for it.

---

## Anomalies

**1. Stage 0's first attempt lost 7 of 18 runs to the walltime.** It was submitted before the
walltime was raised, at 3h, and probe6 runs take 1.9–2.9h with the slow tail past 3h. All 9
`s0_v1enc` runs finished (2.07–2.88h); only 2 of 9 `s0_base` did. Resubmitted as job 53185 at 6h —
the DONE markers make this idempotent, so only the 7 missing re-ran. **The untuned anchor is the
reference for every margin in the campaign, so no A' scoring was done until those landed.**

**2. The untuned baseline is materially slower than v1's adopted config.** `s0_base`
(latent 128, encoder_lr 5e-3) exceeded 3h where `s0_v1enc` (latent 384, encoder_lr 1e-3) ran
2.1–2.9h. The stage-A timing distribution shows the same pattern: the fastest runs are high
`encoder_lr` with high scheduler `patience`, the slowest are low `encoder_lr` or `patience = 5`.
Worth reporting alongside accuracy — under the fixed scheduler, the better settings appear to be
cheaper as well, which is a second axis of benefit v1 could not see.

**3. `volume` exceeds its own historical single-task ceiling.** At step 1 of a probe6 run,
`volume` reaches R² 0.615 against a recorded ceiling of 0.5685. Those ceilings come from H200
hardware in a different container, so this is a reference-frame mismatch, not a result. It
confirms v1's rule that ranking must be against the in-campaign anchor and never against the
ceilings — the ceilings stay as headroom context only. Any deficit-to-ceiling figure that comes
out NEGATIVE for `volume` must be reported as such, not clipped to zero.

**4. Zero run failures across 288 stage-A runs** other than the walltime kills above. Recorded
because it is the kind of thing that is only reassuring if someone actually checked.

---

## Compute budget

Approved: 2750 GPU-h (the user's response to the pessimistic estimate). Projected on measured
timings: **~3550 GPU-h**, i.e. 1.3× the approved figure, driven entirely by the per-run cost being
1.93h rather than the 0.8h PLAN extrapolated from a probe with no kernel-regression heads.

Not cut, and why: the random search (a1r, ~1160 GPU-h) is the one discretionary block — PLAN §7.4
frames it as "when compute is plentiful". By the time the true cost was known it was already 29%
complete, so cancelling would have saved ~820 GPU-h while discarding coverage already paid for.
The finals' 25 seeds are explicitly NOT a candidate for economising: at σ = 3.58% they resolve
1.43%, and resolving ~1.5% is the entire reason the finals exist.

If a cut becomes necessary, the order is: drop a1r's remainder, then trim the B' grid points, then
— last — the finals seed count.
