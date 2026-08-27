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

## Results that changed what the plan assumed

**PLAN's central hypothesis about high learning rates is REFUTED.** §0 argued that v1's
"`encoder_lr = 0.01` diverges" verdict was a symptom of having no working annealing, so with the
cadence fixed "更高的初始 LR 反而可能变好" — a higher start might now be better. That is why the
A'1 grid was extended up to 5e-2, past PLAN's own 3e-2. The data says the opposite, monotonically:

| encoder_lr | mean over the other axes | best reachable |
|---|---:|---:|
| [0.001, 0.002) | −1.85% | **+2.31%** |
| [0.002, 0.005) | −2.99% | +1.89% |
| [0.005, 0.012) | −6.31% | +2.15% |
| [0.012, 0.025) | −10.50% | −0.09% |
| [0.025, 0.06) | −15.32% | −2.27% |

v1's verdict survives the regime change. The upward extension found nothing, which is itself the
answer, and the grid points spent on it are what makes the answer trustworthy rather than assumed.

**The scheduler's `patience` is the one axis with an effect larger than the noise band.** It is
also the axis v1 could not test at all, because before PR #45 it counted batches and was inert.
Short patience — cut the LR early and often — wins by a wide margin:

| patience | 4 | 5 | 8 | 15 | 24 |
|---|---:|---:|---:|---:|---:|
| mean | **+0.46%** | −1.44% | −2.97% | −10.45% | −15.23% |

The spread across this axis is roughly 15%, against a 6.68% seed band. Nothing else in stage A'
comes close.

**All four continuous axes converged to INTERIOR optima, so no a1b extension is owed.** Checked
both ways (see the boundary-logic note below): the leader's `encoder_lr = 0.001671` sits at rank
32 of 206 searched values, and the lowest decile scores slightly WORSE than the interior, so the
trend does not point out of the range.

**Boundary logic had a real gap, found while checking the above.** The original test asked only
whether a short-listed config sat exactly ON an extreme value. With ~200 continuously sampled
values from the random search, none ever does, so a genuine grid edge would have been reported as
clear. The test now also compares the best score reachable in each outer decile against the
interior. On this data both tests agree, but the exact-membership test agreed for the wrong
reason and would not have caught the case it exists for.

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

**5. Every run in BOTH campaigns uses about 9% of its GPU.** From Slurm's own accounting
(`sacct --format=TRESUsageInAve`), not from a sampled `nvidia-smi`:

| campaign | stage | gpuutil | runs measured |
|---|---|---:|---:|
| v2 | A'1 grid | 8–10%, mode **9%** | 394 |
| v1 | A1 probe | 6–8%, mode **7%** | 66 |
| v1 | Stage C, 24 tasks, ~20h | **7%** | job 48088 |

The rest of the footprint says the same thing. A grid run holds a GB200 and uses **1.29 GB of its
189 GB** of device memory; it is allocated 18 CPUs and 400 GB of host RAM and consumes about **one
core** (cpu time 03:09:36 against a 03:07:15 wall clock) and 17.6 GB. So the reservation is
correct — one GPU per run, never more, `devices = 1` in every config — but the reserved GPU idles
through roughly nine tenths of the campaign.

This is NOT a v2 regression. v1's numbers are the same shape, and so is v1's 20-hour Stage C run,
so it is a property of the workload as both campaigns have always run it.

Probable cause, NOT yet proven: the model is small (an MLP encoder at 464→256→latent with small
heads) at `batch_size = 256`, so a single step is trivial for a GB200, while `[data] num_workers =
0` leaves input preparation single-threaded and serialised with training. The measured one-core
CPU usage fits that. Distinguishing "dataloader-bound" from "model-too-small-for-this-GPU" would
need a controlled comparison that has not been run.

**The remedy does not depend on which cause it is.** At 1.29 GB and 9% utilisation, several
independent runs fit on one GPU with room to spare — they are separate processes with separate
seeds and no interaction, so packing them changes throughput and nothing else about the numbers.
Four per GPU would cut the remaining probe work (~680 runs, ~1310 GPU-h) to roughly a third.

Two things to keep straight if that is adopted:

* **wall-clock comparisons stop being comparable across the change.** The observation that high
  `encoder_lr` converges faster was measured one-run-per-GPU; under packing, runs contend, and a
  timing figure from a packed stage cannot be put beside one from an unpacked stage.
* it changes nothing about accuracy, so the tuning conclusions are unaffected either way.

**Packing was then measured, and it scales almost linearly.** Eight grid points that had already
completed unpacked were re-run into a scratch output root at two pack sizes, so the comparison is
against a known per-run baseline for the *same configurations* rather than an average over a
different set:

| | wall clock per run | vs baseline | runs per GPU | throughput |
|---|---:|---:|---:|---:|
| unpacked | 2.19h | 1.00× | 1 | 1× |
| PACK=4 | 2.28h | 1.04× | 4 | **3.84×** |
| PACK=8 | 2.46h | 1.12× | 8 | **7.1×** |

Twelve percent slower per run for seven times the throughput. Every other resource still has room
at PACK=8: 10 GB of 189 GB device memory, 141 GB of the 400 GB host allocation, 8 of 32 CPUs.

**PACK=8 adopted for the remaining stages.** Not extrapolated further — PACK=16 was not measured,
and host RAM (282 GB of 400 GB) is the dimension that would bind first, so raising it would need
its own calibration rather than an assumption.

The site enforces a per-GPU CPU cap that is easy to trip: `--cpus-per-task` above 32 is rejected
outright with `[AI4S] Requested CPUs (64 cpus-per-task x 1 tasks = 64) exceed the per-GPU cap 32`.
The pack sizing is capped there.

Recorded rather than retrofitted: stages 0 and A' ran unpacked and are not re-run. That has one
consequence for the report — **wall-clock comparisons do not cross the boundary.** The observation
that high `encoder_lr` converges faster was measured unpacked; no timing figure from a packed
stage may be placed beside it.

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
