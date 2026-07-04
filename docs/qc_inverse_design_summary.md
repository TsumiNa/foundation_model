# QC inverse-design study — summary

> **Historical note.** This documents the original quasicrystal (QC) study, which ran with a
> then-hardcoded "QC-classification primary" objective. The engine has since been generalized:
> scenario targets are now fully user-specified (`[[inverse.scenarios.targets]]` — regression
> value/direction, kernel-regression curves, classification label(s) high/low, per-target
> weights) with no built-in QC objective. See [configuration.md](configuration.md) and
> [inverse_design_algorithms.md](inverse_design_algorithms.md) for the current schema; the QC
> setup below is reproducible as an ordinary scenario (`classes = [1]`, `weight = 5.0`).

One-page summary of the messages the continual-rehearsal + inverse-design pipeline
(`fm pretrain` → `fm finetune` → `fm inverse`) carries. Written so each bullet maps to either a
slide or a paragraph of the paper.

## Headline messages

### 1. A multi-task foundation model + gradient-based inverse design is an effective recipe for multi-objective materials optimisation.

* The same model is trained once on **11 supervised tasks** (7 regression + 1 kernel
  regression + 3 inverse-design tail tasks: `formation_energy`, `klat`, `material_type`).
  Continual rehearsal (small replay) keeps earlier tasks from collapsing while new ones land —
  what we get out is a single descriptor → latent representation that all heads share.
* On top of that one checkpoint we run **three multi-objective scenarios** with no retraining:
  (a) FE↓ + magnetisation↑, (b) FE↓ + Tc↑ + magnetisation↑, (c) FE↓ + klat↑. All three are
  drivenby gradient descent against a single composite loss:
  $\text{MSE-to-target} + w_{\text{cls}} \cdot (-\log P(\text{QC}))$.
* The takeaway: once the encoder + heads are good enough, *adding a new joint objective is just
  another `task_targets` entry*. No new training, no new data, no per-objective bespoke model.

### 2. For QC, the differentiable-KMD composition path gives more controllable and chemically meaningful results than latent-space optimisation.

* **Latent path** (`optimize_latent`, $\alpha = 0.5$ sweet spot) hits high $P(\text{QC})$
  (~0.92 across scenarios) but produces *predictions in latent space*; the reported recipe has
  to be back-decoded through the AE, which still costs target attainment on the secondary
  regression objectives. Without the AE-alignment term ($\alpha = 0$), the latent drifts off
  the manifold and QC collapses (post-decode 0.97 → 0.35 in PR #18 measurements) — the term is
  doing real work.
* **Composition path** (`optimize_composition`) optimises the element-weight simplex directly:
  $w = \text{softmax}(\theta) \to x = w \cdot K \to \text{heads}$. The optimised `w` *is the
  reported recipe* — no AE round-trip, no fidelity loss between "what the optimiser sees" and
  "what gets written down".
* On the headline `comp (seed, 5% all, element list)` configuration the composition path lands
  at QC ≈ 0.85 — slightly below latent's 0.92 — but the trade-off buys (a) outputs that are
  *valid alloy recipes by construction* (simplex + element whitelist), (b) chemistry-consistent
  outputs that cluster around real QC-prone families (Al-Pd-Pt, Mg-Pd-Al, Au-Ga-RE), and
  (c) a per-knob control surface that materials scientists can actually reason about
  (`allowed_elements`, `element_step_scale`, `seed_blend`).

## Other points worth keeping in the summary

### 3. The two methods are complementary, not competing.

* Latent finds the model's *internal* attractors — it answers "what region of representation
  space does the model think is QC-like, regardless of physical realisability". This is
  scientifically useful as a diagnostic (it surfaces model biases like the "Ti/Pu/F/Mn"
  attractor seen in `comp (random)`).
* Composition is the *recipe generator* — what you'd hand to a synthesist. It's the path
  reported as the "paper-headline" output, with the latent runs kept as the baseline /
  failure-mode control.
* Use them together: latent shows where the model thinks QC lives; composition shows what to
  actually make.

### 4. The model demonstrably learns chemistry beyond the seed set — "element discovery" is real.

* Seed set = 20 compositions (17 top-QC element-system-dedup from training + 3 explicit
  Au-Ga-RE i-QC formers). Crucially, **Pt, Pd, Re, Hf, Ta are *not* in any seed**.
* After running the constrained composition path with the 48-element `ALLOY_PALETTE`:
  * **Pt** is picked up in **6/20 outputs (scenario 1: FE↓ + mag↑)** and **7/20 outputs
    (scenario 3: FE↓ + klat↑)**, as part of an Al-Pd-Pt ternary that the model converges to
    repeatedly. Pt is not seeded — the optimiser introduced it via the `seed_blend = 0.95`
    mechanism (5 % uniform mass over the whitelist) and the gradient signal recognised it as
    QC-favourable.
  * **Pd** also appears in many scenario-1/3 outputs (not in any seed either) — the Mg-Pd-Al
    family was the headline finding in PR #18's smaller-palette run too.
  * **Hf, Ta** are picked up occasionally by latent in scenario 3.
* These are not random noise insertions: the same element families show up consistently across
  seeds and across scenarios with related objectives, which is consistent with the model having
  learned the underlying chemistry of QC-prone compositions, not memorised individual seeds.

### 5. The user-facing knobs are intuitive and on a $[0, 1]$ scale.

* `ae_align_scale` $\in [0, 1]$ (latent): 0 = no manifold constraint (fails: $h$ drifts off the
  AE-learned region); 1 = strict constraint (over-tight, hurts target attainment); 0.5 is the
  empirical sweet spot.
* `diversity_scale` $\in [0, 1]$ (composition): 1 = no entropy penalty (default, lets the
  optimiser pick as many elements as the objective wants); 0 = peaky few-element recipes
  (forces binary / ternary, useful as an ablation).
* `seed_blend` $\in [0, 1]$ (composition): 0.95 default = keep 95 % seed, mix 5 % uniform over
  the allowed elements at the start so the optimiser can actually *introduce new elements*
  (this is the element-discovery enabler).
* The point: no need to read the implementation to use these. The knob name predicts the
  direction.

### 6. The 3 scenarios stress-test conflicting objectives.

* Each scenario combines QC↑ (always primary) with 1–2 regression targets that the model has
  *no a-priori reason* to expect can co-exist with QC. FE↓ is the most aggressive ask (drives
  toward thermodynamically stable phases, often in tension with the metastable QC family);
  klat↑ is also non-obvious for amorphous-leaning compositions.
* The fact that the composition path lands at QC ≈ 0.85 *and* meets the secondary targets
  (scenario 3: FE close to 0, klat ≈ 1.6 / target 2.0) on average shows the model isn't simply
  collapsing to a single trivial "high-QC" point — it's negotiating the trade-off.
* The 8-path × 3-scenario × 20-seed sweep (480 optimisation runs total) gives enough data to
  read the trade-off as a Pareto-like front (the `qc_vs_secondary_scatter.png` figure).

### 7. The pipeline is end-to-end automated and reproducible.

* One run produces, for each scenario:
  * `comparison.png` — 3-panel bar chart with QC + each reg target across all 8 paths.
  * `element_frequency_heatmap.png` — 8 paths × top-25 elements; newly-discovered elements
    (not in any seed) are bold-orange on the x-axis.
  * `qc_vs_secondary_scatter.png` — per-seed cloud, latent = ○ Greens / composition = △ Blues,
    with red dashed target lines.
  * `seed_to_optimized__*.png` × 7 — per-method 1:1 mapping (seed → optimised composition) with
    per-row `(QC%, ΔFE, Δklat, …)` deltas.
  * `trajectories/<path_slug>.{png,gif,html}` per path — **mean-across-seeds** normalised
    per-step target curves (static `.png`), and the same curves animated alongside a per-step
    element bar chart of the best representative seed (default `.gif`, self-contained interactive
    `.html` on request). Raw per-step arrays `(steps, B, T)` for targets and `(steps, B, n_components)`
    for weights persisted as `trajectories/<path_slug>.npz` so re-plots don't need to rerun the
    optimisation.
  * `trajectories_per_seed/seed{NN}/<path_slug>.{png,gif,html}` — **per-(path × seed)** plots
    and animations under a seed-major layout (one folder per seed, all 8 paths inside). Each
    title carries the seed's composition formula in monospace. Default on; opt out with
    `--no-per-seed-trajectories`.
  * `results.json` + `SUMMARY.md` — raw arrays and a markdown table.
* Configs, seeds, and the trained checkpoint are all saved per run, so any figure can be
  regenerated from `results.json` alone (no re-running the optimisation needed for re-plots).
* `fm inverse` writes each scenario into a sibling subfolder so the full study is one directory.

### 8. Per-step optimisation trajectories explain why the same seed → different scenarios → different recipes.

Each path's per-step `(targets, weights)` are now persisted as
`<scenario>/trajectories/<path_slug>.npz`; the corresponding `trajectory__*.png` /
`.gif` / `.html` plots normalise each target to "progress" (0 = seed baseline,
1 = at target) and overlay all targets on one axis. The headline finding is
that **secondary targets converge on very different time-scales**, and the
fastest-converging one locks in the recipe early:

| Scenario | Path | Per-target trajectory (300 steps) | What it tells you |
|---|---|---|---|
| 3: FE↓ + klat↑ | `comp (seed, 5% all, element list)` | **klat overshoots** to progress ≈ 1.5 by step ~100 then plateaus; **FE crawls** to ~0.32 across all 300 steps | klat dominates the gradient early; once a klat-favourable recipe is locked, the remaining 200 steps only nudge FE in the residual subspace |
| 1: FE↓ + mag↑ | same path | **FE** crawls to ~0.26; **magnetisation** essentially flat at ~0.01 | the model can't find compositions that increase magnetisation without dropping QC — magnetisation is a *stuck* target on this manifold |
| 2: FE↓ + tc↑ + mag↑ | same path | **FE and tc** rise together to ~0.22 by step ~200 (coupled); **magnetisation** plateaus at ~0.08 | when two targets pull on similar element directions they couple cleanly; the orthogonal one (mag) again barely moves |

Three consequences for interpreting the per-scenario heatmaps:

* "Same seed, different scenario, different recipe" is not optimisation noise —
  it's the *dominant target* taking over the gradient in the first ~50–100 steps
  and steering the composition into a different chemistry basin. The trajectory
  plot lets you see this happening in real time (left-panel target curves +
  right-panel evolving element bars in the GIF / HTML).
* Most paths have flatlined by step ~150–200, so the configured `inverse_steps =
  300` is enough headroom; further steps would mainly refine the slow tail. The
  bottleneck is not training time, it's the magnetisation-style "model can't
  reach this target from any QC-prone basin" failure mode.
* The klat overshoot (progress > 1.0) is honest signal: the optimiser keeps
  pushing klat past the target because the joint loss is still falling on the
  other axes. Reading the `seed_to_optimized__*.png` per-row Δreg values gives
  the absolute (not relative) numbers if "did it actually overshoot" matters
  for the application.

The "best-per-target representative seed" used in the GIF / HTML's composition
panel is picked by `workflows.inverse_trajectory.best_seed_by_target_distance`
(minimises the joint normalised distance to QC = 1 and every reg target). To
see all 20 seeds individually instead of the mean, rerun with
`--per-seed-trajectories`.

### 9. Constraints and honest limitations.

* The 48-element `ALLOY_PALETTE` is a *chemistry-aware whitelist*, not a synthesisability
  predictor. The optimiser will still happily propose Al-Pd-Pt at a ratio nobody has yet
  reported as quasicrystalline — the model's confidence ≠ experimental confirmation.
* The single-task regression heads are trained on z-scored targets, so "FE = −2" means
  "2 σ below the dataset mean", not "−2 eV/atom" directly. The summary numbers are best read
  as *relative* improvements over the seed baseline (the per-seed `ΔFE` in
  `seed_to_optimized__*.png` is the cleanest view of that).
* The latent path's "α = 0 failure" baseline is *deliberately included* in the comparison
  figure so the AE-alignment term's contribution is visible — readers occasionally interpret
  the α=0 results as the method's overall performance; they're meant to be the *control*, not
  the recommendation.

## Where to go for detail

* **Method math + per-term design intent**: [docs/inverse_design_algorithms.md](inverse_design_algorithms.md)
* **Per-scenario outputs**: `artifacts/inverse/<scenario>/` (gitignored — regenerate with
  `fm inverse --config samples/inverse.toml --checkpoint <ckpt>`).
* **Implementation**: [`workflows/inverse.py`](../src/foundation_model/workflows/inverse.py)
  (scenario × path engine) + [`workflows/inverse_trajectory.py`](../src/foundation_model/workflows/inverse_trajectory.py).
