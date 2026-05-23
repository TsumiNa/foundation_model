# QC inverse-design study — summary

One-page summary of the messages the
[continual-rehearsal + inverse-design pipeline](continual_rehearsal_full_PLAN.md) carries.
Written so each bullet maps to either a slide or a paragraph of the paper.

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
  * `results.json` + `SUMMARY.md` — raw arrays and a markdown table.
* Configs, seeds, and the trained checkpoint are all saved per run, so any figure can be
  regenerated from `results.json` alone (no re-running the optimisation needed for re-plots).
* The orchestrator (`paper_inverse_3scenarios`) writes the three scenarios into sibling
  subfolders so the full study is one directory.

### 8. Constraints and honest limitations.

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
* **Plan and rationale for the 3 scenarios + alloy palette**: [docs/continual_rehearsal_full_PLAN.md](continual_rehearsal_full_PLAN.md)
* **Per-scenario outputs**: `artifacts/inverse_design_run/inverse_design/scenario{1,2,3}_*/`
  (gitignored — regenerate with `paper_inverse_3scenarios`).
* **Implementation**: [`paper_inverse_comparison.py`](../src/foundation_model/scripts/paper_inverse_comparison.py) (single-scenario runner) → [`paper_inverse_3scenarios.py`](../src/foundation_model/scripts/paper_inverse_3scenarios.py) (orchestrator).
