# Inverse-design algorithms — loss & design intent

Reference for the two inverse-design routines in
[`flexible_multi_task_model.py`](../src/foundation_model/models/flexible_multi_task_model.py):
`optimize_latent` (latent-space gradient descent) and `optimize_composition` (differentiable
KMD). Written as a one-stop sheet so the loss formulas, what each term is *for*, and the
user-facing knobs are all in one place — ready to drop into a slide deck or the paper.

## A. Latent-space optimisation (`optimize_latent`, `optimize_space="latent"`)

### Optimisation variable

**latent vector $h$** (the encoder output). One $h$ per seed; each runs independent gradient
descent.

### Loss

$$
\mathcal{L}_{\text{latent}}(h) \;=\; \underbrace{\sum_{i \in \text{targets}} w_i \cdot \text{term}_i(h)}_{\text{(1) objective terms}}
\;+\;\underbrace{\alpha \cdot \bigl\lVert h - \tanh\bigl(E(D(h))\bigr) \bigr\rVert^2}_{\text{(2) AE-alignment term}}
$$

Every target is user-specified (an `OptimizationTarget` / one `[[inverse.scenarios.targets]]`
entry) with its own weight $w_i > 0$; the per-kind terms are:

| Target kind | $\text{term}_i(h)$ | Design intent |
|---|---|---|
| regression, `value = v` | $\bigl(\hat y_t(h) - v\bigr)^2$ | Hit the value (MSE in the task's trained units, typically z-scored). |
| regression, `direction` | $\mp\,\hat y_t(h)$ ($-$ for `"high"`, $+$ for `"low"`) | Push the prediction up/down. **Unbounded** — no stationary point, so the achieved magnitude scales with `steps × lr`; balance with $w_i$. |
| kernel_regression, `points = {(t_k, y_k)}` | $\frac{1}{K}\sum_k \bigl(\hat y_t(h, t_k) - y_k\bigr)^2$ | Pull the predicted curve onto the target curve at the user's $t$ coordinates (the KR head evaluates at any $t$; outside the trained range it extrapolates). |
| classification, `classes = C`, `"high"` | $-\log \sum_{c \in C} P(c \mid h)$ | Push the combined probability of the chosen labels up. |
| classification, `classes = C`, `"low"` | $-\log \sum_{c \notin C} P(c \mid h)$ | Push it down — implemented as *maximising the complement*, which reuses the same logsumexp and stays numerically clean as $P(C) \to 1$. $C$ must be a strict subset of the labels. |

with $D(\cdot)$ = AE decoder (latent → input space $\hat x$) and $E(\cdot)$ = encoder (input →
latent, with the trailing tanh).

**(2) AE-alignment term** is the crux of this method: freely optimised $h$ tends to drift off
the AE-learned manifold → decoded $\hat x$ becomes unphysical → the reported composition can't be
trusted. This term pulls $h$ toward $\tanh(E(D(h)))$, i.e. the fixed-point of one decode→encode
round-trip. $\alpha = 0$ turns the term off (the pre-PR #18 failure mode: the class objective
dropped 0.97 → 0.35 after the round-trip); $\alpha = 1$ over-constrains ($h$ effectively locked
onto the AE manifold, target attainment drops); **empirical sweet spot $\approx 0.5$**.

### Main tunable parameters

| Parameter | Range | Default | Meaning |
|---|---|---|---|
| `ae_align_scale` (= $\alpha$) | $[0, 1]$ | 0.5 | AE-manifold alignment strength (see (2)). |
| per-target `weight` (= $w_i$) | $> 0$ | 1.0 | This target's priority relative to the others. |
| `steps`, `lr` | — | 200, 0.1 | Adam optimisation budget. |
| `num_restarts`, `perturbation_std` | — | 1, 0.0 | Independent restarts with Gaussian jitter on the seed. |

---

## B. Differentiable KMD composition optimisation (`optimize_composition`)

### Optimisation variable

**logits $\theta \in \mathbb{R}^n$**, with $n$ = element-table size (default 94, from KMD's
`DEFAULT_ELEMENTS`). The softmax gives the element-weight simplex `w = softmax(θ)` (each row
non-negative, sums to 1).

### Forward pass

$$
w = \text{softmax}(\theta) \;\to\; x = w \cdot K \;\to\; \tilde h = \tanh(E(x)) \;\to\; \text{heads}
$$

where $K \in \mathbb{R}^{n \times d_x}$ is the precomputed KMD kernel and $x$ is the
descriptor vector. **`w` itself is the recipe you would report** — no AE decode step.

### Loss

$$
\mathcal{L}_{\text{comp}}(\theta) \;=\; \underbrace{\sum_{i \in \text{targets}} w_i \cdot \text{term}_i(w)}_{\text{(1) objective terms}}
\;+\;\underbrace{(1 - d) \cdot H(w)}_{\text{(2) entropy / peakiness term}}
$$

with

- the per-target terms exactly as in section A (regression value/direction, kernel-regression
  curve, classification high/low), evaluated on the forward pass above;
- $H(w) = -\sum_i w_i \log w_i$ — the per-output-row Shannon entropy;
- $d$ = `diversity_scale` $\in [0, 1]$.

### Constraints (not in the loss, but enforced in the implementation)

| Constraint | How it's enforced | Design intent |
|---|---|---|
| **simplex** | `w = softmax(θ)` | Automatically keeps `w` a valid recipe (non-negative, sums to 1). |
| **`allowed_elements` whitelist** | Masks the logits of disallowed elements to $-\infty$ before every softmax step. | Restrict the search to physically realisable elements (e.g. `ALLOY_PALETTE`, 41 symbols), suppressing model biases toward Pu / F / Cs / etc. |
| **`element_step_scale` soft-freeze / hard-lock** | Soft: multiply the element's logit gradient by the scale before each Adam step. Hard (value = 0): rewrite the softmax output to paste seed values back at locked positions and renormalise unlocked positions over the remaining mass. | Let the user pin certain elements to their seed values ("keep the Au-Ga-RE skeleton; you may only change the rare-earth ratios"). |
| **`seed_blend` mixture** | $w_0 \leftarrow \text{seed\_blend} \cdot \text{seed} + (1 - \text{seed\_blend}) \cdot \text{uniform}_{\text{allowed}}$ | Don't start from a 100 % seed (5 % uniform mass lifts every allowed element's logit from $\log(10^{-12}) \approx -27.6$ to $\log(0.05 / \lvert\text{allowed}\rvert) \approx -7.6$, so Adam can introduce new elements within a few hundred steps — this is the **element-discovery** mechanism). |
| **`max_elements` cardinality cap** | Plötz–Roth iterative-softmax K-hot mask $m \in [0, 1]^n$ with $\sum_i m_i = K$, multiplied with `softmax(θ)` and renormalised; temperature $\tau$ annealed from $\tau_\text{start} = 25^{\text{annealing\_scale}}$ down to $\tau_\text{end} = 0.01$ (geometric by default). A hard top-K projection at the end guarantees exactly K-hot (subject to floor below). | Restrict recipes to **at most K non-zero elements** (e.g. "I want a 3-element alloy"). The annealing doubles as a continuation method — the soft τ early on lets the optimiser explore different K-subsets before committing. |
| **`fixed_amounts` user-pin** | Build $\text{locked\_w}_0$ with user-specified values at the named positions, zero elsewhere; reuse the existing lock-paste machinery (no `initial_weights` required since values are given directly). | Pin specific elements at user-given absolute amounts (e.g. `{"Au": 0.65, "Ga": 0.20}` — the optimiser distributes the remaining 0.15 mass across other allowed elements). |
| **`min_nonzero_weight` floor** | After lock-paste, zero unlocked positions with $0 < w < \text{floor}$ and renormalise the unlocked portion to fit the free mass; safe-fallback when dropping would empty a row (leave that row unfloored). | Reject trace-amount appearances (e.g. `Pt = 0.5 %`) that are not synthesisable — "if you use it, use ≥ 10 %". |

### What each loss term is for

| Term | Design intent |
|---|---|
| **(1) objective terms** | Same as latent — the user-specified targets (section A's per-kind table). |
| **(2) entropy / peakiness term** | **The crux of this method.** Larger $H(w)$ ⇒ flatter `w` ⇒ each solution uses more elements; smaller $H(w)$ ⇒ peakier `w` ⇒ a few elements dominate each solution. $(1 - d)$ is the penalty weight: $d = 1$ turns it off (default — the optimiser uses as many elements as the main objective wants); $d = 0$ is the strongest penalty (forced peaky → binary/ternary recipes, useful as an ablation). **Important**: this is a *per-output-complexity* knob, **not** a between-output diversity knob. Whether the $B$ outputs differ from each other is decided by the loss landscape, not by $d$. |

### Main tunable parameters

| Parameter | Range | Default | Meaning |
|---|---|---|---|
| `diversity_scale` (= $d$) | $[0, 1]$ | 1.0 | Per-output element diversity (see (2)). |
| per-target `weight` (= $w_i$) | $> 0$ | 1.0 | This target's priority relative to the others. |
| `seed_blend` | $[0, 1]$ | 0.95 | Fraction of seed kept at the start (the rest is uniform, so new elements can enter). |
| `allowed_elements` | symbol list or `"all"` | `"all"` | Element whitelist (hard constraint). |
| `element_step_scale` | float or `{symbol: float}` | 1.0 | Per-element step scaling; `0` = hard-lock to the seed value. |
| `max_elements` | `int` ∈ $[1, n]$ or `None` | `None` | Cardinality cap — at most K non-zero elements (differentiable soft top-K + final hard projection). |
| `annealing_scale` | $[0, 1]$ | 0.5 | Single-knob softness for the K-hot schedule; maps to $\tau_\text{start} = 25^{\text{scale}}$. |
| `annealing_schedule` | dict or `None` | `None` | Advanced piecewise override of the annealing schedule. |
| `fixed_amounts` | `{symbol: float}` or `None` | `None` | Pin elements at user-specified amounts (e.g. `{"Au": 0.65}`); needs $\sum < 1$. |
| `min_nonzero_weight` | $[0, 1]$ | 0.0 | Drop unlocked positions below this floor (and re-distribute mass). |
| `steps`, `lr` | — | 300, 0.05 | Adam optimisation budget over the logits. |

---

## Side-by-side summary

| | Latent | Composition |
|---|---|---|
| **Optimisation variable** | $h$ (latent vector) | $\theta$, with $w = \text{softmax}(\theta)$ (element-weight simplex) |
| **Where the reported recipe comes from** | $w_{\text{report}}$ inferred from $D(h)$ (an extra AE-decode step) | $w$ itself is the report |
| **Method-specific loss term** | $\alpha \cdot \lVert h - \tanh(E(D(h))) \rVert^2$ (keeps $h$ on the AE manifold) | $(1 - d) \cdot H(w)$ (controls per-solution peakiness) |
| **Failure mode** | $\alpha = 0$: $h$ drifts off the manifold, decoded recipe unphysical (QC 0.97 → 0.35). | `seed_blend = 1.0`: the seed's support set is frozen — no new elements can ever appear. |
| **Method-specific knobs** | `ae_align_scale` | `diversity_scale`, `seed_blend`, `allowed_elements`, `element_step_scale`, `max_elements` + `annealing_scale` / `annealing_schedule`, `fixed_amounts`, `min_nonzero_weight` |

The shared backbone — the user-specified objective terms (regression value/direction,
kernel-regression curves, classification high/low, each with its own weight) — is **identical**
between the two methods. They differ *only* in the method-specific loss term and in which
variable is being optimised.
