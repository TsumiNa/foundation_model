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
\mathcal{L}_{\text{latent}}(h) \;=\; \underbrace{\sum_{t \in \mathcal{T}_{\text{reg}}} \lambda_t \,\bigl\lVert \hat y_t(h) - \text{target}_t \bigr\rVert^2}_{\text{(1) regression term}}
\;+\;\underbrace{w_{\text{cls}} \cdot \bigl(-\log P\bigl(c = \text{QC} \mid h\bigr)\bigr)}_{\text{(2) classification term}}
\;+\;\underbrace{\alpha \cdot \bigl\lVert h - \tanh\bigl(E(D(h))\bigr) \bigr\rVert^2}_{\text{(3) AE-alignment term}}
$$

with

- $\hat y_t(h)$ = prediction of the $t$-th regression head on $h$;
- $P(c = \text{QC} \mid h)$ = softmax probability of the quasicrystal class out of the QC
  classification head on $h$;
- $D(\cdot)$ = AE decoder (latent → input space $\hat x$); $E(\cdot)$ = encoder (input →
  latent, with the trailing tanh);
- $\lambda_t$ = the regression task's internal weight (a scalar fixed at training time).

### What each term is for

| Term | Design intent |
|---|---|
| **(1) regression term** | Push the latent to a place where every regression head hits its `target_t` (MSE in z-scored space). |
| **(2) classification term** | Push the latent to the region where the QC head emits high $P(c = \text{QC})$. $-\log P$ is the cross-entropy against the target class. `w_cls` sets classification priority relative to regression (use $> 1$ when QC is the primary objective and the regression targets are secondary). |
| **(3) AE-alignment term** | **The crux of this method.** Freely optimised $h$ tends to drift off the AE-learned manifold → decoded $\hat x$ becomes unphysical → the reported composition can't be trusted. This term pulls $h$ toward $\tanh(E(D(h)))$, i.e. the fixed-point of one decode→encode round-trip. $\alpha = 0$ turns the term off (the pre-PR #18 failure mode: QC dropped 0.97 → 0.35); $\alpha = 1$ over-constrains ($h$ effectively locked onto the AE manifold, target attainment drops); **empirical sweet spot $\approx 0.5$**. |

### Main tunable parameters

| Parameter | Range | Default | Meaning |
|---|---|---|---|
| `ae_align_scale` (= $\alpha$) | $[0, 1]$ | 0.5 | AE-manifold alignment strength (see (3)). |
| `class_target_weight` (= $w_{\text{cls}}$) | $> 0$ | 1.0 | Classification weight relative to regression. |
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
\mathcal{L}_{\text{comp}}(\theta) \;=\; \underbrace{\sum_{t \in \mathcal{T}_{\text{reg}}} \lambda_t \,\bigl\lVert \hat y_t(w) - \text{target}_t \bigr\rVert^2}_{\text{(1) regression term}}
\;+\;\underbrace{w_{\text{cls}} \cdot \bigl(-\log P\bigl(c = \text{QC} \mid w\bigr)\bigr)}_{\text{(2) classification term}}
\;+\;\underbrace{(1 - d) \cdot H(w)}_{\text{(3) entropy / peakiness term}}
$$

with

- $H(w) = -\sum_i w_i \log w_i$ — the per-output-row Shannon entropy;
- $\hat y_t(w)$ and $P(c = \text{QC} \mid w)$ both come from the forward pass above;
- $d$ = `diversity_scale` $\in [0, 1]$.

### Constraints (not in the loss, but enforced in the implementation)

| Constraint | How it's enforced | Design intent |
|---|---|---|
| **simplex** | `w = softmax(θ)` | Automatically keeps `w` a valid recipe (non-negative, sums to 1). |
| **`allowed_elements` whitelist** | Masks the logits of disallowed elements to $-\infty$ before every softmax step. | Restrict the search to physically realisable elements (e.g. `ALLOY_PALETTE`, 41 symbols), suppressing model biases toward Pu / F / Cs / etc. |
| **`element_step_scale` soft-freeze / hard-lock** | Soft: multiply the element's logit gradient by the scale before each Adam step. Hard (value = 0): rewrite the softmax output to paste seed values back at locked positions and renormalise unlocked positions over the remaining mass. | Let the user pin certain elements to their seed values ("keep the Au-Ga-RE skeleton; you may only change the rare-earth ratios"). |
| **`seed_blend` mixture** | $w_0 \leftarrow \text{seed\_blend} \cdot \text{seed} + (1 - \text{seed\_blend}) \cdot \text{uniform}_{\text{allowed}}$ | Don't start from a 100 % seed (5 % uniform mass lifts every allowed element's logit from $\log(10^{-12}) \approx -27.6$ to $\log(0.05 / \lvert\text{allowed}\rvert) \approx -7.6$, so Adam can introduce new elements within a few hundred steps — this is the **element-discovery** mechanism). |

### What each loss term is for

| Term | Design intent |
|---|---|
| **(1) regression term** | Same as latent — push predictions toward `target_t` via MSE. |
| **(2) classification term** | Same as latent — maximise $P(c = \text{QC})$. |
| **(3) entropy / peakiness term** | **The crux of this method.** Larger $H(w)$ ⇒ flatter `w` ⇒ each solution uses more elements; smaller $H(w)$ ⇒ peakier `w` ⇒ a few elements dominate each solution. $(1 - d)$ is the penalty weight: $d = 1$ turns it off (default — the optimiser uses as many elements as the main objective wants); $d = 0$ is the strongest penalty (forced peaky → binary/ternary recipes, useful as an ablation). **Important**: this is a *per-output-complexity* knob, **not** a between-output diversity knob. Whether the $B$ outputs differ from each other is decided by the loss landscape, not by $d$. |

### Main tunable parameters

| Parameter | Range | Default | Meaning |
|---|---|---|---|
| `diversity_scale` (= $d$) | $[0, 1]$ | 1.0 | Per-output element diversity (see (3)). |
| `class_target_weight` (= $w_{\text{cls}}$) | $> 0$ | 1.0 | Classification weight relative to regression. |
| `seed_blend` | $[0, 1]$ | 0.95 | Fraction of seed kept at the start (the rest is uniform, so new elements can enter). |
| `allowed_elements` | symbol list or `"all"` | `"all"` | Element whitelist (hard constraint). |
| `element_step_scale` | float or `{symbol: float}` | 1.0 | Per-element step scaling; `0` = hard-lock to the seed value. |
| `steps`, `lr` | — | 300, 0.05 | Adam optimisation budget over the logits. |

---

## Side-by-side summary

| | Latent | Composition |
|---|---|---|
| **Optimisation variable** | $h$ (latent vector) | $\theta$, with $w = \text{softmax}(\theta)$ (element-weight simplex) |
| **Where the reported recipe comes from** | $w_{\text{report}}$ inferred from $D(h)$ (an extra AE-decode step) | $w$ itself is the report |
| **Method-specific loss term** | $\alpha \cdot \lVert h - \tanh(E(D(h))) \rVert^2$ (keeps $h$ on the AE manifold) | $(1 - d) \cdot H(w)$ (controls per-solution peakiness) |
| **Failure mode** | $\alpha = 0$: $h$ drifts off the manifold, decoded recipe unphysical (QC 0.97 → 0.35). | `seed_blend = 1.0`: the seed's support set is frozen — no new elements can ever appear. |
| **Method-specific knobs** | `ae_align_scale` | `diversity_scale`, `seed_blend`, `allowed_elements`, `element_step_scale` |

The shared backbone — (1) regression MSE + (2) classification cross-entropy — is **identical**
between the two methods. They differ *only* in the third loss term and in which variable is
being optimised.
