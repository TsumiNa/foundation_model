# Research-slide quality assurance

Use the selected base presentation skill's file and rendering checks alongside the
checks below. Keep build notes and numerical reconciliation outside audience-facing
slides. Validation proves what was actually checked, not that every application will
render a file identically.

## Before final export

1. **Content coverage.** Compare the outline with the requested datasets, methods,
   subjects, conditions, and review comments. Check definitions of target and units.
   Preserve important paired-error plots. Keep raw and smoothed results identifiable.
2. **Numerical reconciliation.** Recompute the displayed central summaries from the
   saved metric tables using the documented aggregation order. Verify representative
   heatmap cells and win counts, primary means, cited medians, subgroup results, and
   all relative improvements that appear as claims. Use unrounded values for calculation.
3. **Claim consistency.** Read each result title, its figure, caption, and conclusion
   together. Check that they share the reference, metric, population, and sign convention.
   Qualify "best" and "improves". Explain local-versus-global or mean-versus-median
   differences when they could otherwise appear contradictory.
4. **Scientific limitations.** Check independent sample counts, additional candidate
   data or compute, the interpretation of spread, and whether a claimed causal effect
   was actually tested. Do not label absence of a visible benefit as statistical equivalence.

## Visual inspection

Render every final slide. A thumbnail contact sheet helps find inconsistent patterns,
but inspect every slide at its intended reading size; inspect dense figures and equations
at full resolution too. A geometry check alone cannot detect unreadable labels or bad
font substitution.

Check:

- Titles and figure explanations have visible separation from the plots.
- Essential labels, cell values, legends, and equations are readable without zooming.
- Text does not overflow or overlap, and bullets have consistent spacing and hanging indents.
- Mathematical glyphs, subscripts, micro symbols, and units render correctly.
- Split panels keep consistent colors, scales, and reference definitions.
- Coverage symbols and all error bars are understandable from their legends.
- No template placeholders, clipped axes, stale captions, or unintended blank pages remain.

Fix defects in the source and render the affected slides again. Complete the final
file checks after the last edit. If all slides were inspected and only a small set
changed, inspect that changed set again and check ordering, numbering, and file integrity
across the full deck; do not repeat unrelated numerical work without a new concern.

## PDF/PPTX parity and review comments

If both formats are delivered, compare their slide count, order, titles, metrics, and
revision. Prefer exporting the PDF from the final PPTX. If the build requires separate
exports, explicitly check corresponding content and representative renders. Do not
deliver a fresh PDF paired with an older PPTX.

For annotated reviews, keep a compact mapping from original slide/page and comment to
the implemented change and new slide/page. Distinguish a figure's data from its layout
when deciding whether a numerical recheck is needed. When a comment says "all slides,"
apply the correction throughout the deck. Preserve an identifiable pre-review version.

## Behavioral checks for this skill

Use these small scenarios when evaluating changes to the instructions. Do not retrain
models or produce a whole demonstration deck merely to run them.

| Scenario | Expected decision |
| --- | --- |
| Three of four task effects are positive, but one large degradation makes their mean negative | Show local benefits and the worse aggregate error; do not declare the candidate globally superior |
| A coverage matrix contains one file per measured condition | Use availability encoding or explicitly label file counts; explain selected conditions |
| Four heatmaps have illegible labels | Split panels across slides with shared scales instead of shrinking labels |
| A box plot includes mean, median, IQR, and whiskers | Define each encoding in a readable caption separated from the plot title |
| A conclusion cites a percent improvement with no denominator | Define the reference and aggregation before presenting that percentage |
| Prediction bands come from ten pretrained checkpoints | Describe checkpoint dispersion, not a calibrated confidence or prediction interval |
| A scaling-law section contains only seven training sizes and no fitted law | Use the preferred section title but explain that the evidence is empirical learning curves |
| The audience has not heard of an internal dataset acronym | Use an understandable dataset name and introduce only necessary abbreviations |
| The user asks why two slides differ | Inspect and answer; do not change the deck without an editing request |
| A new machine has no native presentation skill | Use available tools, preserve the same content rules, and state any unverified export or render |

## Delivery status

Report only material unresolved issues. Do not claim PowerPoint was inspected unless
it was opened there. Do not claim all-page visual checks, numerical validation, or
cross-format parity if they were not completed. Return the requested final artifacts;
temporary renders, private checks, and installation metadata are not presentation content.
