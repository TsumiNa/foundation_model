---
name: research-slides
description: Create, revise, or review scientific and machine-learning experiment slides with clear comparison baselines, mathematical typography, readable figures, and evidence-qualified conclusions. Use alongside an available presentation or PPTX skill for research reports, result decks, and slide review comments.
---

# Research slides

Turn experiment results into a presentation that a researcher can understand without
the preceding conversation. Apply these preferences to scientific and machine-learning
reports. The user's current audience, template, language, scope, and requested slide
count take precedence over defaults here.

## Use an existing presentation skill as the implementation foundation

1. Discover the presentation skills exposed by the current agent or installed locally.
   Typical names include `presentations`, `Presentations`, `pptx`, and `pptx-official`.
   Select the skill appropriate to the requested format and operation, then read its
   entry point and the implementation references it requires before building or editing.
2. Use that skill for file construction, template preservation, editable objects,
   package validation, and rendering. This skill adds research-report requirements.
   Prefer its scientific-content and readability rules over generic decorative defaults
   such as adding an illustration to every slide. Resolve actual implementation conflicts
   using the current agent's instruction hierarchy and the user's request.
3. Use paths and tools discovered in the current environment. Do not assume a particular
   agent, MCP server, library, bundled runtime, or another machine's home directory.
   Read-only deck questions require inspection and an answer, not regeneration.
4. If no suitable base skill is installed, use available presentation tools with the
   requirements below. Record the fallback in private build notes. If export or rendering
   is unavailable, preserve the editable source and state the unverified part accurately.

Do not copy or vendor third-party skills, install unrelated tools, or publish a deck
merely to satisfy this workflow. A slide-production request does not authorize new model
fits, data preprocessing changes, or external messaging.

## Establish the evidence before laying out slides

- Inspect the supplied deck, annotations, result tables, and analysis code as needed.
  Identify the audience's prior knowledge and define unfamiliar names and abbreviations.
  Prefer the collaborator's or dataset's recognizable name when an internal acronym
  would obscure the subject.
- Verify the prediction target, units, available labels, measurement conditions,
  independent experimental units, and train/test split. Distinguish a physical phenomenon
  from a measured numeric target. Check derived labels and labels from other datasets
  before describing what the current measurements contain.
- Determine which metric supports the main comparison and how it is aggregated. Specify
  comparison references, aggregation order, and the source of every displayed spread.
  Read [result interpretation](references/result-interpretation.md) when a deck compares
  models, uses normalized errors, plots heatmaps, or reports uncertainty.
- Keep verified numerical results stable during editorial revisions. Treat changes to
  metrics, training, or targets as analysis changes requiring explicit scope and validation.

## Organize an experiment report

Use the following structure when the user requests a complete results report. Adapt it
for a short update or a narrower question instead of forcing four sections into every deck.

1. **Dataset overview.** Show what was measured, coverage of independent samples and
   conditions, and representative curves or distributions. Dense measurement records
   are not additional independent samples. A raw-versus-smoothed example may appear
   here when preprocessing matters.
2. **Training and prediction.** Explain the experimental design briefly, summarize the
   model architecture and training settings, and lead with an overall baseline comparison
   with a clearly defined measure of dispersion. Then show condition-specific results
   and the requested individual predictions. Preserve informative matched-pair plots.
   Discuss difficult scientific regimes, such as low temperature, where supported.
3. **One-slide conclusions.** State the best-supported overall result with its metric
   and population. Identify supported benefits of the proposed method, localized effects,
   and limitations that change the interpretation. A request to find benefits does not
   justify overstating evidence.
4. **Appendix.** Put sensitivity analyses, preprocessing comparisons, and implementation
   detail here when they are not needed for the main scientific argument. For a report
   whose main experiment uses raw data, keep smoothed-model comparisons in this section
   and label raw and smoothed arms explicitly.

Omit standalone slides about normalization mechanics or sampling density unless the
user asks for them or they are essential to the question. Define a normalized metric
briefly where it is first used; this does not require a preprocessing tutorial.

Use established domain and machine-learning terminology. For training-set-size results,
the preferred section title is **Scaling law analysis**. Describe the plots as empirical
learning curves when no power law was fitted, and state that distinction once. Introduce
FT and CPT before using abbreviations. Avoid invented method names and unqualified
titles such as "Transfer gains" when the evidence contains both gains and losses.

## Lay out slides for reading

Read [layout and mathematical typography](references/layout-and-math.md) before creating
or revising slide layouts. Apply its guidance across the deck, not just to annotated pages.

- Give each slide one purpose. Use short, informative titles and a consistent visual
  hierarchy. Text slides are acceptable when they explain a protocol or conclusion.
  Use native bullet lists with hanging indents and paragraph spacing for parallel points.
- Allocate most of a result slide to the evidence. Separate slide titles, figure titles,
  explanations, legends, and axes with visible whitespace. Remove duplicate titles or
  shorten prose before reducing the figure's size.
- Split crowded multi-panel figures across slides while preserving axes, scales, color
  ranges, panel labels, and method colors. Retain all requested subjects and conditions.
- Typeset every equation mathematically. Use appropriate math fonts for symbols and
  units throughout slide text, diagrams, legends, and axes, including chemical subscripts.
- Make coverage plots self-explanatory. Use an availability symbol or categorical fill
  when the intended message is presence/absence. Display repeated `1` values only when
  file counts are the actual quantity, with that meaning stated explicitly.

## Align figures and conclusions

Every comparison figure must identify the metric, reference, sample unit, and direction
of improvement. Define relative values before presenting them. Keep the same definitions
between figures, tables, body text, and the conclusion.

When a heatmap and an overall summary appear to disagree, check whether one describes
the number of improving tasks and the other the magnitude of errors. Show the relevant
mean, median, or condition-specific result instead of declaring a universal winner.
Use **Transfer effects relative to training from scratch** for mixed-sign comparisons.

Report between-task variation separately from variation across model initializations.
Do not call checkpoint dispersion a calibrated prediction interval. Describe artifacts
and possible explanations separately, and use matched interventions to support causal
claims about preprocessing.

## Verify revisions and deliver

Read [quality assurance](references/quality-assurance.md) before finalizing a new or
revised deck. Render and inspect the final outputs, reconcile the central claims with
the underlying tables, and check every review comment against the new slide numbering.
Keep build checks and temporary files outside the presentation itself.

Deliver the requested editable deck and its PDF companion when useful or requested,
with the same slide order, content, and revision. Keep scientific figure sources and
result provenance in the experiment's existing layout. Use date suffixes when that is
the project's artifact convention, and retain an identifiable pre-review version.
Mention limitations that affect use, without a long production log.

If the user requests accompanying Slack or email text, draft it in the requested
language with the same qualified claims and audience-friendly names. Sending it requires
the user's separate authorization. For installation and shared maintenance of this
skill, see [INSTALL.md](INSTALL.md).
