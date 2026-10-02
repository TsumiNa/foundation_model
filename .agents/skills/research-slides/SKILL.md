---
name: research-slides
description: Create, revise, or review presentations with audience-appropriate structure, readable projected figures, mathematical typography, explicit comparisons, and evidence-qualified conclusions. Use alongside an available presentation or PPTX skill for new decks, slide revisions, and review comments.
---

# Research slides

Build presentations that an audience can understand without the preceding conversation.
Use this skill across topics; derive the narrative from the subject, purpose, and
audience rather than imposing a particular experiment or workflow. The user's template,
language, scope, and requested slide count take precedence over defaults here.

## Build on an available presentation skill

1. Discover the presentation skills exposed by the current agent or installed locally.
   Typical names include `presentations`, `Presentations`, `pptx`, and `pptx-official`.
   Select the skill appropriate to the requested format and operation, then read its
   entry point and the implementation references it requires before building or editing.
2. Use that skill for file construction, template preservation, editable objects,
   package validation, and rendering. This skill adds audience, layout, and interpretation
   requirements. Prefer these readability rules over decorative defaults such as adding
   an illustration to every slide. Resolve actual implementation conflicts using the
   current agent's instruction hierarchy and the user's request.
3. Discover paths and tools in the current environment. Do not assume a particular agent,
   library, bundled runtime, or another machine's home directory. For read-only questions,
   inspect the deck and answer within scope; regeneration is unnecessary.
4. If no suitable base skill is installed, use available presentation tools with the
   requirements below. If export or rendering is unavailable, preserve the editable
   source and identify what remains unverified.

Do not copy or vendor third-party skills, install unrelated tools, or publish a deck
merely to satisfy this workflow. Editorial work does not authorize new experiments,
changes to source data, or external messaging.

## Establish the purpose and source material

- Determine the audience's prior knowledge, the presentation's main question, and what
  the audience should understand or decide afterward. Inspect supplied slides, comments,
  documents, figures, and source tables as needed.
- Use recognizable names and define necessary abbreviations. Verify the meaning of
  quantities, units, categories, and source coverage before describing them.
- For quantitative comparisons, identify the reference, aggregation, and meaning of
  displayed spread. Read [result interpretation](references/result-interpretation.md)
  when using relative values, grouped summaries, heatmaps, error bars, or uncertainty.
- Keep verified facts and numerical results stable during editorial revisions. A change
  to the underlying analysis is a separate change in scope that needs validation.

## Organize around the subject and audience

Choose sections from the actual material. Do not prescribe chapters for a particular
task, method, or domain. A useful full-deck progression is:

- **Necessary context:** introduce the question, terminology, or background needed to
  follow the argument. Omit detail the audience already knows or does not need.
- **Main explanation or evidence:** lead with the central comparison or idea, then show
  supporting detail in the order that makes it easiest to assess. Use diagrams, examples,
  tables, or figures according to the content.
- **Conclusions or implications:** state the strongest supported message and the limits
  that affect its use. A concise closing slide is often sufficient.
- **Supporting material:** use an Appendix for detail that is useful for questions but
  interrupts the main narrative. Include it only when needed.

Adapt this progression for a short update, tutorial, proposal, or narrow review. Avoid
standalone implementation slides unless they help answer the audience's question. Give
definitions near first use without turning them into unnecessary tutorials.

Use established terminology in the relevant field. Avoid invented labels, unnecessary
acronyms, and titles that claim more than the content supports. Match the language to
the user's request and audience.

## Make figures readable during a presentation

Read [layout and mathematical typography](references/layout-and-math.md) before creating,
revising, or reviewing slide layouts, including review-only requests. Apply its guidance
throughout the requested review scope, not just to annotated pages.

- Give each slide one purpose and a clear visual hierarchy. Use native bullet lists
  with hanging indents and paragraph spacing for parallel points on text slides.
- Allocate enough space to the main figure. Keep slide titles, figure titles,
  explanations, legends, and axes visibly separated. Remove duplicate titles or shorten
  prose before reducing the figure's size.
- Use large figure text as a default: figure titles about 24–28 pt and essential axis
  titles, ticks, legends, and annotations about 20–24 pt on the final slide. Check size
  after placement; a large source font can become tiny when its figure is scaled down.
- Split crowded panels across slides before shrinking figures or essential labels.
  Preserve comparable scales, colors, panel identities, and all requested content.
- Typeset equations mathematically. Use appropriate math typography for symbols and
  units throughout text, diagrams, legends, and axes.
- Make chart encodings self-explanatory. Use availability symbols for presence/absence;
  show counts only when their meaning is explicit. Distinguish missing values from zero.

## Keep evidence and claims consistent

Every quantitative comparison must identify its quantity, reference, population or
sample unit, and direction of improvement where relevant. Define relative values before
presenting them and carry the same definitions into captions and conclusions.

When a detailed figure and an overall summary appear to disagree, check whether they
measure frequency, magnitude, or different populations. Explain the difference. Qualify
claims such as "best" with the measure and conditions that support them.

Distinguish variation across cases from uncertainty in an estimate. Describe observed
features and possible explanations separately. Use appropriate evidence for causal
claims; do not present an untested explanation as an established result.

## Verify revisions and deliver

Read [quality assurance](references/quality-assurance.md) before finalizing a new or
revised deck or performing a deck review. For review-only work, apply relevant checks to
the supplied deck or renders within the requested scope; new exports are not required.
For created or revised deliverables, render and inspect final outputs, verify central
claims against their sources, and map review comments to the revised slide numbering.
Keep build checks and temporary files outside the presentation itself.

Deliver the requested editable deck and its PDF companion when useful or requested,
with matching order, content, and revision. Preserve figure sources and provenance in
the project's existing layout. Use date suffixes when that is the artifact convention
and retain an identifiable pre-review version. State material limitations accurately.

If the user requests accompanying Slack or email text, draft it in the requested language
with the same qualified claims and audience-friendly names. Sending it requires separate
authorization. For shared installation and maintenance, see [INSTALL.md](INSTALL.md).
