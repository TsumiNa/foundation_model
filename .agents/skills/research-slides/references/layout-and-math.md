# Layout and mathematical typography

Use this reference when creating, revising, or reviewing slide layouts. Match an
explicitly supplied template. Otherwise use a consistent canvas and a clear hierarchy
that gives the main content enough space. A 16:9 canvas is a useful default.

## Typography for projection

Use these starting sizes for a wide slide, measured after placement on the final slide:

| Element | Starting size |
| --- | --- |
| Slide title | 29–35 pt |
| Body text | 19–23 pt |
| Figure or panel title | 21–23 pt |
| Essential axis titles, ticks, legends, and annotations | 17–19 pt |
| Supporting captions | 15–17 pt |

Adjust to the template, venue, and viewing distance. Essential explanations belong at
body or figure-label size, not in a tiny caption. Font sizes in a plotting script are
not slide sizes: resizing an inserted figure scales its text too. Measure or inspect
the embedded figure at its final size, including the densest panel.

Check the slide at presentation scale without zooming. When possible, preview projected
or at the expected viewing distance. If labels are hard to read, enlarge or re-export
them and simplify or split the slide. Do not shrink essential labels to fit more panels.
High raster resolution removes pixelation but cannot make physically small text readable.

For an unspecified global reduction of one notch, use roughly 10–15% as a starting
adjustment to the established type scale, preserving its hierarchy. Follow an explicit
point size or supplied example instead when provided. Apply it to every slide, including
Appendix slides, tables, equations, captions, and chart titles, ticks, legends, and
annotations. Regenerate embedded figures with smaller text while retaining their
plot area; shrinking the entire figure also shrinks the data and is not equivalent.
Keep section order, content, and numerical values stable. Re-render and inspect the
result at presentation scale. An explicit user-requested reduction takes precedence
over the starting ranges; it is not permission to add more panels or denser content.
For a subsequent request for one point smaller, subtract 1 pt from the current
final-slide sizes rather than repeating the percentage reduction. Account for the
placement scale of embedded figures and equations when regenerating their text.

When the user requests a sample before a full rebuild, render only the requested
number of representative pages, including text, tables, equations, and dense figures
where present. Reuse verified data and assets; do not rerun analysis or rebuild the
whole deck just to assess typography. State which pages the preview covers.

Once the user approves a preview, keep its type hierarchy and relative sizes fixed
when completing the full deck. Do not restart from the default ranges or vary sizes
page by page merely to fill whitespace.

## Titles, lists, and spacing

- Use short titles that name the subject or a supported finding. Avoid universal claims
  based on a favorable subset or a different measure.
- Set body text flush left. Use actual bullet paragraphs with hanging indents and
  explicit paragraph spacing. Do not simulate lists with manual symbols, line breaks
  inside one paragraph, or many disconnected floating text boxes.
- Keep text-only slides when they serve a clear purpose. Short lists, a compact table,
  or a meaningful diagram are sufficient; decoration should not displace useful content.
- Reserve separate space for the slide title, explanatory text, and figure. A 16–24 pt
  gap between major regions is a useful starting point; increase it for wrapped text
  or large mathematical glyphs. Keep figure titles clear of captions and axes.
- Remove repeated prose and duplicate titles before resizing the main content. Retain
  both a slide title and figure title only when they supply different information.

## Large figures and multiple panels

Give figures enough room for readable axes, legends, and annotations. A headline, long
subtitle, figure title, and caption should not all compete above the same axes. Move
secondary explanations into notes, a short footer, or supporting slides.

Split dense panels by category or comparison. Four small heatmaps, for example, can
become two slides with two enlarged panels each. Carry the same comparison definition
and visual scale into both slides. Do not crop axes, omit inconvenient cases, or reduce
labels to meet an arbitrary panel count.

Across related figures:

- Preserve every category or condition requested, across multiple slides if needed.
- Use consistent colors, line styles, and legend meanings.
- Match scales for direct comparisons or explicitly label panel-specific limits.
- Explain logarithmic axes or transformations when they affect interpretation.
- Add a focused view when the full range hides an important feature; retain its units
  and identify the interval actually shown.

Use native charts when supported and appropriate; follow the base skill's editability
requirements. For inserted plots, prefer vector output when the renderer supports it.
Otherwise use sufficient raster resolution and retain the figure source and data.
Keep slide titles, explanatory text, and simple tables editable.

## Equations, symbols, and units

Typeset mathematical expressions using an equation renderer or a genuine math object.
Plain text `sqrt(sum(...)/N)` or an underscore pretending to be a subscript is insufficient.
Use native equations when reliably supported, or rendered LaTeX/MathML as a vector
figure. A raster fallback needs adequate resolution. Preserve the equation source in
notes or build sources when the equation itself cannot remain editable.

Use a mathematical typeface such as Cambria Math, STIX Two Math, or another available
math font. Apply mathematical typography to inline symbols, diagrams, plot labels,
legends, and units too. A font name in OOXML does not prove the export used that font.
Inspect rendered glyphs and PDF font information when available, and verify substitution
through the actual export path.

- Variables are italic where appropriate; units, descriptive subscripts, method names,
  and operators are upright. Use real subscripts and superscripts.
- Use an upright micro prefix. With LaTeX's [upgreek](https://ctan.org/pkg/upgreek)
  package, an example is \(\upmu\mathrm{m}\). In another renderer, use a supported upright
  glyph and verify the export. Separate compound units, as in
  \(\mathrm{m}\,\mathrm{s}^{-1}\), and keep units upright, as in \(t\,[\mathrm{s}]\).
- Align symbols with surrounding text. Avoid fallback glyphs, cramped fractions,
  and equation images with excessive margins.
- Define symbols and relative quantities near first use. An Appendix definition cannot
  substitute for information needed to understand a main-slide figure.

## Self-explanatory charts

Remove words repeated in every tick or category label when a single axis heading,
legend title, or brief explanation can state their shared meaning. For example, use
`1% / 10% / 100%` with one “Training size” heading instead of repeating “training”
three times. Retain the unit and enough context to interpret the values on that slide.

For availability matrices, categorical fills or dots often work better than repeated
`1` values. Explain available, unavailable, and highlighted categories. Blank does not
automatically mean zero. If counts matter, label what is counted and keep the counts.

For signed-effect heatmaps, define the difference, center the color scale at zero, and
state the favorable direction in words. Keep ranges consistent across compared panels
and disclose saturation. Use large cell labels; a count of favorable cells cannot
replace an aggregate magnitude summary.

For distribution plots, explain quartiles, mean markers, median lines, and whiskers once
in a shared caption or legend. Give that explanation visible separation from the figure
title without squeezing the plot or shrinking its labels.

Use the requested language and established terminology for the topic. Define necessary
abbreviations and keep labels consistent. Replace unfamiliar internal names with names
the audience recognizes. An accompanying message may use another requested language,
but its claims must match the deck.
