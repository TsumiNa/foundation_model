# Layout and mathematical typography

Use this reference when laying out or reviewing slides. Match an explicitly supplied
template. Otherwise use a clean research presentation with a consistent canvas and
strong evidence-to-text ratio. A 16:9 canvas is a useful default, not a requirement.

## Text, titles, and spacing

- Establish a small hierarchy of title, body, plot-label, and caption sizes. On a wide
  slide, practical starting points are 30–36 pt titles, 18–24 pt body, and 14–18 pt
  essential plot labels. Treat these as starting points; verify them at final slide size.
  A label legible only after zooming is too small for a presentation.
- Prefer concise noun phrases for dataset, methods, and architecture slides. Use a
  finding as a result title only when the figure supports that claim. Avoid universal
  claims about superiority based on a favorable subset or a different metric.
- Set body text flush left. Use actual bullet paragraphs with hanging indents and
  explicit paragraph spacing for lists. Do not simulate lists with manual symbols,
  line breaks inside one paragraph, or many disconnected floating text boxes.
- Keep text-only slides when they serve a clear scientific purpose. Short lists,
  a compact comparison table, or a simple meaningful diagram are sufficient. Decorative
  icons and stock images should not displace measurements or force smaller plots.
- Reserve separate space for the slide title, a short explanatory caption, and the
  figure. A 12–18 pt gap between major text/figure regions is a useful initial value.
  Increase it when wrapped text or large mathematical glyphs require more room.
- Shorten or remove repeated prose and duplicate titles before resizing evidence.
  Keep a slide title and a figure's title only when they supply different information.

## Large and multi-panel scientific figures

Give the figure enough room for axes, legends, and annotations at reading size. A
headline, a long subtitle, a plot suptitle, and a caption should not all compete above
the same axes. Move secondary explanation into a short footer, notes, or the Appendix.

For dense heatmaps or prediction panels, split by method, condition, or subject group.
For example, four heatmaps can become two slides with two enlarged heatmaps each.
Carry the same comparison definition and visual scale into both slides. Do not crop
axes, omit difficult compounds, or make labels smaller to meet an arbitrary panel count.

For all-subject prediction comparisons:

- Preserve every subject and condition the user requested, across multiple slides if needed.
- Use the same method colors, line styles, legend meaning, and temperature domain.
- Match scales for direct comparisons, or label panel-specific limits explicitly.
- Explain log or symlog axes and transformations when they affect visual interpretation.
- Show a focused low-temperature view when it resolves a scientific issue that a
  full-range plot hides. Use its actual measured range and maintain units.

Use native charts when supported by the chosen implementation and appropriate to the
evidence. Follow the base skill's editability requirements. For scientific plots that
must be inserted as figures, prefer vector output when the renderer supports it;
otherwise provide sufficient raster resolution and retain the plotting source and data.
Keep slide titles, explanatory text, and simple tables editable.

## Equations, symbols, and units

Typeset all mathematical expressions using an equation renderer or a genuine math
object. A plain text `sqrt(sum(...)/N)`, an underscore pretending to be a subscript,
or a slash pretending to be a stacked fraction is not mathematical typesetting.

Suitable methods include native Office equations when reliably supported, or rendered
LaTeX/MathML placed as a vector figure. Use a high-resolution raster fallback only when
needed. Preserve the equation source in notes or build sources if the equation itself
cannot remain editable. Confirm the renderer supports the chosen method before delivery.

Use a proper mathematical typeface such as Cambria Math, STIX Two Math, or another
available math font. Apply math typography to inline symbols, diagrams, plot labels,
legends, and units as well as display equations. A font name in OOXML does not prove
that the export used that font; inspect rendered glyphs and PDF font information when
available. Verify font substitution in the actual export path and embed or render
essential glyphs when necessary and permitted.

Conventions:

- Variables are italic where mathematically appropriate. Units, descriptive subscripts,
  method names, and operators are upright.
- Use real subscripts and superscripts in formulas, axis labels, and chemical compositions.
- Use the correct micro symbol and a space between compound units, for example
  \(\mu\Omega\,\mathrm{cm}\), \(T\,[\mathrm{K}]\), and \(p\,[\mathrm{GPa}]\).
- Keep mathematical symbols on a compatible baseline within surrounding text. Avoid
  broken fallback glyphs, cramped fractions, and equation images with excessive margins.
- Definitions belong near the first use of a symbol or a relative metric. A distant
  Appendix definition is insufficient when the main figure cannot otherwise be read.

## Self-explanatory coverage and comparison charts

For availability matrices, categorical fills or a simple dot often communicate coverage
better than a cell full of repeated `1` values. The legend must distinguish observed,
unobserved, and selected-for-modeling conditions. Blank is not automatically zero.
If counts are scientifically meaningful, label what is counted and retain the counts.

For signed-effect heatmaps, show the reference-minus-candidate definition, zero-centered
color scale, and direction in words. Use a consistent range across panels. Make cell
labels readable and account for saturated values beyond the range. An improving-task
count should not replace the mean-error summary.

For distribution plots, define the visual encodings outside the axes without crowding
their title. Explain Q1–Q3, mean markers, median lines, and whiskers once in a shared
caption. Maintain enough spacing between that caption and the figure's own heading.

## Language and terminology

Default to professional English for research-result decks unless the user requests
another language. Use established scientific and machine-learning terms and concise
sentences. Define unfamiliar abbreviations once and keep method labels consistent.
Use a collaborator-friendly dataset name when the audience does not know an internal
acronym. An accompanying message may use a different requested language, but its
scientific claims must match the deck.
