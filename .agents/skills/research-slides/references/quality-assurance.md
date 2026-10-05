# Presentation quality assurance

Use the selected base presentation skill's file and rendering checks alongside these
checks. For review-only requests, apply relevant content and visual checks within scope
using the supplied deck or renders. Full rendering and export checks apply to created
or revised deliverables; new exports are unnecessary just to provide review comments.
Keep build notes and numerical reconciliation outside audience-facing slides. Validation
proves what was checked, not that every application will render a file identically.

## Content and claims

1. **Coverage and audience fit.** Compare the outline with the purpose, requested topics,
   examples, and review comments. Check necessary definitions and units. Remove imposed
   task-specific chapters that do not serve the subject or audience.
2. **Source reconciliation.** Verify central facts against supplied sources. For numerical
   material, recompute central summaries from source tables using the stated aggregation;
   check representative cells, subgroup summaries, and claimed relative values. Calculate
   from unrounded inputs.
3. **Claim consistency.** Read each title, figure, caption, and conclusion together. Check
   that reference, measure, population, and sign agree. Qualify "best" and "improves".
   Explain local-versus-global or mean-versus-median differences that appear contradictory.
4. **Limits of evidence.** Check independent sample counts, unequal conditions, the meaning
   of spread, and whether a causal claim was tested. Do not mistake absence of a visible
   benefit for statistical equivalence.

Apply numerical checks only where relevant; a conceptual or qualitative presentation
does not require invented metrics, error bars, or a data-analysis section.

## Visual inspection

Render every final slide. Use a contact sheet to find inconsistent patterns, then inspect
each slide at presentation scale without zooming. Inspect dense figures and equations
at full resolution too, but do not use zoomed legibility as proof of projection readability.
A geometry check alone cannot detect small labels or font substitution.

- Verify final embedded figure-text sizes, not just source plotting settings. Aim for
  21–23 pt figure titles and 17–19 pt essential labels on a wide slide. Enlarge, simplify,
  or split when needed; avoid shrinking labels or the main figure to accommodate prose.
  If the user approved a different preview scale, verify against that scale instead.
- Keep titles and explanations visibly separated from figures and their own headings.
- Check text overflow, overlaps, bullet spacing, and hanging indents.
- Verify mathematical glyphs, subscripts, micro prefixes, and units after export.
- Preserve consistent colors, scales, and references when panels are split.
- Explain coverage symbols, missing values, and error bars in readable legends.
- Remove placeholders, clipped axes, stale captions, and unintended blank pages.

Fix defects in the source and render affected slides again. Complete final file checks
after the last edit. If every slide was inspected and only a few changed, inspect those
again and check order, numbering, and file integrity across the deck. Repeat numerical
work only when a new change or concern affects it.

## PDF/PPTX parity and review comments

When both formats are delivered, compare slide count, order, titles, content, and revision.
Prefer exporting the PDF from the final PPTX. For separate exports, check corresponding
content and representative renders explicitly. A new PDF cannot be paired with a stale PPTX.

Map each original page and review comment to its implemented change and new page.
Distinguish data changes from layout changes when deciding whether to recheck numbers.
Apply "all slides" comments throughout the deck and preserve a pre-review version.

## Behavioral checks for this skill

Use these scenarios to assess instruction changes. A full demonstration deck or new
domain analysis is unnecessary merely to evaluate the skill.

| Scenario | Expected decision |
| --- | --- |
| A proposal or tutorial contains no experimental results | Build sections around its purpose; do not impose an experiment-report outline |
| Three of four effects are positive, but the mean is negative | Explain frequent local benefits and the worse aggregate result |
| A coverage matrix repeats unexplained `1` values | Encode availability or explicitly define counts and highlights |
| Large source-chart labels become small when embedded | Enlarge or re-export at final slide size; simplify or split before shrinking essential text |
| The user requests all text one notch smaller | Reduce native text and regenerate embedded chart and equation text by about 10–15%; retain figure areas and check every slide, including the Appendix |
| The user requests one point smaller after a percentage reduction | Subtract 1 pt from current final-slide text sizes, accounting for embedded-asset placement scale |
| The user asks to see about ten pages before a full rebuild | Render a representative preview within that scope; reuse verified data and identify the included pages |
| The user approves a preview and requests the full deck | Preserve its type hierarchy and figure areas throughout the body and Appendix |
| Every category repeats the same qualifier | Move the shared meaning into one axis heading or short explanation; preserve units and context |
| Several plots answer different comparison questions | Separate them into focused sections, from observed outcomes toward controls and explanations |
| Two variants share a name with a size adjective | Explain the concrete difference and relevant dimensions or counts near first use |
| Four heatmaps have illegible labels | Split panels across slides while retaining comparable scales |
| A box plot mixes mean, median, IQR, and whiskers | Define encodings in a readable caption with space around the title |
| A conclusion cites a percentage with no denominator | Define reference, aggregation, and sign before the percentage |
| A band shows variation across repeated runs | Identify descriptive dispersion unless calibrated uncertainty was actually evaluated |
| The audience does not know an internal acronym | Use an understandable name and define only necessary abbreviations |
| The user asks why two slides differ | Inspect and answer; do not edit without an editing request |
| No suitable base presentation skill is installed | Use available tools and accurately identify unverified exports or renders |

## Delivery status

Report material unresolved issues. Do not claim PowerPoint inspection unless it was
opened there, or all-slide checks, source reconciliation, or format parity unless done.
Return requested artifacts; temporary renders and installation metadata are not deck content.
