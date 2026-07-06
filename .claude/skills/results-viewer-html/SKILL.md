---
name: results-viewer-html
description: Generate a self-contained interactive HTML viewer for experiment result matrices (e.g. inverse-design trajectories across mode/order/k/scenario/path). Use when asked to build or extend an interactive results viewer / trajectory viewer HTML.
---

# Self-contained interactive results viewer (HTML)

Canonical reference implementation: `experiments/rikyu_task_scaling/analysis/build_trajectory_viewer.py`
(build: `.venv/bin/python <script> <MIRROR_DIR> <REPLAY_N>`). Adapt it rather than starting from
scratch. The requirements below were converged on with the user across many review rounds —
treat them as hard constraints unless the user says otherwise.

## Hard requirements

1. **Single self-contained file.** Vanilla JS + canvas only — no CDN, no network, no external
   assets. Anyone must be able to open the file directly in a browser.
2. **Payload = gzip + base64**, decoded with the browser-native `DecompressionStream("gzip")`
   (`atob → Blob stream → Response.text → JSON.parse`). Show a "decompressing data…" placeholder,
   and a clear browser-requirement message when `DecompressionStream` is undefined. Target
   ≤ ~20 MB on disk; encode weights as permille integers and reference element names through one
   global symbol table (never repeat strings per data point).
3. **HiDPI-aware canvases.** Scale every canvas by `devicePixelRatio` (CSS size fixed, raster
   size × DPR, `ctx.setTransform(DPR,0,0,DPR,0,0)`), and re-size from the container width on
   `resize`. Plain fixed-attribute canvases look blurry on retina — this was an explicit user
   complaint.
4. **Layout: two columns, left 2/3, right 1/3** (flex; canvases sized from the container).
   - Left top: elements × candidates heatmap (animated over optimisation steps; ▲ marks the
     best-final candidate; click a column to select; hover tooltip shows cell values).
   - Left below: selected candidate's per-target progress curves (3/4 width; 0 = seed,
     1 = target with a dashed line; vertical current-step marker) + its composition as
     **weight-sorted** bars (1/4 width, zero-weight elements hidden).
   - Left bottom: the pretraining task-order timeline for the selected order, **linked to k**
     (first k tasks highlighted, k-th marked as latest; read the realized order from the run's
     own metrics table, don't hardcode).
   - Right top: seeds → final list (fixed height ~400px, scroll, `scrollIntoView` on selection);
     the **selected row expands** to an inline detail line with predicted values at seed and
     final. No hover-tooltips for dense values — the user rejected them.
   - Right bottom: "How to use" + "Experiment" notes so the file is self-explanatory to someone
     with no context (objective incl. the z-score definition, seed strategy, paths, step count).
5. **Controls row:** dropdowns covering the full selection matrix (refresh dependent menus,
   preserve compatible selections), a frame slider (**1 frame per 5 optimisation steps** plus the
   final step), a numeric **jump-to-step** box (+ Enter key), ▶ play/pause, and **raw-data
   download buttons** — current selection (with a `notes` field documenting the JSON schema,
   filename embedding the selection) and the whole dataset.
6. **UI language: English.**

## Known pitfalls (each caused a real bug or user complaint)

- **Capped shared row sets lose per-candidate data.** If the heatmap rows are a truncated union
  (top-N elements), the selected candidate's bar panel MUST NOT read from it — store per-candidate
  overflow pairs (elements outside the rows, ≥1%) per frame and merge at render time. Otherwise a
  candidate like `O0.75 Au0.17 Ac0.05` renders as just `O`.
- **Inline chip sequences need whitespace between elements** (`" › "` separators), or there are
  no soft-wrap points and the line overflows into the neighbouring column.
- **Beware degenerate/model-independent assumptions**: seeds order in lists must match the
  candidate index in trajectory arrays; verify with the run's `seeds.json`.
- Escape/quote carefully when the HTML template lives inside a Python string inside a heredoc —
  prefer `%%PLACEHOLDER%%` substitution over f-strings.

## Validation before delivering (all required)

```bash
node -e "…extract <script>…; new Function(...)(…)"      # JS syntax check
node -e "…zlib.gunzipSync(base64 payload)…"             # payload decompresses; spot-check dims
grep -c 'class="det"' viewer.html                        # feature markers present
```
Also visually confirm one build via the Read tool or a browser screenshot when feasible, and
report the file size (warn if it grew unexpectedly — trim frame rate or thresholds, not features).

## Delivery

Send the built HTML(s) with SendUserFile; commit only the builder script (built HTML and any
result data stay out of git per repo policy — `experiments/**/*.html` is git-ignored).
