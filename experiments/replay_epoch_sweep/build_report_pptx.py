#!/usr/bin/env python3
"""Build results/REPORT_20260802.pptx — replay epoch-resampling sweep + budget variants.

Run: uv run --with python-pptx python experiments/replay_epoch_sweep/build_report_pptx.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
AN = HERE / "analysis"
OUT = HERE / "results" / "REPORT_20260802.pptx"

INK = RGBColor(0x1F, 0x29, 0x37)
MUT = RGBColor(0x6B, 0x72, 0x80)
ACC = RGBColor(0x00, 0x77, 0xBB)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def txt(slide, x, y, w, h, lines, size=14, bold_first=False, color=INK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.color.rgb = color
        if bold_first and i == 0:
            p.font.bold = True
    return box


def title_bar(slide, title, sub=None):
    t = txt(slide, 0.5, 0.25, 12.3, 0.6, [title], size=24)
    t.text_frame.paragraphs[0].font.bold = True
    if sub:
        txt(slide, 0.5, 0.85, 12.3, 0.4, [sub], size=12, color=MUT)


def pic_slide(title, sub, img, top=1.35, bottom=0.2):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, sub)
    from PIL import Image

    with Image.open(img) as im:
        iw, ih = im.size
    max_w, max_h = Inches(12.9), prs.slide_height - Inches(top) - Inches(bottom)
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    s.shapes.add_picture(str(img), Emu(int((prs.slide_width - w) / 2)), Inches(top), Emu(w), Emu(h))
    return s


# 1 — title
s = prs.slides.add_slide(BLANK)
txt(s, 0.7, 2.3, 12, 1.6, ["Replay with Per-Epoch Resampling",
                           "Full fixed-count sweep + training-budget variants"], size=32, bold_first=True)
txt(s, 0.7, 4.1, 12, 1.5, [
    "replay.resample = \"epoch\": redraw each old task's n-label replay subset every epoch",
    "25 runs · 4 arms · 3 machines | 2026-07-29 → 08-02",
    "step-p8 baseline (rikyu GB200, 2026-07) · epoch-p8 (ism 4×A100) · epoch-p24 / epoch-m150 (R-CCS H200)",
], size=15, color=MUT)

# 2 — design (explains the experiment before any results)
s = prs.slides.add_slide(BLANK)
title_bar(s, "The experiment — what is being varied, and what the numbers mean")
txt(s, 0.6, 1.3, 6.4, 5.9, [
    "The setting (same as the 2026-07 sweep):",
    "· 24 material-property tasks are learned ONE AT A TIME on a",
    "  shared encoder (continual pretraining, fixed order, seed 2025)",
    "· while learning each new task, the model also revisits a small",
    "  sample of every earlier task's data — the \"replay\" that fights",
    "  catastrophic forgetting",
    "· the budget knob n = how many labeled examples PER old task",
    "  are replayed at each step; we sweep n = 100 … 2500",
    "· score = mean test R² over the 23 regression tasks AFTER all",
    "  24 steps (higher = less was forgotten)",
    "",
    "What is new — how the n examples are picked:",
    "· historically: chosen once when the step starts, then frozen —",
    "  every epoch re-trains the SAME n examples",
    "· this experiment: redraw a fresh n-example subset EVERY epoch —",
    "  same cost per epoch, but over ~60–100 epochs the model",
    "  eventually sees most of the old task's data",
], size=12.5)
txt(s, 7.2, 1.3, 5.6, 5.9, [
    "The four arms (only these flags differ, configs otherwise identical):",
    "· step-p8 — frozen subsets (the 2026-07 baseline, rikyu GB200)",
    "· epoch-p8 — per-epoch resampling (ism 4×A100)",
    "· epoch-p24 — + early-stop patience 8→24, which in practice",
    "  means every step trains the full 100 epochs (R-CCS H200)",
    "· epoch-m150 — + epoch cap 100→150 (R-CCS H200)",
    "7 budgets × 4 arms = 28 runs, every run idempotent & resumable;",
    "each records its resolved config + git commit (run_provenance.json)",
    "",
    "Caveats accepted by design:",
    "· single seed, single task order — deltas < ~0.02 ≈ noise band",
    "· arms ran on different machines (GB200 / A100 / H200); the",
    "  n2500 convergence between arms (next slide) is the built-in",
    "  negative control showing this does not distort the comparison",
], size=12.5)

# 3 — findings, each carried by one worked example
s = prs.slides.add_slide(BLANK)
title_bar(s, "Five findings — each with one number you can read off the tables",
          "all examples: mean final test R² over the 23 regression tasks; n = replayed labels per old task per step")
txt(s, 0.6, 1.45, 12.2, 5.7, [
    "1. Resampling helps at EVERY budget.  Example: at n=200 the mean rises 0.420 → 0.546 (+0.126) — frozen replay",
    "    needs n≈1000 to reach that level, so redrawing bought a ~5× budget multiplier at identical per-epoch cost.",
    "",
    "2. The gain shrinks as n approaches a task's full data.  Example: at n=2500 — full data for all but the largest",
    "    tasks — the modes nearly tie (0.600 frozen vs 0.622): when there is nothing new left to redraw, resampling",
    "    cannot add information. (This convergence is also the negative control for the hardware caveat.)",
    "",
    "3. Training longer amplifies resampling.  Example: with every step trained the full 100 epochs (p24), n=100 —",
    "    just 100 replayed labels per old task — scores 0.592, matching frozen replay's best with 25× the budget",
    "    (n=2500 → 0.600). More epochs = more distinct redraws = more of the old data eventually seen.",
    "",
    "4. But beyond ~100 epochs nothing more is gained.  Example: raising the cap to 150 epochs moves the mean by",
    "    +0.005 averaged over all 7 budgets (at n=1000: 0.644 → 0.643) — once the redraws have covered the data,",
    "    extra passes are dead weight.",
    "",
    "5. Why training length mattered at all: early stopping was quietly cutting resampling short.  Example: at",
    "    patience 8 a typical step stops after 60–65 epochs (fresh redraws keep improving val loss, vs 52–59 frozen);",
    "    at patience 24, 100% of completed steps ran to the 100-epoch cap — patience 8 had been the hidden binding limit.",
], size=12.5)

# 4-6 — figures
pic_slide("Headline — the saturation curve shifts up-left",
          "mean final test R² over 23 R² tasks vs replay n; every arm at every n above the frozen baseline",
          AN / "mean_final_compare.png")
pic_slide("Per-task saturation with both baselines — step vs epoch, 24 panels",
          "grey dashed = at-intro (step runs), red dashed = at-intro (epoch runs), green = single-task baseline; "
          "largest per-task gains: klat +0.166 @n200, tc +0.163 @n100; classification (material_type) unchanged",
          AN / "per_task_saturation_compare.png")
pic_slide("Forgetting stays event-driven — but small-n collapses are suppressed",
          "per-task trajectories across replay events; blues = frozen (light→dark = n100→n2500), reds = per-epoch resampling",
          AN / "replay_trajectories_compare.png")

# 6b — replay requirement vs task size (both baselines, all budgets)
s = pic_slide("Distance to the single-task ceiling — every budget, every arm, by task size",
              "y = single-task baseline − final; solid 0-line = single-task ceiling, dashed = at-intro level "
              "(multi-task cost); green zone = run ends ABOVE its own introduction level",
              AN / "replay_requirement_vs_size.png", top=1.5, bottom=1.0)
txt(s, 0.5, 6.65, 12.4, 0.75, [
    "big (≥20k): frozen replay never reaches at-intro even at n2500 (11% of own data); epoch+full-epochs recovers to "
    "at-intro from n≈500–1000 and plateaus there — the residual ~0.045 is multi-task cost, not forgetting. "
    "mid (3k–8k): epoch-p24/m150 reach the single-task ceiling itself (deficit ≤0.02 from n≥500) — the historical "
    "\"never past at-intro\" boundary is broken. Requirement scales with task size ⇒ next phase: ratio × epoch resampling.",
], size=11, color=MUT)

# 7 — variants table
s = prs.slides.add_slide(BLANK)
title_bar(s, "Training-budget variants — patience 24 flattens n; epochs saturate near 100")
rows = [
    ("n", "step-p8", "epoch-p8", "epoch-p24", "epoch-m150"),
    ("100", "0.371", "0.475", "0.592", "0.580"),
    ("200", "0.420", "0.546", "0.606", "0.624"),
    ("500", "0.498", "0.582", "0.637", "0.632"),
    ("1000", "0.556", "0.612", "0.644", "0.643"),
    ("1500", "0.578", "0.621", "0.641", "0.660"),
    ("2000", "0.595", "0.632", "0.647", "0.642"),
    ("2500", "0.600", "0.622", "0.639", "0.663"),
]
tbl = s.shapes.add_table(len(rows), 5, Inches(0.8), Inches(1.5), Inches(6.6), Inches(4.6)).table
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = val
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(13)
        p.font.bold = r == 0
txt(s, 7.8, 1.5, 5.0, 5.2, [
    "Reading:",
    "· p24 column is nearly flat (0.59–0.65) — with enough",
    "  epochs, the replay budget n almost stops mattering",
    "· m150 − p24 (all 7 n): −0.012 … +0.024, mean +0.005,",
    "  no n-trend — the epoch budget is saturated at ~100",
    "  everywhere, incl. the smallest n (n100: 0.580 ≤ 0.592)",
    "· best overall: n2500-m150 (0.663), within the noise band",
    "  of the p24 plateau",
    "",
    "Open control: step-p24 (frozen + long training) would",
    "cleanly separate training-time vs coverage — not yet run.",
], size=13)

# 8 — next steps
s = prs.slides.add_slide(BLANK)
title_bar(s, "Practical guidance & next steps")
txt(s, 0.6, 1.4, 12.2, 5.4, [
    "Adopt now:",
    "· default pretrain.replay.resample = \"epoch\" — it never hurt at any n (worst case +0.022, no regression observed)",
    "· with resampling, budget n≈200–500 + patience ≥24 replaces n≥1000 frozen replay at ~5–12× less replay data per epoch;",
    "  the old \"n ≥ 1000\" threshold (measured under frozen subsets) is obsolete — re-anchor at n200–500 with full-epoch training",
    "",
    "Next:",
    "· step-p24 control (4 runs, ism) to split \"train longer\" from \"resample coverage\" at large n",
    "· multi-seed (≥3) confirmation of the headline deltas before publishing numbers",
    "· propagate to the task-scaling protocol: its replay branches (n1000/n1500) were sized under frozen-subset assumptions",
    "",
    "Data & reproduction: experiments/replay_epoch_sweep/ (README, worker/launch/sbatch scripts, analysis/*.py — all in git);",
    "results mt_*.csv + figures + this deck under results/ & analysis/ (rsync policy, not in git); raw runs on ism & R-CCS + local mirror.",
], size=13.5)

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print(f"saved {OUT}")
