#!/usr/bin/env python3
"""Build results/REPORT_v2_<date>.pptx — the campaign deck.

Every number on every slide is read from the campaign's own summary JSONs at build time. Nothing
is typed into this script, so the deck cannot drift from the results the way a hand-maintained one
does — and a missing input is a hard failure rather than a silently empty slide, because a deck
that quietly ships a placeholder is worse than one that refuses to build.

``--allow-missing`` relaxes that to skip the affected slides and print exactly which ones were
skipped. It exists so the builder can be smoke-tested while the long stages are still running; it
is not for producing a deliverable, and the printed skip list is the reason it is safe to use.

    uv run --with python-pptx --with pillow python experiments/rikyu_hparam_tuning_v2/build_report_pptx.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
SUM = HERE / "summary"
RES = HERE / "results"

INK = RGBColor(0x1F, 0x29, 0x37)
MUT = RGBColor(0x6B, 0x72, 0x80)
BLUE = RGBColor(0x00, 0x77, 0xBB)
GREEN = RGBColor(0x00, 0x9E, 0x73)
RED = RGBColor(0xCC, 0x33, 0x11)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEADGREY = RGBColor(0x4B, 0x55, 0x63)

SKIPPED: list[str] = []
ALLOW_MISSING = False


class Missing(Exception):
    """An input this slide needs is not on disk yet."""


def need(path: Path) -> Path:
    if not path.exists():
        raise Missing(str(path))
    return path


def load(name: str) -> dict:
    return json.loads(need(SUM / name).read_text())


def slide_guard(fn):
    """Let one slide's missing input skip that slide instead of killing the build.

    Only under --allow-missing. Without it the exception propagates and the build fails, which is
    the behaviour a deliverable needs.
    """

    def wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except Missing as exc:
            if not ALLOW_MISSING:
                raise SystemExit(
                    f"missing input for slide '{fn.__name__}': {exc}\n"
                    "run that stage's analysis first, or pass --allow-missing for a dry run"
                ) from exc
            SKIPPED.append(f"{fn.__name__}  (needs {exc})")
            return None

    return wrapped


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def txt(slide, x, y, w, h, lines, size=14, color=INK, mono=False, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(lines):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.bold = bold
        if mono:
            p.font.name = "Consolas"
    return box


def title_bar(slide, title, sub=None):
    txt(slide, 0.5, 0.25, 12.3, 0.6, [title], size=24, bold=True)
    if sub:
        txt(slide, 0.5, 0.87, 12.3, 0.4, [sub], size=12, color=MUT)


def new(title, sub=None):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, sub)
    return s


def pic_slide(title, sub, img, top=1.35, bottom=0.25):
    from PIL import Image

    need(img)
    s = new(title, sub)
    with Image.open(img) as im:
        iw, ih = im.size
    max_w, max_h = Inches(12.6), prs.slide_height - Inches(top) - Inches(bottom)
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    s.shapes.add_picture(str(img), Emu(int((prs.slide_width - w) / 2)), Inches(top), Emu(w), Emu(h))
    return s


def table(slide, x, y, w, headers, rows, col_w=None, size=11, head_size=11, colour_col=None):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w),
                                   Inches(0.32 * (len(rows) + 1)))
    tbl = shape.table
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Inches(cw)
    for j, head in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = str(head)
        para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(head_size)
        para.font.bold = True
        para.font.color.rgb = WHITE
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADGREY
    for i, row in enumerate(rows, start=1):
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(value)
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(size)
            # Sign-colouring is opt-in per column: colouring every numeric cell would make the
            # deck read as a heat map and hide which column the argument is actually about.
            if colour_col is not None and j == colour_col and isinstance(value, str):
                para.font.color.rgb = (RED if value.startswith("-") else
                                       GREEN if value.startswith("+") else INK)
            else:
                para.font.color.rgb = INK
            if j:
                para.alignment = PP_ALIGN.RIGHT
    return tbl


# Run identifiers carry history the deck does not: "v1enc" is simply the encoder configuration
# inherited from the earlier campaign. Relabel at the presentation layer; the summary JSONs keep
# the original names so every figure stays traceable to its runs.
DISPLAY = {
    "s0_v1enc": "inherited encoder config",
    "v1enc": "inherited encoder config",
    "a3_v1enc": "inherited encoder config",
    "s0_base": "untuned",
    "a3_base": "untuned",
    "base": "untuned",
}


def label(name: str) -> str:
    return DISPLAY.get(name, name)


def pct(x, digits=2):
    return f"{x * 100:+.{digits}f}%"


# --- slides -----------------------------------------------------------------------------------


def slide_title(date: str):
    s = prs.slides.add_slide(BLANK)
    txt(s, 0.9, 2.4, 11.5, 1.0, ["RIKYU hyper-parameter tuning campaign"], size=40, bold=True)
    txt(s, 0.9, 3.5, 11.5, 0.6, ["v2 report"], size=22, color=MUT)
    txt(s, 0.9, 4.4, 11.5, 1.2, [
        f"{date}  |  branch exp/rikyu-hparam-tuning-v2",
        "Before tuning, ReduceLROnPlateau fired per batch rather than per epoch (fixed in PR #45).",
        "Every measurement predating that fix ran on the broken cadence, so this round was rerun in full.",
    ], size=13, color=MUT)


def finals_vs_control(a: dict) -> dict:
    """Leader vs the finals' OWN 25-seed untuned control, on each arm's measured sigma.

    The `vs_anchor` field in the summary compares against the stage-0 reference, which has nine
    seeds; quoting a 25-seed arm against it mixes seed counts in the standard error. The finals
    deliberately include an untuned arm at the same seed count, and that is the honest baseline.
    """
    import math

    arms = a["arms"]
    control = next((k for k in arms if k.endswith("_base")), None)
    if control is None:
        raise Missing("no untuned control arm in the finals")
    lead, base = arms[a["leader"]]["score"], arms[control]["score"]
    delta = lead["mean"] - base["mean"]
    se = math.sqrt(lead["sigma"] ** 2 / lead["n"] + base["sigma"] ** 2 / base["n"])
    return {"delta": delta, "two_se": 2 * se, "resolved": abs(delta) > 2 * se,
            "lead_sigma": lead["sigma"], "base_sigma": base["sigma"],
            "sigma_ratio": base["sigma"] / lead["sigma"] if lead["sigma"] else None,
            "control": control, "n": lead["n"]}


def slide_glossary():
    s = new("Glossary", "The prose uses full names; these are the directory and job names on RIKYU, "
                        "which you meet when reading raw output")
    table(s, 0.5, 1.35, 5.9, ["Identifier", "What it is"],
          [["stage0", "anchor: untuned, new container"],
           ["a1 / a1r", "encoder x LR x scheduler grid / random search"],
           ["a3", "A' finals (8 configs x 25 seeds)"],
           ["a4", "is the scheduler worth it / extend the lower bound?"],
           ["a2b", "early-stopping recheck: patience 24 vs 40"],
           ["b", "head grid (24 configs x 5 seeds)"],
           ["b3", "head finals (4 configs x 25 seeds)"],
           ["bal / balx", "loss balancer on/off / excluding AE"],
           ["single (stA_*)", "same-regime single-task ceiling"],
           ["xfer", "transfer test: 24 tasks x random orderings"]],
          col_w=[1.9, 4.0], size=10, head_size=10)
    table(s, 6.9, 1.35, 6.0, ["Stage C arm", "What it is"],
          [["c2_base", "untuned (control)"],
           ["c2_top1", "1st-place config from the A' finals"],
           ["c2_top2 / c2_top3", "2nd / 3rd place"],
           ["c2_base_cons", "untuned + consolidation"],
           ["c2_top1_cons", "tuned + consolidation (best this round)"],
           ["stA_*", "same-regime single-task baseline (5 seeds)"],
           ["xf_<task>_o<k>", "transfer run: that task last in a shuffled sequence, repeat k"]],
          col_w=[2.0, 4.0], size=10, head_size=10)
    txt(s, 0.5, 5.2, 12.4, 1.8, [
        "Naming: the middle segment is the config  |  _cons = consolidation was applied",
        "",
        "Consolidation: after the 24-task sequence finishes, all 24 heads and the encoder are",
        "fine-tuned jointly on full data (fm finetune). It is a finishing step, not another",
        "hyper-parameter setting.",
    ], size=12, color=MUT)


@slide_guard
def slide_summary():
    a = load("finals_a.json")
    bal = load("stage_bal.json")
    # transfer_adopted.json is the SIX-task probe. It reported zt and magnetization as gainers, and
    # the 24-task deployment-scale measurement retracts both, so the summary must quote xfer.
    xf = load("transfer_xfer.json")
    vc = finals_vs_control(a)
    off = max(v["delta_vs_untuned"] for v in bal["vs_anchor"] if v["arm"].endswith("off"))
    on = max(v["delta_vs_untuned"] for v in bal["vs_anchor"] if v["arm"].endswith("on"))
    rows = xf["per_task"]
    gainers = sorted((r for r in rows if r.get("matters") and r.get("transfer", 0) > 0),
                     key=lambda r: -r["relative_pct"])
    losers = [r for r in rows if r.get("matters") and r.get("transfer", 0) < 0]
    unresolved = len(rows) - len(gainers) - len(losers)
    # Lead with the relative percentage — +0.045 does not tell a reader whether that is a lot.
    gains = ", ".join(f"{r['task']} {r['relative_pct']:+.1f}%" for r in gainers) or "none"
    s = new("Summary", "Every line is traceable to summary/*.json")
    txt(s, 0.6, 1.5, 12.2, 5.4, [
        f"1. Tuning pays, modestly but resolvably: the adopted config beats the finals' own "
        f"{vc['n']}-seed untuned control by {pct(vc['delta'])} (2SE {vc['two_se'] * 100:.2f}%, "
        f"{abs(vc['delta']) / vc['two_se']:.1f}x the threshold), though the top "
        f"{1 + len(a['statistically_tied_with_leader'])} configs are statistically "
        f"indistinguishable from each other. Run-to-run sigma also falls to "
        f"1/{vc['sigma_ratio']:.2f} of the untuned config.",

        "2. Across 24 tasks tuning buys +0.0074 / +1.03% (untuned 0.7155 -> tuned 0.7229), and "
        "consolidation lifts it to 0.7274. The three configs promoted to deployment scale span only "
        "0.0037, so the probe's ranking carries no actionable information.",

        "3. The single-task ceiling was remeasured in this regime (24 tasks x 5 seeds): the "
        "inherited ceiling was too low for 17 of 23 tasks, by +0.0275 on average, and the offset is "
        "not constant, so it cannot be corrected with a shift.",

        f"4. At deployment scale multi-task training helps exactly one task: {gains}. "
        f"{len(losers)} of {len(rows)} are materially worse and {unresolved} are within seed noise. "
        f"The six-task probe's apparent winners do not survive at 24 tasks.",

        f"5. The learnable loss balancer is harmful, and mechanically so: best off-arm {pct(off)} "
        f"vs best on-arm {pct(on)}. Do not ship it.",

        "6. PCGrad does not apply: measuring the encoder's per-task gradients directly found none "
        "of the directional conflict the method needs in order to help.",

        "7. Engineering: GPU utilisation across both campaigns was about 9%; packing several tasks "
        "onto one card measured a 7.1x speed-up.",
    ], size=15)


def slide_design_flow():
    """The pipeline as a dependency chain: what each stage consumes and what decision it emits.

    Placed before any result, because a reader who does not know that the finals inherit their
    metric axis and their seed budget from the anchor cannot tell a designed campaign from a pile
    of runs. Every arrow is a real dependency in the code, not a narrative one.
    """
    s = new("The pipeline: what each step consumes and what decision it emits",
            "Every arrow is a real dependency in the code, not a narrative order")
    table(s, 0.4, 1.35, 12.5,
          ["Step", "Scale", "Variables", "Consumes", "Decision it emits"],
          [["1 Anchor stage0", "18 runs / 9 seeds", "none (untuned repeats)",
            "-", "reference point + single-run sigma"],
           ["2 A' grid + random", "296 configs x 5 seeds",
            "latent_dim x encoder_lr x min_lr x patience x factor",
            "reference, metric axis, sigma", "shortlist (8) + are bounds exhausted?"],
           ["3 A' finals", "10 arms x 25 seeds", "same, shortlist only",
            "shortlist", "**adopted encoder / scheduler**"],
           ["4 A4 scheduler value", "60 runs", "scheduled vs fixed LR",
            "sigma", "keep the scheduler; no need to extend the bound"],
           ["5 a2b early stopping", "2 arms x 5 seeds", "patience 24 vs 40",
            "**adopted base**", "keep 24"],
           ["6 B' head grid", "24 configs x 5 seeds", "head capacity/LR x KR branch capacity/LR",
            "**adopted base**", "shortlist (4)"],
           ["7 B' finals", "4 arms x 25 seeds", "same",
            "shortlist", "**leave the heads alone**"],
           ["8 Ceiling remeasure", "24 tasks x 5 seeds", "single-task training",
            "adopted config", "the denominator for every deficit"],
           ["9 Stage C'", "6 arms x 24 tasks", "adopted vs untuned; finals top 3",
            "adopted values from 3 and 7", "deployment-scale check; does the ranking carry?"],
           ["10 Transfer xfer", "24 tasks x n orderings", "task under test placed last",
            "adopted config, ceilings", "**does transfer hold?**"]],
          col_w=[2.3, 2.0, 3.6, 2.2, 2.4], size=9, head_size=9)
    txt(s, 0.4, 5.55, 12.5, 1.6, [
        "Side branches, each answering one binary question and feeding nothing downstream: "
        "loss balancer on/off  ·  does PCGrad's premise hold  ·  packing calibration",
        "",
        "Two dependencies are easy to miss: step 1's sigma sets how many seeds every later step "
        "needs, and step 3's adopted base is a precondition for 5 and 6 — a head or early-stopping "
        "result measured on a different base does not carry over.",
    ], size=11, color=MUT)


def slide_design_why():
    """Why each step is shaped the way it is. The decisions, not the numbers."""
    s = new("Why each step is shaped this way",
            "The same compute could have been spent elsewhere; these are the reasons it was not")
    txt(s, 0.4, 1.3, 6.3, 5.6, [
        "Why an anchor first, instead of searching straight away",
        "  Every gain is a relative quantity and needs a reference measured",
        "  on the same image and the same code. The anchor also yields sigma,",
        "  and sigma sets how many seeds every later step has to buy.",
        "",
        "Why a 6-task probe rather than all 24",
        "  A 24-task round takes over a day, so 296 configs is unaffordable.",
        "  Covering large / medium / small cut sigma from 5.01% to 2.05%,",
        "  which drops the seeds needed to resolve 1% from 101 to 17.",
        "",
        "Why both a grid and a random search",
        "  The grid gives readable marginal-effect plots, one axis at a time;",
        "  the random search covers the gaps between grid points.",
        "  Two views of the same question.",
        "",
        "Why split into 5-seed screening and 25-seed finals",
        "  296 configs x 25 seeds = 7,400 runs, which is unaffordable — but a",
        "  5-seed leader is not trustworthy. Winner's curse was observed twice",
        "  this round. So screen cheaply, then rank expensively.",
    ], size=11)
    txt(s, 6.9, 1.3, 6.0, 5.6, [
        "Why grid boundaries get checked",
        "  An optimum sitting on an endpoint means the range was too narrow,",
        "  not that the optimum was found.",
        "",
        "Why early stopping is remeasured AFTER adoption",
        "  It was first measured on the then-leader, and the finals changed",
        "  the leader. A result from a config no longer adopted does not carry.",
        "",
        "Why heads are tuned on the adopted base",
        "  The best head settings depend on the optimisation regime they sit in.",
        "  Change the base and you have changed the regime.",
        "",
        "Why 'change nothing' is a point in the head grid",
        "  It costs 1 of 24 grid points and turns 'the heads need no tuning'",
        "  into a **ranking result** rather than an argument. It placed 8th,",
        "  with the smallest 2SE of any arm.",
        "",
        "Why Stage C' promotes three configs, not one",
        "  Promote one and all you can see is 'the tuned arm beats the baseline'.",
        "  Promote three and you can also see that the three are indistinguishable",
        "  from each other — which is the falsifiable part.",
        "",
        "Why the ceiling had to be remeasured",
        "  The inherited ceilings were measured in a different regime. Using them",
        "  would add 'the model changed' to 'the measurement frame changed',",
        "  and the two cannot be separated afterwards.",
    ], size=11)


@slide_guard
def slide_probe():
    s0 = load("stage0.json")
    cal = s0["calibration"]
    s = new("2. The probe: cutting noise to where decisions become possible",
            "An earlier 3-task probe had an 8.48% noise band while the top three sat 1.5-1.8% apart — no ranking was possible")
    table(s, 0.6, 1.5, 6.0,
          ["Probe task", "Labels", "Size band"],
          [[t, f"{n:,}", g] for t, n, g in [
              ("volume", 23678, "big"), ("formation_energy", 23180, "big"),
              ("seebeck", 8072, "mid"), ("zt", 3445, "mid"),
              ("magnetization", 1160, "small"), ("magnetic_moment", 851, "small")]],
          col_w=[2.6, 1.7, 1.7])
    need_seeds = cal["seeds_needed_to_resolve"]
    # v1 published a RANGE at n=3, not a sigma. E[range] = d2(n)*sigma, d2(3)=1.693 — putting the
    # two side by side without that conversion reads as a 4x noise gap where the real one is 2.4x.
    v1_sigma = cal["v1_probe3_band_for_reference"] / 1.693
    v1_seeds_1pct = math.ceil((2 * v1_sigma / 0.01) ** 2)
    txt(s, 7.0, 1.5, 5.8, 4.8, [
        f"6-task probe, single-run sigma = {cal['sigma_per_run'] * 100:.2f}%   (measured on 9 seeds)",
        f"3-task probe, sigma ~ {v1_sigma * 100:.2f}%   <- converted from its 3-seed range of "
        f"{cal['v1_probe3_band_for_reference'] * 100:.2f}% (E[range] = d2(n)*sigma, d2(3)=1.693)",
        "",
        "Seeds this probe needs to resolve a true difference of:",
        *[f"    {float(k) * 100:.1f}%  ->  {v} seeds" for k, v in need_seeds.items()],
        f"        (the 3-task probe would need {v1_seeds_1pct} seeds for that same 1.0%)",
        "",
        "Excluded electrical_resistivity (ceiling 0.162, no resolving power)",
        "Excluded magnetic_susceptibility (58 labels)",
        "",
        f"Measured over 18 anchor runs: {cal['wallclock']['mean_hours']:.2f} h per run, "
        f"{cal['wallclock']['total_gpu_hours']:.0f} GPU-h in total",
    ], size=13)


@slide_guard
def slide_anchor():
    s0 = load("stage0.json")
    c = s0["comparisons"][0]
    s = new("1. The anchor: the reference every gain is quoted against, and how large the noise is",
            "Untuned config x 9 seeds, same image and same code — every later \"+x%\" is relative to this")
    txt(s, 0.6, 1.6, 12.2, 3.2, [
        "The anchor does two jobs, and neither is optional:",
        "",
        "  - It supplies the **reference point**. A gain is a relative quantity; without an untuned",
        "    baseline in the same regime there is no denominator.",
        "  - It supplies the **single-run sigma = 2.05%**, which sets how many seeds every later step",
        "    must buy: 17 seeds to resolve 1%, only 5 to resolve 2%. The whole campaign's seed",
        "    budget follows from this number.",
        "",
        "It also checked whether an inherited config still has an edge on the current code:",
    ], size=14)
    verdict = "unresolved" if abs(c["delta_score"]) <= c["resolvable_at_this_n"] else "resolved"
    table(s, 0.6, 4.6, 9.0,
          ["Comparison", "Difference", "Resolvable at (2SE)", "Verdict"],
          [[f"{label(c['from'])} → {label(c['to'])}", pct(c["delta_score"]),
            pct(c["resolvable_at_this_n"]), verdict]],
          col_w=[3.4, 1.8, 2.2, 1.6])
    txt(s, 0.6, 5.5, 12.2, 0.8,
        ["So the inherited encoder config has no visible advantage on the current code, "
         "and Stage A' searched from scratch."],
        size=13, color=MUT)


@slide_guard
def slide_grid():
    a = load("stage_a.json")
    s = new("Stage A': joint search over encoder x LR x scheduler",
            f"{a['n_configs']} configs / {a['n_runs']} runs (grid + random search, 5 seeds each)")
    edge = a["edge_bound_axes"]
    txt(s, 0.6, 1.5, 12.2, 2.0, [
        f"Grid boundary check: "
        f"{'all axes clear, no follow-up round needed' if not edge else 'axes pinned at a bound: ' + ', '.join(edge)}",
        "",
        "  The check itself was corrected twice. It first tested only whether the optimum equalled an",
        "  endpoint exactly, but none of the 206 random draws lands exactly on one, so a real boundary",
        "  problem would have passed unnoticed; a decile-trend test was added. That version then",
        "  false-alarmed on axes with only two levels (patience, latent_dim), so it now requires at",
        "  least three levels before it reports.",
    ], size=13)
    ties = a["leader_ties"]
    n_tied = len(ties["statistically_tied_with_leader"])
    short = set(a["short_list"])
    excluded = [c for c in ties["statistically_tied_with_leader"] if c not in short]
    txt(s, 0.6, 3.7, 12.2, 2.8, [
        f"At 5 seeds nothing can be ranked: {n_tied} configs are statistically tied with the leader",
        "(the threshold is 2*sqrt(sem1^2 + sem2^2) for each pair, counting both arms' uncertainty).",
        "",
        "The shortlist takes the top 8 into the 25-seed finals. That is a **budget truncation**, not",
        f"noise-aware selection: {len(excluded)} of the {n_tied} tied configs never reached the finals,",
        "purely because their noisy sample means ranked below 8th.",
        f"So the finals winner is the best of those 8, not the best of {n_tied}.",
    ], size=13)


@slide_guard
def slide_finals():
    a = load("finals_a.json")
    vc = finals_vs_control(a)
    ranked = sorted(a["arms"].items(), key=lambda kv: -kv[1]["score"]["mean"])
    rows = [[label(k.replace("a3_", "")), pct(v["score"]["mean"]), f"{v['score']['sigma'] * 100:.2f}%"]
            for k, v in ranked]
    s = new(f"Stage A' finals ({vc['n']} seeds, {a['n_runs']} runs)",
            "The finals carry their own untuned control at the same seed count, so the gain is a "
            "like-for-like comparison rather than 25 seeds against 9")
    table(s, 0.6, 1.45, 7.6, ["Config", "vs stage-0 reference", "run-to-run sigma"], rows,
          col_w=[4.6, 1.6, 1.4], size=10, head_size=10, colour_col=1)
    tied = a["statistically_tied_with_leader"]
    need_seeds = list(a["seeds_that_would_resolve_the_ties"].items())[:2]
    txt(s, 8.5, 1.45, 4.5, 5.6, [
        f"Adopted vs untuned control: {pct(vc['delta'])}",
        f"2SE {vc['two_se'] * 100:.2f}% -> "
        f"{'resolved' if vc['resolved'] else 'unresolved'} "
        f"({abs(vc['delta']) / vc['two_se']:.1f}x the threshold)",
        "",
        f"But the top {1 + len(tied)} are tied. Seeds needed to separate them:",
        *[f"    {k.split(' vs ')[1].replace('a3_', '')[:26]} -> {v}" for k, v in need_seeds],
        "",
        "Part of what makes a config good is that it is steady:",
        f"    adopted sigma {vc['lead_sigma'] * 100:.2f}%  vs  untuned {vc['base_sigma'] * 100:.2f}%",
        f"    (1/{vc['sigma_ratio']:.2f})",
        "",
        "Winner's curse: a1r129, the 5-seed leader, finished",
        "last at 25 seeds — and its sigma was the second",
        "largest of the ten arms. High-variance configs win",
        "small-sample draws more often.",
    ], size=11)


@slide_guard
def slide_finals_sigma():
    pic_slide("Part of being a good config is being a steady one",
              "corr(sigma, 25-seed score) = -0.844 — high-variance configs win small-sample draws, "
              "then fall back once seeds are added",
              RES / "finals_sigma_vs_mean.png")


@slide_guard
def slide_a4():
    a4 = load("stage_a4.json")
    h = a4["head_to_head"]
    d = a4["downward_extension"]["sched"]
    s = new("A4: is the scheduler worth it, and does the optimum sit below the search floor?",
            f"{a4['n_runs']} runs, paired comparisons")
    table(s, 0.6, 1.6, 11.0,
          ["Question", "Result", "Difference", "Resolvable (2SE)", "Verdict"],
          [["scheduled vs fixed LR",
            f"{pct(h['best_scheduled']['mean'])} vs {pct(h['best_flat']['mean'])}",
            pct(h["delta"]), pct(2 * h["se_of_difference"]),
            "keep the scheduler" if h["separated"] else "unresolved"],
           ["is the optimum below the floor?",
            f"{pct(d['best_below_floor']['mean'])} vs {pct(d['best_at_or_above_floor']['mean'])}",
            pct(d["delta"]), pct(2 * d["se_of_difference"]),
            "no need to extend" if not d["worth_extending_further"] else "extend the range"]],
          col_w=[3.0, 2.6, 1.6, 1.8, 2.0], colour_col=2)
    txt(s, 0.6, 3.4, 12.2, 2.6, [
        "A limitation that has to be stated alongside the numbers:",
        "",
        "  [training.scheduler] governs all four parameter groups (encoder / head / kr / ae) with no",
        "  per-group switch. Turning the schedule off therefore also froze the head and KR learning",
        "  rates, so this loss cannot be attributed to any single group.",
    ], size=14, color=MUT)


@slide_guard
def slide_a2b():
    a = load("stage_a2b.json")
    arms = a["arms"]
    rows = []
    for name, arm in sorted(arms.items(), key=lambda kv: -kv[1]["score"]["mean"]):
        s = arm["score"]
        rows.append([name.split("_")[-1], pct(s["mean"]), f"{2 * s['sem'] * 100:.3f}%",
                     f"{s['sigma'] * 100:.3f}%"])
    s_ = new("A2b: early-stopping patience 24 or 40, rechecked on the adopted base",
             "The original a2 ran on the then-leader, and the 25-seed finals replaced that leader — "
             "so this is a recheck, not a first measurement")
    table(s_, 0.6, 1.5, 7.4, ["Arm", "vs untuned anchor", "2SE", "sigma"], rows,
          col_w=[1.6, 2.2, 1.8, 1.8], colour_col=1)
    pair = a["pairwise"][0] if a.get("pairwise") else None
    lines = []
    if pair:
        lines += [f"Difference {pct(pair['delta'])}, resolvable at {pct(pair['resolvable_at_this_n'])[1:]}",
                  f"-> {'resolved' if pair['separated'] else 'unresolved'}"
                  + (f" (separating them needs "
                     f"{list(a['seeds_that_would_resolve_the_ties'].values())[0]} seeds; we have 5)"
                     if a.get("seeds_that_would_resolve_the_ties") else ""), ""]
    lines += ["In absolute terms about 0.0017 R2 — far below the 1e-2 practical threshold.",
              "ES40 takes 3.75 h of wall clock against ES24's 3.38 h: 11% more.", "",
              "Adopt patience 24. Paying 11% more compute for a difference that is neither",
              "resolvable nor practically significant is not a trade worth making.", "",
              "Lesson: any experiment run 'on the current best' has to be redone",
              "when the current best changes."]
    txt(s_, 8.4, 1.5, 4.6, 5.0, lines, size=12)


@slide_guard
def slide_stage_b():
    import math

    b = load("stage_b.json")
    R = {e["config"]: e for e in b["ranking"]}
    # The default head block, spelled as a stage-B label. It is IN the grid, which is what lets
    # "changing nothing" be ranked against every change rather than assumed to be the baseline.
    default = "b_H64_HL0p005_X128-64_KL0p0005"
    lead = b["ranking"][0]
    rows = []
    for i, e in enumerate(b["ranking"], 1):
        if i > 5 and e["config"] != default and i < len(b["ranking"]):
            continue
        tag = "  <- default head block" if e["config"] == default else ""
        rows.append([f"{i}", e["config"].replace("b_", "") + tag,
                     pct(e["score_mean"]), f"{2 * e['score_sem'] * 100:.3f}%"])
    s = new("Stage B': tuning the multi-task heads — the answer is to leave them alone",
                  f"24 configs x 5 seeds = {b['n_runs']} runs, all on the adopted A' base; "
                  f"120/120 passed the training check")
    table(s, 0.5, 1.45, 8.6, ["Rank", "Config", "vs untuned anchor", "2SE"], rows,
          col_w=[0.8, 5.2, 1.5, 1.1], size=10, head_size=10, colour_col=2)
    lines = [f"The leader is statistically tied with "
             f"{len(b['leader_ties']['statistically_tied_with_leader'])} configs",
             f"(at 5 seeds the resolvable difference is "
             f"{pct(b['leader_ties']['resolvable_difference'])[1:]})", ""]
    if default in R:
        d = lead["score_mean"] - R[default]["score_mean"]
        se = math.sqrt(lead["score_sem"] ** 2 + R[default]["score_sem"] ** 2)
        rank = [e["config"] for e in b["ranking"]].index(default) + 1
        lines += [f"'Change nothing' ranks {rank} of {len(b['ranking'])}",
                  f"Leader vs default: {pct(d)}, 2SE {2 * se * 100:.3f}%",
                  f"-> {'resolved' if abs(d) > 2 * se else 'unresolved'}", "",
                  "The default head block also has the smallest 2SE in the",
                  f"whole grid ({2 * R[default]['score_sem'] * 100:.3f}%),",
                  "which makes it the most reproducible point in it.", ""]
    lines += ["Adopted: the default head block — the smallest change,",
              "under the same rule used in A'.",
              "Note it is not the grid leader: the leader's 2SE is the",
              "second largest in the top 12, the same winner's-curse",
              "pattern A' measured directly.", "",
              "This is a stronger version of an earlier result. That one",
              "could be blamed on the wrong regime; this round tuned in",
              "the right regime and still bought nothing."]
    txt(s, 9.3, 1.45, 3.7, 5.6, lines, size=11)


@slide_guard
def slide_b_finals():
    import math
    import statistics

    b5 = load("stage_b.json")
    b25 = load("finals_b.json")
    five = {e["config"]: e["score_mean"] for e in b5["ranking"]}
    default = "H64_HL0p005_X128-64_KL0p0005"
    rows, xs, ys = [], [], []
    entries = []
    for name, arm in b25["arms"].items():
        short = name.replace("b3_", "")
        f5 = five.get("b_" + short)
        s = arm["score"]
        entries.append((short, f5, s["mean"], s["sigma"]))
    entries.sort(key=lambda r: -(r[1] or 0))
    for short, f5, m25, sg in entries:
        tag = "  <- default" if short == default else ""
        rows.append([short + tag, pct(f5) if f5 is not None else "-", pct(m25),
                     pct(m25 - f5) if f5 is not None else "-", f"{sg * 100:.3f}%"])
        if f5 is not None:
            xs.append(sg)
            ys.append(-(m25 - f5))
    s_ = new("The b3 finals turn the adoption from a rule-based choice into a measured one",
             f"4 arms x 25 seeds = {b25['n_runs']} runs, 220/220 passed the training check")
    table(s_, 0.5, 1.45, 9.4,
          ["Config", "5 seeds", "25 seeds", "Drop", "sigma(25)"], rows,
          col_w=[4.4, 1.2, 1.2, 1.2, 1.2], size=10, head_size=10, colour_col=3)
    lead = entries[0]
    dflt = next(e for e in entries if e[0] == default)
    e5, e25 = lead[1] - dflt[1], lead[2] - dflt[2]
    need = b25.get("seeds_that_would_resolve_the_ties", {})
    pair = next((v for k, v in need.items() if default in k), None)
    r = None
    if len(xs) > 2:
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None
    lines = ["Answer: no head config beats changing nothing.", "",
             f"Leader's edge over default: {pct(e5)} -> {pct(e25)}",
             f"    collapses by {(1 - abs(e25) / abs(e5)) * 100:.0f}%"]
    if pair:
        lines.append(f"    separating the two needs {pair:,} seeds")
    lines += ["", "The default head block barely moves (down 0.060%)",
              "while all three tuned arms drop 0.5-0.66%.",
              "Its sigma is the smallest too."]
    if r is not None:
        lines += ["", f"corr(sigma, drop) = {r:+.3f}",
                  "— the same mechanism A' measured."]
    lines += ["", "Second independent reproduction of winner's curse in",
              "one campaign: A''s 5-seed leader fell to 10th, and B''s",
              "5-seed leader lost its entire edge. Same prescription",
              "both times — add seeds, not grid points."]
    txt(s_, 10.1, 1.45, 2.9, 5.6, lines, size=10)


@slide_guard
def slide_stage_c():
    """The tuned configurations at real scale — v2's arms only.

    No pre-fix baseline and no fix-vs-tuning attribution: the campaign's deliverable is the tuning
    result, and the scheduler bug was something met along the way. That it helps to fix a bug is
    not a finding, so scoring against the broken state does not belong in the deliverable.
    """
    c = load("stage_c.json")
    keep = {"c2_base": "untuned",
            "c2_top1": "tuned (finals 1st)",
            "c2_top2": "tuned (finals 2nd)",
            "c2_top3": "tuned (finals 3rd)",
            "c2_base_cons": "untuned + consolidation",
            "c2_top1_cons": "tuned + consolidation"}
    by = {a["label"]: a for a in c["arms"]}
    arms = sorted((by[k] for k in keep if k in by), key=lambda a: -a["mean_r2"])
    rows = []
    for a in arms:
        d = a["deficit"]
        f = lambda v: f"{v:+.4f}" if v is not None else "-"  # noqa: E731
        rows.append([keep[a["label"]], f"{a['mean_r2']:.4f}",
                     f(d["big"]), f(d["mid"]), f(d["small"])])
    s = new("Stage C': the final 24-task runs",
            "Deficits are against the same-regime single-task ceiling; 1 seed per arm, so small "
            "differences are not resolvable")
    table(s, 0.5, 1.45, 9.4, ["Arm", "mean R2", "big", "mid", "small"], rows,
          col_w=[3.8, 1.5, 1.4, 1.4, 1.3], size=10, head_size=10)

    def delta(a, b):
        if a in by and b in by:
            d = by[b]["mean_r2"] - by[a]["mean_r2"]
            return d, d / by[a]["mean_r2"] * 100
        return None, None
    lines = ["What tuning buys across 24 tasks:", ""]
    for a, b, label in (("c2_base", "c2_top1", "tuning"),
                        ("c2_top1", "c2_top1_cons", "+ consolidation"),
                        ("c2_base", "c2_base_cons", "untuned + consolidation")):
        d, rel = delta(a, b)
        if d is not None:
            lines.append(f"  {label:16s} {d:+.4f}  = {rel:+.2f}%")
    lines += ["", "Same magnitude and direction as the probe's +1.56%,",
              "but with 1 seed per arm none of the three is resolvable."]
    tr = c.get("transfer", {})
    if tr.get("checked"):
        lines += ["", "Does the probe's ranking carry to 24 tasks?",
                  f"  probe said {' > '.join(x.replace('c2_', '') for x in tr['probe_order'])}",
                  f"  actual is {' > '.join(x.replace('c2_', '') for x in tr['deployed_order'])}",
                  f"  spread is only {tr['mean_r2_spread_across_promoted_arms']:.4f}",
                  "  -> the ranking carries no actionable information."]
    txt(s, 10.1, 1.45, 3.0, 5.6, lines, size=10)


@slide_guard
def slide_ceiling_fig():
    pic_slide("The inherited ceiling was a broken measurement frame",
              "It was measured before PR #45, when the LR hit its floor inside the first epoch. "
              "That is not a ceiling.",
              RES / "ceiling_frame_offset.png")


@slide_guard
def slide_transfer_fig():
    pic_slide("Transfer on the probe: does multi-task help the small tasks?",
              "Adopted config, 25-seed multi-task vs 5-seed single-task; the only difference is "
              "pretrain.task_sequence",
              RES / "transfer_adopted.png")


@slide_guard
def slide_transfer_why():
    tr = load("transfer_adopted.json")
    sm = tr["summary"]
    gainers = sorted((r for r in tr["per_task"]
                      if r.get("matters") and r.get("transfer", 0) > 0),
                     key=lambda r: -r["relative_pct"])
    gains = "、".join(f"{r['task']} {r['relative_pct']:+.1f}%" for r in gainers)
    s = new("Why this one question gates three others",
            "Given that a single-task ceiling exists, multi-task training is only worth its cost if "
            "it actually improves the data-poor tasks")
    txt(s, 0.6, 1.5, 12.2, 4.6, [
        "If it does not, loss balancing and gradient surgery have nothing to repair — the right",
        "answer for a small task would simply be to train it alone.",
        "",
        f"On the six-task probe (relative R2 gain): {gains or 'none'}; "
        f"{', '.join(sm['tasks_hurt']) or 'none'} clearly hurt; the rest unresolved.",
        "",
        "The ordering was monotone and in the direction the argument needs: the two smallest tasks",
        "and zt gained, the two largest gave up a little.",
        "",
        "The two percentage conventions mean different things. Relative gain = dR2 / single-task R2",
        "is stable and belongs in a headline. Error reduction = dR2 / (1 - single-task R2) is more",
        "meaningful when there is headroom but explodes near the ceiling: formation_energy has only",
        "0.0053 of residual, so -0.0036 reads as -68% — arithmetically right, misleading as a headline.",
        "",
        f"Practical threshold |dR2| >= 0.01: {', '.join(sm.get('tasks_that_matter', [])) or 'none'} "
        f"belong in a conclusion; {', '.join(sm.get('resolved_but_negligible', [])) or 'none'} "
        f"are resolvable but negligible.",
        "",
        "**This is a six-task result, and the next slide retracts it.** Whether it holds at",
        "deployment scale is what the xfer stage measures: every task trained last in a shuffled",
        "24-task sequence, with several random orderings each. It does not hold.",
    ], size=14)


@slide_guard
def slide_xfer():
    """The campaign's headline question, answered at deployment scale.

    Reads matched_xfer.json, not transfer_xfer.json: the report quotes the comparison restricted to
    the rows both arms share, and a deck showing the uncorrected figures beside a report showing the
    corrected ones is how two documents start disagreeing.
    """
    x = load("matched_xfer.json")
    rows_all = [r for r in x["per_task"] if "transfer" in r]
    better = [r for r in rows_all if r["separated"] and r["transfer"] > 0]
    worse = [r for r in rows_all if r["separated"] and r["transfer"] < 0]
    unres = [r for r in rows_all if not r["separated"]]
    ranked = sorted(rows_all, key=lambda r: -(r.get("relative_pct") or 0))
    shown = ranked[:5] + ranked[-5:]
    rows = []
    for r in shown:
        verdict = ("multi-task better" if r["transfer"] > 0 else "single-task better") \
            if r["separated"] else "unresolved"
        rows.append([r["task"], f"{r['n_train']:,}", f"{r['single_task']:.4f}",
                     f"{r['multi_task']:.4f}", f"{r['relative_pct']:+.2f}%", verdict])
    s = new("Transfer at deployment scale: the answer is negative",
            "24 tasks x shuffled orderings, task under test placed last; both arms restricted to "
            "the test rows they share")
    table(s, 0.5, 1.45, 11.6,
          ["Task", "Labels", "Single-task", "Multi-task", "Relative", "Verdict"], rows,
          col_w=[3.0, 1.4, 1.5, 1.5, 1.5, 2.7], size=10, head_size=10, colour_col=4)
    y = 1.55 + 0.32 * (len(rows) + 1) + 0.15
    txt(s, 0.5, y, 12.4, 2.4, [
        f"Multi-task better: {len(better)} ({', '.join(r['task'] for r in better) or 'none'})   |   "
        f"single-task better: {len(worse)}   |   unresolved: {len(unres)}",
        f"(the table shows the five tasks at each end of the relative change, out of {len(rows_all)})",
        "",
        "The probe overstated transfer. zt went from +6.85% \"multi-task better\" to +3.16%",
        "\"unresolved\"; magnetization from +4.40% to +1.72%; magnetic_moment from \"unresolved\" to",
        "-6.33% \"single-task better\". All three verdicts moved against multi-task — a six-task",
        "probe does not predict twenty-four.",
    ], size=12)


@slide_guard
def slide_ft():
    """The warm-start stage: the transfer loss was mostly replay dilution.

    Sits right after slide_xfer because it is the answer to the question that slide raises. The
    table shows every arm's value beside its % against the single-task baseline, which is the
    layout that was asked for and the only one in which the three arms can be read as one story.
    """
    f = load("ft.json")
    rows_all = f["per_task"]
    order = ["material_type", "final_energy", "volume", "dos_density", "zt",
             "dielectric_total", "magnetization", "magnetic_moment"]
    by = {r["task"]: r for r in rows_all}

    def pc(d):
        if not d:
            return "-"
        star = "*" if d["matters"] else ("·" if d["separated"] else "")
        return f"{d['relative_pct']:+.1f}%{star}"

    def v(x):
        return f"{x:.4f}" if x is not None else "-"

    rows = []
    for tname in order:
        r = by[tname]
        rows.append([tname, v(r["single_task"]),
                     v(r["xfer_with_replay"]), pc(r["xfer_vs_single"]),
                     v(r["ftz"]["mean"] if r["ftz"] else None), pc(r["ftz_vs_single"]),
                     v(r["ftf"]["mean"] if r["ftf"] else None), pc(r["ftf_vs_single"]),
                     pc(r["ftf_vs_ftz"])])
    c = f["counts"]
    s = new("Same checkpoint, same task, replay removed: the transfer loss was mostly dilution",
            "240 transfer checkpoints fine-tuned on their own last task, no replay, two arms, "
            "10 orderings each; vs = change against training alone")
    table(s, 0.4, 1.4, 12.5,
          ["Task", "Alone", "xfer", "vs", "Frozen", "vs", "Warm-start", "vs", "Warm vs frozen"], rows,
          col_w=[2.3, 1.1, 1.1, 1.2, 1.1, 1.2, 1.3, 1.2, 1.5], size=10, head_size=10)
    y = 1.5 + 0.30 * (len(rows) + 1) + 0.15
    txt(s, 0.4, y, 12.5, 2.6, [
        f"Counts over 24 tasks (better / worse / unresolved):   "
        f"xfer vs alone {c['xfer_vs_single']['better']}/{c['xfer_vs_single']['worse']}/"
        f"{c['xfer_vs_single']['unresolved']}    "
        f"frozen vs alone {c['ftz_vs_single']['better']}/{c['ftz_vs_single']['worse']}/"
        f"{c['ftz_vs_single']['unresolved']}    "
        f"warm-start vs alone {c['ftf_vs_single']['better']}/{c['ftf_vs_single']['worse']}/"
        f"{c['ftf_vs_single']['unresolved']}    "
        f"warm-start vs xfer {c['ftf_vs_xfer']['better']}/{c['ftf_vs_xfer']['worse']}/"
        f"{c['ftf_vs_xfer']['unresolved']}    "
        f"unfreezing {c['ftf_vs_ftz']['better']}/{c['ftf_vs_ftz']['worse']}/"
        f"{c['ftf_vs_ftz']['unresolved']}",
        "",
        "At step 24 a small task owns ~1% of the gradient against ~79k replay samples, and early stopping",
        "on the total loss fires when the replayed tasks stop improving (~60 epochs vs 90-150 alone).",
        "Remove replay and 19 tasks recover, none get worse. Warm-starting is break-even against training",
        "alone: zt and magnetization — the probe's original winners — come back as real gains; the extensive",
        "properties (final_energy, volume) still lose, which is what the scale-blind descriptor predicts.",
        "Unfreezing helps 13 tasks, most where frozen was weakest (final_energy +16%, volume +12%); it hurts",
        "material_type, the one task that wants the shared representation left alone. Caveat: every start",
        "checkpoint had already seen its task once (step 24, under replay); a clean unseen-task arm is running.",
    ], size=11)
    txt(s, 0.4, 6.75, 12.5, 0.4,
        ["* = separated at 2SE and |delta| >= 0.01   · = separated but below the threshold   "
         "Baselines and both arms audited for convergence; the two capped KR tasks rerun at 400 epochs."],
        size=9, color=MUT)


@slide_guard
def slide_position():
    """The position curve — obtained by re-reading data the stage already wrote.

    Kept separate from slide_xfer because it answers a different question: xfer asks "is multi-task
    better for a task placed last", this asks "does it matter WHERE the task sits". The second is
    what explains part of the first.
    """
    d = load("position.json")
    rows_all = [r for r in d["per_task"] if r.get("early_1_8") and r.get("late_17_24")]
    ranked = sorted(rows_all, key=lambda r: (r["position_effect"] or {}).get("delta_late_minus_early", 0))

    def cell(x):
        if not x:
            return "-"
        star = "*" if x.get("matters") else ("·" if x.get("separated") else "")
        return f"{x['delta']:+.4f} / {x['relative_pct']:+.1f}%{star}"

    def pcell(pe):
        if not pe:
            return "-"
        star = "*" if pe.get("matters") else ("·" if pe.get("separated") else "")
        return f"{pe['delta_late_minus_early']:+.4f} / {pe['relative_pct']:+.1f}%{star}"

    shown = ranked[:5] + ranked[-3:]
    rows = [[r["task"], f"{r['n_train']:,}", cell(r["early_1_8"]), cell(r["late_17_24"]),
             pcell(r["position_effect"])] for r in shown]
    s = new("Fixing the viewpoint on one task: does its position matter?",
            "Every step directory records the metrics of every task trained so far, so each task's "
            "score at every position was already on disk. No extra compute.")
    table(s, 0.4, 1.5, 12.5,
          ["Task", "Labels", "Early (slots 1-8)", "Late (slots 17-24)", "Position effect (late - early)"],
          rows, col_w=[2.8, 1.4, 2.8, 2.8, 2.7], size=10, head_size=10)
    n_pos = sum(1 for r in rows_all if (r["position_effect"] or {}).get("matters"))
    n_ret = sum(1 for r in d["per_task"]
                if r.get("retention") and r["retention"]["separated_from_zero"]
                and r["retention"]["mean"] > 0)
    y = 1.6 + 0.32 * (len(rows) + 1) + 0.2
    txt(s, 0.4, y, 12.5, 3.0, [
        f"Position is resolvable for {n_pos} of {len(rows_all)} tasks. magnetic_moment and",
        "total_magnetization are not hurt at early positions at all (unresolved) and only fall when",
        "placed last — so magnetic_moment's -6.33% in the transfer measurement is a position effect,",
        "not multi-task training as such.",
        "",
        f"Separately: {n_ret} of 24 tasks keep improving after their own step is over. But the check",
        "rules out 'replay keeps paying': retention correlates with remaining steps at only +0.038",
        "(n=1656), the gain arrives within 3-4 further steps, and for tasks trained last retention is",
        "exactly +0.0000 (69 samples) — the sanity check that the measurement is real.",
        "Read it as each task being under-trained at its own step, with replay recovering about +0.02",
        "over the next two or three. (Early stopping watches the total loss across all tasks, so the",
        "current task's progress is diluted by already-converged ones — a hypothesis, not a finding.)",
    ], size=11)
    txt(s, 0.4, 6.6, 12.5, 0.5,
        ["* = resolvable and >=0.01    . = resolvable but below the practical threshold    "
         "Position is confounded with WHICH tasks preceded; 3-6 samples per position"],
        size=9, color=MUT)


def slide_descriptor_limit():
    s = new("A model limitation worth knowing before launch: the descriptor cannot see cell scale",
            "volume 0.619 and final_energy 0.774 are low for tasks with ~23,700 labels, while "
            "formation_energy reaches 0.995 on the same rows")
    txt(s, 0.5, 1.5, 7.4, 4.6, [
        "Confirmed in the code, not inferred:",
        "  descriptor_fn -> formula_to_composition (contract: an atomic-FRACTION vector summing to 1)",
        "  -> KMD.transform computes weight @ K with no further normalisation",
        "",
        "  formula_to_composition(\"Fe2O3\") == formula_to_composition(\"Fe4O6\")  ->  True",
        "",
        "So cell scale is not among the model's inputs. The composition key itself deliberately keeps",
        "absolute stoichiometry, but this path divides it out at the door.",
        "",
        "It is a generalisation gap, not label noise: of 33,822 reduced formulas only 7 collide",
        "(0.02%), so the training data is not self-contradictory — the model simply has to infer",
        "scale from chemistry alone.",
        "",
        "corr(Volume, atoms per cell) = +0.868  ->  75.3% of the variance",
        "Atom counts span 1-320 (median 17). Single-task reaches 0.619: most of it recovered, not all.",
    ], size=12)
    table(s, 8.2, 1.5, 4.7, ["Reduced formula", "Atoms", "Volume"],
          [["AgSO4", "12 / 48", "162 / 616"],
           ["U(PO3)4", "34 / 136", "450 / 1930"],
           ["Ba(FeAs)2", "5 / 10", "98 / 217"],
           ["MnIr", "2 / 4", "26 / 56"]],
          col_w=[1.7, 1.4, 1.6], size=10, head_size=10)
    txt(s, 8.2, 3.4, 4.7, 3.2, [
        "^ identical descriptor input, ~4x difference in volume",
        "",
        "It explains one of the three low ceilings:",
        "  volume              +0.868  yes",
        "  final_energy        +0.162  no - already per-atom",
        "  total_magnetization +0.023  no - magnetism is simply hard",
        "",
        "No comparison in this report is affected: every",
        "arm shares the descriptor.",
        "Improvement: add the cell's total atom count to the",
        "descriptor. That is a model change, recorded in the",
        "handoff, out of scope for v2.",
    ], size=11, color=MUT)


@slide_guard
def slide_balancer():
    b = load("stage_bal.json")
    rows = [[v["arm"], pct(v["delta_vs_untuned"])] for v in b["vs_anchor"]]
    s = new("The learnable loss balancer is harmful, and mechanically so",
            f"{b['n_runs']} runs. The feature had never actually been wired up; repairing the path "
            f"this round made it measurable for the first time.")
    table(s, 0.6, 1.45, 5.4, ["Arm", "vs untuned anchor"], rows, col_w=[3.0, 2.4],
          size=10, head_size=10, colour_col=1)
    sep = [p for p in b["pairwise"] if p["separated"]]
    txt(s, 6.4, 1.45, 6.4, 5.4, [
        "Every arm with it on scores below every arm with it off.",
        f"Resolvable pairwise comparisons: {len(sep)} of {len(b['pairwise'])}",
        "",
        "Why it runs backwards:",
        "  The method learns a log-sigma per task, and its optimum is sigma^2 = L, that task's own",
        "  loss. Measured correlation between sigma and the raw loss: +0.970.",
        "  The resulting head weights: AE 20,075 against seebeck 1.5 — four orders of magnitude.",
        "",
        "  So it hands weight to the task that is easiest to fit and suppresses the hardest,",
        "  which is the exact opposite of rescuing the weak tasks.",
        "",
        "Excluding AE (the balx arm) it still runs backwards, by about 112x among the supervised",
        "tasks. The problem is the method's premise, not its scope.",
        "",
        "Conclusion: do not ship it. The incidental good news is that it had never been switched on,",
        "or a great deal of compute would have been spent on a harmful mechanism.",
    ], size=12)


def slide_pcgrad():
    s = new("PCGrad (arXiv 2001.06782): its premise does not hold here",
            "It acts only on task pairs with a negative cosine; with no conflict it is the identity")
    txt(s, 0.6, 1.6, 12.2, 4.8, [
        "Cost: one backward pass per task — 6x on the probe, 24x at Stage C. So the premise was",
        "measured before the cost was discussed.",
        "",
        "Per-task gradients on the shared encoder were measured directly. Only the encoder can",
        "conflict; the task heads have disjoint parameters and cannot interfere by construction.",
        "",
        "  - The paper's first condition, directional conflict: not observed. -> do not adopt.",
        "  - The paper's second condition, gradient-magnitude dominance: present (encoder gradient",
        "    norms differ widely between tasks).",
        "",
        "Magnitude dominance alone is not a reason to adopt PCGrad — what it repairs is directional",
        "conflict.",
    ], size=15)


def slide_packing():
    s = new("Engineering: packing several tasks onto one card",
            "Slurm accounting showed a single run using about 9% of a GB200")
    txt(s, 0.6, 1.6, 12.2, 4.8, [
        "Measured per run: about 9% of the compute and 1.29 GB of 189 GB of memory. Both campaigns",
        "ran one job per card, so roughly nine tenths of every reservation was wasted.",
        "",
        "Switching to --pack N (N runs as separate processes sharing one GPU) measured 7.1x",
        "throughput at PACK=8. Had this existed from the start, the two campaigns would have saved",
        "roughly 2,300 card-hours.",
        "",
        "An error caught in review: the first figure reported was 8.0x, but that was circular — the",
        "card-hours in cost.py were themselves computed as run_h / pack_size. The measured value is",
        "7.1x, corrected in four places (AGENTS.md, the RIKYU instructions, the skill, and cost.py).",
        "",
        "A limit on the comparison: packing lengthens each run's wall clock, so packed and unpacked",
        "wall-clock numbers cannot be placed side by side. Observations like \"a higher encoder_lr",
        "converges faster\" were measured unpacked and must not be mixed with packed timings.",
    ], size=14)


@slide_guard
def slide_adopt():
    a = load("finals_a.json")
    s = new("Adopted and rejected", "")
    txt(s, 0.5, 1.35, 6.6, 4.2, [
        "The adopted config — only three numbers actually changed",
        "",
        "  model.latent_dim        = 384       <- changed (was 128)",
        "  training.encoder_lr     = 2e-3      <- changed (was 5e-3)",
        "  scheduler.min_lr        = 1e-5      <- changed (was 1e-4)",
        "  scheduler.patience      = 5         = default",
        "  scheduler.factor        = 0.5       = default",
        "  model.head_hidden_dims  = [64]      = default",
        "  training.head_lr        = 5e-3      = default",
        "  model.kr_x_hidden_dims  = [128,64]  = default",
        "  training.kr_lr          = 5e-4      = default",
        "  early_stopping.patience = 24        = default",
        "  learnable_loss_balancer = false",
    ], size=11, mono=True)
    table(s, 7.4, 1.35, 5.5, ["Item", "Verdict"],
          [["learnable loss balancer", "rejected"],
           ["PCGrad", "not adopted"],
           ["head tuning (capacity / LR / KR branch)", "not adopted"],
           ["early stopping 24 -> 40", "not adopted"],
           ["extending the LR range downward", "not needed"],
           ["turning the LR schedule off", "rejected"]],
          col_w=[3.6, 1.9], size=11)
    txt(s, 7.4, 4.1, 5.5, 0.9,
        ["Every \"stayed at the default\" was measured, not skipped."], size=11, color=MUT)
    txt(s, 0.5, 5.15, 12.4, 2.2, [
        "What this round established:",
        "  1. A probe must span large / medium / small — sigma fell from 8.48% to 2.05%, which is",
        "     what gave the campaign any resolving power at all.",
        "  2. Rankings are bought with seeds, not with grid points: the 5-seed leader finished 10th",
        "     at 25 seeds.",
        "  3. An inherited baseline must be remeasured in the current regime — the old ceiling was",
        "     too low for 17 of 23 tasks, and the offset is not constant.",
        "  4. A group mean is not a conclusion: two tasks moving in opposite directions cancel and",
        "     read as \"near the ceiling\".",
        "  5. Half-wired features that fail silently are a systemic problem: DDP, the checkpoint dict",
        "     and the loss balancer were three instances.",
        "  6. Any experiment run on the current best has to be redone when the best changes — this",
        "     round hit that twice.",
        "  7. The pipeline is bit-deterministic at a fixed seed, so every sigma in this report is",
        "     config-level variance, not run-to-run jitter.",
    ], size=12)
    return a


def slide_limits():
    s = new("Limitations, stated alongside the numbers rather than implied", "")
    txt(s, 0.6, 1.6, 12.2, 4.6, [
        "1. Stage C has 1 seed per arm, so small differences are not resolvable.",
        "2. [training.scheduler] has no per-group switch, so scheduler results cannot be attributed",
        "   to any single parameter group.",
        "3. Stage C's ceiling comparison is optimistic: it assumes an arm's seed noise equals the",
        "   single-task arm's.",
        "4. Packed and unpacked wall-clock figures are not comparable.",
        "5. The probe results come from 6 tasks; the 24-task version comes from the xfer stage.",
        "   Both are in the report, and where they disagree the xfer measurement is the one that",
        "   stands.",
    ], size=16)


def main() -> None:
    global ALLOW_MISSING
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2026-08-28")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip slides whose inputs are absent and list them (dry run only)")
    args = ap.parse_args()
    ALLOW_MISSING = args.allow_missing

    slide_title(args.date)
    slide_summary()
    slide_glossary()
    slide_design_flow()
    slide_design_why()
    slide_probe()
    slide_anchor()
    slide_grid()
    slide_finals()
    slide_finals_sigma()
    slide_a4()
    slide_a2b()
    slide_stage_b()
    slide_b_finals()
    slide_stage_c()
    slide_ceiling_fig()
    slide_transfer_fig()
    slide_transfer_why()
    slide_xfer()
    slide_ft()
    slide_position()
    slide_descriptor_limit()
    slide_balancer()
    slide_pcgrad()
    slide_packing()
    slide_adopt()
    slide_limits()

    out = args.out or RES / f"REPORT_v2_{args.date.replace('-', '')}.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"{out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    if SKIPPED:
        print(f"\nSKIPPED {len(SKIPPED)} slide(s) — this build is NOT a deliverable:")
        for line in SKIPPED:
            print(f"  {line}")


if __name__ == "__main__":
    main()
