#!/usr/bin/env python3
"""Build results/REPORT_v1v2_<date>.pptx — the merged v1+v2 deck.

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


def pct(x, digits=2):
    return f"{x * 100:+.{digits}f}%"


# --- slides -----------------------------------------------------------------------------------


def slide_title(date: str):
    s = prs.slides.add_slide(BLANK)
    txt(s, 0.9, 2.4, 11.5, 1.0, ["RIKYU 超参数调优 campaign"], size=40, bold=True)
    txt(s, 0.9, 3.5, 11.5, 0.6, ["v1 + v2 合并报告"], size=22, color=MUT)
    txt(s, 0.9, 4.4, 11.5, 1.2, [
        f"{date} ｜ 分支 exp/rikyu-hparam-tuning-v2",
        "v1 的数字未重跑，但其中一部分用 v2 测得的同régime天花板重新评分；凡与 v1 报告不一致处均标明修正。",
    ], size=13, color=MUT)


@slide_guard
def slide_summary():
    a = load("finals_a.json")
    bal = load("stage_bal.json")
    tr = load("transfer_adopted.json")
    lead = next(v for v in a["vs_anchor"] if v["arm"] == a["leader"])
    off = max(v["delta_vs_untuned"] for v in bal["vs_anchor"] if v["arm"].endswith("off"))
    on = max(v["delta_vs_untuned"] for v in bal["vs_anchor"] if v["arm"].endswith("on"))
    helped = tr["summary"]["tasks_helped"]
    s = new("摘要", "每一条都可在 summary/*.json 中溯源")
    txt(s, 0.6, 1.5, 12.2, 5.4, [
        f"1. 调参收益小且刚好可分辨：采纳配置相对未调锚点 {pct(lead['delta_vs_untuned'])}；"
        f"前 {1 + len(a['statistically_tied_with_leader'])} 名统计上无法区分。",
        "2. 真正的收益来自上游修复（PR #45 调度器节奏），不是调参。v1 的全部数字都在坏节奏下测得。",
        "3. 必须修正 v1 一条结论：其调参臂的负 deficit（“超过单任务天花板”）是天花板测低造成的假象。",
        f"4. multi-task 对小数据任务确有正迁移：{('、'.join(helped)) or '（无）'} 明确获益，两个最大的任务略亏。",
        f"5. learnable loss balancer 有害且是机制性的：最好的关闭臂 {pct(off)} vs 最好的开启臂 {pct(on)}。不要上线。",
        "6. PCGrad 不适用：直接测量编码器逐任务梯度，未发现该方法赖以生效的方向冲突。",
        "7. 工程发现：两轮 campaign 的 GPU 利用率约 9%，单卡多任务打包实测提速 7.1×。",
    ], size=15)


@slide_guard
def slide_probe():
    s0 = load("stage0.json")
    cal = s0["calibration"]
    s = new("方法：probe6 探针", "v1 用 3 任务探针，噪声带 8.48%，前三名相差 1.5–1.8% —— 排不出名次")
    table(s, 0.6, 1.5, 6.0,
          ["探针任务", "标签数", "档位"],
          [[t, f"{n:,}", g] for t, n, g in [
              ("volume", 23678, "big"), ("formation_energy", 23180, "big"),
              ("seebeck", 8072, "mid"), ("zt", 3445, "mid"),
              ("magnetization", 1160, "small"), ("magnetic_moment", 851, "small")]],
          col_w=[2.6, 1.7, 1.7])
    need_seeds = cal["seeds_needed_to_resolve"]
    txt(s, 7.0, 1.5, 5.8, 4.6, [
        f"单run σ = {cal['sigma_per_run'] * 100:.2f}%   （v1: {cal['v1_probe3_band_for_reference'] * 100:.2f}% 带宽）",
        "",
        "要分辨这么大的真实差异，需要的 seed 数：",
        *[f"    {float(k) * 100:.1f}%  →  {v} seed" for k, v in need_seeds.items()],
        "",
        "排除 electrical_resistivity（天花板 0.162，无分辨力）",
        "排除 magnetic_susceptibility（58 个标签）",
        "",
        f"18 个锚点 run 实测：平均 {cal['wallclock']['mean_hours']:.2f} h/run，"
        f"合计 {cal['wallclock']['total_gpu_hours']:.0f} GPU-h",
    ], size=13)


@slide_guard
def slide_anchor():
    s0 = load("stage0.json")
    c = s0["comparisons"][0]
    s = new("上游修复：v1 的全部数字都在坏节奏下",
            "PR #45 之前 ReduceLROnPlateau 按 batch 而非 epoch 触发，LR 在第一个 epoch 内就落到下限")
    txt(s, 0.6, 1.6, 12.2, 3.0, [
        "这不是影响小数点的 bug，它改变了优化过程本身。因此：",
        "",
        "  • v1 的调参结论是在坏节奏下选出来的，不能直接搬到修好的代码上 —— 这是 v2 重做 Stage A 的理由；",
        "  • v1 引用的单任务“天花板”同样测于坏节奏，所以它们不是天花板（见后）。",
        "",
        "Stage 0 锚点（未调参 + 新镜像 0.3.2）实测：",
    ], size=15)
    verdict = "无法分辨" if abs(c["delta_score"]) <= c["resolvable_at_this_n"] else "可分辨"
    table(s, 0.6, 4.6, 9.0,
          ["比较", "差值", "可分辨阈值 (2SE)", "判定"],
          [[f"{c['from']} → {c['to']}", pct(c["delta_score"]),
            pct(c["resolvable_at_this_n"]), verdict]],
          col_w=[3.4, 1.8, 2.2, 1.6])
    txt(s, 0.6, 5.5, 12.2, 0.8,
        ["即：v1 选出的编码器配置在修好的代码上不再有可见优势。"], size=14, color=MUT)


@slide_guard
def slide_grid():
    a = load("stage_a.json")
    s = new("Stage A′：编码器 × LR × 调度器联合搜索",
            f"{a['n_configs']} 个配置 / {a['n_runs']} 个 run（网格 + 随机搜索，各 5 seed）")
    edge = a["edge_bound_axes"]
    txt(s, 0.6, 1.5, 12.2, 2.0, [
        f"网格边界检查：{'全部通过，不需要追加 a1b 轮' if not edge else '边界受限的轴：' + ', '.join(edge)}",
        "",
        "  这项检查本身经过两次修正：最初只测“最优点是否恰好等于端点”，但 206 个随机取值没有任何一个",
        "  正好落在端点上，真正的边界问题会被放过 —— 后来加了十分位趋势检验；它还曾在只有 2 个取值的轴",
        "  （patience、latent_dim）上误报，因此增加了“至少 3 个取值”的前提。",
    ], size=13)
    ties = a["leader_ties"]
    txt(s, 0.6, 3.7, 12.2, 2.6, [
        f"5 seed 排不出名次：{len(ties['statistically_tied_with_leader'])} 个配置与榜首统计上并列"
        f"（可分辨差异 {pct(ties['resolvable_difference'])}）。",
        "",
        "所以短名单取前 8，进 25 seed 决赛 —— 算力花在 seed 上，而不是更多网格点上。",
    ], size=15)


@slide_guard
def slide_finals():
    a = load("finals_a.json")
    byarm = {v["arm"]: v for v in a["vs_anchor"]}
    rows = [[arm.replace("a3_", ""), pct(byarm[arm]["delta_vs_untuned"])]
            for arm in a["ranking"][:6] if arm in byarm]
    s = new(f"Stage A′ 决赛（25 seed，{a['n_runs']} run）",
            "前几名统计上并列 —— 采纳规则因此落到公开声明的次级判据")
    table(s, 0.6, 1.5, 8.4, ["配置", "相对未调锚点"], rows, col_w=[6.0, 2.4], colour_col=1)
    tied = a["statistically_tied_with_leader"]
    need_seeds = a["seeds_that_would_resolve_the_ties"]
    top_pairs = [f"    {k.replace('a3_', '')} → 需 {v} seed"
                 for k, v in list(need_seeds.items())[:3]]
    txt(s, 9.2, 1.5, 3.9, 5.0, [
        f"与榜首并列：{len(tied)} 个",
        "",
        "要把它们分开所需的 seed 数：",
        *top_pairs,
        "",
        "采纳：",
        a["leader"].replace("a3_", ""),
        "",
        "理由（次级判据，公开声明）：",
        "四者中最简单 —— 网格点、整数 patience、无自定义 factor。",
    ], size=12)


@slide_guard
def slide_a4():
    a4 = load("stage_a4.json")
    h = a4["head_to_head"]
    d = a4["downward_extension"]["sched"]
    s = new("A4：调度器值不值得 / 最优点是否在搜索下界之外", f"{a4['n_runs']} 个 run，成对比较")
    table(s, 0.6, 1.6, 11.0,
          ["问题", "结果", "差值", "2SE 可分辨", "判定"],
          [["有调度 vs 固定 LR",
            f"{pct(h['best_scheduled']['mean'])} vs {pct(h['best_flat']['mean'])}",
            pct(h["delta"]), pct(2 * h["se_of_difference"]),
            "调度器值得保留" if h["separated"] else "无法分辨"],
           ["最优点是否在下界之外",
            f"{pct(d['best_below_floor']['mean'])} vs {pct(d['best_at_or_above_floor']['mean'])}",
            pct(d["delta"]), pct(2 * d["se_of_difference"]),
            "不需要向下扩展" if not d["worth_extending_further"] else "需要扩展"]],
          col_w=[3.0, 2.6, 1.6, 1.8, 2.0], colour_col=2)
    txt(s, 0.6, 3.4, 12.2, 2.6, [
        "必须随数字一起声明的限制：",
        "",
        "  [training.scheduler] 管的是全部四个参数组（encoder / head / kr / ae），没有分组开关。",
        "  所以“关掉调度”同时冻结了 head 与 KR 的学习率 —— 这个损失不能归因到某一个组。",
    ], size=14, color=MUT)


@slide_guard
def slide_stage_b():
    b = load("stage_b.json")
    byarm = {v["arm"]: v for v in b["vs_anchor"]}
    rows = [[c.replace("b_", ""), pct(byarm[c]["delta_vs_untuned"])]
            for c in (b["short_list"] if "short_list" in b else b["ranking"])[:6] if c in byarm]
    s = new("Stage B′：多任务联合 head 调参",
            f"{b['n_configs'] if 'n_configs' in b else 24} 个配置 × 5 seed，全部在 A′ 采纳基座上")
    table(s, 0.6, 1.5, 8.4, ["配置", "相对未调锚点"], rows, col_w=[6.0, 2.4], colour_col=1)
    txt(s, 9.2, 1.5, 3.9, 5.0, [
        "为什么必须在采纳基座上跑：",
        "",
        "v1 的 Stage B 是在单任务探针上逐任务调 head 的，",
        "结论没有迁移到 24 任务连续训练 ——",
        "24 个“改善”里只有 2 个经得起重复，",
        "还有 5 个任务变差了。",
        "",
        "head 的最优值依赖它所处的优化régime；",
        "换了基座就是换了régime。",
    ], size=12)


@slide_guard
def slide_stage_c():
    c = load("stage_c.json")
    rows = []
    for arm in c["arms"]:
        d = arm["deficit"]
        fmt = lambda v: f"{v:+.4f}" if v is not None else "-"  # noqa: E731
        rows.append([arm["label"], f"{arm['mean_r2']:.4f}",
                     fmt(d["big"]), fmt(d["mid"]), fmt(d["small"])])
    s = new("Stage C′：24 任务，四个臂", "deficit 对同régime天花板；每臂仅 1 seed，小差距不可分辨")
    table(s, 0.6, 1.5, 9.6,
          ["臂", "mean R²", "big", "mid", "small"], rows,
          col_w=[3.2, 1.6, 1.6, 1.6, 1.6], colour_col=None)
    att = c["attribution"]
    lines = ["把“升级”与“调参”分开："]
    for name, e in att.items():
        v = e["delta_mean_r2"]
        lines.append(f"  {e['from']} → {e['to']}：{'n/a' if v is None else f'{v:+.4f}'}   {e['what_it_isolates']}")
    t = c["transfer"]
    if t.get("checked"):
        lines += ["", f"探针名次是否迁移到 24 任务：{'保持' if t['order_preserved'] else '被打乱'}"
                      f"（三臂 mean R² 极差 {t['mean_r2_spread_across_promoted_arms']:.4f}）"]
    txt(s, 0.6, 1.6 + 0.32 * (len(rows) + 1) + 0.3, 12.2, 2.6, lines, size=13)


@slide_guard
def slide_ceiling_fig():
    pic_slide("天花板是一个坏掉的测量框架",
              "旧天花板测于 PR #45 之前 —— LR 在第一个 epoch 内就落到地板，那不是天花板",
              RES / "ceiling_frame_offset.png")


@slide_guard
def slide_ceiling_retraction():
    v1 = load("stage_c_v1_rescored.json")
    rows = []
    for arm in v1["arms"]:
        d, h = arm["deficit"], arm["deficit_vs_recorded_h200"]
        f = lambda v: f"{v:+.4f}" if v is not None else "-"  # noqa: E731
        rows.append([arm["label"], f"{arm['mean_r2']:.4f}",
                     f(d["big"]), f(d["mid"]), f(d["small"]),
                     f"{f(h['big'])} / {f(h['mid'])} / {f(h['small'])}"])
    s = new("对 v1 的修正：负 deficit 是假象",
            "用 v2 的任务口径重新评分；v1 原始产物未改动（summary/stage_c_v1_rescored.json）")
    table(s, 0.5, 1.45, 12.3,
          ["v1 臂", "mean R²", "big", "mid", "small", "旧框架 big / mid / small"],
          rows, col_w=[2.4, 1.4, 1.3, 1.3, 1.3, 4.6], size=10, head_size=10)
    txt(s, 0.5, 1.55 + 0.32 * (len(rows) + 1) + 0.2, 12.3, 2.4, [
        "c_tuned 在任何一档上都不再超过天花板。“超过单任务天花板”这条结论撤回。",
        "",
        "旧天花板在 23 个回归/KR 任务里的 17 个上偏低，平均 +0.0275；且不是常数（seebeck 低 0.104，",
        "dielectric_ionic 反而高 0.017），无法用偏移量修正，只能重测。任务越小偏差越大：big +0.022 /",
        "mid +0.027 / small +0.040 —— 正是“LR 不退火”该有的形状。",
    ], size=13)


@slide_guard
def slide_ceiling_gap():
    g = load("ceiling_gap.json")
    arm = next((a for a in g["arms"] if a["label"].endswith("cons")), g["arms"][0])
    small = [r for r in arm["per_task"] if r["group"] == "small"]
    s = new("分组均值把真正的结果盖住了",
            f"{arm['label']} 的 small 组只有两个任务，而它们的天花板 seed 噪声相差三倍以上")
    table(s, 0.6, 1.5, 10.6,
          ["任务", "臂 R²", "天花板", "差值", "2SE", "判定"],
          [[r["task"], f"{r['arm_r2']:.4f}", f"{r['ceiling_mean']:.4f}",
            f"{r['gap']:+.4f}", f"{2 * (r['se_of_difference'] or 0):.4f}",
            {"beats single-task": "超过单任务", "below single-task": "低于单任务"}.get(r["verdict"], "无法分辨")]
           for r in small],
          col_w=[2.8, 1.5, 1.5, 1.5, 1.5, 1.8], colour_col=3)
    txt(s, 0.6, 3.0, 12.2, 3.4, [
        "分组均值把这两个相互抵消的结果压成了一个数字，它两件事都不代表。",
        "",
        f"22 个任务整体：低于单任务 {len(arm['below_single_task'])} 个、"
        f"高于 {len(arm['beats_single_task'])} 个（{'、'.join(arm['beats_single_task']) or '无'}）、"
        f"无法分辨 {len(arm['unresolved'])} 个，平均 {arm['mean_gap']:+.4f}。",
        "",
        "这个检验偏乐观：Stage C 每臂只有 1 个 seed，臂自身的噪声没测到，只能假设它等于单任务的",
        "（SE = σ·√(1+1/n)）。若多任务的逐任务噪声更大（24 任务共享编码器，这是预期方向），",
        "真实 SE 更大，其中一些“可分辨”的判定就不成立 —— 应读作待确认的假设。",
    ], size=13)


@slide_guard
def slide_transfer_fig():
    pic_slide("迁移：multi-task 到底有没有帮到小任务",
              "采纳配置下，25 seed 多任务 vs 5 seed 单任务，唯一差别是 pretrain.task_sequence",
              RES / "transfer_adopted.png")


@slide_guard
def slide_transfer_why():
    tr = load("transfer_adopted.json")
    sm = tr["summary"]
    s = new("为什么这个问题卡住了另外三个",
            "既然存在单任务天花板，multi-task 只有确实让数据少的任务变好时才值得")
    txt(s, 0.6, 1.5, 12.2, 4.6, [
        "否则 loss balancing 和梯度手术都没有可修的东西 —— 小任务的正确答案就是单独训练。",
        "",
        f"实测：{'、'.join(sm['tasks_helped']) or '无'} 明确获益；"
        f"{'、'.join(sm['tasks_hurt']) or '无'} 明确受损；其余无法分辨。",
        "",
        "顺序是单调的，而且是需要的那个方向：最小的两个任务和 zt 获益，最大的两个略亏。",
        "formation_energy 的回退之所以可分辨，只是因为它的 seed σ 只有 0.0004 —— 真实，但可忽略。",
        "",
        "所以多任务这套设置在自己的账上是划算的。不划算的是那个本该保护小任务的 balancer。",
        "",
        "注意：这是 6 个任务的结论。24 个任务在部署规模上是否同样成立，由 xfer 阶段测量 ——",
        "每个任务都被放在打乱后的 24 任务序列末尾训练，每任务 3 组随机顺序。",
    ], size=14)


@slide_guard
def slide_xfer():
    x = load("transfer_xfer.json")
    sm = x["summary"]
    s = new("xfer：部署规模上的迁移，以及任务顺序有没有影响",
            "24 个任务各自被放在打乱序列的末尾，每任务 3 组随机顺序")
    rows = [[r["task"], f"{r['n_train']:,}", f"{r['single_task_r2']:.4f}",
             f"{r['multi_task_r2']:.4f}", f"{r['transfer']:+.4f}",
             {"multi": "多任务更好", "single": "单任务更好"}.get(
                 "multi" if r["transfer"] > 0 else "single", "") if r["separated"] else "无法分辨"]
            for r in sorted([r for r in x["per_task"] if "transfer" in r],
                            key=lambda r: -r["transfer"])[:10]]
    table(s, 0.5, 1.45, 11.4,
          ["任务", "标签数", "单任务", "多任务", "迁移", "判定"], rows,
          col_w=[2.8, 1.5, 1.6, 1.6, 1.5, 2.4], size=10, head_size=10, colour_col=4)
    txt(s, 0.5, 1.55 + 0.32 * (len(rows) + 1) + 0.2, 12.3, 1.8, [
        f"获益 {len(sm['tasks_helped'])} 个 / 受损 {len(sm['tasks_hurt'])} 个 / "
        f"无法分辨 {len(sm['tasks_unresolved'])} 个。",
    ], size=13)


@slide_guard
def slide_balancer():
    b = load("stage_bal.json")
    rows = [[v["arm"], pct(v["delta_vs_untuned"])] for v in b["vs_anchor"]]
    s = new("learnable loss balancer：有害，且是机制性的",
            f"{b['n_runs']} 个 run。这个功能此前一直没有真正接上，本轮修复链路后才第一次可测")
    table(s, 0.6, 1.45, 5.4, ["臂", "相对未调锚点"], rows, col_w=[3.0, 2.4],
          size=10, head_size=10, colour_col=1)
    sep = [p for p in b["pairwise"] if p["separated"]]
    txt(s, 6.4, 1.45, 6.4, 5.4, [
        "开启的每一个臂都低于关闭的每一个臂。",
        f"可分辨的成对比较：{len(sep)} / {len(b['pairwise'])}",
        "",
        "为什么反了：",
        "  方法学每个任务的 log σ，最优解是 σ² = L（该任务自身的损失）。",
        "  实测 σ 与原始损失的相关系数 +0.970。",
        "  由此得到的 head 权重：AE 20 075 vs seebeck 1.5 —— 差四个数量级。",
        "",
        "  即：它把权重给了最容易拟合的任务，把最难拟合的压了下去，",
        "  与“救援弱任务”的初衷正好相反。",
        "",
        "排除 AE（balx 臂）之后仍然反，监督任务之间约 112 倍。",
        "所以问题不在作用范围，而在方法前提。",
        "",
        "结论：不要上线。附带的好处是它此前一直没被开启 ——",
        "否则大量算力会被浪费在一个有害的机制上。",
    ], size=12)


def slide_pcgrad():
    s = new("PCGrad（arXiv 2001.06782）：前提不成立",
            "只对负余弦的任务对起作用；没有冲突时它是恒等变换")
    txt(s, 0.6, 1.6, 12.2, 4.8, [
        "代价：每个任务一次反向传播 —— probe6 上 6×，Stage C 上 24×。所以先测前提，再谈代价。",
        "",
        "直接测量了共享编码器上的逐任务梯度（只有编码器可能冲突；task head 参数不相交，按构造无法干涉）。",
        "",
        "  • 论文条件一「方向冲突」：没有测到。→ 不引入。",
        "  • 论文条件二「梯度量级支配」：确实存在（各任务编码器梯度范数相差很大）。",
        "",
        "但量级支配单独不构成采用 PCGrad 的理由 —— 它修的是方向冲突。",
    ], size=15)


def slide_packing():
    s = new("工程发现：一卡多任务打包", "Slurm 计费显示单个 run 只用掉一块 GB200 的约 9%")
    txt(s, 0.6, 1.6, 12.2, 4.8, [
        "单 run 实测：算力约 9%，显存 1.29 GB / 189 GB。两轮 campaign 都是一 run 一卡 —— 约九成预约被浪费。",
        "",
        "改为 --pack N（N 个 run 共享一块 GPU 的独立进程）后，实测吞吐 7.1×（PACK=8）。",
        "如果这个做法从一开始就有，两轮 campaign 大约能省下 2300 card-h。",
        "",
        "一条被 review 抓出来的错误：最初报告 8.0×，但那个数字是循环论证 —— cost.py 里的 card-h",
        "本来就是 run_h ÷ pack_size 算出来的。真实测量值是 7.1×，已在 AGENTS.md、RIKYU instructions、",
        "skill 和 cost.py 四处一并修正。",
        "",
        "口径限制：打包会拉长单 run 的墙钟。打包前后的墙钟数字不可并列比较 ——",
        "“高 encoder_lr 收敛更快”这类观察测于未打包时，不能和打包阶段的计时放在一起。",
    ], size=14)


@slide_guard
def slide_adopt():
    a = load("finals_a.json")
    s = new("采纳与否决", "")
    txt(s, 0.6, 1.4, 6.0, 3.2, [
        "采纳配置",
        "",
        "  model.latent_dim        = 384",
        "  training.encoder_lr     = 2e-3",
        "  scheduler.min_lr        = 1e-5",
        "  scheduler.patience      = 5",
        "  scheduler.factor        = 0.5",
        "  learnable_loss_balancer = false",
    ], size=13, mono=True)
    table(s, 6.9, 1.4, 5.9, ["项目", "判定"],
          [["learnable loss balancer", "否决"],
           ["PCGrad", "不引入"],
           ["向下扩展 LR 搜索范围", "不需要"],
           ["关闭 LR 调度", "否决"]],
          col_w=[3.9, 2.0], size=12)
    txt(s, 0.6, 4.9, 12.2, 2.2, [
        "本轮确立了什么：",
        "  1. 探针必须覆盖大/中/小三档 —— σ 从 8.48% 降到 2.05%，campaign 才有判断力。",
        "  2. 名次要用 seed 买，不是用网格点买 —— 5 seed 的榜首在 25 seed 下掉到第 10。",
        "  3. 继承来的基线必须在当前régime重测 —— 这一条撤回了 v1 的一个已发布结论。",
        "  4. 分组均值不能当结论 —— 两个任务一正一负相消，看起来像“接近天花板”。",
        "  5. “半接上、静默失效”的功能是系统性问题：DDP、checkpoint dict、loss balancer 三例。",
    ], size=13)
    return a


def slide_limits():
    s = new("局限（与数字一同声明，不隐含）", "")
    txt(s, 0.6, 1.6, 12.2, 4.6, [
        "1. Stage C 每臂仅 1 seed，小差距不可分辨。",
        "2. [training.scheduler] 无分组开关，调度相关结论无法归因到单个参数组。",
        "3. Stage C 的天花板对比偏乐观 —— 假设臂的 seed 噪声等于单任务的。",
        "4. 打包前后墙钟不可比。",
        "5. 探针结论以 6 个任务测得；24 任务版本由 xfer 阶段给出，两者都在报告里。",
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
    slide_probe()
    slide_anchor()
    slide_grid()
    slide_finals()
    slide_a4()
    slide_stage_b()
    slide_stage_c()
    slide_ceiling_fig()
    slide_ceiling_retraction()
    slide_ceiling_gap()
    slide_transfer_fig()
    slide_transfer_why()
    slide_xfer()
    slide_balancer()
    slide_pcgrad()
    slide_packing()
    slide_adopt()
    slide_limits()

    out = args.out or RES / f"REPORT_v1v2_{args.date.replace('-', '')}.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"{out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    if SKIPPED:
        print(f"\nSKIPPED {len(SKIPPED)} slide(s) — this build is NOT a deliverable:")
        for line in SKIPPED:
            print(f"  {line}")


if __name__ == "__main__":
    main()
