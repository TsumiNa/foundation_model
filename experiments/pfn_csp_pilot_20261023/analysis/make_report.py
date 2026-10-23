"""Render results/report.html from results/summary.json and the appendix tables (plan section 5).

uv run python analysis/make_report.py
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
RES = EXP / "results"
APPX = RES / "appendix"
CLASSES = ("AB", "ABX3")
CLS_NAME = {"AB": "AB", "ABX3": "ABX<sub>3</sub>"}
COND_NAME = {
    "C1": "C1 global supervised",
    "C2": "C2 same-family context",
    "C3": "C3 other-family context",
    "C4": "C4 nearest-neighbour context",
}
SERIES = {"C1": 1, "C2": 2, "C3": 3, "C4": 4}
K_A = (5, 10, 20, 50)
K_B = (5, 10, 20)

S = json.loads((RES / "summary.json").read_text())
B = json.loads((APPX / "build_summary.json").read_text())
FIG = [0]
TAB = [0]


def esc(x) -> str:
    return html.escape(str(x))


def pp(x, signed=True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x * 100:+.1f}" if signed else f"{x * 100:.1f}"


def pct(x) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x * 100:.1f}%"


def fig_head(title: str) -> str:
    FIG[0] += 1
    return f'<p class="fig-head"><span>Figure {FIG[0]}</span> {title}</p>'


def tab_head(title: str) -> str:
    TAB[0] += 1
    return f'<p class="fig-head"><span>Table {TAB[0]}</span> {title}</p>'


def caption(text: str) -> str:
    return f'<div class="caption"><span class="cap-label">Reading</span>{text}</div>'


def combo(c: str, eng: str = "tabpfn") -> dict:
    return S["combos"].get(c, {}).get(eng, {})


# ------------------------------------------------------------------ charts (inline SVG)
def learning_curve(cls: str) -> str:
    t2 = pd.DataFrame(combo(f"{cls}_L2_P1")["t2"])
    w, h, ml, mr, mt, mb = 560, 300, 52, 150, 16, 40
    pw, ph = w - ml - mr, h - mt - mb
    lo = max(0.0, math.floor((t2["top1_lo"].min() - 0.03) * 10) / 10)
    hi = min(1.0, math.ceil((t2["top1_hi"].max() + 0.03) * 10) / 10)
    xs = {k: ml + pw * (math.log(k) - math.log(5)) / (math.log(50) - math.log(5)) for k in K_A}

    def y(v):
        return mt + ph * (1 - (v - lo) / (hi - lo))

    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Protocol A learning curves, {cls}">']
    tick = lo
    while tick <= hi + 1e-9:
        out.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>')
        out.append(f'<text class="tick" x="{ml - 8}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick * 100:.0f}</text>')
        tick = round(tick + 0.1, 10)
    for k, x in xs.items():
        out.append(f'<text class="tick" x="{x:.1f}" y="{mt + ph + 18}" text-anchor="middle">{k}</text>')
    out.append(
        f'<text class="axis" x="{ml + pw / 2}" y="{h - 4}" text-anchor="middle">context size k (log scale)</text>'
    )
    out.append(
        f'<text class="axis" transform="translate(14 {mt + ph / 2}) rotate(-90)" text-anchor="middle">Top-1 accuracy (%)</text>'
    )
    ends = []
    for cond in ("C1", "C2", "C3", "C4"):
        d = t2[t2["condition"] == cond].sort_values("k")
        if d.empty:
            continue
        s = SERIES[cond]
        pts = [(xs[k], y(v)) for k, v in zip(d["k"], d["top1"])]
        if cond != "C1":
            band = [(xs[k], y(v)) for k, v in zip(d["k"], d["top1_hi"])] + [
                (xs[k], y(v)) for k, v in zip(d["k"][::-1], d["top1_lo"][::-1])
            ]
            out.append(f'<polygon class="band s{s}" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in band)}"/>')
        dash = ' stroke-dasharray="6 4"' if cond == "C1" else ""
        out.append(f'<polyline class="line s{s}"{dash} points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}"/>')
        for (a, b), (_, r) in zip(pts, d.iterrows()):
            tip = f"{COND_NAME[cond]}, k = {r['k']}: {r['top1'] * 100:.1f}% (95% CI {r['top1_lo'] * 100:.1f}–{r['top1_hi'] * 100:.1f})"
            out.append(f'<circle class="dot s{s}" cx="{a:.1f}" cy="{b:.1f}" r="4"><title>{esc(tip)}</title></circle>')
        ends.append([pts[-1][1], cond, d["top1"].iloc[-1]])
    ends.sort()
    for i in range(1, len(ends)):  # keep direct labels 13 px apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for yy, cond, v in ends:
        out.append(f'<text class="lab" x="{ml + pw + 10}" y="{yy + 4:.1f}">{cond} {v * 100:.0f}%</text>')
    out.append("</svg>")
    return "".join(out)


def family_bars(cls: str) -> str:
    t4 = pd.DataFrame(combo(f"{cls}_L2_P1")["t4"])
    t4 = t4.sort_values("n", ascending=False)
    n = len(t4)
    w, mt, mb, ml, mr = 560, 12, 36, 150, 20
    row = 34
    h = mt + mb + row * n
    pw = w - ml - mr

    def x(v):
        return ml + pw * v

    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Protocol B per held-out family, {cls}">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{mt}" y2="{mt + row * n}"/>')
        out.append(
            f'<text class="tick" x="{x(t):.1f}" y="{mt + row * n + 16}" text-anchor="middle">{t * 100:.0f}</text>'
        )
    out.append(
        f'<text class="axis" x="{ml + pw / 2}" y="{h - 4}" text-anchor="middle">Top-1 accuracy on the held-out family (%)</text>'
    )
    for i, (_, r) in enumerate(t4.iterrows()):
        y0 = mt + i * row
        out.append(
            f'<text class="tick" x="{ml - 8}" y="{y0 + row / 2 + 4}" text-anchor="end">{esc(r["family"])} ({r["n"]})</text>'
        )
        for j, (key, s, name) in enumerate(
            (("C1_B", 1, "C1 trained on the other families"), ("C2_B_k10", 2, "C2 with 10 same-family samples"))
        ):
            v = r.get(key)
            if v is None or not math.isfinite(v):
                continue
            yb = y0 + 5 + j * 12
            out.append(
                f'<rect class="bar s{s}" x="{ml}" y="{yb}" width="{max(x(v) - ml, 1):.1f}" height="10" rx="2"><title>{esc(f"{name}: {v * 100:.1f}%")}</title></rect>'
            )
        ca = r.get("C1_A")
        if ca is not None and math.isfinite(ca):
            out.append(
                f'<line class="ref" x1="{x(ca):.1f}" x2="{x(ca):.1f}" y1="{y0 + 2}" y2="{y0 + 29}"><title>{esc(f"C1 protocol A (family seen in training): {ca * 100:.1f}%")}</title></line>'
            )
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ tables
def t1() -> str:
    rows = []
    for c in CLASSES:
        b = B[c]
        rows.append(
            f"<tr><th>{CLS_NAME[c]}</th><td>{b['N']}</td><td>{b['L1']['K']} / {b['L2']['K']}</td>"
            f"<td>{pct(b['L1']['other_share'])} / {pct(b['L2']['other_share'])}</td>"
            f"<td>{b['P1']['n_families']} / {b['P2']['n_families']}</td>"
            f"<td>{b['P1']['min']}–{b['P1']['max']}</td><td>{b['P2']['min']}–{b['P2']['max']}</td>"
            f"<td>{pct(b['goldschmidt_nan_share']) if b.get('goldschmidt_nan_share') is not None else 'n/a'}</td></tr>"
        )
    return (
        tab_head("Data overview (T1)")
        + '<div class="tw"><table><thead><tr><th>class</th><th>N</th><th>K (L1 / L2)</th>'
        "<th><code>other</code> share (L1 / L2)</th><th>families (P1 / P2)</th><th>P1 size range</th>"
        "<th>P2 size range</th><th>Goldschmidt t missing</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def t2(cls: str) -> str:
    d = pd.DataFrame(combo(f"{cls}_L2_P1")["t2"])
    body = []
    for k in K_A:
        for cond in ("C1", "C2", "C3", "C4"):
            r = d[(d["k"] == k) & (d["condition"] == cond)]
            if r.empty:
                continue
            r = r.iloc[0]
            body.append(
                f"<tr><td>{k}</td><td>{cond}</td><td>{pp(r['top1'], False)} <small>[{pp(r['top1_lo'], False)}, {pp(r['top1_hi'], False)}]</small></td>"
                f"<td>{pp(r['top3'], False)}</td><td>{r['log_loss']:.2f}</td><td>{pp(r['macro_f1'], False)}</td>"
                f"<td>{r['auroc']:.2f}</td><td>{pct(r['coverage'])}</td><td>{pct(r['query_retention'])}</td></tr>"
            )
    return (
        '<div class="tw"><table><thead><tr><th>k</th><th>condition</th><th>Top-1 % [95% CI]</th><th>Top-3 %</th>'
        "<th>log-loss</th><th>macro F1 %</th><th>conf. AUROC</th><th>coverage</th><th>query retention</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def t3(cls: str, combo_name: str | None = None, eng: str = "tabpfn") -> str:
    ce = combo(combo_name or f"{cls}_L2_P1", eng)
    d = pd.DataFrame(ce["t3"])
    sf = {r["k"]: r["same_family_share"] for r in ce["t2"] if r["condition"] == "C4"}
    body = []
    for diff in ("C2-C3", "C2-C4", "C2-C1"):
        cells = []
        for k in K_A:
            r = d[(d["diff"] == diff) & (d["k"] == k)]
            if r.empty:
                cells.append("<td>—</td>")
                continue
            r = r.iloc[0]
            mark = "sig" if r["passes_5pp"] else ("sec" if r["passes_3pp_point"] else "")
            cells.append(f'<td class="{mark}">{pp(r["mean"])} <small>[{pp(r["lo"])}, {pp(r["hi"])}]</small></td>')
        body.append(f"<tr><th>{diff.replace('-', ' − ')}</th>{''.join(cells)}</tr>")
    body.append(
        "<tr><th>C4 same-family share</th>" + "".join(f"<td>{sf.get(k, float('nan')):.2f}</td>" for k in K_A) + "</tr>"
    )
    return (
        '<div class="tw"><table><thead><tr><th>paired difference (pp)</th>'
        + "".join(f"<th>k = {k}</th>" for k in K_A)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def t4(cls: str) -> str:
    ce = combo(f"{cls}_L2_P1")
    d = pd.DataFrame(ce["t4"]).sort_values("n", ascending=False)
    body = []
    for _, r in d.iterrows():
        rep = r.get("repair_k10", float("nan"))
        rep_s = f"{rep:.2f}" if math.isfinite(rep) else "NA"
        body.append(
            f"<tr><td>{esc(r['family'])}</td><td>{r['n']}</td><td>{pp(r['C1_A'], False)}</td><td>{pp(r['C1_B'], False)}</td>"
            + "".join(f"<td>{pp(r.get(f'C2_B_k{k}'), False)}</td>" for k in K_B)
            + f"<td>{rep_s}</td><td>{esc(r['min_k'])}</td></tr>"
        )
    b = ce["B"]
    body.append(
        f"<tr class='tot'><td>|F|-weighted</td><td>{int(d['n'].sum())}</td><td>{pp(b['weighted_C1_A'], False)}</td>"
        f"<td>{pp(b['weighted_C1_B'], False)}</td>"
        + "".join(f"<td>{pp(b.get(f'weighted_C2_B_k{k}'), False)}</td>" for k in K_B)
        + f"<td>{b['pooled_repair_k10']:.2f} <small>pooled</small></td><td></td></tr>"
    )
    return (
        '<div class="tw"><table><thead><tr><th>held-out family</th><th>|F|</th><th>C1 protocol A</th><th>C1 protocol B</th>'
        + "".join(f"<th>C2 k = {k}</th>" for k in K_B)
        + "<th>repair ratio k = 10</th><th>min k reaching C1 A</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def checks_table() -> str:
    body = []
    for c in S["consistency"]:
        extra = ""
        if "gap_A" in c:
            extra = f"gap A {pp(c['gap_A'], False)} pp, gap B {pp(c['gap_B'], False)} pp"
        if "by_k" in c:
            extra = ", ".join(f"k={k}: {v:.2f}" for k, v in c["by_k"].items())
        status = '<span class="pill ok">pass</span>' if c["pass"] else '<span class="pill bad">fail</span>'
        body.append(
            f"<tr><td>{CLS_NAME[c['class']]}</td><td>{esc(c['check'])}</td><td>{status}</td><td>{esc(extra)}</td></tr>"
        )
    return (
        '<div class="tw"><table><thead><tr><th>class</th><th>check (plan 4.3)</th><th>result</th><th>detail</th></tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table></div>"
    )


def all_combos_table() -> str:
    body = []
    for c, ce in S["combos"].items():
        for eng in ("tabpfn", "tabicl"):
            e = ce.get(eng)
            if not e or "rule_A" not in e:
                continue
            ra, rb = e["rule_A"], e.get("rule_B", {})
            cls, lab, part = c.split("_")
            body.append(
                f"<tr><td>{CLS_NAME[cls]}</td><td>{lab}</td><td>{part}</td><td>{'TabPFN-3.5' if eng == 'tabpfn' else 'TabICLv2'}</td>"
                f"<td>{pp(ra['d23_k20'])}</td><td>{pp(ra['d24_k20'])}</td><td>{pp(ra['C2_minus_C1_k20'])}</td>"
                f"<td>{esc(ra['rule'])}</td><td>{pp(rb.get('gap_pp'))}</td><td>{rb.get('pooled_repair_k10', float('nan')):.2f}</td>"
                f"<td>{esc(rb.get('rule', '—'))}</td><td>{esc(e.get('decision', '—'))}</td></tr>"
            )
    return (
        '<div class="tw"><table><thead><tr><th>class</th><th>label</th><th>partition</th><th>engine</th>'
        "<th>C2 − C3 k=20</th><th>C2 − C4 k=20</th><th>C2 − C1 k=20</th><th>rule A</th>"
        "<th>C1 A − C1 B</th><th>pooled repair k=10</th><th>rule B</th><th>decision</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def main(text: dict) -> None:
    sections = []
    sections.append(f'<section id="conclusion"><h2>1. Conclusion</h2>{text["conclusion"]}</section>')
    sections.append(
        f'<section id="data"><h2>2. Data</h2><p>{text["data"]}</p>' + t1() + caption(text["t1_caption"]) + "</section>"
    )
    a_parts = [
        f'<section id="protocol-a"><h2>3. Protocol A: random split within each family</h2><p>{text["protocol_a"]}</p>'
    ]
    for cls in CLASSES:
        a_parts.append(f"<h3>{CLS_NAME[cls]}, L2 × P1</h3>")
        a_parts.append(fig_head(f"Top-1 accuracy against context size, {CLS_NAME[cls]} (F1)"))
        a_parts.append(
            '<div class="legend"><span><i class="sw-line dash s1"></i>C1 global supervised</span>'
            '<span><i class="sw-line s2"></i>C2 same-family</span><span><i class="sw-line s3"></i>C3 other-family</span>'
            '<span><i class="sw-line s4"></i>C4 nearest-neighbour</span><span>bands: 95% CI</span></div>'
        )
        a_parts.append(f'<div class="chart">{learning_curve(cls)}</div>')
        a_parts.append(caption(text[f"f1_{cls}"]))
        a_parts.append(tab_head(f"Conditions by context size, {CLS_NAME[cls]} (T2)") + t2(cls))
        a_parts.append(tab_head(f"Paired differences, {CLS_NAME[cls]} (T3)") + t3(cls))
        a_parts.append(caption(text["t3_caption"]))
    a_parts.append("</section>")
    sections.append("".join(a_parts))
    b_parts = [f'<section id="protocol-b"><h2>4. Protocol B: one family held out</h2><p>{text["protocol_b"]}</p>']
    for cls in CLASSES:
        b_parts.append(f"<h3>{CLS_NAME[cls]}, L2 × P1</h3>")
        b_parts.append(fig_head(f"Held-out family accuracy, {CLS_NAME[cls]} (F2)"))
        b_parts.append(
            '<div class="legend"><span><i class="sw s1"></i>C1 trained on the other families</span>'
            '<span><i class="sw s2"></i>C2 with 10 same-family samples</span>'
            '<span><i class="tick-ref"></i>C1 when the family is in training (protocol A)</span></div>'
        )
        b_parts.append(f'<div class="chart">{family_bars(cls)}</div>')
        b_parts.append(caption(text[f"f2_{cls}"]))
        b_parts.append(tab_head(f"Per held-out family, {CLS_NAME[cls]} (T4)") + t4(cls))
    b_parts.append("</section>")
    sections.append("".join(b_parts))
    sections.append(
        '<section id="checks"><h2>5. Consistency checks</h2>'
        f"<p>{text['checks']}</p>"
        + tab_head("Plan 4.3 checks")
        + checks_table()
        + tab_head("Decision rules for every combination")
        + all_combos_table()
        + caption(text["combos_caption"])
        + "</section>"
    )
    sections.append(f'<section id="deviations"><h2>6. Deviations from the plan</h2>{text["deviations"]}</section>')
    sections.append(f'<section id="next"><h2>7. Next step</h2>{text["next"]}</section>')
    sections.append(f'<section id="appendix"><h2>Appendix</h2>{text["appendix"]}</section>')
    page = TEMPLATE.replace("{{BODY}}", "".join(sections)).replace("{{LEDE}}", text["lede"])
    (RES / "report.html").write_text(page)
    print("wrote", RES / "report.html")


TEMPLATE = """<title>PFN × CSP Pilot</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column (~70ch) with wide tables scrolling inside their own boxes. */
:root {
  --bg: #fbfbfa; --fg: #17191c; --muted: #565b63; --rule: #dfe1e4; --tint: #eef2f7; --accent: #1f5fa8;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --ok: #0ca30c; --bad: #d03b3b;
  --display: "Source Serif 4", Georgia, "Times New Roman", serif;
  --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #17181a; --fg: #eceef0; --muted: #a9aeb5; --rule: #34373b; --tint: #20262e; --accent: #7fb0ec;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --ok: #0ca30c; --bad: #e66767; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #17181a; --fg: #eceef0; --muted: #a9aeb5; --rule: #34373b; --tint: #20262e; --accent: #7fb0ec;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --ok: #0ca30c; --bad: #e66767; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); }
main { max-width: 860px; margin: 0 auto; padding-block: 40px 80px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header { border-bottom: 1px solid var(--rule); padding-bottom: 20px; margin-bottom: 8px; }
.eyebrow { font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
h1 { font: 600 34px/1.15 var(--display); margin: 10px 0 12px; text-wrap: balance; }
h2 { font: 600 23px/1.25 var(--display); margin: 44px 0 10px; text-wrap: balance; }
h3 { font: 600 16px/1.3 var(--body); margin: 28px 0 6px; }
p, li { max-width: 70ch; }
.lede { font-size: 16px; color: var(--muted); }
code, .mono { font-family: var(--mono); font-size: .9em; }
small { color: var(--muted); font-size: 11.5px; }
.verdict { background: var(--tint); border-left: 3px solid var(--accent); padding: 14px 18px; margin: 12px 0; }
.verdict p { margin: 6px 0; }
.fig-head { font-weight: 600; margin: 22px 0 6px; }
.fig-head span { font: 500 12px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--accent); margin-right: 6px; }
.caption { background: var(--tint); border-radius: 4px; padding: 8px 12px; font-size: 13.5px; margin: 6px 0 18px; max-width: 70ch; }
.cap-label { font: 600 11px var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-right: 8px; }
.tw { overflow-x: auto; margin: 4px 0 8px; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; min-width: 100%; }
th, td { text-align: left; padding: 5px 10px 5px 0; border-bottom: 1px solid var(--rule); vertical-align: top; white-space: nowrap; }
thead th { font-weight: 600; color: var(--muted); font-size: 12px; }
td.sig { font-weight: 600; color: var(--accent); }
td.sec { color: var(--accent); }
tr.tot td { font-weight: 600; }
.pill { font: 500 11px var(--mono); padding: 2px 8px; border-radius: 10px; border: 1px solid currentColor; }
.pill.ok { color: var(--ok); } .pill.bad { color: var(--bad); }
.chart svg { width: 100%; height: auto; max-width: 100%; display: block; }
.chart text { fill: var(--fg); font-family: var(--body); }
.chart .tick { font-size: 11px; fill: var(--muted); }
.chart .axis { font-size: 12px; fill: var(--muted); }
.chart .lab { font-size: 12px; font-weight: 500; }
.chart .grid { stroke: var(--rule); stroke-width: 1; }
.chart .line { fill: none; stroke-width: 2; }
.chart .band { opacity: .14; stroke: none; }
.chart .dot { stroke: var(--bg); stroke-width: 2; }
.chart .ref { stroke: var(--fg); stroke-width: 2; }
.s1 { stroke: var(--s1); fill: var(--s1); } .s2 { stroke: var(--s2); fill: var(--s2); }
.s3 { stroke: var(--s3); fill: var(--s3); } .s4 { stroke: var(--s4); fill: var(--s4); }
.chart .line.s1, .chart .line.s2, .chart .line.s3, .chart .line.s4 { fill: none; }
.bar { stroke: none; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 12.5px; color: var(--muted); margin: 2px 0 4px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 10px; border-radius: 2px; display: inline-block; }
.sw.s1 { background: var(--s1); } .sw.s2 { background: var(--s2); }
.sw.s3 { background: var(--s3); } .sw.s4 { background: var(--s4); }
.sw-line { width: 18px; height: 0; border-top: 2px solid; display: inline-block; }
.sw-line.s1 { border-color: var(--s1); } .sw-line.s2 { border-color: var(--s2); }
.sw-line.s3 { border-color: var(--s3); } .sw-line.s4 { border-color: var(--s4); }
.sw-line.dash { border-top-style: dashed; }
.tick-ref { width: 2px; height: 14px; background: var(--fg); display: inline-block; }
ul { padding-left: 20px; }
a { color: var(--accent); }
@media (max-width: 480px) { h1 { font-size: 27px; } }
</style>
<main>
<header>
<div class="eyebrow">Pilot experiment · Materials Project 2026.09.28 · 2026-10-23</div>
<h1>PFN × CSP pilot: is there family-level rule variation in composition → structure type?</h1>
<p class="lede">{{LEDE}}</p>
</header>
{{BODY}}
</main>
"""

if __name__ == "__main__":
    text = json.loads((EXP / "analysis" / "report_text.json").read_text())
    main(text)
