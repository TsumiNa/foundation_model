"""Render results/followups.html: a plain-language reading of the pilot and the three follow-ups.

uv run python analysis/make_followup_report.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_report as mr  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
RES = EXP / "results"
F = json.loads((RES / "followups.json").read_text())
S = mr.S
CLS = mr.CLS_NAME


def pc(x) -> str:
    return f"{x * 100:.1f}"


def d(dd) -> str:
    return f"{dd['mean'] * 100:+.1f} pp <small>[{dd['lo'] * 100:+.1f}, {dd['hi'] * 100:+.1f}]</small>"


# ------------------------------------------------------------------ charts
def retrieval_chart(cls: str) -> str:
    rows = F["retrieval"][f"{cls}_tabpfn"]["rows"]
    ks = [r["k"] for r in rows]
    w, h, ml, mr_, mt, mb = 560, 280, 52, 130, 14, 40
    pw, ph = w - ml - mr_, h - mt - mb
    lo, hi = 0.3, 0.8
    xs = {k: ml + pw * (math.log(k) - math.log(ks[0])) / (math.log(ks[-1]) - math.log(ks[0])) for k in ks}

    def y(v):
        return mt + ph * (1 - (v - lo) / (hi - lo))

    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Retrieval follow-up, {cls}">']
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        out.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        out.append(f'<text class="tick" x="{ml - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t * 100:.0f}</text>')
    for k, x in xs.items():
        out.append(f'<text class="tick" x="{x:.1f}" y="{mt + ph + 18}" text-anchor="middle">{k}</text>')
    out.append(
        f'<text class="axis" x="{ml + pw / 2}" y="{h - 4}" text-anchor="middle">number of examples shown, k (log scale)</text>'
    )
    out.append(
        f'<text class="axis" transform="translate(14 {mt + ph / 2}) rotate(-90)" text-anchor="middle">Top-1 accuracy (%)</text>'
    )
    ends = []
    for key, s, name, dash in (
        ("C1_top1", 1, "global model", True),
        ("C4_top1", 4, "most similar k", False),
        ("C5_top1", 3, "random k", False),
    ):
        pts = [(xs[r["k"]], y(r[key]["mean"])) for r in rows]
        band = [(xs[r["k"]], y(r[key]["hi"])) for r in rows] + [(xs[r["k"]], y(r[key]["lo"])) for r in rows[::-1]]
        if not dash:
            out.append(f'<polygon class="band s{s}" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in band)}"/>')
        da = ' stroke-dasharray="6 4"' if dash else ""
        out.append(f'<polyline class="line s{s}"{da} points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}"/>')
        for (a, b), r in zip(pts, rows):
            tip = f"{name}, k = {r['k']}: {pc(r[key]['mean'])}% (95% CI {pc(r[key]['lo'])}–{pc(r[key]['hi'])})"
            out.append(
                f'<circle class="dot s{s}" cx="{a:.1f}" cy="{b:.1f}" r="4"><title>{mr.esc(tip)}</title></circle>'
            )
        ends.append([pts[-1][1], name, rows[-1][key]["mean"]])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for yy, name, v in ends:
        out.append(f'<text class="lab" x="{ml + pw + 10}" y="{yy + 4:.1f}">{name} {v * 100:.0f}%</text>')
    out.append("</svg>")
    return "".join(out)


def bars(rows: list[dict], series: list[tuple[str, int, str]], ref_key: str | None) -> str:
    rows = sorted(rows, key=lambda r: -r["n"])
    n, ns = len(rows), len(series)
    w, mt, mb, ml, mr_ = 560, 12, 36, 170, 20
    row = 10 + 12 * ns
    h = mt + mb + row * n
    pw = w - ml - mr_

    def x(v):
        return ml + pw * v

    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="per-family accuracy">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{mt}" y2="{mt + row * n}"/>')
        out.append(
            f'<text class="tick" x="{x(t):.1f}" y="{mt + row * n + 16}" text-anchor="middle">{t * 100:.0f}</text>'
        )
    out.append(
        f'<text class="axis" x="{ml + pw / 2}" y="{h - 4}" text-anchor="middle">Top-1 accuracy on the held-out family (%)</text>'
    )
    for i, r in enumerate(rows):
        y0 = mt + i * row
        out.append(
            f'<text class="tick" x="{ml - 8}" y="{y0 + row / 2 + 4}" text-anchor="end">{mr.esc(r["family"])} ({r["n"]})</text>'
        )
        for j, (key, s, name) in enumerate(series):
            v = r[key]
            out.append(
                f'<rect class="bar s{s}" x="{ml}" y="{y0 + 5 + j * 12}" width="{max(x(v) - ml, 1):.1f}" height="10" rx="2"><title>{mr.esc(f"{name}: {v * 100:.1f}%")}</title></rect>'
            )
        if ref_key:
            v = r[ref_key]
            out.append(
                f'<line class="ref" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{y0 + 2}" y2="{y0 + row - 4}"><title>{mr.esc(f"family seen in training: {v * 100:.1f}%")}</title></line>'
            )
    out.append("</svg>")
    return "".join(out)


def legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(f'<span><i class="{c}"></i>{t}</span>' for c, t in items) + "</div>"


# ------------------------------------------------------------------ page
def main() -> None:
    a_ab, a_x = S["combos"]["AB_L2_P1"]["tabpfn"], S["combos"]["ABX3_L2_P1"]["tabpfn"]
    r_ab, r_x = F["retrieval"]["AB_tabpfn"], F["retrieval"]["ABX3_tabpfn"]
    rk = {c: {r["k"]: r for r in F["retrieval"][f"{c}_tabpfn"]["rows"]} for c in ("AB", "ABX3")}
    sp = F["split"]["AB_L2_P1x_tabpfn"]
    sp_l1, sp_icl = F["split"]["AB_L1_P1x_tabpfn"], F["split"]["AB_L2_P1x_tabicl"]
    m_ab, m_x, m_abx = (F["mixed"][k] for k in ("AB_L2_P1_tabpfn", "ABX3_L2_P1_tabpfn", "AB_L2_P1x_tabpfn"))
    m_ab_icl, m_x_icl = F["mixed"]["AB_L2_P1_tabicl"], F["mixed"]["ABX3_L2_P1_tabicl"]
    mr.FIG[0] = 0
    mr.TAB[0] = 0
    body = []

    body.append(f"""<section id="answer"><h2>The answer in three lines</h2><div class="verdict">
<p><strong>No sign of family-specific rules.</strong> Composition → structure type behaves like one smooth map over the descriptor space:
the {r_ab["best_k"]} most similar training compounds predict as well as the full training set
(AB {d(r_ab["best_C4_minus_C1"])}; ABX<sub>3</sub> {d(r_x["best_C4_minus_C1"])} at k = {r_x["best_k"]}).</p>
<p><strong>Few examples of a new family help only when that family is chemically new to the model,</strong> and then they help most on top of the
global data, not instead of it: ten own examples added to everything else raise a held-out family by {d(m_x["by_k"]["10"]["M_minus_C1B"])} (ABX<sub>3</sub>)
and {d(m_abx["by_k"]["10"]["M_minus_C1B"])} (AB, intermetallics split).</p>
<p><strong>For the roadmap:</strong> this argues against building a library of family rules as a PFN prior, and for a generic PFN (or a kNN-style model)
fed with a retrieved context, plus more data where chemistry is poorly covered.</p></div></section>""")

    body.append(f"""<section id="pilot"><h2>1. What the pilot did, in plain terms</h2>
<p>We asked whether each chemical family (halides, oxides, intermetallics, …) follows its own rule for which structure type a composition takes.
If it does, a model that sees a handful of examples from a family should pick up that family's rule, and that is the case where a PFN with a
library of family rules as its prior would pay off.</p>
<h3>Test 1: within each family, which examples help the model most?</h3>
<p>20% of every family is held out as the exam. The model answers each exam question after seeing one of four kinds of examples:</p>
<div class="tw"><table><thead><tr><th>examples shown</th><th>analogy</th><th>AB, k = 20</th><th>ABX<sub>3</sub>, k = 20</th></tr></thead><tbody>
<tr><td>the whole training set</td><td>read the whole textbook</td><td>{pc(a_ab["t2"][8]["top1"])}%</td><td>{pc(a_x["t2"][8]["top1"])}%</td></tr>
<tr><td>20 random examples from the same family</td><td>a few worked problems of the same kind</td><td>{pc(a_ab["t2"][9]["top1"])}%</td><td>{pc(a_x["t2"][9]["top1"])}%</td></tr>
<tr><td>20 random examples from other families</td><td>worked problems of another kind</td><td>{pc(a_ab["t2"][10]["top1"])}%</td><td>{pc(a_x["t2"][10]["top1"])}%</td></tr>
<tr><td>the 20 most similar examples, any family</td><td>the problems that look most like this one</td><td>{pc(a_ab["t2"][11]["top1"])}%</td><td>{pc(a_x["t2"][11]["top1"])}%</td></tr>
</tbody></table></div>
<p><strong>Hoped for:</strong> same-family examples beat both other-family and most-similar examples; that would mean a family carries a rule beyond resemblance.
<strong>Got:</strong> same-family beats other-family, but the most similar examples beat same-family in every setting, and nothing beats the whole textbook.</p>
<h3>Test 2: can the model handle a family it has never seen, and do a few examples fix it?</h3>
<p>One family at a time is removed from training entirely; the model is then tested on it, first with nothing from that family, then with 5–20 of its members as the only examples.</p>
<p><strong>Hoped for:</strong> the unseen family is predicted badly, and ten of its members recover at least half of the loss.
<strong>Got:</strong> AB lost 40 points and ten examples recovered half (right on the threshold); ABX<sub>3</sub> lost 13 points and ten own examples did worse than the model trained on the other anions.
The AB recovery came from one family: the intermetallics, half of AB, with no metal-metal compound left in training once they were removed.</p>
<p>The two classes pointed to different routes, so the plan's verdict was “uncertain”. Hence three follow-ups.</p></section>""")

    body.append(f"""<section id="fu1"><h2>2. Follow-up 1: are the “most similar examples” enough?</h2>
<p><strong>Question.</strong> If similarity is what matters, a model that only sees the k most similar training compounds should match the model trained on everything once k is large enough.
<strong>Done.</strong> Same exams as test 1; most-similar examples (any family) against the same number of random examples, k from 5 to 200.
<strong>Criterion fixed beforehand.</strong> “Retrieval suffices” if at some k the most-similar context is within 3 points of the full model and at least 5 points (CI excluding 0) above random examples.</p>
{mr.fig_head("Most similar against random examples, AB")}
{legend([("sw-line dash s1", "global model (all ~900 training rows)"), ("sw-line s4", "k most similar examples"), ("sw-line s3", "k random examples"), ("", "bands: 95% CI")])}
<div class="chart">{retrieval_chart("AB")}</div>
{mr.caption(f"With 100 similar examples the gap to the full model is {d(rk['AB'][100]['C4_minus_C1'])}; 100 random examples are {rk['AB'][100]['C4_minus_C5']['mean'] * 100:.1f} points worse. The criterion is met from k = 50 (TabICLv2: from k = 20).")}
{mr.fig_head("Most similar against random examples, ABX<sub>3</sub>")}
<div class="chart">{retrieval_chart("ABX3")}</div>
{mr.caption(f"Met from k = 50 with both engines; at k = 200 the gap is {d(rk['ABX3'][200]['C4_minus_C1'])}. Random examples catch up as k approaches the size of the training set (about 630 rows).")}
<p><strong>Result: retrieval suffices.</strong> The global model behaves like a lookup of the nearest ~100 compounds in descriptor space. This is the locality explanation of test 1, now confirmed directly.</p></section>""")

    body.append(f"""<section id="fu2"><h2>3. Follow-up 2: was the AB “few examples fix it” result one family?</h2>
<p><strong>Done.</strong> Test 2 again with the intermetallics split by the periodic-table blocks of their two elements (rare earth–transition metal f–d, transition–transition d–d, and so on),
giving 12 families of 18–202 compounds instead of one family of 572.
<strong>Criterion.</strong> The plan's own: a loss of at least 10 points on unseen families and at least half of it recovered by ten own examples.</p>
{mr.fig_head("Unseen-family accuracy with intermetallics split, AB")}
{legend([("sw s1", "trained on the other families"), ("sw s2", "10 examples of the family only"), ("tick-ref", "family seen in training")])}
<div class="chart">{bars(sp["t4"], [("C1_B", 1, "trained on the other families"), ("C2_B_k10", 2, "10 examples of the family only")], "C1_A")}</div>
{mr.caption(f"Loss on unseen families: {sp['gap_pp'] * 100:.1f} points (was 39.9 with one intermetallic family). Pooled recovery with ten own examples: {sp['pooled_repair_k10']:.2f} (95% CI {sp['pooled_repair_k10_ci'][1]:.2f} to {sp['pooled_repair_k10_ci'][2]:.2f}); TabICLv2 {sp_icl['pooled_repair_k10']:.2f}; L1 labels {sp_l1['pooled_repair_k10']:.2f}.")}
<p><strong>Result: it was one family.</strong> Once no single family removes a whole kind of chemistry from training, an unseen family loses about 14 points and ten own examples recover little of it: {sp["pooled_repair_k10"]:.2f} with TabPFN, {sp_icl["pooled_repair_k10"]:.2f} with TabICLv2, {sp_l1["pooled_repair_k10"]:.2f} with L1 labels, all below the 0.5 threshold.
The pilot's AB recovery measured missing coverage (no metals in training), not a family rule.</p></section>""")

    body.append(f"""<section id="fu3"><h2>4. Follow-up 3: everything else plus a few examples of the new family</h2>
<p><strong>Done.</strong> Test 2 with the realistic setup: the model sees all other families plus k members of the unseen family (the same k members as before), and is tested on the rest of that family.
<strong>Criterion.</strong> Useful if ten added examples raise accuracy on the unseen family by at least 5 points (CI excluding 0); no added value below 3 points.</p>
<div class="tw"><table><thead><tr><th>setting</th><th>other families only</th><th>10 own examples only</th><th>other families + 10 own</th><th>gain from the 10</th><th>verdict</th></tr></thead><tbody>
<tr><td>ABX<sub>3</sub> (anion families)</td><td>{pc(m_x["by_k"]["10"]["C1_B"]["mean"])}%</td><td>{pc(m_x["by_k"]["10"]["C2_B"]["mean"])}%</td><td>{pc(m_x["by_k"]["10"]["M"]["mean"])}%</td><td>{d(m_x["by_k"]["10"]["M_minus_C1B"])}</td><td>useful (TabICLv2 {d(m_x_icl["by_k"]["10"]["M_minus_C1B"])})</td></tr>
<tr><td>AB, intermetallics split</td><td>{pc(m_abx["by_k"]["10"]["C1_B"]["mean"])}%</td><td>{pc(m_abx["by_k"]["10"]["C2_B"]["mean"])}%</td><td>{pc(m_abx["by_k"]["10"]["M"]["mean"])}%</td><td>{d(m_abx["by_k"]["10"]["M_minus_C1B"])}</td><td>between thresholds; {d(m_abx["by_k"]["20"]["M_minus_C1B"])} at k = 20</td></tr>
<tr><td>AB, one intermetallic family</td><td>{pc(m_ab["by_k"]["10"]["C1_B"]["mean"])}%</td><td>{pc(m_ab["by_k"]["10"]["C2_B"]["mean"])}%</td><td>{pc(m_ab["by_k"]["10"]["M"]["mean"])}%</td><td>{d(m_ab["by_k"]["10"]["M_minus_C1B"])}</td><td>useful (TabICLv2 {d(m_ab_icl["by_k"]["10"]["M_minus_C1B"])})</td></tr>
</tbody></table></div>
{mr.fig_head("Unseen-family accuracy with ten own examples added, ABX<sub>3</sub>")}
{legend([("sw s1", "other families only"), ("sw s2", "10 own examples only"), ("sw s3", "other families + 10 own")])}
<div class="chart">{bars(m_x["families"], [("C1_B", 1, "other families only"), ("C2_B_k10", 2, "10 own examples only"), ("M_k10", 3, "other families + 10 own")], None)}</div>
{mr.caption("Adding ten own examples costs at most 4 points (chlorides) and helps most where the other families transfer worst (fluorides +25 points, selenides +13). Ten own examples alone are worse than the global model for six of the seven anions.")}
<p><strong>Result.</strong> A few examples of a new family are worth adding to the global data, by 3–6 points when the family is chemically related to the training data and by 25 points when it is not.
This is ordinary in-context adaptation; it does not need family rules.</p></section>""")

    body.append("""<section id="method"><h2>5. Notes</h2><ul>
<li>Same Materials Project data, features, seeds (0–4) and decision thresholds as the pilot; TabPFN-3.5 primary, TabICLv2 as the engine check. All three follow-ups agree between engines.</li>
<li>The follow-up criteria were written into the experiment README before the runs. Follow-ups 1 and 3 are not part of the original plan; follow-up 2 reuses the plan's protocol-B rule with a new partition.</li>
<li>“Points” are percentage points of Top-1 accuracy; intervals are 95% bootstrap CIs over five seeds.</li>
<li>Compute: RIKYU GB200, 6 GPUs, 22 min wall clock for the 95 follow-up runs.</li>
<li>Pilot report with the plan's tables T1–T4: <a href="https://claude.ai/artifact/7fnHZtx5At8RYFrnj88K3B">PFN × CSP Pilot</a>.</li></ul></section>""")

    page = mr.TEMPLATE.replace("{{BODY}}", "".join(body))
    page = page.replace(
        "{{LEDE}}",
        "A plain-language reading of the pilot, and three follow-up experiments that settle what it left open.",
    )
    page = page.replace("<title>PFN × CSP Pilot</title>", "<title>PFN × CSP Follow-ups</title>")
    page = page.replace(
        "<h1>PFN × CSP pilot: is there family-level rule variation in composition → structure type?</h1>",
        "<h1>Do chemical families need their own structure rules?</h1>",
    )
    page = page.replace(
        "Pilot experiment · Materials Project 2026.09.28 · 2026-10-23",
        "Follow-up experiments · Materials Project 2026.09.28 · 2026-10-23",
    )
    (RES / "followups.html").write_text(page)
    print("wrote", RES / "followups.html")


if __name__ == "__main__":
    main()
