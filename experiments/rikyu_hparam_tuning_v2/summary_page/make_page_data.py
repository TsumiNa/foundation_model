#!/usr/bin/env python3
"""Regenerate the data constants of page.js (ROWS, COUNTS) from summary/ft.json.

ROWS carries, per task: name, N_train, alone, then for xfer / frozen / warm-start the value, the
relative % against alone and a significance mark ("*" separated at 2×SE and |Δ| ≥ 0.01, "·" separated
only, "" otherwise), then warm − frozen % and mark, the metric, and — appended for the unseen-encoder
arms — frozen-unseen and warm-start-unseen value / % / mark (None when the arm has not run).

    python summary_page/make_page_data.py      # then python summary_page/gen_page.py
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent; EXP = HERE.parent
ft = json.load(open(EXP / "summary" / "ft.json"))

def mark(d):
    if not d: return ""
    return "*" if d.get("matters") else ("·" if d.get("separated") else "")
def pct(d): return round(d["relative_pct"], 1) if d else None
def val(a): return round(a["mean"], 4) if a else None

rows = []
for r in sorted(ft["per_task"], key=lambda r: -r["n_train"]):
    rows.append([r["task"], r["n_train"], round(r["single_task"], 4),
                 round(r["xfer_with_replay"], 4) if r.get("xfer_with_replay") is not None else None, pct(r["xfer_vs_single"]), mark(r["xfer_vs_single"]),
                 val(r["ftz"]), pct(r["ftz_vs_single"]), mark(r["ftz_vs_single"]),
                 val(r["ftf"]), pct(r["ftf_vs_single"]), mark(r["ftf_vs_single"]),
                 pct(r["ftf_vs_ftz"]), mark(r["ftf_vs_ftz"]), r["metric"],
                 val(r.get("ftzu")), pct(r.get("ftzu_vs_single")), mark(r.get("ftzu_vs_single")),
                 val(r.get("ftfu")), pct(r.get("ftfu_vs_single")), mark(r.get("ftfu_vs_single"))])
js = HERE / "page.js"; t = js.read_text(encoding="utf-8")
t = re.sub(r"^const ROWS = .*$", "const ROWS = " + json.dumps(rows, ensure_ascii=False) + ";", t, count=1, flags=re.M)
t = re.sub(r"^const COUNTS = .*$", "const COUNTS = " + json.dumps(ft["counts"], ensure_ascii=False) + ";", t, count=1, flags=re.M)
js.write_text(t, encoding="utf-8")
n_u = sum(1 for r in ft["per_task"] if r.get("ftfu"))
print(f"  page.js: ROWS for {len(rows)} tasks ({n_u} with unseen arms), COUNTS with {len(ft['counts'])} comparisons")
