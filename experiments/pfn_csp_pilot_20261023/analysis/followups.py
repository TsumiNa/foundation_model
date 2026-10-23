"""Summarize the three follow-ups (README, Follow-ups) against their pre-stated criteria.

    uv run python analysis/followups.py

Writes results/followups.json and results/tables/FU*_*.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize as sm  # noqa: E402
from metrics import correct_top1, seed_ci  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "results"
K_R = (5, 10, 20, 50, 100, 200)
K_B = (5, 10, 20)


def ci_dict(values) -> dict:
    m, lo, hi = seed_ci(values)
    return {"mean": m, "lo": lo, "hi": hi, "seeds": [float(v) for v in values]}


def excludes0(d: dict) -> bool:
    return d["lo"] > 0 or d["hi"] < 0


# ------------------------------------------------------------------ 1. retrieval
def retrieval(cls: str, engine: str) -> dict | None:
    runs = sm.load(f"{cls}_L2_P1", engine, "R")
    if not runs:
        return None
    rows, verdict_rows = [], {}
    for k in K_R:
        top = {"C1": [], "C4": [], "C5": []}
        d41, d45, same = [], [], []
        for run in runs:
            y, res = run["meta"]["y"], run["res"]
            if ("C4", k) not in res:
                break
            test = res[("C1", 0)]["query"]
            cor = {c: correct_top1(sm.rows_of(res[(c, 0 if c == "C1" else k)], test)[0], y[test]) for c in top}
            for c in top:
                top[c].append(float(cor[c].mean()))
            d41.append(float((cor["C4"] - cor["C1"]).mean()))
            d45.append(float((cor["C4"] - cor["C5"]).mean()))
            same.append(float(np.nanmean(res[("C4", k)]["same_family_share"])))
        if len(d41) < len(runs):
            continue
        r = {
            "k": k,
            **{f"{c}_top1": ci_dict(v) for c, v in top.items()},
            "C4_minus_C1": ci_dict(d41),
            "C4_minus_C5": ci_dict(d45),
            "C4_same_family_share": float(np.mean(same)),
        }
        rows.append(r)
        verdict_rows[k] = r
    suff = [
        k
        for k, r in verdict_rows.items()
        if r["C4_minus_C1"]["mean"] >= -0.03 and r["C4_minus_C5"]["mean"] >= 0.05 and excludes0(r["C4_minus_C5"])
    ]
    helps = any(r["C4_minus_C5"]["mean"] >= 0.05 and excludes0(r["C4_minus_C5"]) for r in verdict_rows.values())
    if suff:
        v = f"retrieval suffices (from k = {min(suff)})"
    elif helps:
        v = "retrieval helps, global data still adds"
    elif verdict_rows.get(20, {}).get("C4_minus_C5", {}).get("mean", 1) < 0.03:
        v = "no locality effect"
    else:
        v = "uncertain"
    best = max(verdict_rows.values(), key=lambda r: r["C4_minus_C1"]["mean"])
    return {"rows": rows, "verdict": v, "best_k": best["k"], "best_C4_minus_C1": best["C4_minus_C1"]}


# ------------------------------------------------------------------ 2. intermetallics split
def split_b(label: str, engine: str) -> dict | None:
    combo = f"AB_{label}_P1x"
    a = sm.protocol_a(combo, engine)
    b = sm.protocol_b(combo, engine, a["c1a_by_family"] if a else None)
    if a is None or b is None:
        return None
    out = {
        "t4": b["t4"].to_dict(orient="records"),
        "C1_A_all_top1": a["c1_all_top1"],
        **{k: v for k, v in b.items() if k != "t4"},
    }
    if engine != "catboost":
        out["rule_B"] = sm.rule_b(b)
        out["rule_A"] = sm.rule_a(a["t2"], a["t3"])
    return out


# ------------------------------------------------------------------ 3. mixed context
def mixed(combo: str, engine: str) -> dict | None:
    rm, rb = sm.load(combo, engine, "M"), sm.load(combo, engine, "B")
    a = sm.protocol_a(combo, engine)
    if not rm or not rb or a is None:
        return None
    y, fam = rb[0]["meta"]["y"], rb[0]["meta"]["fam"]
    fams = rb[0]["meta"]["families"]
    c1a = a["c1a_by_family"]
    res: dict = {"by_k": {}, "families": []}
    for k in K_B:
        fs = [f for f in range(len(fams)) if ("M", f, k) in rm[0]["res"]]
        n = np.array([(fam == f).sum() for f in fs])
        per_seed = {"M": [], "C1_B": [], "C2_B": [], "M_minus_C1B": [], "M_minus_C2B": [], "repair": []}
        for s_i in range(len(rm)):
            accs = {"M": [], "C1_B": [], "C2_B": []}
            for f in fs:
                held = np.flatnonzero(fam == f)
                accs["M"].append(correct_top1(sm.rows_of(rm[s_i]["res"][("M", f, k)], held)[0], y[held]).mean())
                accs["C1_B"].append(correct_top1(sm.rows_of(rb[s_i]["res"][("C1", f, 0)], held)[0], y[held]).mean())
                accs["C2_B"].append(correct_top1(sm.rows_of(rb[s_i]["res"][("C2", f, k)], held)[0], y[held]).mean())
            w = {key: float(np.average(v, weights=n)) for key, v in accs.items()}
            for key in ("M", "C1_B", "C2_B"):
                per_seed[key].append(w[key])
            per_seed["M_minus_C1B"].append(w["M"] - w["C1_B"])
            per_seed["M_minus_C2B"].append(w["M"] - w["C2_B"])
            num = float(np.sum(n * (np.array(accs["M"]) - np.array(accs["C1_B"]))))
            den = float(np.sum(n * (np.array([c1a[f] for f in fs]) - np.array(accs["C1_B"]))))
            per_seed["repair"].append(num / den if den else float("nan"))
        res["by_k"][k] = {key: ci_dict(v) for key, v in per_seed.items()} | {"n_families": len(fs)}
        if k == 10:
            for i, f in enumerate(fs):
                held = np.flatnonzero(fam == f)
                m = np.mean([correct_top1(sm.rows_of(r["res"][("M", f, k)], held)[0], y[held]).mean() for r in rm])
                c1 = np.mean([correct_top1(sm.rows_of(r["res"][("C1", f, 0)], held)[0], y[held]).mean() for r in rb])
                c2 = np.mean([correct_top1(sm.rows_of(r["res"][("C2", f, k)], held)[0], y[held]).mean() for r in rb])
                res["families"].append(
                    {
                        "family": fams[f],
                        "n": int(n[i]),
                        "C1_A": c1a[f],
                        "C1_B": float(c1),
                        "C2_B_k10": float(c2),
                        "M_k10": float(m),
                    }
                )
    g = res["by_k"][10]["M_minus_C1B"]
    if g["mean"] >= 0.05 and excludes0(g):
        res["verdict"] = "few own samples help on top of global data"
    elif g["mean"] < 0.03:
        res["verdict"] = "no added value"
    else:
        res["verdict"] = "uncertain"
    return res


def main() -> None:
    out: dict = {"retrieval": {}, "split": {}, "mixed": {}}
    for cls in ("AB", "ABX3"):
        for eng in ("tabpfn", "tabicl"):
            r = retrieval(cls, eng)
            if r:
                out["retrieval"][f"{cls}_{eng}"] = r
                print("R", cls, eng, r["verdict"], "best k", r["best_k"], round(r["best_C4_minus_C1"]["mean"], 3))
    for lab in ("L1", "L2"):
        for eng in ("tabpfn", "tabicl", "catboost"):
            r = split_b(lab, eng)
            if r:
                out["split"][f"AB_{lab}_P1x_{eng}"] = r
                if "rule_B" in r:
                    print("P1x", lab, eng, r["rule_B"], r["pooled_repair_k10_ci"])
    for combo in ("AB_L2_P1", "ABX3_L2_P1", "AB_L2_P1x"):
        for eng in ("tabpfn", "tabicl"):
            r = mixed(combo, eng)
            if r:
                out["mixed"][f"{combo}_{eng}"] = r
                g = r["by_k"][10]
                print(
                    "M", combo, eng, r["verdict"], {k: round(v["mean"], 3) for k, v in g.items() if isinstance(v, dict)}
                )
    (OUT / "followups.json").write_text(
        json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    )


if __name__ == "__main__":
    main()
