"""Collect run outputs into the plan's tables T1-T4, consistency checks and decision rules.

    uv run python analysis/summarize.py

Reads artifacts/runs/<combo>/<engine>/prot{A,B}_s<seed>.pkl and results/appendix/build_summary.json;
writes results/tables/*.csv and results/summary.json (consumed by analysis/make_report.py).
"""

from __future__ import annotations

import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import all_metrics, correct_top1, seed_ci  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
RUNS = EXP / "artifacts" / "runs"
OUT = EXP / "results"
SEEDS = (0, 1, 2, 3, 4)
K_A = (5, 10, 20, 50)
K_B = (5, 10, 20)
SIG_PP, SECONDARY_PP = 0.05, 0.03
COMBOS = [f"{c}_{lab}_{p}" for c, lab, p in product(("AB", "ABX3"), ("L1", "L2"), ("P1", "P2"))]
MAIN = ("AB_L2_P1", "ABX3_L2_P1")


def load(combo: str, engine: str, prot: str) -> list[dict]:
    runs = []
    for s in SEEDS:
        f = RUNS / combo / engine / f"prot{prot}_s{s}.pkl"
        if not f.exists():
            return []
        with open(f, "rb") as fh:
            runs.append(pickle.load(fh))
    return runs


def rows_of(res: dict, idx: np.ndarray):
    pos = {int(q): j for j, q in enumerate(res["query"])}
    r = np.array([pos[int(i)] for i in idx])
    return res["prob"][r], res["n_valid"][r], res["n_covered"][r]


# ---------------------------------------------------------------- protocol A
def protocol_a(combo: str, engine: str) -> dict | None:
    runs = load(combo, engine, "A")
    if not runs:
        return None
    per_seed = []  # list of dict[(cond,k)] -> metrics, plus correctness vectors
    for run in runs:
        y = run["meta"]["y"]
        res = run["res"]
        test = res[("C1", 0)]["query"]
        seed_out: dict = {}
        p, nv, nc = rows_of(res[("C1", 0)], test)
        seed_out[("C1", "all")] = {**all_metrics(p, y[test], nv, nc), "correct": None}
        fam = run["meta"]["fam"]
        c1_fam = {}
        for f in np.unique(fam[test]):
            qf = test[fam[test] == f]
            pf, _, _ = rows_of(res[("C1", 0)], qf)
            c1_fam[int(f)] = (float(correct_top1(pf, y[qf]).mean()), len(qf))
        seed_out["C1_by_family"] = c1_fam
        for k in K_A:
            q = res.get(("eligible", k))
            if q is None:
                continue
            seed_out[("retention", k)] = len(q) / len(test)
            if len(q) == 0:
                continue
            for cond in ("C1", "C2", "C3", "C4"):
                key = (cond, 0) if cond == "C1" else (cond, k)
                if key not in res:
                    continue
                p, nv, nc = rows_of(res[key], q)
                m = all_metrics(p, y[q], nv, nc)
                m["correct"] = correct_top1(p, y[q])
                if cond == "C4":
                    m["same_family_share"] = float(np.nanmean(res[key]["same_family_share"]))
                seed_out[(cond, k)] = m
        per_seed.append(seed_out)

    t2, t3 = [], []
    for k in K_A:
        ret = [s.get(("retention", k), np.nan) for s in per_seed]
        for cond in ("C1", "C2", "C3", "C4"):
            ms = [s.get((cond, k)) for s in per_seed]
            if any(m is None for m in ms):
                continue
            row = {"k": k, "condition": cond, "query_retention": float(np.mean(ret))}
            for met in ("top1", "top3", "log_loss", "macro_f1", "auroc", "coverage", "coverage_any"):
                mu, lo, hi = seed_ci([m[met] for m in ms])
                row.update({met: mu, f"{met}_lo": lo, f"{met}_hi": hi})
            if cond == "C4":
                row["same_family_share"] = float(np.mean([m["same_family_share"] for m in ms]))
            t2.append(row)
        for a, b in (("C2", "C3"), ("C2", "C4"), ("C2", "C1")):
            if any(s.get((a, k)) is None or s.get((b, k)) is None for s in per_seed):
                continue
            d = [float((s[(a, k)]["correct"] - s[(b, k)]["correct"]).mean()) for s in per_seed]
            mu, lo, hi = seed_ci(d)
            excl0 = lo > 0 or hi < 0
            t3.append(
                {
                    "k": k,
                    "diff": f"{a}-{b}",
                    "mean": mu,
                    "lo": lo,
                    "hi": hi,
                    "seed_values": d,
                    "ci_excludes_0": excl0,
                    "passes_5pp": bool(mu >= SIG_PP and excl0),
                    "passes_3pp_point": bool(mu >= SECONDARY_PP),
                }
            )
    c1_all = seed_ci([s[("C1", "all")]["top1"] for s in per_seed])
    # C1_A(F): protocol-A C1 top-1 on F's whole test part, mean over seeds
    fams = sorted(per_seed[0]["C1_by_family"])
    c1a_f = {f: float(np.mean([s["C1_by_family"][f][0] for s in per_seed])) for f in fams}
    return {
        "t2": pd.DataFrame(t2),
        "t3": pd.DataFrame(t3),
        "c1_all_top1": c1_all,
        "c1a_by_family": c1a_f,
        "families": runs[0]["meta"]["families"],
    }


# ---------------------------------------------------------------- protocol B
def protocol_b(combo: str, engine: str, c1a_by_family: dict | None) -> dict | None:
    runs = load(combo, engine, "B")
    if not runs:
        return None
    y, fam = runs[0]["meta"]["y"], runs[0]["meta"]["fam"]
    fams = runs[0]["meta"]["families"]
    rows, seed_c1, seed_c2 = [], {}, {}
    for f in range(len(fams)):
        held = np.flatnonzero(fam == f)
        row = {"family": fams[f], "n": len(held)}
        c1 = [float(correct_top1(rows_of(r["res"][("C1", f, 0)], held)[0], y[held]).mean()) for r in runs]
        row["C1_B"] = float(np.mean(c1))
        seed_c1[f] = c1
        for k in K_B:
            key = ("C2", f, k)
            if key in runs[0]["res"]:
                v = [float(correct_top1(rows_of(r["res"][key], held)[0], y[held]).mean()) for r in runs]
                row[f"C2_B_k{k}"] = float(np.mean(v))
                seed_c2[(f, k)] = v
                row[f"C2_B_k{k}_coverage"] = float(
                    np.mean([(r["res"][key]["n_covered"] / r["res"][key]["n_valid"]).mean() for r in runs])
                )
                row[f"n_draws_k{k}"] = float(np.mean([r["res"][key]["n_draws"] for r in runs]))
            else:
                row[f"C2_B_k{k}"] = np.nan
        if c1a_by_family is not None:
            row["C1_A"] = c1a_by_family.get(f, np.nan)
            den = row["C1_A"] - row["C1_B"]
            for k in K_B:
                row[f"repair_k{k}"] = (row[f"C2_B_k{k}"] - row["C1_B"]) / den if den >= 0.02 else np.nan
            mk = [k for k in K_B if np.isfinite(row[f"C2_B_k{k}"]) and row[f"C2_B_k{k}"] >= row["C1_A"]]
            row["min_k"] = str(mk[0]) if mk else "> 20"
        rows.append(row)
    t4 = pd.DataFrame(rows)
    out = {"t4": t4, "weighted_C1_B": float(np.average(t4["C1_B"], weights=t4["n"]))}
    if c1a_by_family is not None:
        out["weighted_C1_A"] = float(np.average(t4["C1_A"], weights=t4["n"]))
        out["gap_pp"] = out["weighted_C1_A"] - out["weighted_C1_B"]
        for k in K_B:
            ok = t4[np.isfinite(t4[f"C2_B_k{k}"])]
            num = float((ok["n"] * (ok[f"C2_B_k{k}"] - ok["C1_B"])).sum())
            den = float((ok["n"] * (ok["C1_A"] - ok["C1_B"])).sum())
            out[f"pooled_repair_k{k}"] = num / den if den != 0 else float("nan")
            out[f"pooled_families_k{k}"] = int(len(ok))
            # seed-level pooled ratio: C1_A stays the seed mean (plan 3.3), C1_B and C2_B per seed
            fs = [f for f in range(len(fams)) if (f, k) in seed_c2]
            n_f = {f: int((fam == f).sum()) for f in fs}
            c1a = {f: c1a_by_family[f] for f in fs}
            per_seed = []
            for s_i in range(len(runs)):
                nm = sum(n_f[f] * (seed_c2[(f, k)][s_i] - seed_c1[f][s_i]) for f in fs)
                dn = sum(n_f[f] * (c1a[f] - seed_c1[f][s_i]) for f in fs)
                per_seed.append(nm / dn if dn != 0 else float("nan"))
            out[f"pooled_repair_k{k}_seeds"] = per_seed
            out[f"pooled_repair_k{k}_ci"] = seed_ci(per_seed)
            out[f"weighted_C2_B_k{k}"] = (
                float(np.average(ok[f"C2_B_k{k}"], weights=ok["n"])) if len(ok) else float("nan")
            )
    return out


# ---------------------------------------------------------------- decision rules (plan 4.2)
def rule_a(t2: pd.DataFrame, t3: pd.DataFrame) -> dict:
    def d(name, k):
        r = t3[(t3["diff"] == name) & (t3["k"] == k)]
        return r.iloc[0] if len(r) else None

    def top1(cond, k):
        r = t2[(t2["condition"] == cond) & (t2["k"] == k)]
        return float(r["top1"].iloc[0]) if len(r) else float("nan")

    d23, d24 = d("C2-C3", 20), d("C2-C4", 20)
    d23_10, d24_10 = d("C2-C3", 10), d("C2-C4", 10)
    notes = []
    sign_ok = all(
        x is not None and y is not None and np.sign(x["mean"]) == np.sign(y["mean"])
        for x, y in ((d23, d23_10), (d24, d24_10))
    )
    ret50 = t2[(t2["k"] == 50)]["query_retention"]
    k_r3 = 50 if (len(ret50) and float(ret50.iloc[0]) >= 0.30) else 20
    if k_r3 == 20:
        notes.append("k = 50 query retention < 30 %: rule 3 read at k = 20 (deviation from plan)")
    c2_minus_c1_r3 = top1("C2", k_r3) - top1("C1", k_r3)
    out = {
        "d23_k20": d23["mean"],
        "d24_k20": d24["mean"],
        "C2_minus_C1_k20": top1("C2", 20) - top1("C1", 20),
        "k_for_rule3": k_r3,
        "C2_minus_C1_rule3": c2_minus_c1_r3,
        "sign_k10_equals_k20": sign_ok,
        "notes": notes,
    }
    c4 = t2[t2["condition"] == "C4"].set_index("k")["same_family_share"]
    c4_same_high = bool(any(float(c4.get(k, 0.0)) >= 0.8 for k in (10, 20)))
    out["C4_same_family_share_k10_k20"] = [float(c4.get(10, np.nan)), float(c4.get(20, np.nan))]
    if c4_same_high:
        notes.append("C4 same-family share >= 0.8 at k = 10/20: C2 - C4 cannot be read for H1' (plan 4.3)")
    if not sign_ok:
        out["rule"], out["verdict"] = "uncertain", "sign of C2-C3 or C2-C4 differs between k = 10 and k = 20"
    elif c4_same_high and not (d23["mean"] < SECONDARY_PP and abs(c2_minus_c1_r3) < 0.03):
        out["rule"], out["verdict"] = (
            "uncertain",
            (
                "C2-C3 passes but C2-C4 is not readable (C4 neighbours >= 80 % same family)"
                if d23["passes_5pp"]
                else "no row of the rule table applies"
            ),
        )
    elif d23["passes_5pp"] and d24["passes_5pp"] and out["C2_minus_C1_k20"] >= -0.03:
        out["rule"], out["verdict"] = "A1", "multiple worlds; family rule identifiable from few samples; not locality"
    elif d23["passes_5pp"] and d24["mean"] < SECONDARY_PP:
        out["rule"], out["verdict"] = "A2", "gain comes from locality in descriptor space"
    elif d23["mean"] < SECONDARY_PP and abs(c2_minus_c1_r3) < 0.03:
        out["rule"], out["verdict"] = "A3", "one global rule, already expressed by the features"
    else:
        why = []
        if d23["mean"] >= SIG_PP and not d23["ci_excludes_0"]:
            why.append("C2-C3 >= 5 pp but CI includes 0")
        if SECONDARY_PP <= d23["mean"] < SIG_PP:
            why.append("C2-C3 between 3 and 5 pp")
        out["rule"], out["verdict"] = "uncertain", "; ".join(why) or "no row of the rule table applies"
    return out


def rule_b(b: dict) -> dict:
    gap, rep = b["gap_pp"], b["pooled_repair_k10"]
    if gap >= 0.10 and rep >= 0.5:
        r, v = "B1", "cross-family extrapolation fails and k <= 10 same-family samples repair it"
    elif gap >= 0.10:
        r, v = "B2", "cross-family extrapolation fails and few samples do not repair it"
    elif gap < 0.05:
        r, v = "B3", "the global rule extrapolates across families"
    else:
        r, v = "uncertain", "gap between 5 and 10 pp"
    return {"rule": r, "verdict": v, "gap_pp": gap, "pooled_repair_k10": rep}


ROUTE = {
    "A1": "rule-prior PFN route",
    "A2": "locality model (kNN or generic PFN)",
    "A3": "composition-conditioned generative model",
    "B1": "rule-prior PFN route",
    "B2": "more labels or conditional generative model; no rule prior",
}


def combine(ra: dict, rb: dict) -> tuple[str, str]:
    """Protocol B decides; protocol A refines inside 'no rule prior' (plan 4.2 last paragraph)."""
    if rb["rule"] in ("B1", "B2"):
        return rb["rule"], ROUTE[rb["rule"]]
    if rb["rule"] == "B3":
        if ra["rule"] == "A1":
            return "uncertain", "B3 with A1"
        if ra["rule"] in ("A2", "A3"):
            return f"B3+{ra['rule']}", ROUTE[ra["rule"]]
        return "uncertain", "B3 with protocol A uncertain"
    return "uncertain", "protocol B uncertain"


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    summary: dict = {"combos": {}}
    for combo in COMBOS:
        entry: dict = {}
        for engine in ("tabpfn", "tabicl", "catboost"):
            a = protocol_a(combo, engine)
            b = protocol_b(combo, engine, a["c1a_by_family"] if a else None)
            if a is None and b is None:
                continue
            e: dict = {}
            if a is not None:
                a["t2"].to_csv(OUT / "tables" / f"T2_{combo}_{engine}.csv", index=False)
                if len(a["t3"]):
                    a["t3"].to_csv(OUT / "tables" / f"T3_{combo}_{engine}.csv", index=False)
                e["C1_A_all_top1"] = a["c1_all_top1"]
                if engine != "catboost":
                    e["rule_A"] = rule_a(a["t2"], a["t3"])
                    e["t2"] = a["t2"].to_dict(orient="records")
                    e["t3"] = a["t3"].to_dict(orient="records")
            if b is not None:
                b["t4"].to_csv(OUT / "tables" / f"T4_{combo}_{engine}.csv", index=False)
                e["B"] = {k: v for k, v in b.items() if k != "t4"}
                e["t4"] = b["t4"].to_dict(orient="records")
                if engine != "catboost" and "gap_pp" in b:
                    e["rule_B"] = rule_b(b)
            if "rule_A" in e and "rule_B" in e:
                e["decision"], e["route"] = combine(e["rule_A"], e["rule_B"])
            entry[engine] = e
        if entry:
            summary["combos"][combo] = entry

    # consistency checks (plan 4.3)
    checks = []
    for cls in ("AB", "ABX3"):
        main = f"{cls}_L2_P1"
        for diff in ("d23_k20", "d24_k20"):

            def sgn(combo, eng="tabpfn"):
                r = summary["combos"].get(combo, {}).get(eng, {}).get("rule_A")
                return np.sign(r[diff]) if r else None

            checks.append(
                {
                    "class": cls,
                    "check": f"engine sign ({diff})",
                    "pass": sgn(main) is not None and sgn(main) == sgn(main, "tabicl"),
                }
            )
            checks.append(
                {
                    "class": cls,
                    "check": f"label scheme sign ({diff})",
                    "pass": all(sgn(f"{cls}_L1_{p}") == sgn(f"{cls}_L2_{p}") for p in ("P1", "P2")),
                }
            )
            checks.append(
                {
                    "class": cls,
                    "check": f"partition sign ({diff})",
                    "pass": all(sgn(f"{cls}_{lab}_P1") == sgn(f"{cls}_{lab}_P2") for lab in ("L1", "L2")),
                }
            )
        for combo in [c for c in COMBOS if c.startswith(cls + "_")]:
            ce = summary["combos"].get(combo, {})
            tp, cb = ce.get("tabpfn", {}), ce.get("catboost", {})
            if "C1_A_all_top1" in tp and "C1_A_all_top1" in cb:
                ga = abs(tp["C1_A_all_top1"][0] - cb["C1_A_all_top1"][0])
                gb = abs(tp["B"]["weighted_C1_B"] - cb["B"]["weighted_C1_B"])
                checks.append(
                    {
                        "class": cls,
                        "check": f"CatBoost vs TabPFN C1 ({combo})",
                        "pass": bool(ga < 0.05 and gb < 0.05),
                        "gap_A": ga,
                        "gap_B": gb,
                    }
                )
            if "t2" in tp:
                sf = {r["k"]: r["same_family_share"] for r in tp["t2"] if r["condition"] == "C4"}
                checks.append(
                    {
                        "class": cls,
                        "check": f"C4 same-family share < 0.8 ({combo})",
                        "pass": bool(all(v < 0.8 for v in sf.values())),
                        "by_k": sf,
                    }
                )
    summary["consistency"] = checks
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    )
    for combo, ce in summary["combos"].items():
        tp = ce.get("tabpfn", {})
        print(
            combo,
            tp.get("rule_A", {}).get("rule"),
            tp.get("rule_B", {}).get("rule"),
            tp.get("decision"),
            tp.get("route"),
        )


if __name__ == "__main__":
    main()
