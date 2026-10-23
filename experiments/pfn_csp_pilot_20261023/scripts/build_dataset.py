"""Build the per-stoichiometry wide tables: labels L1/L2, features, families P1/P2 (plan section 2).

    uv run python scripts/build_dataset.py

Inputs : artifacts/data/mp_raw_{AB,ABX3}.json.gz (fetch_mp.py), configs/references.toml
Outputs: artifacts/data/table_{cls}.parquet, results/appendix/{exclusions,prototype_map,
         family_merges,reference_structures}_{cls}.csv, results/appendix/build_summary.json
"""

from __future__ import annotations

import gzip
import json
import math
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csp_features import (  # noqa: E402
    goldschmidt,
    magpie_features,
    merge_small_families,
    p1_family,
    p2_communities,
    site_features,
)
from csp_labels import (  # noqa: E402
    NOBLE_GASES,
    RoleError,
    assign_roles,
    match_aristotype,
    top_k_labels,
    wyckoff_skeleton,
)

EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "artifacts" / "data"
APPX = EXP / "results" / "appendix"
K0 = {"AB": 12, "ABX3": 8}

_REFS: list[tuple[str, Structure]] = []


def _init_refs(refs: list[tuple[str, dict]]) -> None:
    global _REFS
    _REFS = [(name, Structure.from_dict(d)) for name, d in refs]


def _match(sdict: dict) -> tuple[str | None, float, list[str]]:
    return match_aristotype(Structure.from_dict(sdict), _REFS)


def _mid_num(mid: str) -> int:
    return int(mid.split("-")[1])


def build(cls: str, refs_cfg: dict) -> dict:
    raw = json.load(gzip.open(DATA / f"mp_raw_{cls}.json.gz"))
    by_id = {r["material_id"]: r for r in raw}
    excl, kept = [], []
    for r in raw:
        comp = Composition(r["formula_pretty"])
        syms = {e.symbol for e in comp.elements}

        def drop(reason: str) -> None:
            excl.append(
                {
                    "material_id": r["material_id"],
                    "formula": r["formula_pretty"],
                    "energy_above_hull": r["energy_above_hull"],
                    "reason": reason,
                }
            )

        if syms & NOBLE_GASES:
            drop("contains a noble gas")
            continue
        if len(syms) >= 4:
            drop(">= 4 elements")
            continue
        if not r["icsd_ids"]:
            drop("no ICSD id (secondary theoretical check)")
            continue
        try:
            roles = assign_roles(comp, cls)
        except RoleError as e:
            drop(f"role rule: {e}")
            continue
        if cls == "AB" and roles["B"] == "H":
            drop("hydride (B = H)")
            continue
        sk = wyckoff_skeleton(Structure.from_dict(r["structure"]), roles)
        if sk is None:
            drop("spglib returned no symmetry dataset")
            continue
        kept.append(
            {
                **{k: r[k] for k in ("material_id", "formula_pretty", "energy_above_hull")},
                "reduced_formula": comp.reduced_formula,
                "roles": roles,
                "spacegroup": sk.spacegroup,
                "skeleton": sk.string,
                "icsd_ids": ";".join(r["icsd_ids"]),
            }
        )
    df = pd.DataFrame(kept)
    df["_mid"] = df["material_id"].map(_mid_num)
    df = df.sort_values(["reduced_formula", "energy_above_hull", "_mid"])
    dup = df[df.duplicated("reduced_formula", keep="first")]
    for _, r in dup.iterrows():
        excl.append(
            {
                "material_id": r["material_id"],
                "formula": r["formula_pretty"],
                "energy_above_hull": r["energy_above_hull"],
                "reason": "not the lowest energy_above_hull entry of its composition",
            }
        )
    df = df.drop_duplicates("reduced_formula", keep="first").drop(columns="_mid")
    df = df.sort_values("reduced_formula").reset_index(drop=True)
    for role in df["roles"].iloc[0]:
        df[role] = df["roles"].map(lambda d, role=role: d[role])
    df = df.drop(columns="roles")
    role_cols = ["A", "B"] + (["X"] if cls == "ABX3" else [])

    # References: skeleton -> prototype name for display; structures for L2 matching.
    ref_rows, refs = [], []
    for name, spec in refs_cfg.items():
        r = by_id[spec["material_id"]]
        assert r["formula_pretty"] == spec["formula"], (name, r["formula_pretty"])
        roles = assign_roles(Composition(r["formula_pretty"]), cls)
        sk = wyckoff_skeleton(Structure.from_dict(r["structure"]), roles)
        ref_rows.append(
            {
                "prototype": name,
                "aristotype": spec["aristotype"],
                "material_id": spec["material_id"],
                "formula": spec["formula"],
                "skeleton": sk.string,
            }
        )
        refs.append((spec["aristotype"], r["structure"]))
    proto_name = {}
    for rr in ref_rows:
        proto_name.setdefault(rr["skeleton"], rr["prototype"])

    cache = DATA / f"l2_matches_{cls}.json"
    ref_key = json.dumps(sorted((n, s["material_id"]) for n, s in refs_cfg.items()))
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    if cached.get("ref_key") != ref_key:
        cached = {"ref_key": ref_key, "matches": {}}
    todo = [m for m in df["material_id"] if m not in cached["matches"]]
    if todo:
        structs = [by_id[m]["structure"] for m in todo]
        with ProcessPoolExecutor(max_workers=14, initializer=_init_refs, initargs=(refs,)) as ex:
            for mid, res in zip(todo, ex.map(_match, structs, chunksize=1)):
                cached["matches"][mid] = list(res)
        cache.write_text(json.dumps(cached))
    matches = [cached["matches"][m] for m in df["material_id"]]
    timeouts = [
        {"material_id": m, "formula": f, "timed_out_references": ";".join(t[2])}
        for m, f, t in zip(df["material_id"], df["formula_pretty"], matches)
        if t[2]
    ]
    df["L1_raw"] = df["skeleton"]
    df["L1_name"] = df["skeleton"].map(lambda s: proto_name.get(s, ""))
    df["L2_match"] = [m[0] or "" for m in matches]
    df["L2_rms"] = [m[1] if m[1] is not None else math.nan for m in matches]
    df["L2_raw"] = [m[0] if m[0] else s for m, s in zip(matches, df["skeleton"])]
    df["L1"], k1, o1 = top_k_labels(list(df["L1_raw"]), K0[cls])
    df["L2"], k2, o2 = top_k_labels(list(df["L2_raw"]), K0[cls])

    # Features (plan 2.4)
    mag = magpie_features(list(df["reduced_formula"]))
    site = pd.DataFrame([site_features({r: row[r] for r in role_cols}) for _, row in df.iterrows()])
    feats = pd.concat([mag, site], axis=1)
    tol_nan_reasons: dict[str, int] = {}
    if cls == "ABX3":
        tol = []
        for _, row in df.iterrows():
            t, why = goldschmidt(row["reduced_formula"], {r: row[r] for r in role_cols})
            tol.append(t)
            if why:
                tol_nan_reasons[why] = tol_nan_reasons.get(why, 0) + 1
        feats["site::goldschmidt_t"] = tol
    feat_cols = list(feats.columns)
    df = pd.concat([df, feats], axis=1)

    # Families (plan 2.5)
    xf = df[feat_cols].to_numpy(dtype=float)
    df["P1_raw"] = [p1_family({r: row[r] for r in role_cols}, cls) for _, row in df.iterrows()]
    df["P2_raw"] = [f"community {c}" for c in p2_communities(df[role_cols], role_cols)]
    merges = []
    for p in ("P1", "P2"):
        lab, log = merge_small_families(list(df[f"{p}_raw"]), xf)
        df[p] = lab
        merges += [{"partition": p, **m} for m in log]

    APPX.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA / f"table_{cls}.parquet", index=False)
    pd.DataFrame(excl).to_csv(APPX / f"exclusions_{cls}.csv", index=False)
    pd.DataFrame(merges).to_csv(APPX / f"family_merges_{cls}.csv", index=False)
    pd.DataFrame(ref_rows).to_csv(APPX / f"reference_structures_{cls}.csv", index=False)
    pd.DataFrame(timeouts, columns=["material_id", "formula", "timed_out_references"]).to_csv(
        APPX / f"l2_match_timeouts_{cls}.csv", index=False
    )
    pmap = (
        df.groupby(["L1_raw", "L1_name", "L2_match"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    )
    pmap.to_csv(APPX / f"prototype_map_{cls}.csv", index=False)

    def fam_stats(p: str) -> dict:
        s = df[p].dropna().value_counts()
        return {
            "n_families": int(len(s)),
            "min": int(s.min()),
            "max": int(s.max()),
            "dropped_rows": int(df[p].isna().sum()),
            "sizes": {str(k): int(v) for k, v in s.items()},
        }

    return {
        "N": int(len(df)),
        "n_raw": len(raw),
        "n_excluded": len(excl),
        "exclusion_reasons": pd.DataFrame(excl)["reason"].value_counts().to_dict(),
        "L1": {"K": k1, "other_share": o1, "n_raw_classes": int(df["L1_raw"].nunique())},
        "L2": {"K": k2, "other_share": o2, "n_raw_classes": int(df["L2_raw"].nunique())},
        "P1": fam_stats("P1"),
        "P2": fam_stats("P2"),
        "n_features": len(feat_cols),
        "l2_match_timeouts": len(timeouts),
        "goldschmidt_nan_share": (float(np.isnan(df["site::goldschmidt_t"]).mean()) if cls == "ABX3" else None),
        "goldschmidt_nan_reasons": tol_nan_reasons,
        "feature_nan_share": float(np.isnan(xf).mean()),
    }


def main() -> None:
    refs_all = tomllib.loads((EXP / "configs" / "references.toml").read_text())
    summary = {cls: build(cls, refs_all[cls]) for cls in ("AB", "ABX3")}
    summary["fetch"] = json.loads((DATA / "mp_fetch_meta.json").read_text())
    (APPX / "build_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(
        json.dumps(
            {c: {k: v for k, v in s.items() if k not in ("P1", "P2")} for c, s in summary.items() if c != "fetch"},
            indent=1,
            default=float,
        )
    )
    for c in ("AB", "ABX3"):
        print(c, "P1", summary[c]["P1"]["sizes"])
        print(c, "P2", summary[c]["P2"]["n_families"], summary[c]["P2"]["min"], summary[c]["P2"]["max"])


if __name__ == "__main__":
    main()
