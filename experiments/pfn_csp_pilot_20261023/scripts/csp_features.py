"""Features (plan 2.4) and family partitions (plan 2.5)."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.utils.data import MagpieData
from pymatgen.core import Composition, Element

_MAGPIE = MagpieData()
_ROMAN = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIV": 14,
}


def magpie_features(formulas: list[str]) -> pd.DataFrame:
    ep = ElementProperty.from_preset("magpie")
    df = pd.DataFrame({"composition": [Composition(f) for f in formulas]})
    out = ep.featurize_dataframe(df, "composition", ignore_errors=False, pbar=False)
    feats = out.drop(columns=["composition"])
    feats.columns = [f"magpie::{c.replace('MagpieData ', '')}" for c in feats.columns]
    return feats.reset_index(drop=True)


def _num(v) -> float:
    return float(v) if v is not None else math.nan


def site_features(roles: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for role, sym in roles.items():
        el = Element(sym)
        out[f"site::{role}_mendeleev_no"] = _num(el.mendeleev_no)
        out[f"site::{role}_X"] = _num(el.X)
        out[f"site::{role}_atomic_radius"] = _num(el.atomic_radius)
        out[f"site::{role}_row"] = float(el.row)
        out[f"site::{role}_group"] = float(el.group)
        out[f"site::{role}_NValence"] = _num(_MAGPIE.get_elemental_property(el, "NValence"))
    out["site::rA_over_rB"] = out["site::A_atomic_radius"] / out["site::B_atomic_radius"]
    if "X" in roles:
        out["site::rB_over_rX"] = out["site::B_atomic_radius"] / out["site::X_atomic_radius"]
    return out


def shannon_radius(symbol: str, oxi: float, cn: str, highest_if_missing: bool) -> float:
    """Shannon ionic radius; high spin preferred; for A the highest available CN when XII is absent."""
    if oxi != int(oxi):
        return math.nan
    table = Element(symbol).data.get("Shannon radii", {}).get(str(int(oxi)))
    if not table:
        return math.nan
    if cn not in table:
        if not highest_if_missing:
            return math.nan
        cn = max(table, key=lambda c: _ROMAN.get(c, 0))
    spins = table[cn]
    for spin in ("High Spin", ""):
        if spin in spins:
            return float(spins[spin]["ionic_radius"])
    return float(next(iter(spins.values()))["ionic_radius"])


def goldschmidt(formula: str, roles: dict[str, str]) -> tuple[float, str]:
    """t = (r_A + r_X) / (sqrt(2) (r_B + r_X)); oxidation states = first oxi_state_guesses() solution."""
    guesses = Composition(formula).oxi_state_guesses()
    if not guesses:
        return math.nan, "no oxidation-state solution"
    g = guesses[0]
    ra = shannon_radius(roles["A"], g[roles["A"]], "XII", highest_if_missing=True)
    rb = shannon_radius(roles["B"], g[roles["B"]], "VI", highest_if_missing=False)
    rx = shannon_radius(roles["X"], g[roles["X"]], "VI", highest_if_missing=False)
    if any(math.isnan(r) for r in (ra, rb, rx)):
        return math.nan, "no Shannon radius for the guessed state"
    return (ra + rx) / (math.sqrt(2) * (rb + rx)), ""


def p1_family(roles: dict[str, str], cls: str) -> str:
    if cls == "ABX3":
        return f"X={roles['X']}"
    b = Element(roles["B"])
    if b.is_metal:
        return "intermetallic"
    names = {13: "boron group", 14: "carbon group", 15: "pnictogen", 16: "chalcogen", 17: "halogen"}
    return names.get(b.group, f"group {b.group}")


def p1x_family(roles: dict[str, str], cls: str) -> str:
    """Follow-up partition: P1 with the AB intermetallic family split by the blocks of A and B.

    Added after the pilot (README, follow-up 2) because one intermetallic family held half of AB.
    """
    fam = p1_family(roles, cls)
    if fam != "intermetallic":
        return fam
    return f"intermetallic {Element(roles['A']).block}-{Element(roles['B']).block}"


def p2_communities(role_table: pd.DataFrame, role_cols: list[str]) -> list[int]:
    """Substitution graph (exactly one role element differs) -> Louvain, resolution 1.0, seed 0."""
    g = nx.Graph()
    g.add_nodes_from(range(len(role_table)))
    for drop in role_cols:
        keep = [c for c in role_cols if c != drop]
        for _, idx in role_table.groupby(keep).groups.items():
            idx = list(idx)
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    g.add_edge(idx[i], idx[j])
    comms = nx.community.louvain_communities(g, resolution=1.0, seed=0)
    comms = sorted(comms, key=lambda c: (-len(c), min(c)))
    label = [0] * len(role_table)
    for k, c in enumerate(comms):
        for i in c:
            label[i] = k
    return label


def zscore_imputed(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Median-impute NaN then z-score; statistics from `ref` (default: x itself)."""
    ref = x if ref is None else ref
    med = np.nanmedian(ref, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    xr = np.where(np.isnan(ref), med, ref)
    mu, sd = xr.mean(axis=0), xr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    xi = np.where(np.isnan(x), med, x)
    return (xi - mu) / sd


def merge_small_families(fam: list[str], feats: np.ndarray, min_size: int = 15):
    """Merge families with < min_size members into the nearest-centroid family with >= min_size.

    Smallest family first; centroids recomputed after every merge; a small family with no
    eligible target is dropped (label None). Returns (new labels, log rows).
    """
    z = zscore_imputed(feats)
    lab = np.array(fam, dtype=object)
    log = []
    while True:
        names, counts = np.unique(lab[lab != None], return_counts=True)  # noqa: E711
        size = dict(zip(names, counts))
        small = sorted([n for n in names if size[n] < min_size], key=lambda n: (size[n], str(n)))
        if not small:
            break
        src = small[0]
        targets = [n for n in names if size[n] >= min_size]
        if not targets:
            lab[lab == src] = None
            log.append({"family": src, "n": int(size[src]), "action": "dropped", "into": ""})
            continue
        c_src = z[lab == src].mean(axis=0)
        dist = {t: float(np.linalg.norm(z[lab == t].mean(axis=0) - c_src)) for t in targets}
        dst = min(dist, key=lambda t: (dist[t], str(t)))
        lab[lab == src] = dst
        log.append(
            {"family": src, "n": int(size[src]), "action": "merged", "into": dst, "centroid_distance": dist[dst]}
        )
    return list(lab), log
