"""Site roles, Wyckoff skeletons and aristotype matching (plan sections 2.1-2.3)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import spglib
from pymatgen.analysis.structure_matcher import FrameworkComparator, StructureMatcher
from pymatgen.core import Composition, Element, Structure

SYMPREC = 0.1
NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"}
ABX3_ANIONS = {"O", "F", "Cl", "Br", "I", "S", "Se"}


def _radius(el: Element) -> float:
    r = el.atomic_radius
    return float(r) if r is not None else math.nan


def _x(el: Element) -> float:
    x = el.X
    return float(x) if x is not None else math.nan


class RoleError(ValueError):
    """The composition does not admit the role assignment of plan section 2.2."""


def assign_roles(comp: Composition, cls: str) -> dict[str, str]:
    """Return {"A": el, "B": el[, "X": el]} following the plan 2.2 table, or raise RoleError."""
    # Ratios from the raw amounts: pymatgen keeps peroxide-like formulas (K2O2) unreduced.
    raw = comp.get_el_amt_dict()
    lo = min(raw.values())
    amounts = {el: round(v / lo, 6) for el, v in raw.items()}
    els = [Element(e) for e in amounts]
    if any(math.isnan(_x(e)) for e in els):
        raise RoleError("missing Pauling electronegativity")
    if cls == "AB":
        if len(els) != 2 or sorted(amounts.values()) != [1.0, 1.0]:
            raise RoleError("not a 1:1 binary")
        e1, e2 = els
        if _x(e1) != _x(e2):
            a, b = (e1, e2) if _x(e1) < _x(e2) else (e2, e1)
        else:
            r1, r2 = _radius(e1), _radius(e2)
            if math.isnan(r1) or math.isnan(r2) or r1 == r2:
                raise RoleError("electronegativity and radius tie")
            a, b = (e1, e2) if r1 > r2 else (e2, e1)
        return {"A": a.symbol, "B": b.symbol}
    if cls == "ABX3":
        if len(els) != 3 or sorted(amounts.values()) != [1.0, 1.0, 3.0]:
            raise RoleError("not a 1:1:3 ternary")
        x3 = [e for e in els if amounts[e.symbol] == 3.0][0]
        most_en = max(els, key=_x)
        if x3 != most_en or sum(_x(e) == _x(most_en) for e in els) > 1:
            raise RoleError("coefficient-3 element is not the most electronegative")
        if x3.symbol not in ABX3_ANIONS:
            raise RoleError(f"X={x3.symbol} not in the anion list")
        e1, e2 = [e for e in els if e != x3]
        r1, r2 = _radius(e1), _radius(e2)
        if math.isnan(r1) or math.isnan(r2):
            raise RoleError("missing atomic radius")
        if r1 != r2:
            a, b = (e1, e2) if r1 > r2 else (e2, e1)
        else:
            if _x(e1) == _x(e2):
                raise RoleError("radius and electronegativity tie")
            a, b = (e1, e2) if _x(e1) < _x(e2) else (e2, e1)
        return {"A": a.symbol, "B": b.symbol, "X": x3.symbol}
    raise ValueError(cls)


@dataclass(frozen=True)
class Skeleton:
    spacegroup: int
    string: str


def strip_orientation(symbol: str) -> str:
    """Site-symmetry symbol without spglib's orientation dots ('4/mm.m' -> '4/mmm')."""
    return symbol.replace(".", "")


def wyckoff_skeleton(structure: Structure, roles: dict[str, str]) -> Skeleton | None:
    """Space group + sorted (multiplicity, site symmetry) per role on the conventional cell.

    Returns None when spglib finds no symmetry dataset (plan 2.1 step 3).
    """
    cell = (structure.lattice.matrix, structure.frac_coords, [sp.Z for sp in structure.species])
    std = spglib.standardize_cell(cell, to_primitive=False, no_idealize=False, symprec=SYMPREC)
    if std is None:
        return None
    ds = spglib.get_symmetry_dataset(std, symprec=SYMPREC)
    if ds is None:
        return None
    numbers = np.asarray(std[2])
    orbits = np.asarray(ds.equivalent_atoms)
    sym_by_z = {Element(el).Z: role for role, el in roles.items()}
    per_role: dict[str, list[tuple[int, str]]] = {role: [] for role in roles}
    for rep in np.unique(orbits):
        members = np.flatnonzero(orbits == rep)
        role = sym_by_z[int(numbers[rep])]
        per_role[role].append((len(members), strip_orientation(ds.site_symmetry_symbols[rep])))
    parts = [str(ds.number)]
    for role in roles:
        sites = sorted(per_role[role])
        parts.append(f"{role}:" + "+".join(f"{m}({s})" for m, s in sites))
    return Skeleton(int(ds.number), "|".join(parts))


MATCHER = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, attempt_supercell=True, comparator=FrameworkComparator())


class _MatchTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _MatchTimeout


def match_aristotype(
    structure: Structure, references: list[tuple[str, Structure]], timeout_s: int = 60
) -> tuple[str | None, float, list[str]]:
    """Best-matching reference aristotype (smallest RMS distance), or (None, nan).

    Each (structure, reference) comparison is capped at `timeout_s` seconds (SIGALRM, so call
    it from a process's main thread); a comparison that times out counts as no match and its
    reference name is returned in the third element so the caller can log it.
    """
    import signal

    best: tuple[str | None, float] = (None, math.inf)
    timed_out: list[str] = []
    old = signal.signal(signal.SIGALRM, _alarm)
    try:
        for name, ref in references:
            signal.alarm(timeout_s)
            try:
                if not MATCHER.fit(structure, ref):
                    continue
                rms = MATCHER.get_rms_dist(structure, ref)
            except _MatchTimeout:
                timed_out.append(name)
                continue
            finally:
                signal.alarm(0)
            d = rms[0] if rms is not None else math.inf
            if d < best[1]:
                best = (name, d)
    finally:
        signal.signal(signal.SIGALRM, old)
    if best[0] is None:
        return None, math.nan, timed_out
    return best[0], best[1], timed_out


def top_k_labels(labels: list[str], k0: int, other_cap: float = 0.30, k_max: int = 20):
    """Keep the k most frequent labels, the rest -> 'other'; grow k by 2 while other > cap."""
    from collections import Counter

    counts = Counter(labels)
    ranked = [lab for lab, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    k = k0
    while True:
        keep = set(ranked[:k])
        other = sum(c for lab, c in counts.items() if lab not in keep) / len(labels)
        if other <= other_cap or k >= k_max:
            break
        k = min(k + 2, k_max)
    return [lab if lab in keep else "other" for lab in labels], k, other
