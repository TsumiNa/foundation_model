from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

_FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"

_MAGNETIC_MOMENT_RE = re.compile(
    rf"^\s*({_FLOAT_RE})(?:\s*±\s*{_FLOAT_RE})?\s*[μµ]B\s*/\s*f\.?\s*u\.?\s*(?P<tail>.*)$",
    re.IGNORECASE,
)
_EMU_PER_G_RE = re.compile(
    rf"^\s*({_FLOAT_RE})(?:\s*±\s*{_FLOAT_RE})?\s*emu\s*/\s*g\s*(?P<tail>.*)$",
    re.IGNORECASE,
)
_A_M2_PER_KG_RE = re.compile(
    rf"^\s*({_FLOAT_RE})(?:\s*±\s*{_FLOAT_RE})?\s*A\s*m(?:\^?2|²)\s*/\s*kg\s*(?P<tail>.*)$",
    re.IGNORECASE,
)
_A_M2_PER_MOL_RE = re.compile(
    rf"^\s*({_FLOAT_RE})(?:\s*±\s*{_FLOAT_RE})?\s*A\s*[·.]?\s*m(?:\^?2|²)\s*/\s*mol\s*(?P<tail>.*)$",
    re.IGNORECASE,
)
_TEMPERATURE_IN_K_RE = re.compile(r"(?<![A-Za-z])K\b", re.IGNORECASE)
_MEASUREMENT_UNIT_RE = re.compile(
    r"[μµ]B\s*/\s*f\.?\s*u\.?|emu\s*/\s*g|A\s*[·.]?\s*m(?:\^?2|²)\s*/\s*(?:kg|mol)",
    re.IGNORECASE,
)
_CANONICAL_COMPOSITION_COLUMN = "Material_Name"
_LEGACY_COMPOSITION_COLUMN = "New_Column_Concatenated"


def convert_mass_magnetization(
    comp: Composition, mass_magnetization: float | np.ndarray, cgs_to_si: bool = True
) -> float | np.ndarray:
    molar_mass = comp.weight
    magnetization_array = np.asarray(mass_magnetization, dtype=float)

    if cgs_to_si:
        result_array = magnetization_array * molar_mass * 1e-3
    else:
        result_array = magnetization_array * 1e3 / molar_mass

    if result_array.ndim == 0:
        return float(result_array)
    return result_array


def convert_composition(
    comp: str, require_computable_mass: bool = True, as_formula: bool = True
) -> str | Composition | None:
    try:
        composition = Composition(comp)
    except Exception:
        return None

    if len(composition) == 0:
        return None

    if require_computable_mass:
        try:
            composition.weight
        except Exception:
            return None

    if as_formula:
        return composition.formula

    return composition


def load_and_clean_magnetic_csv(path: str | Path) -> pd.DataFrame:
    return clean_magnetic_dataframe(pd.read_csv(path))


def clean_magnetic_dataframe(
    magnetic: pd.DataFrame,
    composition_column: str = _CANONICAL_COMPOSITION_COLUMN,
) -> pd.DataFrame:
    if (
        composition_column == _CANONICAL_COMPOSITION_COLUMN
        and _CANONICAL_COMPOSITION_COLUMN not in magnetic.columns
        and _LEGACY_COMPOSITION_COLUMN in magnetic.columns
    ):
        magnetic = magnetic.rename(columns={_LEGACY_COMPOSITION_COLUMN: _CANONICAL_COMPOSITION_COLUMN})

    required_columns = {
        composition_column,
        "Magnetic_Moment",
        "Magnetization",
        "Curie",
        "Curie(Tc)",
        "Neel",
        "Neel(Tn)",
    }
    missing_columns = required_columns.difference(magnetic.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = magnetic.copy()
    compositions = cleaned[composition_column].map(_convert_composition_for_mass)

    cleaned["Magnetic_Moment"] = cleaned["Magnetic_Moment"].map(_parse_magnetic_moment)
    cleaned["Magnetization"] = [
        _parse_magnetization(value, comp) for value, comp in zip(cleaned["Magnetization"], compositions, strict=False)
    ]

    curie_mask = cleaned["Curie(Tc)"].map(_is_kelvin_temperature)
    cleaned["Curie"] = pd.to_numeric(cleaned["Curie"].where(curie_mask), errors="coerce")

    neel_mask = cleaned["Neel(Tn)"].map(_is_kelvin_temperature)
    cleaned["Neel"] = pd.to_numeric(cleaned["Neel"].where(neel_mask), errors="coerce")

    return cleaned


def _is_kelvin_temperature(value: object) -> bool:
    if _is_missing_value(value):
        return False

    text = _normalize_text(str(value))
    if "°c" in text.lower() or "deg c" in text.lower():
        return False
    return bool(_TEMPERATURE_IN_K_RE.search(text))


def _parse_magnetic_moment(value: object) -> float:
    parsed = _parse_measurement(value, _MAGNETIC_MOMENT_RE)
    return float("nan") if parsed is None else parsed


def _parse_magnetization(value: object, comp: Composition | None) -> float:
    emu_per_g = _parse_measurement(value, _EMU_PER_G_RE)
    if emu_per_g is not None:
        return _convert_mass_value_to_molar(emu_per_g, comp)

    a_m2_per_kg = _parse_measurement(value, _A_M2_PER_KG_RE)
    if a_m2_per_kg is not None:
        return _convert_mass_value_to_molar(a_m2_per_kg, comp)

    a_m2_per_mol = _parse_measurement(value, _A_M2_PER_MOL_RE)
    if a_m2_per_mol is not None:
        return a_m2_per_mol

    return float("nan")


def _convert_mass_value_to_molar(value: float, comp: Composition | None) -> float:
    if comp is None or not math.isfinite(value):
        return float("nan")
    return float(convert_mass_magnetization(comp, value, cgs_to_si=True))


def _convert_composition_for_mass(comp: str) -> Composition | None:
    composition = convert_composition(comp, require_computable_mass=True, as_formula=False)
    return composition if isinstance(composition, Composition) else None


def _parse_measurement(value: object, pattern: re.Pattern[str]) -> float | None:
    if _is_missing_value(value):
        return None

    normalized = _normalize_text(str(value))
    match = pattern.match(normalized)
    if match is None:
        return None

    tail = match.group("tail")
    if _has_ambiguous_tail(tail):
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def _has_ambiguous_tail(tail: str) -> bool:
    stripped_tail = tail.strip()
    if not stripped_tail:
        return False

    if re.search(r",|;|\bto\b", stripped_tail, re.IGNORECASE):
        return True

    return bool(_MEASUREMENT_UNIT_RE.search(stripped_tail))


def _normalize_text(text: str) -> str:
    return (
        text.replace("µ", "μ")
        .replace("·", " ")
        .replace("⋅", " ")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("²", "²")
        .replace("\u00a0", " ")
        .strip()
    )


def _is_missing_value(value: object) -> bool:
    if value is None or value is pd.NA:
        return True

    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))

    return False
