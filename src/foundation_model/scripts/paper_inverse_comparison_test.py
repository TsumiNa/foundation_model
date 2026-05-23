# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure helpers in :mod:`paper_inverse_comparison`.

The main ``run()`` function needs a trained checkpoint + KMD kernel to exercise end-to-end (see
the smoke runs under ``artifacts/inverse_design_run/``); this file targets the *units that don't*
need either — the formula parser, and the two output plot helpers we added in this PR.
"""

from __future__ import annotations

from foundation_model.scripts.paper_inverse_comparison import (
    _parse_formula_to_fractions,
    _plot_seed_to_optimized_mapping,
)


# --- _parse_formula_to_fractions ----------------------------------------------------------


def test_parse_raw_amount_formula_normalises_to_fractions():
    # Seeds typically come in raw-amount form like "Au65 Ga20 Gd15"; the parser must normalise
    # so the same downstream code can read it as fractions.
    out = _parse_formula_to_fractions("Au65 Ga20 Gd15")
    assert sorted(out.keys()) == ["Au", "Ga", "Gd"]
    assert abs(sum(out.values()) - 1.0) < 1e-12
    assert abs(out["Au"] - 0.65) < 1e-12
    assert abs(out["Ga"] - 0.20) < 1e-12
    assert abs(out["Gd"] - 0.15) < 1e-12


def test_parse_pre_fractional_formula_kept_as_fractions():
    # Decoded compositions land here in fractional form ("Mg0.691 Cd0.309 …"); they must round-trip.
    out = _parse_formula_to_fractions("Mg0.691 Cd0.309")
    assert abs(sum(out.values()) - 1.0) < 1e-12
    assert abs(out["Mg"] - 0.691) < 1e-12
    assert abs(out["Cd"] - 0.309) < 1e-12


def test_parse_handles_missing_amount_as_unit():
    # A bare element symbol ("Mg") gets unit amount, then normalised.
    out = _parse_formula_to_fractions("Mg Cu Ni")
    # 3 elements, equal amounts, fractions = 1/3 each.
    assert sorted(out.keys()) == ["Cu", "Mg", "Ni"]
    for v in out.values():
        assert abs(v - 1.0 / 3.0) < 1e-12


def test_parse_empty_formula_returns_empty_dict():
    assert _parse_formula_to_fractions("") == {}


# --- _plot_seed_to_optimized_mapping ------------------------------------------------------


def test_plot_seed_to_optimized_mapping_writes_png(tmp_path):
    seeds = [
        "Mg12 Cu3 Ni3",
        "Au65 Ga20 Gd15",
        "Al6 Co1 Cu3",
    ]
    decoded = [
        "Mg0.50 Cu0.30 Ni0.20",
        "Au0.55 Ga0.30 Gd0.15",
        "Al0.60 Pd0.20 Ti0.20",  # introduces Pd / Ti not in seeds
    ]
    out = tmp_path / "seed_to_optimized.png"
    _plot_seed_to_optimized_mapping(seeds, decoded, out, title="test scenario")
    assert out.exists()


def test_plot_seed_to_optimized_mapping_skips_on_length_mismatch(tmp_path, caplog):
    """Mismatched seeds / decoded lengths must not crash — log a warning and skip the write."""
    out = tmp_path / "should_not_exist.png"
    _plot_seed_to_optimized_mapping(["Mg1 Cu1"], ["Mg0.5 Cu0.5", "Al1.0"], out, title="bad")
    assert not out.exists()


def test_plot_seed_to_optimized_mapping_skips_on_empty(tmp_path):
    out = tmp_path / "should_not_exist.png"
    _plot_seed_to_optimized_mapping([], [], out, title="empty")
    assert not out.exists()
