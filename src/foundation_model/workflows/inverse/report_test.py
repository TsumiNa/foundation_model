# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the report's text formatting.

The seed→optimised map lays its rows out by hand — measured widths, kind-aware number formats,
shortened task names — because matplotlib will happily let two columns overlap. These pin the
pieces that decide the layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from foundation_model.workflows.inverse.report import (
    _format_channel,
    _format_channel_delta,
    _row_parts,
    _row_width_pt,
    _short_task,
    _target_direction_arrow,
    _top_fractions,
)

from .conftest import _catalog, _spec


def test_short_task_names_stay_compact() -> None:
    assert _short_task("formation_energy") == "FE"  # curated override
    assert _short_task("thermal_conductivity_300k") == "TC3"  # initials of the underscore parts
    assert _short_task("tc") == "tc"  # already short
    assert _short_task("magnetisability") == "magn"  # single long word → prefix


def test_channel_formatting_is_kind_aware(data_dir) -> None:
    cat = _catalog(data_dir)
    cls = _spec(cat, task="mat", classes=[1])
    reg = _spec(cat, task="b", direction="low")
    # Class channels are probabilities: shown as percent, moved in percentage points.
    assert _format_channel(cls, 0.886) == "88.6%" and _format_channel_delta(cls, -0.087) == "-8.7pp"
    assert _format_channel(reg, -1.234) == "-1.23" and _format_channel_delta(reg, 0.5) == "+0.50"


def test_direction_arrows_match_target_intent(data_dir) -> None:
    cat = _catalog(data_dir)
    assert _target_direction_arrow(_spec(cat, task="b", direction="high")) == "↑"
    assert _target_direction_arrow(_spec(cat, task="b", direction="low")) == "↓"
    assert _target_direction_arrow(_spec(cat, task="mat", classes=[1])) == "↑"
    assert _target_direction_arrow(_spec(cat, task="k", points=[[0.0, 1.0]])) == "↓"  # RMSE → 0
    assert _target_direction_arrow(_spec(cat, task="a", value=-1.0)) == ""  # point goal, no direction


def test_top_fractions_truncates_and_renormalises() -> None:
    from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS

    row = np.zeros(len(DEFAULT_ELEMENTS))
    for i in range(8):  # 8 elements at descending weights, summing to 1
        row[i] = (8 - i) / 36
    top = _top_fractions(row, top_k=6)
    assert len(top) == 6  # the two smallest are dropped, matching `format_weights`
    assert sum(top.values()) == pytest.approx(1.0)  # ... and what is left is renormalised
    assert list(top) == [DEFAULT_ELEMENTS[i] for i in range(6)]  # largest first
    assert len(_top_fractions(row, top_k=None)) == 8  # seeds keep every element above eps


def test_row_width_grows_with_target_count() -> None:
    # The map sizes its columns from the text; more targets ⇒ a longer parenthetical ⇒ a wider
    # column. Without that the two columns collide as soon as a scenario adds a target.
    comp = {"Ti": 0.6, "Al": 0.4}
    narrow = _row_width_pt(_row_parts(comp, " (obj=0.20, FE=0.83)", None))
    wide = _row_width_pt(_row_parts(comp, " (obj=0.20, FE=0.83, mag=-0.14, QC=88.6%)", None))
    assert wide > narrow > 0
