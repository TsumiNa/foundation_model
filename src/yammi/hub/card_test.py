# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model card generator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from yammi.hub import read_config, read_manifest
from yammi.hub.card import render_card

from .export_test import _checkpoint, _export, catalog_raw  # noqa: F401  (catalog_raw is a fixture)


@pytest.fixture
def bundle(catalog_raw: dict, tmp_path: Path) -> Path:  # noqa: F811
    runs = [_checkpoint(catalog_raw, tmp_path / f"r{i}.pt", i) for i in range(2)]
    return _export(catalog_raw, runs, tmp_path / "bundle")


def _front(card: str) -> dict[str, Any]:
    assert card.startswith("---\n")
    return yaml.safe_load(card.split("---\n")[1])


def _card(bundle: Path, license_id: str | None = None, *, gated: bool = True) -> str:
    config = read_config(bundle)
    if license_id is not None:
        config = replace(config, license=license_id)
    return render_card(config, read_manifest(bundle), repo_id="owner/fm", gated=gated)


def test_front_matter_and_body(bundle: Path) -> None:
    card = _card(bundle)
    meta = _front(card)
    assert meta["license"] == "cc-by-nc-sa-4.0" and meta["library_name"] == "yammi"
    assert "non-commercial" in meta["extra_gated_prompt"] and "from scratch" in meta["extra_gated_prompt"]
    assert any("non-commercial purposes only" in k for k in meta["extra_gated_fields"])
    assert "| `band_gap` | regression | value | eV |" in card
    assert "dimensionless" in card  # the shape-only DOS head
    assert "temperature ∈ [300" in card and "2 members; default `m000`" in card
    assert 'load_pretrained("owner/fm")' in card and "rehearsal" in card


@pytest.mark.parametrize(
    ("license_id", "says", "never"),
    [
        ("cc-by-nc-sa-4.0", "must carry the same licence", "may not be shared"),
        ("cc-by-nc-4.0", "stay non-commercial", "same licence"),
        ("cc-by-nc-nd-4.0", "may not be shared", "same licence"),
        ("cc-by-4.0", "including commercially", "non-commercial purposes only"),
    ],
)
def test_licence_terms_follow_the_licence(bundle: Path, license_id: str, says: str, never: str) -> None:
    card = _card(bundle, license_id)
    meta = _front(card)
    assert says in meta["extra_gated_prompt"] and says in card.split("## Licence")[1]
    assert never not in meta["extra_gated_prompt"] and never not in card.split("## Licence")[1]
    has_nc_box = any("non-commercial purposes only" in k for k in meta["extra_gated_fields"])
    assert has_nc_box == ("-nc" in license_id)
    # The Hub refuses the upload when any metadata key contains "." or "$".
    assert not [k for k in meta["extra_gated_fields"] if "." in k or "$" in k]


def test_custom_licence_is_quoted_and_not_paraphrased(bundle: Path) -> None:
    card = _card(bundle, "Lab: Research-1.0 #internal")
    meta = _front(card)  # parses: YAML-significant characters are quoted everywhere
    assert meta["license"] == "other" and meta["license_name"] == "Lab: Research-1.0 #internal"
    assert "I accept the licence terms stated above" in meta["extra_gated_fields"]
    assert "its own terms apply" in card and "non-commercial" not in card.split("## Licence")[1]


def test_install_line_points_at_the_code_repository(bundle: Path) -> None:
    card = _card(bundle, "cc-by-4.0")
    assert "pip install 'yammi[hub] @ git+https://github.com/TsumiNa/yammi'" in card
    assert "pip install 'yammi[hub]'\n" not in card  # not on PyPI yet


def test_no_gating_form_when_not_gated(bundle: Path) -> None:
    meta = _front(_card(bundle, gated=False))
    assert "extra_gated_fields" not in meta and "extra_gated_prompt" not in meta


def test_heads_grouped_by_kind_and_long_class_lists_abbreviated(bundle: Path) -> None:
    config = read_config(bundle)
    phase = next(h for h in config.heads if h.kind == "classification")
    assert phase.classes is not None
    many = replace(phase, num_classes=151, classes=replace(phase.classes, classes=tuple(f"c{i}" for i in range(151))))
    config = replace(config, heads=tuple(many if h is phase else h for h in reversed(config.heads)))
    card = render_card(config, read_manifest(bundle), repo_id="owner/fm", gated=True)
    rows = [line for line in card.splitlines() if line.startswith("| `")]
    kinds = [row.split(" | ")[1] for row in rows]
    assert kinds == sorted(kinds, key=["regression", "classification", "kernel regression"].index)
    assert "class: 151 classes" in card and "c150" not in card
    assert "SaturationWarning" in card
