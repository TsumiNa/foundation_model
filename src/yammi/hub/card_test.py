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
    assert 'load_suite("owner/fm")' in card and 'load_pretrained("owner/fm#m001")' in card and "rehearsal" in card
    assert "yammi-pretrained" in card and ".agents/skills/" in card  # agent skill install hint


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


def test_commercial_contact_offers_a_separate_licence(bundle: Path) -> None:
    config, manifest = read_config(bundle), read_manifest(bundle)
    card = render_card(config, manifest, repo_id="owner/fm", gated=True, commercial_contact="a.b@example.org")
    meta = _front(card)
    assert "separate agreement with the authors is available: contact a.b@example.org." in meta["extra_gated_prompt"]
    nc_box = [k for k in meta["extra_gated_fields"] if "non-commercial purposes only" in k]
    assert nc_box == [
        "I will use these weights and any model derived from them for non-commercial purposes only, "
        "unless I have a separate agreement with the authors"
    ]
    assert not [k for k in meta["extra_gated_fields"] if "." in k or "$" in k]
    licence = card.split("## Licence")[1]
    assert "must carry the same licence" in licence  # the public terms stay as they are
    assert "contact [a.b@example.org](mailto:a.b@example.org)" in licence and "joint research agreement" in licence
    assert "commercial licence by separate agreement (see Licence)" in card
    url = render_card(config, manifest, gated=False, commercial_contact="https://example.org/licensing")
    assert "contact https://example.org/licensing." in url and "mailto:" not in url
    # Without a contact the card says nothing about it.
    plain = _card(bundle)
    assert "separate agreement" not in plain and "joint research" not in plain


@pytest.mark.parametrize(("license_id", "contact"), [("cc-by-4.0", "a@b.org"), ("apache-2.0", "a@b.org"), (None, " ")])
def test_commercial_contact_rejections(bundle: Path, license_id: str | None, contact: str) -> None:
    config = read_config(bundle)
    if license_id is not None:
        config = replace(config, license=license_id)
    with pytest.raises(ValueError, match="commercial_contact"):
        render_card(config, read_manifest(bundle), commercial_contact=contact)


def test_custom_licence_is_quoted_and_not_paraphrased(bundle: Path) -> None:
    card = _card(bundle, "Lab: Research-1.0 #internal")
    meta = _front(card)  # parses: YAML-significant characters are quoted everywhere
    assert meta["license"] == "other" and meta["license_name"] == "Lab: Research-1.0 #internal"
    assert "I accept the licence terms stated above" in meta["extra_gated_fields"]
    assert "its own terms apply" in card and "non-commercial" not in card.split("## Licence")[1]


def test_install_line_points_at_the_code_repository(bundle: Path) -> None:
    card = _card(bundle, "cc-by-4.0")
    assert "git clone https://github.com/TsumiNa/yammi.git" in card and "uv sync --frozen --no-dev --extra hub" in card
    assert "pip install 'yammi[hub] @ git+https://github.com/TsumiNa/yammi'" in card
    assert "pip install 'yammi[hub]'\n" not in card  # never the bare name: on PyPI it is a placeholder


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


def test_publish_will_not_drop_the_commercial_offer_from_a_kept_card(bundle: Path) -> None:
    from yammi.hub.publish import publish

    (bundle / "README.md").write_text("# hand-written card\n")
    with pytest.raises(ValueError, match="overwrite_card"):
        publish(bundle, "owner/fm", commercial_contact="a@b.org", dry_run=True)
    card = publish(
        bundle, "owner/fm", visibility="gated", commercial_contact="a@b.org", overwrite_card=True, dry_run=True
    )
    assert "separate agreement" in Path(card).read_text()
