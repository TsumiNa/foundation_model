# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Model card and ``fm publish`` (the Hub API is faked; nothing leaves the machine)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]
from click.testing import CliRunner

from foundation_model.cli.main import main
from foundation_model.hub import read_config, read_manifest
from foundation_model.hub.card import render_card
from foundation_model.hub.publish import publish


from .export_test import _checkpoint, _export, catalog_raw  # noqa: F401  (catalog_raw is a fixture)

# The package re-exports the function `publish`, which shadows the module of the same name.
publish_mod = importlib.import_module("foundation_model.hub.publish")


@pytest.fixture
def bundle(catalog_raw: dict, tmp_path: Path) -> Path:  # noqa: F811
    runs = [_checkpoint(catalog_raw, tmp_path / f"r{i}.pt", i) for i in range(2)]
    return _export(catalog_raw, runs, tmp_path / "bundle")


class FakeApi:
    calls: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def create_repo(self, repo_id: str, **kw: Any) -> str:
        FakeApi.calls.append(("create_repo", {"repo_id": repo_id, **kw}))
        return f"https://huggingface.co/{repo_id}"

    def update_repo_settings(self, repo_id: str, **kw: Any) -> None:
        FakeApi.calls.append(("update_repo_settings", {"repo_id": repo_id, **kw}))

    def upload_folder(self, **kw: Any) -> None:
        FakeApi.calls.append(("upload_folder", kw))

    def create_tag(self, repo_id: str, **kw: Any) -> None:
        FakeApi.calls.append(("create_tag", {"repo_id": repo_id, **kw}))


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> type[FakeApi]:
    FakeApi.calls = []
    monkeypatch.setattr(publish_mod, "_api", lambda token: FakeApi(token))
    return FakeApi


def _front(card: str) -> dict[str, Any]:
    assert card.startswith("---\n")
    return yaml.safe_load(card.split("---\n")[1])


def test_card_front_matter_and_body(bundle: Path) -> None:
    card = render_card(read_config(bundle), read_manifest(bundle), repo_id="owner/fm", gated=True)
    meta = _front(card)
    assert meta["license"] == "cc-by-nc-sa-4.0" and meta["library_name"] == "foundation-model"
    assert "non-commercial" in meta["extra_gated_prompt"] and "from scratch" in meta["extra_gated_prompt"]
    assert any("non-commercial purposes only" in k for k in meta["extra_gated_fields"])
    assert "| `band_gap` | regression | value | eV |" in card
    assert "dimensionless" in card  # the shape-only DOS head
    assert "temperature ∈ [300" in card and "2 members; default `m000`" in card
    assert 'load_pretrained("owner/fm")' in card and "rehearsal" in card


def test_card_without_gating_and_unknown_licence(bundle: Path) -> None:
    config = read_config(bundle)
    from dataclasses import replace

    custom = replace(config, license="Lab-Research-1.0")
    meta = _front(render_card(custom, read_manifest(bundle), gated=False))
    assert meta["license"] == "other" and meta["license_name"] == "Lab-Research-1.0"
    assert "extra_gated_fields" not in meta


def test_publish_gated_with_tag(bundle: Path, fake_api: type[FakeApi], tmp_path: Path) -> None:
    extra = "## Training data\n\n- Materials Project (CC BY 4.0)."
    url = publish(bundle, "owner/fm", tag="v1.0.0", visibility="gated", card_extra=[extra])
    assert url == "https://huggingface.co/owner/fm"
    names = [c[0] for c in fake_api.calls]
    assert names == ["create_repo", "update_repo_settings", "upload_folder", "create_tag"]
    assert fake_api.calls[0][1]["private"] is False
    assert fake_api.calls[1][1] == {"repo_id": "owner/fm", "gated": "manual", "private": False}
    assert fake_api.calls[2][1]["folder_path"] == str(bundle) and fake_api.calls[3][1]["tag"] == "v1.0.0"
    card = (bundle / "README.md").read_text()
    assert "Materials Project (CC BY 4.0)" in card and "extra_gated_fields" in card


def test_publish_defaults_to_private_and_keeps_an_existing_card(bundle: Path, fake_api: type[FakeApi]) -> None:
    (bundle / "README.md").write_text("hand-written card\n")
    publish(bundle, "owner/fm")
    assert fake_api.calls[0][1]["private"] is True
    assert "update_repo_settings" not in [c[0] for c in fake_api.calls] and "create_tag" not in [
        c[0] for c in fake_api.calls
    ]
    assert (bundle / "README.md").read_text() == "hand-written card\n"
    publish(bundle, "owner/fm", overwrite_card=True)
    assert (bundle / "README.md").read_text().startswith("---\n")


def test_dry_run_touches_no_network(bundle: Path, fake_api: type[FakeApi]) -> None:
    result = publish(bundle, "owner/fm", visibility="gated", dry_run=True)
    assert result == str(bundle / "README.md") and fake_api.calls == []


def test_publish_refuses_a_broken_bundle(bundle: Path, fake_api: type[FakeApi]) -> None:
    (bundle / "members" / "m001" / "model.safetensors").write_bytes(b"not weights")
    with pytest.raises(Exception):
        publish(bundle, "owner/fm")
    assert fake_api.calls == []


def test_fm_publish_cli(bundle: Path, fake_api: type[FakeApi], tmp_path: Path) -> None:
    extra = tmp_path / "citations.md"
    extra.write_text("## Citations\n\n- HYPOD-X.\n")
    result = CliRunner().invoke(
        main, ["publish", str(bundle), "--repo", "owner/fm", "--tag", "v1.0.0", "--visibility", "gated",
               "--card-extra", str(extra)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "published to https://huggingface.co/owner/fm" in result.output
    assert "HYPOD-X" in (bundle / "README.md").read_text()
    result = CliRunner().invoke(main, ["publish", str(bundle), "--repo", "owner/fm", "--visibility", "secret"])
    assert result.exit_code != 0
