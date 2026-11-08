# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``yammi publish`` — write the model card and push a bundle to a Hugging Face repository.

Visibility is one of ``private`` (default), ``gated`` (public page, downloads after an approved
request) or ``public``. Every member is loaded back strictly before anything is uploaded. A tag such
as ``v1.0.0`` pins the release. The token comes from ``HF_TOKEN`` / ``hf auth login`` unless given.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from .bundle import BundleError, load_member, read_config, read_manifest
from .card import render_card

__all__ = ["Visibility", "publish"]

Visibility = Literal["private", "gated", "public"]
CARD_FILE = "README.md"


def _api(token: str | None) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("publishing needs the optional extra: pip install 'yammi[hub]'") from exc
    return HfApi(token=token)


def publish(
    bundle: str | Path,
    repo_id: str,
    *,
    tag: str | None = None,
    visibility: Visibility = "private",
    title: str | None = None,
    card_extra: Sequence[str] = (),
    overwrite_card: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
    dry_run: bool = False,
) -> str:
    """Check, card and upload ``bundle`` to ``repo_id``; returns the repository URL.

    ``dry_run`` stops after writing the card and returns its path: nothing touches the network.
    """
    if visibility not in ("private", "gated", "public"):
        raise ValueError(f"visibility must be private, gated or public, got {visibility!r}.")
    path = Path(bundle)
    config, manifest = read_config(path), read_manifest(path)
    for member in manifest.members:  # never upload a bundle that does not load
        load_member(path, member.id)

    card = path / CARD_FILE
    if card.exists() and not overwrite_card:
        logger.info(f"keeping the existing {card} (pass overwrite_card to regenerate it).")
    else:
        card.write_text(
            render_card(config, manifest, repo_id=repo_id, title=title, gated=visibility == "gated", extra=card_extra),
            encoding="utf-8",
        )
    if visibility == "public" and "nc" in config.license.lower().split("-"):
        logger.warning(f"publishing non-commercial weights ({config.license}) publicly without gating.")
    if dry_run:
        logger.info(f"dry run: card written to {card}; nothing uploaded.")
        return str(card)

    api = _api(token)
    url = api.create_repo(repo_id, private=visibility == "private", exist_ok=True, repo_type="model")
    # create_repo ignores visibility for an existing repository, so set it explicitly every time, before
    # any weights are uploaded.
    if visibility == "private":
        api.update_repo_settings(repo_id, private=True)
    elif visibility == "gated":
        api.update_repo_settings(repo_id, gated="manual", private=False)
    else:
        api.update_repo_settings(repo_id, gated=False, private=False)
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(path),
        repo_type="model",
        commit_message=commit_message or f"Release {tag or 'bundle'} ({len(manifest.members)} member(s))",
        ignore_patterns=[".*", "**/.*"],
    )
    if tag is not None:
        # Tag the commit this upload created, not whatever the branch head is by the time we get here.
        api.create_tag(
            repo_id,
            tag=tag,
            revision=getattr(commit, "oid", None),
            tag_message=f"yammi {config.yammi_version}, schema {config.schema_version}",
        )
    if not str(url):
        raise BundleError(f"the Hub returned no URL for {repo_id}.")
    logger.info(f"published {path} → {url}{'@' + tag if tag else ''} ({visibility}).")
    return str(url)
