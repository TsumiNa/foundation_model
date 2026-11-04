# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Bundles derived from a released bundle: the licence is inherited and the lineage recorded.

A model fine-tuned from released weights, or saved again after loading, is a derivative. Its bundle
keeps the source bundle's licence: a ShareAlike licence such as CC BY-NC-SA 4.0 cannot be swapped for
another. It also records ``provenance.derived_from`` (source, revision, member, licence, fm version).
Earlier lineage stays under ``provenance.lineage``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.workflows.recording import _git_info

from .bundle import BundleConfig, BundleError, HeadSpec, MemberInfo, load_member, write_bundle

__all__ = ["derived_config", "save_derived"]

_TRAINING_ONLY_PREFIXES = ("task_log_sigmas.",)


def derived_config(
    source: BundleConfig,
    *,
    heads: Sequence[HeadSpec],
    derived_from: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> BundleConfig:
    """``source``'s descriptor, architecture and licence with new ``heads`` and the lineage appended."""
    lineage = list(source.provenance.get("lineage", []))
    if "derived_from" in source.provenance:
        lineage.append(source.provenance["derived_from"])
    return BundleConfig(
        descriptor=source.descriptor,
        model=source.model,
        heads=tuple(heads),
        license=source.license,
        provenance={
            "git": _git_info(),
            **dict(provenance or {}),
            "derived_from": dict(derived_from),
            **({"lineage": lineage} if lineage else {}),
        },
    )


def save_derived(
    out: str | Path,
    source: BundleConfig,
    model: FlexibleMultiTaskModel,
    *,
    heads: Sequence[HeadSpec],
    task_order: Sequence[str],
    derived_from: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    member_id: str = "m000",
) -> Path:
    """Write ``model`` as a one-member derived bundle and check it loads back with identical weights."""
    config = derived_config(source, heads=heads, derived_from=derived_from, provenance=provenance)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith(_TRAINING_ONLY_PREFIXES)}
    info = MemberInfo(id=member_id, task_order=tuple(task_order), source=str(derived_from.get("source")))
    path = write_bundle(out, config, [(info, state)])
    loaded, _, _ = load_member(path, member_id)
    for key, tensor in loaded.state_dict().items():
        if not torch.equal(tensor, state[key].detach().cpu()):
            raise BundleError(f"derived bundle at {path}: tensor {key} did not round-trip.")
    return path
