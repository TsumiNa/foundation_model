# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Use a released model: load a bundle (local or from the Hugging Face Hub) and predict in physical units.

A *source* is a local bundle directory, or a Hub repository written ``owner/name[@revision][#member]``.
Hub access needs the optional extra ``foundation-model[hub]``. A gated repository also needs a token:
``HF_TOKEN`` or ``hf auth login``.

    >>> from foundation_model.hub import load_pretrained
    >>> model = load_pretrained("TsumiNa/fm@v1.0.0")                 # default member
    >>> model.predict(["Fe2O3", "NaCl"], tasks=["band_gap"])             # eV, not the training scale
    >>> model.predict_curve(["Bi2Te3"], "seebeck")                        # on the training grid

Outputs are in the head's physical unit (``HeadSpec.unit``). The exceptions are shape-only heads
(``output_scale = "shape"``), which return per-row max-abs normalised curves, and classification
heads, which return class names (and optionally probabilities).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from foundation_model.data.targets import ClipStep
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.task_head.kernel_regression import expand_for_kernel_regression
from foundation_model.utils.kmd_plus import formula_to_composition

from .bundle import (
    CONFIG_FILE,
    MANIFEST_FILE,
    MEMBERS_DIR,
    BundleConfig,
    BundleError,
    HeadSpec,
    Manifest,
    MemberInfo,
    load_member,
    read_manifest,
)

__all__ = [
    "ExtrapolationWarning",
    "PretrainedModel",
    "SaturationWarning",
    "Suite",
    "load_pretrained",
    "load_suite",
    "parse_source",
    "resolve",
]

_REPO_RE = re.compile(r"^(?P<repo>[\w.-]+/[\w.-]+)(?:@(?P<rev>[^#]+))?(?:#(?P<member>.+))?$")


class ExtrapolationWarning(UserWarning):
    """A kernel-regression ``t`` lies outside the range the head was trained on."""


class SaturationWarning(UserWarning):
    """A prediction lies past the training range a head's transform can invert.

    Bundles exported with 1.0.1 or later clip there, so the value comes back as the most extreme training
    value. A bundle exported earlier has no such clip, and the value comes back NaN.
    """


def _to_physical(head: HeadSpec, raw: np.ndarray) -> np.ndarray:
    """Training-scale head outputs → physical units, warning when values saturate or cannot be inverted."""
    if head.transform is None:
        return raw
    with np.errstate(invalid="ignore", over="ignore"):
        values = head.transform.inverse(raw)
    lost = int((np.isfinite(raw) & ~np.isfinite(values)).sum())
    if lost:
        warnings.warn(
            f"{head.name}: {lost} prediction(s) lie past the range this head's transform can invert and are "
            "NaN. Re-export the bundle with foundation-model >= 1.0.1 to saturate them at the training range.",
            SaturationWarning,
            stacklevel=3,
        )
    last = head.transform.steps[-1] if head.transform.steps else None
    if isinstance(last, ClipStep):
        beyond = int((raw > last.upper).sum()) if last.upper is not None else 0
        beyond += int((raw < last.lower).sum()) if last.lower is not None else 0
        if beyond:
            warnings.warn(
                f"{head.name}: {beyond} prediction(s) lie beyond the training range and are returned as the "
                "most extreme training value.",
                SaturationWarning,
                stacklevel=3,
            )
    return values


@dataclass(frozen=True)
class Source:
    """A parsed source: a local bundle path, or a Hub repository with an optional revision and member."""

    path: Path | None = None
    repo: str | None = None
    revision: str | None = None
    member: str | None = None


def parse_source(source: str | Path, *, revision: str | None = None, member: str | None = None) -> Source:
    """Resolve ``source``. An existing directory, optionally ``<dir>#member``, is local; otherwise
    ``owner/name[@rev][#member]``.

    Keyword ``revision`` / ``member`` override the ones written into the string.
    """
    path = Path(source)
    if not path.exists() and "#" in str(source):
        head, _, tail = str(source).rpartition("#")
        if tail and Path(head).is_dir():
            path, member = Path(head), member or tail
    if path.exists():
        if not path.is_dir():
            raise BundleError(f"{path} is a file; a bundle is a directory with {CONFIG_FILE}.")
        return Source(path=path, member=member)
    match = _REPO_RE.match(str(source))
    if match is None:
        raise BundleError(
            f"{source!r} is neither an existing bundle directory nor a Hub repo 'owner/name[@rev][#member]'."
        )
    return Source(repo=match["repo"], revision=revision or match["rev"], member=member or match["member"])


def _snapshot(repo: str, revision: str | None, patterns: list[str], token: str | bool | None, cache_dir: Any) -> Path:
    """Download ``patterns`` of a Hub repository into the local cache; returns the snapshot directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "loading from the Hugging Face Hub needs the optional extra: pip install 'foundation-model[hub]'"
        ) from exc
    return Path(
        snapshot_download(repo_id=repo, revision=revision, allow_patterns=patterns, token=token, cache_dir=cache_dir)
    )


_METADATA = [CONFIG_FILE, MANIFEST_FILE, "README.md"]


_METADATA = [CONFIG_FILE, MANIFEST_FILE, "README.md"]


def _commit(repo: str, revision: str | None, token: str | bool | None) -> str:
    """The commit a (possibly symbolic) Hub revision points to right now."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "loading from the Hugging Face Hub needs the optional extra: pip install 'foundation-model[hub]'"
        ) from exc
    sha = HfApi().model_info(repo, revision=revision, token=token).sha
    if not sha:
        raise BundleError(f"the Hub did not resolve {repo}@{revision or 'main'} to a commit.")
    return str(sha)


def _pin(src: Source, token: str | bool | None) -> Source:
    """Resolve a Hub source's revision to its commit once, so every later download reads one snapshot.

    A branch or tag can move between the metadata download and the weights download; a commit cannot.
    """
    if src.repo is None:
        return src
    return Source(repo=src.repo, revision=_commit(src.repo, src.revision, token), member=src.member)


def _fetch(src: Source, members: Sequence[str], token: str | bool | None, cache_dir: Any) -> Path:
    """The local snapshot of a pinned Hub source with its metadata and ``members`` (empty = metadata only)."""
    assert src.repo is not None
    patterns = [*_METADATA, *(f"{MEMBERS_DIR}/{m}/*" for m in members)]
    return _snapshot(src.repo, src.revision, patterns, token, cache_dir)


class PretrainedModel:
    """One loaded member of a bundle, with a physical-unit prediction API."""

    def __init__(
        self,
        model: FlexibleMultiTaskModel,
        config: BundleConfig,
        member: MemberInfo,
        source: str,
        *,
        revision: str | None = None,
    ) -> None:
        self.model = model.eval()
        self.config = config
        self.member = member
        self.source = source
        self.revision = revision
        self._kmd = config.descriptor.kmd()

    def save(self, out: str | Path, *, provenance: dict[str, Any] | None = None) -> Path:
        """Write this member as a one-member bundle at ``out``.

        The bundle keeps the source licence and records ``derived_from`` (source, revision, member,
        licence). Saving a fine-tuned model therefore carries the ShareAlike terms forward.
        """
        from .derive import save_derived

        return save_derived(
            out,
            self.config,
            self.model,
            heads=self.config.heads,
            task_order=self.member.task_order,
            derived_from=self.derived_from(),
            provenance=provenance,
        )

    def derived_from(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "revision": self.revision,
            "member": self.member.id,
            "license": self.config.license,
            "fm_version": self.config.fm_version,
        }

    def __repr__(self) -> str:
        return f"PretrainedModel({self.source!r}, member={self.member.id!r}, heads={len(self.config.heads)})"

    @property
    def heads(self) -> tuple[HeadSpec, ...]:
        return self.config.heads

    @property
    def tasks(self) -> list[str]:
        return [h.name for h in self.config.heads]

    @property
    def license(self) -> str:
        return self.config.license

    def units(self) -> dict[str, str | None]:
        """Unit of each head's output (``"shape"`` for shape-only heads, ``None`` for classification)."""
        return {h.name: ("shape" if h.output_scale == "shape" else h.unit) for h in self.config.heads}

    # -- inputs ----------------------------------------------------------------------------------

    def descriptors(self, compositions: Sequence[str]) -> np.ndarray:
        """KMD descriptors, identical to training. Raises naming every composition it cannot parse."""
        rows, bad = [], []
        for comp in compositions:
            try:
                rows.append(formula_to_composition(comp))
            except (ValueError, TypeError, KeyError) as exc:
                bad.append(f"{comp!r} ({exc})")
        if bad:
            raise ValueError(f"cannot build descriptors for {len(bad)} composition(s): {'; '.join(bad[:5])}")
        if not rows:
            return np.zeros((0, self.config.descriptor.dim()))
        return np.asarray(self._kmd.transform(np.stack(rows)))

    def latent(self, compositions: Sequence[str]) -> np.ndarray:
        """The shared latent representation (``tanh`` of the encoder output)."""
        return self._latent(compositions).cpu().numpy()

    def _latent(self, compositions: Sequence[str]) -> torch.Tensor:
        x = torch.tensor(self.descriptors(list(compositions)), dtype=torch.float32)
        with torch.no_grad():
            return torch.tanh(self.model.encoder(x))

    def _head(self, task: str, kinds: tuple[str, ...]) -> HeadSpec:
        head = self.config.head(task)
        if head.kind not in kinds:
            raise ValueError(
                f"task {task!r} is a {head.kind} head; use {'predict_curve' if head.kind == 'kernel_regression' else 'predict'}."
            )
        return head

    # -- outputs ---------------------------------------------------------------------------------

    def predict(
        self, compositions: Sequence[str], tasks: Sequence[str] | None = None, *, probabilities: bool = False
    ) -> pd.DataFrame:
        """Scalar heads on ``compositions``: one column per task, in physical units.

        ``tasks = None`` predicts every regression and classification head. A classification task gives
        the class name; ``probabilities=True`` adds a ``"<task>:<class>"`` column per class.
        """
        names = tasks or [h.name for h in self.config.heads if h.kind != "kernel_regression"]
        heads = [self._head(n, ("regression", "classification")) for n in names]
        comps = list(compositions)
        h = self._latent(comps)
        out = pd.DataFrame({"composition": comps})
        with torch.no_grad():
            for head in heads:
                raw = self.model.task_heads[head.name](h)
                if head.kind == "regression":
                    values = raw.squeeze(-1).cpu().numpy().astype(float)
                    out[head.name] = _to_physical(head, values)
                else:
                    assert head.classes is not None
                    prob = torch.softmax(raw, dim=-1).cpu().numpy()
                    out[head.name] = head.classes.names(prob.argmax(axis=1))
                    if probabilities:
                        for j, cls in enumerate(head.classes.classes):
                            out[f"{head.name}:{cls}"] = prob[:, j]
        out.attrs["units"] = {head.name: self.units()[head.name] for head in heads}
        return out

    def predict_curve(self, compositions: Sequence[str], task: str, t: Any = None) -> pd.DataFrame:
        """A kernel-regression head as a long table ``composition, <t_name>, <task>``.

        ``t = None`` uses the head's default grid. ``t`` may also be one 1-D grid shared by every
        composition, or a list with one grid per composition (any lengths). Values outside the observed
        training range raise :class:`ExtrapolationWarning`.
        """
        head = self._head(task, ("kernel_regression",))
        comps = list(compositions)
        grids = self._grids(head, comps, t)
        h = self._latent(comps)
        with torch.no_grad():
            eh, et = expand_for_kernel_regression(h, [torch.tensor(g, dtype=torch.float32) for g in grids])
            raw = self.model.task_heads[head.name](eh, t=et).squeeze(-1).cpu().numpy().astype(float)
        values = _to_physical(head, raw)
        t_col = head.t_name or "t"
        out = pd.DataFrame(
            {
                "composition": np.repeat(comps, [len(g) for g in grids]),
                t_col: np.concatenate(grids) if grids else np.array([]),
                head.name: values,
            }
        )
        out.attrs["units"] = {head.name: self.units()[head.name], t_col: head.t_unit}
        return out

    @staticmethod
    def _grids(head: HeadSpec, comps: list[str], t: Any) -> list[np.ndarray]:
        if t is None:
            grids = [np.asarray(head.t_grid, dtype=float)] * len(comps)
        elif isinstance(t, (list, tuple)) and t and np.ndim(t[0]) == 1:
            if len(t) != len(comps):
                raise ValueError(f"{len(t)} t grids for {len(comps)} compositions.")
            grids = [np.asarray(g, dtype=float) for g in t]
        else:
            shared = np.asarray(t, dtype=float)
            if shared.ndim != 1 or shared.size == 0:
                raise ValueError("t must be a non-empty 1-D grid, or one 1-D grid per composition.")
            grids = [shared] * len(comps)
        if head.t_range is not None and grids:
            lo, hi = head.t_range
            span = np.concatenate(grids)
            outside = int(((span < lo) | (span > hi)).sum())
            if outside:
                warnings.warn(
                    f"{head.name}: {outside} t value(s) outside the training range [{lo:g}, {hi:g}]"
                    f"{' ' + head.t_unit if head.t_unit else ''}; predictions there are extrapolated.",
                    ExtrapolationWarning,
                    stacklevel=3,
                )
        return grids


class Suite:
    """All members of a bundle. Members are downloaded and loaded on first access, one at a time."""

    def __init__(
        self,
        root: Path,
        source: str,
        *,
        fetch: Callable[[str], Path] | None = None,
        revision: str | None = None,
    ) -> None:
        self.root = root
        self.source = source
        self.revision = revision
        self.manifest: Manifest = read_manifest(root)
        self._fetch = fetch
        self._loaded: dict[str, PretrainedModel] = {}

    @property
    def member_ids(self) -> list[str]:
        return [m.id for m in self.manifest.members]

    def __len__(self) -> int:
        return len(self.manifest.members)

    def __getitem__(self, member_id: str) -> PretrainedModel:
        if member_id not in self._loaded:
            self.manifest.member(member_id)  # KeyError naming the members, before any download
            root = self._fetch(member_id) if self._fetch is not None else self.root
            model, config, info = load_member(root, member_id)
            self._loaded[member_id] = PretrainedModel(model, config, info, self.source, revision=self.revision)
        return self._loaded[member_id]

    def __iter__(self) -> Iterator[PretrainedModel]:
        for member_id in self.member_ids:
            yield self[member_id]


def load_pretrained(
    source: str | Path,
    *,
    member: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
) -> PretrainedModel:
    """Load one member (the bundle's default unless ``member`` is given) from a bundle or Hub repository.

    On the Hub the revision is pinned to its commit first; then the metadata and only that member are
    downloaded.
    """
    root, src = resolve(source, member=member, revision=revision, token=token, cache_dir=cache_dir)
    model, config, info = load_member(root, src.member)
    return PretrainedModel(model, config, info, str(source), revision=src.revision)


def resolve(
    source: str | Path,
    *,
    member: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Path, Source]:
    """A local directory holding ``source``'s metadata and one member, plus the parsed source.

    The returned source names the member (the default member when none was asked for) and, for the
    Hub, the commit its revision was pinned to.
    """
    src = parse_source(source, revision=revision, member=member)
    if src.path is not None:
        wanted = src.member or read_manifest(src.path).default
        return src.path, Source(path=src.path, member=wanted)
    pinned = _pin(src, token)
    chosen: str = pinned.member or read_manifest(_fetch(pinned, [], token, cache_dir)).default
    root = _fetch(pinned, [chosen], token, cache_dir)
    return root, Source(repo=pinned.repo, revision=pinned.revision, member=chosen)


def load_suite(
    source: str | Path,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
) -> Suite:
    """Every member of a bundle or Hub repository. Only the metadata is read up front; each member is
    downloaded (Hub) and loaded the first time it is accessed."""
    src = parse_source(source, revision=revision)
    if src.member is not None:
        raise BundleError(f"{source!r} names member {src.member!r}; load_suite loads all members.")
    if src.path is not None:
        return Suite(src.path, str(source))
    pinned = _pin(src, token)
    root = _fetch(pinned, [], token, cache_dir)
    return Suite(
        root,
        str(source),
        fetch=lambda member_id: _fetch(pinned, [member_id], token, cache_dir),
        revision=pinned.revision,
    )
