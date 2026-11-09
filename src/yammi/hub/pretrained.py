# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Use a released model: load a bundle (local or from the Hugging Face Hub) and predict in physical units.

A *source* is a local bundle directory, or a Hub repository written ``owner/name[@revision][#member]``.
Hub access needs the optional extra ``yammi[hub]``. A gated repository also needs a token:
``HF_TOKEN`` or ``hf auth login``.

    >>> from yammi.hub import load_pretrained
    >>> model = load_pretrained("TsumiNa/yammi@v1.0.0")                 # default member
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

from yammi.data.targets import ClipStep
from yammi.models.flexible_multi_task_model import FlexibleMultiTaskModel
from yammi.models.task_head.kernel_regression import expand_for_kernel_regression
from yammi.utils.kmd_plus import formula_to_composition

from .bundle import (
    CONFIG_FILE,
    MANIFEST_FILE,
    MEMBER_FILE,
    MEMBERS_DIR,
    WEIGHTS_FILE,
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


#: How :meth:`PretrainedModel.predict` reports an output past the range a head can express in physical units.
#: ``"nan"`` (default): NaN, with a :class:`SaturationWarning`. Measured on the v1 suite's test split, such
#: outputs were almost always far off when returned as the most extreme training value (R² of
#: ``total_magnetization`` fell from 0.25 to -24 on 3 % of rows). ``"clip"``: that extreme training value.
SATURATED_MODES = ("nan", "clip")


def _to_physical(head: HeadSpec, raw: np.ndarray, saturated: str = "nan") -> np.ndarray:
    """Training-scale head outputs → physical units, warning when values saturate or cannot be inverted."""
    if saturated not in SATURATED_MODES:
        raise ValueError(f"saturated must be one of {SATURATED_MODES}, got {saturated!r}.")
    if head.transform is None:
        return raw
    with np.errstate(invalid="ignore", over="ignore"):
        values = head.transform.inverse(raw)
    lost = int((np.isfinite(raw) & ~np.isfinite(values)).sum())
    if lost:
        warnings.warn(
            f"{head.name}: {lost} prediction(s) lie past the range this head's transform can invert and are "
            "NaN. Re-export the bundle with yammi >= 1.0.1 to saturate them at the training range.",
            SaturationWarning,
            stacklevel=3,
        )
    last = head.transform.steps[-1] if head.transform.steps else None
    if isinstance(last, ClipStep):
        beyond = np.zeros(np.shape(raw), dtype=bool)
        if last.upper is not None:
            beyond |= raw > last.upper
        if last.lower is not None:
            beyond |= raw < last.lower
        n = int(beyond.sum())
        if n and last.guard and saturated == "nan":
            values = np.where(beyond, np.nan, values)
            warnings.warn(
                f"{head.name}: {n} prediction(s) lie past the range this head can express in physical units and "
                "are NaN. The combined prediction (load_suite(...).predict, or yammi predict without #member) keeps the "
                "members that stay in range; saturated='clip' returns the most extreme training value instead.",
                SaturationWarning,
                stacklevel=3,
            )
        elif n:
            warnings.warn(
                f"{head.name}: {n} prediction(s) lie beyond the training range and are returned as the "
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


def _explained(exc: BaseException, repo: str, revision: str | None) -> BaseException | None:
    """A BundleError that says what to do about a Hub access error, or None if ``exc`` is something else."""
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
    except ImportError:  # pragma: no cover
        return None
    page = f"https://huggingface.co/{repo}"
    if isinstance(exc, GatedRepoError):
        return BundleError(
            f"{repo} is gated: request access on {page}, then log in with `hf auth login` (or set HF_TOKEN). "
            "In a container, mount ~/.cache/huggingface or pass HF_TOKEN."
        )
    if isinstance(exc, RevisionNotFoundError):
        tags: list[str] = []
        try:
            from huggingface_hub import HfApi

            tags = [t.name for t in HfApi().list_repo_refs(repo).tags]
        except Exception:  # noqa: BLE001 - the listing is a courtesy
            pass
        return BundleError(
            f"{repo} has no revision {revision!r}" + (f"; its release tags are {tags}." if tags else ".")
        )
    if isinstance(exc, RepositoryNotFoundError):
        return BundleError(
            f"no model repository {repo} on the Hub, or it is private and you are not logged in (`hf auth login`)."
        )
    return None


def _snapshot(
    repo: str,
    revision: str | None,
    patterns: list[str],
    token: str | bool | None,
    cache_dir: Any,
    required: Sequence[str] = (),
) -> Path:
    """Download ``patterns`` of a Hub repository into the local cache; returns the snapshot directory.

    When ``required`` (paths relative to the snapshot) are all in the cache already, the cached snapshot
    is returned without contacting the Hub, so a cached load prints no download progress bars.
    ``revision`` should be a commit (see ``_pin``): a commit's cached files cannot be stale.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "loading from the Hugging Face Hub needs the `hub` extra: "
            "pip install 'yammi[hub] @ git+https://github.com/TsumiNa/yammi' (or `uv sync --extra hub` in a clone)"
        ) from exc
    if required:
        try:
            cached = Path(
                snapshot_download(
                    repo_id=repo,
                    revision=revision,
                    allow_patterns=patterns,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
            )
        except Exception:  # noqa: BLE001 - not cached (or partly): download below
            cached = None
        if cached is not None and all((cached / rel).is_file() for rel in required):
            return cached
    try:
        snap = snapshot_download(
            repo_id=repo, revision=revision, allow_patterns=patterns, token=token, cache_dir=cache_dir
        )
    except Exception as exc:
        explained = _explained(exc, repo, revision)
        if explained is None:
            raise
        raise explained from exc
    # When the Hub cannot be asked, huggingface_hub returns the cached snapshot folder without checking that
    # it holds the files asked for.
    missing = [rel for rel in required if not (Path(snap) / rel).is_file()]
    if missing:
        raise BundleError(
            f"{repo}@{revision}: {', '.join(missing)} not in the local cache, and the Hub did not provide them "
            "(no network, offline mode, or access refused). Download them once where the Hub answers."
        )
    return Path(snap)


_METADATA = [CONFIG_FILE, MANIFEST_FILE, "README.md"]


def _network_errors() -> tuple[type[BaseException], ...]:
    """Transport errors of huggingface_hub's HTTP client (httpx2 since huggingface_hub 2.0)."""
    import httpx2

    return (httpx2.TransportError,)


_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _commit(repo: str, revision: str | None, token: str | bool | None, cache_dir: Any = None) -> str:
    """The commit a (possibly symbolic) Hub revision points to: asked of the Hub, else read from the cache.

    A commit needs no lookup. A branch or tag resolved by the Hub is recorded in the local cache
    (``refs/<revision>``, as ``snapshot_download`` does), because the files are then downloaded by commit.
    When the Hub cannot answer (``HF_HUB_OFFLINE=1``, no network, as on many compute nodes) or refuses
    (access withdrawn, repository made private, server error), the cache answers: a revision downloaded once
    loads anywhere that shares the cache. A refusal is warned about; a revision the Hub does not have raises.
    """
    if revision is not None and _COMMIT_RE.fullmatch(revision):
        return revision
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled, RevisionNotFoundError
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "loading from the Hugging Face Hub needs the `hub` extra: "
            "pip install 'yammi[hub] @ git+https://github.com/TsumiNa/yammi' (or `uv sync --extra hub` in a clone)"
        ) from exc
    unreachable: tuple[type[BaseException], ...] = (OfflineModeIsEnabled, *_network_errors())
    try:
        sha = HfApi().model_info(repo, revision=revision, token=token).sha
    except unreachable as online_error:
        return _cached_commit(repo, revision, cache_dir, online_error)
    except RevisionNotFoundError as exc:
        raise (_explained(exc, repo, revision) or exc) from exc
    except HfHubHTTPError as exc:
        refused = _explained(exc, repo, revision) or exc
        sha = _cached_commit(repo, revision, cache_dir, refused)
        warnings.warn(
            f"the Hub refused {repo}@{revision or 'main'} ({refused}); loading the copy in the local cache "
            f"(commit {sha[:12]}).",
            stacklevel=3,
        )
        return sha
    if not sha:
        raise BundleError(f"the Hub did not resolve {repo}@{revision or 'main'} to a commit.")
    _record_revision(repo, revision, str(sha), cache_dir)
    return str(sha)


def _cache_folder(repo: str, cache_dir: Any) -> Path:
    """The local cache folder of a Hub model repository (huggingface_hub's layout)."""
    from huggingface_hub import constants

    return Path(cache_dir if cache_dir is not None else constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"


def _record_revision(repo: str, revision: str | None, sha: str, cache_dir: Any) -> None:
    """Write ``refs/<revision>`` = ``sha`` into the cache, so the revision resolves later without the Hub."""
    name = revision or "main"
    parts = name.split("/")
    if name == sha or name.startswith("/") or any(part in ("", ".", "..") for part in parts):
        return
    ref = _cache_folder(repo, cache_dir) / "refs" / Path(*parts)
    try:
        if not ref.is_file() or ref.read_text() != sha:
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(sha)
    except OSError:  # a read-only cache: this load works, and the revision just stays unrecorded
        pass


def _cached_commit(repo: str, revision: str | None, cache_dir: Any, online_error: BaseException) -> str:
    """The commit of a revision already in the local cache (its snapshot directory is named after it)."""
    from huggingface_hub import snapshot_download

    try:
        snap = snapshot_download(
            repo_id=repo,
            revision=revision,
            allow_patterns=[CONFIG_FILE],
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception:
        snapshots = _cache_folder(repo, cache_dir) / "snapshots"
        cached = (
            sorted(d.name for d in snapshots.glob("*") if (d / CONFIG_FILE).is_file()) if snapshots.is_dir() else []
        )
        hint = (
            f" The cache holds {repo} at commit {', '.join(cached)} but no record of which commit "
            f"{revision or 'main'} is (downloads before yammi 2.1.2 did not record it); load one by commit, "
            f"for example {repo}@{cached[0]}."
            if cached
            else " Download it once where the Hub answers."
        )
        if isinstance(online_error, BundleError):
            raise BundleError(f"{online_error}{hint}") from online_error
        raise BundleError(
            f"cannot reach the Hub for {repo}@{revision or 'main'} ({online_error}) and it is not in the "
            f"local cache.{hint}"
        ) from online_error
    return Path(snap).name


def _pin(src: Source, token: str | bool | None, cache_dir: Any = None) -> Source:
    """Resolve a Hub source's revision to its commit once, so every later download reads one snapshot.

    A branch or tag can move between the metadata download and the weights download; a commit cannot.
    """
    if src.repo is None:
        return src
    return Source(repo=src.repo, revision=_commit(src.repo, src.revision, token, cache_dir), member=src.member)


def _fetch(src: Source, members: Sequence[str], token: str | bool | None, cache_dir: Any) -> Path:
    """The local snapshot of a pinned Hub source with its metadata and ``members`` (empty = metadata only)."""
    assert src.repo is not None
    patterns = [*_METADATA, *(f"{MEMBERS_DIR}/{m}/*" for m in members)]
    required = [CONFIG_FILE, MANIFEST_FILE]
    required += [f"{MEMBERS_DIR}/{m}/{name}" for m in members for name in (MEMBER_FILE, WEIGHTS_FILE)]
    return _snapshot(src.repo, src.revision, patterns, token, cache_dir, required=required)


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
            derived_from=self._as_source(),
            provenance=provenance,
        )

    def derived_from(self) -> dict[str, Any] | None:
        """Where this bundle came from, if it was derived (fine-tuned or saved) from another; else None.

        The record holds ``source``, ``revision`` (commit), ``member``, ``license`` and the yammi version
        that wrote the source. Earlier generations are in ``config.provenance["lineage"]``.
        """
        record = self.config.provenance.get("derived_from")
        return dict(record) if record is not None else None

    def _as_source(self) -> dict[str, Any]:
        """This model as the ``derived_from`` record of a bundle made from it."""
        return {
            "source": self.source,
            "revision": self.revision,
            "member": self.member.id,
            "license": self.config.license,
            "yammi_version": self.config.yammi_version,
            "fm_version": self.config.yammi_version,  # pre-2.1.1 key, kept for existing readers
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
        self,
        compositions: Sequence[str],
        tasks: Sequence[str] | None = None,
        *,
        probabilities: bool = False,
        saturated: str = "nan",
    ) -> pd.DataFrame:
        """Scalar heads on ``compositions``: one column per task, in physical units.

        ``tasks = None`` predicts every regression and classification head. A classification task gives
        the class name; ``probabilities=True`` adds a ``"<task>:<class>"`` column per class. An output
        past the range a head can express is NaN (``saturated="nan"``) or the most extreme training value
        (``"clip"``); see :data:`SATURATED_MODES`.
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
                    out[head.name] = _to_physical(head, values, saturated)
                else:
                    assert head.classes is not None
                    prob = torch.softmax(raw, dim=-1).cpu().numpy()
                    out[head.name] = head.classes.names(prob.argmax(axis=1))
                    if probabilities:
                        for j, cls in enumerate(head.classes.classes):
                            out[f"{head.name}:{cls}"] = prob[:, j]
        out.attrs["units"] = {head.name: self.units()[head.name] for head in heads}
        return out

    def predict_curve(
        self, compositions: Sequence[str], task: str, t: Any = None, *, saturated: str = "nan"
    ) -> pd.DataFrame:
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
        values = _to_physical(head, raw, saturated)
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

    def predict(
        self, compositions: Sequence[str], tasks: Sequence[str] | None = None, *, probabilities: bool = False
    ) -> pd.DataFrame:
        """Every member's prediction, combined: the recommended way to predict with a suite.

        Regression: the mean over the members whose output stays in range, and ``"<task>:std"``, their
        standard deviation (a spread, usable as an uncertainty). NaN only when every member is out of
        range. Classification: the class with the highest mean probability over members;
        ``probabilities=True`` adds the mean ``"<task>:<class>"`` columns. On the v1 suite's test split the
        combination had a higher R² than the default member on 27 of 29 regression heads.
        """
        models = list(self)
        names = list(tasks) if tasks else [h.name for h in models[0].heads if h.kind != "kernel_regression"]
        kinds = {n: models[0]._head(n, ("regression", "classification")).kind for n in names}  # noqa: SLF001
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SaturationWarning)  # summarised below, once per head
            per = [m.predict(compositions, names, probabilities=True, saturated="nan") for m in models]
        out = pd.DataFrame({"composition": list(compositions)})
        for name in names:
            if kinds[name] == "regression":
                stack = np.stack([p[name].to_numpy(float) for p in per])
                stack[~np.isfinite(stack)] = np.nan  # an overflowed inverse (inf) is out of range too
                kept = np.isfinite(stack).sum(axis=0)
                with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns stay NaN
                    out[name] = np.nanmean(stack, axis=0)
                    out[f"{name}:std"] = np.nanstd(stack, axis=0)
                _warn_dropped(name, kept, len(models))
            else:
                head = models[0].config.head(name)
                assert head.classes is not None
                cols = [f"{name}:{c}" for c in head.classes.classes]
                prob = np.mean([p[cols].to_numpy(float) for p in per], axis=0)
                out[name] = head.classes.names(prob.argmax(axis=1))
                if probabilities:
                    out[cols] = prob
        out.attrs["units"] = {n: models[0].units()[n] for n in names}
        out.attrs["members"] = [m.member.id for m in models]
        return out

    def predict_curve(self, compositions: Sequence[str], task: str, t: Any = None) -> pd.DataFrame:
        """A curve head combined over members: the mean of in-range members and ``"<task>:std"``."""
        models = list(self)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SaturationWarning)
            per = [m.predict_curve(compositions, task, t, saturated="nan") for m in models]
        out = per[0].drop(columns=[task]).copy()
        stack = np.stack([p[task].to_numpy(float) for p in per])
        stack[~np.isfinite(stack)] = np.nan
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            out[task] = np.nanmean(stack, axis=0)
            out[f"{task}:std"] = np.nanstd(stack, axis=0)
        _warn_dropped(task, np.isfinite(stack).sum(axis=0), len(models))
        out.attrs = {**per[0].attrs, "members": [m.member.id for m in models]}
        return out


def _warn_dropped(name: str, kept: np.ndarray, n_members: int) -> None:
    """One warning per head: rows where some members were out of range, and rows left NaN."""
    partial = int(((kept > 0) & (kept < n_members)).sum())
    empty = int((kept == 0).sum())
    if partial or empty:
        warnings.warn(
            f"{name}: on {partial} row(s) some of the {n_members} members were out of range and left out of the "
            f"mean; {empty} row(s) are NaN because every member was.",
            SaturationWarning,
            stacklevel=3,
        )


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
    pinned = _pin(src, token, cache_dir)
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
    pinned = _pin(src, token, cache_dir)
    root = _fetch(pinned, [], token, cache_dir)
    return Suite(
        root,
        str(source),
        fetch=lambda member_id: _fetch(pinned, [member_id], token, cache_dir),
        revision=pinned.revision,
    )
