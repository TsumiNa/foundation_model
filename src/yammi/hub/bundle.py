# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""The release bundle (schema v2; v1 still read): a self-describing model directory that needs no training data.

Layout::

    <bundle>/
      config.json                     shared by every member: descriptor, architecture, heads, licence
      manifest.json                   members, default member, per-member task order / seed / source
      members/<id>/model.safetensors  one member's weights (no pickle)
      members/<id>/member.json        the member's own manifest entry, for a single-member download

Everything a consumer needs to rebuild a member's architecture, map its outputs to physical units and
check that its inputs match training lives in ``config.json``. :func:`build_model` constructs the model
from that file alone, and :func:`load_member` loads a member with ``strict=True``. The design is in
``docs/refactor/pretrained-release/00-overview.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from yammi.data.targets import ClassLabels, TargetTransform
from yammi.models.flexible_multi_task_model import FlexibleMultiTaskModel
from yammi.models.model_config import (
    ClassificationTaskConfig,
    KernelRegressionTaskConfig,
    OptimizerConfig,
    RegressionTaskConfig,
)
from yammi.utils.kmd_plus import DEFAULT_ELEMENTS, KMD, element_features
from yammi.workflows._engine import AE_NAME, build_encoder_config
from yammi.workflows._sections import ModelSectionConfig, TrainingSectionConfig, build_model_section
from yammi.workflows.task_catalog import TaskConfig

__all__ = [
    "BundleError",
    "BundleConfig",
    "DescriptorSpec",
    "HeadSpec",
    "Manifest",
    "MemberInfo",
    "SCHEMA_VERSION",
    "build_model",
    "load_member",
    "read_config",
    "read_manifest",
    "write_bundle",
]

#: Schema history. v1 (1.0.0): the format below. v2 (1.0.1): a ``clip`` step may carry ``guard: true``,
#: marking the saturation clip that export adds where a transform's inverse is undefined past the training
#: range. Readers accept every older schema: a v1 bundle has no guards, so all its clips are intrinsic.
SCHEMA_VERSION = 2
CONFIG_FILE = "config.json"
MANIFEST_FILE = "manifest.json"
MEMBERS_DIR = "members"
WEIGHTS_FILE = "model.safetensors"
MEMBER_FILE = "member.json"
_KINDS = ("regression", "classification", "kernel_regression")
_OUTPUT_SCALES = ("physical", "shape")
_ELEMENT_FEATURES = Path(__file__).resolve().parents[1] / "utils" / "element_features.csv"
# A member id names a directory under members/: one portable path component, no separators or dots-only.
_MEMBER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_HEAD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class BundleError(ValueError):
    """A bundle is malformed, from a newer schema, or does not match this installation."""


def yammi_version() -> str:
    try:
        return version("yammi")
    except PackageNotFoundError:  # pragma: no cover - running from an uninstalled tree
        return "unknown"


def _writer_version(data: Mapping[str, Any]) -> str:
    """The yammi version that wrote a config.json; bundles before yammi 2.1.1 call it ``fm_version``."""
    return str(data.get("yammi_version", data.get("fm_version", "unknown")))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


# --- descriptor --------------------------------------------------------------------------------


@dataclass(frozen=True)
class DescriptorSpec:
    """The KMD descriptor the model was trained on; :meth:`check` refuses a mismatching installation."""

    n_grids: int
    elements: tuple[str, ...]
    features_sha256: str
    kind: str = "kmd"
    method: str = "1d"
    sigma: str = "auto"
    scale: bool = True

    @classmethod
    def current(cls, n_grids: int) -> DescriptorSpec:
        """The descriptor this installation computes for ``n_grids``."""
        digest = hashlib.sha256(_ELEMENT_FEATURES.read_bytes()).hexdigest()
        return cls(n_grids=n_grids, elements=tuple(DEFAULT_ELEMENTS), features_sha256=digest)

    def check(self) -> None:
        if self.kind != "kmd":
            raise BundleError(f"descriptor kind {self.kind!r} is not supported; bundles carry KMD descriptors only.")
        here = DescriptorSpec.current(self.n_grids)
        if (self.method, self.sigma, self.scale) != (here.method, here.sigma, here.scale):
            raise BundleError(f"KMD settings differ from this installation: {self.method}/{self.sigma}/{self.scale}.")
        if self.elements != here.elements:
            raise BundleError("the bundle's element list differs from this installation's element_features.csv.")
        if self.features_sha256 != here.features_sha256:
            raise BundleError(
                "element_features.csv differs from the one the model was trained with "
                f"(bundle {self.features_sha256[:12]}…, installed {here.features_sha256[:12]}…)."
            )

    def kmd(self) -> KMD:
        # check() pins method / sigma / scale to the installation's values, so these are the literals.
        if self.method != "1d" or self.sigma != "auto":
            raise BundleError(f"unsupported KMD settings method={self.method!r}, sigma={self.sigma!r}.")
        return KMD(element_features.values, method="1d", n_grids=self.n_grids, sigma="auto", scale=self.scale)

    def dim(self) -> int:
        return int(self.kmd().transform(np.eye(1, len(self.elements))).shape[1])

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["elements"] = list(self.elements)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DescriptorSpec:
        data = dict(data)
        data["elements"] = tuple(data["elements"])
        return cls(**data)


# --- heads ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadSpec:
    """One task head, with its widths resolved so the model's ``[model]`` defaults no longer matter.

    ``hidden_dims`` are the interior widths of a regression / classification head. ``x_hidden_dims``,
    ``t_hidden_dims`` and ``n_kernel`` describe a kernel-regression head, which also records its
    coordinate: name, unit, the observed training range ``t_range``, and the default prediction grid
    ``t_grid``. ``transform`` maps physical values to the training scale; it is ``None`` for
    classification and for shape-only heads (``output_scale = "shape"``).
    """

    name: str
    kind: str
    unit: str | None = None
    output_scale: str = "physical"
    source_column: str | None = None
    hidden_dims: tuple[int, ...] = ()
    num_classes: int | None = None
    classes: ClassLabels | None = None
    x_hidden_dims: tuple[int, ...] = ()
    t_hidden_dims: tuple[int, ...] = ()
    n_kernel: int | None = None
    t_name: str | None = None
    t_unit: str | None = None
    t_range: tuple[float, float] | None = None
    t_grid: tuple[float, ...] = ()
    transform: TargetTransform | None = None

    def __post_init__(self) -> None:
        # Head names become file names (e.g. yammi predict's "<task>_curve.csv"): one portable component only.
        if not isinstance(self.name, str) or not _HEAD_NAME.match(self.name):
            raise BundleError(
                f"head name {self.name!r} must be letters, digits, '.', '_' or '-' (starting with a letter or "
                "digit, at most 128 characters); no path separators."
            )
        if self.kind not in _KINDS:
            raise BundleError(f"head {self.name!r}: kind must be one of {_KINDS}, got {self.kind!r}.")
        if self.output_scale not in _OUTPUT_SCALES:
            raise BundleError(f"head {self.name!r}: output_scale must be one of {_OUTPUT_SCALES}.")
        if self.kind == "classification":
            if self.num_classes is None or self.classes is None or len(self.classes.classes) != self.num_classes:
                raise BundleError(f"head {self.name!r}: a classification head needs num_classes class names.")
            if self.transform is not None or self.output_scale != "physical":
                raise BundleError(f"head {self.name!r}: a classification head takes no transform or output_scale.")
        elif (self.transform is None) == (self.output_scale == "physical"):
            raise BundleError(
                f"head {self.name!r}: a {self.kind} head needs a transform (physical output) "
                "or output_scale = 'shape' (no transform), not both or neither."
            )
        if self.kind == "kernel_regression":
            if not self.x_hidden_dims or not self.t_hidden_dims or not self.n_kernel:
                raise BundleError(f"head {self.name!r}: kernel regression needs x/t hidden widths and n_kernel.")
            if self.t_range is None or not self.t_grid:
                raise BundleError(f"head {self.name!r}: kernel regression needs t_range and t_grid.")
        elif not self.hidden_dims:
            raise BundleError(f"head {self.name!r}: hidden_dims is required.")

    def task_config(self, latent_dim: int) -> TaskConfig:
        """The model-side head config; mirrors ``TaskCatalog.build_task_config`` so weights load strictly."""
        optimizer = OptimizerConfig(lr=1e-3, weight_decay=0.0, scheduler_enabled=False)  # inference placeholder
        column = self.source_column or self.name
        if self.kind == "regression":
            return RegressionTaskConfig(
                name=self.name, data_column=column, dims=[latent_dim, *self.hidden_dims, 1], optimizer=optimizer
            )
        if self.kind == "classification":
            assert self.num_classes is not None
            return ClassificationTaskConfig(
                name=self.name,
                data_column=column,
                dims=[latent_dim, *self.hidden_dims],
                num_classes=self.num_classes,
                optimizer=optimizer,
            )
        assert self.n_kernel is not None
        return KernelRegressionTaskConfig(
            name=self.name,
            data_column=column,
            t_column=self.t_name or "t",
            x_dim=[latent_dim, *self.x_hidden_dims],
            t_dim=list(self.t_hidden_dims),
            kernel_num_centers=self.n_kernel,
            kernel_centers_init=None,
            kernel_sigmas_init=None,
            kernel_learnable_centers=True,
            kernel_learnable_sigmas=True,
            enable_mu3=False,
            optimizer=optimizer,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "unit": self.unit,
            "output_scale": self.output_scale,
        }
        out["source_column"] = self.source_column
        if self.kind == "kernel_regression":
            out.update(
                x_hidden_dims=list(self.x_hidden_dims),
                t_hidden_dims=list(self.t_hidden_dims),
                n_kernel=self.n_kernel,
                t_name=self.t_name,
                t_unit=self.t_unit,
                t_range=list(self.t_range) if self.t_range else None,
                t_grid=list(self.t_grid),
            )
        else:
            out["hidden_dims"] = list(self.hidden_dims)
        if self.kind == "classification":
            assert self.classes is not None
            out.update(num_classes=self.num_classes, classes=list(self.classes.classes))
        out["transform"] = self.transform.to_dict() if self.transform is not None else None
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HeadSpec:
        d = dict(data)
        for key in ("hidden_dims", "x_hidden_dims", "t_hidden_dims", "t_grid"):
            if d.get(key) is not None:
                d[key] = tuple(d[key])
        if d.get("t_range") is not None:
            d["t_range"] = tuple(d["t_range"])
        if d.get("classes") is not None:
            d["classes"] = ClassLabels(tuple(d["classes"]))
        if d.get("transform") is not None:
            d["transform"] = TargetTransform.from_dict(d["transform"])
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise BundleError(f"head {d.get('name')!r}: unknown keys {sorted(unknown)}.")
        return cls(**d)


# --- config / manifest ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleConfig:
    """``config.json``: everything shared by the members of one bundle."""

    descriptor: DescriptorSpec
    model: ModelSectionConfig
    heads: tuple[HeadSpec, ...]
    license: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    yammi_version: str = field(default_factory=yammi_version)

    def __post_init__(self) -> None:
        object.__setattr__(self, "heads", tuple(self.heads))
        if not self.license or not str(self.license).strip():
            raise BundleError("a bundle needs a licence identifier (e.g. 'cc-by-nc-sa-4.0').")
        names = [h.name for h in self.heads]
        if len(set(names)) != len(names) or AE_NAME in names:
            raise BundleError(f"head names must be unique and must not be {AE_NAME!r}: {names}.")

    @property
    def fm_version(self) -> str:
        """Deprecated name of ``yammi_version`` (kept for code written against yammi < 2.1.1)."""
        return self.yammi_version

    def head(self, name: str) -> HeadSpec:
        for h in self.heads:
            if h.name == name:
                return h
        raise KeyError(f"no head {name!r}; heads are {[h.name for h in self.heads]}.")

    def to_dict(self) -> dict[str, Any]:
        model = _jsonable(asdict(self.model))
        if model.get("encoder_type") == "mlp":
            model.pop("transformer", None)  # build_model_section rejects [model.transformer] for MLPs
        return {
            "schema_version": self.schema_version,
            "yammi_version": self.yammi_version,
            "license": self.license,
            "descriptor": self.descriptor.to_dict(),
            "model": model,
            "heads": [h.to_dict() for h in self.heads],
            "provenance": _jsonable(dict(self.provenance)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BundleConfig:
        schema = data.get("schema_version")
        if not isinstance(schema, int):
            raise BundleError("config.json has no integer schema_version.")
        if schema > SCHEMA_VERSION:
            raise BundleError(
                f"bundle schema {schema} is newer than this installation supports ({SCHEMA_VERSION}); "
                f"it was written by yammi {_writer_version(data)}. Upgrade yammi."
            )
        return cls(
            descriptor=DescriptorSpec.from_dict(data["descriptor"]),
            model=build_model_section(data["model"]),
            heads=tuple(HeadSpec.from_dict(h) for h in data["heads"]),
            license=data.get("license", ""),
            provenance=dict(data.get("provenance") or {}),
            schema_version=schema,
            yammi_version=_writer_version(data),
        )


@dataclass(frozen=True)
class MemberInfo:
    """One member: how it was pretrained (task order, seed) and where it came from."""

    id: str
    task_order: tuple[str, ...]
    seed: int | None = None
    source: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _MEMBER_ID.match(self.id):
            raise BundleError(
                f"member id {self.id!r} must be one path component of letters, digits, '.', '_' or '-' "
                "(starting with a letter or digit, at most 64 characters)."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_order": list(self.task_order),
            "seed": self.seed,
            "source": self.source,
            "metrics": _jsonable(dict(self.metrics)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemberInfo:
        d = dict(data)
        d["task_order"] = tuple(d.get("task_order", ()))
        return cls(**d)


@dataclass(frozen=True)
class Manifest:
    members: tuple[MemberInfo, ...]
    default: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))
        ids = [m.id for m in self.members]
        if not ids:
            raise BundleError("a bundle needs at least one member.")
        if len(set(ids)) != len(ids):
            raise BundleError(f"member ids must be unique: {ids}.")
        if self.default not in ids:
            raise BundleError(f"default member {self.default!r} is not among {ids}.")

    def member(self, member_id: str | None = None) -> MemberInfo:
        wanted = self.default if member_id is None else member_id
        for m in self.members:
            if m.id == wanted:
                return m
        raise KeyError(f"no member {wanted!r}; members are {[m.id for m in self.members]}.")

    def to_dict(self) -> dict[str, Any]:
        return {"default": self.default, "members": [m.to_dict() for m in self.members]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        return cls(members=tuple(MemberInfo.from_dict(m) for m in data["members"]), default=data["default"])


# --- model construction and I/O ------------------------------------------------------------------


def build_model(config: BundleConfig, training: TrainingSectionConfig | None = None) -> FlexibleMultiTaskModel:
    """The bundle's architecture with freshly initialised weights, in eval mode.

    Without ``training`` every parameter group gets an inference placeholder optimizer. With it, the
    groups get ``[training]``'s learning rates and weight decays, so the model can be fine-tuned. The
    encoder, the AE and each head use the same rules as ``yammi pretrain`` / ``yammi finetune``. The
    learnable loss balancer is never enabled: its parameters are not part of a bundle.
    """
    config.descriptor.check()
    section = config.model

    def opt(lr: float, weight_decay: float) -> OptimizerConfig:
        if training is None:
            return OptimizerConfig(lr=1e-3, weight_decay=1e-2, scheduler_enabled=False)
        return training.optimizer_config(lr=lr, weight_decay=weight_decay)

    model = FlexibleMultiTaskModel(
        task_configs=[],
        encoder_config=build_encoder_config(section, config.descriptor.dim()),
        enable_autoencoder=True,
        autoencoder_hidden_dims=section.autoencoder_hidden_dims,
        shared_block_optimizer=opt(training.encoder_lr, training.encoder_weight_decay) if training else opt(0, 0),
    )
    if AE_NAME in model.task_configs_map:
        model.task_configs_map[AE_NAME].optimizer = (
            opt(training.ae_lr, training.ae_weight_decay) if training else opt(0, 0)
        )
    for head in config.heads:
        task_config = head.task_config(section.latent_dim)
        if training is not None:
            if head.kind == "kernel_regression":
                task_config.optimizer = opt(training.kr_lr, training.kr_weight_decay)
            else:
                task_config.optimizer = opt(training.head_lr, training.head_weight_decay)
        model.add_task(task_config)
    return model.eval()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_bundle(
    out: str | Path,
    config: BundleConfig,
    members: Sequence[tuple[MemberInfo, Mapping[str, torch.Tensor]]],
    *,
    default: str | None = None,
) -> Path:
    """Write a bundle to an empty or missing directory ``out``; the default member is the first.

    The bundle is assembled in a sibling staging directory and moved into place only when complete, so
    an interrupted write (e.g. a full disk) leaves ``out`` as it was.
    """
    out = Path(out)
    if not members:
        raise BundleError("a bundle needs at least one member.")
    if out.exists() and any(out.iterdir()):
        raise BundleError(f"{out} is not empty; a bundle is written to a fresh directory.")
    manifest = Manifest(members=tuple(info for info, _ in members), default=default or members[0][0].id)
    reference = build_model(config).state_dict()
    for info, state in members:
        _check_state(state, reference, info.id)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.", suffix=".partial", dir=out.parent))
    try:
        _write_json(stage / CONFIG_FILE, config.to_dict())
        _write_json(stage / MANIFEST_FILE, manifest.to_dict())
        for info, state in members:
            member_dir = stage / MEMBERS_DIR / info.id
            member_dir.mkdir(parents=True)
            # Clone so tensors that share storage (or are views) serialise as independent, contiguous blobs.
            tensors = {k: v.detach().cpu().contiguous().clone() for k, v in state.items()}
            save_file(tensors, str(member_dir / WEIGHTS_FILE))
            _write_json(member_dir / MEMBER_FILE, info.to_dict())
        if out.exists():
            out.rmdir()  # empty, checked above
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return out


def _check_state(state: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor], member: str) -> None:
    missing = sorted(set(reference) - set(state))
    unexpected = sorted(set(state) - set(reference))
    if missing or unexpected:
        raise BundleError(
            f"member {member!r} does not match the bundle architecture: "
            f"missing {missing[:8]}{'…' if len(missing) > 8 else ''}, "
            f"unexpected {unexpected[:8]}{'…' if len(unexpected) > 8 else ''}."
        )
    wrong = [k for k in reference if tuple(state[k].shape) != tuple(reference[k].shape)]
    if wrong:
        k = wrong[0]
        raise BundleError(
            f"member {member!r}: {len(wrong)} tensor(s) have the wrong shape, e.g. {k} "
            f"{tuple(state[k].shape)} vs {tuple(reference[k].shape)}."
        )


def read_config(path: str | Path) -> BundleConfig:
    file = Path(path) / CONFIG_FILE
    if not file.exists():
        raise BundleError(f"{Path(path)} is not a bundle: {CONFIG_FILE} is missing.")
    return BundleConfig.from_dict(json.loads(file.read_text(encoding="utf-8")))


def read_manifest(path: str | Path) -> Manifest:
    file = Path(path) / MANIFEST_FILE
    if not file.exists():
        raise BundleError(f"{Path(path)} is not a bundle: {MANIFEST_FILE} is missing.")
    return Manifest.from_dict(json.loads(file.read_text(encoding="utf-8")))


def load_member(
    path: str | Path, member: str | None = None, *, training: TrainingSectionConfig | None = None
) -> tuple[FlexibleMultiTaskModel, BundleConfig, MemberInfo]:
    """Build the architecture from ``config.json`` and load one member's weights strictly.

    ``training`` sets the optimizers for fine-tuning (see :func:`build_model`).
    """
    path = Path(path)
    config = read_config(path)
    info = read_manifest(path).member(member)
    weights = path / MEMBERS_DIR / info.id / WEIGHTS_FILE
    if not weights.exists():
        raise BundleError(f"member {info.id!r} has no weights at {weights}.")
    model = build_model(config, training)
    state = load_file(str(weights))
    _check_state(state, model.state_dict(), info.id)
    model.load_state_dict(state, strict=True)
    return model.eval(), config, info
