# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``yammi export`` — turn one or more training runs into a release bundle, and prove it.

Each run becomes one member. The shared ``config.json`` is built from the task catalog:

* head widths, resolved against the ``[model]`` defaults;
* target transforms and class names from each task's ``scaler``;
* units and the shape-only marker from the task's release metadata;
* for kernel-regression heads, the observed training range of ``t`` and a deterministic default grid.

After writing, every member is loaded back through :func:`~yammi.hub.bundle.load_member`
(strict) and compared with the same checkpoint loaded through the catalog path that ``yammi predict`` uses.
They must produce bit-identical outputs on a fixed set of training compositions, or the export fails.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from yammi.data.targets import guard_inverse_domain
from yammi.models.task_head.kernel_regression import expand_for_kernel_regression
from yammi.workflows._engine import (
    AE_NAME,
    as_float_array,
    build_model_for_checkpoint,
    checkpoint_task_order,
)
from yammi.workflows._sections import ModelSectionConfig
from yammi.workflows.recording import _git_info, load_checkpoint_state
from yammi.workflows.task_catalog import TaskCatalog, TaskCatalogConfig, TaskKind

from .bundle import (
    BundleConfig,
    BundleError,
    DescriptorSpec,
    HeadSpec,
    MemberInfo,
    build_model,
    load_member,
    write_bundle,
)

__all__ = ["default_grid", "export_runs", "resolve_checkpoint"]

# Training-only parameters a checkpoint may carry that an inference model has no slot for.
_TRAINING_ONLY_PREFIXES = ("task_log_sigmas.",)


def resolve_checkpoint(run: str | Path) -> Path:
    """A run directory (``training/final_model.pt`` inside) or a checkpoint file."""
    path = Path(run)
    if path.is_file():
        return path
    candidate = path / "training" / "final_model.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"{path}: neither a checkpoint file nor a run directory with training/final_model.pt.")


def default_grid(rows: Sequence[np.ndarray]) -> tuple[tuple[float, float], tuple[float, ...]]:
    """Observed range and default prediction grid of a kernel-regression task's training rows.

    The grid is the shared one when every row uses the same ``t`` values (e.g. the DOS energy mesh);
    otherwise ``n`` evenly spaced points over the observed range, ``n`` being the most common row
    length (e.g. transport curves, resampled per row between that row's own bounds).
    """
    rows = [np.asarray(r, dtype=float) for r in rows if np.asarray(r).size]
    if not rows:
        raise BundleError("no training t values to derive a grid from.")
    values = np.concatenate(rows)
    values = values[np.isfinite(values)]
    lo, hi = float(values.min()), float(values.max())
    first = rows[0]
    if all(r.shape == first.shape and np.array_equal(r, first) for r in rows[1:]):
        return (lo, hi), tuple(float(v) for v in first)
    n = Counter(r.size for r in rows).most_common(1)[0][0]
    return (lo, hi), tuple(float(v) for v in np.linspace(lo, hi, int(n)))


def _train_targets(catalog: TaskCatalog, name: str) -> np.ndarray:
    """A task's training-split targets on the training scale, curves flattened."""
    spec = catalog.task_spec(name)
    frame = catalog.task_frames([name])[name]
    mask = frame[spec.column].notna()
    if "split" in frame.columns:
        mask &= frame["split"] == "train"
    cells = frame.loc[mask, spec.column]
    if spec.kind is TaskKind.KERNEL_REGRESSION:
        rows = [as_float_array(c) for c in cells]
        return np.concatenate(rows) if rows else np.array([])
    return cells.to_numpy(dtype=float)


def _head_spec(catalog: TaskCatalog, model_cfg: ModelSectionConfig, name: str) -> HeadSpec:
    spec = catalog.task_spec(name)
    kind = spec.kind
    common: dict[str, Any] = {"name": name, "unit": spec.unit, "source_column": spec.column}
    if kind is TaskKind.CLASSIFICATION:
        labels = catalog.class_labels(name)
        if labels is None:
            raise BundleError(
                f"task '{name}': a classification head needs class names; point its [tasks.scaler] at the "
                "LabelEncoder used to encode the labels."
            )
        return HeadSpec(
            kind="classification",
            hidden_dims=tuple(spec.hidden_dims or model_cfg.head_hidden_dims),
            num_classes=spec.num_classes,
            classes=labels,
            **common,
        )

    if spec.output_scale == "shape":
        transform = None
        scale = "shape"
    else:
        transform = catalog.target_transform(name)
        if transform is None:
            raise BundleError(
                f"task '{name}': no target transform. Configure [tasks.scaler] (the scaler used to normalise "
                'the column) or mark a per-row-normalised head with output_scale = "shape".'
            )
        transform = guard_inverse_domain(transform, _train_targets(catalog, name))
        scale = "physical"

    if kind is TaskKind.REGRESSION:
        return HeadSpec(
            kind="regression",
            hidden_dims=tuple(spec.hidden_dims or model_cfg.head_hidden_dims),
            output_scale=scale,
            transform=transform,
            **common,
        )

    assert spec.t_column is not None
    frame = catalog.task_frames([name])[name]
    mask = frame[spec.column].notna()
    if "split" in frame.columns:
        mask &= frame["split"] == "train"
    rows = [as_float_array(cell) for cell in frame.loc[mask, spec.t_column].dropna()]
    t_range, t_grid = default_grid(rows)
    return HeadSpec(
        kind="kernel_regression",
        x_hidden_dims=tuple(spec.x_hidden_dims or model_cfg.kr_x_hidden_dims),
        t_hidden_dims=tuple(spec.t_hidden_dims or model_cfg.kr_t_hidden_dims),
        n_kernel=spec.n_kernel or model_cfg.n_kernel,
        t_name=spec.t_column,
        t_unit=spec.t_unit,
        t_range=t_range,
        t_grid=t_grid,
        output_scale=scale,
        transform=transform,
        **common,
    )


def _inference_state(state: Mapping[str, torch.Tensor], source: Path) -> dict[str, torch.Tensor]:
    kept: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for key, tensor in state.items():
        if key.startswith(_TRAINING_ONLY_PREFIXES):
            dropped.append(key)
        else:
            kept[key] = tensor
    if dropped:
        logger.info(f"{source}: dropped {len(dropped)} training-only tensors ({dropped[0]}, …).")
    return kept


def _run_seed(checkpoint: Path) -> int | None:
    """The training seed recorded beside a run (``run_provenance.json``), when there is one."""
    for parent in (checkpoint.parent.parent, checkpoint.parent):
        prov = parent / "run_provenance.json"
        if prov.exists():
            try:
                seeds = json.loads(prov.read_text(encoding="utf-8")).get("seeds") or {}
            except json.JSONDecodeError:
                return None
            seed = seeds.get("seed", seeds.get("training.seed"))
            return int(seed) if isinstance(seed, (int, float)) else None
    return None


def _outputs(model: Any, heads: Sequence[HeadSpec], x: torch.Tensor) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        h = torch.tanh(model.encoder(x))
        out["__latent__"] = h
        for head in heads:
            module = model.task_heads[head.name]
            if head.kind == "kernel_regression":
                grid = torch.tensor(head.t_grid, dtype=torch.float32)
                eh, et = expand_for_kernel_regression(h, [grid] * h.shape[0])
                out[head.name] = module(eh, t=et)
            else:
                out[head.name] = module(h)
    return out


def _verify(
    bundle: Path,
    config: BundleConfig,
    members: Sequence[tuple[MemberInfo, Path]],
    catalog: TaskCatalog,
    model_cfg: ModelSectionConfig,
    n_compositions: int,
) -> None:
    names = [h.name for h in config.heads]
    pool: dict[str, None] = {}  # insertion-ordered set: O(1) membership, stops at n_compositions
    for name in names:
        for comp in catalog.task_frames([name])[name].index:
            if len(pool) >= n_compositions:
                break
            pool.setdefault(comp, None)
        if len(pool) >= n_compositions:
            break
    desc = catalog.descriptor_fn()(list(pool))
    x = torch.tensor(desc.values, dtype=torch.float32)
    for info, checkpoint in members:
        reference = build_model_for_checkpoint(catalog, model_cfg, names)
        reference.load_state_dict(load_checkpoint_state(checkpoint)["model"], strict=False)
        reference.eval()
        loaded, _, _ = load_member(bundle, info.id)
        expected, got = _outputs(reference, config.heads, x), _outputs(loaded, config.heads, x)
        for key, tensor in expected.items():
            if not torch.equal(tensor, got[key]):
                diff = float((tensor - got[key]).abs().max())
                raise BundleError(
                    f"member {info.id!r}: output '{key}' differs from the checkpoint (max |diff| {diff:.3e})."
                )
    logger.info(f"verified {len(members)} member(s) on {x.shape[0]} compositions: outputs identical.")


def export_runs(
    catalog_config: TaskCatalogConfig,
    model_config: ModelSectionConfig,
    runs: Sequence[str | Path],
    out: str | Path,
    *,
    license: str,
    member_ids: Sequence[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
    verify_compositions: int = 64,
) -> Path:
    """Export ``runs`` (run dirs or checkpoint files) as one bundle at ``out``; returns ``out``."""
    if not runs:
        raise BundleError("export needs at least one run.")
    if catalog_config.descriptor.kind != "kmd":
        raise BundleError("only KMD-descriptor models can be exported; precomputed descriptors are not portable.")
    checkpoints = [resolve_checkpoint(r) for r in runs]
    ids = list(member_ids) if member_ids is not None else [f"m{i:03d}" for i in range(len(checkpoints))]
    if len(ids) != len(checkpoints):
        raise BundleError(f"{len(ids)} member ids for {len(checkpoints)} runs.")

    catalog = TaskCatalog(catalog_config)
    states = [load_checkpoint_state(c) for c in checkpoints]
    head_sets = [set(checkpoint_task_order(s)) for s in states]
    if any(hs != head_sets[0] for hs in head_sets[1:]):
        odd = next(i for i, hs in enumerate(head_sets) if hs != head_sets[0])
        raise BundleError(
            f"members must share one set of heads: {checkpoints[odd]} differs from {checkpoints[0]} "
            f"(only in one: {sorted(head_sets[0] ^ head_sets[odd])})."
        )
    names = checkpoint_task_order(states[0])
    unknown = [n for n in names if n not in catalog.task_names]
    if unknown:
        raise BundleError(f"checkpoint heads {unknown} are not defined in the catalog.")

    config = BundleConfig(
        descriptor=DescriptorSpec.current(catalog_config.descriptor.n_grids),
        model=model_config,
        heads=tuple(_head_spec(catalog, model_config, n) for n in names),
        license=license,
        provenance={"git": _git_info(), **dict(provenance or {})},
    )
    reference = build_model(config).state_dict()
    members: list[tuple[MemberInfo, dict[str, torch.Tensor]]] = []
    for member_id, checkpoint, state in zip(ids, checkpoints, states):
        weights = _inference_state(state["model"], checkpoint)
        # The AE head is always present in a bundle model; it must be in the checkpoint too.
        if not any(k.startswith(f"task_heads.{AE_NAME}.") for k in weights) and any(
            k.startswith(f"task_heads.{AE_NAME}.") for k in reference
        ):
            raise BundleError(f"{checkpoint}: no autoencoder head; it is required for inverse design.")
        order = tuple(state.get("task_sequence") or checkpoint_task_order(state))
        info = MemberInfo(id=member_id, task_order=order, seed=_run_seed(checkpoint), source=str(checkpoint))
        members.append((info, weights))

    out = Path(out)
    write_bundle(out, config, members)  # staged: a failed write leaves nothing behind
    try:
        _verify(
            out,
            config,
            [(info, c) for (info, _), c in zip(members, checkpoints)],
            catalog,
            model_config,
            verify_compositions,
        )
    except Exception:
        shutil.rmtree(out, ignore_errors=True)  # never leave an unverified bundle behind
        raise
    return out
