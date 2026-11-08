# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``yammi finetune`` — frozen-encoder fine-tuning of selected task heads.

Rebuilds the checkpoint's model from the :class:`TaskCatalog`, loads its weights with
``strict=False``, optionally adds heads absent from the checkpoint, then fine-tunes only the
requested heads. Freeze policy (rewrite of ``finetune_inverse_heads.freeze_except`` with one
intentional change): the encoder and every non-target head are frozen, **except the built-in
autoencoder head which always stays trainable** (at ``ae_lr``); ``task_log_sigmas`` are frozen.
Non-target heads are disabled for the fit and restored before saving so the checkpoint keeps
every head.

From a released bundle (``--pretrained`` / ``[finetune].pretrained``) the bundle supplies the
architecture and every existing head, so the catalog lists only the user's own tasks. Their data are in
physical units:

* an existing head's targets go through the bundle's transform (or the per-row shape normalisation);
* a new head fits ``StandardScaler → PowerTransformer`` on its training split.

The run also writes a derived bundle (``<output>/bundle``). It carries the source licence and
``derived_from``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from loguru import logger

from ._engine import (
    AE_NAME,
    DropLastTrainCompoundDataModule,
    as_float_array,
    build_empty_model,
    guard_single_device,
    task_names_from_state,
    build_head_config,
    build_trainer_extras,
    evaluate_task,
    reapply_class_weights,
)
from ._sections import (
    ModelSectionConfig,
    TrainingSectionConfig,
    build_model_section,
    build_training_section,
    reject_unknown,
)
from .recording import RunRecorder, load_checkpoint_state
from .task_catalog import TaskCatalog, TaskCatalogConfig, build_task_catalog_config


_FINETUNE_ROOT_KEYS = {"data", "descriptor", "datasets", "tasks", "model", "training", "finetune", "output"}
_CATALOG_KEYS = {"data", "descriptor", "datasets", "tasks"}


@dataclass(kw_only=True)
class FinetuneConfig:
    """``[finetune]`` + shared sections — frozen-encoder head fine-tuning config."""

    catalog: TaskCatalogConfig
    model: ModelSectionConfig
    training: TrainingSectionConfig
    checkpoint: Path | None  # checkpoint to fine-tune from (--checkpoint or [finetune].checkpoint)
    tasks: list[str]  # heads to fine-tune; other heads stay frozen (the AE head always trains)
    output_dir: Path
    # Released bundle to fine-tune from instead of a checkpoint: a directory or owner/name[@rev][#member].
    pretrained: str | None = None
    epochs: int = 20  # fine-tune epochs (distinct from [training].max_epochs)
    freeze_encoder: bool = True  # freeze the shared encoder + non-target heads (BatchNorm buffers too)
    add_new_tasks: bool = True  # if a target task isn't in the checkpoint, add a fresh head for it

    def __post_init__(self) -> None:
        if (self.checkpoint is None) == (self.pretrained is None):
            raise ValueError("give exactly one of a checkpoint (--checkpoint) or a released bundle (--pretrained).")
        if self.checkpoint is not None:
            self.checkpoint = Path(self.checkpoint)
        self.output_dir = Path(self.output_dir)
        if not self.tasks:
            raise ValueError("finetune.tasks must be non-empty.")
        if self.epochs < 1:
            raise ValueError(f"finetune.epochs must be >= 1, got {self.epochs}.")
        catalog_tasks = {t.name for t in self.catalog.tasks}
        unknown = [t for t in self.tasks if t not in catalog_tasks]
        if unknown:
            raise ValueError(f"finetune.tasks references unknown task(s): {unknown}.")


def _with_class_counts(raw: Mapping[str, Any], pretrained: str) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """Fill ``num_classes`` of classification tasks that leave it out, when fine-tuning a released bundle.

    An existing head takes the bundle's count (the user's data may not show every class). A new head
    with string labels counts the distinct labels in its column; one with integer labels (class indices)
    takes ``max index + 1``. Explicit values are kept and checked later.
    """
    from yammi.workflows.task_catalog import TaskKind, _coerce_task_kind

    def is_classification(task: Mapping[str, Any]) -> bool:
        try:
            return _coerce_task_kind(task.get("kind")) is TaskKind.CLASSIFICATION
        except ValueError:
            return False  # the catalog reports the bad kind

    tasks = list(raw.get("tasks", []))
    if not any(is_classification(t) and "num_classes" not in t for t in tasks):
        return raw, ()
    from yammi.data.composition_sources import read_data_file
    from yammi.hub.bundle import read_config
    from yammi.hub.pretrained import resolve

    heads = {h.name: h for h in read_config(resolve(pretrained)[0]).heads}
    datasets = raw.get("datasets", {})
    filled, counted = [], []
    for task in tasks:
        if is_classification(task) and "num_classes" not in task:
            task = dict(task)
            head = heads.get(task.get("name", ""))
            if head is not None and head.num_classes:
                task["num_classes"] = head.num_classes
            else:
                counted.append(str(task.get("name")))
                path = datasets.get(task.get("dataset", ""), {}).get("path")
                if path is None or "column" not in task:
                    raise ValueError(f"Task '{task.get('name')}': give num_classes, or a dataset and column to count.")
                labels = read_data_file(str(path))[task["column"]].dropna()
                task["num_classes"] = _class_count(str(task.get("name")), labels)
        filled.append(task)
    return {**raw, "tasks": filled}, tuple(counted)


def _recount_after_filtering(raw: Mapping[str, Any], counted: Sequence[str]) -> Mapping[str, Any]:
    """Recount inferred class counts on the rows the catalog keeps (min_elements, sample, dropped rows).

    The first count, on the raw file, only lets the catalog config be built; this one sees the same
    composition-keyed rows that :func:`_target_inputs` builds the class vocabulary from.
    """
    if not counted:
        return raw
    catalog = TaskCatalog(build_task_catalog_config({k: raw[k] for k in _CATALOG_KEYS if k in raw}))
    frames = catalog.task_frames(list(counted))
    tasks = []
    for task in raw.get("tasks", []):
        if task.get("name") in counted:
            labels = frames[task["name"]][catalog.task_spec(task["name"]).column].dropna()
            task = {**task, "num_classes": _class_count(str(task["name"]), labels)}
        tasks.append(task)
    return {**raw, "tasks": tasks}


def _class_count(name: str, labels: Any) -> int:
    """Number of classes a label column implies: distinct names, or ``max + 1`` for class indices."""
    if labels.map(lambda v: isinstance(v, str)).all():
        return int(labels.nunique())
    values = np.asarray(labels, dtype=float)
    if not (np.isfinite(values).all() and (values >= 0).all() and (values == np.round(values)).all()):
        raise ValueError(f"Task '{name}': numeric class labels must be non-negative integer indices.")
    return int(values.max()) + 1 if values.size else 0


def build_finetune_config(
    raw: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    pretrained: str | None = None,
) -> FinetuneConfig:
    """Normalize a parsed-TOML tree into a :class:`FinetuneConfig`."""

    reject_unknown("<root>", raw, _FINETUNE_ROOT_KEYS)
    ft_raw = dict(raw.get("finetune", {}))
    reject_unknown(
        "finetune", ft_raw, {"checkpoint", "pretrained", "tasks", "epochs", "freeze_encoder", "add_new_tasks"}
    )
    resolved_pretrained = pretrained if pretrained is not None else ft_raw.get("pretrained")
    if resolved_pretrained is not None:
        raw, counted = _with_class_counts(raw, str(resolved_pretrained))
        raw = _recount_after_filtering(raw, counted)
    catalog = build_task_catalog_config({k: raw[k] for k in _CATALOG_KEYS if k in raw})
    training = build_training_section(raw.get("training", {}))

    resolved_checkpoint = checkpoint if checkpoint is not None else ft_raw.get("checkpoint")
    if resolved_pretrained is not None:
        if "model" in raw:
            raise ValueError(
                "[model] does not apply when fine-tuning a released bundle: the bundle defines the architecture."
            )
        if resolved_checkpoint is not None:
            raise ValueError("give either a checkpoint or a released bundle (--pretrained), not both.")
    elif resolved_checkpoint is None:
        raise ValueError("checkpoint must be given via --checkpoint or [finetune].checkpoint (or --pretrained).")
    model = build_model_section(raw.get("model", {}))
    resolved_output = output_dir if output_dir is not None else raw.get("output", {}).get("dir")
    if resolved_output is None:
        raise ValueError("output directory must be given via --output-dir or [output].dir.")

    return FinetuneConfig(
        catalog=catalog,
        model=model,
        training=training,
        checkpoint=Path(resolved_checkpoint) if resolved_checkpoint is not None else None,
        pretrained=str(resolved_pretrained) if resolved_pretrained is not None else None,
        tasks=list(ft_raw.get("tasks", [])),
        output_dir=Path(resolved_output),
        epochs=int(ft_raw.get("epochs", 20)),
        freeze_encoder=bool(ft_raw.get("freeze_encoder", True)),
        add_new_tasks=bool(ft_raw.get("add_new_tasks", True)),
    )


def apply_freeze_policy(model: Any, target_tasks: Sequence[str], *, freeze_encoder: bool) -> dict[str, bool]:
    """Freeze the encoder (optional) + non-target heads (except the AE) + task_log_sigmas.

    Returns the resulting ``{param_name: requires_grad}`` snapshot (taken AFTER applying the
    policy) for provenance. The autoencoder head (``"__reconstruction__"``) always stays
    trainable — the intentional behaviour change vs. the legacy ``freeze_except`` (which froze it).
    """
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    keep = set(target_tasks) | {AE_NAME}  # AE always trainable
    for head_name, head in model.task_heads.items():
        trainable = head_name in keep
        for p in head.parameters():
            p.requires_grad_(trainable)
    for p in model.task_log_sigmas.parameters():
        p.requires_grad_(False)
    return {name: p.requires_grad for name, p in model.named_parameters()}


class _FrozenEncoderEval(Callback):
    """Keep a frozen encoder in eval mode during fit.

    Freezing ``requires_grad`` stops gradients but Lightning still puts the module in train mode,
    so the encoder's ``BatchNorm1d`` ``running_mean``/``running_var`` would keep updating and the
    saved encoder would differ. Forcing eval mode each train epoch freezes those buffers too.
    """

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        pl_module.encoder.eval()


def run(cfg: FinetuneConfig, recorder: RunRecorder | None = None) -> dict[str, Any]:
    """Fine-tune ``cfg.tasks`` on top of ``cfg.checkpoint``; return the finetune summary dict."""

    if cfg.pretrained is not None:
        return _run_pretrained(cfg, recorder)
    assert cfg.checkpoint is not None
    catalog = TaskCatalog(cfg.catalog)
    owns_recorder = recorder is None
    rec = recorder or RunRecorder(cfg.output_dir)
    seed_everything(cfg.training.seed, workers=True)

    try:
        state = load_checkpoint_state(cfg.checkpoint)
        ckpt_tasks = list(state.get("task_sequence") or task_names_from_state(state["model"]))
        catalog_tasks = {t.name for t in cfg.catalog.tasks}
        missing_from_catalog = [t for t in ckpt_tasks if t not in catalog_tasks]
        if missing_from_catalog:
            raise ValueError(
                f"Checkpoint tasks {missing_from_catalog} are not defined in the catalog "
                f"(available: {sorted(catalog_tasks)})."
            )

        model = build_empty_model(catalog, cfg.model, cfg.training)
        for name in ckpt_tasks:
            model.add_task(build_head_config(catalog, cfg.model, cfg.training, name, masking_ratio=1.0))
        incompatible = model.load_state_dict(state["model"], strict=False)
        if incompatible.missing_keys:
            logger.info(
                f"load_state_dict missing keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys[:8]}"
            )
        if incompatible.unexpected_keys:
            logger.info(
                f"load_state_dict unexpected keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys[:8]}"
            )
        # A classification head stores its class weights as a buffer, so the checkpoint just
        # overwrote them with whatever the pretraining used. The fine-tune's own config decides the
        # weighting (`[[tasks]] class_weights`), so re-apply it from the head configs built above.
        reapply_class_weights(model, ckpt_tasks)

        added_tasks = [t for t in cfg.tasks if t not in model.task_heads]
        if added_tasks and not cfg.add_new_tasks:
            raise ValueError(
                f"Heads {added_tasks} are absent from the checkpoint and add_new_tasks=False "
                f"(checkpoint heads: {sorted(model.task_heads)})."
            )
        for name in added_tasks:
            logger.info(f"Adding new head '{name}' (absent from checkpoint).")
            model.add_task(build_head_config(catalog, cfg.model, cfg.training, name, masking_ratio=1.0))

        fit = _fit(model, catalog, cfg, rec)
        rec.save_final_model(model, ckpt_tasks + added_tasks, _task_spec_dump(catalog, ckpt_tasks + added_tasks))
        summary = {
            "from_checkpoint": str(cfg.checkpoint),
            "finetuned_heads": list(cfg.tasks),
            "added_tasks": added_tasks,
            "checkpoint_tasks": ckpt_tasks,
            "epochs": cfg.epochs,
            **fit,
        }
        (rec.paths.training / "finetune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"Fine-tuned {cfg.tasks} for {fit['epochs_run']} epochs → {rec.paths.training / 'final_model.pt'}")
        return summary
    finally:
        if owns_recorder:
            rec.close()


def _target_inputs(cfg: FinetuneConfig, bundle_config: Any) -> dict[str, Any]:
    """How each fine-tuned task's physical-unit data reach the training scale.

    Existing heads use the bundle's transform, its class names, or the shape normalisation. New heads
    get a transform fitted on their own training split, or class names from their labels.
    """
    from yammi.data.targets import ClassLabels, fit_default, strip_inverse_guard

    raw_catalog = TaskCatalog(cfg.catalog)  # untransformed, to read the user's own values
    heads = {h.name: h for h in bundle_config.heads}
    inputs: dict[str, Any] = {}
    for name in cfg.tasks:
        spec = raw_catalog.task_spec(name)
        if spec.scaler is not None:
            raise ValueError(
                f"Task '{name}': no [tasks.scaler] when fine-tuning a released bundle. Give the data in physical "
                "units; the bundle (existing heads) or the fit on your training split (new heads) normalises it."
            )
        kind = spec.kind.value
        if name in heads:
            head = heads[name]
            if head.kind != kind:
                raise ValueError(f"Task '{name}': the bundle's head is {head.kind}, the config says {kind}.")
            # Units are not converted: data must be in the head's own units. An omitted unit means "as the head".
            for label, given, expected in (("unit", spec.unit, head.unit), ("t_unit", spec.t_unit, head.t_unit)):
                if given is not None and expected is not None and given != expected:
                    raise ValueError(
                        f"Task '{name}': {label} {given!r} differs from the bundle head's {expected!r}; "
                        f"give the data in {expected} (units are not converted)."
                    )
            if kind == "classification":
                if spec.num_classes != head.num_classes:
                    raise ValueError(f"Task '{name}': num_classes {spec.num_classes} != the head's {head.num_classes}.")
                inputs[name] = head.classes
                continue
            # output_scale is inherited from the head when omitted; only an explicit conflict is an error.
            if spec.output_scale is not None and spec.output_scale != head.output_scale:
                raise ValueError(
                    f"Task '{name}': output_scale {spec.output_scale!r} disagrees with the bundle's head ({head.output_scale})."
                )
            # Labels are normalised with the unguarded transform, so held-out values past the source's
            # training range are not clipped; the guard only applies when predictions are inverted.
            inputs[name] = "shape" if head.output_scale == "shape" else strip_inverse_guard(head.transform)
            continue
        frame = raw_catalog.task_frames([name])[name]
        labeled = frame[spec.column].notna()
        mask = labeled & (frame["split"] == "train") if "split" in frame.columns else labeled
        train = frame.loc[mask, spec.column]
        if kind == "classification":
            assert spec.num_classes is not None
            every = frame.loc[labeled, spec.column]  # the class vocabulary: every labelled row, not only train
            if every.map(lambda v: isinstance(v, str)).all():
                names = sorted({str(v) for v in every})
                if len(names) != spec.num_classes:
                    raise ValueError(
                        f"Task '{name}': {len(names)} class names in the data the catalog keeps, num_classes = "
                        f"{spec.num_classes}. Set num_classes to match, or check rows dropped by min_elements."
                    )
                inputs[name] = ClassLabels(tuple(names))
            else:
                inputs[name] = ClassLabels(tuple(str(i) for i in range(spec.num_classes)))
        elif spec.output_scale == "shape":
            inputs[name] = "shape"
        else:
            values = train if kind == "regression" else [as_float_array(c) for c in train]
            flat = np.concatenate(values) if kind == "kernel_regression" and len(values) else np.asarray(values, float)
            # Unguarded here, so every split's labels keep their values; the derived bundle's head spec
            # (``_head_spec``) adds the guard for inverting predictions.
            inputs[name] = fit_default(flat)
    return inputs


def _reguarded(head: Any, catalog: TaskCatalog) -> Any:
    """A fine-tuned existing head whose inverse guard covers the source's range and the new training data."""
    from dataclasses import replace

    from yammi.data.targets import ClipStep, guard_inverse_domain, strip_inverse_guard
    from yammi.hub.export import _train_targets

    if head.transform is None:
        return head
    old = head.transform.steps[-1] if head.transform.steps else None
    base = strip_inverse_guard(head.transform)
    # Keep the source's guard bounds, so the clip only widens; a source without a guard (schema v1, or an
    # everywhere-invertible transform) still gets one where it is needed.
    kept = [b for b in (old.lower, old.upper) if b is not None] if isinstance(old, ClipStep) and old.guard else []
    values = np.concatenate([_train_targets(catalog, head.name), np.asarray(kept, dtype=float)])
    guarded = guard_inverse_domain(base, values)
    return head if guarded == head.transform else replace(head, transform=guarded)


def _refreshed_curve(head: Any, catalog: TaskCatalog) -> Any:
    """A fine-tuned existing curve head's metadata, covering both the source and the new training ``t``.

    The observed range becomes the union. The default grid is kept when it already spans that range;
    otherwise it is re-spaced over the union with the same number of points.
    """
    from dataclasses import replace

    spec = catalog.task_spec(head.name)
    assert spec.t_column is not None
    frame = catalog.task_frames([head.name])[head.name]
    mask = frame[spec.column].notna()
    if "split" in frame.columns:
        mask &= frame["split"] == "train"
    rows = [as_float_array(c) for c in frame.loc[mask, spec.t_column].dropna()]
    values = np.concatenate(rows) if rows else np.array([])
    values = values[np.isfinite(values)]
    if values.size == 0 or head.t_range is None:
        return head
    lo, hi = min(head.t_range[0], float(values.min())), max(head.t_range[1], float(values.max()))
    grid = head.t_grid
    if grid and (grid[0] > lo or grid[-1] < hi):
        grid = tuple(float(v) for v in np.linspace(lo, hi, len(grid)))
    return replace(head, t_range=(lo, hi), t_grid=grid)


def _run_pretrained(cfg: FinetuneConfig, recorder: RunRecorder | None) -> dict[str, Any]:
    """Fine-tune from a released bundle; writes the legacy checkpoint and a derived bundle."""
    from yammi.hub.bundle import load_member
    from yammi.hub.derive import save_derived
    from yammi.hub.export import _head_spec
    from yammi.hub.pretrained import resolve

    assert cfg.pretrained is not None
    owns_recorder = recorder is None
    rec = recorder or RunRecorder(cfg.output_dir)
    seed_everything(cfg.training.seed, workers=True)
    try:
        if cfg.training.learnable_loss_balancer:
            logger.warning("learnable_loss_balancer is ignored when fine-tuning a released bundle.")
        root, src = resolve(cfg.pretrained)
        model, bundle_config, member = load_member(root, src.member, training=cfg.training)
        desc = cfg.catalog.descriptor
        if desc.kind != "kmd" or desc.n_grids != bundle_config.descriptor.n_grids:
            raise ValueError(
                f'[descriptor] must match the bundle: kind = "kmd", n_grids = {bundle_config.descriptor.n_grids} '
                f"(the config has kind = {desc.kind!r}, n_grids = {desc.n_grids})."
            )
        catalog = TaskCatalog(cfg.catalog, target_inputs=_target_inputs(cfg, bundle_config))
        section = bundle_config.model
        existing = [h.name for h in bundle_config.heads]

        # Classification heads being fine-tuned weight their loss from the user's labels, as configured.
        for name in cfg.tasks:
            spec = catalog.task_spec(name)
            if name in existing and spec.class_weights == "balanced":
                model.task_configs_map[name].class_weights = catalog._class_weights(name)  # noqa: SLF001
        reapply_class_weights(model, [n for n in cfg.tasks if n in existing])
        # Per-task lr / weight_decay overrides apply to existing heads exactly as to new ones.
        for name in cfg.tasks:
            spec = catalog.task_spec(name)
            if name in existing and (spec.lr is not None or spec.weight_decay is not None):
                is_kr = spec.kind.value == "kernel_regression"
                lr = spec.lr if spec.lr is not None else (cfg.training.kr_lr if is_kr else cfg.training.head_lr)
                wd = (
                    spec.weight_decay
                    if spec.weight_decay is not None
                    else (cfg.training.kr_weight_decay if is_kr else cfg.training.head_weight_decay)
                )
                model.task_configs_map[name].optimizer = cfg.training.optimizer_config(lr=lr, weight_decay=wd)

        added = [t for t in cfg.tasks if t not in existing]
        if added and not cfg.add_new_tasks:
            raise ValueError(
                f"Heads {added} are not in the bundle and add_new_tasks = false (bundle heads: {existing})."
            )
        for name in added:
            logger.info(f"Adding new head '{name}' to the released model.")
            model.add_task(build_head_config(catalog, section, cfg.training, name, masking_ratio=1.0))

        fit = _fit(model, catalog, cfg, rec)
        order = list(member.task_order) + added
        rec.save_final_model(model, existing + added, _task_spec_dump(catalog, [t for t in cfg.tasks]))

        heads = [
            _reguarded(_refreshed_curve(h, catalog) if h.kind == "kernel_regression" else h, catalog)
            if h.name in cfg.tasks
            else h
            for h in bundle_config.heads
        ] + [_head_spec(catalog, section, name) for name in added]
        derived_from = {
            "source": cfg.pretrained,
            "revision": src.revision,
            "member": member.id,
            "license": bundle_config.license,
            "yammi_version": bundle_config.yammi_version,
            "fm_version": bundle_config.yammi_version,  # pre-2.1.1 key, kept for existing readers
        }
        bundle_dir = save_derived(
            rec.paths.root / "bundle",
            bundle_config,
            model.eval(),
            heads=heads,
            task_order=order,
            derived_from=derived_from,
            provenance={"finetuned_heads": list(cfg.tasks), "added_heads": added},
        )
        summary = {
            "from_pretrained": cfg.pretrained,
            "member": member.id,
            "license": bundle_config.license,
            "finetuned_heads": list(cfg.tasks),
            "added_tasks": added,
            "bundle": str(bundle_dir),
            "epochs": cfg.epochs,
            **fit,
        }
        (rec.paths.training / "finetune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"Fine-tuned {cfg.tasks} from {cfg.pretrained} → {bundle_dir} (licence {bundle_config.license})")
        return summary
    finally:
        if owns_recorder:
            rec.close()


def _fit(model: Any, catalog: TaskCatalog, cfg: FinetuneConfig, rec: RunRecorder) -> dict[str, Any]:
    """Freeze, fit ``cfg.tasks`` and evaluate before / after; non-target heads are restored afterwards."""
    apply_freeze_policy(model, cfg.tasks, freeze_encoder=cfg.freeze_encoder)
    frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)

    # Disable non-target supervised heads for the fit (keep the AE active + trainable); restore
    # before saving so the checkpoint retains every head.
    disabled = [n for n in list(model.task_heads) if n not in cfg.tasks and n != AE_NAME]
    if disabled:
        model.disable_task(*disabled)

    datamodule = catalog.build_datamodule(
        cfg.tasks,
        masking_ratios={t: 1.0 for t in cfg.tasks},
        datamodule_cls=DropLastTrainCompoundDataModule,  # drop size-1 tail batch (BatchNorm)
    )
    datamodule.setup("fit")
    test_keys = _test_keys(datamodule)

    before = {
        name: evaluate_task(
            model, catalog, name, rec, rec.paths.training / "before_finetune", is_new=False, test_keys=test_keys
        )
        for name in cfg.tasks
    }

    callbacks, loggers, enable_ckpt = build_trainer_extras(
        cfg.training,
        log_dir=rec.paths.root / "logs",
        ckpt_dir=rec.paths.training / "finetune" / "lightning",
        run_name="finetune",
    )
    if cfg.freeze_encoder:
        callbacks.append(_FrozenEncoderEval())  # keep frozen encoder's BatchNorm buffers fixed
    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=loggers,
        enable_checkpointing=enable_ckpt,
        enable_progress_bar=False,
        callbacks=callbacks,
    )
    guard_single_device(trainer)
    trainer.fit(model, datamodule=datamodule)

    if disabled:
        model.enable_task(*disabled)

    eval_dir = rec.paths.training / "finetune"
    after: dict[str, dict[str, float]] = {}
    for name in cfg.tasks:
        after[name] = evaluate_task(model, catalog, name, rec, eval_dir, is_new=True, test_keys=test_keys)
        lost = int(after[name].get("nonfinite", 0) or 0)
        if lost:
            # These metrics leave such rows out; a bundle made from this model returns them as NaN (or, with
            # saturated="clip", as the most extreme training value), so its predictions score lower.
            logger.warning(
                f"{name}: {lost} test prediction(s) fall outside the range the head's transform can invert and "
                "are left out of the metrics above."
            )

    return {
        "epochs_run": trainer.current_epoch,
        "freeze_encoder": cfg.freeze_encoder,
        "frozen_params": frozen,
        "trainable_params": trainable,
        "metrics_before": before,
        "metrics_after": after,
    }


def _test_keys(datamodule: Any) -> set[str] | None:
    if datamodule.split_series is None:
        return None
    resolved = datamodule.split_series
    return set(resolved.index[resolved == "test"].astype(str))


def _task_spec_dump(catalog: TaskCatalog, task_names: Sequence[str]) -> dict[str, Any]:
    dump: dict[str, Any] = {}
    for name in task_names:
        spec = catalog.task_spec(name)
        dump[name] = {"kind": spec.kind.value, "column": spec.column, "source": spec.dataset}
    return dump
