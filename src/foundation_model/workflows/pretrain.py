# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``fm pretrain`` — continual-rehearsal pre-training engine with an n-runs sweep.

Migrated from ``scripts/continual_rehearsal_full.py`` (``ContinualRehearsalFullRunner.run`` /
``_build_empty_model`` / ``_evaluate_task``) with three deliberate changes vs. the legacy runner:

1. **Rehearsal interval** (new): on step ``s`` (1-based) already-learned tasks join training only
   when ``s % interval == 0``; other steps train the new task + AE alone. ``interval == 1``
   reproduces the legacy "every step replays every old task" behaviour. Evaluation still covers
   **all** learned heads at every step (the forgetting trajectory is interval-independent).
2. **Replay amount** replaces the legacy ``fixed_tail`` / ``replay_ratio_high`` two-tier design:
   a global ``default_replay`` plus per-task overrides, each a fraction (< 1) or an absolute
   label count (>= 1, converted to a ratio against the task's valid-label count).
3. **n-runs sweep**: ``n_runs > 1`` writes ``runs/runNN/`` subdirectories with a per-run seed and
   (optionally) a reshuffled task order, aggregated into a top-level ``experiment_records.json``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from lightning import Trainer, seed_everything
from loguru import logger

from . import plots
from ._engine import (
    DropLastTrainCompoundDataModule,
    build_empty_model,
    build_head_config,
    build_trainer_extras,
    checkpoint_task_order,
    evaluate_task,
)
from ._sections import (
    ModelSectionConfig,
    TrainingSectionConfig,
    build_model_section,
    build_training_section,
    reject_unknown,
)
from .recording import RunRecorder, load_checkpoint_state
from .task_catalog import TaskCatalog, TaskCatalogConfig, TaskKind, build_task_catalog_config

# Stable qualitative palette so each task keeps one colour across the forgetting figure.
_TASK_PALETTE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
    "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#E377C2", "#17BECF",
)  # fmt: skip

_PRETRAIN_ROOT_KEYS = {"data", "descriptor", "datasets", "tasks", "model", "training", "pretrain", "output"}
_CATALOG_KEYS = {"data", "descriptor", "datasets", "tasks"}


class TaskOrder(str, Enum):
    FIXED = "fixed"
    RANDOM = "random"


# --- config dataclasses -------------------------------------------------------------------


def _validate_replay_amount(value: float | int, *, where: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{where}: replay must be a number, got bool {value!r}.")
    if isinstance(value, float):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{where}: replay float must be in (0, 1), got {value}.")
    elif isinstance(value, int):
        if value < 1:
            raise ValueError(f"{where}: replay int must be >= 1, got {value}.")
    else:
        raise ValueError(f"{where}: replay must be a float in (0, 1) or int >= 1, got {value!r}.")


@dataclass(kw_only=True)
class RehearsalConfig:
    """[pretrain.rehearsal] — interval schedule + replay amounts."""

    interval: int = 1  # old tasks rejoin training every Nth step (1 = always replay)
    default_replay: float | int = (
        0.05  # replay amount per old task: float in (0,1) = fraction of its labels, int >= 1 = count
    )
    per_task: dict[str, float | int] = field(default_factory=dict)  # override default_replay for named tasks

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError(f"rehearsal.interval must be >= 1, got {self.interval}.")
        _validate_replay_amount(self.default_replay, where="rehearsal.default_replay")
        for task, value in self.per_task.items():
            _validate_replay_amount(value, where=f"rehearsal.per_task.{task}")


@dataclass(kw_only=True)
class PretrainConfig:
    """``[pretrain]`` + shared sections — continual-rehearsal pre-training config."""

    catalog: TaskCatalogConfig
    model: ModelSectionConfig
    training: TrainingSectionConfig
    rehearsal: RehearsalConfig
    output_dir: Path
    task_sequence: list[str] = field(default_factory=list)  # order tasks are introduced; [] = [[tasks]] order
    n_runs: int = 1  # independent repeats (different seeds) written to runs/runNN/
    task_order: TaskOrder = TaskOrder.FIXED  # "fixed" (task_sequence order) or "random" (per-run shuffle)
    # Random-order controls (both require task_order = "random"). task_order_seed decouples the
    # shuffle RNG from the run seed: run i shuffles with task_order_seed + i (None = the run seed,
    # i.e. [training].seed + i — the historical behavior, still reproducible). task_order_groups
    # constrains the shuffle: tasks are shuffled within each group and the groups are concatenated
    # in the listed order; the groups must exactly partition task_sequence. Lets e.g. expensive
    # kernel-regression tasks stay in a final block while the order within blocks is randomized.
    task_order_seed: int | None = None
    task_order_groups: list[list[str]] = field(default_factory=list)
    # Warm-start: load encoder + heads from this checkpoint, treat its tasks as already learned, and
    # continue the sequence with the task_sequence tasks it doesn't already contain. None = from scratch.
    checkpoint: Path | None = None
    # Resume: on (re)start, if a run's output dir already holds step checkpoints, warm-start from the
    # latest and continue in place — so a job killed mid-sequence picks up at the next task. Completed
    # runs (final_model.pt present) are skipped. Optimizer state is NOT restored (each step trains a
    # fresh optimizer anyway); the resume granularity is one completed task-step.
    resume: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.checkpoint is not None:
            self.checkpoint = Path(self.checkpoint)
        if not isinstance(self.task_order, TaskOrder):
            self.task_order = TaskOrder(str(self.task_order))
        if self.n_runs < 1:
            raise ValueError(f"pretrain.n_runs must be >= 1, got {self.n_runs}.")
        catalog_tasks = {t.name for t in self.catalog.tasks}
        if not self.task_sequence:
            self.task_sequence = [t.name for t in self.catalog.tasks]
        unknown = [t for t in self.task_sequence if t not in catalog_tasks]
        if unknown:
            raise ValueError(f"pretrain.task_sequence references unknown task(s): {unknown}.")
        bad_replay = [t for t in self.rehearsal.per_task if t not in catalog_tasks]
        if bad_replay:
            raise ValueError(f"rehearsal.per_task references unknown task(s): {bad_replay}.")
        if self.task_order is not TaskOrder.RANDOM:
            if self.task_order_seed is not None:
                raise ValueError('pretrain.task_order_seed requires task_order = "random".')
            if self.task_order_groups:
                raise ValueError('pretrain.task_order_groups requires task_order = "random".')
        if self.task_order_groups:
            flat = [t for group in self.task_order_groups for t in group]
            dupes = sorted({t for t in flat if flat.count(t) > 1})
            if dupes:
                raise ValueError(f"pretrain.task_order_groups lists task(s) more than once: {dupes}.")
            missing = sorted(set(self.task_sequence) - set(flat))
            extra = sorted(set(flat) - set(self.task_sequence))
            if missing or extra:
                raise ValueError(
                    "pretrain.task_order_groups must exactly partition task_sequence "
                    f"(missing: {missing}, not in task_sequence: {extra})."
                )


# --- builder ------------------------------------------------------------------------------


def build_pretrain_config(
    raw: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    resume: bool = False,
) -> PretrainConfig:
    """Normalize a parsed-TOML tree into a :class:`PretrainConfig`."""

    reject_unknown("<root>", raw, _PRETRAIN_ROOT_KEYS)
    catalog = build_task_catalog_config({k: raw[k] for k in _CATALOG_KEYS if k in raw})
    model = build_model_section(raw.get("model", {}))
    training = build_training_section(raw.get("training", {}))

    pretrain_raw = dict(raw.get("pretrain", {}))
    reject_unknown(
        "pretrain",
        pretrain_raw,
        {
            "task_sequence",
            "n_runs",
            "task_order",
            "task_order_seed",
            "task_order_groups",
            "rehearsal",
            "checkpoint",
            "resume",
        },
    )
    rehearsal_raw = dict(pretrain_raw.get("rehearsal", {}))
    reject_unknown("pretrain.rehearsal", rehearsal_raw, {"interval", "default_replay", "per_task"})
    rehearsal = RehearsalConfig(
        interval=rehearsal_raw.get("interval", 1),
        default_replay=rehearsal_raw.get("default_replay", 0.05),
        per_task=dict(rehearsal_raw.get("per_task", {})),
    )

    resolved_output = output_dir if output_dir is not None else raw.get("output", {}).get("dir")
    if resolved_output is None:
        raise ValueError("output directory must be given via --output-dir or [output].dir.")
    resolved_checkpoint = checkpoint if checkpoint is not None else pretrain_raw.get("checkpoint")

    return PretrainConfig(
        catalog=catalog,
        model=model,
        training=training,
        rehearsal=rehearsal,
        output_dir=Path(resolved_output),
        task_sequence=list(pretrain_raw.get("task_sequence", [])),
        n_runs=int(pretrain_raw.get("n_runs", 1)),
        task_order=TaskOrder(str(pretrain_raw.get("task_order", "fixed"))),
        task_order_seed=(None if pretrain_raw.get("task_order_seed") is None else int(pretrain_raw["task_order_seed"])),
        task_order_groups=[list(group) for group in pretrain_raw.get("task_order_groups", [])],
        checkpoint=Path(resolved_checkpoint) if resolved_checkpoint is not None else None,
        resume=resume or bool(pretrain_raw.get("resume", False)),
    )


# --- pure helpers -------------------------------------------------------------------------


def active_old_tasks(step: int, learned: Sequence[str], interval: int) -> list[str]:
    """Old tasks that participate in training at 1-based ``step`` (empty on non-interval steps)."""
    if interval <= 1 or step % interval == 0:
        return list(learned)
    return []


def replay_to_ratio(replay: float | int, n_valid: int) -> float:
    """Convert a replay amount (fraction or absolute count) to a keep-ratio in [0, 1]."""
    if isinstance(replay, float):
        return replay
    if n_valid <= 0:
        return 1.0
    return min(1.0, replay / n_valid)


# --- engine -------------------------------------------------------------------------------


def run(cfg: PretrainConfig, recorder: RunRecorder | None = None) -> dict[str, Any]:
    """Execute the pretraining sweep. Returns the aggregated cross-run records dict.

    ``recorder`` is the root recorder (already carrying provenance); per-run recorders are created
    under it. When ``n_runs == 1`` the single run writes directly under ``output_dir``.
    """

    catalog = TaskCatalog(cfg.catalog)
    owns_recorder = recorder is None  # close only the recorder we create (leave callers' alone)
    root_recorder = recorder or RunRecorder(cfg.output_dir)
    aggregate: list[dict[str, Any]] = []

    try:
        for run_idx in range(cfg.n_runs):
            run_seed = cfg.training.seed + run_idx
            order = _run_task_order(cfg, run_seed, run_idx)
            if cfg.n_runs == 1:
                run_recorder = root_recorder
            else:
                run_recorder = RunRecorder(cfg.output_dir / "runs" / f"run{run_idx:02d}")

            # Resume: prefer a run's own latest step checkpoint (continue in place after a kill);
            # fall back to the explicit warm-start checkpoint. A finished run is skipped.
            source = cfg.checkpoint
            if cfg.resume:
                training_dir = run_recorder.paths.training
                if (training_dir / "final_model.pt").exists():
                    logger.info(f"=== pretrain run {run_idx + 1}/{cfg.n_runs}: already complete, skipping (resume) ===")
                    continue
                source = _latest_step_checkpoint(training_dir) or cfg.checkpoint
                if source is not None:
                    logger.info(f"resuming run {run_idx + 1}/{cfg.n_runs} from {source}")

            logger.info(f"=== pretrain run {run_idx + 1}/{cfg.n_runs} (seed={run_seed}) order={order} ===")
            records = _run_single(
                cfg,
                catalog,
                run_recorder,
                task_order=order,
                seed=run_seed,
                warm_start_source=source,
                is_resume=cfg.resume,
            )
            for rec in records:
                aggregate.append({"run": run_idx, **rec})
            if run_recorder is not root_recorder:
                run_recorder.close()

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (cfg.output_dir / "experiment_records.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        return {"records": aggregate}
    finally:
        if owns_recorder:
            root_recorder.close()


def _run_task_order(cfg: PretrainConfig, run_seed: int, run_idx: int = 0) -> list[str]:
    if cfg.task_order is not TaskOrder.RANDOM:
        return list(cfg.task_sequence)
    seed = run_seed if cfg.task_order_seed is None else cfg.task_order_seed + run_idx
    rng = np.random.default_rng(seed)
    order: list[str] = []
    for group in cfg.task_order_groups or [cfg.task_sequence]:
        block = list(group)
        rng.shuffle(block)
        order.extend(block)
    return order


def _latest_step_checkpoint(training_dir: Path) -> Path | None:
    """The highest-numbered ``stepNN_*/checkpoint.pt`` under a run's training dir (for --resume)."""
    if not training_dir.exists():
        return None
    candidates = sorted(training_dir.glob("step*/checkpoint.pt"), key=lambda p: p.parent.name)
    return candidates[-1] if candidates else None


def _warm_start(
    cfg: PretrainConfig, catalog: TaskCatalog, source: Path | None
) -> tuple[Any, list[str], dict[str, Any]]:
    """Build the run's start model. With ``source`` set, load its encoder + heads (warm-start) and
    return those tasks as already-learned; otherwise an empty (AE-only) model."""
    model = build_empty_model(catalog, cfg.model, cfg.training)
    if source is None:
        return model, [], {}
    state = load_checkpoint_state(source)
    preloaded = checkpoint_task_order(state)
    catalog_tasks = {t.name for t in cfg.catalog.tasks}
    missing = [t for t in preloaded if t not in catalog_tasks]
    if missing:
        raise ValueError(f"pretrain.checkpoint tasks {missing} are not in the catalog (have {sorted(catalog_tasks)}).")
    built: dict[str, Any] = {}
    for name in preloaded:
        built[name] = build_head_config(catalog, cfg.model, cfg.training, name, init_from_data=False)
        model.add_task(built[name])
    incompatible = model.load_state_dict(state["model"], strict=False)
    if incompatible.missing_keys:
        logger.info(f"warm-start: {len(incompatible.missing_keys)} missing key(s) e.g. {incompatible.missing_keys[:6]}")
    logger.info(f"warm-started from {source} with heads {preloaded}")
    return model, preloaded, built


def _run_single(
    cfg: PretrainConfig,
    catalog: TaskCatalog,
    recorder: RunRecorder,
    *,
    task_order: Sequence[str],
    seed: int,
    warm_start_source: Path | None = None,
    is_resume: bool = False,
) -> list[dict[str, Any]]:
    seed_everything(seed, workers=True)
    model, preloaded, built = _warm_start(cfg, catalog, warm_start_source)

    # New tasks = task_order entries not already loaded from the checkpoint; preloaded tasks are
    # already trained and only participate as rehearsal/eval targets.
    new_tasks = [t for t in task_order if t not in preloaded]
    if not new_tasks:
        if is_resume:  # killed after the last step but before final_model was written — just finalize
            recorder.save_final_model(model, list(preloaded), _task_spec_dump(catalog, preloaded))
            logger.info("resume: every task already trained; wrote final_model.")
            return []
        raise ValueError("nothing to train: every task in task_sequence is already in pretrain.checkpoint.")
    full_order = [*preloaded, *new_tasks]

    task_colors = {name: _TASK_PALETTE[i % len(_TASK_PALETTE)] for i, name in enumerate(full_order)}
    clf_tasks = frozenset(n for n in full_order if catalog.task_spec(n).kind is TaskKind.CLASSIFICATION)
    metric_history: dict[str, list[tuple[int, float]]] = {name: [] for name in full_order}
    records: list[dict[str, Any]] = []

    # Global step numbering continues past the preloaded tasks so the rehearsal-interval schedule
    # picks up where the checkpoint left off (step == 1 for a from-scratch run, offset == 0).
    offset = len(preloaded)
    for i, task_name in enumerate(new_tasks):
        step = offset + i + 1
        logger.info(f"--- step {step}: add '{task_name}' (new {i + 1}/{len(new_tasks)}) ---")
        built[task_name] = build_head_config(catalog, cfg.model, cfg.training, task_name, masking_ratio=1.0)
        model.add_task(built[task_name])

        learned = [*preloaded, *new_tasks[:i]]
        participating = active_old_tasks(step, learned, cfg.rehearsal.interval)
        active = [task_name, *participating]
        for name in participating:
            built[name].task_masking_ratio = _replay_ratio_for(cfg, catalog, name)
        built[task_name].task_masking_ratio = 1.0

        datamodule = DropLastTrainCompoundDataModule(
            task_configs=[built[name] for name in active],
            descriptor_fn=catalog.descriptor_fn(),
            task_frames={name: catalog.task_frames([name])[name] for name in active},
            composition_column=cfg.catalog.data.composition_column,
            random_seed=cfg.catalog.data.split_random_seed,
            val_split=cfg.catalog.data.val_split,
            test_split=cfg.catalog.data.test_split,
            batch_size=cfg.catalog.data.batch_size,
            num_workers=cfg.catalog.data.num_workers,
        )
        callbacks, loggers, enable_ckpt = build_trainer_extras(
            cfg.training,
            log_dir=recorder.paths.root / "logs",
            ckpt_dir=recorder.paths.step_dir(step, task_name) / "lightning",
            run_name=f"step{step:02d}_{task_name}",
        )
        trainer = Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator=cfg.training.accelerator,
            devices=cfg.training.devices,
            logger=loggers,
            enable_checkpointing=enable_ckpt,
            enable_progress_bar=False,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule=datamodule)

        test_keys: set[str] | None = None
        if datamodule.split_series is not None:
            resolved = datamodule.split_series
            test_keys = set(resolved.index[resolved == "test"].astype(str))

        step_dir = recorder.paths.step_dir(step, task_name)
        step_metrics: dict[str, dict[str, float]] = {}
        for name in [*preloaded, *new_tasks[: i + 1]]:  # ALL learned heads (preloaded + new-so-far)
            metric = evaluate_task(
                model, catalog, name, recorder, step_dir, is_new=(name == task_name), test_keys=test_keys
            )
            step_metrics[name] = metric
            metric_history[name].append((step, metric["primary"]))

        recorder.save_step_checkpoint(step, task_name, model, list(active))
        record = {"step": step, "new_task": task_name, "epochs_run": trainer.current_epoch, "metrics": step_metrics}
        records.append(record)
        recorder.append_record(record)

    fig = plots.plot_forgetting(metric_history, task_colors=task_colors, clf_tasks=clf_tasks)
    recorder.save_figure("training/forgetting_trajectory.png", fig)
    plots.plt.close(fig)
    recorder.write_records()
    recorder.write_metrics_table()
    recorder.save_final_model(model, list(full_order), _task_spec_dump(catalog, full_order))
    return records


def _replay_ratio_for(cfg: PretrainConfig, catalog: TaskCatalog, name: str) -> float:
    replay = cfg.rehearsal.per_task.get(name, cfg.rehearsal.default_replay)
    if isinstance(replay, float):
        return replay
    return replay_to_ratio(replay, _n_valid_train_labels(catalog, name))


def _n_valid_train_labels(catalog: TaskCatalog, name: str) -> int:
    spec = catalog.task_spec(name)
    frame = catalog.task_frames([name])[name]
    mask = frame[spec.column].notna()
    if "split" in frame.columns:
        mask &= frame["split"] == "train"
    return int(mask.sum())


def _task_spec_dump(catalog: TaskCatalog, task_order: Sequence[str]) -> dict[str, Any]:
    dump: dict[str, Any] = {}
    for name in task_order:
        spec = catalog.task_spec(name)
        dump[name] = {"kind": spec.kind.value, "column": spec.column, "source": spec.dataset}
    return dump
