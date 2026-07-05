# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``fm`` — the unified foundation-model CLI.

This layer is intentionally THIN: each subcommand loads a TOML file, applies ``--set`` /
first-class-flag overrides onto the raw tree, builds the workflow config dataclass, writes
provenance, and calls exactly one ``workflows.<mod>.run(cfg)``. No business logic lives here.
"""

from __future__ import annotations

import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import click

from foundation_model.workflows.finetune import FinetuneConfig, build_finetune_config
from foundation_model.workflows.finetune import run as finetune_run
from foundation_model.workflows.inverse import InverseConfig, build_inverse_config
from foundation_model.workflows.inverse import run as inverse_run
from foundation_model.workflows.predict import PredictConfig, build_predict_config
from foundation_model.workflows.predict import run as predict_run
from foundation_model.workflows.pretrain import PretrainConfig, build_pretrain_config
from foundation_model.workflows.pretrain import run as pretrain_run
from foundation_model.workflows.recording import RunRecorder


def _parse_toml_value(value: str) -> Any:
    """Parse a ``--set`` value with TOML semantics (strings must be quoted, e.g. ``'"foo"'``)."""
    try:
        return tomllib.loads(f"_v_ = {value}")["_v_"]
    except tomllib.TOMLDecodeError as exc:
        raise click.BadParameter(f"could not parse override value {value!r} as TOML: {exc}") from exc


def _set_dotted(tree: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = tree
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def load_raw_config(
    config_path: str | Path,
    overrides: Sequence[str] = (),
    *,
    seed: int | None = None,
    accelerator: str | None = None,
    sample: int | None = None,
    seed_key: str = "training.seed",
    accelerator_key: str = "training.accelerator",
) -> dict[str, Any]:
    """Load a TOML file and apply ``--set`` overrides plus common first-class flags.

    ``seed_key`` / ``accelerator_key`` route ``--seed`` / ``--accelerator`` to the right section
    per subcommand (``[training]`` for pretrain/finetune, ``[inverse]`` / ``[predict]`` for those),
    so they never inject a section the config builder would reject.
    """

    with open(config_path, "rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)

    for override in overrides:
        if "=" not in override:
            raise click.BadParameter(f"--set expects SECTION.KEY=VALUE, got {override!r}")
        key, _, value = override.partition("=")
        _set_dotted(raw, key.strip(), _parse_toml_value(value.strip()))

    if seed is not None:
        _set_dotted(raw, seed_key, seed)
    if accelerator is not None:
        _set_dotted(raw, accelerator_key, accelerator)
    if sample is not None:
        for name in raw.get("datasets", {}):
            _set_dotted(raw, f"datasets.{name}.sample", sample)
    return raw


def common_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Attach the options shared by every subcommand."""
    options = [
        click.option(
            "--config",
            "config_path",
            required=True,
            type=click.Path(exists=True, dir_okay=False),
            help="Path to the run's TOML config file (required).",
        ),
        click.option(
            "--output-dir", "output_dir", default=None, help="Override [output].dir (the run output directory)."
        ),
        click.option(
            "--set",
            "overrides",
            multiple=True,
            metavar="SECTION.KEY=VALUE",
            help="Override one TOML value (repeatable); VALUE uses TOML syntax, so quote strings: --set data.batch_size=64.",
        ),
        click.option(
            "--seed", type=int, default=None, help="Override the run seed (routed to the subcommand's section)."
        ),
        click.option(
            "--accelerator", default=None, help='Override the accelerator ("auto" | "cpu"), routed per subcommand.'
        ),
        click.option("--sample", type=int, default=None, help="Cap rows for every [datasets.*] (fast smoke runs)."),
    ]
    for option in reversed(options):
        func = option(func)
    return func


@click.group()
@click.version_option(package_name="foundation-model", prog_name="fm")
def main() -> None:
    """Unified foundation-model CLI (pretrain / finetune / inverse / predict)."""


def _pretrain_config(
    config_path: str,
    overrides: Sequence[str],
    output_dir: str | None,
    seed: int | None,
    accelerator: str | None,
    sample: int | None,
    max_epochs: int | None,
    checkpoint: str | None,
    resume: bool,
) -> PretrainConfig:
    raw = load_raw_config(config_path, overrides, seed=seed, accelerator=accelerator, sample=sample)
    if max_epochs is not None:
        _set_dotted(raw, "training.max_epochs", max_epochs)
    return build_pretrain_config(raw, output_dir=output_dir, checkpoint=checkpoint, resume=resume)


@main.command("pretrain")
@common_options
@click.option("--max-epochs", type=int, default=None, help="Override training.max_epochs.")
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(dir_okay=False),
    help="Warm-start from this checkpoint (continue the sequence).",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from the latest step checkpoint in the output dir (continue after a kill).",
)
def pretrain_cmd(
    config_path: str,
    output_dir: str | None,
    overrides: tuple[str, ...],
    seed: int | None,
    accelerator: str | None,
    sample: int | None,
    max_epochs: int | None,
    checkpoint: str | None,
    resume: bool,
) -> None:
    """Continual-rehearsal pre-training (multi-run sweep)."""
    cfg = _pretrain_config(
        config_path, overrides, output_dir, seed, accelerator, sample, max_epochs, checkpoint, resume
    )
    recorder = RunRecorder(cfg.output_dir)
    recorder.write_provenance(config=cfg, argv=list(sys.argv), seeds={"seed": cfg.training.seed})
    pretrain_run(cfg, recorder)
    recorder.close()


def _finetune_config(
    config_path: str,
    overrides: Sequence[str],
    output_dir: str | None,
    seed: int | None,
    accelerator: str | None,
    sample: int | None,
    checkpoint: str | None,
    tasks: str | None,
    epochs: int | None,
) -> FinetuneConfig:
    raw = load_raw_config(config_path, overrides, seed=seed, accelerator=accelerator, sample=sample)
    if tasks is not None:
        _set_dotted(raw, "finetune.tasks", [t.strip() for t in tasks.split(",") if t.strip()])
    if epochs is not None:
        _set_dotted(raw, "finetune.epochs", epochs)
    return build_finetune_config(raw, output_dir=output_dir, checkpoint=checkpoint)


@main.command("finetune")
@common_options
@click.option("--checkpoint", default=None, type=click.Path(dir_okay=False), help="Checkpoint to fine-tune from.")
@click.option("--tasks", default=None, help="Comma-separated heads to fine-tune (overrides finetune.tasks).")
@click.option("--epochs", type=int, default=None, help="Override finetune.epochs.")
def finetune_cmd(
    config_path: str,
    output_dir: str | None,
    overrides: tuple[str, ...],
    seed: int | None,
    accelerator: str | None,
    sample: int | None,
    checkpoint: str | None,
    tasks: str | None,
    epochs: int | None,
) -> None:
    """Frozen-encoder fine-tuning of selected task heads."""
    cfg = _finetune_config(config_path, overrides, output_dir, seed, accelerator, sample, checkpoint, tasks, epochs)
    recorder = RunRecorder(cfg.output_dir)
    recorder.write_provenance(config=cfg, argv=list(sys.argv), seeds={"seed": cfg.training.seed})
    finetune_run(cfg, recorder)
    recorder.close()


def _inverse_config(
    config_path: str,
    overrides: Sequence[str],
    output_dir: str | None,
    seed: int | None,
    accelerator: str | None,
    sample: int | None,
    checkpoint: str | None,
    steps: int | None,
    no_trajectory: bool,
    animation_formats: str | None,
) -> InverseConfig:
    raw = load_raw_config(
        config_path,
        overrides,
        seed=seed,
        accelerator=accelerator,
        sample=sample,
        seed_key="inverse.seed",
        accelerator_key="inverse.accelerator",
    )
    if steps is not None:
        _set_dotted(raw, "inverse.steps", steps)
    if no_trajectory:
        _set_dotted(raw, "inverse.record_trajectory", False)
    if animation_formats is not None:
        fmts = [f.strip() for f in animation_formats.split(",") if f.strip()]
        _set_dotted(raw, "inverse.animation_formats", fmts)
    return build_inverse_config(raw, output_dir=output_dir, checkpoint=checkpoint)


@main.command("inverse")
@common_options
@click.option("--checkpoint", default=None, type=click.Path(dir_okay=False), help="Checkpoint to inverse-design from.")
@click.option("--scenario", "scenarios", multiple=True, help="Run only the named scenario(s) (repeatable).")
@click.option("--steps", type=int, default=None, help="Override inverse.steps.")
@click.option("--no-trajectory", is_flag=True, default=False, help="Disable trajectory recording.")
@click.option("--animation-formats", default=None, help="Comma list of {gif,html,svg} (overrides config).")
def inverse_cmd(
    config_path: str,
    output_dir: str | None,
    overrides: tuple[str, ...],
    seed: int | None,
    accelerator: str | None,
    sample: int | None,
    checkpoint: str | None,
    scenarios: tuple[str, ...],
    steps: int | None,
    no_trajectory: bool,
    animation_formats: str | None,
) -> None:
    """Inverse design (scenarios × algorithm paths)."""
    cfg = _inverse_config(
        config_path,
        overrides,
        output_dir,
        seed,
        accelerator,
        sample,
        checkpoint,
        steps,
        no_trajectory,
        animation_formats,
    )
    recorder = RunRecorder(cfg.output_dir)
    recorder.write_provenance(config=cfg, argv=list(sys.argv), seeds={"seed": 2025})
    inverse_run(cfg, recorder, only_scenarios=list(scenarios) or None)
    recorder.close()


def _predict_config(
    config_path: str,
    overrides: Sequence[str],
    output_dir: str | None,
    seed: int | None,
    accelerator: str | None,
    sample: int | None,
    checkpoint: str | None,
    tasks: str | None,
    split: str | None,
    compositions: str | None,
    no_metrics: bool,
) -> PredictConfig:
    raw = load_raw_config(
        config_path,
        overrides,
        seed=seed,
        accelerator=accelerator,
        sample=sample,
        seed_key="predict.seed",
        accelerator_key="predict.accelerator",
    )
    if tasks is not None:
        _set_dotted(raw, "predict.tasks", [t.strip() for t in tasks.split(",") if t.strip()])
    if split is not None:
        _set_dotted(raw, "predict.split", split)
    if compositions is not None:
        _set_dotted(raw, "predict.compositions", [c.strip() for c in compositions.split(",") if c.strip()])
    if no_metrics:
        _set_dotted(raw, "predict.with_metrics", False)
    return build_predict_config(raw, output_dir=output_dir, checkpoint=checkpoint)


@main.command("predict")
@common_options
@click.option("--checkpoint", default=None, type=click.Path(dir_okay=False), help="Checkpoint to predict with.")
@click.option("--tasks", default=None, help="Comma-separated heads to predict (default: all checkpoint heads).")
@click.option("--split", default=None, help="train | val | test | all.")
@click.option("--compositions", default=None, help="Comma-separated compositions (overrides split).")
@click.option("--no-metrics", is_flag=True, default=False, help="Skip metric computation.")
def predict_cmd(
    config_path: str,
    output_dir: str | None,
    overrides: tuple[str, ...],
    seed: int | None,
    accelerator: str | None,
    sample: int | None,
    checkpoint: str | None,
    tasks: str | None,
    split: str | None,
    compositions: str | None,
    no_metrics: bool,
) -> None:
    """Evaluate / predict with an arbitrary checkpoint."""
    cfg = _predict_config(
        config_path,
        overrides,
        output_dir,
        seed,
        accelerator,
        sample,
        checkpoint,
        tasks,
        split,
        compositions,
        no_metrics,
    )
    recorder = RunRecorder(cfg.output_dir)
    recorder.write_provenance(config=cfg, argv=list(sys.argv), seeds={"seed": cfg.seed})
    predict_run(cfg, recorder)
    recorder.close()


if __name__ == "__main__":  # pragma: no cover
    main()
