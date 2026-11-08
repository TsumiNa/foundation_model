# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``yammi`` — the unified yammi CLI.

This layer is intentionally THIN: each subcommand loads a TOML file, applies ``--set`` /
first-class-flag overrides onto the raw tree, builds the workflow config dataclass, writes
provenance, and calls exactly one ``workflows.<mod>.run(cfg)``. No business logic lives here.
"""

from __future__ import annotations

import functools
import json
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import click

from yammi.workflows.finetune import FinetuneConfig, build_finetune_config
from yammi.workflows.finetune import run as finetune_run
from yammi.workflows.inverse import InverseConfig, build_inverse_config
from yammi.workflows.inverse import run as inverse_run
from yammi.workflows.predict import PredictConfig, build_predict_config
from yammi.workflows.predict import run as predict_run
from yammi.workflows.pretrain import PretrainConfig, build_pretrain_config
from yammi.workflows.pretrain import run as pretrain_run
from yammi.workflows.recording import RunRecorder


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


def common_options(func: Callable[..., Any], *, config_required: bool = True) -> Callable[..., Any]:
    """Attach the options shared by every subcommand."""
    options = [
        click.option(
            "--config",
            "config_path",
            required=config_required,
            type=click.Path(exists=True, dir_okay=False),
            help="Path to the run's TOML config file" + (" (required)." if config_required else "."),
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
@click.version_option(package_name="yammi", prog_name="yammi")
def main() -> None:
    """Unified yammi CLI (pretrain / finetune / inverse / predict / export / publish)."""


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
    """Replay-based continual pre-training (multi-run sweep)."""
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
    pretrained: str | None = None,
) -> FinetuneConfig:
    raw = load_raw_config(config_path, overrides, seed=seed, accelerator=accelerator, sample=sample)
    if tasks is not None:
        _set_dotted(raw, "finetune.tasks", [t.strip() for t in tasks.split(",") if t.strip()])
    if epochs is not None:
        _set_dotted(raw, "finetune.epochs", epochs)
    return build_finetune_config(raw, output_dir=output_dir, checkpoint=checkpoint, pretrained=pretrained)


@main.command("finetune")
@common_options
@click.option("--checkpoint", default=None, type=click.Path(dir_okay=False), help="Checkpoint to fine-tune from.")
@click.option(
    "--pretrained",
    default=None,
    help="Released bundle to fine-tune from (directory or owner/name[@rev][#member]); the TOML lists only your tasks.",
)
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
    pretrained: str | None,
    tasks: str | None,
    epochs: int | None,
) -> None:
    """Frozen-encoder fine-tuning of selected task heads (from a checkpoint or a released bundle)."""
    cfg = _finetune_config(
        config_path, overrides, output_dir, seed, accelerator, sample, checkpoint, tasks, epochs, pretrained
    )
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
    pretrained: str | None = None,
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
    return build_inverse_config(raw, output_dir=output_dir, checkpoint=checkpoint, pretrained=pretrained)


@main.command("inverse")
@common_options
@click.option("--checkpoint", default=None, type=click.Path(dir_okay=False), help="Checkpoint to inverse-design from.")
@click.option(
    "--pretrained",
    default=None,
    help="Released bundle (directory or owner/name[@rev][#member]); targets in physical units, no catalog or data.",
)
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
    pretrained: str | None,
    scenarios: tuple[str, ...],
    steps: int | None,
    no_trajectory: bool,
    animation_formats: str | None,
) -> None:
    """Inverse design (scenarios × algorithm paths) from a checkpoint or a released bundle."""
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
        pretrained,
    )
    recorder = RunRecorder(cfg.output_dir)
    recorder.write_provenance(config=cfg, argv=list(sys.argv), seeds={"seed": cfg.seed})
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
@functools.partial(common_options, config_required=False)
@click.option("--checkpoint", default=None, type=click.Path(dir_okay=False), help="Checkpoint to predict with.")
@click.option(
    "--pretrained",
    default=None,
    help="Released bundle: a local directory or owner/name[@revision][#member]. Needs no --config or data.",
)
@click.option(
    "--compositions-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="With --pretrained: a text file of compositions, one per line.",
)
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
    pretrained: str | None,
    compositions_file: str | None,
) -> None:
    """Evaluate / predict with a checkpoint (catalog path) or a released bundle (--pretrained)."""
    if pretrained is not None:
        ignored = [
            flag
            for flag, value in (
                ("--set", overrides),
                ("--seed", seed),
                ("--accelerator", accelerator),
                ("--sample", sample),
            )
            if value
        ]
        if ignored:
            raise click.UsageError(f"{', '.join(ignored)} do not apply to --pretrained (no config or data is read).")
        _predict_pretrained(
            pretrained,
            config_path=config_path,
            checkpoint=checkpoint,
            split=split,
            output_dir=output_dir,
            tasks=tasks,
            compositions=compositions,
            compositions_file=compositions_file,
        )
        return
    if config_path is None:
        raise click.UsageError("--config is required unless --pretrained is given.")
    if compositions_file is not None:
        raise click.UsageError("--compositions-file applies to --pretrained only; use --compositions.")
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


def _predict_pretrained(
    source: str,
    *,
    config_path: str | None,
    checkpoint: str | None,
    split: str | None,
    output_dir: str | None,
    tasks: str | None,
    compositions: str | None,
    compositions_file: str | None,
) -> None:
    """``yammi predict --pretrained``: physical-unit predictions from a bundle, no catalog or data."""
    from yammi.hub import load_pretrained

    for flag, value in (("--config", config_path), ("--checkpoint", checkpoint), ("--split", split)):
        if value is not None:
            raise click.UsageError(f"{flag} does not apply to --pretrained (the bundle carries the model).")
    if output_dir is None:
        raise click.UsageError("--pretrained needs --output-dir.")
    comps = [c.strip() for c in (compositions or "").split(",") if c.strip()]
    if compositions_file is not None:
        comps += [ln.strip() for ln in Path(compositions_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not comps:
        raise click.UsageError("--pretrained needs --compositions and/or --compositions-file.")

    from yammi.hub import load_suite
    from yammi.hub.pretrained import parse_source

    # No "#member": combine every member (median of in-range members, plus a :std column). The suite's
    # spread is an uncertainty, and the combination was more accurate than one member on most heads.
    suite = load_suite(source) if parse_source(source).member is None else None
    combine = suite is not None and len(suite) > 1
    first = suite[suite.member_ids[0]] if suite is not None and combine else load_pretrained(source)
    model: Any = suite if combine else first
    wanted = [t.strip() for t in tasks.split(",")] if tasks else first.tasks
    heads = {h.name: h for h in first.heads}
    unknown = [t for t in wanted if t not in heads]
    if unknown:
        raise click.UsageError(f"unknown task(s) {unknown}; the bundle has {first.tasks}.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scalar = [t for t in wanted if heads[t].kind != "kernel_regression"]
    written: list[str] = []
    if scalar:
        model.predict(comps, scalar, probabilities=True).to_csv(out / "predictions.csv", index=False)
        written.append("predictions.csv")
    for task in (t for t in wanted if heads[t].kind == "kernel_regression"):
        model.predict_curve(comps, task).to_csv(out / f"{task}_curve.csv", index=False)
        written.append(f"{task}_curve.csv")
    members = suite.member_ids if suite is not None and combine else [first.member.id]
    info = {
        "source": source,
        "members": members,
        "combined": "median of the members whose output stays in range; <task>:std is their spread"
        if combine
        else None,
        "license": first.license,
        "fm_version": first.config.fm_version,
        "units": {t: first.units()[t] for t in wanted},
        "files": written,
    }
    (out / "prediction_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    who = f"{len(members)} members combined" if combine else f"member {members[0]}"
    click.echo(f"wrote {', '.join(written)} to {out} ({who}, licence {first.license})")


_EXPORT_CATALOG_KEYS = ("data", "descriptor", "datasets", "tasks")


@main.command("export")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="TOML with the task catalog and [model] the runs were trained with (other sections are ignored).",
)
@click.option(
    "--run",
    "runs",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Run directory (training/final_model.pt inside) or checkpoint file; repeat for a suite, one member each.",
)
@click.option(
    "--out", "out", required=True, type=click.Path(file_okay=False), help="Bundle directory (must be empty or new)."
)
@click.option(
    "--license",
    "license_id",
    required=True,
    help="Licence identifier of the weights, e.g. cc-by-nc-sa-4.0. No default: a model trained from scratch is yours.",
)
@click.option("--member-id", "member_ids", multiple=True, help="Member ids in --run order (default m000, m001, …).")
@click.option("--set", "overrides", multiple=True, metavar="KEY=VALUE", help="Override a config value (TOML).")
def export_cmd(
    config_path: str,
    runs: tuple[str, ...],
    out: str,
    license_id: str,
    member_ids: tuple[str, ...],
    overrides: tuple[str, ...],
) -> None:
    """Export training runs as a release bundle, verified against the source checkpoints."""
    from yammi.hub.export import export_runs
    from yammi.workflows._sections import build_model_section
    from yammi.workflows.task_catalog import build_task_catalog_config

    raw = load_raw_config(config_path, overrides)
    catalog = build_task_catalog_config({k: raw[k] for k in _EXPORT_CATALOG_KEYS if k in raw})
    model = build_model_section(raw.get("model", {}))
    path = export_runs(
        catalog,
        model,
        list(runs),
        out,
        license=license_id,
        member_ids=list(member_ids) or None,
        provenance={"export_config": str(config_path), "argv": list(sys.argv)},
    )
    click.echo(f"wrote bundle {path}")


@main.command("publish")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False))
@click.option("--repo", "repo_id", required=True, help="Hugging Face repository, owner/name.")
@click.option("--tag", default=None, help="Release tag, e.g. v1.0.0 (recommended: pins the release).")
@click.option(
    "--visibility",
    type=click.Choice(["private", "gated", "public"]),
    default="private",
    show_default=True,
    help="private: only you / your organisation; gated: public page, downloads after manual approval; public.",
)
@click.option("--title", default=None, help="Model card title (default: the repository name).")
@click.option(
    "--card-extra",
    "card_extra",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Markdown file appended to the model card (repeatable), e.g. data citations.",
)
@click.option("--overwrite-card", is_flag=True, default=False, help="Regenerate README.md even if the bundle has one.")
@click.option("--dry-run", is_flag=True, default=False, help="Check the bundle and write the card; upload nothing.")
def publish_cmd(
    bundle: str,
    repo_id: str,
    tag: str | None,
    visibility: str,
    title: str | None,
    card_extra: tuple[str, ...],
    overwrite_card: bool,
    dry_run: bool,
) -> None:
    """Write the model card and push a bundle to the Hugging Face Hub (token from HF_TOKEN / hf auth login)."""
    from yammi.hub.card import read_extra
    from yammi.hub.publish import publish

    result = publish(
        bundle,
        repo_id,
        tag=tag,
        visibility=visibility,  # type: ignore[arg-type]
        title=title,
        card_extra=read_extra(card_extra),
        overwrite_card=overwrite_card,
        dry_run=dry_run,
    )
    click.echo(f"card written to {result} (dry run)" if dry_run else f"published to {result}")


if __name__ == "__main__":  # pragma: no cover
    main()
