# Copyright 2025 TsumiNa.
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
) -> dict[str, Any]:
    """Load a TOML file and apply ``--set`` overrides plus common first-class flags."""

    with open(config_path, "rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)

    for override in overrides:
        if "=" not in override:
            raise click.BadParameter(f"--set expects SECTION.KEY=VALUE, got {override!r}")
        key, _, value = override.partition("=")
        _set_dotted(raw, key.strip(), _parse_toml_value(value.strip()))

    if seed is not None:
        _set_dotted(raw, "training.seed", seed)
    if accelerator is not None:
        _set_dotted(raw, "training.accelerator", accelerator)
    if sample is not None:
        for name in raw.get("datasets", {}):
            _set_dotted(raw, f"datasets.{name}.sample", sample)
    return raw


def common_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Attach the options shared by every subcommand."""
    options = [
        click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False)),
        click.option("--output-dir", "output_dir", default=None, help="Override the run output directory."),
        click.option("--set", "overrides", multiple=True, metavar="SECTION.KEY=VALUE", help="Override a TOML value."),
        click.option("--seed", type=int, default=None, help="Override training.seed."),
        click.option("--accelerator", default=None, help="Override training.accelerator."),
        click.option("--sample", type=int, default=None, help="Cap rows for every dataset (smoke runs)."),
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
) -> PretrainConfig:
    raw = load_raw_config(config_path, overrides, seed=seed, accelerator=accelerator, sample=sample)
    if max_epochs is not None:
        _set_dotted(raw, "training.max_epochs", max_epochs)
    return build_pretrain_config(raw, output_dir=output_dir)


@main.command("pretrain")
@common_options
@click.option("--max-epochs", type=int, default=None, help="Override training.max_epochs.")
def pretrain_cmd(
    config_path: str,
    output_dir: str | None,
    overrides: tuple[str, ...],
    seed: int | None,
    accelerator: str | None,
    sample: int | None,
    max_epochs: int | None,
) -> None:
    """Continual-rehearsal pre-training (multi-run sweep)."""
    cfg = _pretrain_config(config_path, overrides, output_dir, seed, accelerator, sample, max_epochs)
    recorder = RunRecorder(cfg.output_dir)
    recorder.write_provenance(config=cfg, argv=list(sys.argv), seeds={"seed": cfg.training.seed})
    pretrain_run(cfg, recorder)
    recorder.close()


if __name__ == "__main__":  # pragma: no cover
    main()
