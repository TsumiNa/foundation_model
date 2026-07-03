# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`foundation_model.cli.main`."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from foundation_model.cli.main import (
    _finetune_config,
    _parse_toml_value,
    _pretrain_config,
    load_raw_config,
    main,
)

_FINETUNE_CONFIG = """
[descriptor]
kind = "kmd"

[datasets.d1]
path = "data/x.parquet"

[[tasks]]
name = "a"
kind = "regression"
dataset = "d1"
column = "a"

[[tasks]]
name = "b"
kind = "regression"
dataset = "d1"
column = "b"

[finetune]
tasks = ["a"]
checkpoint = "ck.pt"

[output]
dir = "out"
"""

_CONFIG = """
[descriptor]
kind = "kmd"

[datasets.d1]
path = "data/x.parquet"

[[tasks]]
name = "a"
kind = "regression"
dataset = "d1"
column = "a"

[training]
max_epochs = 10

[output]
dir = "out"
"""


def _write_config(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(_CONFIG)
    return str(path)


def test_set_override_applies_to_built_config(tmp_path) -> None:
    cfg = _pretrain_config(
        _write_config(tmp_path), ("training.max_epochs=3",), None, None, None, None, None, None, False
    )
    assert cfg.training.max_epochs == 3


def test_max_epochs_flag_overrides(tmp_path) -> None:
    cfg = _pretrain_config(_write_config(tmp_path), (), None, None, None, None, 5, None, False)
    assert cfg.training.max_epochs == 5


def test_seed_and_sample_flags_map_into_tree(tmp_path) -> None:
    raw = load_raw_config(_write_config(tmp_path), seed=99, sample=50)
    assert raw["training"]["seed"] == 99
    assert raw["datasets"]["d1"]["sample"] == 50


def test_output_dir_flag_overrides(tmp_path) -> None:
    cfg = _pretrain_config(_write_config(tmp_path), (), "/tmp/custom", None, None, None, None, None, False)
    assert str(cfg.output_dir) == "/tmp/custom"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("3", 3), ("1.5", 1.5), ('"foo"', "foo"), ("true", True), ("[1, 2]", [1, 2])],
)
def test_parse_toml_value(value: str, expected: object) -> None:
    assert _parse_toml_value(value) == expected


def test_set_without_equals_errors(tmp_path) -> None:
    with pytest.raises(click.BadParameter):
        load_raw_config(_write_config(tmp_path), ("training.max_epochs",))


def test_unknown_subcommand_exits_nonzero() -> None:
    result = CliRunner().invoke(main, ["frobnicate"])
    assert result.exit_code != 0


def test_missing_config_file_errors() -> None:
    result = CliRunner().invoke(main, ["pretrain", "--config", "/no/such/file.toml"])
    assert result.exit_code != 0
    assert "does not exist" in result.output.lower()


def test_help_lists_subcommands() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "pretrain" in result.output
    assert "finetune" in result.output


def test_finetune_tasks_and_checkpoint_flags(tmp_path) -> None:
    path = tmp_path / "ft.toml"
    path.write_text(_FINETUNE_CONFIG)
    cfg = _finetune_config(str(path), (), None, None, None, None, "override.pt", "a,b", 7)
    assert cfg.tasks == ["a", "b"]  # --tasks overrides finetune.tasks
    assert str(cfg.checkpoint) == "override.pt"  # --checkpoint wins
    assert cfg.epochs == 7


_PREDICT_CONFIG = """
[descriptor]
kind = "kmd"

[datasets.d1]
path = "data/x.parquet"

[[tasks]]
name = "a"
kind = "regression"
dataset = "d1"
column = "a"

[predict]
split = "test"

[output]
dir = "out"
"""


def test_predict_flags(tmp_path) -> None:
    from foundation_model.cli.main import _predict_config

    path = tmp_path / "pred.toml"
    path.write_text(_PREDICT_CONFIG)
    cfg = _predict_config(str(path), (), None, None, None, None, "ck.pt", "a", "all", "Fe2 O3,Al2 O3", True)
    assert cfg.tasks == ["a"]
    assert cfg.split == "all"
    assert cfg.compositions == ["Fe2 O3", "Al2 O3"]
    assert cfg.with_metrics is False  # --no-metrics


def test_predict_seed_and_accelerator_route_to_predict_section(tmp_path) -> None:
    # regression: --seed / --accelerator used to inject [training] and crash predict's builder.
    from foundation_model.cli.main import _predict_config

    path = tmp_path / "pred.toml"
    path.write_text(_PREDICT_CONFIG)
    cfg = _predict_config(str(path), (), None, 7, "cpu", None, "ck.pt", None, None, None, False)
    assert cfg.seed == 7 and cfg.accelerator == "cpu"
