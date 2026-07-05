# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Workflow engines behind the unified ``fm`` CLI.

Each ``workflows.<name>.run(cfg)`` takes a validated config dataclass and executes one
subcommand's flow. Only :mod:`foundation_model.workflows.recording` writes artifacts for the
training/predict flows; the other modules hand it dataframes/dicts/figures.
"""
