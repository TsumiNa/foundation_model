# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Workflow engines behind the unified ``yammi`` CLI.

Each ``workflows.<name>.run(cfg)`` takes a validated config dataclass and executes one
subcommand's flow. Only :mod:`yammi.workflows.recording` writes artifacts for the
training/predict flows; the other modules hand it dataframes/dicts/figures.
"""
