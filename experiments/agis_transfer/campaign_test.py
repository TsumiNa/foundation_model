# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import pytest
import torch

from campaign import (
    CampaignSettings,
    Route,
    RunUnit,
    build_units,
    final_config,
    fit_spec,
    initialize_target,
    warm_config,
)


@pytest.fixture
def manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "data/agis_preprocessing_20261001/fold_01"
    folder.mkdir(parents=True)
    sections = []
    compositions = [
        "La3Ni2O7",
        "La2.9Sr0.1Ni2O7",
        "La2NdNi2O7",
        "La1.9NdSr0.1Ni2O7",
        "La1.8NdSr0.2Ni2O7",
        "La2EuNi2O7",
        "La1.9EuSr0.1Ni2O7",
        "La1.8EuSr0.2Ni2O7",
    ]
    for pressure in (0, 10, 20):
        path = folder / f"p{pressure}.parquet"
        pd.DataFrame(
            {
                "composition": compositions,
                "y": [np.linspace(i, i + 1, 300) for i in range(8)],
                "t": [np.linspace(6, 290, 300)] * 8,
                "split": ["test"] + ["train"] * 7,
            }
        ).to_parquet(path)
        sections.append(f'[datasets.p{pressure}]\npath = "{path}"\n')
        sections.append(
            f'[[tasks]]\nname = "agis_rho_{pressure}gpa"\nkind = "kernel_regression"\n'
            f'dataset = "p{pressure}"\ncolumn = "y"\nt_column = "t"\n'
        )
    (folder / "tasks_20261001.toml").write_text("\n".join(sections))
    return {
        "settings": vars(CampaignSettings()),
        "base_config": {
            "data": {"batch_size": 256},
            "descriptor": {"kind": "kmd", "n_grids": 4},
            "datasets": {},
            "tasks": [],
            "model": {"latent_dim": 8, "encoder_hidden_dims": [16], "n_kernel": 4},
            "training": {},
            "pretrain": {"replay": {"amount": 0.3, "interval": 1, "resample": "epoch"}},
        },
    }


def test_exact_experiment_counts_and_scratch_has_no_repeats() -> None:
    counts = Counter(unit.route for unit in build_units())
    assert counts == {Route.DIRECT: 240, Route.WARM: 240, Route.SCRATCH: 24}
    assert counts[Route.DIRECT] * 2 + counts[Route.WARM] * 2 + counts[Route.SCRATCH] == 984
    assert len({unit.name for unit in build_units()}) == 504
    assert all(unit.checkpoint_index is None for unit in build_units() if unit.route == Route.SCRATCH)
    assert (
        sum(
            unit.route == Route.WARM and unit.checkpoint_index is not None and unit.checkpoint_index < 5
            for unit in build_units()
        )
        == 120
    )


@pytest.mark.parametrize("pressure", [0, 10, 20])
def test_warm_start_excludes_target_and_replays_all_seven_other_pressure_samples(manifest: dict, pressure: int) -> None:
    unit = RunUnit(route=Route.WARM, fold=1, checkpoint_index=0, pressure=pressure)
    raw = warm_config(manifest, unit)
    assert raw["pretrain"]["task_sequence"] == [f"agis_rho_{p}gpa" for p in (0, 10, 20) if p != pressure]
    assert raw["pretrain"]["replay"]["per_task"] == {f"agis_rho_{p}gpa": 7 for p in (0, 10, 20)}
    assert raw["data"]["batch_size"] == 256
    assert raw["training"]["early_stopping"]["monitor"] == "val_final_loss"
    final = final_config(manifest, unit)
    assert final["data"]["batch_size"] == 7
    assert final["finetune"]["epochs"] == 1000
    assert final["training"]["early_stopping"]["enabled"] is False
    assert final["training"]["scheduler"]["monitor"] == "train_final_loss_epoch"


def test_head_initialization_is_identical_across_sources_and_scratch(manifest: dict, tmp_path: Path) -> None:
    unit = RunUnit(route=Route.DIRECT, fold=1, checkpoint_index=0, pressure=0)
    raw = final_config(manifest, unit)
    initial_states = []
    for i, task_count in enumerate((24, 26)):
        source = tmp_path / f"source{i}.pt"
        encoder = torch.full((2, 2), float(i))
        torch.save(
            {"model": {"encoder.custom_weight": encoder}, "task_sequence": [f"old{n}" for n in range(task_count)]},
            source,
        )
        path = initialize_target(raw, source, tmp_path / f"initialized{i}.pt")
        state = torch.load(path, weights_only=False)
        assert torch.equal(state["model"]["encoder.custom_weight"], encoder)
        assert state["task_sequence"][-1] == "agis_rho_0gpa"
        initial_states.append({k: v for k, v in state["model"].items() if k.startswith("task_heads.agis_")})
    scratch = final_config(manifest, RunUnit(route=Route.SCRATCH, fold=1, checkpoint_index=None, pressure=0))
    path = initialize_target(scratch, None, tmp_path / "scratch.pt")
    state = torch.load(path, weights_only=False)
    assert state["task_sequence"] == ["agis_rho_0gpa"]
    initial_states.append({k: v for k, v in state["model"].items() if k.startswith("task_heads.agis_")})
    for other in initial_states[1:]:
        assert initial_states[0].keys() == other.keys()
        for key in other:
            assert torch.equal(initial_states[0][key], other[key]), key


def test_already_learned_target_is_rejected(manifest: dict, tmp_path: Path) -> None:
    raw = final_config(manifest, RunUnit(route=Route.DIRECT, fold=1, checkpoint_index=0, pressure=0))
    source = tmp_path / "source.pt"
    torch.save({"model": {}, "task_sequence": ["agis_rho_0gpa"]}, source)
    with pytest.raises(ValueError, match="target pressure must be absent"):
        initialize_target(raw, source, tmp_path / "initial.pt")


@pytest.mark.parametrize("kwargs", [{"fold": 0}, {"pressure": 5}, {"checkpoint_index": None}])
def test_invalid_transfer_unit(kwargs: dict) -> None:
    values: dict[str, Any] = {"route": Route.DIRECT, "fold": 1, "pressure": 0, "checkpoint_index": 0}
    values.update(kwargs)
    with pytest.raises(ValueError):
        RunUnit(**values)


def test_scratch_cannot_have_checkpoint_repeat() -> None:
    with pytest.raises(ValueError, match="no pretrained checkpoint"):
        RunUnit(route=Route.SCRATCH, fold=1, checkpoint_index=0, pressure=0)


def test_scratch_fit_runs_with_seven_train_compounds_and_no_validation(manifest: dict, tmp_path: Path) -> None:
    raw = final_config(manifest, RunUnit(route=Route.SCRATCH, fold=1, checkpoint_index=None, pressure=0))
    raw["training"].update(accelerator="cpu", max_epochs=2)
    raw["finetune"]["epochs"] = 2
    output = tmp_path / "fit"
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"raw": raw, "mode": "finetune", "source": None, "output": str(output), "heldout": "La3 Ni2 O7"})
    )
    fit_spec(spec)
    assert (output / "DONE").is_file()
    summary = json.loads((output / "training/finetune_summary.json").read_text())
    assert summary["epochs_run"] == 2 and summary["freeze_encoder"] is False
