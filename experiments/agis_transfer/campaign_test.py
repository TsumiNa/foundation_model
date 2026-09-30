# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import pytest
import torch

import campaign

from campaign import (
    CampaignSettings,
    Route,
    RunUnit,
    build_units,
    final_config,
    execute_unit,
    file_sha256,
    fit_spec,
    initialize_target,
    plan_campaign,
    source_fingerprint,
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
        json.dumps(
            {
                "raw": raw,
                "mode": "finetune",
                "source": None,
                "output": str(output),
                "heldout": "La3 Ni2 O7",
                "source_sha256": source_fingerprint(),
            }
        )
    )
    fit_spec(spec)
    assert (output / "DONE").is_file()
    summary = json.loads((output / "training/finetune_summary.json").read_text())
    assert summary["epochs_run"] == 2 and summary["freeze_encoder"] is False


def test_plan_records_the_consumed_input_artifacts(manifest: dict, tmp_path: Path) -> None:
    selection = tmp_path / "data/agis_pretrained_20261001/selection_20261001.json"
    selection.parent.mkdir()
    selection.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "sha256": str(i),
                        "hyperparameters": {"latent_dim": 8, "encoder_hidden_dims": [16], "head_hidden_dims": [8]},
                    }
                    for i in range(10)
                ]
            }
        )
    )
    folder = tmp_path / "data/agis_preprocessing_20261001/fold_01"
    scaler = folder / "scalers_20261001.joblib"
    scaler.write_bytes(b"serialized scaler")
    tasks = folder / "tasks_20261001.toml"
    with tasks.open("a") as stream:
        stream.write(f'\n[tasks.scaler]\npath = "{scaler}"\n')
    curves = tmp_path / "data/curves.parquet"
    curves.write_bytes(b"audit curves")
    preprocessing = folder.parent / "manifest_20261001.json"
    preprocessing.write_text(
        json.dumps(
            {"n_compounds": 8, "n_curves": 24, "curves_path": str(curves), "folds": [{"directory": str(folder)}] * 8}
        )
    )
    base = tmp_path / "base.toml"
    base.write_text("[datasets]\n[model]\n[pretrain.replay]\n")
    path = plan_campaign(base, tmp_path / "plan", CampaignSettings())
    recorded = json.loads(path.read_text())["input_sha256"]
    assert str(tasks) in recorded and str(scaler) in recorded and str(curves) in recorded
    assert all(str(folder / f"p{p}.parquet") in recorded for p in (0, 10, 20))
    assert all(file_sha256(Path(filename)) == digest for filename, digest in recorded.items())


@pytest.fixture
def completed_unit(manifest: dict, tmp_path: Path) -> tuple[Path, Path, Path]:
    unit = RunUnit(route=Route.SCRATCH, fold=1, checkpoint_index=None, pressure=0)
    input_path = tmp_path / "input.parquet"
    input_path.write_bytes(b"original input")
    manifest.update(
        units=[vars(unit)],
        input_sha256={str(input_path): file_sha256(input_path)},
        preprocessing={"folds": [{"heldout_composition": "La3 Ni2 O7"}]},
        source_sha256=source_fingerprint(),
    )
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(manifest))
    root = tmp_path / "outputs" / unit.name
    root.mkdir(parents=True)
    (root / "DONE").write_text("completed")
    (root / "campaign_identity.json").write_text(
        json.dumps({"manifest_sha256": file_sha256(path), "source_sha256": source_fingerprint()})
    )
    return path, root, input_path


def test_completed_unit_rejects_input_drift(completed_unit: tuple[Path, Path, Path]) -> None:
    path, root, input_path = completed_unit
    input_path.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="input hash mismatch"):
        execute_unit(path, 0, root.parent)


def test_completed_unit_rejects_a_different_manifest(completed_unit: tuple[Path, Path, Path]) -> None:
    path, root, _ = completed_unit
    value = json.loads(path.read_text())
    value["settings"]["final_epochs"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="different campaign manifest"):
        execute_unit(path, 0, root.parent)


def test_unbound_outputs_cannot_be_reused(completed_unit: tuple[Path, Path, Path]) -> None:
    path, root, _ = completed_unit
    (root / "campaign_identity.json").unlink()
    with pytest.raises(ValueError, match="no campaign identity"):
        execute_unit(path, 0, root.parent)


def test_matching_completed_unit_revalidates_final_outputs(
    completed_unit: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, root, _ = completed_unit
    seen = []
    monkeypatch.setattr(campaign, "verify_final", lambda *args: seen.append(args))
    execute_unit(path, 0, root.parent)
    assert seen == [(root / "unfrozen", "agis_rho_0gpa", "La3 Ni2 O7", 1000)]


def test_worker_requires_an_array_submission() -> None:
    env = dict(os.environ)
    env.pop("SLURM_ARRAY_TASK_ID", None)
    result = subprocess.run(
        ["bash", str(Path(__file__).with_name("array.sbatch"))], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Submit this worker with sbatch --array" in result.stderr


def test_completed_unit_rejects_source_drift(
    completed_unit: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, root, _ = completed_unit
    monkeypatch.setattr(campaign, "source_fingerprint", lambda: "changed implementation")
    with pytest.raises(ValueError, match="planned implementation"):
        execute_unit(path, 0, root.parent)


def test_fit_rejects_source_changes_between_parent_and_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "fit.json"
    path.write_text(json.dumps({"source_sha256": source_fingerprint()}))
    monkeypatch.setattr(campaign, "source_fingerprint", lambda: "changed implementation")
    with pytest.raises(ValueError, match="source changed"):
        fit_spec(path)
