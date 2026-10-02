"""Tests for paired initialization, configuration round trips and composition-weighted scoring."""

from copy import deepcopy
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest
import torch

from foundation_model.workflows._engine import build_empty_model, build_head_config
from foundation_model.workflows._sections import build_model_section, build_training_section
from foundation_model.workflows.pretrain import build_pretrain_config
from foundation_model.workflows.recording import load_checkpoint_state
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

from benchmark import (
    DATE,
    SOURCE_TASKS,
    composition_metrics,
    initialize_target,
    load_protocol,
    workflow_config,
    write_toml,
)

PROTOCOL = Path(__file__).parents[1] / "configs/protocol.toml"


@pytest.fixture
def data_dir(tmp_path):
    source = pd.DataFrame(
        {
            "composition": ["FeO", "SiO2", "Al2O3", "NaCl"],
            "split": ["train", "train", "val", "val"],
            "material_type": [0, 4, 4, 4],
        }
    )
    for name in SOURCE_TASKS[1:]:
        source[name] = [np.ones(2)] * 4 if name in {"seebeck", "zt"} else [0.0, 1.0, 0.0, 1.0]
    source["seebeck_t"] = [np.array([1.0, 2.0])] * 4
    source["zt_t"] = [np.array([1.0, 2.0])] * 4
    source.to_parquet(tmp_path / f"source_{DATE}.parquet", index=False)
    target = source[["composition", "split"]].copy()
    target["dielectric_total"] = [0.0, 1.0, 0.0, 1.0]
    target.to_parquet(tmp_path / "target.parquet", index=False)
    manifest = {
        "source": {"file": f"source_{DATE}.parquet"},
        "targets": [
            {"task": "dielectric_total", "seed": 0, "fraction": 1, "file": "target.parquet", "scaler": "unused.joblib"}
        ],
    }
    (tmp_path / f"manifest_{DATE}.json").write_text(json.dumps(manifest))
    return tmp_path


def test_registered_encoders_build_and_workflow_toml_roundtrips(data_dir, tmp_path):
    config, _, arms = load_protocol(PROTOCOL)
    assert len(arms) == 8 and config.source_counts == [1, 3, 7]
    for name in arms:
        raw = workflow_config(PROTOCOL, data_dir, name, 0)
        raw["pretrain"] = {"task_sequence": list(SOURCE_TASKS)}
        raw["output"] = {"dir": str(tmp_path / name)}
        path = tmp_path / f"{name}.toml"
        write_toml(raw, path)
        cfg = build_pretrain_config(tomllib.loads(path.read_text()))
        assert cfg.model.latent_dim == 384


def test_target_head_initialization_is_identical_for_scratch_and_different_source_counts(data_dir, tmp_path):
    raw = workflow_config(PROTOCOL, data_dir, "mlp_tuned", 0, target="dielectric_total")
    raw["model"] = {
        "latent_dim": 6,
        "encoder_hidden_dims": [8],
        "head_hidden_dims": [8],
        "autoencoder_hidden_dims": [8],
    }
    states = []
    for count in (0, 1, 3):
        source_path = None
        if count:
            catalog = TaskCatalog(
                build_task_catalog_config(
                    {key: deepcopy(raw[key]) for key in ("data", "descriptor", "datasets", "tasks")}
                )
            )
            model_cfg, training_cfg = build_model_section(raw["model"]), build_training_section(raw["training"])
            model = build_empty_model(catalog, model_cfg, training_cfg)
            for name in SOURCE_TASKS[:count]:
                model.add_task(build_head_config(catalog, model_cfg, training_cfg, name))
            source_path = tmp_path / f"source{count}.pt"
            torch.save({"model": model.state_dict(), "task_sequence": list(SOURCE_TASKS[:count])}, source_path)
        path = tmp_path / f"init{count}" / "initializer.pt"
        info = initialize_target(raw, "dielectric_total", path, source=source_path)
        assert info["source_task_count"] == count
        state = load_checkpoint_state(path)["model"]
        states.append({key: value for key, value in state.items() if key.startswith("task_heads.dielectric_total.")})
        if source_path:
            original = load_checkpoint_state(source_path)["model"]
            for key in original:
                if key.startswith("encoder."):
                    torch.testing.assert_close(original[key], state[key])
    assert states[0]
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]) and torch.equal(states[0][key], states[2][key])


def test_rmse_weights_compositions_equally_instead_of_curve_sampling_density():
    frame = pd.DataFrame({"composition": ["A"] * 3 + ["B"], "true": [0.0] * 4, "pred": [2.0, 2.0, 2.0, 4.0]})
    metric = composition_metrics(frame, 2)
    assert metric["rmse"] == pytest.approx(np.sqrt(10))
    assert metric["standardized_rmse"] == pytest.approx(np.sqrt(10) / 2)
    assert metric["mae"] == 3 and metric["compositions"] == 2


def test_invalid_predictions_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match="finite"):
        composition_metrics(pd.DataFrame({"composition": ["A"], "true": [1.0], "pred": [np.nan]}), 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seeds", [True]),
        ("source_counts", [1.5]),
        ("fractions", [True]),
        ("fractions", [0.101, 0.102]),
        ("max_epochs", 0),
    ],
)
def test_protocol_rejects_invalid_identifiers_and_bounds(tmp_path, field, value):
    from benchmark import BenchmarkConfig

    raw = tomllib.loads(PROTOCOL.read_text())["benchmark"]
    raw[field] = value
    with pytest.raises(ValueError):
        BenchmarkConfig(**raw)


def test_nested_inverse_array_tables_roundtrip(tmp_path):
    raw = {
        "inverse": {
            "scenarios": [
                {"name": "one", "targets": [{"task": "a", "direction": "high"}, {"task": "b", "direction": "low"}]},
                {"name": "two", "targets": [{"task": "a", "direction": "low"}]},
            ],
            "paths": [{"name": "latent", "method": "latent"}, {"name": "composition", "method": "composition"}],
        }
    }
    path = tmp_path / "inverse.toml"
    write_toml(raw, path)
    assert tomllib.loads(path.read_text()) == raw


def test_functional_batch_cap_does_not_modify_scientific_protocol(data_dir):
    path = data_dir / f"manifest_{DATE}.json"
    manifest = json.loads(path.read_text())
    manifest["smoke_batch_size"] = 16
    path.write_text(json.dumps(manifest))
    assert workflow_config(PROTOCOL, data_dir, "mlp_tuned", 0)["data"]["batch_size"] == 128
    manifest["functional_smoke"] = True
    path.write_text(json.dumps(manifest))
    assert workflow_config(PROTOCOL, data_dir, "mlp_tuned", 0)["data"]["batch_size"] == 16
