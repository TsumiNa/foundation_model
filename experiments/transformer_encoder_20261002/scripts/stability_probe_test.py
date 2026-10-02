from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from foundation_model.models.model_config import TransformerEncoderConfig

import stability_probe
from stability_probe import checkpoint_diagnostics, diagnostic_compositions, probe_cases, representation_statistics


ROOT = Path(__file__).parents[1]


def test_generated_lightning_checkpoint_with_config_and_disabled_head(tmp_path, monkeypatch):
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(3, 2)
    model.task_heads = torch.nn.ModuleDict({"a": torch.nn.BatchNorm1d(2)})
    state = {k.replace("task_heads.", "disabled_task_heads."): v for k, v in model.state_dict().items()}
    checkpoint = tmp_path / "best.ckpt"
    torch.save(
        {"state_dict": state, "hyper_parameters": {"encoder": TransformerEncoderConfig(input_dim=3)}}, checkpoint
    )
    monkeypatch.setattr(stability_probe, "build_model_for_checkpoint", lambda *args: model)
    result = checkpoint_diagnostics(checkpoint, None, {"model": {}}, ["a"], torch.randn(4, 3))
    assert result["batch_norm"][0]["module"] == "task_heads.a"
    assert result["batch_norm"][0]["running_var_median"] == 1
    assert len(result["checkpoint_sha256"]) == 64


def test_registered_probe_pairs_rates_seeds_and_control():
    config, cases = probe_cases(ROOT / "configs/protocol.toml", ROOT / "configs/stability.toml")
    assert config.source_count == 3 and config.max_epochs == 12
    assert len(cases) == 27
    for arm in config.arms:
        for lr in config.encoder_learning_rates:
            assert [c["seed"] for c in cases if c["arm"] == arm and c["encoder_lr"] == lr] == [0, 1, 2]
    assert all(c["control"] and c["arm"] == "grouped_cls" for c in cases[-3:])
    # Three adjacent seeds share an architecture/rate, allowing calibrated packing.
    for offset in range(0, len(cases), 3):
        assert len({(c["arm"], c["encoder_lr"]) for c in cases[offset : offset + 3]}) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"encoder_learning_rates": [float("nan")]},
        {"encoder_learning_rates": [True]},
        {"encoder_learning_rates": [0]},
        {"encoder_learning_rates": [1e-4, 1e-4]},
        {"seeds": [0, 0]},
        {"seeds": [True]},
        {"source_count": 8},
        {"max_epochs": 0},
        {"diagnostic_per_class": False},
        {"control_arm": "legacy_cls"},
    ],
)
def test_invalid_probe_configuration(changes):
    config, _ = probe_cases(ROOT / "configs/protocol.toml", ROOT / "configs/stability.toml")
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_diagnostics_are_fixed_stratified_validation_only():
    frame = pd.DataFrame(
        {
            "composition": [f"x{i}" for i in range(12)],
            "split": ["train"] * 3 + ["val"] * 6 + ["test"] * 3,
            "material_type": [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    selected = diagnostic_compositions(frame, 2)
    assert selected == diagnostic_compositions(frame, 2)
    sample = frame.set_index("composition").loc[selected]
    assert sample["split"].eq("val").all()
    assert sample["material_type"].value_counts().to_dict() == {0: 2, 1: 2}
    with pytest.raises(ValueError, match="validation"):
        diagnostic_compositions(frame.loc[frame.split.eq("test")], 2)


def test_representation_probe_detects_saturation_and_collapse():
    bad = representation_statistics(torch.full((32, 8), 20.0))
    assert bad["saturated_fraction"] == 1
    assert bad["low_variance_dimension_fraction"] == 1
    healthy = representation_statistics(torch.linspace(-1, 1, 32).unsqueeze(1).expand(-1, 8))
    assert healthy["saturated_fraction"] == 0
    assert healthy["low_variance_dimension_fraction"] == 0
    for invalid in [torch.ones(1, 8), torch.tensor([[0.0], [float("inf")]])]:
        with pytest.raises(ValueError):
            representation_statistics(invalid)
