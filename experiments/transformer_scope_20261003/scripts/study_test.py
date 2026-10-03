from pathlib import Path
import tomllib
import numpy as np
import pandas as pd
import pytest
import torch

from study import StudyConfig, cases, normalization, shuffle_labels, make_encoder, fit_target

CONFIG = Path(__file__).parents[1] / "configs/study.toml"


def test_matrix_and_disjoint_seeds():
    raw = tomllib.loads(CONFIG.read_text())
    cfg = StudyConfig(**raw["study"])
    assert len(cases(raw, "pilot")) == 48
    formal = cases(raw, "formal", {a: v["learning_rates"][0] for a, v in raw["arms"].items()})
    assert (
        len(formal) == 1026
        and len({(c["arm"], c["split_seed"], c["seed"], c["condition"], c["steps"]) for c in formal}) == 1026
    )
    with pytest.raises(ValueError):
        StudyConfig(**{**raw["study"], "seeds": cfg.pilot_seeds})
    with pytest.raises(ValueError):
        cases(raw, "formal", {})


def test_train_only_normalization_and_shuffled_masks():
    y = np.array([1.0, 3.0, 1000.0, np.nan])
    normalized, mean, scale = normalization(y, np.array([True, True, False, False]))
    assert mean == 2 and scale == 1 and normalized[2] == 998
    matrix = np.arange(60, dtype=float).reshape(20, 3)
    matrix[0, 1] = np.nan
    splits = np.array(["train"] * 10 + ["val"] * 5 + ["test"] * 5)
    shuffled = shuffle_labels(matrix, splits, 4)
    assert np.array_equal(np.isnan(shuffled), np.isnan(matrix))
    assert np.array_equal(shuffled[15:], matrix[15:])
    for split in ["train", "val"]:
        assert np.allclose(
            np.sort(shuffled[splits == split], axis=0), np.sort(matrix[splits == split], axis=0), equal_nan=True
        )
    assert not np.allclose(shuffled[:10], matrix[:10], equal_nan=True)


def test_full_and_frozen_target_training(tmp_path):
    torch.set_num_threads(1)
    cfg = StudyConfig(**tomllib.loads(CONFIG.read_text())["study"])
    cfg.head_epochs = 2
    cfg.batch_size = 8
    cfg.latent_dim = 8
    enc = make_encoder({"kind": "mlp", "hidden": [12]}, 16, 8, 1)
    x = torch.randn(32, 16)
    enc.eval()
    with torch.no_grad():
        z = enc(x).tanh()
    y = np.linspace(-1, 1, 32)
    tr = np.arange(32) < 20
    va = ~tr
    original = {k: v.clone() for k, v in enc.state_dict().items()}
    for mode in ["frozen", "full"]:
        r = fit_target(enc, x, z, y, tr, va, mode, 0.001, 2, cfg, tmp_path / mode)
        assert r["epochs"] == 2 and np.isfinite(r["val_mse"])
        assert (tmp_path / mode / "best.pt").is_file() and (tmp_path / mode / "last.pt").is_file()
    assert all(torch.equal(v, enc.state_dict()[k]) for k, v in original.items())


@pytest.mark.parametrize("condition", ["real1_structure", "shuffled12"])
def test_source_fit_budget_checkpoints_and_completed_reload(tmp_path, condition):
    from study import source_fit
    from prepare import SOURCE

    cfg = StudyConfig(**tomllib.loads(CONFIG.read_text())["study"])
    cfg.latent_dim = 8
    cfg.batch_size = 4
    cfg.validation_interval = 1
    cfg.checkpoint_steps = [1, 2]
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(30, len(SOURCE))), columns=SOURCE)
    frame["split"] = ["train"] * 20 + ["val"] * 10
    enc = make_encoder({"kind": "mlp", "hidden": [12]}, 16, 8, 1)
    x = torch.randn(30, 16)
    case = dict(condition=condition, seed=1, lr=0.001)
    result = source_fit(enc, x, frame, case, cfg, tmp_path, 2)
    assert result["steps"] == 2 and sum(result["exposures"]) == 8
    assert (tmp_path / "source_step1.pt").is_file() and (tmp_path / "source_step2.pt").is_file()
    history = pd.read_csv(tmp_path / "source_history.csv")
    assert history.step.tolist() == [1, 2] and np.isfinite(history.to_numpy()).all()
    saved = {k: v.clone() for k, v in enc.state_dict().items()}
    with torch.no_grad():
        for p in enc.parameters():
            p.zero_()
    assert source_fit(enc, x, frame, case, cfg, tmp_path, 2) == result
    assert all(torch.equal(v, enc.state_dict()[k]) for k, v in saved.items())


def test_source_uses_registered_task_indices(tmp_path):
    from study import source_fit
    from prepare import SOURCE

    torch.set_num_threads(1)
    cfg = StudyConfig(**tomllib.loads(CONFIG.read_text())["study"])
    cfg.latent_dim = 8
    cfg.batch_size = 4
    cfg.validation_interval = 1
    cfg.checkpoint_steps = [2]
    frame = pd.DataFrame({n: np.full(30, np.nan) for n in SOURCE})
    frame[SOURCE[3]] = np.linspace(1, 3, 30)
    frame["split"] = ["train"] * 20 + ["val"] * 10
    enc = make_encoder({"kind": "mlp", "hidden": [12]}, 16, 8, 1)
    result = source_fit(
        enc, torch.randn(30, 16), frame, dict(condition="real1_structure", seed=2, lr=0.001), cfg, tmp_path, 2
    )
    assert result["task_indices"] == [3] and result["exposures"] == [8]


def test_capacity_matching_and_feature_inputs():
    torch.set_num_threads(1)
    raw = tomllib.loads(CONFIG.read_text())
    for feature, width in [("kmd", 464), ("composition", 94)]:
        counts = []
        for kind in ["mlp", "transformer", "no_attention"]:
            enc = make_encoder(raw["arms"][f"{feature}_{kind}"], width, 384, 1)
            enc.eval()
            with torch.no_grad():
                assert enc(torch.rand(2, width)).shape == (2, 384)
            counts.append(sum(p.numel() for p in enc.parameters()))
        assert max(counts) / min(counts) < 1.01


def test_invalid_grids_and_source_sets():
    raw = tomllib.loads(CONFIG.read_text())["study"]
    for field, value in [
        ("source_budgets", []),
        ("source_sets", {"bad": [12]}),
        ("source_sets", {"bad": [1, 1]}),
        ("full_conditions", ["unknown"]),
    ]:
        with pytest.raises(ValueError):
            StudyConfig(**(raw | {field: value}))
