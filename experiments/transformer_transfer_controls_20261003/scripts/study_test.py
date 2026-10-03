from pathlib import Path
import tomllib
import numpy as np
import pandas as pd
import pytest
import torch

from study import StudyConfig, cases, normalization, shuffle_labels, make_encoder, fit_target
from prepare import isolate

CONFIG = Path(__file__).parents[1] / "configs/study.toml"


def test_matrix_and_disjoint_seeds():
    raw = tomllib.loads(CONFIG.read_text())
    cfg = StudyConfig(**raw["study"])
    assert len(cases(raw, "pilot")) == 24
    formal = cases(raw, "formal", {a: v["learning_rates"][0] for a, v in raw["arms"].items()})
    assert len(formal) == 200 and len({(c["arm"], c["seed"], c["condition"]) for c in formal}) == 200
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


def test_alias_split_precedence_preserves_original_formula():
    f = isolate(pd.DataFrame({"composition": ["Fe4 O6", "Fe2 O3", "NaCl"], "split": ["train", "test", "val"]}))
    assert f.composition.tolist() == ["Fe4 O6", "NaCl"] and f.split.tolist() == ["test", "val"]


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
