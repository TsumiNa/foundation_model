import json
from pathlib import Path

import numpy as np
import pytest

from collect_head_probe import collect, linear_cka

EXP = Path(__file__).resolve().parents[1]


def test_cka_rotation_scale_and_degenerate_inputs():
    x = np.random.default_rng(0).normal(size=(40, 8))
    q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(8, 8)))
    assert linear_cka(x, 5 * x @ q) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        linear_cka(x, np.zeros((40, 8)))
    with pytest.raises(ValueError):
        linear_cka(x, x[:2])


def test_empty_collection_and_smoke_rejection(tmp_path):
    frame, audit = collect(tmp_path, EXP / "configs/head_probe.toml", EXP / "configs/protocol.toml")
    assert frame.empty and audit["expected_fits"] == 162 and not audit["complete"]
    lane = tmp_path / "mlp_tuned_s0_k1"
    lane.mkdir()
    identity = {"smoke": True}
    (lane / "identity.json").write_text(json.dumps(identity))
    (lane / "done.json").write_text(json.dumps({"identity": identity, "encoder_unchanged": True}))
    with pytest.raises(ValueError, match="smoke"):
        collect(tmp_path, EXP / "configs/head_probe.toml", EXP / "configs/protocol.toml")
