import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collect_head_probe import collect, linear_cka, prediction_disagreement

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


def test_nonempty_audit_and_corrupt_predictions(tmp_path):

    base = tomllib.loads((EXP / "configs/protocol.toml").read_text())["benchmark"]
    lane = tmp_path / "mlp_tuned_s0_k1"
    folder = lane / "f010"
    folder.mkdir(parents=True)
    identity = dict(
        arm="mlp_tuned",
        seed=0,
        source_count=1,
        smoke=False,
        revision="a" * 40,
        config_sha256=hashlib.sha256((EXP / "configs/head_probe.toml").read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((EXP / "configs/protocol.toml").read_bytes()).hexdigest(),
        manifest_sha256=base["data_manifest_sha256"],
        sif_sha256=base["sif_sha256"],
    )
    row = dict(arm="mlp_tuned", seed=0, k=1, fraction=0.1, readout="ridge_post", rmse=1.0)
    done = dict(identity=identity, encoder_unchanged=True, selected=[row])
    (lane / "identity.json").write_text(json.dumps(identity))
    (lane / "done.json").write_text(json.dumps(done))
    predictions = pd.DataFrame(dict(composition=[f"c{i}" for i in range(697)], pred=np.ones(697), true=np.zeros(697)))
    predictions.to_parquet(folder / "ridge_post_pred.parquet")
    frame, audit = collect(tmp_path, EXP / "configs/head_probe.toml", EXP / "configs/protocol.toml")
    assert len(frame) == 1 and audit["selected_fits"] == 1 and not audit["complete"]
    done["selected"] = [row, row]
    (lane / "done.json").write_text(json.dumps(done))
    with pytest.raises(ValueError, match="Duplicate"):
        collect(tmp_path, EXP / "configs/head_probe.toml", EXP / "configs/protocol.toml")
    done["selected"] = [row]
    (lane / "done.json").write_text(json.dumps(done))
    predictions.loc[0, "pred"] = float("nan")
    predictions.to_parquet(folder / "ridge_post_pred.parquet")
    with pytest.raises(ValueError, match="nonfinite"):
        collect(tmp_path, EXP / "configs/head_probe.toml", EXP / "configs/protocol.toml")


def test_disagreement_aligns_by_composition_and_rejects_wrong_truth(tmp_path):

    rows = []
    for arm, delta in [("mlp_tuned", 1.0), ("grouped_mean", 2.0)]:
        folder = tmp_path / f"{arm}_s0_k1" / "f100"
        folder.mkdir(parents=True)
        pd.DataFrame(
            dict(composition=["a", "b", "c"], pred=np.array([1.0, 2.0, 3.0]) + delta, true=[1.0, 2.0, 3.0])
        ).iloc[::-1].to_parquet(folder / "ridge_post_pred.parquet")
        rows.append(dict(arm=arm, seed=0, k=1, fraction=1.0, readout="ridge_post"))
    summary, paired = prediction_disagreement(tmp_path, pd.DataFrame(rows))
    assert len(paired) == 3 and summary.iloc[0].difference_over_mlp_rmse == pytest.approx(1.0)
    p = tmp_path / "grouped_mean_s0_k1/f100/ridge_post_pred.parquet"
    pred = pd.read_parquet(p)
    pred.loc[0, "true"] = 100
    pred.to_parquet(p)
    with pytest.raises(ValueError, match="references"):
        prediction_disagreement(tmp_path, pd.DataFrame(rows))
