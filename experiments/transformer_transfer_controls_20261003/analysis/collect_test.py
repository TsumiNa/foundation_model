import numpy as np
import pytest
from collect import interval


def test_paired_interval_constant_effect_and_invalid_data():
    assert interval(np.full(10, -0.25)) == (-0.25, -0.25, -0.25)
    for x in [np.array([]), np.array([np.nan])]:
        with pytest.raises(ValueError):
            interval(x)


def test_partial_collector_rejects_smoke_and_missing_endpoints(tmp_path):
    import json
    import tomllib
    from pathlib import Path
    from collect import collect, file_hash

    cfg = Path(__file__).parents[1] / "configs/study.toml"
    raw = tomllib.loads(cfg.read_text())
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"learning_rates": {a: v["learning_rates"][0] for a, v in raw["arms"].items()}}))
    root = tmp_path / "runs"
    lane = root / "case000"
    lane.mkdir(parents=True)
    case = dict(arm="mlp", seed=10, condition="random", lr=0.0005)
    identity = dict(
        case=case,
        phase="formal",
        smoke=False,
        cpu_test=False,
        config=file_hash(cfg),
        selection=file_hash(selection),
        manifest=raw["study"]["data_manifest_sha256"],
        revision="rev",
        script="script",
        preparation_script="prep",
        package="0.5.0",
    )
    metrics = [
        dict(target=t, fraction=f, mode=m, rmse=1.0, mae=0.5, standardized_rmse=0.1, val_mse=0.01)
        for t in ["Dielectric total", "Bulk modulus", "Shear modulus", "Piezoelectric max"]
        for f in [0.1, 1.0]
        for m in ["ridge", "frozen", "full"]
    ]
    (lane / "identity.json").write_text(json.dumps(identity))
    (lane / "done.json").write_text(json.dumps(dict(case=case, metrics=metrics)))
    result = collect(root, cfg, selection, tmp_path / "results")
    assert result["partial"] and result["complete_lanes"] == 1 and result["metrics"] == 24
    identity["smoke"] = True
    (lane / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="identity"):
        collect(root, cfg, selection, tmp_path / "results")
    identity["smoke"] = False
    (lane / "identity.json").write_text(json.dumps(identity))
    metrics[0]["target"] = "unexpected"
    (lane / "done.json").write_text(json.dumps(dict(case=case, metrics=metrics)))
    with pytest.raises(ValueError, match="endpoint"):
        collect(root, cfg, selection, tmp_path / "results")
