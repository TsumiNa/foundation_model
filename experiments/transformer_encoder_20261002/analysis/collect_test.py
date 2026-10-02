import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from collect import collect, paired_comparisons

PROTOCOL = Path(__file__).parents[1] / "configs/protocol.toml"


def test_pairing_uses_same_encoder_seed_target_and_budget():
    frame = pd.DataFrame(
        [
            dict(
                arm="mlp_tuned",
                seed=0,
                target="power_factor",
                fraction=0.1,
                source_count=0,
                mode="scratch",
                test_rmse=2.0,
                test_standardized_rmse=2.0,
            ),
            dict(
                arm="grouped_concat",
                seed=0,
                target="power_factor",
                fraction=0.1,
                source_count=0,
                mode="scratch",
                test_rmse=4.0,
                test_standardized_rmse=4.0,
            ),
            dict(
                arm="grouped_concat",
                seed=0,
                target="power_factor",
                fraction=0.1,
                source_count=7,
                mode="full",
                test_rmse=3.0,
                test_standardized_rmse=3.0,
            ),
        ]
    )
    paired = paired_comparisons(frame).iloc[0]
    assert paired["relative_change_pct"] == -25
    assert paired["mlp_scratch_rmse"] == 2  # Own-baseline gain is still worse than MLP scratch.


def test_collector_refuses_smoke_and_accepts_partial_registered_results(tmp_path):
    lane = tmp_path / "mlp_tuned_s0"
    lane.mkdir()
    settings = tomllib.loads(PROTOCOL.read_text())["benchmark"]
    runtime = dict(
        cpu_smoke=False,
        functional_smoke=True,
        max_epochs_override=None,
        image_revision=settings["image_revision"],
        sif_sha256=settings["sif_sha256"],
    )
    (lane / "runtime.json").write_text(json.dumps(runtime))
    with pytest.raises(ValueError, match="refuses"):
        collect(tmp_path, PROTOCOL)
    runtime["functional_smoke"] = False
    (lane / "runtime.json").write_text(json.dumps(runtime))
    frame, expected = collect(tmp_path, PROTOCOL)
    assert frame.empty and expected == 672
