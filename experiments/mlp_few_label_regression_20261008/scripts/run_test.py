"""Paired recipe isolation and actual-training validation."""

from pathlib import Path

import pandas as pd
import pytest

from run import audit_fit, recipe, restart_incomplete


def test_recipe_preserves_protocol_and_resolves_size():
    m = {
        "recipes": {
            a: {
                "datasets": {"qc": {"path": "old"}, "other": {"path": "other.parquet"}},
                "training": {},
                "pretrain": {},
                "finetune": {},
            }
            for a in ("scratch", "transfer")
        },
        "subsets": [{"n": 10, "seed": 1, "file": "n10s1.parquet"}],
    }
    c = {"n": 10, "seed": 1, "task": "band_gap"}
    r = recipe(m, c, "transfer", Path("/data"), False)
    assert r["datasets"]["qc"]["path"] == "/data/n10s1.parquet"
    assert r["training"]["seed"] == 2026
    assert r["finetune"]["epochs"] == 150
    assert r["finetune"]["tasks"] == ["band_gap"]
    assert not r["finetune"]["freeze_encoder"]
    assert m["recipes"]["transfer"]["training"] == {}
    assert recipe(m, c, "scratch", Path("/data"), True)["training"]["max_epochs"] == 2


def test_metrics_and_missing_training(tmp_path):
    step = tmp_path / "training" / "finetune"
    step.mkdir(parents=True)
    pd.DataFrame({"composition": ["a", "b"], "true": [0.0, 2.0], "pred": [0.5, 1.5]}).to_parquet(
        step / "x_pred.parquet"
    )
    logs = tmp_path / "logs" / "finetune" / "version_0"
    logs.mkdir(parents=True)
    p = logs / "metrics.csv"
    pd.DataFrame({"epoch": [0, 1], "step": [3, 7], "train_final_loss_epoch": [2.0, 1.0]}).to_csv(p, index=False)
    m = audit_fit(tmp_path, "x", "transfer")
    assert m["rmse"] == 0.5 and m["r2"] == 0.75 and m["epochs"] == 2
    pd.DataFrame({"epoch": [0], "step": [0], "train_final_loss_epoch": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="actual"):
        audit_fit(tmp_path, "x", "transfer")


def test_incomplete_fit_quarantined_and_retry_bounded(tmp_path):
    dest = tmp_path / "scratch"
    dest.mkdir()
    (dest / "previous.log").write_text("failure evidence")
    restart_incomplete(tmp_path, "scratch")
    assert not dest.exists()
    assert next(tmp_path.glob("failed_scratch_*/previous.log")).read_text() == "failure evidence"
    dest.mkdir()
    with pytest.raises(RuntimeError, match="exhausted"):
        restart_incomplete(tmp_path, "scratch")
