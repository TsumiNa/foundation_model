"""Regression coverage for the final report's scientific aggregates."""

import json

import numpy as np
import pandas as pd
import pytest

from summarize_head_probe import audit_histories, best_last_table, gpu_cost


def test_best_last_pairing_aggregation_and_corruption(tmp_path):
    selected = pd.DataFrame(
        [dict(arm="mlp_tuned", seed=0, k=7, fraction=0.1, readout="linear_output", lr=0.002, rmse=1.0)]
    )
    folder = tmp_path / "mlp_tuned_s0_k7/f010"
    folder.mkdir(parents=True)
    best = pd.DataFrame(dict(composition=["a", "b"], true=[0.0, 2.0], pred=[1.0, 3.0]))
    last = best.assign(pred=[2.0, 4.0]).iloc[::-1]
    best.to_parquet(folder / "linear_output_pred.parquet")
    last.to_parquet(folder / "linear_output_selected_last_pred.parquet")
    (folder / "linear_output_selected_last.json").write_text(json.dumps(dict(lr=0.002, rmse=2.0)))
    table = best_last_table(tmp_path, selected)
    assert table.last_minus_best.tolist() == [1.0] and table.last_rmse.tolist() == [2.0]
    last.loc[0, "true"] = 10
    last.to_parquet(folder / "linear_output_selected_last_pred.parquet")
    with pytest.raises(ValueError, match="references"):
        best_last_table(tmp_path, selected)


def test_history_audit_checks_best_epoch_and_finiteness(tmp_path):
    selected = pd.DataFrame([dict(arm="mlp_tuned", seed=0, k=7, fraction=0.1, readout="linear_output")])
    folder = tmp_path / "mlp_tuned_s0_k7/f010/linear_output/lr0.002"
    folder.mkdir(parents=True)
    history = pd.DataFrame(dict(epoch=[0, 1, 2], train_mse=[2.0, 1.0, 0.5], val_mse=[2.0, 1.0, 1.2], lr=[0.002] * 3))
    history.to_csv(folder / "history.csv", index=False)
    done = dict(epochs=3, best_epoch=1, best_val_mse=1.0)
    (folder / "done.json").write_text(json.dumps(done))
    result = audit_histories(tmp_path, selected, [0.002])
    assert result["neural_trials"] == 1 and result["trial_epoch_max"] == 3
    done["best_epoch"] = 2
    (folder / "done.json").write_text(json.dumps(done))
    with pytest.raises(ValueError, match="checkpoint"):
        audit_histories(tmp_path, selected, [0.002])
    history.loc[0, "train_mse"] = np.nan
    history.to_csv(folder / "history.csv", index=False)
    with pytest.raises(ValueError, match="nonfinite"):
        audit_histories(tmp_path, selected, [0.002])


def test_cost_excludes_batch_rows_and_rejects_failure(tmp_path):
    path = tmp_path / "sacct.txt"
    path.write_text("1_0|COMPLETED|60|0:0|gres/gpu=1,cpu=8|\n1_0.batch|COMPLETED|60|0:0|gres/gpu=1|\n")
    result = gpu_cost(path)
    assert result["gpu_hours"] == pytest.approx(1 / 60) and result["estimated_jpy_pre_tax"] == pytest.approx(5)
    path.write_text("1_0|FAILED|60|1:0|gres/gpu=1|\n")
    with pytest.raises(ValueError, match="successful"):
        gpu_cost(path)
