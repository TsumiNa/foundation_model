import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from benchmark import DATE, SOURCE_TASKS, file_hash
import run_lane

PROTOCOL = Path(__file__).parents[1] / "configs/protocol.toml"


def test_lane_matrix_resume_skip_and_stale_identity(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    source = pd.DataFrame({"composition": ["C", "FeO"], "split": ["train", "val"]})
    for name in SOURCE_TASKS:
        source[name] = 1.0
    source.to_parquet(data / "source.parquet", index=False)
    (data / f"source_scalers_{DATE}.joblib").write_bytes(b"source")
    source_entry = dict(
        file="source.parquet",
        sha256=file_hash(data / "source.parquet"),
        scaler_sha256=file_hash(data / f"source_scalers_{DATE}.joblib"),
    )
    targets = []
    for target in ("dielectric_total", "power_factor"):
        for fraction in (0.1, 1):
            filename = f"{target}_{fraction}"
            source.to_parquet(data / f"{filename}.parquet", index=False)
            (data / f"{filename}.joblib").write_bytes(b"target")
            targets.append(
                dict(
                    task=target,
                    fraction=fraction,
                    seed=0,
                    file=f"{filename}.parquet",
                    scaler=f"{filename}.joblib",
                    sha256=file_hash(data / f"{filename}.parquet"),
                    scaler_sha256=file_hash(data / f"{filename}.joblib"),
                )
            )
    (data / f"manifest_{DATE}.json").write_text(
        json.dumps(dict(source=source_entry, targets=targets, functional_smoke=True, smoke_batch_size=2))
    )
    calls = []

    def source_fit(raw, output, protocol, count):
        for k in (1, 3, 7):
            path = output / "training" / f"step{k:02d}_task" / "checkpoint.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"checkpoint")
        run_lane.atomic_json(output / "done.json", {})

    def target_fit(raw, source, output, target, **options):
        calls.append((target, options["fraction"], options["k"], options["mode"]))

    monkeypatch.setattr(run_lane, "train_source", source_fit)
    monkeypatch.setattr(run_lane, "train_target", target_fit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lane",
            "--protocol",
            str(PROTOCOL),
            "--data-dir",
            str(data),
            "--output-root",
            str(tmp_path / "runs"),
            "--arm",
            "mlp_tuned",
            "--seed",
            "0",
            "--cpu-smoke",
            "--max-epochs",
            "1",
        ],
    )
    run_lane.main()
    assert len(set(calls)) == 28
    assert {mode for _, _, _, mode in calls} == {"scratch", "frozen", "full"}
    assert {k for _, _, k, mode in calls if mode != "scratch"} == {1, 3, 7}
    lane = tmp_path / "runs/mlp_tuned_s0"
    identity = json.loads((lane / "identity.json").read_text())
    identity["protocol_sha256"] = "wrong"
    run_lane.atomic_json(lane / "identity.json", identity)
    with pytest.raises(ValueError, match="different protocol"):
        run_lane.main()


def test_completed_fit_skips_subprocesses(tmp_path, monkeypatch):
    run_lane.atomic_json(tmp_path / "done.json", {})
    monkeypatch.setattr(run_lane, "run_command", lambda *args: pytest.fail("Completed fit was relaunched"))
    run_lane.train_target(
        {}, None, tmp_path, "power_factor", mode="full", k=7, arm="mlp_tuned", seed=0, fraction=1, protocol=PROTOCOL
    )


def test_source_refuses_zero_training_epochs(tmp_path, monkeypatch):
    source_path = tmp_path / "source.parquet"
    frame = pd.DataFrame({"split": ["train"] * 2})
    for name in SOURCE_TASKS:
        frame[name] = 1.0
    frame.to_parquet(source_path)
    raw = {"datasets": {"source": {"path": str(source_path)}}, "data": {"batch_size": 2}}
    monkeypatch.setattr(run_lane, "build_pretrain_config", lambda raw: None)

    def command(*args):
        path = tmp_path / "run/training"
        path.mkdir(parents=True)
        (path / "final_model.pt").write_bytes(b"checkpoint")
        (path / "experiment_records.json").write_text(json.dumps([{"epochs_run": 0}]))

    monkeypatch.setattr(run_lane, "run_command", command)
    with pytest.raises(RuntimeError, match="without training"):
        run_lane.train_source(raw, tmp_path / "run", PROTOCOL, 1)


@pytest.mark.parametrize("records", [None, [], [{"step": 3, "epochs_run": 2}]])
def test_resumed_source_uses_cumulative_checkpoints_not_invocation_record_count(tmp_path, monkeypatch, records):
    source_path = tmp_path / "source.parquet"
    frame = pd.DataFrame({"split": ["train"] * 3})
    for name in SOURCE_TASKS:
        frame[name] = 1.0
    frame.to_parquet(source_path)
    output = tmp_path / "run"
    for k, task in enumerate(SOURCE_TASKS[:3], start=1):
        directory = output / f"training/step{k:02d}_{task}"
        directory.mkdir(parents=True)
        torch.save(
            {"model": {}, "task_sequence": list(SOURCE_TASKS[:k]), "step": k, "new_task": task},
            directory / "checkpoint.pt",
        )
        log = output / f"logs/step{k:02d}_{task}/version_0"
        log.mkdir(parents=True)
        pd.DataFrame({"epoch": [0, 1], "train_final_loss_epoch": [1.0, 0.5]}).to_csv(log / "metrics.csv", index=False)
    torch.save({"model": {}, "task_sequence": list(SOURCE_TASKS[:3])}, output / "training/final_model.pt")
    if records is not None:
        (output / "training/experiment_records.json").write_text(json.dumps(records))
    monkeypatch.setattr(run_lane, "build_pretrain_config", lambda raw: None)
    monkeypatch.setattr(run_lane, "run_command", lambda *args: None)  # CLI resumes/skips completed stages.
    run_lane.train_source(
        {"datasets": {"source": {"path": str(source_path)}}, "data": {"batch_size": 2}}, output, PROTOCOL, 3
    )
    assert json.loads((output / "done.json").read_text())["source_count"] == 3
