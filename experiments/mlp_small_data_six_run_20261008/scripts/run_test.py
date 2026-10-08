"""Paired recipe isolation and actual-training validation."""

from pathlib import Path
import os
import subprocess

import pandas as pd
import pytest
import torch

from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import MLPEncoderConfig, RegressionTaskConfig

import importlib.util
import sys

spec = importlib.util.spec_from_file_location("six_run", Path(__file__).with_name("run.py"))
run = importlib.util.module_from_spec(spec)
sys.modules["six_run"] = run
spec.loader.exec_module(run)
audit_fit, recipe, restart_incomplete = run.audit_fit, run.recipe, run.restart_incomplete
verify_scripts, digest, paired_target_initialization = run.verify_scripts, run.digest, run.paired_target_initialization


def test_target_initialization_paired_despite_source_heads():
    hashes = []
    states = []
    original = FlexibleMultiTaskModel.add_task
    for old_count in (0, 3):
        torch.manual_seed(2025)
        model = FlexibleMultiTaskModel(
            encoder_config=MLPEncoderConfig(hidden_dims=[4, 8]), task_configs=[], enable_autoencoder=True
        )
        for i in range(old_count):
            model.add_task(RegressionTaskConfig(name=f"source{i}", dims=[8, 12, 1], data_column="y"))
        rng = torch.random.get_rng_state().clone()
        encoder = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        with paired_target_initialization("target", 12025) as audit:
            model.add_task(RegressionTaskConfig(name="target", dims=[8, 12, 1], data_column="y"))
        hashes.append(audit)
        states.append(model.task_heads["target"].state_dict())
        assert torch.equal(rng, torch.random.get_rng_state())
        assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in encoder.items())
        assert FlexibleMultiTaskModel.add_task is original
    assert hashes[0] == hashes[1]
    assert all(torch.equal(v, states[1][k]) for k, v in states[0].items())


def test_target_initialization_requires_fresh_head():
    with pytest.raises(ValueError, match="not freshly"):
        with paired_target_initialization("target", 12025):
            pass


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


def test_staged_script_hashes_reject_code_drift(tmp_path):
    for name in ("run.py", "array.sbatch"):
        (tmp_path / name).write_text("registered code")
    manifest = {"script_sha256": {name: digest(tmp_path / name) for name in ("run.py", "array.sbatch")}}
    verify_scripts(manifest, tmp_path)
    (tmp_path / "run.py").write_text("unreviewed edit")
    with pytest.raises(ValueError, match="checksum"):
        verify_scripts(manifest, tmp_path)


@pytest.mark.parametrize("pack,cpus", [(5, 16), (4, 8)])
def test_batch_rejects_cpu_oversubscription_before_workers(tmp_path, pack, cpus):
    # Stub only cluster-specific module/image probing; execute the real Bash guard.
    result = subprocess.run(
        [
            "bash",
            "-c",
            'module(){ :; }; git(){ printf "revision\\n"; }; singularity(){ printf "expected\\n"; }; sha256sum(){ printf "expected  image\\n"; }; source "$1"',
            "test",
            str(Path(__file__).with_name("array.sbatch")),
        ],
        env={
            **os.environ,
            "FM_WORKSPACE": str(tmp_path),
            "FM_IMAGE": "image",
            "FM_REVISION": "revision",
            "FM_DATA": str(tmp_path),
            "FM_IMAGE_HASH": "expected",
            "FM_OUTPUT": str(tmp_path / "output"),
            "SLURM_JOB_PARTITION": "ai-h200-brc-pu",
            "SLURM_JOB_ID": "test",
            "SLURM_CPUS_PER_TASK": str(cpus),
            "PACK": str(pack),
            "FIRST_CASE": "0",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("revision,pin", [("wrong", "expected"), ("revision", "wrong")])
def test_batch_rejects_wrong_commit_and_unregistered_image(tmp_path, revision, pin):
    result = subprocess.run(
        [
            "bash",
            "-c",
            'module(){ :; }; git(){ printf "revision\\n"; }; singularity(){ printf "%s\\n" "$TEST_PIN"; }; sha256sum(){ printf "expected  image\\n"; }; source "$1"',
            "test",
            str(Path(__file__).with_name("array.sbatch")),
        ],
        env={
            **os.environ,
            "FM_WORKSPACE": str(tmp_path),
            "FM_IMAGE": "image",
            "FM_IMAGE_HASH": "expected",
            "FM_REVISION": revision,
            "FM_DATA": str(tmp_path),
            "TEST_PIN": pin,
            "SLURM_JOB_PARTITION": "ai-h200-brc-pu",
            "SLURM_JOB_ID": "test",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    assert not (tmp_path / "output").exists()


def test_worker_still_identical_to_reused_campaign():
    sibling = Path(__file__).resolve().parents[2] / "mlp_few_label_regression_20261008/scripts/run.py"
    assert Path(__file__).with_name("run.py").read_bytes() == sibling.read_bytes()
