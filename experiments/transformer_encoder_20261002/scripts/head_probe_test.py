"""Scientific controls: paired heads, validation-only fitting, and frozen-feature boundaries."""

import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from foundation_model.models.model_config import RegressionTaskConfig
from foundation_model.models.task_head.regression import RegressionHead
from head_probe import (
    ProbeConfig,
    Readout,
    fit_head,
    make_head,
    metrics,
    representation_stats,
    ridge_search,
    validate_frame,
)


def config(**updates):
    return ProbeConfig(
        **{
            **dict(
                arms=["mlp_tuned", "grouped_mean"],
                seeds=[0, 1, 2],
                source_counts=[1, 3, 7],
                fractions=[0.1, 1.0],
                learning_rates=[0.002],
                ridge_alphas=[0.01, 1.0],
                max_epochs=3,
                patience=2,
                batch_size=8,
                source_revision="a" * 40,
            ),
            **updates,
        }
    )


def test_paired_heads_match_package_except_explicit_output():
    seed = 23
    torch.manual_seed(seed)
    package = RegressionHead(RegressionTaskConfig(name="dielectric_total", dims=[6, 128, 64, 1])).eval()
    legacy = make_head(6, Readout.LEGACY, seed).eval()
    linear = make_head(6, Readout.LINEAR_OUTPUT, seed).eval()
    assert all(torch.equal(v, linear.state_dict()[k]) for k, v in legacy.state_dict().items())
    x = torch.randn(12, 6)
    assert torch.equal(package(x), legacy(x))
    assert torch.equal(legacy(x), nn.functional.leaky_relu(linear(x), 0.1))
    # Deterministically force negative output to ensure the control is not vacuous.
    for head in (legacy, linear):
        head.layers[-1].layer.weight.data.zero_()
        head.layers[-1].layer.bias.data.fill_(-2)
    torch.testing.assert_close(legacy(x), torch.full((12, 1), -0.2))
    torch.testing.assert_close(linear(x), torch.full((12, 1), -2.0))


def test_validation_selection_and_training_only_scaler():
    x = np.arange(60, dtype=float).reshape(20, 3)
    y = x[:, 0] * 2 + 1
    xv = np.ones((4, 3)) * 1000
    yv = np.full(4, 2001.0)
    scaler, model, trials = ridge_search(x, y, xv, yv, [0.001, 10000.0])
    np.testing.assert_allclose(scaler.mean_, x.mean(0))
    assert model.alpha == min(trials, key=lambda t: t["val_mse"])["alpha"]
    assert trials[0]["val_mse"] < trials[1]["val_mse"]


def test_fit_preserves_features_and_best_last(tmp_path: Path):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(32, 6)).astype("float32")
    y = x[:, 0].copy()
    before = x.copy()
    result = fit_head(x[:24], y[:24], x[24:], y[24:], Readout.LINEAR_OUTPUT, 13, 0.002, config(), tmp_path, "cpu")
    history = pd.read_csv(tmp_path / "history.csv")
    assert result["best_val_mse"] == pytest.approx(history.val_mse.min())
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "last.pt").is_file()
    assert result["epochs"] > 0
    assert (
        fit_head(x[:24], y[:24], x[24:], y[24:], Readout.LINEAR_OUTPUT, 13, 0.002, config(), tmp_path, "cpu") == result
    )
    np.testing.assert_array_equal(x, before)


def test_matrix_counts_and_invalid_inputs():
    cfg = config()
    assert len(cfg.cases()) == 18
    with pytest.raises(ValueError):
        config(learning_rates=[float("nan")])
    with pytest.raises(ValueError):
        config(source_counts=[1, 1])
    with pytest.raises(ValueError):
        config(max_epochs=True)
    with pytest.raises(ValueError):
        metrics(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        representation_stats(np.ones((1, 4)))
    frame = pd.DataFrame(
        {"composition": ["a", "b", "c"], "split": ["train", "val", "test"], "dielectric_total": [1, 2, 3]}
    )
    validate_frame(frame)
    frame.loc[2, "composition"] = "a"
    with pytest.raises(ValueError):
        validate_frame(frame)


def test_launcher_stops_partial_pack(tmp_path):

    exp = tmp_path / "experiments/transformer_encoder_20261002"
    (exp / "configs").mkdir(parents=True)
    fake = tmp_path / "apptainer"
    fake.write_text("""#!/bin/bash
if [[ " $* " == *" --nv "* ]]; then
  while [[ $# -gt 0 ]]; do
    if [[ $1 = --case ]]; then echo "case=$2"; exit 0; fi
    shift
  done
elif [[ " $* " == *"math.prod"* ]]; then echo 18
else echo abc
fi
""")
    fake.chmod(0o755)
    sha = tmp_path / "sha256sum"
    sha.write_text('#!/bin/bash\necho "abc image"\n')
    sha.chmod(0o755)
    script = Path(__file__).with_name("head_probe.sbatch").resolve()
    output = tmp_path / "output"
    env = {
        **os.environ,
        "PATH": str(tmp_path) + ":" + os.environ["PATH"],
        "APPTAINER": str(fake),
        "FM_WORKSPACE": str(tmp_path),
        "FM_SOURCE_WORKSPACE": str(tmp_path / "source"),
        "FM_SOURCE_ROOT": str(tmp_path / "source/artifacts"),
        "FM_DATA_DIR": str(tmp_path / "data"),
        "FM_OUTPUT_ROOT": str(output),
        "FM_IMAGE": "unused",
        "FM_BENCHMARK_REVISION": "a" * 40,
        "SLURM_JOB_ID": "test",
        "SLURM_ARRAY_TASK_ID": "4",
        "SLURM_CPUS_PER_TASK": "8",
        "PACK": "4",
    }
    subprocess.run(
        ["bash", "-c", 'module() { :; }; export -f module; exec bash "$1"', "test", str(script)], env=env, check=True
    )
    assert {p.name for p in output.glob("*.log")} == {"case-16-test.log", "case-17-test.log"}
