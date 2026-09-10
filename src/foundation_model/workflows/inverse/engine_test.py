# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke tests for ``fm inverse``.

A real run over the synthetic catalog: seeds selected, both paths optimised, every artefact
written. Slow relative to the rest of this package's tests, and the only thing that would catch a
break in how the five modules fit together.
"""

from __future__ import annotations

import json

import numpy as np

from foundation_model.workflows.inverse import run as inverse_run
from foundation_model.workflows.recording import RunRecorder

from .conftest import _checkpoint, _inverse_cfg


def test_trajectory_static_and_svg_animation_emitted(data_dir, tmp_path) -> None:
    ckpt = tmp_path / "ck.pt"
    _checkpoint(data_dir, ckpt)
    out = tmp_path / "inv"
    cfg = _inverse_cfg(data_dir, out, ckpt, animation='["svg"]')  # svg avoids slow FuncAnimation writers
    rec = RunRecorder(out)
    rec.write_provenance(config=cfg, argv=["fm", "inverse"], seeds={"seed": 2025})
    inverse_run(cfg, rec)
    rec.close()
    traj = out / "sc1" / "trajectories"
    assert (traj / "latent_align1_trajectory.png").exists()  # static plot always
    assert (traj / "latent_align1_trajectory.svg").exists()  # requested animation format
    npz = np.load(traj / "latent_align1.npz", allow_pickle=False)
    assert npz["targets"].shape[2] == 4  # one channel per target
    assert [str(v) for v in npz["labels"]] == ["a→-1", "b↑", "P(mat∈{1})↑", "k~curve(2pts)"]


def test_inverse_smoke_end_to_end(data_dir, tmp_path) -> None:
    ckpt = tmp_path / "ck.pt"
    _checkpoint(data_dir, ckpt)
    out = tmp_path / "inv"
    cfg = _inverse_cfg(data_dir, out, ckpt)
    rec = RunRecorder(out)
    rec.write_provenance(config=cfg, argv=["fm", "inverse"], seeds={"seed": 2025})
    summary = inverse_run(cfg, rec)
    rec.close()

    assert (out / "seeds.json").exists()
    assert (out / "inverse_design.json").exists()
    assert (out / "run_provenance.json").exists()
    sc = out / "sc1"
    for name in (
        "scenario.json",
        "results.json",
        "summary.json",
        "targets.json",
        "comparison.png",
        "objective_vs_targets_scatter.png",
        "element_frequency_heatmap.png",
    ):
        assert (sc / name).exists(), name
    assert (sc / "seed_to_optimized__latent_align1.png").exists()

    scenario = json.loads((sc / "scenario.json").read_text())
    assert [t["task"] for t in scenario["targets"]] == ["a", "b", "mat", "k"]
    assert [t["kind"] for t in scenario["targets"]] == ["value", "direction", "class", "curve"]

    payload = json.loads((sc / "results.json").read_text())
    assert set(payload["seed_predictions"]["channels"]) == {"a", "b", "mat", "k"}
    results = payload["results"]
    assert [r["path"] for r in results] == ["latent_align1", "comp_seed_blend95"]
    assert "objective_after_decode" in results[0] and "decoded_composition" in results[0]
    assert set(results[0]["channels_after_decode"]) == {"a", "b", "mat", "k"}
    assert "sc1" in summary and len(summary["sc1"]) == 2
    assert all("objective_mean" in row for row in summary["sc1"])
