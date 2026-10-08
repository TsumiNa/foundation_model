"""Six-repeat registry, reproducible nested masks and reuse preconditions."""

import copy
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("six_prepare", Path(__file__).with_name("prepare.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def frame():
    data = pd.DataFrame({"split": ["train"] * 12500 + ["val"] * 5 + ["test"] * 5})
    data.index = pd.Index([f"compound_{i}" for i in range(len(data))])
    for col in prepare.TARGETS.values():
        data[col] = np.arange(len(data), dtype=float)
    return data


@pytest.mark.parametrize("seed", [0, 2, 5])
def test_nested_counts_preserve_reused_prefix_and_holdout(seed):
    data = frame()
    outputs, selection = prepare.mask_nested(data, seed)
    for i, (task, col) in enumerate(prepare.TARGETS.items()):
        pool = data.index[data.split == "train"].to_numpy()
        old_order = np.random.default_rng(np.random.SeedSequence([20261008, seed, i])).permutation(pool)[:100]
        assert selection[task][:100] == old_order.tolist()
        for n in prepare.COUNTS:
            actual = outputs[n].index[(outputs[n].split == "train") & outputs[n][col].notna()]
            assert len(actual) == n and set(actual) == set(selection[task][:n])
            pd.testing.assert_series_equal(
                outputs[n].loc[data.split != "train", col], data.loc[data.split != "train", col]
            )
    assert data.notna().all().all()


def test_insufficient_other_task_pool_and_invalid_inputs():
    data = frame().iloc[:120]
    outputs, selection = prepare.mask_nested(data, 0)
    assert len(selection["piezoelectric_max"]) == 120
    assert outputs[10000][prepare.TARGETS["piezoelectric_max"]].notna().sum() == 120
    with pytest.raises(ValueError, match="Insufficient"):
        prepare.mask_nested(data.iloc[:99], 0)
    with pytest.raises(ValueError, match="Missing"):
        prepare.mask_nested(data.drop(columns=[prepare.TARGETS["cbm"]]), 0)
    with pytest.raises(ValueError, match="unique"):
        prepare.mask_nested(pd.concat([data, data]), 0)


def test_six_source_frozen_draw_and_substitution(tmp_path, monkeypatch):
    models = [{"run": f"m{i:03d}", "sha256": str(i)} for i in range(240)]
    path = tmp_path / "population.json"
    path.write_text(json.dumps({"models": models, "n_models": 240, "library": prepare.LIBRARY}))
    monkeypatch.setattr(prepare, "POPULATION_SHA256", prepare.sha256(path))
    selection = {
        "sampling_seed": 20261001,
        "source_library": prepare.LIBRARY,
        "population": 240,
        "selection_method": "Uniform random sample without replacement; no AGIS evaluation used",
        "models": random.Random(20261001).sample(models, 10),
    }
    prepare.validate_selection(selection, path)
    wrong = copy.deepcopy(selection)
    wrong["models"][5] = wrong["models"][0]
    with pytest.raises(ValueError, match="random draw"):
        prepare.validate_selection(wrong, path)


def test_refuse_unregistered_reuse_before_preparing(tmp_path):
    reuse = tmp_path / "reuse"
    reuse.mkdir()
    (reuse / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Reuse manifest"):
        prepare.prepare(
            tmp_path / "source",
            tmp_path / "checkpoints",
            tmp_path / "configs",
            tmp_path / "new",
            reuse,
            tmp_path / "results",
            tmp_path / "plan",
        )
    assert not (tmp_path / "new").exists()


def test_registered_counts_cover_93_points():
    # The plan is an explicitly documented external input, not a hidden runtime dependency.
    path = Path(__file__).resolve().parents[2] / "rikyu_hparam_tuning_v2/summary/lowdata_n_plan.json"
    plan = json.loads(path.read_text())
    assert sum(3 + len(plan["plan"][t]) for t in prepare.TARGETS) == 93
    assert len(prepare.SEEDS) == 6


@pytest.mark.parametrize("failure", [None, "subset", "checkpoint", "recipe"])
def test_full_preparation_reuses_only_equivalent_completed_cases(tmp_path, monkeypatch, failure):
    import torch

    old_script = Path(__file__).resolve().parents[2] / "mlp_few_label_regression_20261008/scripts/prepare.py"
    old_spec = importlib.util.spec_from_file_location("original_prepare_fixture", old_script)
    old_module = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(old_module)
    for module in (prepare, old_module):
        monkeypatch.setattr(module, "canonical_key", lambda value, normalize: value)
    data = frame()
    data["composition"] = data.index
    source = tmp_path / "source.parquet"
    data.to_parquet(source)
    for name in (
        "NEMAD_magnetic_20260419_norm.parquet",
        "NEMAD_superconductor_20260425_norm.parquet",
        "phonix-db-filtered_20260425_norm.parquet",
    ):
        data.to_parquet(tmp_path / name)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    models = [{"run": f"m{i:03d}", "sha256": "unused"} for i in range(240)]
    chosen = random.Random(20261001).sample(models, 10)
    state = {
        "model": {
            "encoder.shared.layers.0.layer.weight": torch.zeros(256, 464),
            "encoder.shared.layers.1.layer.weight": torch.zeros(384, 256),
        },
        "task_sequence": ["density"],
    }
    for model in chosen[:6]:
        path = checkpoints / (model["run"] + ".pt")
        torch.save(state, path)
        model["sha256"] = prepare.sha256(path)
    population = checkpoints / "population_20261001.json"
    population.write_text(json.dumps({"models": models, "n_models": 240, "library": prepare.LIBRARY}))
    for module in (prepare, old_module):
        monkeypatch.setattr(module, "POPULATION_SHA256", prepare.sha256(population))
    selection = {
        "models": chosen,
        "population": 240,
        "source_library": prepare.LIBRARY,
        "sampling_seed": 20261001,
        "selection_method": "Uniform random sample without replacement; no AGIS evaluation used",
    }
    (checkpoints / "selection_20261001.json").write_text(json.dumps(selection))
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in ("probe6_mp2026_lowdata.toml", "ft_lowdata_mp2026.toml"):
        (configs / name).write_text(
            '[model]\nlatent_dim=384\n[training]\nencoder_lr=0.002\n[datasets.qc]\npath="source.parquet"\n'
        )
    old_data = tmp_path / "reuse-data"
    old = old_module.prepare(source, checkpoints, configs, old_data)
    monkeypatch.setattr(prepare, "REUSE_MANIFEST_SHA256", prepare.sha256(old_data / "manifest.json"))
    old_root = tmp_path / "old-root"
    identity = {
        "manifest": prepare.REUSE_MANIFEST_SHA256,
        "revision": "630e973dfa51bde36c0ce8dbaee34919ad9f73ed",
        "image_hash": "f90adc81f1148db2fff0ae94551b751e77f82c5d52d032ea54641f746dbe2f67",
        "smoke": False,
        "script_sha256": old["script_sha256"],
    }
    for case in old["cases"]:
        lane = old_root / f"case{case['case']:03d}"
        lane.mkdir(parents=True)
        (lane / "done.json").write_text(json.dumps({"case": case, "identity": identity}))
    plan = Path(__file__).resolve().parents[2] / "rikyu_hparam_tuning_v2/summary/lowdata_n_plan.json"
    if failure == "subset":
        (old_data / old["subsets"][0]["file"]).write_bytes(b"corrupted")
    if failure == "checkpoint":
        (checkpoints / (chosen[5]["run"] + ".pt")).write_bytes(b"corrupted")
    if failure == "recipe":
        path = configs / "ft_lowdata_mp2026.toml"
        path.write_text(path.read_text().replace("latent_dim=384", "latent_dim=256"))
    destination = tmp_path / "new-data"
    if failure:
        with pytest.raises(ValueError, match="checksum|hash mismatch|differs"):
            prepare.prepare(source, checkpoints, configs, destination, old_data, old_root, plan)
    else:
        manifest = prepare.prepare(source, checkpoints, configs, destination, old_data, old_root, plan)
        assert (
            len(manifest["cases"]) == 402 and len(manifest["reuse_cases"]) == 156 and len(manifest["all_cases"]) == 558
        )
        assert manifest["fits"] == 1116 and len(manifest["checkpoints"]) == 6 and len(manifest["subsets"]) == 48
        for subset in old["subsets"]:
            assert (old_data / subset["file"]).read_bytes() == (destination / subset["file"]).read_bytes()
        old_keys = {(c["task"], c["n"], c["seed"]) for c in old["cases"]}
        new_keys = {(c["task"], c["n"], c["seed"]) for c in manifest["cases"]}
        assert not old_keys & new_keys and len(old_keys | new_keys) == 558
