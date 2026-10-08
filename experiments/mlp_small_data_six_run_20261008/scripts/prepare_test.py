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
