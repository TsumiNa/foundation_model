import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import figures


@pytest.fixture
def example(tmp_path):
    raw = {"study": {"source_budgets": [6000, 24000], "split_seeds": [123], "seeds": [4]}}
    cases = [
        dict(arm=f"kmd_{arm}", condition=condition, steps=steps, split_seed=123, seed=4)
        for arm in ["no_attention", "mlp", "transformer"]
        for condition, steps in [("real12", 24000), ("random", 0)]
    ]
    for i, case in enumerate(cases):
        lane = tmp_path / f"case{i:03d}"
        target = lane / "target0_f010"
        target.mkdir(parents=True)
        frame = pd.DataFrame({"composition": ["B", "A", "C"], "true": [2.0, 1.0, 3.0], "pred": [2.5, -0.5, 3.5]})
        frame.sample(frac=1, random_state=i).to_parquet(target / "full_pred.parquet", index=False)
        metric = dict(
            target=figures.TARGETS[0],
            fraction=0.1,
            mode="full",
            rmse=float(np.sqrt(np.mean((frame.pred - frame.true) ** 2))),
        )
        (lane / "done.json").write_text(json.dumps({"metrics": [metric]}))
    return tmp_path, cases, raw


def test_metadata_selection_and_aligned_truths(example):
    root, cases, raw = example
    panels = figures.load_panels(root, cases, raw, "kmd", 0)
    assert [(p.row, p.column, p.arm) for p in panels] == [
        (row, col, f"kmd_{arm}") for col, arm in enumerate(["mlp", "transformer", "no_attention"]) for row in [0, 1]
    ]
    assert all(p.frame.composition.tolist() == ["A", "B", "C"] for p in panels)
    assert all(p.frame.pred.min() == -0.5 for p in panels)
    figures.parity(panels, raw, "kmd", 0, root)
    assert (root / "kmd_target0_same_composition_parity_20261003.png").stat().st_size > 1000


@pytest.mark.parametrize("fault", ["duplicate", "composition", "truth", "nan", "rmse"])
def test_reject_misaligned_or_invalid_predictions(example, fault):
    root, cases, raw = example
    path = root / "case000/target0_f010/full_pred.parquet"
    frame = pd.read_parquet(path)
    if fault == "duplicate":
        frame.loc[0, "composition"] = frame.loc[1, "composition"]
    elif fault == "composition":
        frame.loc[0, "composition"] = "X"
    elif fault == "truth":
        frame.loc[0, "true"] += 1
    elif fault == "nan":
        frame.loc[0, "pred"] = np.nan
    else:
        frame.loc[0, "pred"] += 5
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        figures.load_panels(root, cases, raw, "kmd", 0)


def test_missing_registered_case(example):
    root, cases, raw = example
    with pytest.raises(ValueError, match="registered"):
        figures.load_panels(root, cases[:-1], raw, "kmd", 0)


def test_refuse_partial_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "collect", lambda *args: {"partial": True})
    with pytest.raises(ValueError, match="complete audited"):
        figures.render(tmp_path, Path("missing"), Path("missing"), tmp_path)
    assert not list(tmp_path.glob("*.png"))
