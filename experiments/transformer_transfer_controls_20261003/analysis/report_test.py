from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from report import aligned_predictions, effects
import report as report_module


def test_effects_pairs_seeds_before_averaging():
    rows = []
    for seed, scale in [(10, 1.0), (11, 10.0)]:
        for condition, value in [("random", 2), ("real1", 1.5), ("real3", 1.2), ("real7", 1), ("shuffled7", 2.5)]:
            rows.append(
                dict(
                    arm="transformer",
                    target="Bulk modulus",
                    fraction=0.1,
                    mode="full",
                    seed=seed,
                    condition=condition,
                    rmse=scale * value,
                )
            )
    frame = pd.DataFrame(rows).sample(frac=1, random_state=3)
    r = effects(frame, [10, 11]).set_index("baseline")
    assert r.loc["random", "relative_mean"] == -50
    assert r.loc["random", "delta_rmse_mean"] == -5.5
    assert r.loc["random", "improved_seeds"] == 2
    with pytest.raises(ValueError, match="Complete paired"):
        effects(frame[~((frame.seed == 11) & (frame.condition == "real7"))], [10, 11])
    frame.loc[frame.condition == "random", "rmse"] = 0
    with pytest.raises(ValueError, match="positive denominator"):
        effects(frame, [10, 11])


def test_predictions_align_compositions_and_reject_truth_mismatch(tmp_path: Path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pd.DataFrame(dict(composition=["B", "A"], true=[2.0, 1.0], pred=[2.1, 1.1])).to_parquet(first)
    frame = pd.DataFrame(dict(composition=["A", "B"], true=[1.0, 2.0], pred=[0.9, 1.9]))
    frame.to_parquet(second)
    index, truth, values = aligned_predictions([first, second])
    assert index.tolist() == ["A", "B"]
    np.testing.assert_allclose(truth, [1, 2])
    np.testing.assert_allclose(values, [[1.1, 2.1], [0.9, 1.9]])
    frame.loc[0, "true"] = 7
    frame.to_parquet(second)
    with pytest.raises(ValueError, match="ground-truth mismatch"):
        aligned_predictions([first, second])
    frame.loc[1, "composition"] = "A"
    frame.to_parquet(second)
    with pytest.raises(ValueError, match="Invalid prediction"):
        aligned_predictions([second])


def test_partial_report_stops_before_effects_or_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(report_module, "collect", lambda *args: {"partial": True})

    def forbidden(*args):
        pytest.fail("Partial campaign reached final analysis")

    monkeypatch.setattr(report_module, "effects", forbidden)
    monkeypatch.setattr(report_module, "figures", forbidden)
    with pytest.raises(ValueError, match="complete registered"):
        report_module.report(tmp_path, tmp_path / "missing.toml", tmp_path, tmp_path)
    assert not (tmp_path / "REPORT_EN_20261003.md").exists()


def test_complete_report_uses_only_audited_case_range(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("[study]\nseeds=[10,11]\n")
    pd.DataFrame({"rmse": [1.0]}).to_csv(tmp_path / "metrics.csv", index=False)
    (tmp_path / "case999").mkdir()  # Stale matching directory must never enter figures.
    audit = dict(partial=False, complete_lanes=2, expected_lanes=2, metrics=48)
    monkeypatch.setattr(report_module, "collect", lambda *args: audit)
    paired = pd.DataFrame(
        [
            dict(
                arm="mlp",
                target="Bulk modulus",
                fraction=1.0,
                mode="full",
                baseline="random",
                relative_mean=-10.0,
                relative_lo95=-12.0,
                relative_hi95=-8.0,
                improved_seeds=2,
                n_seeds=2,
            )
        ]
    )
    monkeypatch.setattr(report_module, "effects", lambda frame, seeds: paired)
    seen = []

    def fake_figures(frame, effects, lanes, output, seeds):
        seen.extend(lanes)
        assert seeds == [10, 11]

    monkeypatch.setattr(report_module, "figures", fake_figures)
    assert report_module.report(tmp_path, config, tmp_path, tmp_path) == audit
    assert seen == [tmp_path / "case000", tmp_path / "case001"]
    assert (tmp_path / "paired_relative_effects.csv").is_file()
    text = (tmp_path / "REPORT_EN_20261003.md").read_text()
    assert "-10.00%" in text and "Audited 2 lanes and 48 selected endpoints" in text
