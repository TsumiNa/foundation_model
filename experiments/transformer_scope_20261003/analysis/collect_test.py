from pathlib import Path
import json
import tomllib

import numpy as np
import pandas as pd
import pytest

from collect import collect, digest, expected_cases, paired_summary, contrasts

CONFIG = Path(__file__).parents[1] / "configs/study.toml"


def test_registered_matrix_and_partial_audit(tmp_path):
    raw = tomllib.loads(CONFIG.read_text())
    rates = {a: v["learning_rates"][0] for a, v in raw["arms"].items()}
    expected = expected_cases(raw, rates)
    assert len(expected) == 1026
    assert len({tuple(c.values()) for c in expected}) == 1026
    selected = tmp_path / "selection.json"
    selected.write_text(
        json.dumps(
            dict(
                learning_rates=rates, config_sha256=digest(CONFIG), manifest_sha256=raw["study"]["data_manifest_sha256"]
            )
        )
    )
    audit = collect(tmp_path / "absent", CONFIG, selected, tmp_path / "out")
    assert audit["expected_metrics"] == 16848 and audit["partial"] and audit["complete_lanes"] == 0
    lane = tmp_path / "absent/case000"
    lane.mkdir(parents=True)
    case = expected[0]
    identity = dict(
        case=case,
        phase="formal",
        config=digest(CONFIG),
        manifest=raw["study"]["data_manifest_sha256"],
        script=digest(Path(__file__).parents[1] / "scripts/study.py"),
        preparation_script=digest(Path(__file__).parents[1] / "scripts/prepare.py"),
        selection=digest(selected),
        smoke=True,
        cpu_test=False,
        package="0.5.0",
        revision="a" * 40,
    )
    (lane / "identity.json").write_text(json.dumps(identity))
    (lane / "done.json").write_text(json.dumps(dict(case=case)))
    (lane / "run_provenance.json").write_text(json.dumps(dict(runtime={})))
    with pytest.raises(ValueError, match="identity"):
        collect(tmp_path / "absent", CONFIG, selected, tmp_path / "out")
    identity["smoke"] = False
    (lane / "identity.json").write_text(json.dumps(identity))
    metrics = [
        dict(target=t, fraction=f, mode=m, rmse=1.0, mae=0.5, standardized_rmse=0.1, val_mse=0.01)
        for t in ["Dielectric total", "Bulk modulus", "Shear modulus", "Piezoelectric max"]
        for f in raw["study"]["fractions"]
        for m in ["ridge", "full"]
    ]
    result = dict(case=case, source=dict(steps=0), metrics=metrics)
    (lane / "done.json").write_text(json.dumps(result))
    (lane / "run_provenance.json").write_text(
        json.dumps(
            dict(
                runtime=dict(
                    device="cuda",
                    sif_sha256=raw["study"]["sif_sha256"],
                    image_revision=raw["study"]["image_revision"],
                    package_path="/opt/venv/lib/site-packages/foundation_model/__init__.py",
                )
            )
        )
    )
    audit = collect(tmp_path / "absent", CONFIG, selected, tmp_path / "out")
    assert audit["complete_lanes"] == 1 and audit["metrics"] == 24 and audit["partial"]
    result["metrics"][0]["rmse"] = float("nan")
    (lane / "done.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Nonfinite"):
        collect(tmp_path / "absent", CONFIG, selected, tmp_path / "out")


def test_hierarchical_pairing_and_incomplete_intervals():
    index = pd.MultiIndex.from_product([[1, 2, 3], [20, 21, 22]], names=["split_seed", "seed"])
    values = pd.Series([-1.0] * 9, index=index)
    r = paired_summary(values, [1, 2, 3], [20, 21, 22])
    assert r["mean"] == r["lo95"] == r["hi95"] == -1 and r["all_split_means_negative"]
    r = paired_summary(values.iloc[:3], [1, 2, 3], [20, 21, 22])
    assert r["partial"] and r["lo95"] is None and r["n_splits"] == 1
    with pytest.raises(ValueError):
        paired_summary(values * float("nan"), [1, 2, 3], [20, 21, 22])


def test_equal_count_source_sets_are_paired_directly(tmp_path):
    raw = tomllib.loads(CONFIG.read_text())
    raw["study"]["split_seeds"] = [1, 2, 3]
    names = [c for c, v in raw["study"]["source_sets"].items() if len(v) in [1, 3]]
    rows = [
        dict(
            input="kmd",
            arm="kmd_transformer",
            target="Bulk modulus",
            fraction=0.1,
            mode="ridge",
            split_seed=sp,
            seed=seed,
            steps=steps,
            condition=c,
            standardized_rmse=float(i + 1),
        )
        for i, c in enumerate(names)
        for sp in [1, 2, 3]
        for seed in [20, 21, 22]
        for steps in [6000, 24000]
    ]
    contrasts(pd.DataFrame(rows), raw, tmp_path)
    pairs = pd.read_csv(tmp_path / "paired_contrasts.csv")
    assert len(pairs) == 12 and not pairs.partial.any()
    q = pairs[(pairs.condition == "real1_energy") & (pairs.baseline == "real1_electronic")]
    assert len(q) == 2 and np.allclose(q["mean"], -1)
    assert (
        (pairs.condition.str.startswith("real1") & pairs.baseline.str.startswith("real1"))
        | (pairs.condition.str.startswith("real3") & pairs.baseline.str.startswith("real3"))
    ).all()


def test_family_interaction_requires_all_targets_and_pairs(tmp_path):
    raw = tomllib.loads(CONFIG.read_text())
    raw["study"]["split_seeds"] = [1, 2, 3]
    records = []
    for sp in [1, 2, 3]:
        for seed in [20, 21, 22]:
            for target in ["Dielectric total", "Bulk modulus", "Shear modulus", "Piezoelectric max"]:
                for arm in ["kmd_transformer", "kmd_no_attention", "kmd_mlp"]:
                    for condition, steps in [("random", 0), ("real12", 6000), ("real12", 24000)]:
                        value = 2 if condition == "random" else (1 if arm == "kmd_transformer" else 1.5)
                        records.append(
                            dict(
                                input="kmd",
                                target=target,
                                fraction=0.1,
                                mode="full",
                                arm=arm,
                                condition=condition,
                                steps=steps,
                                split_seed=sp,
                                seed=seed,
                                standardized_rmse=value,
                            )
                        )
    contrasts(pd.DataFrame(records).sample(frac=1, random_state=3), raw, tmp_path)
    q = pd.read_csv(tmp_path / "family_interactions.csv")
    assert len(q) == 4 and np.allclose(q["mean"], -0.5) and not q.partial.any()
