import json

import pandas as pd

from benchmark import DATE, SOURCE_TASKS, TARGET_TASKS, file_hash
from make_smoke_data import make_smoke


def test_smoke_preserves_nested_budgets_and_updates_counts_hashes(tmp_path):
    source_dir, output = tmp_path / "prepared", tmp_path / "smoke"
    source_dir.mkdir()
    source = pd.DataFrame({"composition": [f"id{i}" for i in range(60)], "split": ["train"] * 50 + ["val"] * 10})
    for name in SOURCE_TASKS:
        source[name] = 1.0
    source_file = f"source_{DATE}.parquet"
    source.to_parquet(source_dir / source_file, index=False)
    (source_dir / f"source_scalers_{DATE}.joblib").write_bytes(b"source-scaler")
    target = pd.DataFrame(
        {"composition": [f"id{i}" for i in range(80)], "split": ["train"] * 60 + ["val"] * 10 + ["test"] * 10}
    )
    entries = []
    for task in TARGET_TASKS:
        for seed in (0, 1, 2):
            for fraction in (0.1, 1):
                frame = target if fraction == 1 else target.loc[target["split"].ne("train") | (target.index < 6)]
                stem = f"{task}_{seed}_{fraction}"
                frame.to_parquet(source_dir / f"{stem}.parquet", index=False)
                (source_dir / f"{stem}.joblib").write_bytes(b"target-scaler")
                entries.append(
                    dict(
                        task=task,
                        seed=seed,
                        fraction=fraction,
                        file=f"{stem}.parquet",
                        scaler=f"{stem}.joblib",
                        scaler_sha256=file_hash(source_dir / f"{stem}.joblib"),
                    )
                )
    manifest = {"source": {"file": source_file}, "targets": entries}
    (source_dir / f"manifest_{DATE}.json").write_text(json.dumps(manifest))
    make_smoke(source_dir, output)
    result = json.loads((output / f"manifest_{DATE}.json").read_text())
    assert result["functional_smoke"] and result["smoke_batch_size"] == 16
    assert result["source"]["sha256"] == file_hash(output / source_file)
    assert result["source"]["counts"] == source["split"].value_counts().to_dict()
    for entry in result["targets"]:
        frame = pd.read_parquet(output / entry["file"])
        assert entry["sha256"] == file_hash(output / entry["file"])
        assert entry["scaler_sha256"] == file_hash(output / entry["scaler"])
        assert entry["counts"] == frame["split"].value_counts().to_dict()
        assert set(entry["training_compositions"]) == set(frame.loc[frame["split"].eq("train"), "composition"])
    low, high = result["targets"][:2]
    assert set(low["training_compositions"]) < set(high["training_compositions"])
    low_frame, high_frame = pd.read_parquet(output / low["file"]), pd.read_parquet(output / high["file"])
    pd.testing.assert_frame_equal(
        low_frame.loc[low_frame["split"].ne("train")].reset_index(drop=True),
        high_frame.loc[high_frame["split"].ne("train")].reset_index(drop=True),
    )
