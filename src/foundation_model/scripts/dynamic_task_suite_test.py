import textwrap
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pandas.testing as pdt
import pytest

from foundation_model.models.model_config import OptimizerConfig
from foundation_model.scripts.dynamic_task_suite import (
    DynamicTaskSuiteRunner,
    SuiteConfig,
    parse_arguments,
)


def _make_suite_config(tmp_path: Path, **overrides) -> SuiteConfig:
    base_kwargs: dict[str, Any] = {
        "descriptor_path": tmp_path / "descriptors.parquet",
        "pretrain_data_path": tmp_path / "pretrain.parquet",
        "finetune_data_path": tmp_path / "finetune.parquet",
        "output_dir": tmp_path / "outputs",
    }
    base_kwargs.update(overrides)
    return SuiteConfig(**base_kwargs)


class DummyScaler:
    def __init__(self, label: str):
        self.label = label

    def inverse_transform(self, values):  # pragma: no cover - simple passthrough
        return values


def test_build_regression_task_with_head_lr(tmp_path: Path):
    config = _make_suite_config(tmp_path, head_lr=2e-3)
    runner = DynamicTaskSuiteRunner(config)

    task_config = runner._build_regression_task("density", "density_column")

    assert task_config.optimizer is not None
    assert isinstance(task_config.optimizer, OptimizerConfig)
    assert task_config.optimizer.lr == pytest.approx(2e-3)
    assert task_config.optimizer.weight_decay == pytest.approx(1e-5)
    assert task_config.dims[0] == runner.encoder_latent_dim
    assert task_config.dims[-1] == 1


def test_build_regression_task_without_head_lr(tmp_path: Path):
    config = _make_suite_config(tmp_path)
    runner = DynamicTaskSuiteRunner(config)

    task_config = runner._build_regression_task("density", "density_column")

    assert task_config.optimizer is None
    assert task_config.dims[0] == runner.encoder_latent_dim


def test_parse_arguments_respects_overrides(tmp_path: Path):
    args = [
        "--descriptor-path",
        str(tmp_path / "descriptors.parquet"),
        "--pretrain-data-path",
        str(tmp_path / "pretrain.parquet"),
        "--finetune-data-path",
        str(tmp_path / "finetune.parquet"),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--head-lr",
        "5e-4",
    ]

    suite_config = parse_arguments(args)

    assert suite_config.head_lr == pytest.approx(5e-4)


def test_parse_arguments_from_toml_config(tmp_path: Path):
    config_path = tmp_path / "suite.toml"
    descriptor_path = tmp_path / "desc.parquet"
    pretrain_path = tmp_path / "pre.parquet"
    finetune_path = tmp_path / "fin.parquet"
    output_dir = tmp_path / "out"
    config_path.write_text(
        textwrap.dedent(
            f"""
            descriptor_path = "{descriptor_path}"
            pretrain_data_path = "{pretrain_path}"
            finetune_data_path = "{finetune_path}"
            output_dir = "{output_dir}"
            head_lr = 0.0005
            shared_block_dims = [190, 256, 128]
            encoder_config = {{ type = "transformer", d_model = 256, nhead = 4 }}
            pretrain_tasks = ["density", "Rg"]
            finetune_tasks = ["density"]
            """
        ).strip(),
        encoding="utf-8",
    )

    suite_config = parse_arguments(["--config-file", str(config_path)])

    assert suite_config.descriptor_path == descriptor_path
    assert suite_config.head_lr == pytest.approx(5e-4)
    assert suite_config.pretrain_tasks == ["density", "Rg"]
    assert suite_config.finetune_tasks == ["density"]

    # encoder_config is a dict when loaded from TOML
    assert isinstance(suite_config.encoder_config, dict)
    assert suite_config.encoder_config["type"] == "transformer"
    assert suite_config.encoder_config["d_model"] == 256


def test_config_file_overridden_by_cli(tmp_path: Path):
    config_path = tmp_path / "suite.toml"
    descriptor_path = tmp_path / "desc.parquet"
    pretrain_path = tmp_path / "pre.parquet"
    finetune_path = tmp_path / "fin.parquet"
    output_dir = tmp_path / "out"
    config_path.write_text(
        textwrap.dedent(
            f"""
            descriptor_path = "{descriptor_path}"
            pretrain_data_path = "{pretrain_path}"
            finetune_data_path = "{finetune_path}"
            output_dir = "{output_dir}"
            head_hidden_dim = 128
            head_lr = 0.0005
            """
        ).strip(),
        encoding="utf-8",
    )

    suite_config = parse_arguments(
        [
            "--config-file",
            str(config_path),
            "--head-hidden-dim",
            "64",
        ]
    )

    assert suite_config.head_hidden_dim == 64
    assert suite_config.head_lr == pytest.approx(5e-4)


def test_load_datasets_uses_phase_specific_descriptors(tmp_path: Path):
    pre_desc_path = tmp_path / "pre_desc.parquet"
    fin_desc_path = tmp_path / "fin_desc.parquet"
    pretrain_descriptor = pd.DataFrame({"feat": [1.0, 2.0]}, index=[0, 1])
    finetune_descriptor = pd.DataFrame({"feat": [9.0, 8.0]}, index=[10, 11])
    pretrain_targets = pd.DataFrame({"density": [0.2, 0.3]}, index=[0, 1])
    finetune_targets = pd.DataFrame({"Tg": [100.0, 110.0]}, index=[10, 11])
    pre_desc_path.parent.mkdir(parents=True, exist_ok=True)
    pretrain_descriptor.to_parquet(pre_desc_path)
    finetune_descriptor.to_parquet(fin_desc_path)
    pretrain_targets_path = tmp_path / "pre_targets.parquet"
    finetune_targets_path = tmp_path / "fin_targets.parquet"
    pretrain_targets.to_parquet(pretrain_targets_path)
    finetune_targets.to_parquet(finetune_targets_path)

    config = _make_suite_config(
        tmp_path,
        descriptor_path=pre_desc_path,
        pretrain_descriptor_path=pre_desc_path,
        finetune_descriptor_path=fin_desc_path,
        pretrain_data_path=pretrain_targets_path,
        finetune_data_path=finetune_targets_path,
        pretrain_tasks=["density"],
        finetune_tasks=["Tg"],
        use_normalized_targets=False,
    )

    runner = DynamicTaskSuiteRunner(config)
    runner._load_datasets()

    assert runner.pretrain_features is not None
    assert runner.finetune_features is not None
    pdt.assert_frame_equal(runner.pretrain_features, pretrain_descriptor)
    pdt.assert_frame_equal(runner.finetune_features, finetune_descriptor)


def test_load_datasets_accepts_phase_specific_scalers(tmp_path: Path):
    pre_desc_path = tmp_path / "pre_desc.parquet"
    fin_desc_path = tmp_path / "fin_desc.parquet"
    pretrain_descriptor = pd.DataFrame({"feat": [1.0, 2.0]}, index=[0, 1])
    finetune_descriptor = pd.DataFrame({"feat": [9.0, 8.0]}, index=[10, 11])
    pretrain_targets = pd.DataFrame({"density (normalized)": [0.2, 0.3]}, index=[0, 1])
    finetune_targets = pd.DataFrame({"Tg (normalized)": [100.0, 110.0]}, index=[10, 11])
    pretrain_descriptor.to_parquet(pre_desc_path)
    finetune_descriptor.to_parquet(fin_desc_path)
    pretrain_targets_path = tmp_path / "pre_targets.parquet"
    finetune_targets_path = tmp_path / "fin_targets.parquet"
    pretrain_targets.to_parquet(pretrain_targets_path)
    finetune_targets.to_parquet(finetune_targets_path)

    global_scaler_path = tmp_path / "scalers.joblib"
    pretrain_scaler_path = tmp_path / "pre_scalers.joblib"
    finetune_scaler_path = tmp_path / "fin_scalers.joblib"
    joblib.dump({"density": DummyScaler("global"), "Tg": DummyScaler("global")}, global_scaler_path)
    joblib.dump({"density": DummyScaler("pre")}, pretrain_scaler_path)
    joblib.dump({"Tg": DummyScaler("fin")}, finetune_scaler_path)

    config = _make_suite_config(
        tmp_path,
        descriptor_path=pre_desc_path,
        pretrain_descriptor_path=pre_desc_path,
        finetune_descriptor_path=fin_desc_path,
        pretrain_data_path=pretrain_targets_path,
        finetune_data_path=finetune_targets_path,
        scaler_path=global_scaler_path,
        pretrain_scaler_path=pretrain_scaler_path,
        finetune_scaler_path=finetune_scaler_path,
        pretrain_tasks=["density"],
        finetune_tasks=["Tg"],
    )

    runner = DynamicTaskSuiteRunner(config)
    runner._load_datasets()

    assert isinstance(runner.property_scalers["density"], DummyScaler)
    assert runner.property_scalers["density"].label == "pre"
    assert isinstance(runner.property_scalers["Tg"], DummyScaler)
    assert runner.property_scalers["Tg"].label == "fin"


def test_finetune_datamodule_retains_split_column(tmp_path: Path):
    config = _make_suite_config(tmp_path)
    runner = DynamicTaskSuiteRunner(config)

    index = pd.Index(["a", "b", "c", "d"], name="sample")
    runner.finetune_features = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0]}, index=index)
    runner.finetune_targets = pd.DataFrame(
        {
            "Density (normalized)": [0.1, 0.2, 0.3, 0.4],
            "split": ["train", "test", "val", "test"],
        },
        index=index,
    )
    runner.finetune_target_columns = {"Density": "Density (normalized)"}

    dm = runner._build_finetune_datamodule("Density")
    dm.setup(stage="test")

    assert set(dm.test_idx) == {"b", "d"}


# ============================================================================
# Tests for distributed prediction and result ordering
# ============================================================================


class TestPredictionOrdering:
    """Test prediction result ordering in distributed environments."""

    def test_prediction_rows_deduplication(self):
        """Test that duplicate predictions are correctly removed."""
        # Simulate predictions from multiple ranks with duplicates (due to padding)
        prediction_rows = [
            {"task": "task1", "dataset_index": 0, "actual": 1.0, "predicted": 1.1, "sample_index": 0},
            {"task": "task1", "dataset_index": 1, "actual": 2.0, "predicted": 2.1, "sample_index": 1},
            {"task": "task1", "dataset_index": 2, "actual": 3.0, "predicted": 3.1, "sample_index": 2},
            # Duplicates from padding
            {"task": "task1", "dataset_index": 0, "actual": 1.0, "predicted": 1.1, "sample_index": 3},
            {"task": "task1", "dataset_index": 1, "actual": 2.0, "predicted": 2.1, "sample_index": 4},
        ]

        # Simulate deduplication logic
        seen_keys = set()
        deduplicated_rows = []
        for row in prediction_rows:
            task_name = str(row.get("task"))
            dataset_idx = int(row.get("dataset_index", -1))
            key = (task_name, dataset_idx)

            if key not in seen_keys:
                seen_keys.add(key)
                deduplicated_rows.append(row)

        # Should have exactly 3 unique samples
        assert len(deduplicated_rows) == 3, f"Expected 3 unique samples, got {len(deduplicated_rows)}"

        # Verify correct samples were kept (first occurrence)
        assert deduplicated_rows[0]["dataset_index"] == 0
        assert deduplicated_rows[1]["dataset_index"] == 1
        assert deduplicated_rows[2]["dataset_index"] == 2

    def test_prediction_rows_sorting(self):
        """Test that predictions are sorted by task and dataset_index."""
        # Simulated predictions in random order
        prediction_rows = [
            {"task": "task2", "dataset_index": 1, "actual": 5.0, "predicted": 5.1},
            {"task": "task1", "dataset_index": 2, "actual": 3.0, "predicted": 3.1},
            {"task": "task1", "dataset_index": 0, "actual": 1.0, "predicted": 1.1},
            {"task": "task2", "dataset_index": 0, "actual": 4.0, "predicted": 4.1},
            {"task": "task1", "dataset_index": 1, "actual": 2.0, "predicted": 2.1},
        ]

        # Sort by task name, then by dataset_index
        sorted_rows = sorted(prediction_rows, key=lambda r: (str(r.get("task")), int(r.get("dataset_index", 0))))

        # Verify ordering
        expected_order = [
            ("task1", 0),
            ("task1", 1),
            ("task1", 2),
            ("task2", 0),
            ("task2", 1),
        ]

        actual_order = [(row["task"], row["dataset_index"]) for row in sorted_rows]
        assert actual_order == expected_order, f"Expected {expected_order}, got {actual_order}"

    def test_sample_index_reassignment(self):
        """Test that sample_index is reassigned sequentially per task after sorting."""
        # Sorted predictions
        prediction_rows = [
            {"task": "task1", "dataset_index": 0, "actual": 1.0, "predicted": 1.1},
            {"task": "task1", "dataset_index": 1, "actual": 2.0, "predicted": 2.1},
            {"task": "task1", "dataset_index": 2, "actual": 3.0, "predicted": 3.1},
            {"task": "task2", "dataset_index": 0, "actual": 4.0, "predicted": 4.1},
            {"task": "task2", "dataset_index": 1, "actual": 5.0, "predicted": 5.1},
        ]

        # Reassign sample_index per task
        sample_counters = {}
        for row in prediction_rows:
            task_name = str(row.get("task"))
            current_index = sample_counters.get(task_name, 0)
            row["sample_index"] = current_index
            sample_counters[task_name] = current_index + 1

        # Verify sample_index values
        assert prediction_rows[0]["sample_index"] == 0  # task1, first sample
        assert prediction_rows[1]["sample_index"] == 1  # task1, second sample
        assert prediction_rows[2]["sample_index"] == 2  # task1, third sample
        assert prediction_rows[3]["sample_index"] == 0  # task2, first sample
        assert prediction_rows[4]["sample_index"] == 1  # task2, second sample


class TestDistributedIndexMapping:
    """Test dataset index mapping for distributed prediction."""

    @staticmethod
    def _distributed_sampler_indices(dataset_size: int, world_size: int, rank: int) -> list[int]:
        """Helper that mirrors torch DistributedSampler output for comparison."""
        from torch.utils.data.distributed import DistributedSampler

        dataset: Any = list(range(dataset_size))
        sampler: Any = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        return list(iter(sampler))

    def test_index_calculation_for_ranks(self):
        """Test that each rank gets correct dataset indices."""
        dataset_size = 100
        world_size = 3
        num_samples = (dataset_size + world_size - 1) // world_size
        total_size = num_samples * world_size
        base_indices = list(range(dataset_size))
        if len(base_indices) < total_size:
            base_indices.extend(base_indices[: total_size - len(base_indices)])

        # Simulate index calculation for each rank
        for rank in range(world_size):
            indices_for_rank = base_indices[rank:total_size:world_size]
            assert len(indices_for_rank) == num_samples, (
                f"Rank {rank} should have {num_samples} samples, got {len(indices_for_rank)}"
            )

            expected_from_sampler = self._distributed_sampler_indices(dataset_size, world_size, rank)
            assert indices_for_rank == expected_from_sampler, (
                f"Rank {rank} indices mismatch DistributedSampler: {indices_for_rank} vs {expected_from_sampler}"
            )

    def test_index_mapping_with_uneven_distribution(self):
        """Test index mapping when dataset size is not divisible by world_size."""
        dataset_size = 101  # 101 % 3 = 2
        world_size = 3

        # Collect all indices from all ranks
        all_indices = []
        samples_per_rank = []

        for rank in range(world_size):
            num_samples = (dataset_size + world_size - 1) // world_size
            total_size = num_samples * world_size
            base_indices = list(range(dataset_size))
            if len(base_indices) < total_size:
                base_indices.extend(base_indices[: total_size - len(base_indices)])
            indices_for_rank = base_indices[rank:total_size:world_size]

            samples_per_rank.append(len(indices_for_rank))
            all_indices.extend(indices_for_rank)

        # All ranks should have equal samples
        assert all(count == samples_per_rank[0] for count in samples_per_rank), (
            "All ranks should process equal number of samples (with padding)"
        )

        # After deduplication, should cover exactly dataset_size unique indices
        unique_indices = set(all_indices)
        assert len(unique_indices) == dataset_size, (
            f"After deduplication, should have {dataset_size} unique samples, got {len(unique_indices)}"
        )

    def test_batch_index_slicing(self):
        """Test that batch indices are correctly sliced from rank indices."""
        indices_for_rank = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]  # Rank 0 with world_size=3
        global_sample_idx = 0
        batch_size = 4

        # First batch
        batch_indices = indices_for_rank[global_sample_idx : global_sample_idx + batch_size]
        assert batch_indices == [0, 3, 6, 9], f"First batch indices incorrect: {batch_indices}"

        # Second batch
        global_sample_idx += batch_size
        batch_indices = indices_for_rank[global_sample_idx : global_sample_idx + batch_size]
        assert batch_indices == [12, 15, 18, 21], f"Second batch indices incorrect: {batch_indices}"

        # Last batch (partial)
        global_sample_idx += batch_size
        batch_indices = indices_for_rank[global_sample_idx : global_sample_idx + batch_size]
        assert batch_indices == [24, 27], f"Last batch indices incorrect: {batch_indices}"


class TestEndToEndPredictionFlow:
    """Test the complete prediction flow with deduplication and sorting."""

    def test_complete_flow_with_3_gpus(self):
        """Test complete prediction flow simulating 3 GPUs."""
        total_samples = 10
        world_size = 3

        # Simulate predictions from each rank
        # Each rank processes different samples but with padding
        rank_predictions = {}

        num_samples = (total_samples + world_size - 1) // world_size
        total_size = num_samples * world_size
        base_indices = list(range(total_samples))
        if len(base_indices) < total_size:
            base_indices.extend(base_indices[: total_size - len(base_indices)])

        for rank in range(world_size):
            indices = base_indices[rank:total_size:world_size]

            # Simulate predictions for this rank
            predictions = []
            for idx in indices:
                predictions.append(
                    {
                        "task": "task1",
                        "dataset_index": idx,
                        "actual": float(idx),
                        "predicted": float(idx) + 0.1,
                        "run": 1,
                        "phase": "test",
                        "stage": 1,
                    }
                )

            rank_predictions[rank] = predictions

        # Gather all predictions (simulating all_gather_object)
        all_predictions = []
        for rank in range(world_size):
            all_predictions.extend(rank_predictions[rank])

        # Deduplication
        seen_keys = set()
        deduplicated_rows = []
        for row in all_predictions:
            task_name = str(row.get("task"))
            dataset_idx = int(row.get("dataset_index", -1))
            key = (task_name, dataset_idx)

            if key not in seen_keys:
                seen_keys.add(key)
                deduplicated_rows.append(row)

        # Sorting
        deduplicated_rows.sort(key=lambda r: (str(r.get("task")), int(r.get("dataset_index", 0))))

        # Reassign sample_index
        sample_counters = {}
        for row in deduplicated_rows:
            task_name = str(row.get("task"))
            current_index = sample_counters.get(task_name, 0)
            row["sample_index"] = current_index
            sample_counters[task_name] = current_index + 1

        # Verify final results
        assert len(deduplicated_rows) == total_samples, (
            f"Expected {total_samples} final predictions, got {len(deduplicated_rows)}"
        )

        # Verify ordering
        for i, row in enumerate(deduplicated_rows):
            assert row["dataset_index"] == i, f"Row {i} should have dataset_index={i}, got {row['dataset_index']}"
            assert row["sample_index"] == i, f"Row {i} should have sample_index={i}, got {row['sample_index']}"
            assert row["actual"] == float(i), f"Row {i} should have actual={i}, got {row['actual']}"

        # Build aggregated metrics from deduplicated rows
        aggregated = {}
        for row in deduplicated_rows:
            task = str(row["task"])
            entry = aggregated.setdefault(task, {"preds": [], "targets": []})
            entry["preds"].append(row["predicted"])
            entry["targets"].append(row["actual"])

        assert aggregated["task1"]["targets"] == [float(i) for i in range(total_samples)]
        assert aggregated["task1"]["preds"] == [float(i) + 0.1 for i in range(total_samples)]

    def test_metrics_deduplicate_padding(self):
        """Ensure metrics aggregation ignores duplicates introduced by padding."""
        total_samples = 5
        world_size = 2
        num_samples = (total_samples + world_size - 1) // world_size
        total_size = num_samples * world_size

        base_indices = list(range(total_samples))
        if len(base_indices) < total_size:
            base_indices.extend(base_indices[: total_size - len(base_indices)])

        all_rank_predictions = []
        for rank in range(world_size):
            indices_for_rank = base_indices[rank:total_size:world_size]
            predictions = []
            for idx in indices_for_rank:
                predictions.append(
                    {
                        "task": "task1",
                        "dataset_index": idx,
                        "actual": float(idx),
                        "predicted": float(idx) + 0.1,
                    }
                )
            all_rank_predictions.extend(predictions)

        seen = set()
        deduplicated_rows = []
        for row in all_rank_predictions:
            key = (row["task"], row["dataset_index"])
            if key in seen:
                continue
            seen.add(key)
            deduplicated_rows.append(row)

        deduplicated_rows.sort(key=lambda r: (str(r["task"]), int(r["dataset_index"])))

        aggregated = {}
        for row in deduplicated_rows:
            entry = aggregated.setdefault(row["task"], {"preds": [], "targets": []})
            entry["preds"].append(row["predicted"])
            entry["targets"].append(row["actual"])

        assert len(deduplicated_rows) == total_samples
        assert aggregated["task1"]["targets"] == [float(i) for i in range(total_samples)]
        assert aggregated["task1"]["preds"] == [float(i) + 0.1 for i in range(total_samples)]


class TestDistributedPredictionFiles:
    """Validate per-rank prediction file merging helpers."""

    def test_merge_rank_prediction_rows(self, tmp_path):
        output_dir = tmp_path
        rows_rank0 = [
            {"task": "task1", "dataset_index": 0, "actual": 0.0, "predicted": 0.1},
            {"task": "task1", "dataset_index": 3, "actual": 3.0, "predicted": 3.1},
        ]
        rows_rank1 = [
            {"task": "task1", "dataset_index": 1, "actual": 1.0, "predicted": 1.1},
            {"task": "task1", "dataset_index": 2, "actual": 2.0, "predicted": 2.1},
        ]
        pd.DataFrame(rows_rank0).to_parquet(output_dir / "predictions_rank000.parquet", index=False)
        pd.DataFrame(rows_rank1).to_parquet(output_dir / "predictions_rank001.parquet", index=False)

        merged = DynamicTaskSuiteRunner._merge_rank_prediction_rows(output_dir, world_size=2)

        assert len(merged) == 4
        assert all((output_dir / f"predictions_rank00{i}.parquet").exists() is False for i in range(2))
