from pathlib import Path
import textwrap

import pytest

from foundation_model.models.model_config import OptimizerConfig
from foundation_model.scripts.dynamic_task_suite import (
    DynamicTaskSuiteRunner,
    SuiteConfig,
    parse_arguments,
)


def _make_suite_config(tmp_path: Path, **overrides) -> SuiteConfig:
    base_kwargs = {
        "descriptor_path": tmp_path / "descriptors.parquet",
        "pretrain_data_path": tmp_path / "pretrain.parquet",
        "finetune_data_path": tmp_path / "finetune.parquet",
        "output_dir": tmp_path / "outputs",
    }
    base_kwargs.update(overrides)
    return SuiteConfig(**base_kwargs)


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
        "--disable-deposit-layer",
    ]

    suite_config = parse_arguments(args)

    assert suite_config.head_lr == pytest.approx(5e-4)
    assert suite_config.use_deposit_layer is False


def test_parse_arguments_enable_deposit_layer(tmp_path: Path):
    args = [
        "--descriptor-path",
        str(tmp_path / "descriptors.parquet"),
        "--pretrain-data-path",
        str(tmp_path / "pretrain.parquet"),
        "--finetune-data-path",
        str(tmp_path / "finetune.parquet"),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--enable-deposit-layer",
    ]

    suite_config = parse_arguments(args)

    assert suite_config.use_deposit_layer is True


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
            use_deposit_layer = false
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
    assert suite_config.use_deposit_layer is False
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
            use_deposit_layer = false
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
            "--enable-deposit-layer",
        ]
    )

    assert suite_config.head_hidden_dim == 64
    assert suite_config.head_lr == pytest.approx(5e-4)
    assert suite_config.use_deposit_layer is True


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

    def test_index_calculation_for_ranks(self):
        """Test that each rank gets correct dataset indices."""
        total_size = 100
        world_size = 3

        # Simulate index calculation for each rank
        for rank in range(world_size):
            indices_for_rank = list(range(rank, total_size, world_size))

            # Verify rank 0 gets: 0, 3, 6, 9, ...
            # Verify rank 1 gets: 1, 4, 7, 10, ...
            # Verify rank 2 gets: 2, 5, 8, 11, ...
            assert all(idx % world_size == rank for idx in indices_for_rank), (
                f"Rank {rank} should only get indices where idx % world_size == rank"
            )

            # Verify expected count
            expected_count = (total_size + world_size - 1 - rank) // world_size
            assert len(indices_for_rank) == expected_count, (
                f"Rank {rank} should have {expected_count} indices, got {len(indices_for_rank)}"
            )

    def test_index_mapping_with_uneven_distribution(self):
        """Test index mapping when dataset size is not divisible by world_size."""
        total_size = 101  # 101 % 3 = 2
        world_size = 3

        # Collect all indices from all ranks
        all_indices = []
        samples_per_rank = []

        for rank in range(world_size):
            indices_for_rank = list(range(rank, total_size, world_size))

            # DistributedSampler pads to make equal sizes
            num_samples = (total_size + world_size - 1) // world_size  # Ceiling division
            if len(indices_for_rank) < num_samples:
                # Pad with indices from the beginning
                padding_needed = num_samples - len(indices_for_rank)
                indices_for_rank.extend(indices_for_rank[:padding_needed])

            samples_per_rank.append(len(indices_for_rank))
            all_indices.extend(indices_for_rank)

        # All ranks should have equal samples
        assert all(count == samples_per_rank[0] for count in samples_per_rank), (
            "All ranks should process equal number of samples (with padding)"
        )

        # After deduplication, should have exactly total_size unique indices
        unique_indices = set(all_indices)
        assert len(unique_indices) == total_size, (
            f"After deduplication, should have {total_size} unique samples, got {len(unique_indices)}"
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

        for rank in range(world_size):
            indices = list(range(rank, total_samples, world_size))

            # Add padding to make equal sizes
            num_samples = (total_samples + world_size - 1) // world_size
            if len(indices) < num_samples:
                padding_needed = num_samples - len(indices)
                indices.extend(indices[:padding_needed])

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
