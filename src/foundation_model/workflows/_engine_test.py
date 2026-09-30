# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared engine internals (currently the replay resampling callback)."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import joblib
from lightning import Trainer
import numpy as np
import pandas as pd

import pytest
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.model_config import RegressionTaskConfig
from ._engine import (
    ReplayResampleCallback,
    DropLastTrainCompoundDataModule,
    as_float_array,
    build_empty_model,
    build_head_config,
    evaluate_task,
)
from ._sections import ModelSectionConfig, TrainingSectionConfig
from .recording import RunRecorder
from .task_catalog import TaskCatalog, build_task_catalog_config


@pytest.mark.parametrize("cell", [None, np.nan, pd.NA, np.array(np.nan), np.array([np.nan])])
def test_missing_sequence_is_empty(cell: object) -> None:
    assert as_float_array(cell).size == 0


@pytest.mark.parametrize("cell", [[0, 1], np.array([0, 1]), "[0, 1]"])
def test_sequence_cells_convert_without_scalar_truth_test(cell: object) -> None:
    np.testing.assert_array_equal(as_float_array(cell), [0, 1])


def test_kr_evaluation_exports_and_plots_both_values_in_original_units(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "curve.parquet"
    pd.DataFrame({"composition": ["Fe2O3"], "y": [[0.0, 1.0]], "t": [[6.0, 7.0]], "split": ["test"]}).to_parquet(
        dataset
    )
    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump(StandardScaler().fit([[100.0], [200.0]]), scaler_path)
    catalog = TaskCatalog(
        build_task_catalog_config(
            {
                "descriptor": {"kind": "kmd", "n_grids": 4},
                "datasets": {"curve": {"path": str(dataset)}},
                "tasks": [
                    {
                        "name": "curve",
                        "kind": "kernel_regression",
                        "dataset": "curve",
                        "column": "y",
                        "t_column": "t",
                        "scaler": {"path": str(scaler_path)},
                    }
                ],
            }
        )
    )
    model_config = ModelSectionConfig(latent_dim=8, encoder_hidden_dims=[16], n_kernel=4)
    training = TrainingSectionConfig(max_epochs=1, accelerator="cpu")
    model = build_empty_model(catalog, model_config, training)
    model.add_task(build_head_config(catalog, model_config, training, "curve"))
    for parameter in model.task_heads["curve"].parameters():
        parameter.data.zero_()
    plotted = {}

    def capture_plot(*args, **kwargs):
        plotted["true"] = args[2]
        plotted["pred"] = args[3]
        return None

    monkeypatch.setattr("foundation_model.workflows._engine.plots.plot_kr_sequences", capture_plot)
    recorder = RunRecorder(tmp_path / "out")
    step_dir = recorder.paths.training / "step01_curve"
    try:
        metrics = evaluate_task(model, catalog, "curve", recorder, step_dir, is_new=True, test_keys=None)
    finally:
        recorder.close()
    result = pd.read_parquet(step_dir / "curve_pred.parquet")
    np.testing.assert_allclose(result.true, [150, 200])
    np.testing.assert_allclose(result.pred, [150, 150])
    np.testing.assert_allclose(plotted["true"][0], [150, 200])
    np.testing.assert_allclose(plotted["pred"], [150, 150])
    assert metrics["mae"] == 25


def _masked_datamodule() -> CompoundDataModule:
    comps = [f"m{i}" for i in range(20)]
    descriptors = pd.DataFrame({"f0": np.arange(20.0), "f1": np.ones(20)}, index=comps)
    frame = pd.DataFrame({"y": np.arange(20.0)}, index=comps)

    def descriptor_fn(compositions):
        present = [c for c in compositions if c in descriptors.index]
        return descriptors.loc[present]

    dm = CompoundDataModule(
        task_configs=[RegressionTaskConfig(name="task_a", data_column="y", dims=[2, 8, 1], task_masking_ratio=0.5)],
        descriptor_fn=descriptor_fn,
        task_frames={"task_a": frame},
        val_split=0.0,
        test_split=0.0,
        random_seed=42,
    )
    dm.setup("fit")
    return dm


def _dense_mask(dm: CompoundDataModule, task: str) -> torch.Tensor:
    assert dm.train_dataset is not None
    mask = dm.train_dataset.task_masks_dict[task]
    assert isinstance(mask, torch.Tensor)
    return mask


def test_replay_resample_callback_redraws_with_trainer_epoch() -> None:
    dm = _masked_datamodule()
    epoch0 = _dense_mask(dm, "task_a").clone()

    callback = ReplayResampleCallback()
    callback.on_train_epoch_start(SimpleNamespace(datamodule=dm, current_epoch=1), None)
    epoch1 = _dense_mask(dm, "task_a").clone()
    assert not torch.equal(epoch0, epoch1)

    # The callback draws the same mask the datamodule would for the same epoch.
    dm.resample_train_masks(epoch=1)
    assert torch.equal(epoch1, _dense_mask(dm, "task_a"))

    # Epoch 0 redraw restores the construction-time draw (idempotent first epoch).
    callback.on_train_epoch_start(SimpleNamespace(datamodule=dm, current_epoch=0), None)
    assert torch.equal(epoch0, _dense_mask(dm, "task_a"))


def test_replay_resample_callback_ignores_foreign_datamodule() -> None:
    callback = ReplayResampleCallback()
    callback.on_train_epoch_start(SimpleNamespace(datamodule=None, current_epoch=3), None)
    callback.on_train_epoch_start(SimpleNamespace(datamodule=object(), current_epoch=3), None)


def test_replay_resample_callback_rejects_persistent_workers() -> None:
    dm = _masked_datamodule()
    assert dm.train_dataset is not None
    # Constructing with num_workers=1 spawns no processes until iteration — safe in a test.
    loader = DataLoader(dm.train_dataset, batch_size=4, num_workers=1, persistent_workers=True)
    trainer = SimpleNamespace(datamodule=dm, current_epoch=0, train_dataloader=loader)
    with pytest.raises(ValueError, match="persistent_workers"):
        ReplayResampleCallback().on_train_epoch_start(trainer, None)


def test_drop_last_datamodule_propagates_persistent_workers() -> None:
    from foundation_model.workflows._engine import DropLastTrainCompoundDataModule

    comps = [f"m{i}" for i in range(20)]
    descriptors = pd.DataFrame({"f0": np.arange(20.0), "f1": np.ones(20)}, index=comps)

    def descriptor_fn(compositions):
        present = [c for c in compositions if c in descriptors.index]
        return descriptors.loc[present]

    dm = DropLastTrainCompoundDataModule(
        task_configs=[RegressionTaskConfig(name="task_a", data_column="y", dims=[2, 8, 1])],
        descriptor_fn=descriptor_fn,
        task_frames={"task_a": pd.DataFrame({"y": np.arange(20.0)}, index=comps)},
        val_split=0.0,
        test_split=0.0,
        num_workers=1,
        persistent_workers=True,
        pin_memory=False,
        prefetch_factor=3,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    assert loader is not None and loader.drop_last and loader.persistent_workers
    assert loader.pin_memory is False and loader.prefetch_factor == 3  # rebuild keeps tuning knobs


def test_fixed_budget_trainer_can_fit_without_validation_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "property.parquet"
    pd.DataFrame(
        {"composition": ["Fe2O3", "Al2O3", "NaCl"], "y": [0.0, 1.0, 2.0], "split": ["train", "train", "test"]}
    ).to_parquet(dataset)
    catalog = TaskCatalog(
        build_task_catalog_config(
            {
                "data": {"batch_size": 2, "val_split": 0.0, "test_split": 0.0},
                "descriptor": {"kind": "kmd", "n_grids": 4},
                "datasets": {"property": {"path": str(dataset)}},
                "tasks": [{"name": "property", "kind": "regression", "dataset": "property", "column": "y"}],
            }
        )
    )
    config = ModelSectionConfig(latent_dim=8, encoder_hidden_dims=[16])
    training = TrainingSectionConfig(max_epochs=1, accelerator="cpu")
    model = build_empty_model(catalog, config, training)
    model.add_task(build_head_config(catalog, config, training, "property"))
    datamodule = catalog.build_datamodule(["property"], datamodule_cls=DropLastTrainCompoundDataModule)
    trainer = Trainer(
        max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False
    )
    trainer.fit(model, datamodule=datamodule)
    assert trainer.current_epoch == 1
    assert datamodule.val_dataset is None
    assert datamodule.train_dataset is not None and len(datamodule.train_dataset) == 2


def test_replay_resample_callback_accepts_non_persistent_workers() -> None:
    dm = _masked_datamodule()
    assert dm.train_dataset is not None
    epoch0 = _dense_mask(dm, "task_a").clone()
    loader = DataLoader(dm.train_dataset, batch_size=4, num_workers=1)  # persistent_workers=False
    trainer = SimpleNamespace(datamodule=dm, current_epoch=1, train_dataloader=loader)
    ReplayResampleCallback().on_train_epoch_start(trainer, None)
    assert not torch.equal(epoch0, _dense_mask(dm, "task_a"))  # redraw went through


def test_build_empty_model_routes_the_loss_balancer_flag(tmp_path):
    """The link that was missing: [training] -> build_empty_model -> the model.

    Every other piece of uncertainty weighting was already implemented — registration on task
    activation, the objective term, inclusion in the main optimizer — but nothing carried a value
    from config to constructor, so the feature had never been switched on in any run. Asserting
    the route here is what stops it silently detaching again.
    """
    from foundation_model.workflows._engine import build_empty_model
    from foundation_model.workflows.task_catalog import TaskCatalog
    from foundation_model.workflows._sections import build_model_section, build_training_section

    # build_empty_model only reads descriptor_dim off the catalog; cast rather than
    # constructing a full TaskCatalog, which would need data files on disk.
    catalog = cast(TaskCatalog, SimpleNamespace(descriptor_dim=32))
    model_cfg = build_model_section({"latent_dim": 8, "encoder_hidden_dims": [16]})

    off = build_empty_model(catalog, model_cfg, build_training_section({}))
    on = build_empty_model(catalog, model_cfg, build_training_section({"learnable_loss_balancer": True}))

    assert off.enable_learnable_loss_balancer is False
    assert on.enable_learnable_loss_balancer is True


def _stub_catalog(descriptor_dim: int = 32):
    """Enough of a TaskCatalog for the model builders: they only read ``descriptor_dim`` and,
    for ``build_model_for_checkpoint``, ``build_task_config``. A real one would need data files."""
    from foundation_model.models.model_config import OptimizerConfig, RegressionTaskConfig, TaskType
    from foundation_model.workflows.task_catalog import TaskCatalog

    def build_task_config(name, *, latent_dim, lr, **_kwargs):
        return RegressionTaskConfig(
            name=name,
            type=TaskType.REGRESSION,
            data_column=name,
            dims=[latent_dim, 8, 1],
            optimizer=OptimizerConfig(lr=lr),
        )

    return cast(TaskCatalog, SimpleNamespace(descriptor_dim=descriptor_dim, build_task_config=build_task_config))


@pytest.mark.parametrize("task_names", [[], ["density"]])
def test_build_model_for_checkpoint_groups_agree_on_scheduler_policy(task_names):
    """Every parameter group of an inference-only model must share one scheduler policy.

    A single optimizer carries a single scheduler, so ``configure_optimizers`` rejects groups that
    disagree. This builder switches the scheduler off on the encoder and on every head it adds —
    but the AE head is auto-created by the constructor, never passes through that loop, and its
    ``optimizer = None`` reads back as the defaults, i.e. scheduler ON. Nothing calls
    ``configure_optimizers`` on this model today (predict/inverse build no Trainer), which is
    exactly why the disagreement would sit here unnoticed until something did.
    """
    from foundation_model.workflows._engine import AE_NAME, build_model_for_checkpoint
    from foundation_model.workflows._sections import build_model_section

    model_cfg = build_model_section({"latent_dim": 8, "encoder_hidden_dims": [16]})
    built = build_model_for_checkpoint(_stub_catalog(), model_cfg, task_names)

    assert built.task_configs_map[AE_NAME].optimizer is not None, "the AE head must carry a config"
    optimizer = built.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW), "no scheduler on an inference-only model"


def test_build_empty_model_groups_agree_on_a_non_default_scheduler_policy():
    """The training builder must stay policy-consistent under a customised [training.scheduler].

    All four groups draw their scheduler settings from that one block, so they cannot disagree —
    this pins that, including for the AE head, whose config the builder has to fill in by hand.
    """
    from foundation_model.workflows._engine import build_empty_model
    from foundation_model.workflows._sections import build_model_section, build_training_section

    model_cfg = build_model_section({"latent_dim": 8, "encoder_hidden_dims": [16]})
    training = build_training_section(
        {"scheduler": {"patience": 11, "factor": 0.25, "monitor": "train_final_loss_epoch", "min_lr": 1e-6}}
    )
    built = build_empty_model(_stub_catalog(), model_cfg, training)

    result = cast(dict, built.configure_optimizers())
    scheduler = result["lr_scheduler"]["scheduler"]
    assert scheduler.patience == 11 and scheduler.factor == 0.25
