# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared engine internals (currently the replay resampling callback)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.model_config import RegressionTaskConfig
from foundation_model.workflows._engine import ReplayResampleCallback


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


def test_replay_resample_callback_accepts_non_persistent_workers() -> None:
    dm = _masked_datamodule()
    assert dm.train_dataset is not None
    epoch0 = _dense_mask(dm, "task_a").clone()
    loader = DataLoader(dm.train_dataset, batch_size=4, num_workers=1)  # persistent_workers=False
    trainer = SimpleNamespace(datamodule=dm, current_epoch=1, train_dataloader=loader)
    ReplayResampleCallback().on_train_epoch_start(trainer, None)
    assert not torch.equal(epoch0, _dense_mask(dm, "task_a"))  # redraw went through
