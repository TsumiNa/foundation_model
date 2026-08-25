#!/usr/bin/env python3
"""Exercise an installed foundation-model image on CPU or a RIKYU GPU."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version

import torch

from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import MLPEncoderConfig, RegressionTaskConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-arch")
    args = parser.parse_args()

    architecture = platform.machine()
    if args.expected_arch and architecture != args.expected_arch:
        raise RuntimeError(f"Expected architecture {args.expected_arch!r}, got {architecture!r}")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlexibleMultiTaskModel(
        task_configs=[RegressionTaskConfig(name="smoke", data_column="smoke", dims=[4, 8, 1], norm=False)],
        encoder_config=MLPEncoderConfig(hidden_dims=[8, 16, 4], norm=False),
    ).to(device)
    inputs = torch.randn(4, 8, device=device)
    predictions = model(inputs)["smoke"]
    loss = predictions.square().mean()
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke loss: {loss.item()}")
    if device.type == "cuda":
        torch.cuda.synchronize()

    result = {
        "architecture": architecture,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "foundation_model": version("foundation-model"),
        "loss": loss.item(),
        "torch": torch.__version__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print("RIKYU_CONTAINER_SMOKE_PASS")


if __name__ == "__main__":
    main()
