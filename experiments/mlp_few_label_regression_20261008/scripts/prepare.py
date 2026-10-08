"""Prepare exact-count regression labels without altering historical inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from foundation_model.data.composition_sources import canonical_key, normalize_composition


TARGETS = {
    "band_gap": "Band gap (normalized)",
    "density_atomic": "Density atomic (normalized)",
    "magnetization_per_volume": "Total magnetization per volume (normalized)",
    "magnetization_per_fu": "Total magnetization per formula unit (normalized)",
    "reaction_energy": "Equilibrium reaction energy per atom (normalized)",
    "cbm": "CBM (normalized)",
    "vbm": "VBM (normalized)",
    "bulk_modulus": "Bulk modulus (normalized)",
    "shear_modulus": "Shear modulus (normalized)",
    "poisson_ratio": "Poisson ratio (normalized)",
    "universal_anisotropy": "Universal anisotropy (normalized)",
    "refractive_index": "Refractive index (normalized)",
    "piezoelectric_max": "Piezoelectric max (normalized)",
}
COUNTS = (10, 20, 50, 100)
POPULATION_SHA256 = "c0254aad322ed4507c146947829fdf56fc8fadc9998d8de4f697b37188332f01"
LIBRARY = "rikyu_hparam_tuning_v2 / transfer stage final encoders"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_selection(selection: dict, population_path: Path) -> None:
    """Reproduce the October 1 draw against the immutable 240-model population."""
    if sha256(population_path) != POPULATION_SHA256:
        raise ValueError("Frozen checkpoint population checksum differs")
    population = json.loads(population_path.read_text())
    if (
        selection["sampling_seed"] != 20261001
        or selection["source_library"] != LIBRARY
        or population["library"] != LIBRARY
        or selection["population"] != 240
        or population["n_models"] != 240
        or selection["selection_method"] != "Uniform random sample without replacement; no AGIS evaluation used"
    ):
        raise ValueError("Checkpoint sampling provenance differs")
    models = sorted(population["models"], key=lambda m: m["run"])
    if len(models) != 240 or len({m["run"] for m in models}) != 240:
        raise ValueError("Require 240 distinct population models")
    expected = random.Random(20261001).sample(models, 10)
    if [(m["run"], m["sha256"]) for m in selection["models"]] != [(m["run"], m["sha256"]) for m in expected]:
        raise ValueError("Selected checkpoints do not reproduce the frozen random draw")
    if len({m["run"] for m in selection["models"][:3]}) != 3:
        raise ValueError("Require three distinct checkpoints")


def mask_nested(frame: pd.DataFrame, seed: int) -> tuple[dict[int, pd.DataFrame], dict[str, list[str]]]:
    """Mask only training targets; retain exact counts and nested subsets."""
    if not frame.index.is_unique or "split" not in frame:
        raise ValueError("Input must have unique composition keys and explicit splits")
    outputs = {n: frame.copy() for n in COUNTS}
    selected = {}
    for i, (task, col) in enumerate(TARGETS.items()):
        if col not in frame:
            raise ValueError(f"Missing target column: {col}")
        pool = frame.index[(frame["split"] == "train") & frame[col].notna()]
        if len(pool) < max(COUNTS):
            raise ValueError(f"Insufficient training labels for {task}")
        rng = np.random.default_rng(np.random.SeedSequence([20261008, seed, i]))
        order = rng.permutation(pool.to_numpy())[: max(COUNTS)].tolist()
        selected[task] = order
        for n, out in outputs.items():
            out.loc[pool.difference(order[:n]), col] = np.nan
            assert out.loc[out["split"] == "train", col].notna().sum() == n
            pd.testing.assert_series_equal(
                out.loc[frame["split"] != "train", col], frame.loc[frame["split"] != "train", col]
            )
    return outputs, selected


def prepare(source: Path, checkpoint_dir: Path, config_dir: Path, destination: Path) -> dict:
    """Freeze inputs and generate the 156 paired cases."""
    destination.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(source)
    keys = [canonical_key(v, normalize_composition) for v in frame["composition"]]
    frame["composition"] = keys
    frame = frame.drop_duplicates("composition", keep="first").set_index("composition", drop=False)
    source_hashes = {source.name: sha256(source)}
    auxiliary = []
    for name in (
        "NEMAD_magnetic_20260419_norm.parquet",
        "NEMAD_superconductor_20260425_norm.parquet",
        "phonix-db-filtered_20260425_norm.parquet",
    ):
        p = source.parent / name
        (destination / name).write_bytes(p.read_bytes())
        auxiliary.append({"file": name, "sha256": sha256(p)})
    subsets = []
    for seed in range(3):
        outputs, chosen = mask_nested(frame, seed)
        for n, out in outputs.items():
            name = f"qc_n{n:03d}_s{seed}_20261008.parquet"
            out.reset_index(drop=True).to_parquet(destination / name)
            subsets.append({"seed": seed, "n": n, "file": name, "sha256": sha256(destination / name)})
        (destination / f"selected_s{seed}.json").write_text(json.dumps(chosen, indent=2))
    selection_path = checkpoint_dir / "selection_20261001.json"
    selection = json.loads(selection_path.read_text())
    population_path = checkpoint_dir / "population_20261001.json"
    validate_selection(selection, population_path)
    source_hashes[selection_path.name] = sha256(selection_path)
    source_hashes[population_path.name] = sha256(population_path)
    for p in (selection_path, population_path):
        (destination / p.name).write_bytes(p.read_bytes())
    checkpoints = []
    for m in selection["models"][:3]:
        p = checkpoint_dir / f"{m['run']}.pt"
        digest = sha256(p)
        if digest != m["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {p.name}")
        state = torch.load(p, map_location="cpu", weights_only=False)
        if set(TARGETS) & set(state["task_sequence"]):
            raise ValueError(f"Target leakage in checkpoint {p.name}")
        (destination / p.name).write_bytes(p.read_bytes())
        checkpoints.append({"file": p.name, "sha256": digest, "task_sequence": state["task_sequence"]})
    recipes = {}
    for arm, name in [("scratch", "probe6_mp2026_lowdata.toml"), ("transfer", "ft_lowdata_mp2026.toml")]:
        p = config_dir / name
        raw = tomllib.loads(p.read_text())
        source_hashes[name] = sha256(p)
        raw["model"]["latent_dim"] = 384
        raw["training"].update(encoder_lr=0.002)
        raw["training"]["scheduler"] = {"patience": 5, "factor": 0.5, "min_lr": 1e-5}
        for spec in raw["datasets"].values():
            spec["path"] = Path(spec["path"]).name
        recipes[arm] = copy.deepcopy(raw)
    cases = [
        {"case": i, "task": task, "n": n, "seed": seed, "checkpoint": checkpoints[seed]["file"]}
        for i, (task, n, seed) in enumerate((t, n, s) for t in TARGETS for n in COUNTS for s in range(3))
    ]
    manifest = {
        "study": "mlp_few_label_regression_20261008",
        "package": "0.5.0",
        "source_hashes": source_hashes,
        "subsets": subsets,
        "checkpoints": checkpoints,
        "auxiliary": auxiliary,
        "recipes": recipes,
        "cases": cases,
        "targets": TARGETS,
        "counts": COUNTS,
        "seeds": [0, 1, 2],
        "paired_cases": 156,
        "fits": 312,
        "sampling": "Nested uniform training-label subsets after keep-first canonicalization",
        "scope": "Training labels only; historical validation/test and unlabeled reconstruction pool retained",
        "historical_checkpoint_cohort_reused": False,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--configs", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    ap.add_argument("--destination", type=Path, required=True)
    args = ap.parse_args()
    result = prepare(args.source, args.checkpoints, args.configs, args.destination)
    print(f"Prepared {len(result['subsets'])} subsets; {result['fits']} fits")
