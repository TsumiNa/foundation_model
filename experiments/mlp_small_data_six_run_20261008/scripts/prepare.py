"""Register six-run regression curves and prove completed paired fits can be reused."""

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
COUNTS = (10, 20, 50, 100, 300, 1000, 3000, 10000)
SEEDS = tuple(range(6))
REUSE_MANIFEST_SHA256 = "fda38283df55e0292072cad30876023c9d08551c974b39cb222525db3a9e63b8"
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
    if len({m["run"] for m in selection["models"][:6]}) != 6:
        raise ValueError("Require six distinct checkpoints")


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
        if len(pool) < 100:
            raise ValueError(f"Insufficient training labels for {task}")
        rng = np.random.default_rng(np.random.SeedSequence([20261008, seed, i]))
        order = rng.permutation(pool.to_numpy())[: max(COUNTS)].tolist()
        selected[task] = order
        for n, out in outputs.items():
            out.loc[pool.difference(order[:n]), col] = np.nan
            assert out.loc[out["split"] == "train", col].notna().sum() == min(n, len(pool))
            pd.testing.assert_series_equal(
                out.loc[frame["split"] != "train", col], frame.loc[frame["split"] != "train", col]
            )
    return outputs, selected


def prepare(
    source: Path,
    checkpoint_dir: Path,
    config_dir: Path,
    destination: Path,
    reuse_data: Path,
    reuse_root: Path,
    plan_path: Path,
) -> dict:
    """Register 558 pairs; train only the 402 not already completed under this protocol."""
    old_path = reuse_data / "manifest.json"
    if sha256(old_path) != REUSE_MANIFEST_SHA256:
        raise ValueError("Reuse manifest differs from the completed October 8 campaign")
    old = json.loads(old_path.read_text())
    if sha256(Path(__file__).with_name("run.py")) != old["script_sha256"]["run.py"]:
        raise ValueError("Scientific worker changed; existing fits cannot be pooled")
    plan = json.loads(plan_path.read_text())
    if plan["max_share"] != 0.8:
        raise ValueError("Expected the historical 80-percent eligibility limit")
    task_counts = {t: [10, 20, 50, *plan["plan"][t]] for t in TARGETS}
    if sum(map(len, task_counts.values())) != 93:
        raise ValueError("Expected 93 regression task/count points")
    destination.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(source)
    keys = [canonical_key(v, normalize_composition) for v in frame["composition"]]
    frame["composition"] = keys
    frame = frame.drop_duplicates("composition", keep="first").set_index("composition", drop=False)
    source_hashes = {source.name: sha256(source), plan_path.name: sha256(plan_path)}
    available = {t: int(((frame["split"] == "train") & frame[col].notna()).sum()) for t, col in TARGETS.items()}
    if any(n > 0.8 * available[t] for t, counts in task_counts.items() for n in counts):
        raise ValueError("Registered count exceeds 80 percent of canonical training-label pool")
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
    for seed in SEEDS:
        outputs, chosen = mask_nested(frame, seed)
        for n, out in outputs.items():
            name = f"qc_n{n:03d}_s{seed}_20261008.parquet"
            if seed < 3 and n <= 100:
                archived = next(v for v in old["subsets"] if v["seed"] == seed and v["n"] == n)
                path = reuse_data / archived["file"]
                if sha256(path) != archived["sha256"]:
                    raise ValueError("Reused input subset checksum differs")
                pd.testing.assert_frame_equal(out.reset_index(drop=True), pd.read_parquet(path))
                (destination / name).write_bytes(path.read_bytes())
            else:
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
    for m in selection["models"][:6]:
        p = checkpoint_dir / f"{m['run']}.pt"
        digest = sha256(p)
        if digest != m["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {p.name}")
        state = torch.load(p, map_location="cpu", weights_only=False)
        if set(TARGETS) & set(state["task_sequence"]):
            raise ValueError(f"Target leakage in checkpoint {p.name}")
        for key, shape in (
            ("encoder.shared.layers.0.layer.weight", (256, 464)),
            ("encoder.shared.layers.1.layer.weight", (384, 256)),
        ):
            if key not in state["model"] or tuple(state["model"][key].shape) != shape:
                raise ValueError(f"Source encoder architecture differs: {p.name}")
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
    for name, digest in old["source_hashes"].items():
        if source_hashes.get(name) != digest:
            raise ValueError(f"Reused source or recipe input differs: {name}")
    if recipes != old["recipes"] or auxiliary != old["auxiliary"] or checkpoints[:3] != old["checkpoints"]:
        raise ValueError("Reused resolved recipes or auxiliary/checkpoint inputs differ")
    reuse_identity = json.loads((reuse_root / "case000/done.json").read_text())["identity"]
    expected_reuse = {
        "manifest": REUSE_MANIFEST_SHA256,
        "revision": "630e973dfa51bde36c0ce8dbaee34919ad9f73ed",
        "image_hash": "f90adc81f1148db2fff0ae94551b751e77f82c5d52d032ea54641f746dbe2f67",
        "smoke": False,
        "script_sha256": old["script_sha256"],
    }
    if reuse_identity != expected_reuse:
        raise ValueError("Reuse runtime identity differs")
    for case in old["cases"]:
        marker = json.loads((reuse_root / f"case{case['case']:03d}/done.json").read_text())
        if marker != {"case": case, "identity": expected_reuse}:
            raise ValueError("Incomplete or mismatched reused pair")
    old_keys = {(c["task"], c["n"], c["seed"]): c for c in old["cases"]}
    all_cases = [
        {"task": task, "n": n, "seed": seed, "checkpoint": checkpoints[seed]["file"]}
        for task in TARGETS
        for n in task_counts[task]
        for seed in SEEDS
    ]
    cases = [c for c in all_cases if (c["task"], c["n"], c["seed"]) not in old_keys]
    cases = [{"case": i, **c} for i, c in enumerate(cases)]
    if len(cases) != 402 or len(old_keys) != 156 or len(all_cases) != 558:
        raise ValueError("Unexpected new/reused case coverage")
    manifest = {
        "study": "mlp_small_data_six_run_20261008",
        "package": "0.5.0",
        "source_hashes": source_hashes,
        "script_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("run.py", "array.sbatch")},
        "subsets": subsets,
        "checkpoints": checkpoints,
        "auxiliary": auxiliary,
        "recipes": recipes,
        "cases": cases,
        "targets": TARGETS,
        "counts": COUNTS,
        "seeds": SEEDS,
        "task_counts": task_counts,
        "available_training_labels": available,
        "all_cases": all_cases,
        "reuse_cases": old["cases"],
        "reuse_identity": expected_reuse,
        "reuse_input_manifest": old,
        "reuse_proof": "Identical worker, resolved recipes, data source, auxiliary inputs, first three checkpoints and exact first 100 nested labels; subset bytes preserved",
        "new_pairs": 402,
        "new_fits": 804,
        "paired_cases": 558,
        "fits": 1116,
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
    ap.add_argument("--reuse-data", type=Path, required=True)
    ap.add_argument("--reuse-root", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    args = ap.parse_args()
    result = prepare(
        args.source, args.checkpoints, args.configs, args.destination, args.reuse_data, args.reuse_root, args.plan
    )
    print(f"Prepared {len(result['subsets'])} subsets; {result['fits']} fits")
