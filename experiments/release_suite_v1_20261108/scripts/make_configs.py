"""Generate the release-suite configs from the release task index.

    .venv/bin/python experiments/release_suite_v1_20261108/scripts/make_configs.py

Writes, for a task set (``--set v1`` = the 41 tasks trained before; ``--set full`` = all 64):

- ``configs/pretrain_<set>.toml``: catalog + adopted model / training values + hybrid replay recipe;
- ``configs/consolidate_<set>.toml``: the same catalog + a full-model joint retrain of every head;
- ``configs/export_<set>.toml``: the same catalog **with** each task's scaler / label encoder, + ``[model]``;
- ``configs/members_<set>.txt``: one line per suite member (run id, task-order seed).

Every task carries its unit, its ``t`` unit and, for the DOS curve, ``output_scale = "shape"``. Only the
export config names the scalers. With a scaler configured, the per-step evaluation runs in physical
units, and a Yeo-Johnson inverse with lambda < 0 is undefined above a bound that the training maximum
sits on for 14 of these tasks: an early, unconverged head then yields NaN and the run stops. Training
without scalers keeps the evaluation in the normalised space every earlier campaign reported, and the
scalers never touch the weights, so the export config can add them afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
DATA = "data/fm_task_datasets_20261108_release"
MANIFEST = f"{DATA}/split_manifest_20261016.csv"
MP_SCALERS = "data/preprocessing_objects_20260912.pkl.z"
BINARY_ENCODERS = f"{DATA}/label_encoders_20261108.pkl.z"
ORDER_SEED = 20261108
N_MEMBERS = 10

# Task -> (scaler path, key). JSON keys are the normalised column names (data/*_transforms.json).
SCALERS: dict[str, tuple[str, str]] = {
    "final_energy": (MP_SCALERS, "final_energy_per_atom_scaler"),
    "formation_energy": (MP_SCALERS, "formation_energy_per_atom_scaler"),
    "reaction_energy": (MP_SCALERS, "equilibrium_reaction_energy_per_atom_scaler"),
    "volume": (MP_SCALERS, "volume_scaler"),
    "density": (MP_SCALERS, "density_scaler"),
    "density_atomic": (MP_SCALERS, "density_atomic_scaler"),
    "efermi": (MP_SCALERS, "efermi_scaler"),
    "band_gap": (MP_SCALERS, "band_gap_scaler"),
    "cbm": (MP_SCALERS, "cbm_scaler"),
    "vbm": (MP_SCALERS, "vbm_scaler"),
    "total_magnetization": (MP_SCALERS, "total_magnetization_scaler"),
    "magnetization_per_fu": (MP_SCALERS, "total_magnetization_per_formula_unit_scaler"),
    "magnetization_per_volume": (MP_SCALERS, "total_magnetization_per_volume_scaler"),
    "dielectric_total": (MP_SCALERS, "dielectric_total_scaler"),
    "dielectric_ionic": (MP_SCALERS, "dielectric_ionic_scaler"),
    "dielectric_electronic": (MP_SCALERS, "dielectric_electronic_scaler"),
    "refractive_index": (MP_SCALERS, "refractive_index_scaler"),
    "bulk_modulus": (MP_SCALERS, "bulk_modulus_scaler"),
    "shear_modulus": (MP_SCALERS, "shear_modulus_scaler"),
    "poisson_ratio": (MP_SCALERS, "poisson_ratio_scaler"),
    "universal_anisotropy": (MP_SCALERS, "universal_anisotropy_scaler"),
    "piezoelectric_max": (MP_SCALERS, "piezoelectric_max_scaler"),
    "material_type": (MP_SCALERS, "material_type_label_encoder"),
    "magnetic_ordering": (MP_SCALERS, "magnetic_ordering_label_encoder"),
    "space_group": (MP_SCALERS, "space_group_task_label_encoder"),
    "is_metal": (BINARY_ENCODERS, "is_metal_label_encoder"),
    "is_gap_direct": (BINARY_ENCODERS, "is_gap_direct_label_encoder"),
    "seebeck": (MP_SCALERS, "seebeck_coefficient_scaler"),
    "zt": (MP_SCALERS, "zt_scaler"),
    "power_factor": (MP_SCALERS, "power_factor_scaler"),
    "thermal_conductivity": (MP_SCALERS, "thermal_conductivity_scaler"),
    "electrical_resistivity": (MP_SCALERS, "electrical_resistivity_scaler"),
    "magnetic_susceptibility": (MP_SCALERS, "magnetic_susceptibility_scaler"),
    "magnetic_moment": ("data/NEMAD_magnetic_20260419_transforms.json", "Magnetic moment (normalized)"),
    "magnetization": ("data/NEMAD_magnetic_20260419_transforms.json", "Magnetization (normalized)"),
    "curie": ("data/NEMAD_magnetic_20260419_transforms.json", "Curie temperature (normalized)"),
    "neel": ("data/NEMAD_magnetic_20260419_transforms.json", "Neel temperature (normalized)"),
    "tc": ("data/NEMAD_superconductor_20260425_transforms.json", "Transition temperature (normalized)"),
    "kp": ("data/phonix-db-filtered_20260425_transforms.json", "kp (normalized)"),
    "klat": ("data/phonix-db-filtered_20260425_transforms.json", "klat (normalized)"),
}
# Units as they appear in model cards and prediction files. Curves: (y unit, t unit).
CURVE_UNITS = {
    "dos_density": ("a.u. (each curve scaled to max |y| = 1)", "eV (E - E_F)"),
    "seebeck": ("V/K", "K"),
    "zt": ("dimensionless", "K"),
    "power_factor": ("W/(m·K²)", "K"),
    "thermal_conductivity": ("W/(m·K)", "K"),
    "electrical_resistivity": ("Ω·m", "K"),
    "magnetic_susceptibility": ("not recorded by the source", "K"),
}
SHAPE_HEADS = {"dos_density"}

MODEL = """[model]
latent_dim = 384
encoder_hidden_dims = [256]
head_hidden_dims = [64]
kr_x_hidden_dims = [128, 64]
kr_t_hidden_dims = [16, 8]
n_kernel = 15
"""
TRAINING = """[training]
max_epochs = {max_epochs}
encoder_lr = 0.002
head_lr = 0.005
kr_lr = 0.0005
kr_weight_decay = 5e-5
ae_lr = 0.005
learnable_loss_balancer = false
accelerator = "gpu"
devices = 1
seed = 2025

[training.scheduler]
min_lr = 1e-05
patience = 5
factor = 0.5

[training.early_stopping]
enabled = true
monitor = "val_final_loss"
mode = "min"
patience = 24
min_delta = 1e-4

[training.logging]
csv = true
"""


def q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def tasks_for(index: pd.DataFrame, which: str) -> pd.DataFrame:
    sel = index[index.trained_in_campaign_v2] if which == "v1" else index
    return sel.reset_index(drop=True)


def catalog(tasks: pd.DataFrame, scalers: bool) -> str:
    files = sorted(tasks.file.unique())
    names = {f: f.removesuffix(".parquet") for f in files}
    out = [
        "[data]",
        'composition_column = "composition"',
        f"split_manifest = {q(MANIFEST)}",
        "batch_size = 256",
        "num_workers = 0",
        "",
        "[descriptor]",
        'kind = "kmd"',
        "n_grids = 8",
        "",
    ]
    for f in files:
        out += [f"[datasets.{names[f]}]", f"path = {q(DATA + '/' + f)}", ""]
    for t in tasks.itertuples():
        kind = {"curve": "kernel_regression"}.get(t.kind, t.kind)
        out += ["[[tasks]]", f"name = {q(t.task)}", f"kind = {q(kind)}", f"dataset = {q(names[t.file])}"]
        out.append(f"column = {q(t.column)}")
        if kind == "kernel_regression":
            unit, t_unit = CURVE_UNITS[t.task]
            out += [f"t_column = {q(t.t_column)}", f"unit = {q(unit)}", f"t_unit = {q(t_unit)}"]
        elif kind == "regression":
            out.append(f"unit = {q(t.unit)}")
        if kind == "classification":
            out += [f"num_classes = {int(t.n_classes)}", 'class_weights = "none"']
        if t.task in SHAPE_HEADS:
            out.append('output_scale = "shape"')
        elif scalers:
            path, key = SCALERS[t.task]
            out += [f"scaler = {{ path = {q(path)}, key = {q(key)} }}"]
        out.append("")
    return "\n".join(out)


def pretrain(tasks: pd.DataFrame, which: str) -> str:
    curves = [t for t, k in zip(tasks.task, tasks.kind) if k == "curve"]
    others = [t for t in tasks.task if t not in curves]
    floor = {t: 1500 for t, n in zip(tasks.task, tasks.n_train) if 0.3 * n < 1500}
    seq = others + curves

    def lst(xs: list[str]) -> str:
        return "[\n" + "".join(f"    {q(x)},\n" for x in xs) + "]"

    return "\n".join(
        [
            f"# GENERATED by scripts/make_configs.py --set {which} — do not hand-edit.",
            f"# Release suite: {len(seq)} tasks; members differ only in task order (task_order_seed).",
            "",
            catalog(tasks, scalers=False),
            MODEL,
            TRAINING.format(max_epochs=150),
            "[pretrain]",
            f"task_sequence = {lst(seq)}",
            "n_runs = 1",
            'task_order = "random"',
            'run_seed = "fixed"',
            f"task_order_seed = {ORDER_SEED}  # member i runs with --set pretrain.task_order_seed={ORDER_SEED}+i",
            "# scalar and classification heads shuffled first, the kernel-regression curves shuffled in a final",
            "# block: curve replay dominates wall time, so an early curve would be replayed at every later step.",
            f"task_order_groups = [{lst(others)}, {lst(curves)}]",
            "",
            "[pretrain.replay]",
            "interval = 1",
            "amount = 0.30",
            'resample = "epoch"',
            "# hybrid rule max(1500, 0.3 N): every task with 0.3 * n_train < 1500 gets 1500 (clamped to N)",
            "per_task = { " + ", ".join(f"{t} = {v}" for t, v in floor.items()) + " }",
            "",
            "[output]",
            f'dir = "artifacts/release_suite_{which}/pretrain"',
            "",
        ]
    )


def consolidate(tasks: pd.DataFrame, which: str) -> str:
    return "\n".join(
        [
            f"# GENERATED by scripts/make_configs.py --set {which} — do not hand-edit.",
            "# Full-model joint retrain of every head from a member's pretrain checkpoint (the recipe's end pass).",
            "",
            catalog(tasks, scalers=False),
            MODEL,
            TRAINING.format(max_epochs=100),
            "[finetune]",
            "tasks = [\n" + "".join(f"    {q(t)},\n" for t in tasks.task) + "]",
            "epochs = 150",
            "freeze_encoder = false",
            "add_new_tasks = false",
            "",
            "[output]",
            f'dir = "artifacts/release_suite_{which}/consolidate"',
            "",
        ]
    )


def export(tasks: pd.DataFrame, which: str) -> str:
    return "\n".join(
        [
            f"# GENERATED by scripts/make_configs.py --set {which} — do not hand-edit.",
            "# fm export reads only the catalog and [model]; this catalog adds every task's scaler / label encoder.",
            "",
            catalog(tasks, scalers=True),
            MODEL,
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", dest="which", choices=["v1", "full"], default="v1")
    args = p.parse_args()
    tasks = tasks_for(pd.read_csv(ROOT / DATA / "task_index.csv"), args.which)
    missing = set(tasks.task) - set(SCALERS) - SHAPE_HEADS
    if missing:
        # --set full stops here until SCALERS covers the HEA / QMOF / Catalysis-Hub tasks (README, plan step 1).
        raise SystemExit(f"--set {args.which} is not supported yet: no scaler mapping for {sorted(missing)}")
    (EXP / "configs" / f"pretrain_{args.which}.toml").write_text(pretrain(tasks, args.which), encoding="utf-8")
    (EXP / "configs" / f"consolidate_{args.which}.toml").write_text(consolidate(tasks, args.which), encoding="utf-8")
    (EXP / "configs" / f"export_{args.which}.toml").write_text(export(tasks, args.which), encoding="utf-8")
    members = "".join(f"m{i:03d}\t{ORDER_SEED + i}\n" for i in range(N_MEMBERS))
    (EXP / "configs" / f"members_{args.which}.txt").write_text(members, encoding="utf-8")
    print(f"{args.which}: {len(tasks)} tasks, {N_MEMBERS} members")


if __name__ == "__main__":
    main()
