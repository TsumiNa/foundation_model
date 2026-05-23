# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Continual multi-task rehearsal + inverse-design — **full / formal** run.

A larger, "formal training" sibling of :mod:`continual_rehearsal_demo`. It covers the complete
inorganic task catalogue (24 supervised tasks + always-on autoencoder) over four datasets and,
relative to the demo, adds:

* **Tiered rehearsal** — a configurable high-replay set (the inverse-design-relevant tail tasks,
  e.g. formation_energy / magnetic_moment / tc / klat / material_type) keeps ``replay_ratio_high``
  of its labels when replayed as an *old* task, while every other learned task keeps ``replay_ratio``.
* **EarlyStopping** on ``val_final_loss`` (full data ⇒ ``max_epochs_per_step`` is just a ceiling).
* **Per-stage raw artifacts** — at every step, every active head's test ``(composition, true, pred)``
  is dumped to parquet (kernel heads additionally store the ``t`` series), alongside a per-task
  ``<task>_metrics.json`` and a per-step ``checkpoint.pt`` (model state + active-task metadata).
  Everything lives under ``training/stepNN_<task>/`` so any intermediate stage can be revisited.
* **Final checkpoint** — ``training/final_model.pt`` + ``training/final_model_taskconfigs.json``.
* **Multiple inverse-design scenarios** — the same final model is optimized through **four PR #18
  paths per scenario** (latent with cycle-consistency + composition strict / alloy-palette /
  random init), with results, a 4-path comparison plot, an element-frequency heatmap (discovered
  elements highlighted), and `targets.json` written to ``inverse_design/<scenario>/``.
* **Slide-prep deliverables (no auto PPT / HTML)** — the runner emits ``SLIDE_PREP.md`` (9-section
  outline + raw-data pointers), ``ANALYSIS.md`` (long-form English narrative), ``README.md``
  (directory index), and per-scenario ``comparison.png`` / ``element_frequency_heatmap.png``
  inside ``inverse_design/<scenario>/``. The three scenarios are first-class results — the runner
  does **not** promote any single scenario as the headline (that was a demo-only convention).
  The slide author builds the deck externally; every figure is reproducible from the raw arrays
  without retraining.

No layers are frozen: every step jointly trains the shared encoder + all active task heads
(``freeze_shared_encoder=False``, per-task ``freeze_parameters=False``). The "continual" behaviour
comes purely from the rehearsal mask, not from freezing.

Run:
    ./run_continual_rehearsal_full.sh samples/continual_rehearsal_full_config.toml
    python -m foundation_model.scripts.continual_rehearsal_full --config-file <toml>
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless

import joblib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score  # type: ignore[import-untyped]

from foundation_model.data.composition_sources import normalize_composition
from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import (
    ClassificationTaskConfig,
    KernelRegressionTaskConfig,
    MLPEncoderConfig,
    OptimizerConfig,
    RegressionTaskConfig,
)
from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS, KMD, element_features, formula_to_composition

# Reuse the spec-independent helpers + HTML shell from the demo (no behaviour change to the demo).
from foundation_model.scripts.continual_rehearsal_demo import (
    _PALETTE,
    _SCATTER_COLOR,
    _apply_plot_style,
    _as_float_array,
    _composition_key,
    _init_kernels,
)

# --- Task catalogue ----------------------------------------------------------
# source: dataset the task's targets come from. qc columns are pre-normalized; raw NEMAD/phonix
# regression columns are log1p + z-scored (train-only stats) + clipped at load time.
TASK_SPECS: dict[str, dict[str, Any]] = {
    # --- qc: regression (9) ---
    "density": {"source": "qc", "kind": "reg", "column": "Density (normalized)"},
    "efermi": {"source": "qc", "kind": "reg", "column": "Efermi (normalized)"},
    "final_energy": {"source": "qc", "kind": "reg", "column": "Final energy per atom (normalized)"},
    "formation_energy": {"source": "qc", "kind": "reg", "column": "Formation energy per atom (normalized)"},
    "total_magnetization": {"source": "qc", "kind": "reg", "column": "Total magnetization (normalized)"},
    "volume": {"source": "qc", "kind": "reg", "column": "Volume (normalized)"},
    "dielectric_total": {"source": "qc", "kind": "reg", "column": "Dielectric total (normalized)"},
    "dielectric_ionic": {"source": "qc", "kind": "reg", "column": "Dielectric ionic (normalized)"},
    "dielectric_electronic": {"source": "qc", "kind": "reg", "column": "Dielectric electronic (normalized)"},
    # --- qc: kernel regression (7) ---
    "dos_density": {"source": "qc", "kind": "kr", "column": "DOS density (normalized)", "t_column": "DOS energy"},
    "electrical_resistivity": {
        "source": "qc",
        "kind": "kr",
        "column": "Electrical resistivity (normalized)",
        "t_column": "Electrical resistivity (T/K)",
    },
    "power_factor": {
        "source": "qc",
        "kind": "kr",
        "column": "Power factor (normalized)",
        "t_column": "Power factor (T/K)",
    },
    "seebeck": {
        "source": "qc",
        "kind": "kr",
        "column": "Seebeck coefficient (normalized)",
        "t_column": "Seebeck coefficient (T/K)",
    },
    "thermal_conductivity": {
        "source": "qc",
        "kind": "kr",
        "column": "Thermal conductivity (normalized)",
        "t_column": "Thermal conductivity (T/K)",
    },
    "zt": {"source": "qc", "kind": "kr", "column": "ZT (normalized)", "t_column": "ZT (T/K)"},
    "magnetic_susceptibility": {
        "source": "qc",
        "kind": "kr",
        "column": "Magnetic susceptibility (normalized)",
        "t_column": "Magnetic susceptibility (T/K)",
    },
    # --- qc: classification (1) ---
    "material_type": {"source": "qc", "kind": "clf", "column": "Material type (label)", "num_classes": 3},
    # --- phonix-db: regression (2) ---
    "kp": {"source": "phonix", "kind": "reg", "column": "kp[W/mK]"},
    "klat": {"source": "phonix", "kind": "reg", "column": "klat[W/mK]"},
    # --- NEMAD superconductor: regression (1) ---
    "tc": {"source": "superconductor", "kind": "reg", "column": "Transition temperature[K]"},
    # --- NEMAD magnetic: regression (4) ---
    "magnetic_moment": {"source": "magnetic", "kind": "reg", "column": "Magnetic moment[μB/f.u.]"},
    "magnetization": {"source": "magnetic", "kind": "reg", "column": "Magnetization[A·m²/mol]"},
    "curie": {"source": "magnetic", "kind": "reg", "column": "Curie temperature[K]"},
    "neel": {"source": "magnetic", "kind": "reg", "column": "Neel temperature[K]"},
}

# Raw (non-qc) regression targets span orders of magnitude; log1p-compress, z-score, clip tails.
_RAW_TARGET_CLIP = 5.0

# Default 24-task sequence: 19 free-order tasks, then the fixed inverse-design tail (kept freshest).
DEFAULT_SEQUENCE = [
    # qc regression (free)
    "density",
    "efermi",
    "final_energy",
    "total_magnetization",
    "volume",
    "dielectric_total",
    "dielectric_ionic",
    "dielectric_electronic",
    # qc kernel regression (free)
    "dos_density",
    "electrical_resistivity",
    "power_factor",
    "seebeck",
    "thermal_conductivity",
    "zt",
    "magnetic_susceptibility",
    # magnetic + phonix (free)
    "magnetization",
    "curie",
    "neel",
    "kp",
    # fixed tail (inverse-design heads, freshest at the end)
    "formation_energy",
    "magnetic_moment",
    "tc",
    "klat",
    "material_type",
]
# The inverse-design-relevant tail: kept at the higher replay ratio when replayed as an old task.
DEFAULT_FIXED_TAIL = ["formation_energy", "magnetic_moment", "tc", "klat", "material_type"]

# 5 fine labels merged into AC / QC / others (index == merged class id).
_MATERIAL_TYPE_MERGE = {0: 0, 2: 0, 1: 1, 3: 1, 4: 2}
MATERIAL_TYPE_CLASSES = ["AC", "QC", "others"]
MATERIAL_TYPE_DISPLAY_ORDER = ["others", "AC", "QC"]
QC_CLASSES = [1]  # merged quasicrystal class index — inverse-design classification objective.

# --- Presentation -------------------------------------------------------------
TASK_DISPLAY: dict[str, str] = {
    "density": "Density",
    "efermi": "E_Fermi",
    "final_energy": "Final Energy / atom",
    "formation_energy": "Formation Energy",
    "total_magnetization": "Total Magnetization",
    "volume": "Volume",
    "dielectric_total": "Dielectric (total)",
    "dielectric_ionic": "Dielectric (ionic)",
    "dielectric_electronic": "Dielectric (electronic)",
    "dos_density": "DOS Density",
    "electrical_resistivity": "Electrical Resistivity",
    "power_factor": "Power Factor",
    "seebeck": "Seebeck Coefficient",
    "thermal_conductivity": "Thermal Conductivity",
    "zt": "ZT",
    "magnetic_susceptibility": "Magnetic Susceptibility",
    "material_type": "Material Type",
    "kp": "Phonon Conductivity (κₚ)",
    "klat": "Lattice Conductivity (κ_lat)",
    "tc": "Critical Temperature (Tc)",
    "magnetic_moment": "Magnetic Moment",
    "magnetization": "Magnetization",
    "curie": "Curie Temperature",
    "neel": "Néel Temperature",
}
SOURCE_DISPLAY = {
    "qc": "qc_ac_te_mp",
    "phonix": "phonix-db",
    "superconductor": "NEMAD superconductor",
    "magnetic": "NEMAD magnetic",
}
KIND_LABEL = {"reg": "regression", "kr": "kernel regression", "clf": "classification"}

# --- Inverse design — paths + element constraints ----------------------------
# 41-element alloy palette for the composition-space ``C-alloy`` path (plan §5). Covers classic
# i-QC / d-QC formers (Mg–Zn–RE, Al–Mn, Al–Cu–Fe, Al–Ni–Co, Au–Ga–RE …), the Sc–Zn 4th-period TMs,
# the Y–Cd 5th-period TMs (Tc excluded for radioactivity), Au (Au–Ga–Ln seeds need it), group 13/14
# enablers (B/Al/Ga/In/Tl, Si/Ge), and the 12 easy lanthanides. Pm/Tc are radioactive; Tm/Lu are
# scarce. The three explicit-append Au–Ga–Ln seeds (Gd/Tb/Dy) all fit in this palette.
ALLOY_PALETTE: list[str] = [
    "Mg",
    "Ca",
    "B",
    "Al",
    "Ga",
    "In",
    "Tl",
    "Si",
    "Ge",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Au",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Yb",
]

# Inverse-design comparison configurations, one row per box in ``comparison.png``. Mirrors the
# PR #18 demo's ``paper_inverse_comparison.py``: a 3-point ``ae_align_scale`` sweep on the latent
# side (failure α=0 / mid α=0.25 / max α=1.0) plus five composition configurations that layer
# blend, palette and diversity-scale knobs against a random-init control. The ``allowed`` field
# uses the sentinel ``"__palette__"`` to refer to ``config.inverse_composition_allowed_elements``
# (the 41-element ``ALLOY_PALETTE`` by default); every other field is fixed at the module level so
# the comparison is a stable plan-§5 ablation across runs.
_PALETTE_SENTINEL = "__palette__"
INVERSE_PATH_CONFIGS: list[dict[str, Any]] = [
    {"key": "latent_align0p0", "label": "latent α=0", "method": "latent", "ae_align_scale": 0.0},
    {"key": "latent_align0p25", "label": "latent α=0.25", "method": "latent", "ae_align_scale": 0.25},
    {"key": "latent_align1p0", "label": "latent α=1", "method": "latent", "ae_align_scale": 1.0},
    {
        "key": "comp_seed",
        "label": "comp (seed)",
        "method": "composition",
        "init": "seed",
        "blend": 1.0,
        "allowed": "all",
        "diversity": 1.0,
    },
    {
        "key": "comp_seed_blend",
        "label": "comp (seed, 5% all)",
        "method": "composition",
        "init": "seed",
        "blend": 0.95,
        "allowed": "all",
        "diversity": 1.0,
    },
    {
        "key": "comp_seed_blend_palette",
        "label": "comp (seed, 5% all, element list)",
        "method": "composition",
        "init": "seed",
        "blend": 0.95,
        "allowed": _PALETTE_SENTINEL,
        "diversity": 1.0,
    },
    {
        # Ablation: clamp diversity to 0 → max entropy penalty → forced peaky few-element recipes.
        "key": "comp_seed_blend_palette_lowdiv",
        "label": "comp (seed, 5% all, element list, low diversity)",
        "method": "composition",
        "init": "seed",
        "blend": 0.95,
        "allowed": _PALETTE_SENTINEL,
        "diversity": 0.0,
    },
    {
        "key": "comp_random",
        "label": "comp (random)",
        "method": "composition",
        "init": "random",
        "blend": 0.95,
        "allowed": "all",
        "diversity": 1.0,
    },
]
INVERSE_PATHS: list[str] = [c["key"] for c in INVERSE_PATH_CONFIGS]

# Per-regression-task panel title (units + arrow). Matches the demo's REG_TASK_TITLES so plots
# read the same across both runners. Falls back to the bare task name if a task isn't listed.
REG_TASK_TITLES: dict[str, str] = {
    "formation_energy": "Formation energy [eV/atom] ↓",
    "klat": "klat [W/mK] ↑",
    "magnetic_moment": "Magnetic moment [μB/f.u.] ↑",
    "tc": "Critical temperature [K] ↑",
}


def _seed_weights_from_compositions(seeds: list[str], n_components: int) -> torch.Tensor:
    """Element-weight tensor ``(B, n_components)`` for seeding ``optimize_composition``.

    Order matches DEFAULT_ELEMENTS. Raises if any seed cannot be parsed — we fail fast rather than
    silently dropping rows (callers rely on per-seed correspondence with the latent path).
    """
    rows = []
    for c in seeds:
        w = formula_to_composition(c)
        if w is None:
            raise ValueError(f"Cannot parse seed composition '{c}' to element weights.")
        rows.append(np.asarray(w, dtype=np.float64))
    return torch.tensor(np.stack(rows), dtype=torch.float64)


def _format_weights(weights: np.ndarray, top_k: int = 6, eps: float = 1e-3) -> list[str]:
    """Render element-weight rows as compact formula strings (top-K elements above ``eps``)."""
    out: list[str] = []
    for row in weights:
        order = np.argsort(row)[::-1]
        parts = [f"{DEFAULT_ELEMENTS[i]}{row[i]:.3f}" for i in order[:top_k] if row[i] > eps]
        out.append(" ".join(parts) if parts else "<empty>")
    return out


def _display(task: str) -> str:
    return TASK_DISPLAY.get(task, task.replace("_", " ").title())


def _scale_label(task: str) -> str:
    return "normalized" if TASK_SPECS[task]["source"] == "qc" else "log1p, z-scored"


def _title(task: str) -> str:
    return f"{_display(task)}  ({_scale_label(task)})"


def _arrow(value: float) -> str:
    return "↓" if value < 0 else "↑"


@dataclass
class InverseScenario:
    """One inverse-design objective set (primary = QC probability; secondary = regression targets)."""

    name: str
    reg_tasks: list[str]
    reg_targets: list[float]

    def __post_init__(self) -> None:
        if len(self.reg_tasks) != len(self.reg_targets):
            raise ValueError(f"Scenario '{self.name}': reg_tasks and reg_targets must have equal length.")


@dataclass
class ContinualRehearsalFullConfig:
    """Configuration for the full continual rehearsal + inverse-design run."""

    qc_data_path: Path = Path("data/qc_ac_te_mp_dos_reformat_20260515.pd.parquet")
    qc_preprocessing_path: Path | None = None
    superconductor_path: Path = Path("data/NEMAD_superconductor_20260425.parquet")
    magnetic_path: Path = Path("data/NEMAD_magnetic_20260419.parquet")
    phonix_path: Path = Path("data/phonix-db-filtered_20260425.parquet")
    output_dir: Path = Path("artifacts/continual_rehearsal_full")

    task_sequence: list[str] = field(default_factory=lambda: list(DEFAULT_SEQUENCE))
    fixed_tail: list[str] = field(default_factory=lambda: list(DEFAULT_FIXED_TAIL))
    replay_ratio: float = 0.05  # ordinary old-task replay ratio
    replay_ratio_high: float = 0.10  # replay ratio for fixed_tail tasks when replayed as old
    sample_per_dataset: int | None = None  # cap rows per dataset (for fast/smoke runs)

    max_epochs_per_step: int = 100  # ceiling; EarlyStopping usually stops sooner
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    batch_size: int = 256
    num_workers: int = 0

    n_grids: int = 8
    latent_dim: int = 128
    encoder_hidden: int = 256
    head_hidden_dim: int = 64
    head_lr: float = 5e-3
    encoder_lr: float = 5e-3
    n_kernel: int = 15
    kr_lr: float = 5e-4
    kr_decay: float = 5e-5

    # Inverse design (shared across scenarios). Primary objective is QC probability ↑; each
    # scenario runs the four PR #18 paths (latent + 3 composition configs) — see plan §5.
    inverse_n_seeds: int = 20  # 17 top-QC dedup + 3 explicit Au-Ga-Ln formers (plan §5)
    inverse_steps: int = 300
    inverse_lr: float = 0.05
    inverse_class_weight: float = 5.0
    # 41-element ``ALLOY_PALETTE`` for the composition rows that whitelist elements. Configurable
    # in case the slide author wants a wider or narrower palette; everything else (ae_align_scale
    # sweep, seed_blend, diversity_scale) is fixed at the module level in ``INVERSE_PATH_CONFIGS``
    # so the comparison is a stable ablation across runs.
    inverse_composition_allowed_elements: list[str] = field(default_factory=lambda: list(ALLOY_PALETTE))
    inverse_seed_strategy: str = "top_qc"  # "top_qc" | "random" | "explicit"
    inverse_seed_split: str = "train"  # "train" | "val" | "test" | "all"
    inverse_seed_compositions: list[str] = field(default_factory=list)
    # Compositions appended to the strategy-selected seeds regardless of QC ranking. Each must
    # have a computable descriptor (fail-fast in _select_seeds). The strategy budget is reduced
    # by len(explicit_append) so total seeds == inverse_n_seeds. Defaults to the three Au-Ga-Ln
    # i-QC formers used in plan §5 (Au65 Ga20 Gd/Tb/Dy15).
    inverse_seed_explicit_append: list[str] = field(
        default_factory=lambda: ["Au65 Ga20 Gd15", "Au65 Ga20 Tb15", "Au65 Ga20 Dy15"]
    )
    inverse_scenarios: list[InverseScenario] = field(
        default_factory=lambda: [
            InverseScenario("scenario1_fe_down_moment_up", ["formation_energy", "magnetic_moment"], [-2.0, 2.0]),
            InverseScenario("scenario2_fe_tc_moment", ["formation_energy", "tc", "magnetic_moment"], [-2.0, 2.0, 2.0]),
            InverseScenario("scenario3_fe_down_klat_up", ["formation_energy", "klat"], [-2.0, 2.0]),
        ]
    )

    random_seed: int = 2025
    datamodule_random_seed: int = 42
    accelerator: str = "auto"
    devices: int = 1

    def __post_init__(self) -> None:
        unknown = [t for t in self.task_sequence if t not in TASK_SPECS]
        if unknown:
            raise ValueError(f"Unknown task(s) {unknown}. Available: {sorted(TASK_SPECS)}")
        if len(set(self.task_sequence)) != len(self.task_sequence):
            raise ValueError("task_sequence contains duplicates.")
        bad_tail = [t for t in self.fixed_tail if t not in self.task_sequence]
        if bad_tail:
            raise ValueError(f"fixed_tail tasks {bad_tail} are not in task_sequence.")
        for ratio_name, ratio in (("replay_ratio", self.replay_ratio), ("replay_ratio_high", self.replay_ratio_high)):
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{ratio_name} must be in [0, 1].")
        if not self.inverse_composition_allowed_elements:
            raise ValueError("inverse_composition_allowed_elements must be non-empty.")
        unknown_palette = [e for e in self.inverse_composition_allowed_elements if e not in DEFAULT_ELEMENTS]
        if unknown_palette:
            raise ValueError(
                f"inverse_composition_allowed_elements contains symbols not in DEFAULT_ELEMENTS: {unknown_palette}"
            )
        if self.inverse_seed_strategy not in {"top_qc", "random", "explicit"}:
            raise ValueError("inverse_seed_strategy must be 'top_qc', 'random', or 'explicit'.")
        if self.inverse_seed_split not in {"train", "val", "test", "all"}:
            raise ValueError("inverse_seed_split must be 'train', 'val', 'test', or 'all'.")
        if self.inverse_seed_strategy == "explicit" and not self.inverse_seed_compositions:
            raise ValueError("inverse_seed_strategy='explicit' requires inverse_seed_compositions.")
        # Every scenario's tasks must be regression tasks present in the sequence.
        for sc in self.inverse_scenarios:
            for t in sc.reg_tasks:
                if t not in self.task_sequence:
                    raise ValueError(f"Scenario '{sc.name}': task '{t}' not in task_sequence.")
                if TASK_SPECS[t]["kind"] != "reg":
                    raise ValueError(f"Scenario '{sc.name}': task '{t}' must be a (scalar) regression task.")
        if "material_type" not in self.task_sequence:
            raise ValueError("task_sequence must contain 'material_type' (QC classifier for inverse design).")


class ContinualRehearsalFullRunner:
    def __init__(self, config: ContinualRehearsalFullConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        # Plan §6 layout: training/ for per-step artifacts (incl. final_model.pt and forgetting
        # trajectory), inverse_design/ for the dual-path scenarios, slide-prep / analysis / readme
        # at the top level. Subdirs are created lazily where needed.
        self.training_dir = self.output_dir / "training"
        self.inverse_root = self.output_dir / "inverse_design"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        _apply_plot_style()
        self._task_colors = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(config.task_sequence)}
        self._kmd = KMD(element_features.values, method="1d", n_grids=config.n_grids, sigma="auto", scale=True)
        self.x_dim = int(self._kmd.transform(np.eye(1, len(DEFAULT_ELEMENTS))).shape[1])
        self._desc_cache: dict[str, np.ndarray] = {}
        self._load_data()

    # ------------------------------------------------------------------ data

    def _load_data(self) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.datamodule_random_seed)
        self.task_frames: dict[str, pd.DataFrame] = {}
        split_by_key: dict[str, str] = {}

        sources = {
            "qc": self._load_qc(),
            "superconductor": pd.read_parquet(cfg.superconductor_path),
            "magnetic": pd.read_parquet(cfg.magnetic_path),
            "phonix": pd.read_parquet(cfg.phonix_path),
        }

        keyed: dict[str, pd.DataFrame] = {}
        for name, df in sources.items():
            df = df.copy()
            if cfg.sample_per_dataset is not None and cfg.sample_per_dataset < len(df):
                if name == "qc" and "Material type (label)" in df.columns:
                    df = self._stratified_qc_sample(df, cfg.sample_per_dataset, rng)
                else:
                    df = df.iloc[rng.choice(len(df), size=cfg.sample_per_dataset, replace=False)]
            df["__key__"] = [_composition_key(v) for v in df["composition"]]
            df = df.dropna(subset=["__key__"]).drop_duplicates(subset="__key__", keep="first").set_index("__key__")
            keyed[name] = df
            if "split" in df.columns:
                for k, s in df["split"].items():
                    split_by_key.setdefault(str(k), str(s))
            else:
                for k in df.index:
                    split_by_key.setdefault(str(k), rng.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15]))

        for task_name in cfg.task_sequence:
            spec = TASK_SPECS[task_name]
            df = keyed[spec["source"]]
            col = spec["column"]
            if col not in df.columns:
                raise KeyError(f"Task '{task_name}': column '{col}' missing in {spec['source']} data.")
            frame = pd.DataFrame(index=df.index)
            values = df[col]
            if task_name == "material_type":
                values = values.map(_MATERIAL_TYPE_MERGE)
            if spec["source"] != "qc" and spec["kind"] == "reg":
                v = np.log1p(df[col].astype(float).clip(lower=0.0))
                is_train = np.array([split_by_key.get(str(k)) == "train" for k in df.index])
                ref = v[is_train] if is_train.any() else v
                mean = float(ref.mean())
                std = float(ref.std(ddof=0)) or 1.0
                values = ((v - mean) / std).clip(-_RAW_TARGET_CLIP, _RAW_TARGET_CLIP)
            frame[col] = values
            if spec["kind"] == "kr":
                frame[spec["t_column"]] = df[spec["t_column"]]
            frame["split"] = [split_by_key.get(str(k), "train") for k in frame.index]
            self.task_frames[task_name] = frame

        self.split_by_key = split_by_key
        n_keys = len(set().union(*[set(f.index) for f in self.task_frames.values()]))
        logger.info(f"Built {len(self.task_frames)} task frames over {n_keys} unique compositions; x_dim={self.x_dim}.")

    def _load_qc(self) -> pd.DataFrame:
        cfg = self.config
        df = pd.read_parquet(cfg.qc_data_path)
        if cfg.qc_preprocessing_path is not None and Path(cfg.qc_preprocessing_path).exists():
            dropped = joblib.load(cfg.qc_preprocessing_path).get("dropped_idx", [])
            df = df.loc[~df.index.isin(dropped)]
        return df

    @staticmethod
    def _stratified_qc_sample(df: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
        """Cap qc rows while keeping every minority (non-"others") material-type row."""
        labels = df["Material type (label)"]
        minority = df[labels != 4]
        others = df[labels == 4]
        n_others = max(cap - len(minority), 0)
        if n_others < len(others):
            others = others.iloc[rng.choice(len(others), size=n_others, replace=False)]
        out = pd.concat([minority, others])
        if len(out) > cap:
            out = out.iloc[rng.choice(len(out), size=cap, replace=False)]
        return out

    def _class_weights(self, task_name: str) -> list[float]:
        spec = TASK_SPECS[task_name]
        frame = self.task_frames[task_name]
        num_classes = int(spec["num_classes"])
        train = frame.loc[frame["split"] == "train", spec["column"]].dropna().astype(int)
        counts = np.bincount(train, minlength=num_classes).astype(float)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (num_classes * counts)
        return weights.tolist()

    def descriptor_fn(self, compositions: list[str]) -> pd.DataFrame:
        uncached = [c for c in dict.fromkeys(compositions) if c not in self._desc_cache]
        if uncached:
            weights = np.zeros((len(uncached), len(DEFAULT_ELEMENTS)), dtype=float)
            valid: list[str] = []
            for key in uncached:
                try:
                    w = formula_to_composition(key)
                except Exception:
                    w = None
                if w is None or float(w.sum()) <= 0:
                    continue
                weights[len(valid)] = w
                valid.append(key)
            if valid:
                desc = self._kmd.transform(weights[: len(valid)])
                for j, key in enumerate(valid):
                    self._desc_cache[key] = desc[j]
        present = [c for c in compositions if c in self._desc_cache]
        if not present:
            return pd.DataFrame()
        return pd.DataFrame(np.stack([self._desc_cache[c] for c in present]), index=present)

    # ------------------------------------------------------------------ configs

    def _build_task_config(self, task_name: str):
        cfg = self.config
        spec = TASK_SPECS[task_name]
        ld, hd = cfg.latent_dim, cfg.head_hidden_dim
        if spec["kind"] == "reg":
            return RegressionTaskConfig(
                name=task_name,
                data_column=spec["column"],
                dims=[ld, hd, 1],
                optimizer=OptimizerConfig(lr=cfg.head_lr, weight_decay=1e-5),
            )
        if spec["kind"] == "clf":
            return ClassificationTaskConfig(
                name=task_name,
                data_column=spec["column"],
                dims=[ld, hd, 32],
                num_classes=spec["num_classes"],
                class_weights=self._class_weights(task_name),
                optimizer=OptimizerConfig(lr=cfg.head_lr, weight_decay=1e-5),
            )
        train_t = self._collect_train_t(task_name)
        centers, sigmas = _init_kernels(train_t, cfg.n_kernel)
        return KernelRegressionTaskConfig(
            name=task_name,
            data_column=spec["column"],
            t_column=spec["t_column"],
            x_dim=[ld, 128, 64],
            t_dim=[16, 8],
            kernel_num_centers=cfg.n_kernel,
            kernel_centers_init=centers or None,
            kernel_sigmas_init=sigmas or None,
            kernel_learnable_centers=True,
            kernel_learnable_sigmas=True,
            enable_mu3=False,
            optimizer=OptimizerConfig(lr=cfg.kr_lr, weight_decay=cfg.kr_decay),
        )

    def _collect_train_t(self, task_name: str) -> np.ndarray:
        spec = TASK_SPECS[task_name]
        frame = self.task_frames[task_name]
        mask = frame[spec["column"]].notna() & (frame["split"] == "train")
        cells = frame.loc[mask, spec["t_column"]].dropna()
        if cells.empty:
            return np.array([])
        return np.concatenate([_as_float_array(c) for c in cells])

    # ------------------------------------------------------------------ run

    def _build_empty_model(self) -> FlexibleMultiTaskModel:
        """The bare model used as the starting point for both ``run`` and ``run_inverse_only``."""
        cfg = self.config
        encoder_config = MLPEncoderConfig(hidden_dims=[self.x_dim, cfg.encoder_hidden, cfg.latent_dim])
        return FlexibleMultiTaskModel(
            task_configs=[],
            encoder_config=encoder_config,
            enable_autoencoder=True,
            shared_block_optimizer=OptimizerConfig(lr=cfg.encoder_lr, weight_decay=1e-2),
        )

    def _build_full_model(self) -> FlexibleMultiTaskModel:
        """Rebuild the post-training model (all tasks added in sequence order) so a saved
        ``final_model.pt`` ``state_dict`` can be loaded for inverse-only runs."""
        model = self._build_empty_model()
        for task_name in self.config.task_sequence:
            model.add_task(self._build_task_config(task_name))
        return model

    def run(self) -> None:
        cfg = self.config
        seed_everything(cfg.random_seed, workers=True)
        model = self._build_empty_model()

        task_configs: dict[str, Any] = {}
        metric_history: dict[str, list[tuple[int, float]]] = {name: [] for name in cfg.task_sequence}
        records: list[dict[str, Any]] = []
        fixed_tail = set(cfg.fixed_tail)

        for step, task_name in enumerate(cfg.task_sequence):
            logger.info(f"=== Step {step + 1}/{len(cfg.task_sequence)}: add task '{task_name}' ===")
            task_configs[task_name] = self._build_task_config(task_name)
            model.add_task(task_configs[task_name])

            active = cfg.task_sequence[: step + 1]
            # New task fully active; old tasks replayed — fixed-tail tasks at the higher ratio.
            for name in active:
                if name == task_name:
                    ratio = 1.0
                elif name in fixed_tail:
                    ratio = cfg.replay_ratio_high
                else:
                    ratio = cfg.replay_ratio
                task_configs[name].task_masking_ratio = ratio

            datamodule = CompoundDataModule(
                task_configs=[task_configs[name] for name in active],
                descriptor_fn=self.descriptor_fn,
                task_frames={name: self.task_frames[name] for name in active},
                composition_column="composition",
                random_seed=cfg.datamodule_random_seed,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )
            callbacks: list[Callback] = [
                EarlyStopping(
                    monitor="val_final_loss",
                    mode="min",
                    patience=cfg.early_stop_patience,
                    min_delta=cfg.early_stop_min_delta,
                )
            ]
            trainer = Trainer(
                max_epochs=cfg.max_epochs_per_step,
                accelerator=cfg.accelerator,
                devices=cfg.devices,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=callbacks,
            )
            trainer.fit(model, datamodule=datamodule)

            test_keys: set[str] | None = None
            if datamodule.split_series is not None:
                resolved = datamodule.split_series
                test_keys = set(resolved.index[resolved == "test"].astype(str))

            step_dir = self.training_dir / f"step{step + 1:02d}_{task_name}"
            step_dir.mkdir(parents=True, exist_ok=True)
            step_metrics: dict[str, dict[str, float]] = {}
            for name in active:
                # Plot only the freshly-added head; dump raw (composition, true, pred) + per-task
                # metrics.json for every active head so the forgetting trajectory is backed by
                # raw data and per-task numbers at each stage.
                metric = self._evaluate_task(model, name, step_dir, is_new=(name == task_name), test_keys=test_keys)
                step_metrics[name] = metric
                metric_history[name].append((step + 1, metric["primary"]))
            # Per-step model checkpoint (mirrors the demo, PR #18). Lets analysts revisit any
            # intermediate stage ("what did the encoder look like just after task K was added?")
            # without retraining the prefix, and feeds downstream finetune scripts.
            step_ckpt = step_dir / "checkpoint.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "task_sequence": list(cfg.task_sequence),
                    "step": step + 1,
                    "new_task": task_name,
                    "active_tasks": list(active),
                },
                step_ckpt,
            )
            records.append(
                {"step": step + 1, "new_task": task_name, "epochs_run": trainer.current_epoch, "metrics": step_metrics}
            )
            summary = ", ".join(f"{k}={v['primary']:.3f}" for k, v in step_metrics.items())
            rel_ckpt = step_ckpt.relative_to(self.output_dir)
            logger.info(f"Step {step + 1} ({trainer.current_epoch} epochs): {summary} (ckpt: {rel_ckpt})")

        self._plot_forgetting(metric_history)
        (self.training_dir / "experiment_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        self._write_metrics_table(records)
        self._save_final_model(model, task_configs)

        inverse = self._inverse_design(model)
        (self.inverse_root / "inverse_design.json").write_text(json.dumps(inverse, indent=2), encoding="utf-8")

        # Slide-prep deliverables (plan §6) — no more PPT/HTML; the slide author works from
        # SLIDE_PREP.md + the raw arrays + the standard image set. The three scenarios are
        # treated as equal first-class results — no demo-style "headline scenario" promotion.
        self._write_inverse_summary_md(inverse)
        self._write_analysis_md(records, inverse)
        self._write_slide_prep_md(records, inverse)
        self._write_readme(records, inverse)
        logger.info(f"Done. Outputs in {self.output_dir}")

    def _save_final_model(self, model, task_configs: dict[str, Any]) -> None:
        # Schema matches the demo's ``final_model.pt`` (PR #18) so the same downstream consumers —
        # ``paper_inverse_comparison.py`` / ``finetune_inverse_heads.py`` / ``--inverse-only`` —
        # can ingest checkpoints from either runner without translation.
        ckpt = self.training_dir / "final_model.pt"
        torch.save({"model": model.state_dict(), "task_sequence": list(self.config.task_sequence)}, ckpt)
        spec_dump = {
            name: {
                "kind": TASK_SPECS[name]["kind"],
                "column": TASK_SPECS[name]["column"],
                "source": TASK_SPECS[name]["source"],
            }
            for name in self.config.task_sequence
        }
        (self.training_dir / "final_model_taskconfigs.json").write_text(
            json.dumps(spec_dump, indent=2), encoding="utf-8"
        )
        logger.info(f"Saved final model checkpoint to {ckpt}")

    def run_inverse_only(self, ckpt_path: Path) -> None:
        """Skip training; load a saved ``final_model.pt`` and run only the inverse-design stage.

        Use this to iterate on inverse-design knobs (palette, seeds, scenarios, …) without
        repeating the multi-hour training. Data loading + descriptor computation still happen —
        they're prerequisites for seed selection and the composition-path kernel — but no
        ``Trainer.fit`` is called.
        """
        logger.info(f"=== Inverse-only mode: loading model checkpoint {ckpt_path} ===")
        seed_everything(self.config.random_seed, workers=True)
        model = self._build_full_model()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict)
        model.eval()
        inverse = self._inverse_design(model)
        (self.inverse_root / "inverse_design.json").write_text(json.dumps(inverse, indent=2), encoding="utf-8")
        self._write_inverse_summary_md(inverse)
        logger.info(f"Inverse-only done. Outputs in {self.output_dir}")

    def _write_metrics_table(self, records: list[dict[str, Any]]) -> None:
        final = records[-1]["metrics"] if records else {}
        intro = {r["new_task"]: r["metrics"][r["new_task"]] for r in records}
        rows = []
        for task in self.config.task_sequence:
            spec = TASK_SPECS[task]
            metric_name = "accuracy" if spec["kind"] == "clf" else "R2"
            rows.append(
                {
                    "task": task,
                    "display": _display(task),
                    "type": KIND_LABEL[spec["kind"]],
                    "dataset": SOURCE_DISPLAY[spec["source"]],
                    "metric": metric_name,
                    "at_intro": intro.get(task, {}).get("primary", float("nan")),
                    "final": final.get(task, {}).get("primary", float("nan")),
                    "final_mae": final.get(task, {}).get("mae", float("nan")),
                    "samples": final.get(task, {}).get("samples", 0),
                }
            )
        pd.DataFrame(rows).to_csv(self.training_dir / "metrics_table.csv", index=False)

    # ------------------------------------------------------------------ eval

    def _test_rows(self, task_name: str, test_keys: set[str] | None = None) -> list[str]:
        spec = TASK_SPECS[task_name]
        frame = self.task_frames[task_name]
        mask = frame[spec["column"]].notna()
        mask &= frame.index.isin(test_keys) if test_keys is not None else (frame["split"] == "test")
        return list(frame.index[mask])

    def _descriptor_tensor(self, comps: list[str], device) -> tuple[torch.Tensor, list[str]]:
        desc = self.descriptor_fn(comps)
        comps = [c for c in comps if c in desc.index]
        return torch.tensor(desc.loc[comps].values, dtype=torch.float32, device=device), comps

    def _evaluate_task(self, model, task_name, step_dir, *, is_new, test_keys=None) -> dict[str, float]:
        spec = TASK_SPECS[task_name]
        kind = spec["kind"]
        model.eval()
        device = next(model.parameters()).device
        comps = self._test_rows(task_name, test_keys)
        if not comps:
            return {"primary": float("nan"), "samples": 0}
        frame = self.task_frames[task_name]
        head = model.task_heads[task_name]

        with torch.no_grad():
            if kind in ("reg", "clf"):
                x, comps = self._descriptor_tensor(comps, device)
                if not comps:
                    return {"primary": float("nan"), "samples": 0}
                h = torch.tanh(model.encoder(x))
                if kind == "reg":
                    pred = head(h).squeeze(-1).cpu().numpy()
                    true = frame.loc[comps, spec["column"]].astype(float).to_numpy()
                    r2 = float(r2_score(true, pred))
                    metric = {
                        "r2": r2,
                        "mae": float(mean_absolute_error(true, pred)),
                        "samples": len(comps),
                        "primary": r2,
                    }
                    self._dump_predictions(task_name, step_dir, comps=list(comps), true=true, pred=pred)
                    self._dump_metrics(task_name, step_dir, metric)
                    if is_new:
                        self._plot_parity(true, pred, task_name, r2, step_dir)
                    return metric
                logits = head(h)
                pred = logits.argmax(dim=-1).cpu().numpy()
                true = frame.loc[comps, spec["column"]].astype(int).to_numpy()
                acc = float(accuracy_score(true, pred))
                metric = {
                    "accuracy": acc,
                    "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
                    "samples": len(comps),
                    "primary": acc,
                }
                self._dump_predictions(task_name, step_dir, comps=list(comps), true=true, pred=pred)
                self._dump_metrics(task_name, step_dir, metric)
                if is_new:
                    self._plot_confusion(true, pred, task_name, acc, step_dir, spec["num_classes"])
                return metric

            # kernel regression
            keep, t_list, true_parts = [], [], []
            for comp in comps:
                if comp not in self._desc_cache and self.descriptor_fn([comp]).empty:
                    continue
                y_arr = _as_float_array(frame.at[comp, spec["column"]])
                t_arr = _as_float_array(frame.at[comp, spec["t_column"]])
                if y_arr.size == 0 or y_arr.size != t_arr.size:
                    continue
                keep.append(comp)
                t_list.append(torch.tensor(t_arr, dtype=torch.float32, device=device))
                true_parts.append(y_arr)
            if not keep:
                return {"primary": float("nan"), "samples": 0}
            xk, _ = self._descriptor_tensor(keep, device)
            h_k = torch.tanh(model.encoder(xk))
            expanded_h, expanded_t = model._expand_for_kernel_regression(h_k, t_list)
            pred = head(expanded_h, t=expanded_t).squeeze(-1).cpu().numpy()
            true = np.concatenate(true_parts)
            r2 = float(r2_score(true, pred))
            metric = {
                "r2": r2,
                "mae": float(mean_absolute_error(true, pred)),
                "samples": len(keep),
                "points": int(true.size),
                "primary": r2,
            }
            self._dump_kr_predictions(
                task_name,
                step_dir,
                comps=keep,
                t_list=[t.cpu().numpy() for t in t_list],
                true_parts=true_parts,
                pred=pred,
            )
            self._dump_metrics(task_name, step_dir, metric)
            if is_new:
                self._plot_kr_sequences(keep, t_list, true_parts, pred, task_name, step_dir)
            return metric

    # --- per-task artifact dump helpers (PR #18 demo factoring) ---------------

    def _dump_predictions(self, task_name: str, step_dir: Path, *, comps: list[str], true, pred) -> None:
        """Persist ``(composition, true, pred)`` for a regression or classification task."""
        pd.DataFrame({"composition": comps, "true": true, "pred": pred}).to_parquet(
            step_dir / f"{task_name}_pred.parquet"
        )

    def _dump_kr_predictions(
        self,
        task_name: str,
        step_dir: Path,
        *,
        comps: list[str],
        t_list: list[np.ndarray],
        true_parts: list[np.ndarray],
        pred,
    ) -> None:
        """Persist KR test predictions in long-form: one row per ``(composition, t)``."""
        rows: list[dict[str, object]] = []
        offset = 0
        for comp, t_arr, y_true in zip(comps, t_list, true_parts):
            n = int(y_true.size)
            for k in range(n):
                rows.append(
                    {
                        "composition": comp,
                        "t": float(t_arr[k]),
                        "true": float(y_true[k]),
                        "pred": float(pred[offset + k]),
                    }
                )
            offset += n
        pd.DataFrame(rows).to_parquet(step_dir / f"{task_name}_pred.parquet")

    def _dump_metrics(self, task_name: str, step_dir: Path, metric: dict[str, float]) -> None:
        """Persist the per-task metric dict alongside the parquet for easy human / scripted inspection."""
        (step_dir / f"{task_name}_metrics.json").write_text(json.dumps(metric, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ inverse design

    @staticmethod
    def _element_system(composition: str) -> frozenset[str]:
        """Element symbols (no amounts) in a composition string — used for system-level dedup."""
        return frozenset(re.findall(r"[A-Z][a-z]?", composition))

    @classmethod
    def _dedupe_by_element_system(cls, candidates: list[str], n: int) -> list[str]:
        """Walk ``candidates`` in order, keep the first occurrence of each element set, cap at ``n``."""
        seen: set[frozenset[str]] = set()
        out: list[str] = []
        for comp in candidates:
            key = cls._element_system(comp)
            if key in seen:
                continue
            seen.add(key)
            out.append(comp)
            if len(out) >= n:
                break
        return out

    def _select_seeds(self, model, device, qc_prob_fn) -> dict[str, list[str]]:
        """Pick seed compositions for inverse design (mirrors demo's PR #18 behaviour).

        Returns ``{"strategy_seeds": […], "explicit_seeds": […]}``. Element-system dedup keeps the
        best representative per element set (so 17 strategy seeds = 17 distinct alloy families,
        not 17 ratio variants of three). ``inverse_seed_explicit_append`` is fail-fast validated
        (each appended composition must have a computable descriptor) and the strategy budget is
        reduced by its length so the total length equals ``inverse_n_seeds``.
        """
        cfg = self.config
        n = cfg.inverse_n_seeds

        # Pre-validate the explicit-append seeds so we fail fast on bad input.
        appended: list[str] = []
        for raw in cfg.inverse_seed_explicit_append:
            norm = normalize_composition(raw) or str(raw)
            if norm not in self._desc_cache and self.descriptor_fn([norm]).empty:
                raise ValueError(
                    f"inverse_seed_explicit_append entry {raw!r} has no computable descriptor "
                    "(check the formula and that all elements are in DEFAULT_ELEMENTS)."
                )
            appended.append(norm)
        # Dedup the appended list itself (in case the user listed near-duplicates).
        appended = self._dedupe_by_element_system(appended, len(appended))
        n_strategy = max(0, n - len(appended))

        def _finalise(strategy_seeds: list[str]) -> dict[str, list[str]]:
            """Combine strategy seeds + explicit-append, skipping any duplicate element systems."""
            seen_keys = {self._element_system(c) for c in appended}
            kept_strategy = [c for c in strategy_seeds if self._element_system(c) not in seen_keys][:n_strategy]
            return {"strategy_seeds": kept_strategy, "explicit_seeds": appended}

        if cfg.inverse_seed_strategy == "explicit":
            seeds = [normalize_composition(c) or str(c) for c in cfg.inverse_seed_compositions]
            seeds = [c for c in seeds if c in self._desc_cache or not self.descriptor_fn([c]).empty]
            return _finalise(self._dedupe_by_element_system(seeds, n_strategy))

        # Candidate pool: chosen split of the material_type frame, with a valid descriptor.
        frame = self.task_frames["material_type"]
        index = (
            frame.index if cfg.inverse_seed_split == "all" else frame.index[frame["split"] == cfg.inverse_seed_split]
        )
        pool = [c for c in index if c in self._desc_cache or not self.descriptor_fn([c]).empty]
        if not pool:
            return {"strategy_seeds": [], "explicit_seeds": appended}

        if cfg.inverse_seed_strategy == "random":
            rng = np.random.default_rng(cfg.random_seed)
            shuffled = [pool[i] for i in rng.permutation(len(pool))]
            return _finalise(self._dedupe_by_element_system(shuffled, n_strategy))

        # "top_qc": highest predicted QC probability, then element-system dedup.
        x, pool = self._descriptor_tensor(pool, device)
        probs = qc_prob_fn(x)
        ranked = [pool[i] for i in np.argsort(probs)[::-1]]
        return _finalise(self._dedupe_by_element_system(ranked, n_strategy))

    def _decode_compositions_from_descriptor(self, descriptors: np.ndarray) -> list[str]:
        """Latent-path composition output: AE-decoded descriptor → KMD.inverse → formula string."""
        try:
            weights = self._kmd.inverse(descriptors)
        except Exception as exc:  # pragma: no cover - QP edge cases
            logger.warning(f"KMD.inverse failed ({exc}); skipping composition decoding.")
            return ["<undecodable>"] * descriptors.shape[0]
        return _format_weights(weights)

    def _inverse_design(self, model) -> dict[str, Any]:
        """Run the 8 inverse-design configurations against each scenario on the same seeds.

        The configurations are defined at module level in :data:`INVERSE_PATH_CONFIGS`, mirroring
        the demo's ``paper_inverse_comparison.py``:

          * **latent** (3 rows): ``optimize_latent`` with ``ae_align_scale ∈ {0.0, 0.25, 1.0}``
            (failure / mid / max alignment).
          * **composition** (5 rows): ``optimize_composition`` with seed_blend / palette / diversity
            knobs swept — strict seed, blended seed, blended + palette, blended + palette + low
            diversity, and random init (no seed) as the no-seed-bias control.

        Saves per-path JSON + plot under ``inverse_design/<scenario>/<path>/`` plus a per-scenario
        ``summary.json`` aggregating headline stats, and a top-level ``seeds.json`` recording the
        strategy- vs explicit-appended seed split.
        """
        cfg = self.config
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        model.eval()
        inv_root = self.output_dir / "inverse_design"
        inv_root.mkdir(parents=True, exist_ok=True)

        def _qc_prob(x: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                h = torch.tanh(model.encoder(x))
                probs = torch.softmax(model.task_heads["material_type"](h), dim=-1)
                return probs[:, QC_CLASSES].sum(dim=-1).cpu().numpy()

        def _reg_preds(x: torch.Tensor, tasks: list[str]) -> dict[str, np.ndarray]:
            with torch.no_grad():
                h = torch.tanh(model.encoder(x))
                return {t: model.task_heads[t](h).squeeze(-1).cpu().numpy() for t in tasks}

        # Same seeds for every scenario, so the four paths are directly comparable.
        seed_split = self._select_seeds(model, device, _qc_prob)
        seeds_all = seed_split["strategy_seeds"] + seed_split["explicit_seeds"]
        if not seeds_all:
            logger.warning("No seeds available for inverse design.")
            return {}
        x_seed, seeds = self._descriptor_tensor(seeds_all, device)
        if not seeds:
            logger.warning("No seeds have computable descriptors; aborting inverse design.")
            return {}

        # Composition path shares: kernel + per-seed initial weight tensor (B, n_components).
        kmd_kernel = self._kmd.kernel_torch(device=device, dtype=dtype)
        w_seed = _seed_weights_from_compositions(seeds, n_components=len(DEFAULT_ELEMENTS)).to(
            device=device, dtype=dtype
        )

        # Top-level seeds.json with the strategy / explicit split (single source of truth across
        # all scenarios). Per-path subdirs record their own ``seeds`` field for completeness.
        seeds_meta = {
            "strategy_strategy": cfg.inverse_seed_strategy,
            "strategy_split": cfg.inverse_seed_split,
            "n_target": cfg.inverse_n_seeds,
            "n_used": len(seeds),
            "strategy_seeds": [c for c in seed_split["strategy_seeds"] if c in seeds],
            "explicit_seeds": [c for c in seed_split["explicit_seeds"] if c in seeds],
            "all_seeds_used": seeds,
        }
        (inv_root / "seeds.json").write_text(json.dumps(seeds_meta, indent=2), encoding="utf-8")

        # Union of element symbols present in any seed — used by the element-frequency
        # heatmap to flag "discovered" elements (high occurrence but not in any seed).
        seed_element_pool: set[str] = set()
        for c in seeds:
            seed_element_pool |= self._element_system(c)

        out: dict[str, Any] = {"seeds": seeds_meta, "scenarios": {}}
        for sc in cfg.inverse_scenarios:
            logger.info(f"=== Inverse design [{sc.name}]: targets={dict(zip(sc.reg_tasks, sc.reg_targets))} ===")
            sc_dir = inv_root / sc.name
            sc_dir.mkdir(parents=True, exist_ok=True)
            reg_targets = {t: v for t, v in zip(sc.reg_tasks, sc.reg_targets)}

            # Per-scenario targets.json (plan §5) — separate from results so a slide author can
            # quote the objective without parsing the full result dump.
            (sc_dir / "targets.json").write_text(
                json.dumps(
                    {
                        "name": sc.name,
                        "primary": {"task": "material_type", "class_indices": QC_CLASSES, "direction": "max"},
                        "secondary": [
                            {"task": t, "target": v, "direction": "min" if v < 0 else "max"}
                            for t, v in reg_targets.items()
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            before_qc = _qc_prob(x_seed)
            before_reg = _reg_preds(x_seed, sc.reg_tasks)

            paths: dict[str, dict[str, Any]] = {}
            for path_cfg in INVERSE_PATH_CONFIGS:
                key = path_cfg["key"]
                path_dir = sc_dir / key
                if path_cfg["method"] == "latent":
                    paths[key] = self._run_latent_path(
                        model,
                        x_seed,
                        seeds,
                        reg_targets,
                        path_dir,
                        ae_align_scale=path_cfg["ae_align_scale"],
                        label=path_cfg["label"],
                        _qc_prob_fn=_qc_prob,
                        _reg_preds_fn=_reg_preds,
                    )
                else:
                    # Composition row: resolve the palette sentinel and seed/random init.
                    allowed = (
                        list(cfg.inverse_composition_allowed_elements)
                        if path_cfg["allowed"] == _PALETTE_SENTINEL
                        else path_cfg["allowed"]
                    )
                    init = path_cfg["init"]
                    paths[key] = self._run_composition_path(
                        model,
                        kmd_kernel,
                        w_seed if init == "seed" else None,
                        seeds,
                        reg_targets,
                        path_dir,
                        init=init,
                        blend=path_cfg["blend"] if init == "seed" else None,
                        allowed=allowed,
                        diversity=path_cfg["diversity"],
                        label=path_cfg["label"],
                        _qc_prob_fn=_qc_prob,
                        _reg_preds_fn=_reg_preds,
                    )

            scenario_summary = {
                "name": sc.name,
                "reg_targets": reg_targets,
                "n_seeds": len(seeds),
                "qc_before_mean": float(before_qc.mean()),
                "paths": {
                    path_name: {
                        "qc_after_mean": float(np.mean(p["qc_after_decode"])),
                        "qc_after_std": float(np.std(p["qc_after_decode"])),
                        "reg_after_decode_mean": {t: float(np.mean(p["reg_after_decode"][t])) for t in reg_targets},
                        "reg_after_decode_std": {t: float(np.std(p["reg_after_decode"][t])) for t in reg_targets},
                    }
                    for path_name, p in paths.items()
                },
            }
            (sc_dir / "summary.json").write_text(json.dumps(scenario_summary, indent=2), encoding="utf-8")
            self._plot_inverse_scenario(sc, before_qc, before_reg, paths, reg_targets, sc_dir)
            self._element_frequency_heatmap(sc.name, paths, seed_element_pool, sc_dir / "element_frequency_heatmap.png")

            qc_summary = " · ".join(
                f"{name}={paths[name]['qc_after_decode'] and np.mean(paths[name]['qc_after_decode']):.3f}"
                for name in INVERSE_PATHS
            )
            logger.info(f"[{sc.name}] QC after-decode mean — {qc_summary}")

            out["scenarios"][sc.name] = {**scenario_summary, "paths_details": paths}
        return out

    # --- inverse path runners -------------------------------------------------

    def _run_latent_path(
        self,
        model,
        x_seed: torch.Tensor,
        seeds: list[str],
        reg_targets: dict[str, float],
        path_dir: Path,
        *,
        ae_align_scale: float,
        label: str,
        _qc_prob_fn,
        _reg_preds_fn,
    ) -> dict[str, Any]:
        """Latent-space optimisation with cycle-consistency at a fixed ``ae_align_scale``."""
        cfg = self.config
        path_dir.mkdir(parents=True, exist_ok=True)
        reg_names = list(reg_targets)

        before_qc = _qc_prob_fn(x_seed)
        before_reg = _reg_preds_fn(x_seed, reg_names)

        res = model.optimize_latent(
            initial_input=x_seed,
            task_targets=reg_targets,
            class_targets={"material_type": QC_CLASSES},
            class_target_weight=cfg.inverse_class_weight,
            ae_align_scale=ae_align_scale,
            optimize_space="latent",
            steps=cfg.inverse_steps,
            lr=cfg.inverse_lr,
        )
        achieved_latent = res.optimized_target[:, 0, :].cpu().numpy()
        optimized_desc = res.optimized_input[:, 0, :]
        optimized_desc_np = optimized_desc.detach().cpu().numpy()
        after_qc = _qc_prob_fn(optimized_desc)
        after_reg = _reg_preds_fn(optimized_desc, reg_names)
        try:
            optimized_weights = self._kmd.inverse(optimized_desc_np)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"KMD.inverse failed for latent path ({exc}); weights left empty.")
            optimized_weights = np.zeros((optimized_desc_np.shape[0], len(DEFAULT_ELEMENTS)))
        decoded = _format_weights(optimized_weights)

        result = {
            "method": "latent",
            "label": label,
            "ae_align_scale": ae_align_scale,
            "seeds": list(seeds),
            "qc_before": before_qc.tolist(),
            "qc_after_decode": after_qc.tolist(),
            "reg_before": {t: before_reg[t].tolist() for t in reg_names},
            "reg_achieved_latent": {t: achieved_latent[:, j].tolist() for j, t in enumerate(reg_names)},
            "reg_after_decode": {t: after_reg[t].tolist() for t in reg_names},
            "decoded_composition": decoded,
            "optimized_descriptor": optimized_desc_np.tolist(),
            "optimized_weights": optimized_weights.tolist(),
        }
        (path_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _run_composition_path(
        self,
        model,
        kmd_kernel: torch.Tensor,
        w_seed: torch.Tensor | None,
        seeds: list[str],
        reg_targets: dict[str, float],
        path_dir: Path,
        *,
        init: str,
        blend: float | None,
        allowed: str | list[str],
        diversity: float,
        label: str,
        _qc_prob_fn,
        _reg_preds_fn,
    ) -> dict[str, Any]:
        """Composition-space optimisation via differentiable KMD (``optimize_composition``).

        ``init="seed"`` uses ``w_seed`` + ``seed_blend``; ``init="random"`` ignores ``w_seed`` and
        runs ``n_starts = len(seeds)`` so the per-row budget matches the latent run.
        """
        cfg = self.config
        path_dir.mkdir(parents=True, exist_ok=True)
        reg_names = list(reg_targets)

        if init == "seed":
            if w_seed is None:
                raise ValueError("Composition path with init='seed' requires w_seed.")
            init_kwargs: dict[str, Any] = {"initial_weights": w_seed, "seed_blend": blend}
        elif init == "random":
            init_kwargs = {"initial_weights": None, "n_starts": len(seeds)}
        else:
            raise ValueError(f"Unknown init mode in composition path: {init!r}")

        res = model.optimize_composition(
            kmd_kernel,
            task_targets=reg_targets,
            class_targets={"material_type": QC_CLASSES},
            class_target_weight=cfg.inverse_class_weight,
            diversity_scale=diversity,
            allowed_elements=allowed,
            steps=cfg.inverse_steps,
            lr=cfg.inverse_lr,
            **init_kwargs,
        )
        # Composition's result tensors are 2D — ``(B, x_dim)`` / ``(B, n_components)`` /
        # ``(B, T)`` — no restart axis, so no ``[:, 0, :]`` slicing (unlike ``optimize_latent``).
        optimized_desc = res.optimized_descriptor  # (B, x_dim) — w @ K, no AE round-trip
        optimized_desc_np = optimized_desc.detach().cpu().numpy()
        w_final = res.optimized_weights.detach().cpu().numpy()
        achieved_latent = res.optimized_target.detach().cpu().numpy()  # (B, T)
        after_qc = _qc_prob_fn(optimized_desc)
        after_reg = _reg_preds_fn(optimized_desc, reg_names)
        decoded = _format_weights(w_final)

        # Random init has no per-row correspondence with the seed list — preserve the seed list
        # only when the init was seeded; otherwise label the rows as random restarts.
        seed_labels = list(seeds) if init == "seed" else [f"random_start_{i}" for i in range(len(seeds))]

        result = {
            "method": "composition",
            "label": label,
            "init": init,
            "seed_blend": blend,
            "allowed_elements": allowed,
            "diversity_scale": diversity,
            "seeds": seed_labels,
            "qc_after_decode": after_qc.tolist(),
            "reg_achieved_latent": {t: achieved_latent[:, j].tolist() for j, t in enumerate(reg_names)},
            "reg_after_decode": {t: after_reg[t].tolist() for t in reg_names},
            "decoded_composition": decoded,
            "optimized_descriptor": optimized_desc_np.tolist(),
            "optimized_weights": w_final.tolist(),
        }
        (path_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    # ------------------------------------------------------------------ plots

    def _plot_parity(self, true, pred, task_name, r2, step_dir):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(true, pred, s=14, alpha=0.55, color=_SCATTER_COLOR, edgecolor="none")
        lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], color="#444444", ls="--", lw=1.2, label="ideal")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(_title(task_name))
        ax.text(
            0.04,
            0.96,
            f"R² = {r2:.3f}\nn = {len(true)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d0d0d0", alpha=0.9),
        )
        ax.legend(loc="lower right")
        fig.savefig(step_dir / f"{task_name}_parity.png")
        plt.close(fig)

    def _plot_confusion(self, true, pred, task_name, acc, step_dir, num_classes):
        counts = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(true, pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                counts[t, p] += 1
        if task_name == "material_type":
            labels = MATERIAL_TYPE_DISPLAY_ORDER[:num_classes]
            perm = [MATERIAL_TYPE_CLASSES.index(lbl) for lbl in labels]
        else:
            labels = [str(i) for i in range(num_classes)]
            perm = list(range(num_classes))
        counts = counts[np.ix_(perm, perm)]
        row_sums = counts.sum(axis=1, keepdims=True)
        row_frac = np.divide(counts, row_sums, out=np.zeros(counts.shape, dtype=float), where=row_sums > 0)
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        im = ax.imshow(row_frac, cmap="Blues", vmin=0.0, vmax=1.0, origin="lower")
        fig.colorbar(im, ax=ax, label="row-normalized fraction (recall)", fraction=0.046, pad=0.04)
        ax.set_xticks(range(num_classes), labels, rotation=45, ha="right")
        ax.set_yticks(range(num_classes), labels)
        for i in range(num_classes):
            for j in range(num_classes):
                if counts[i, j]:
                    ax.text(
                        j,
                        i,
                        f"{row_frac[i, j] * 100:.0f}%\n{counts[i, j]}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if row_frac[i, j] > 0.5 else "#333333",
                    )
        ax.grid(False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(_display(task_name))
        ax.text(
            0.5,
            -0.22,
            f"accuracy = {acc:.3f}  ·  n = {int(counts.sum())}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        fig.savefig(step_dir / f"{task_name}_confusion.png")
        plt.close(fig)

    def _plot_kr_sequences(self, comps, t_list, true_parts, pred, task_name, step_dir):
        k = min(3, len(comps))
        fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 3.7), squeeze=False)
        offset = 0
        line_true = line_pred = None
        for i in range(k):
            ax = axes[0][i]
            n = true_parts[i].size
            t = t_list[i].cpu().numpy()
            true_i = np.asarray(true_parts[i])
            pred_i = pred[offset : offset + n]
            order = np.argsort(t)
            (line_true,) = ax.plot(t[order], true_i[order], color="#444444", lw=1.8, label="True")
            # Same blue as every regression parity scatter — keeps "Predicted" colour consistent
            # across regression / kernel-regression panels (mirrors the demo's fix in PR #18).
            (line_pred,) = ax.plot(t[order], pred_i[order], color=_SCATTER_COLOR, lw=1.6, ls="--", label="Predicted")
            ax.set_xlabel("t")
            if i == 0:
                ax.set_ylabel("Value")
            r2_i = float(r2_score(true_i, pred_i)) if n >= 2 and float(np.var(true_i)) > 0 else float("nan")
            ax.text(
                0.96,
                0.96,
                f"R² = {r2_i:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d0d0d0", alpha=0.9),
            )
            ax.set_title(comps[i], fontsize=9)
            offset += n
        if line_true is not None:
            fig.legend(
                [line_true, line_pred],
                ["True", "Predicted"],
                loc="lower left",
                ncol=2,
                bbox_to_anchor=(0.0, 1.10),
                bbox_transform=axes[0][0].transAxes,
            )
        fig.suptitle(_title(task_name), y=1.24)
        fig.savefig(step_dir / f"{task_name}_sequences.png")
        plt.close(fig)

    def _plot_forgetting(self, metric_history):
        n_tasks = sum(1 for pts in metric_history.values() if pts)
        fig, ax = plt.subplots(figsize=(14, max(5.5, 0.32 * n_tasks + 3)))
        all_steps: set[int] = set()
        for task_name, points in metric_history.items():
            if not points:
                continue
            steps = [s for s, _ in points]
            vals = [v for _, v in points]
            all_steps.update(steps)
            is_clf = TASK_SPECS[task_name]["kind"] == "clf"
            ax.plot(
                steps,
                vals,
                marker="s" if is_clf else "o",
                ms=5,
                ls="--" if is_clf else "-",
                color=self._task_colors.get(task_name, "#888888"),
                label=_display(task_name) + (" · accuracy" if is_clf else ""),
            )
        if all_steps:
            ax.set_xticks(sorted(all_steps))
        ax.set_xlabel("Continual finetuning step (a new task is added at each step)")
        ax.set_ylabel("Primary metric  ·  R² (regression) / accuracy (classification)")
        ax.set_title("Per-task performance across continual finetuning")
        ncol = 1 if n_tasks <= 20 else 2
        ax.legend(fontsize=8, ncol=ncol, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        out_path = self.training_dir / "forgetting_trajectory.png"
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved forgetting trajectory to {out_path}")

    def _plot_inverse_scenario(
        self,
        scenario,
        before_qc: np.ndarray,
        before_reg: dict[str, np.ndarray],
        paths: dict[str, dict[str, Any]],
        reg_targets: dict[str, float],
        sc_dir: Path,
    ) -> None:
        """Compare the 8 inverse-design configurations side-by-side on QC + each reg target.

        Mirrors the demo's ``paper_inverse_comparison.py`` plot — same suptitle, panel titles
        (via ``REG_TASK_TITLES``), x-tick labels (``INVERSE_PATH_CONFIGS[*]["label"]``), and
        two-tone colour code (green ``#55A868`` for latent rows, blue ``#2563EB`` for composition
        rows). We keep our boxplot style (vs the demo's bar+errorbar) to surface the full per-seed
        distribution. Per the user override, the QC panel title is ``"Probability (QC)"``.
        """
        reg_names = list(reg_targets)
        n_panels = 1 + len(reg_names)
        fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 5.6), squeeze=False)
        axes = axes[0]

        configs_in_order = [c for c in INVERSE_PATH_CONFIGS if c["key"] in paths]
        path_labels = [c["label"] for c in configs_in_order]
        # Two-tone colour code, matching the demo.
        face_colors = ["#55A868" if c["method"] == "latent" else "#2563EB" for c in configs_in_order]
        x_pos = list(range(len(configs_in_order)))

        def _boxplot(ax, vals_per_path: list[list[float]]) -> None:
            """Two-tone per-row boxplot. Box face matches the row's method colour at α=0.25."""
            bp = ax.boxplot(
                vals_per_path,
                positions=x_pos,
                widths=0.6,
                patch_artist=True,
                medianprops=dict(color="#222222", lw=1.4),
                flierprops=dict(marker="o", mec="none", ms=3, alpha=0.55),
            )
            for patch, fc in zip(bp["boxes"], face_colors):
                patch.set(facecolor=fc, alpha=0.25, edgecolor=fc)
            for whisker, fc in zip(bp["whiskers"], [c for c in face_colors for _ in range(2)]):
                whisker.set_color(fc)
            for cap, fc in zip(bp["caps"], [c for c in face_colors for _ in range(2)]):
                cap.set_color(fc)
            for flier, fc in zip(bp["fliers"], face_colors):
                flier.set(markerfacecolor=fc)

        def _set_xticks(ax) -> None:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(path_labels, rotation=45, ha="right", fontsize=9)

        # Panel 1: QC probability. Title is the user-specified override "Probability (QC)";
        # ylabel + target line follow the demo.
        axq = axes[0]
        qc_vals = [list(paths[c["key"]]["qc_after_decode"]) for c in configs_in_order]
        _boxplot(axq, qc_vals)
        axq.axhline(1.0, color="#C44E52", ls="--", lw=1.4, label="target = 1.0")
        _set_xticks(axq)
        axq.set_ylim(-0.02, 1.05)
        axq.set_ylabel("P(quasicrystal)")
        axq.set_title("Probability (QC)")
        axq.legend(fontsize=9, loc="lower right")

        # Remaining panels: per regression target. Title pulled from REG_TASK_TITLES with units
        # and an arrow indicating whether the target is below (↓) or above (↑) the model's baseline.
        for ax, (t, tgt) in zip(axes[1:], reg_targets.items()):
            vals = [list(paths[c["key"]]["reg_after_decode"][t]) for c in configs_in_order]
            _boxplot(ax, vals)
            ax.axhline(tgt, color="#C44E52", ls="--", lw=1.4, label=f"target = {tgt:+.1f}")
            _set_xticks(ax)
            ax.set_ylabel("Predicted value")
            ax.set_title(REG_TASK_TITLES.get(t, t))
            ax.legend(fontsize=9, loc="best")

        fig.suptitle(
            "Inverse-design comparison: latent (ae_align_scale sweep) vs differentiable KMD (configs)",
            y=1.00,
        )
        out = sc_dir / "comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved inverse-design comparison plot to {out}")

    # ------------------------------------------------------------------ slide-prep (plan §6)

    def _counts(self) -> dict[str, int]:
        seq = self.config.task_sequence
        return {
            "n_tasks": len(seq),
            "n_reg": sum(1 for t in seq if TASK_SPECS[t]["kind"] == "reg"),
            "n_kr": sum(1 for t in seq if TASK_SPECS[t]["kind"] == "kr"),
            "n_clf": sum(1 for t in seq if TASK_SPECS[t]["kind"] == "clf"),
        }

    def _dataset_summary(self) -> list[tuple[str, int, int]]:
        """(dataset display, #tasks, #unique compositions used) per source, in stable order."""
        rows = []
        for src in ("qc", "phonix", "superconductor", "magnetic"):
            tasks = [t for t in self.config.task_sequence if TASK_SPECS[t]["source"] == src]
            if not tasks:
                continue
            keys = set().union(*[set(self.task_frames[t].index) for t in tasks])
            rows.append((SOURCE_DISPLAY[src], len(tasks), len(keys)))
        return rows

    def _final_target_metrics(self, records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        """Final-step metrics for the headline tasks the summary must report."""
        final = records[-1]["metrics"] if records else {}
        headline = ["formation_energy", "magnetic_moment", "tc", "klat", "material_type"]
        return {t: final.get(t, {}) for t in headline if t in self.config.task_sequence}

    # --- element-frequency heatmap (plan §6 / §5 evaluation) ------------------

    def _element_frequency_heatmap(
        self,
        scenario_name: str,
        paths: dict[str, dict[str, Any]],
        seed_element_pool: set[str],
        out_path: Path,
        *,
        top_k: int = 25,
        eps: float = 1e-3,
    ) -> None:
        """Per-path × top-K-element occurrence heatmap (rows = path, cols = element).

        ``optimized_weights`` in each path's ``result.json`` gives the (B, n_components) recipes;
        an element is "present" in a recipe when its weight > ``eps``. Cell value = #recipes
        containing the element (0..B). Elements absent from any seed (``seed_element_pool``) are
        highlighted on the x-axis label (**bold + orange**) as the inverse-design
        **element-discovery signal**. Orange (#E67E22) is chosen for high contrast against the
        Blues heatmap cmap and to stay visually distinct from the project's blue / green / red
        palette used elsewhere (composition bars / latent bars / target lines).
        """
        path_names = [p for p in INVERSE_PATHS if p in paths]
        if not path_names:
            return
        # Build per-path occurrence vector over all elements.
        n_elem = len(DEFAULT_ELEMENTS)
        occ = np.zeros((len(path_names), n_elem), dtype=int)
        for i, p in enumerate(path_names):
            w = np.asarray(paths[p].get("optimized_weights", []), dtype=float)
            if w.size == 0:
                continue
            occ[i] = (w > eps).sum(axis=0)
        total = occ.sum(axis=0)
        order = np.argsort(total)[::-1]
        keep = [int(k) for k in order if total[k] > 0][:top_k]
        if not keep:
            return
        labels = [DEFAULT_ELEMENTS[k] for k in keep]
        sub = occ[:, keep]

        fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(labels) + 2.0), 0.55 * len(path_names) + 2.4))
        im = ax.imshow(sub, cmap="Blues", aspect="auto")
        ax.set_yticks(range(len(path_names)), path_names)
        ax.set_xticks(range(len(labels)), labels, rotation=0, fontsize=9)
        # Bold + orange for "discovered" elements (not in any seed). No underline — bold + a
        # contrasting non-palette colour is enough to read at a glance, and underlining glyphs
        # under rotated/tight tick labels was visually noisy.
        for idx, sym in enumerate(labels):
            tick = ax.get_xticklabels()[idx]
            if sym not in seed_element_pool:
                tick.set_fontweight("bold")
                tick.set_color("#E67E22")
        # Cell annotations (counts).
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                if sub[i, j]:
                    ax.text(
                        j,
                        i,
                        str(int(sub[i, j])),
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if sub[i, j] > sub.max() * 0.55 else "#333333",
                    )
        fig.colorbar(im, ax=ax, label="# recipes containing element", fraction=0.025, pad=0.02)
        ax.set_title(
            f"{scenario_name} — element frequency (top {len(labels)})\nbold orange = discovered (not in any seed)",
            fontsize=11,
        )
        ax.grid(False)
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved element-frequency heatmap to {out_path}")

    # --- markdown writers (plan §6) -------------------------------------------

    def _write_inverse_summary_md(self, inverse: dict[str, Any]) -> None:
        """Compact cross-scenario summary (plan §6).

        Scenarios have **heterogeneous** regression-target sets (e.g. scenario2 has 3 reg targets
        vs 2 for the others), so a single flat table would let later rows spill past the header.
        We keep the cross-scenario table to **QC only** (the metric every scenario shares), and
        emit a per-scenario reg-target block underneath.
        """
        scenarios = inverse.get("scenarios", {}) if isinstance(inverse, dict) else {}
        if not scenarios:
            return
        lines: list[str] = [
            "# Inverse design — compact cross-scenario summary\n",
            "Auto-generated. The headline QC table aggregates across all scenarios; per-scenario "
            "reg-target tables follow. Full per-seed arrays in "
            "`inverse_design/<scenario>/<path>/result.json`.\n",
        ]

        # Cross-scenario QC table — the one metric every scenario shares.
        lines.append("## QC probability after decode\n")
        lines.append("| scenario | path | QC mean | QC std |")
        lines.append("|---|---|---:|---:|")
        for name, data in scenarios.items():
            paths_meta = data.get("paths", {})
            for path_name in INVERSE_PATHS:
                meta = paths_meta.get(path_name, {})
                qc_m = meta.get("qc_after_mean", float("nan"))
                qc_s = meta.get("qc_after_std", float("nan"))
                lines.append(f"| {name} | {path_name} | {qc_m:.3f} | {qc_s:.3f} |")
        lines.append("")

        # Per-scenario regression targets (columns match that scenario's reg_targets).
        for name, data in scenarios.items():
            reg_targets = data.get("reg_targets", {})
            paths_meta = data.get("paths", {})
            lines.append(f"## {name} — regression targets (after decode)\n")
            secondary = " · ".join(f"{_display(t)} {_arrow(v)} {v:+.1f}" for t, v in reg_targets.items())
            lines.append(f"Targets: {secondary}\n")
            header = ["path", *[_display(t) for t in reg_targets]]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for path_name in INVERSE_PATHS:
                meta = paths_meta.get(path_name, {})
                row = [path_name]
                for t in reg_targets:
                    row.append(f"{meta.get('reg_after_decode_mean', {}).get(t, float('nan')):+.2f}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        (self.inverse_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved inverse-design SUMMARY.md to {self.inverse_root / 'SUMMARY.md'}")

    def _write_analysis_md(self, records: list[dict[str, Any]], inverse: dict[str, Any]) -> None:
        """Long-form analysis (English, plan §0a). Reads as speaker-notes feedstock for SLIDE_PREP."""
        c = self._counts()
        intro = {r["new_task"]: r["metrics"][r["new_task"]]["primary"] for r in records}
        final = records[-1]["metrics"] if records else {}
        lines: list[str] = []
        lines.append("# Analysis — continual rehearsal + inverse design\n")
        lines.append(
            "Long-form narrative analysis of this run. The structured slide outline lives in\n"
            "[`SLIDE_PREP.md`](SLIDE_PREP.md); the compact cross-scenario table lives in\n"
            "[`inverse_design/SUMMARY.md`](inverse_design/SUMMARY.md). Numbers below are\n"
            "regenerable from the raw arrays under `training/` and `inverse_design/`.\n",
        )

        lines.append("## Run scale\n")
        lines.append(
            f"- **{c['n_tasks']} supervised tasks**: {c['n_reg']} regression · "
            f"{c['n_kr']} kernel regression · {c['n_clf']} classification, plus the always-on autoencoder.\n"
        )
        lines.append("- Datasets (tasks · unique compositions used):")
        for name, ntask, nkeys in self._dataset_summary():
            lines.append(f"  - {name}: {ntask} · {nkeys}")
        lines.append("")

        lines.append("## Continual learning — is there forgetting?\n")
        drops = []
        for task in self.config.task_sequence:
            i = intro.get(task)
            f_v = final.get(task, {}).get("primary")
            if i is not None and f_v is not None and np.isfinite(i) and np.isfinite(f_v):
                drops.append((task, i, f_v, f_v - i))
        early = drops[: max(1, len(drops) // 2)]
        mean_early_delta = float(np.mean([d for *_, d in early])) if early else float("nan")
        verdict = "stable (no clear forgetting)" if mean_early_delta > -0.05 else "some forgetting"
        lines.append(
            f"Mean (final − at-intro) primary metric over the *earlier-trained half* is "
            f"**{mean_early_delta:+.3f}** → **{verdict}**. The full per-step trajectory is in "
            "`training/forgetting_trajectory.png`; per-task raw `(composition, true, pred)` for "
            "every step is in `training/stepNN_<task>/<task>_pred.parquet` + `<task>_metrics.json` "
            "— rebuild any panel from those without retraining.\n"
        )
        lines.append("| task | at intro | final | Δ |")
        lines.append("|---|---:|---:|---:|")
        for task, i, f_v, d in drops:
            lines.append(f"| {_display(task)} | {i:+.3f} | {f_v:+.3f} | {d:+.3f} |")
        lines.append("")

        lines.append("## Final model — headline targets (inverse-design heads)\n")
        lines.append("| task | metric | value |")
        lines.append("|---|---|---:|")
        for task, m in self._final_target_metrics(records).items():
            spec = TASK_SPECS[task]
            metric_name = "accuracy" if spec["kind"] == "clf" else "R²"
            val = m.get("primary", float("nan"))
            lines.append(f"| {_display(task)} | {metric_name} | {val:+.3f} |")
        lines.append("")

        lines.append("## Inverse design — 3 scenarios × 4 paths\n")
        lines.append(
            "Each scenario shares the same 20 seeds (17 top-QC element-system-dedup + 3 explicit "
            "Au-Ga-Ln). Path semantics: **latent** uses `optimize_latent(ae_align_scale=0.5)` "
            "(PR #18 sweet spot); **composition_strict** locks the seed element support "
            "(`seed_blend=1.0`); **composition_alloy** is the paper-headline path "
            "(`seed_blend≈0.95`, 41-element ALLOY_PALETTE — allows discovery of QC-prone "
            "elements outside the seeds); **composition_random** ablates the seed entirely "
            "(`n_starts=N`) to surface the model's global QC attractor — useful to motivate the "
            "need for chemistry-constrained palettes when the global attractor falls on "
            "unsynthesisable elements.\n"
        )
        scenarios = inverse.get("scenarios", {}) if isinstance(inverse, dict) else {}
        for name, data in scenarios.items():
            reg_targets = data.get("reg_targets", {})
            paths_meta = data.get("paths", {})
            paths_details = data.get("paths_details", {})
            secondary = ", ".join(f"{_display(t)} {_arrow(v)} {v:+.1f}" for t, v in reg_targets.items())
            lines.append(f"### {name}\n")
            lines.append(f"- Secondary targets: {secondary}")
            lines.append(f"- Seed mean QC (before): **{data.get('qc_before_mean', float('nan')):.3f}**")
            lines.append("")
            header_cells = ["path", "QC after (mean ± std)"] + [_display(t) for t in reg_targets]
            lines.append("| " + " | ".join(header_cells) + " |")
            lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
            for path_name in INVERSE_PATHS:
                meta = paths_meta.get(path_name, {})
                qc_m = meta.get("qc_after_mean", float("nan"))
                qc_s = meta.get("qc_after_std", float("nan"))
                row_cells = [path_name, f"{qc_m:.3f} ± {qc_s:.3f}"]
                for t in reg_targets:
                    row_cells.append(f"{meta.get('reg_after_decode_mean', {}).get(t, float('nan')):+.2f}")
                lines.append("| " + " | ".join(row_cells) + " |")
            lines.append("")
            lines.append("One decoded example per path:")
            for path_name in INVERSE_PATHS:
                decoded = paths_details.get(path_name, {}).get("decoded_composition", [])
                if decoded:
                    lines.append(f"- **{path_name}**: `{decoded[0]}`")
            lines.append("")
            lines.append(
                f"Element-discovery heatmap: `inverse_design/{name}/element_frequency_heatmap.png`. "
                f"Side-by-side path comparison: `inverse_design/{name}/comparison.png`. "
                f"Per-path raw arrays: `inverse_design/{name}/<path>/result.json` (keys `optimized_weights` "
                "(B, 94), `optimized_descriptor` (B, x_dim), `qc_after_decode`, `reg_after_decode`).\n"
            )

        (self.output_dir / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved Markdown analysis to {self.output_dir / 'ANALYSIS.md'}")

    def _write_slide_prep_md(self, records: list[dict[str, Any]], inverse: dict[str, Any]) -> None:
        """9-slide structured handoff for the external slide author.

        Mirrors the polish level of the demo's ``inverse_design_run/SLIDE_PREP.md``:
        every section names a takeaway, slide content, speaker notes, and the canonical figure
        (with raw-data paths so the slide author can rebuild the figure if the auto-emitted one
        doesn't fit their layout). Numbers are computed from this run's data; interpretation
        threads are templated stubs the slide author fills in after sanity-checking against the
        plan §5 expected baselines (also reproduced inline).

        When ``sample_per_dataset`` is set (i.e. this is a smoke / partial run rather than the
        formal full run), a disclaimer is rendered at the top of the document; the numbers are
        still real but the magnitudes will not match the plan §5 expected baselines.
        """
        cfg = self.config
        counts = self._counts()
        intro = {r["new_task"]: r["metrics"][r["new_task"]]["primary"] for r in records}
        final = records[-1]["metrics"] if records else {}
        scenarios = inverse.get("scenarios", {}) if isinstance(inverse, dict) else {}
        seeds_meta = inverse.get("seeds", {}) if isinstance(inverse, dict) else {}
        strategy_seeds = list(seeds_meta.get("strategy_seeds", []))
        explicit_seeds = list(seeds_meta.get("explicit_seeds", []))
        all_seeds = strategy_seeds + explicit_seeds
        seed_pool: set[str] = set()
        for s in all_seeds:
            seed_pool |= self._element_system(s)

        is_smoke = cfg.sample_per_dataset is not None or cfg.max_epochs_per_step < 20
        run_date = _datetime.date.today().isoformat()

        def _discovered(
            path_data: dict[str, Any], threshold: float = 0.95, eps: float = 1e-3
        ) -> list[tuple[str, float]]:
            """Elements present in ≥ ``threshold`` fraction of a path's outputs but **0** in any seed."""
            w = np.asarray(path_data.get("optimized_weights", []), dtype=float)
            if w.size == 0:
                return []
            occ = (w > eps).mean(axis=0)
            out: list[tuple[str, float]] = []
            for i, frac in enumerate(occ):
                sym = DEFAULT_ELEMENTS[i]
                if frac >= threshold and sym not in seed_pool:
                    out.append((sym, float(frac)))
            out.sort(key=lambda kv: -kv[1])
            return out

        def _headline(task: str) -> str:
            spec = TASK_SPECS.get(task, {"kind": "reg"})
            metric_name = "accuracy" if spec["kind"] == "clf" else "R²"
            val = final.get(task, {}).get("primary", float("nan"))
            return f"`{task}` ({metric_name} = **{val:+.3f}**)"

        lines: list[str] = []
        # ── Header ────────────────────────────────────────────────────────────────────────
        lines.append("# Slide-prep document — handoff for the slide author (claude coworker)\n")
        lines.append(
            "> **What this is.** A structured outline a slide author can convert directly into deck\n"
            "> pages. Each section corresponds to one slide / slide group and lists: (a) the\n"
            "> takeaway, (b) what to put on the slide, (c) which file in this folder is the visual,\n"
            "> (d) speaker-note bullets. The slide author has **full creative freedom** for layout,\n"
            "> colours, and visual style — this document only specifies *what* to communicate, not\n"
            "> *how*.\n"
        )
        lines.append(f"**Folder this document lives in:** `{self.output_dir.name}/`")
        lines.append(f"**Run date:** {run_date}")
        lines.append(
            "**Data sources for every number cited:** "
            "`training/experiment_records.json` (per-task metrics across the "
            f"{counts['n_tasks']} training stages), "
            "`training/metrics_table.csv` (flat per-task at-intro / final), "
            "`training/stepNN_<task>/<task>_pred.parquet` (per-step raw test predictions for every "
            "active head), `inverse_design/inverse_design.json` (full nested inverse-design dump), "
            "and per-path `inverse_design/<scenario>/<path>/result.json` (raw per-seed arrays)."
        )
        lines.append(
            "**Companion docs:** [`README.md`](README.md) (folder index), [`ANALYSIS.md`](ANALYSIS.md) (long-form writeup), [`inverse_design/SUMMARY.md`](inverse_design/SUMMARY.md) (compact cross-scenario table).\n"
        )

        if is_smoke:
            lines.append(
                "> **⚠️ Run quality note — this is a SMOKE / partial run.**\n"
                f"> `sample_per_dataset = {cfg.sample_per_dataset}`, "
                f"`max_epochs_per_step = {cfg.max_epochs_per_step}` "
                "(formal full run uses `sample_per_dataset = null` and "
                "`max_epochs_per_step = 100` + EarlyStopping). The artifact tree is structurally\n"
                "> complete (every section below has real numbers from THIS run), but the\n"
                "> *magnitudes* will not match the formal full-run expected baselines documented in\n"
                "> [`docs/continual_rehearsal_full_PLAN.md`](../../docs/continual_rehearsal_full_PLAN.md) §5.\n"
                "> The expected-baseline tables below give the slide author the magnitudes to\n"
                "> sanity-check against before quoting numbers from this smoke run.\n"
            )

        lines.append("---\n")

        # ── Slide 1 — Experimental goal ───────────────────────────────────────────────────
        lines.append("## Slide 1 — Experimental goal: multi-property joint optimisation\n")
        lines.append(
            "**Takeaway.** Real materials development asks for *several properties at once* (is "
            "the material a quasi-crystal? does it have low formation energy? does it have high "
            "Tc / high κ_lat / high magnetic moment?). Single-property inverse-design tools don't "
            "help. We need a joint-optimisation framework around a model that learned all those "
            "properties together.\n"
        )
        lines.append("**Slide content.**")
        lines.append('- Opening line: *"The materials-design question is rarely about a single property."*')
        lines.append(
            "- 2–3 illustrative property combinations to ground the audience — pulled from this run's scenarios:"
        )
        for name, data in scenarios.items():
            reg_targets = data.get("reg_targets", {})
            arrowed = ", ".join(f"{_display(t)} {_arrow(v)}" for t, v in reg_targets.items())
            lines.append(f"  - **{name}** — QC ↑ + {arrowed}")
        lines.append('- A "wishlist → recipe" arrow showing the inverse direction: target properties → composition.\n')
        lines.append("**Speaker notes.**")
        lines.append(
            "- DFT / experiment loops are prohibitively expensive for joint searches over many target dimensions."
        )
        lines.append(
            "- A surrogate model that maps composition → multiple properties + supports gradient-based inverse design lets us search jointly.\n"
        )
        lines.append("**Visual asset.** Slide author draws; no pre-rendered figure.\n")
        lines.append("---\n")

        # ── Slide 2 — Model structure ─────────────────────────────────────────────────────
        lines.append("## Slide 2 — Model structure + inverse-design strategies\n")
        lines.append(
            "**Takeaway.** A shared-encoder foundation model with multiple task heads; **two** "
            "inverse-design paths (latent vs composition) operate on the **same trained model** "
            "so the comparison is a fair head-to-head test.\n"
        )
        lines.append("**Slide content.**")
        lines.append(
            "- Architecture diagram: "
            "`composition → KMD-1d descriptor x → encoder → latent h → tanh → {head_1, …, head_K}`."
        )
        lines.append(
            "- Highlight the always-on autoencoder head (decoder back to descriptor) — required by the latent path."
        )
        lines.append("- Two strategy boxes:")
        lines.append(
            "  - **Latent path** (`optimize_latent`): gradient-descend on `h`, decode with AE back to descriptor, "
            "evaluate heads. Failure mode without `ae_align_scale > 0`: AE round-trip drift drops QC."
        )
        lines.append(
            '  - **Composition path** (`optimize_composition`, "differentiable KMD"): gradient-descend directly '
            "on the 94-d element-weight simplex `w`, descriptor = `w · K`. No AE in the loop."
        )
        lines.append("- Two user knobs, both on `[0, 1]` (bigger = more of the named thing):")
        lines.append(
            "  - `ae_align_scale` — latent path; 0 = no AE-alignment penalty (failure-mode "
            "baseline), 1 = strongest alignment to AE fixed set. Compared at 0 / 0.25 / 1 in this run."
        )
        lines.append(
            "  - `diversity_scale` — composition path; 0 = peaky few-element recipes, 1 = "
            "multi-element recipes (default). Compared at 1.0 and 0.0 (low-diversity ablation) in this run."
        )
        lines.append(
            "- Optional composition add-ons: `allowed_elements` (whitelist palette), `seed_blend` (5 % uniform mix lets non-seed elements have reachable logits).\n"
        )
        lines.append("**Speaker notes.**")
        lines.append("- KMD-1d is differentiable in PR #17 → composition-space optimisation possible at all.")
        lines.append(
            '- Knob naming follows "bigger value = more of the named thing"; user doesn\'t need to read the docstring.'
        )
        lines.append(
            "- Same model handles both paths, so latent vs composition is a fair experiment, not an architecture comparison.\n"
        )
        lines.append("**Visual asset.** Slide author draws; no pre-rendered figure.\n")
        lines.append("---\n")

        # ── Slide 3 — Datasets + task types ──────────────────────────────────────────────
        lines.append("## Slide 3 — Datasets and task types\n")
        lines.append(
            f"**Takeaway.** The framework is trained on a heterogeneous task suite "
            f"({counts['n_tasks']} tasks across 4 data sources × 3 task types) joined by composition formula.\n"
        )
        lines.append("**Slide content (suggested 3-column layout).**\n")
        lines.append("| Task type | Count | Tasks |")
        lines.append("|---|---:|---|")
        for kind, label in (("reg", "Regression"), ("kr", "Kernel regression"), ("clf", "Classification")):
            tasks = [t for t in cfg.task_sequence if TASK_SPECS[t]["kind"] == kind]
            if tasks:
                lines.append(f"| **{label}** | {len(tasks)} | {', '.join(f'`{t}`' for t in tasks)} |")
        lines.append("")
        lines.append("Datasets supplying these tasks:\n")
        for name, ntask, nkeys in self._dataset_summary():
            lines.append(f"- **{name}** — {ntask} tasks · {nkeys} unique compositions used")
        lines.append("")
        lines.append("**Speaker notes.**")
        lines.append(
            "- Cross-source joining: every dataset has a `composition` column; the canonical formula is the join key."
        )
        lines.append(
            "- Kernel regression predicts an entire `(t, value)` series per composition — one head learns the shape vs `t` (DOS energy or temperature)."
        )
        lines.append(
            '- Classification uses inverse-frequency `class_weights` so the rare QC / AC classes stay alive against ~48k "others" rows in the qc dataset.\n'
        )
        lines.append(
            "**Visual asset.** Slide author renders the 3-column callout. Optional teaser: [`training/forgetting_trajectory.png`](training/forgetting_trajectory.png).\n"
        )
        lines.append(
            "**Raw-data pointer.** [`training/metrics_table.csv`](training/metrics_table.csv) is the flat task / type / dataset / at-intro / final / metric table.\n"
        )
        lines.append("---\n")

        # ── Slide 4 — Continual training ──────────────────────────────────────────────────
        lines.append("## Slide 4 — Continual training without catastrophic forgetting\n")
        lines.append(
            "**Takeaway.** Tasks are introduced one at a time across "
            f"**{counts['n_tasks']} stages**; tiered rehearsal (5 %/10 %) keeps the older heads "
            "alive. The forgetting trajectory shows every head holds its R² / accuracy as new "
            "tasks are added.\n"
        )
        lines.append(
            "**Primary figure:** [`training/forgetting_trajectory.png`](training/forgetting_trajectory.png) "
            "— per-step metric for every active head across all stages."
        )
        lines.append(
            "Annotate the fixed-tail tasks (the last 5 steps, "
            f"`{cfg.fixed_tail[0]} → {cfg.fixed_tail[1]} → {cfg.fixed_tail[2]} → {cfg.fixed_tail[3]} → {cfg.fixed_tail[4]}`) "
            "as the focus for the inverse-design section that follows.\n"
        )
        lines.append("**Final-step metrics for the inverse-design heads** (the heads inverse design actually uses):\n")
        lines.append("| Head | Type | Final-step metric |")
        lines.append("|---|---|---:|")
        for t in ["formation_energy", "magnetic_moment", "tc", "klat", "material_type"]:
            if t in final:
                spec = TASK_SPECS[t]
                metric_name = "accuracy" if spec["kind"] == "clf" else "R²"
                val = final.get(t, {}).get("primary", float("nan"))
                lines.append(f"| `{t}` | {KIND_LABEL[spec['kind']]} | **{val:+.3f}** ({metric_name}) |")
        lines.append("")
        lines.append("**Speaker notes.**")
        lines.append(
            f"- Rehearsal: `replay_ratio = {cfg.replay_ratio}` for ordinary old tasks, "
            f"`replay_ratio_high = {cfg.replay_ratio_high}` for the inverse-design tail (every step). "
            "No layer is frozen — encoder + every active head train jointly."
        )
        lines.append(
            "- Task ordering minimises rehearsal cost: 12 regression first (any order), then 7 "
            "kernel-regression tasks **ascending by row count** (cheapest first), then the 5 fixed-"
            "tail tasks — see plan §2 for the cost argument."
        )
        lines.append(
            "- **Per-step parquets + per-step checkpoints** are available under "
            "`training/stepNN_<task>/` so any per-task / per-step drill-down can be made later "
            "**without retraining**."
        )
        lines.append("- Raw data:")
        lines.append(
            "  - [`training/forgetting_trajectory.png`](training/forgetting_trajectory.png) — the headline curve."
        )
        lines.append(
            "  - `training/stepNN_<task>/<task>_pred.parquet` — `(composition, true, pred)` for every active head at every step."
        )
        lines.append("  - `training/stepNN_<task>/<task>_metrics.json` — per-task metric dict at that step.")
        lines.append(
            "  - `training/stepNN_<task>/checkpoint.pt` — model state at that step (payload `{model, task_sequence, step, new_task, active_tasks}`)."
        )
        lines.append(
            "  - [`training/experiment_records.json`](training/experiment_records.json) — every step × every active head, both at-intro and running metrics."
        )
        lines.append("  - [`training/metrics_table.csv`](training/metrics_table.csv) — flat aggregated table.\n")
        lines.append("---\n")

        # ── Slide 5 — Inverse design scenario setup ──────────────────────────────────────
        lines.append("## Slide 5 — Inverse design: scenario setup\n")
        lines.append(
            "**Takeaway.** Three scenarios share the same model, the same 20 seeds, and the "
            "same primary objective (**P(QC) ↑**). Secondary objectives differ — picking which "
            "scenario to feature in the talk is the slide author's narrative choice.\n"
        )
        lines.append("**Slide content.** A small table or three pill boxes:\n")
        lines.append("| Scenario | Primary | Secondary objectives |")
        lines.append("|---|---|---|")
        for name, data in scenarios.items():
            reg_targets = data.get("reg_targets", {})
            secondary = ", ".join(f"{_display(t)} {_arrow(v)} {v:+.1f}" for t, v in reg_targets.items())
            lines.append(f"| `{name}` | P(QC) ↑ (target 1.0) | {secondary} |")
        lines.append("")
        lines.append("**Methodology (constant across scenarios).**")
        lines.append("- 20 seeds shared across scenarios (slide 6 details the split).")
        lines.append(
            f"- Optimisation budget: **{cfg.inverse_steps} Adam steps**, **`lr = {cfg.inverse_lr}`**, "
            f"**`class_target_weight = {cfg.inverse_class_weight}`** (so QC dominates the loss)."
        )
        lines.append(
            "- All metrics evaluated **after** decoding the optimised descriptor back to a real composition (round-trip)."
        )
        lines.append("- 8 configurations per scenario (3 latent α + 5 composition) — see slide 6.\n")
        lines.append("**Speaker notes.**")
        lines.append(
            '- All three scenarios are first-class — the runner does not pick a "headline" scenario. Slide author chooses which to feature based on the talk\'s narrative.'
        )
        lines.append("- Plan §5 lists the rationale for each scenario.\n")
        lines.append('**Visual asset.** Slide author can draw a small "target dial" visual. No pre-rendered figure.\n')
        lines.append(
            "**Raw-data pointer.** [`inverse_design/seeds.json`](inverse_design/seeds.json) (seeds), `inverse_design/<scenario>/targets.json` (objective definitions per scenario).\n"
        )
        lines.append("---\n")

        # ── Slide 6 — Seeds + palette + config table ─────────────────────────────────────
        lines.append("## Slide 6 — Initial seeds, the element palette, and the 8 configurations\n")
        lines.append(
            f"**Takeaway.** Three ingredients shape the search: (a) **{len(all_seeds)} seeds** "
            "for the optimiser to start from, (b) the **41-element `ALLOY_PALETTE`** the "
            "constrained composition paths are allowed to use, (c) **8 configurations** isolating "
            "ae_align_scale / seed_blend / palette / diversity / random-init effects.\n"
        )
        lines.append("### Seeds\n")
        lines.append(
            f"**N = {len(all_seeds)}** = {len(strategy_seeds)} top-QC dedup + {len(explicit_seeds)} explicit-append. "
            "Element-system dedup keeps the best representative per element set so the seed list spans "
            "**different alloy families** rather than ratio variants of a few.\n"
        )
        lines.append(
            f"- **{len(strategy_seeds)} top-QC dedup seeds** (from the training-set material_type frame, picked by predicted QC probability):"
        )
        for s in strategy_seeds[:8]:
            lines.append(f"  - `{s}`")
        if len(strategy_seeds) > 8:
            lines.append(f"  - … ({len(strategy_seeds) - 8} more in `inverse_design/seeds.json`)")
        lines.append(
            f"- **{len(explicit_seeds)} explicit-append seeds** (forced regardless of QC score — known Au–Ga–RE i-QC formers):"
        )
        for s in explicit_seeds:
            lines.append(f"  - `{s}`")
        lines.append("")

        lines.append("### `ALLOY_PALETTE` (41 elements, slide author renders periodic-table highlight)\n")
        lines.append(
            "Range design: covers classic i-QC / d-QC formers + easy 4th/5th-period TMs + accessible lanthanides + Au (so Au–Ga–Ln seeds are reachable). Pm / Tc and Pu-class radioactives are excluded; Tm / Lu excluded as rare and expensive.\n"
        )
        lines.append("- **Light alkaline earth:** Mg, Ca")
        lines.append("- **Group 13:** B, Al, Ga, In, Tl")
        lines.append("- **Group 14:** Si, Ge")
        lines.append("- **4th-period TM (10):** Sc Ti V Cr Mn Fe Co Ni Cu Zn")
        lines.append("- **5th-period TM (9, Tc excluded as radioactive):** Y Zr Nb Mo Ru Rh Pd Ag Cd")
        lines.append("- **6th-period noble (needed for Au–Ga–RE seeds):** Au")
        lines.append("- **Accessible lanthanides (12, Pm/Tm/Lu excluded):** La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Yb\n")

        lines.append("### The 8 configurations — what each isolates\n")
        lines.append("3 latent points (along `ae_align_scale`) + 5 composition configs:\n")
        lines.append("| Config (x-axis label in `comparison.png`) | Knobs | What it tests |")
        lines.append("|---|---|---|")
        lines.append(
            "| `latent α=0` | `ae_align_scale = 0` | AE-alignment off → failure mode in PR #18's paper-baseline run (QC collapses). With `dos_density` in the training mix the latent geometry may be more robust — check this run's number. |"
        )
        lines.append("| `latent α=0.25` | `ae_align_scale = 0.25` | Low alignment — intermediate point. |")
        lines.append(
            "| `latent α=1` | `ae_align_scale = 1.0` | Max alignment — strongest cycle-consistency constraint. |"
        )
        lines.append(
            "| `comp (seed)` | `seed_blend = 1.0`, all elements allowed | Strict-seed baseline. Optimiser can only rebalance the seed's existing elements — no new element can enter the support set. |"
        )
        lines.append(
            "| `comp (seed, 5% all)` | `seed_blend = 0.95`, all allowed | Adds 5 % uniform mass over all 94 elements so non-seed elements have reachable logits. Optimiser *can* introduce new elements but otherwise unconstrained. |"
        )
        lines.append(
            "| `comp (seed, 5% all, element list)` | (above) + `allowed_elements = ALLOY_PALETTE` | Restricts the support set to the 41 feasible alloy elements. **Practical materials-design mode.** |"
        )
        lines.append(
            "| `comp (seed, 5% all, element list, low diversity)` | (above) + `diversity_scale = 0` | Adds max entropy penalty → forces peaky few-element recipes. Tests whether peaky recipes still satisfy the targets. |"
        )
        lines.append(
            '| `comp (random)` | `initial_weights = None`, all allowed | No seed, no palette. Pure "let the optimiser explore" — the no-bias control. |'
        )
        lines.append("")
        lines.append("**Speaker notes.**")
        lines.append("- Each row of `inverse_design/<scenario>/comparison.png` x-axis maps to one of these configs.")
        lines.append('- Labels read as "config A, then add knob B, then add knob C" — each comma = a knob change.')
        lines.append(
            '- "low diversity" = `diversity_scale = 0`, the most penalised end of the diversity knob → fewest elements per output.\n'
        )
        lines.append(
            "**Visual asset.** Slide author renders the periodic-table highlight from the 41-element list above. No pre-rendered palette figure.\n"
        )
        lines.append(
            "**Raw-data pointer.** [`inverse_design/seeds.json`](inverse_design/seeds.json) for the seed list; palette literal in [`samples/continual_rehearsal_full_config.toml`](../../samples/continual_rehearsal_full_config.toml).\n"
        )
        lines.append("---\n")

        # ── Slide 7 — Results & discussion (the central section) ─────────────────────────
        lines.append("## Slide 7 — Results & discussion\n")
        lines.append(
            "**Takeaway** (templated stub — fill in based on the per-scenario tables below + "
            "discovered-elements list). Typical claims the slide author chooses among:\n"
        )
        lines.append(
            "- **Headline claim.** `comp (seed, 5% all, element list)` is the practical winner on the scenario you pick to feature — tight, physically credible alloy recipes; element discovery (specific elements present in 100 % of outputs but 0 % of seeds)."
        )
        lines.append(
            "- **Constraints-matter claim.** `comp (random)` lands the optimiser on the model's unconstrained global QC attractor — often physically implausible elements; demonstrates that the palette + seed are doing real work, not just regularising."
        )
        lines.append(
            "- **Latent-knob claim.** The `ae_align_scale` sweep on `latent α=0 / 0.25 / 1` traces the AE-alignment effect on the three target axes."
        )
        lines.append("")
        lines.append(
            "Pick the claim(s) the actual numbers support; the per-scenario tables below carry every figure you need.\n"
        )

        lines.append("**Primary figures (per scenario).**")
        for name in scenarios:
            lines.append(
                f"- [`inverse_design/{name}/comparison.png`](inverse_design/{name}/comparison.png) — 8-config boxplot across P(QC) + each reg target."
            )
        lines.append("")
        lines.append("**Supporting figures (per scenario).**")
        for name in scenarios:
            lines.append(
                f'- [`inverse_design/{name}/element_frequency_heatmap.png`](inverse_design/{name}/element_frequency_heatmap.png) — path × top-25 elements; **bold orange** x-tick labels = elements NOT in any seed → "discovered".'
            )
        lines.append("")

        # Per-scenario per-config table + discovered elements + open questions
        for name, data in scenarios.items():
            reg_targets = data.get("reg_targets", {})
            paths_meta = data.get("paths", {})
            paths_details = data.get("paths_details", {})

            lines.append(f"### Scenario: `{name}`\n")
            secondary = ", ".join(f"{_display(t)} {_arrow(v)} {v:+.1f}" for t, v in reg_targets.items())
            lines.append(
                f"Targets: **P(QC) ↑ (target 1.0)**, {secondary}. "
                f"Seed mean QC (before): **{data.get('qc_before_mean', float('nan')):.3f}**.\n"
            )

            # Per-config table (one row per config, columns: QC mean ± std, each reg target mean)
            header = ["config", "QC after (mean ± std)"] + [REG_TASK_TITLES.get(t, t) for t in reg_targets]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
            for path_cfg in INVERSE_PATH_CONFIGS:
                key = path_cfg["key"]
                label = path_cfg["label"]
                meta = paths_meta.get(key, {})
                qc_m = meta.get("qc_after_mean", float("nan"))
                qc_s = meta.get("qc_after_std", float("nan"))
                row = [f"`{label}`", f"{qc_m:.3f} ± {qc_s:.3f}"]
                for t in reg_targets:
                    row.append(f"{meta.get('reg_after_decode_mean', {}).get(t, float('nan')):+.2f}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

            # Discovered elements per config (≥ 95 % occupancy, 0 in seeds)
            lines.append(
                "**Element discovery** (occurrence ≥ 95 % in this config's 20 outputs, **and** 0 occurrence in any seed):"
            )
            any_discovered = False
            for path_cfg in INVERSE_PATH_CONFIGS:
                key = path_cfg["key"]
                disc = _discovered(paths_details.get(key, {}))
                if disc:
                    any_discovered = True
                    payload = ", ".join(f"**{sym}** ({int(round(frac * 100))}%)" for sym, frac in disc)
                    lines.append(f"- `{path_cfg['label']}` → {payload}")
            if not any_discovered:
                lines.append(
                    "- *(none in this run — no element passes the ≥95 % occurrence + 0-in-seeds bar. "
                    "Either the optimiser is just rebalancing seed elements, or the run is too early "
                    "to surface discoveries. Smoke runs typically have none; the formal full run "
                    "is expected to surface discovered elements in `comp (seed, 5% all, element list)`.)*"
                )
            lines.append("")

            # Decoded example per config
            lines.append("**One decoded example per config** (highest-QC seed of that config):")
            for path_cfg in INVERSE_PATH_CONFIGS:
                key = path_cfg["key"]
                decoded = paths_details.get(key, {}).get("decoded_composition", [])
                if decoded:
                    lines.append(f"- `{path_cfg['label']}` → `{decoded[0]}`")
            lines.append("")

        # Three discussion-thread stubs (templated for the slide author)
        lines.append("### Discussion threads (templated stubs — verify against numbers above)\n")
        lines.append(
            "1. **Element discovery is the headline.** *Fill in:* in `comp (seed, 5% all, element list)`, "
            "which element(s) appear in ≥95 % of outputs and 0 % of seeds? (See the discovery list "
            'per scenario above.) If non-empty, this is the central claim — "the model found '
            "something we didn't tell it about\".\n"
        )
        lines.append(
            "2. **Constraints matter.** *Fill in:* `comp (random)` QC vs `comp (seed, 5% all, element list)` QC. "
            "If random-init lands far from the constrained QC, the seed + palette are doing real "
            "work (not regularising). If random-init still finds high QC but with implausible "
            "elements (Pu / F / Mn-rich), the *physicality* of the recipe is the constraint payoff, "
            "not raw QC.\n"
        )
        lines.append(
            "3. **Latent path α-knob role.** *Fill in:* compare `latent α=0` vs `latent α=1` QC + reg "
            "targets. In PR #18's pre-`dos_density` baseline α=0 was a catastrophe (QC ~ 0.39). "
            "With `dos_density` in this run's training mix, check whether α=0 is still a "
            "catastrophe (claim the failure-mode story), or whether the latent geometry is now "
            'robust to α=0 (claim the α-knob has shifted from "rescue QC" to "trade QC bias '
            'against secondary-target reach").\n'
        )

        lines.append("### Plan §5 expected baselines (for sanity-check; slide author must verify)\n")
        lines.append(
            "Plan §5 reports the following PR #18 + 41-elem-smoke baselines for a single "
            "scenario (QC↑ / FE↓ / klat↑, 16 seeds). The formal full run should land in similar "
            "magnitudes; smoke / partial runs will not.\n"
        )
        lines.append("| Config | QC after | FE after | klat after | pairwise L1 | mean #elems |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append("| latent α=0 (failure) | 0.386 ± 0.315 | +2.46 ± 0.59 | −0.44 ± 0.27 | 1.07 | 5.2 |")
        lines.append("| latent α=0.5 (sweet) | **0.960 ± 0.027** | +0.92 ± 1.16 | +1.07 ± 0.31 | 0.82 | 3.4 |")
        lines.append("| latent α=1.0 (max) | 0.951 ± 0.027 | +0.40 ± 1.04 | +1.20 ± 0.35 | 1.06 | 3.6 |")
        lines.append("| C-strict | 0.887 ± 0.053 | +1.27 ± 0.24 | +0.76 ± 0.67 | 1.42 | 2.6 |")
        lines.append("| **C-alloy (12 elem)** | 0.870 ± 0.012 | +0.84 ± 0.03 | **+1.81 ± 0.07** | 0.17 | 5.6 |")
        lines.append("| **C-alloy (41 elem)** | 0.842 ± 0.018 | +0.68 ± 0.07 | **+1.84 ± 0.06** | 1.02 | 6.0 |")
        lines.append("| C-rand | 0.793 ± 0.005 | −0.78 ± 0.03 | +1.77 ± 0.02 | 0.10 | 6.0 |")
        lines.append("")

        lines.append("### Open questions to flag\n")
        lines.append(
            "- **`comp (seed)` variance.** If `comp (seed)` σ is large (≥0.2 in PR #18 paper run), "
            "per-seed audit: which seeds fail? Drill down via `inverse_design/<scenario>/comp_seed/result.json` "
            "(`qc_after_decode` per seed; `seeds` list in same file)."
        )
        lines.append(
            "- **Au–Ga–Ln seeds.** The 3 explicit Au–Ga–Ln seeds are known QC candidates. Their "
            "*per-seed* QC in `comp (seed)` should be high — if not, that's itself a notable finding."
        )
        lines.append(
            "- **Scenario coverage.** This run has 3 scenarios; the deck may not need all three. "
            "Pick 1–2 the audience cares about and footnote the others.\n"
        )
        lines.append("---\n")

        # ── Slide 8 — Summary ────────────────────────────────────────────────────────────
        lines.append("## Slide 8 — Summary\n")
        lines.append("**Takeaway** (three bullets for the slide; numbers fill in from above).\n")
        lines.append(
            f"1. A shared-encoder foundation model trained continually across "
            f"**{counts['n_tasks']} heterogeneous tasks** with tiered rehearsal — no catastrophic "
            "forgetting on the inverse-design heads (slide 4 numbers)."
        )
        lines.append(
            "2. Two inverse-design paths on the same model, both exposed as user-friendly `[0, 1]` "
            "knobs (`ae_align_scale`, `diversity_scale`). Eight configurations per scenario "
            "isolate every effect (slide 6 table)."
        )
        lines.append(
            "3. On the scenario(s) you feature: the constrained composition path delivers "
            "physically credible recipes; element-discovery signal surfaces "
            "(see scenario-specific table in slide 7)."
        )
        lines.append("")
        lines.append("**Failure modes (also first-class — claim them honestly).**")
        lines.append("- AE-roundtrip drift without `ae_align_scale > 0` (latent path).")
        lines.append("- Seed-init support-set lock without `seed_blend < 1` (composition path with strict seed).")
        lines.append("- Non-physical attractors without `allowed_elements` (composition random init).\n")
        lines.append(
            "**Slide content.** Three takeaway bullets + a thumbnail of one of the "
            "`inverse_design/<scenario>/comparison.png` files (slide author picks).\n"
        )
        lines.append("---\n")

        # ── Slide 9 — Future work ────────────────────────────────────────────────────────
        lines.append("## Slide 9 — Future work\n")
        lines.append(
            "**Takeaway.** The current framework is the foundation; the next step is to wrap it "
            "in an agent system, then later wire into the broader AI4S agent ecosystem.\n"
        )
        lines.append("### Beat 6 — agent-based inverse-design workbench\n")
        lines.append('- Natural-language goals from the user ("I want a low-density QC formed from common metals").')
        lines.append(
            '- An AI agent decomposes the goal + applies domain knowledge ("QC + common metals → use `allowed_elements = ALLOY_PALETTE − lanthanides`").'
        )
        lines.append(
            "- Agent automatically sets optimiser knobs (`ae_align_scale`, `diversity_scale`, seed strategy, palette, target dict)."
        )
        lines.append("- Runs `optimize_*`, decodes outputs, generates a visualisation + PDF report.\n")
        lines.append("### Beat 7 — wider AI4S agent ecosystem\n")
        lines.append(
            "- Foundation model becomes the fast predictor + candidate generator in the centre of a larger stack."
        )
        lines.append(
            "- Other agents wrap DFT / MD simulators (slow but accurate validation), automated synthesis platforms (closed-loop experimental feedback)."
        )
        lines.append(
            "- Pipeline: user request → foundation-model candidates → DFT validation → robotic synthesis → results loop back to retrain the foundation model.\n"
        )
        lines.append(
            "**Slide content.** One bullet per beat, plus a concentric-circles sketch (foundation model at the centre, agent wrappers around it, the user / world outside).\n"
        )
        lines.append("---\n")

        # ── Quick reference ──────────────────────────────────────────────────────────────
        lines.append("## Quick reference — files in this run folder\n")
        lines.append("| File | Used by which slide |")
        lines.append("|---|---|")
        lines.append(
            "| [`training/forgetting_trajectory.png`](training/forgetting_trajectory.png) | Slide 4 (primary) |"
        )
        lines.append("| `training/stepNN_<task>/*.png` | Slide 4 appendix (drill-down per task) |")
        lines.append("| `training/stepNN_<task>/*_pred.parquet` | Replot any per-step figure without retraining |")
        lines.append("| `training/stepNN_<task>/*_metrics.json` | Per-task metric dict at that step |")
        lines.append("| `training/stepNN_<task>/checkpoint.pt` | Restore the model at any intermediate stage |")
        lines.append(
            "| [`training/experiment_records.json`](training/experiment_records.json) | Full records (step × head, at-intro + running) |"
        )
        lines.append(
            "| [`training/metrics_table.csv`](training/metrics_table.csv) | Flat task / type / dataset / at-intro / final table |"
        )
        lines.append(
            "| [`training/final_model.pt`](training/final_model.pt) | Final model state_dict + task_sequence |"
        )
        lines.append(
            "| `inverse_design/<scenario>/comparison.png` | Slide 7 (primary, per scenario), Slide 8 (thumbnail) |"
        )
        lines.append(
            "| `inverse_design/<scenario>/element_frequency_heatmap.png` | Slide 7 (supporting, per scenario) |"
        )
        lines.append(
            "| `inverse_design/<scenario>/<config>/result.json` | Per-config raw arrays — `optimized_weights` (20, 94), `optimized_descriptor` (20, x_dim), per-seed predictions |"
        )
        lines.append(
            "| `inverse_design/<scenario>/summary.json` | Per-scenario aggregated stats (per-config means + stds) |"
        )
        lines.append("| `inverse_design/<scenario>/targets.json` | Primary + secondary objective definitions |")
        lines.append(
            "| [`inverse_design/seeds.json`](inverse_design/seeds.json) | Slide 6 (seed names + strategy/explicit split) |"
        )
        lines.append(
            "| [`inverse_design/SUMMARY.md`](inverse_design/SUMMARY.md) | Cross-scenario compact summary table |"
        )
        lines.append(
            "| [`inverse_design/inverse_design.json`](inverse_design/inverse_design.json) | Full nested inverse-design dump (every scenario × every path) |"
        )
        lines.append("| [`ANALYSIS.md`](ANALYSIS.md) | Speaker-note source (long-form analysis) |")
        lines.append("| [`README.md`](README.md) | Run-folder reference / directory map |")
        lines.append("")

        # ── Slide-author freedom ──────────────────────────────────────────────────────────
        lines.append("## What the slide author has freedom over (and what they don't)\n")
        lines.append("**Free:**")
        lines.append("- Visual style (theme, colours, fonts, slide template).")
        lines.append("- Layout and slide breaks.")
        lines.append('- Diagrams (slides 1, 2, 3, 5, 6, 9 explicitly say "slide author draws this").')
        lines.append("- Order: this document is in narrative order, but the slide author may reshuffle.")
        lines.append("- Which scenario(s) to feature: the runner does not pick a headline scenario.")
        lines.append(
            "- Which discussion thread(s) in slide 7 to make the central claim — pick the one(s) the numbers actually support.\n"
        )
        lines.append("**Not free (these are the claims):**")
        lines.append(
            "- All numbers in the per-scenario tables of slide 7 — quoted from `inverse_design/<scenario>/<config>/result.json`."
        )
        lines.append(
            "- The element-discovery list — computed as occurrence ≥ 95 % in a config's outputs AND 0 in any seed (the bar must be cleared to claim discovery)."
        )
        lines.append("- The two-knob naming (`ae_align_scale`, `diversity_scale`) — these are the public API.")
        lines.append("- The 8 configuration names (x-axis labels of every `comparison.png`).")
        lines.append("- The 3 scenario names + target dicts (slide 5 table is canonical).\n")
        lines.append("---\n")

        # ── Raw-data cheat sheet ──────────────────────────────────────────────────────────
        lines.append("## Where the raw data lives — full cheat-sheet\n")
        lines.append(
            "Every figure above is fully reproducible from the raw arrays — **no need to "
            "retrain or rerun the optimisation** to change a plot's style / axis / colour scheme.\n"
        )
        lines.append(
            "- `training/stepNN_<task>/<task>_pred.parquet` — `(composition, true, pred)` (KR has `t` too). Plot any per-task parity / confusion / KR-sequence at any stage."
        )
        lines.append("- `training/stepNN_<task>/<task>_metrics.json` — per-task metric dict at that step.")
        lines.append(
            "- `training/stepNN_<task>/checkpoint.pt` — model state at that step (payload: `{model, task_sequence, step, new_task, active_tasks}`)."
        )
        lines.append(
            "- `training/experiment_records.json` — every step × every active head metric (at-intro and running)."
        )
        lines.append("- `training/metrics_table.csv` — flat task/type/dataset/at-intro/final/metric.")
        lines.append(
            "- `training/final_model.pt` — final model state_dict + task_sequence (consumed by `--inverse-only` / `paper_inverse_comparison.py` / `finetune_inverse_heads.py`)."
        )
        lines.append("- `training/forgetting_trajectory.png` — per-step × per-task primary-metric curves.")
        lines.append("- `inverse_design/seeds.json` — seeds in two segments (`strategy_seeds`, `explicit_seeds`).")
        lines.append("- `inverse_design/<scenario>/targets.json` — primary + secondary target definitions.")
        lines.append(
            "- `inverse_design/<scenario>/<config>/result.json` — per-config full record: `optimized_weights` `(B, 94)`, `optimized_descriptor` `(B, x_dim)`, `qc_after_decode`, `reg_before` / `reg_achieved_latent` / `reg_after_decode`, `decoded_composition`."
        )
        lines.append("- `inverse_design/<scenario>/summary.json` — per-scenario aggregated stats.")
        lines.append("- `inverse_design/<scenario>/comparison.png` — 8-config boxplot comparison.")
        lines.append(
            "- `inverse_design/<scenario>/element_frequency_heatmap.png` — config × element occurrence heatmap; discovered-element x-tick labels are bold + orange."
        )
        lines.append("- `inverse_design/SUMMARY.md` — compact cross-scenario table.\n")
        lines.append(
            "Element order in `optimized_weights`: "
            "`foundation_model.utils.kmd_plus.DEFAULT_ELEMENTS` (94 symbols). "
            "Composition-formula round-trip: `KMD.inverse(descriptor)` (or directly use `optimized_weights` which already lives on the simplex).\n"
        )

        (self.output_dir / "SLIDE_PREP.md").write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved SLIDE_PREP.md to {self.output_dir / 'SLIDE_PREP.md'}")

    def _write_readme(self, records: list[dict[str, Any]], inverse: dict[str, Any]) -> None:
        """Top-level run index — what's in this directory and where to start reading."""
        c = self._counts()
        scenarios = inverse.get("scenarios", {}) if isinstance(inverse, dict) else {}
        lines = [
            "# Continual rehearsal + inverse-design — run directory",
            "",
            f"{c['n_tasks']} supervised tasks ({c['n_reg']} reg · {c['n_kr']} kr · "
            f"{c['n_clf']} clf) + autoencoder · 3 inverse-design scenarios × 4 paths.",
            "",
            "## Start here",
            "- [`SLIDE_PREP.md`](SLIDE_PREP.md) — 9-section slide outline for the external slide author.",
            "- [`ANALYSIS.md`](ANALYSIS.md) — long-form narrative analysis (speaker-note material).",
            "- [`inverse_design/SUMMARY.md`](inverse_design/SUMMARY.md) — compact cross-scenario table.",
            "- `inverse_design/<scenario>/comparison.png` + `element_frequency_heatmap.png` — per-scenario figures (three scenarios, all first-class — no demo-style single-scenario headline).",
            "",
            "## Directory map",
            "```",
            "training/",
            "  stepNN_<task>/                       # one dir per training step",
            "    <task>_pred.parquet                # (composition, true, pred) for every active head",
            "    <task>_metrics.json                # per-task metric dict (R²/acc/MAE/…)",
            "    <task>_parity.png | _confusion.png | _sequences.png   # newest-head plot only",
            "    checkpoint.pt                      # model state at that step",
            "  forgetting_trajectory.png            # per-step × per-task primary metric",
            "  experiment_records.json              # full records (every step × every head)",
            "  metrics_table.csv                    # flat per-task at-intro / final table",
            "  final_model.pt                       # final model state_dict + task_sequence",
            "  final_model_taskconfigs.json         # task-config metadata for rebuilding the model",
            "inverse_design/",
            "  seeds.json                           # 20 seeds (17 top-QC dedup + 3 Au-Ga-Ln)",
            "  inverse_design.json                  # full nested result dump",
            "  SUMMARY.md                           # cross-scenario compact table",
            "  <scenario>/",
            "    targets.json                       # primary + secondary objectives",
            "    summary.json                       # per-path mean / std headline stats",
            "    comparison.png                     # 4-path boxplot (QC + each reg target)",
            "    element_frequency_heatmap.png      # path × top-25 elements (discovered = bold orange)",
            "    <path>/result.json                 # raw per-seed arrays, optimized_weights, …",
            "SLIDE_PREP.md                          # slide outline + raw-data pointers",
            "ANALYSIS.md                            # long-form analysis",
            "README.md                              # this file",
            "```",
            "",
            "## Scenarios",
        ]
        for name, data in scenarios.items():
            reg_targets = data.get("reg_targets", {})
            secondary = ", ".join(f"{_display(t)} {_arrow(v)} {v:+.1f}" for t, v in reg_targets.items())
            lines.append(f"- **{name}** — primary: QC ↑; secondary: {secondary}")
        lines.append("")
        (self.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved README.md to {self.output_dir / 'README.md'}")


# --- CLI ---------------------------------------------------------------------


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _parse_args(argv: list[str] | None = None) -> tuple[ContinualRehearsalFullConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Continual rehearsal + inverse-design — full run.")
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-per-dataset", type=int, default=None)
    parser.add_argument("--max-epochs-per-step", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument(
        "--inverse-only",
        type=Path,
        default=None,
        metavar="CKPT",
        help="Skip training; load a final_model.pt checkpoint and rerun only the inverse-design stage.",
    )
    args = parser.parse_args(argv)

    data = _load_toml(args.config_file) if args.config_file else {}
    for key in ("output_dir", "sample_per_dataset", "max_epochs_per_step", "accelerator"):
        val = getattr(args, key)
        if val is not None:
            data[key] = val

    field_names = set(ContinualRehearsalFullConfig.__dataclass_fields__)
    path_fields = {
        "qc_data_path",
        "qc_preprocessing_path",
        "superconductor_path",
        "magnetic_path",
        "phonix_path",
        "output_dir",
    }
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_names:
            logger.warning(f"Ignoring unknown config key '{key}'.")
            continue
        if key == "inverse_scenarios":
            kwargs[key] = [InverseScenario(**sc) if isinstance(sc, dict) else sc for sc in value]
        elif key in path_fields:
            # Empty string means "unset" (e.g. qc_preprocessing_path with no matching pkl).
            kwargs[key] = Path(value) if value not in (None, "") else None
        else:
            kwargs[key] = value
    return ContinualRehearsalFullConfig(**kwargs), args


def main(argv: list[str] | None = None) -> None:
    config, args = _parse_args(argv)
    runner = ContinualRehearsalFullRunner(config)
    if args.inverse_only is not None:
        runner.run_inverse_only(args.inverse_only)
    else:
        runner.run()


if __name__ == "__main__":
    main()
