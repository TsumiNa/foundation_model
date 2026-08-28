#!/usr/bin/env python3
"""Shared scoring primitives for the v2 campaign.

The conventions here are v1's, deliberately unchanged, because the final deliverable is a MERGED
v1+v2 report and a scale that shifted between the two halves would make the halves incomparable:

  * a run is read from the per-step JSONs, never from ``metrics_table.csv`` — a ``--resume``'d run
    writes a PARTIAL table (only the resuming process's in-memory records) while the per-step
    JSONs are always complete (PLAN §6.4);
  * a configuration is scored by the MEAN of its per-task RELATIVE deltas against the in-campaign
    untuned control, never against the historical single-task ceilings, which were measured on
    other hardware in a different container. Relative, because absolute deltas cannot be averaged
    across tasks whose metric lives on different scales — formation_energy's MAE is ~0.06 while a
    small task's is ~0.4, so an absolute mean is just the small task wearing a disguise;
  * a config has to win ACROSS the probe's tasks, so it cannot buy the mean by exploiting one;
  * every margin is quoted against the seed band, and this module computes the band rather than
    leaving it to each caller to invent one.

The ceilings are carried for headroom context only — they are never a target and never enter a
score.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

# Single-task ceilings from experiments/rikyu_replay_sweep/results/warm_restart.csv (H200,
# untuned architecture). Context for how much headroom a probe task has — never a target.
CEILING = {
    "formation_energy": 0.9950, "density": 0.9882, "material_type": 0.9840, "efermi": 0.9140,
    "dielectric_electronic": 0.8626, "tc": 0.7985, "curie": 0.7657, "magnetization": 0.7462,
    "total_magnetization": 0.7204, "klat": 0.6959, "final_energy": 0.6872, "kp": 0.6745,
    "neel": 0.6703, "dielectric_total": 0.6694, "thermal_conductivity": 0.6618, "zt": 0.6532,
    "magnetic_moment": 0.6408, "power_factor": 0.6336, "dielectric_ionic": 0.6078,
    "seebeck": 0.6026, "dos_density": 0.5999, "volume": 0.5685,
    "electrical_resistivity": 0.1622, "magnetic_susceptibility": 0.1238,
}  # fmt: skip

# Single-task ceilings measured IN THIS REGIME: 0.3.2 container, the adopted configuration, five
# seeds each, differing from a campaign run only in `pretrain.task_sequence`. These supersede
# CEILING above for anything that reports a deficit.
#
# The inherited H200 numbers were measured before PR #45, i.e. under the per-batch scheduler
# cadence that drove the LR to its floor inside the first epoch, so those runs were undertrained
# and the "ceilings" they produced sit BELOW what single-task training actually reaches.
#
# Across the 23 regression / kernel-regression tasks the old ceiling is too low in 17, by
# +0.0275 on average (median +0.0212, sd 0.0319, range -0.0168 to +0.1036). It is emphatically
# NOT a constant, so it cannot be repaired by subtracting an offset — seebeck is understated by
# 0.104 while dielectric_ionic is overstated by 0.017. It also grows as tasks get smaller
# (big +0.022, mid +0.027, small +0.040), which is the pattern an LR that never anneals would
# produce: the tasks with least data need the most optimisation to converge.
#
# material_type is EXCLUDED from that comparison and must stay excluded: the old entry is
# accuracy (0.984) and the same-regime run reports macro-F1 (0.571). Their difference is a
# metric mismatch, not an offset, and averaging it in produces a meaningless -0.41.
#
# Consequence for anything already published against the old ceilings: deficits computed there are
# systematically too small, and v1's negative mid/small deficits ("we passed the ceiling") are that
# artefact rather than a result.
CEILING_SAME_REGIME = {
    "curie": 0.8027,
    "density": 0.9898,
    "dielectric_electronic": 0.8574,
    "dielectric_ionic": 0.5910,
    "dielectric_total": 0.6587,
    "dos_density": 0.6250,
    "efermi": 0.9073,
    "electrical_resistivity": 0.1767,
    "final_energy": 0.7739,
    "formation_energy": 0.9947,
    "klat": 0.7220,
    "kp": 0.6957,
    "magnetic_moment": 0.6980,
    "magnetic_susceptibility": 0.1712,
    "magnetization": 0.7611,
    "material_type": 0.5710,
    "neel": 0.7176,
    "power_factor": 0.6936,
    "seebeck": 0.7062,
    "tc": 0.8153,
    "thermal_conductivity": 0.7194,
    "total_magnetization": 0.7183,
    "volume": 0.6191,
    "zt": 0.6600,
}  # fmt: skip

N_TRAIN = {
    "density": 23678, "efermi": 23668, "final_energy": 23678, "total_magnetization": 23678,
    "volume": 23678, "dielectric_total": 3124, "dielectric_ionic": 3124,
    "dielectric_electronic": 3124, "magnetization": 1160, "curie": 6272, "neel": 3466,
    "kp": 3875, "magnetic_susceptibility": 58, "zt": 3445, "power_factor": 3638,
    "thermal_conductivity": 4272, "electrical_resistivity": 5051, "dos_density": 7009,
    "seebeck": 8072, "formation_energy": 23180, "magnetic_moment": 851, "tc": 7207,
    "klat": 3863, "material_type": 33556,
}  # fmt: skip

PROBE6 = ["volume", "formation_energy", "seebeck", "zt", "magnetization", "magnetic_moment"]
KR_TASKS = {"seebeck", "zt", "power_factor", "thermal_conductivity", "electrical_resistivity",
            "dos_density", "magnetic_susceptibility"}  # fmt: skip

LOWER_IS_BETTER = {"mae"}

# A task whose R2 spread across the whole comparison set is below this has no resolution left on
# R2 and is scored on MAE instead (PLAN §6.3). formation_energy is the known case: its single-task
# ceiling is 0.995, so R2 there measures rounding, not learning.
R2_RESOLUTION_FLOOR = 0.005


def size_group(task: str) -> str:
    n = N_TRAIN.get(task, 0)
    return "big" if n >= 20000 else ("mid" if n >= 3000 else "small")


def final_metrics(run_dir: Path) -> tuple[dict[str, dict], int]:
    """Every task's metrics at the LAST completed step, from the authoritative step JSONs."""
    training = run_dir / "training"
    steps: dict[int, Path] = {}
    for d in training.glob("step*_*"):
        m = re.match(r"step(\d+)_", d.name)
        if m:
            steps[int(m.group(1))] = d
    if not steps:
        raise FileNotFoundError(f"{run_dir}: no step directories under {training}")
    last = max(steps)
    out = {}
    for jf in sorted(steps[last].glob("*_metrics.json")):
        out[jf.name[: -len("_metrics.json")]] = json.load(open(jf))
    return out, last


def fnum(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load_runs(outroot: Path, pattern: str, tasks: list[str]) -> dict[str, dict[str, dict]]:
    """{runid: {task: metrics}} for every DONE run matching ``pattern``.

    A run missing the marker, or missing a probe task, is dropped and reported by the caller —
    never silently averaged over, because a partial run reads as a slightly worse config rather
    than as a broken one.
    """
    out: dict[str, dict[str, dict]] = {}
    for run in sorted(outroot.glob(pattern)):
        if not run.is_dir() or not (run / "DONE").exists():
            continue
        try:
            metrics, _ = final_metrics(run)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if any(t not in metrics for t in tasks):
            continue
        out[run.name] = metrics
    return out


def split_seed(runid: str) -> tuple[str, int | None]:
    """('a1_L384_E0p003_M1e-06_P15', 2027) from 'a1_L384_E0p003_M1e-06_P15_s2027'."""
    m = re.match(r"^(.*)_s(\d+)$", runid)
    return (m.group(1), int(m.group(2))) if m else (runid, None)


def group_by_config(runs: dict[str, dict[str, dict]]) -> dict[str, dict[int, dict[str, dict]]]:
    """{config_label: {seed: {task: metrics}}}."""
    out: dict[str, dict[int, dict[str, dict]]] = {}
    for runid, metrics in runs.items():
        label, seed = split_seed(runid)
        out.setdefault(label, {})[seed if seed is not None else -1] = metrics
    return out


def pick_metric_per_task(
    configs: dict[str, dict[int, dict[str, dict]]], tasks: list[str]
) -> dict[str, str]:
    """R2 for each task unless R2 has run out of resolution there, in which case MAE.

    Decided across the WHOLE comparison set rather than per run, so every configuration is scored
    on the same axis — a per-run choice would silently compare different quantities.
    """
    chosen: dict[str, str] = {}
    for task in tasks:
        values = [
            fnum(seed_metrics[task].get("r2"))
            for per_seed in configs.values()
            for seed_metrics in per_seed.values()
            if task in seed_metrics
        ]
        values = [v for v in values if v is not None]
        spread = (max(values) - min(values)) if values else 0.0
        chosen[task] = "r2" if values and spread >= R2_RESOLUTION_FLOOR else "mae"
    return chosen


def task_values(per_seed: dict[int, dict[str, dict]], task: str, metric: str) -> list[float]:
    out = []
    for seed_metrics in per_seed.values():
        v = fnum(seed_metrics.get(task, {}).get(metric))
        if v is not None:
            out.append(v)
    return out


def relative_score(
    seed_metrics: dict[str, dict],
    reference: dict[str, float],
    metric_by_task: dict[str, str],
    tasks: list[str],
) -> float | None:
    """Mean relative improvement over the reference, across the probe's tasks.

    Sign-corrected so higher is always better, including on MAE.
    """
    deltas = []
    for task in tasks:
        metric = metric_by_task[task]
        value = fnum(seed_metrics.get(task, {}).get(metric))
        ref = reference.get(task)
        if value is None or ref is None or ref == 0:
            return None
        sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
        deltas.append(sign * (value - ref) / abs(ref))
    return statistics.fmean(deltas) if deltas else None


def band(values: list[float]) -> dict:
    """Seed-spread statistics, plus what those seeds can actually resolve.

    ``range`` is reported because v1 quoted its band that way (8.48% over three seeds) and the
    merged report has to line up with it. ``sigma`` is carried alongside because range depends on
    the seed COUNT — E[range] grows from 1.69σ at n=3 to 2.97σ at n=9 — so comparing a 3-seed
    range to a 9-seed range without that correction would read as a noise increase that is purely
    an artefact of counting more seeds.

    ``resolves`` is the honest headline: the smallest true difference two arms of this size can be
    told apart at, ~2 standard errors of the mean.
    """
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(values)
    sigma = statistics.stdev(values) if n > 1 else 0.0
    sem = sigma / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "sigma": sigma,
        "sem": sem,
        "resolves": 2 * sem,
        "values": sorted(values),
    }


def seeds_needed(sigma: float, target: float) -> int:
    """Seeds required for two arms to resolve a true difference of ``target`` (~2 SE each side).

    This is the arithmetic that decided v1's honest 'three-way tie': at σ≈5.0% three seeds resolve
    5.8%, and its top three were 1.5-1.8% apart, so the ranking it wanted was not purchasable at
    that seed count. It is recomputed here from v2's OWN measured σ rather than inherited, because
    probe6 exists precisely to change σ.
    """
    if target <= 0 or sigma <= 0:
        return 1
    return max(1, math.ceil((2 * sigma / target) ** 2))


def exceeds_band(delta: float, band_stats: dict) -> tuple[bool, float]:
    """(is the delta outside the noise band, how many band-widths it is).

    Deliberately returns the multiple as well: 'outside the band' is a threshold, and a threshold
    alone invites reporting 1.01x and 5x as the same finding.
    """
    width = band_stats.get("range") or 0.0
    if width <= 0:
        return (False, 0.0)
    return (abs(delta) > width, delta / width)
