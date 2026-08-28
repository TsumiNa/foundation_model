#!/usr/bin/env python3
"""Emit the grid files consumed by ``scripts/fm_array.sbatch`` for the v2 campaign.

One line per array task: ``<runid>\\t<shell-quoted fm overrides>``. Every knob a grid point
changes is written explicitly, so a run's identity is fully recoverable from its runid and the
grid file is the campaign's registry of what was actually tried.

    python scripts/make_grids.py smoke
    python scripts/make_grids.py s0
    python scripts/make_grids.py a1
    python scripts/make_grids.py a3 --winners a1_L384_E0p003_M1e-06_P15 ... --seeds-n 25

WHAT IS DIFFERENT FROM v1
-------------------------
v1 tuned ``encoder_lr`` as a standalone knob. It could not have done otherwise: before PR #45 the
LR scheduler stepped once per BATCH, so the LR hit its ``min_lr`` floor inside the first epoch and
the whole run trained at the floor. ``min_lr`` was the de-facto training LR and ``patience`` /
``factor`` were inert.

With the cadence fixed, ``encoder_lr`` (start), ``min_lr`` (floor), ``factor`` (cut size) and
``patience`` (how long before cutting) jointly define ONE LR trajectory. Tuning any of them alone
reproduces v1's mistake in a new regime, so stage A' searches them together.

THE min_lr CONSTRAINT IS LOAD-BEARING
-------------------------------------
``OptimizerConfig`` rejects ``min_lr >= lr`` outright (model_config.py) — a floor at or above the
start silently disables annealing, so 0.3.2 refuses it rather than warning. One
``[training.scheduler]`` block serves ALL FOUR parameter groups (encoder / head / kr / ae), so the
binding constraint is the SMALLEST group LR, which is ``kr_lr = 5e-4`` — not the encoder LR the
grid is nominally about. A stage-B' grid that lowers ``kr_lr`` to 1e-4 under an adopted
``min_lr = 1e-4`` would therefore die on a ValueError at construction, as a whole column of runs.
``validate_point`` catches that here, at generation time, instead of at 3am in an array job.
"""

from __future__ import annotations

import argparse
import math
import random
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

# --- the untuned 0.3.2 baseline (= configs/probe6.toml, = stage-0's `base` anchor) ------------
BASE = {
    "model.latent_dim": 128,
    "model.encoder_hidden_dims": [256],
    "model.head_hidden_dims": [64],
    "model.kr_x_hidden_dims": [128, 64],
    "model.kr_t_hidden_dims": [16, 8],
    "model.n_kernel": 15,
    "training.encoder_lr": 5e-3,
    "training.head_lr": 5e-3,
    "training.kr_lr": 5e-4,
    "training.ae_lr": 5e-3,
    # 0.3.2 [training.scheduler] defaults — the axis that first became real in PR #45.
    "training.scheduler.factor": 0.5,
    "training.scheduler.patience": 5,
    "training.scheduler.min_lr": 1e-4,
}

# v1's ADOPTED encoder (stage A4-confirmed): the second stage-0 anchor. Its two per-task heads
# (density, dos_density) are not in the probe6 sequence, so only the encoder part is expressible
# here — which is exactly the part stage 0 is asking about.
V1_ADOPTED = {
    "model.latent_dim": 384,
    "training.encoder_lr": 1e-3,
}

PROBE6 = ["volume", "formation_energy", "seebeck", "zt", "magnetization", "magnetic_moment"]

# Seeds are drawn from one contiguous block so a larger seed count always CONTAINS the smaller
# one: the 5-seed grid runs are a prefix of the 25-seed finals, and re-running a promoted config
# at 25 seeds reuses the 5 already computed (DONE markers skip them).
SEED0 = 2025


def seeds(n: int) -> list[int]:
    return [SEED0 + i for i in range(n)]


def fmt(value) -> str:
    """TOML literal for a ``--set`` value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(fmt(v) for v in value) + "]"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, float):
        return repr(value)
    return str(value)


def tag(value) -> str:
    """Filesystem-safe token for a runid."""
    if isinstance(value, list):
        return "-".join(str(v) for v in value)
    return str(value).replace("+", "").replace(".", "p")


def overrides(**kw) -> str:
    """Shell-quoted ``--set`` string for the given dotted keys (``__`` means ``.``)."""
    parts: list[str] = []
    for key, value in kw.items():
        parts += ["--set", shlex.quote(f"{key.replace('__', '.')}={fmt(value)}")]
    return " ".join(parts)


# --- the constraint that kills whole columns of runs if it is not checked here ----------------

GROUP_LR_KEYS = {
    "encoder": "training__encoder_lr",
    "head": "training__head_lr",
    "kr": "training__kr_lr",
    "ae": "training__ae_lr",
}


def validate_point(point: dict) -> str | None:
    """Why 0.3.2 would refuse this point, or None if it is constructible.

    One scheduler serves four optimizer groups, so `min_lr` must sit below the SMALLEST of them.
    """
    min_lr = point.get("training__scheduler__min_lr", BASE["training.scheduler.min_lr"])
    for group, key in GROUP_LR_KEYS.items():
        lr = point.get(key)
        if lr is None:
            lr = BASE[key.replace("__", ".")]
        if min_lr >= lr:
            return f"min_lr={min_lr:g} >= {group}_lr={lr:g} (OptimizerConfig rejects this)"
    factor = point.get("training__scheduler__factor", BASE["training.scheduler.factor"])
    if not 0.0 < factor < 1.0:
        return f"factor={factor} outside (0, 1)"
    return None


def write(name: str, rows: list[tuple[str, str]], skipped: list[str] | None = None) -> Path:
    path = EXP / "configs" / f"grid_{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    lines = []
    for runid, ov in rows:
        if runid in seen:
            raise SystemExit(f"duplicate runid {runid!r} in grid {name}")
        seen.add(runid)
        lines.append(f"{runid}\t{ov}")
    path.write_text("\n".join(lines) + "\n")
    print(f"{path}  ({len(lines)} runs)")
    if skipped:
        # Never let a bounded grid look like a complete one (PLAN §7.6: no silent caps).
        print(f"  SKIPPED {len(skipped)} invalid point(s):")
        for s in skipped:
            print(f"    {s}")
    return path


def expand(label: str, point: dict, seed_list: list[int]) -> list[tuple[str, str]]:
    """One configuration at several seeds. Seed is an `fm` flag, not a `--set` key."""
    return [(f"{label}_s{s}", overrides(**point) + f" --seed {s}") for s in seed_list]


# --- smoke -----------------------------------------------------------------------------------


def smoke() -> list[tuple[str, str]]:
    """Full-chain check in the real execution environment, before anything long runs.

    Three points rather than one: the untuned baseline, a point that drives every scheduler knob
    the campaign will override (so the `--set training.scheduler.*` path is exercised, not just
    assumed), and the widest architecture the grid will ask for (so the biggest memory footprint
    is proven on a GB200 before 480 runs depend on it).
    """
    points = {
        "smoke_base": {},
        "smoke_sched": {
            "training__encoder_lr": 3e-2,
            "training__scheduler__min_lr": 1e-6,
            "training__scheduler__patience": 15,
            "training__scheduler__factor": 0.3,
        },
        "smoke_wide": {"model__latent_dim": 384, "training__encoder_lr": 1e-3},
    }
    rows = []
    for label, point in points.items():
        bad = validate_point(point)
        if bad:
            raise SystemExit(f"smoke point {label} is invalid: {bad}")
        rows.append((label, overrides(**point)))
    return rows


# --- stage 0: the anchor ----------------------------------------------------------------------


def stage0(n_seeds: int) -> list[tuple[str, str]]:
    """Untuned-on-0.3.2 and v1's-adopted-on-0.3.2, on probe6.

    This is the reference frame for everything else. Without the untuned point measured on the
    NEW image, v2's tuning gain cannot be separated from the gain PR #45 already delivered — and
    conflating the two is the specific mistake v1 made (PLAN §1).

    It does double duty: it also supplies the two numbers the rest of the campaign is sized from,
    the real per-run wall-clock of probe6 and its seed band. Both are currently ESTIMATES
    (0.8h/run, band inherited from a different probe), and both are load-bearing.
    """
    rows: list[tuple[str, str]] = []
    rows += expand("s0_base", {}, seeds(n_seeds))
    rows += expand(
        "s0_v1enc",
        {"model__latent_dim": 384, "training__encoder_lr": 1e-3},
        seeds(n_seeds),
    )
    return rows


# --- stage A'1: the joint encoder x LR x scheduler grid ---------------------------------------
#
# encoder_lr is densified relative to PLAN §2's four values because v1 measured it as the dominant
# knob (~2/3 of the total gain came from it alone) and because §7.5 budgets 96 points. The extra
# values are spent on THAT axis rather than on new axes: 5e-3 is included so the untuned default
# sits inside the grid as an interior point, and the range runs past PLAN's 3e-2 up to 5e-2
# because v1's "0.01 diverges" verdict was a symptom of having no working annealing, and where the
# ceiling actually sits is now an open question.

A1_ENCODER_LRS = [1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2]
A1_MIN_LRS = [1e-6, 1e-5, 1e-4]
A1_PATIENCE = [5, 15]
A1_LATENTS = [128, 384]


def point_label(prefix: str, p: dict) -> str:
    bits = [prefix]
    if "model__latent_dim" in p:
        bits.append(f"L{p['model__latent_dim']}")
    if "training__encoder_lr" in p:
        bits.append(f"E{tag(p['training__encoder_lr'])}")
    if "training__scheduler__min_lr" in p:
        bits.append(f"M{tag(p['training__scheduler__min_lr'])}")
    if "training__scheduler__patience" in p:
        bits.append(f"P{p['training__scheduler__patience']}")
    if "training__scheduler__factor" in p:
        bits.append(f"F{tag(p['training__scheduler__factor'])}")
    return "_".join(bits)


def stage_a1(n_seeds: int) -> tuple[list[tuple[str, str]], list[str]]:
    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    for latent in A1_LATENTS:
        for lr in A1_ENCODER_LRS:
            for min_lr in A1_MIN_LRS:
                for patience in A1_PATIENCE:
                    point = {
                        "model__latent_dim": latent,
                        "training__encoder_lr": lr,
                        "training__scheduler__min_lr": min_lr,
                        "training__scheduler__patience": patience,
                    }
                    bad = validate_point(point)
                    label = point_label("a1", point)
                    if bad:
                        skipped.append(f"{label}: {bad}")
                        continue
                    rows += expand(label, point, seeds(n_seeds))
    return rows, skipped


def stage_a1r(n_points: int, n_seeds: int, rng_seed: int) -> tuple[list[tuple[str, str]], list[str]]:
    """Random search over the same space plus `factor`, which the grid holds fixed.

    Four continuous dimensions is where a pure grid stops being efficient: it spends its points on
    axis-aligned combinations and leaves the interior unsampled. Random search covers the interior;
    the grid keeps the readable marginal-effect plots. Both are ranked together (PLAN §7.4).

    Sampling is seeded so the point set is reproducible from this file alone.
    """
    rng = random.Random(rng_seed)
    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    seen: set[str] = set()
    # The floor is capped by the SMALLEST group LR, and that is kr_lr = 5e-4 — not the encoder LR
    # this stage is nominally searching. Sampling against encoder_lr alone throws away precisely
    # the high-LR draws v2 exists to explore (measured: ~30% of points lost, all of them fast-LR).
    # So cap the draw at both bounds and keep every sample inside the constructible region.
    floor_ceiling = min(BASE[k] for k in ("training.kr_lr", "training.head_lr", "training.ae_lr"))
    for i in range(n_points):
        lr = 10 ** rng.uniform(math.log10(1e-3), math.log10(5e-2))
        # Sample the floor as a RATIO below the start, not on an absolute scale: what the
        # scheduler can do is set by how much annealing room it has, and an absolute floor sampled
        # independently would put most of its draws in the no-op region.
        hi = min(lr / 10.0, floor_ceiling * 0.9)
        min_lr = 10 ** rng.uniform(math.log10(1e-8), math.log10(hi))
        point = {
            "model__latent_dim": rng.choice(A1_LATENTS),
            "training__encoder_lr": float(f"{lr:.4g}"),
            "training__scheduler__min_lr": float(f"{min_lr:.4g}"),
            "training__scheduler__patience": rng.randint(3, 30),
            "training__scheduler__factor": round(rng.uniform(0.2, 0.8), 2),
        }
        bad = validate_point(point)
        label = f"a1r{i:03d}_" + point_label("", point).lstrip("_")
        if bad:
            skipped.append(f"{label}: {bad}")
            continue
        if label in seen:
            continue
        seen.add(label)
        rows += expand(label, point, seeds(n_seeds))
    return rows, skipped


# Longest prefix first: "HL0p001" must not be read as tag "H" with value "L0p001".
_TAGS: list[tuple[str, str, object]] = [
    ("HL", "training__head_lr", float),
    ("KL", "training__kr_lr", float),
    ("L", "model__latent_dim", int),
    ("E", "training__encoder_lr", float),
    ("M", "training__scheduler__min_lr", float),
    ("P", "training__scheduler__patience", int),
    ("F", "training__scheduler__factor", float),
    ("H", "model__head_hidden_dims", "dims"),
    ("X", "model__kr_x_hidden_dims", "dims"),
]


def parse_point(runid: str) -> dict:
    """Recover the settings a runid encodes (inverse of ``point_label`` and stage_b's label).

    Handles BOTH the stage-A tags (L/E/M/P/F) and the stage-B head tags (H/HL/X/KL).

    It used to handle only the first set, and it SKIPPED anything it did not recognise. That made
    ``parse_point("b_H64_HL0p005_X128-64_KL0p0005")`` return an empty dict — so a b3 finals built
    from stage-B winners would have expanded into a hundred runs of the untuned baseline, all
    identical, with no error anywhere. Unrecognised uppercase-led tokens now raise: a tag this
    function cannot read is a bug, not a value to drop.

    Lowercase tokens are stage prefixes and anchor names ("a1r129", "b", "base", "v1enc") and are
    still skipped, which is what lets a full runid be passed in.

    NOTE a stage-B label carries head settings ONLY — the encoder and scheduler it was measured on
    come from the A'-adopted base and must be merged by the caller.
    """
    point: dict = {}
    for bit in runid.split("_"):
        if not bit or bit[0].islower():
            continue  # stage prefix, anchor name
        if bit.startswith("s") and bit[1:].isdigit():
            continue  # seed suffix
        for tag, key, kind in _TAGS:
            if not bit.startswith(tag):
                continue
            raw = bit[len(tag):]
            if kind == "dims":
                point[key] = [int(v) for v in raw.split("-")]
            elif kind is int:
                if not raw.isdigit():
                    break  # e.g. a future "P" tag carrying a non-integer: fall through to raise
                point[key] = int(raw)
            else:
                point[key] = float(raw.replace("p", "."))
            break
        else:
            raise SystemExit(f"parse_point: unrecognised tag {bit!r} in runid {runid!r}")
        if key not in point and bit.startswith(tag):
            raise SystemExit(f"parse_point: bad value in token {bit!r} of runid {runid!r}")
    return point


def stage_a1b(winners: list[str], lrs: list[float], min_lrs: list[float], n_seeds: int):
    """Re-open a grid edge the optimum landed on.

    An optimum at the edge of the searched range is not an optimum, it is the range running out.
    v1 got this right once (its A1b extended encoder_lr downward and confirmed an interior point)
    and it is the one v1 procedure carried over unchanged.
    """
    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for w in winners:
        base = parse_point(w)
        # A None entry means "leave this axis at the winner's value" — an edge is reopened on one
        # axis at a time unless both are given.
        for lr in lrs or [None]:
            for min_lr in min_lrs or [None]:
                point = dict(base)
                if lr is not None:
                    point["training__encoder_lr"] = lr
                if min_lr is not None:
                    point["training__scheduler__min_lr"] = min_lr
                bad = validate_point(point)
                label = point_label("a1b", point)
                if bad:
                    skipped.append(f"{label}: {bad}")
                    continue
                if label in seen:
                    continue
                seen.add(label)
                rows += expand(label, point, seeds(n_seeds))
    return rows, skipped


def stage_a2(winner: str, n_seeds: int) -> tuple[list[tuple[str, str]], list[str]]:
    """`factor` and early-stopping patience on the A'1 champion.

    The new regime early-stops sooner than the old one did (v1 measured epoch 114 vs 136), so the
    150-epoch ceiling and patience 24 may no longer bind. This measures whether they do.
    """
    base = parse_point(winner)
    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    for factor in (0.3, 0.5, 0.7):
        for es_patience in (24, 40):
            point = dict(base, training__scheduler__factor=factor)
            bad = validate_point(point)
            label = point_label("a2", point) + f"_ES{es_patience}"
            if bad:
                skipped.append(f"{label}: {bad}")
                continue
            full = dict(point, training__early_stopping__patience=es_patience)
            rows += expand(label, full, seeds(n_seeds))
    return rows, skipped


def stage_a4(n_seeds: int) -> tuple[list[tuple[str, str]], list[str]]:
    """Does the LR schedule earn its place, and is the optimum below the searched floor?

    Two stage-A' findings point the same way. `encoder_lr` declines monotonically as it rises, and
    short scheduler `patience` — cut the LR early and often — is the only axis whose effect clears
    the seed band. Read together they suggest the model simply wants a low effective LR quickly,
    in which case the schedule may be an elaborate way of arriving somewhere a constant low LR
    reaches directly.

    Stage A' cannot answer that. Its `encoder_lr` floor was 1e-3, so BELOW that is unmeasured, and
    every point it ran had the scheduler enabled. The one hint it does carry argues weakly the
    other way — the lowest decile scored a little worse than the interior — but that decile spans
    only [1e-3, ~1.3e-3] and the gap is inside the noise.

    So: a full 2 x 6 factorial, schedule on/off crossed with an `encoder_lr` axis extended a decade
    lower, everything else held identical. Both questions get answered by the same runs, and
    "schedule vs none" is measured AT each LR rather than confounded with it.

    Two honest limits on what this can conclude:
      * `[training.scheduler]` governs all four parameter groups, so `enabled = false` also
        freezes the head / KR / AE learning rates. A loss for the flat arm therefore does not say
        WHICH group needed annealing. There is no per-group switch to decompose it with.
      * `min_lr = 1e-8` on the scheduled arm, far below the 1e-4 default, so the schedule has room
        to anneal across the whole extended LR range rather than hitting its floor immediately.
    """
    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    for enabled in (True, False):
        for lr in (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3):
            point = {
                "model__latent_dim": 384,
                "training__encoder_lr": lr,
                "training__scheduler__enabled": enabled,
                "training__scheduler__min_lr": 1e-8,
                "training__scheduler__patience": 5,
                "training__scheduler__factor": 0.5,
            }
            label = f"a4{'sched' if enabled else 'flat'}_E{tag(lr)}"
            # validate_point checks min_lr against every group LR. With the scheduler off the
            # engine skips that rejection entirely, so the check only has to hold for the
            # scheduled arm — which it does at min_lr 1e-8.
            bad = validate_point(point) if enabled else None
            if bad:
                skipped.append(f"{label}: {bad}")
                continue
            rows += expand(label, point, seeds(n_seeds))
    return rows, skipped


ALL24 = [
    "density", "efermi", "final_energy", "total_magnetization", "volume",
    "dielectric_total", "dielectric_ionic", "dielectric_electronic",
    "magnetization", "curie", "neel", "kp",
    "magnetic_susceptibility", "zt", "power_factor", "thermal_conductivity",
    "electrical_resistivity", "dos_density", "seebeck",
    "formation_energy", "magnetic_moment", "tc", "klat", "material_type",
]  # fmt: skip


def stage_xfer(base: str, n_orders: int, rng_seed: int, tasks: list[str]):
    """Transfer into each task from the other 23, with the task under test placed LAST.

    Measures what the probe cannot. A six-task probe says something about six tasks; the model
    that ships trains on 24, and the question that matters for a small task is whether arriving
    last — after an encoder has been shaped by 23 others — leaves it better off than training it
    alone. Comparing the final task's R2 against its same-regime single-task baseline answers
    exactly that, per task, at deployment scale.

    THREE ORDERS PER TASK, not three seeds of one order. The 23 preceding tasks are shuffled
    independently each time, so the spread across the three includes any effect of ordering as
    well as seed noise. If that spread matches the seed noise already measured, ordering does not
    matter — which is the claim being tested rather than assumed.

    Shuffling is seeded, so the 72 sequences are reproducible from this file alone.
    """
    rng = random.Random(rng_seed)
    point = parse_point(base)
    bad = validate_point(point)
    if bad:
        return [], [f"{base}: {bad}"]
    rows: list[tuple[str, str]] = []
    for task in tasks:
        others = [t for t in ALL24 if t != task]
        for k in range(n_orders):
            order = others[:]
            rng.shuffle(order)
            order.append(task)  # the task under test always arrives last
            rows.append((
                f"xf_{task}_o{k}",
                # --resume, unlike the probe stages. Those omit it because a probe run is short
                # enough that a walltime kill is cheaper to redo than the partial
                # metrics_table.csv it leaves behind. An xfer run is a full 24-task sequence —
                # stage-C length — and it writes a per-step checkpoint.pt exactly as stage C
                # does, so a kill without this flag throws away up to a day of work that was
                # already recoverable.
                overrides(pretrain__task_sequence=order, **point)
                + f" --seed {SEED0 + k} --resume",
            ))
    return rows, []


def stage_single(base: str, n_seeds: int, tasks: list[str], prefix: str = "st") -> tuple[list[tuple[str, str]], list[str]]:
    """Single-task ceilings measured IN THIS REGIME, one run per probe task.

    The recorded ceilings this campaign quotes come from the replay campaign's warm-restart
    control: different hardware (H200), different container, different code. Five of the six probe
    tasks sit ABOVE them under multi-task training, which reads as positive transfer — but a
    consistent offset is exactly what a change of measurement frame also produces, so that
    comparison cannot separate the two.

    This measures the same six tasks alone, on the same image, the same probe config and the same
    adopted hyper-parameters, differing only in `pretrain.task_sequence`. Multi-task minus this is
    transfer; multi-task minus the old ceilings is transfer plus an unknown offset.

    The question it settles is larger than it looks. If multi-task training does not beat training
    a small task by itself, there is nothing for a loss balancer or a gradient-surgery method to
    repair — the answer is to train it separately.
    """
    rows: list[tuple[str, str]] = []
    point = parse_point(base)
    bad = validate_point(point)
    if bad:
        return [], [f"{base}: {bad}"]
    for task in tasks:
        rows += expand(f"{prefix}_{task}", dict(point, pretrain__task_sequence=[task]), seeds(n_seeds))
    return rows, []


def stage_balx(bases: list[str], n_seeds: int) -> tuple[list[tuple[str, str]], list[str]]:
    """Balancer ON only, against a source tree that excludes the autoencoder head.

    The OFF arms are NOT regenerated. With the balancer disabled no sigmas are registered at all,
    so the AE-exclusion patch cannot touch that path — the existing `bal*off` runs are the valid
    control for these, and re-running them would burn compute to reproduce identical numbers.
    """
    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    for i, base in enumerate(bases):
        point = parse_point(base)
        bad = validate_point(point)
        if bad:
            skipped.append(f"{base}: {bad}")
            continue
        rows += expand(f"balx{i}on", dict(point, training__learnable_loss_balancer=True), seeds(n_seeds))
    return rows, skipped


def stage_bal(bases: list[str], n_seeds: int) -> tuple[list[tuple[str, str]], list[str]]:
    """Learnable loss balancer ON vs OFF, at several bases.

    Uncertainty weighting (Kendall/Gal/Cipolla, CVPR 2018) exists to stop multi-task training
    collapsing onto whichever tasks descend fastest — a live risk on a 24-task sequence spanning
    three orders of magnitude in label count. The model has implemented it all along; nothing ever
    routed a value to it, so it has never run once.

    SEVERAL BASES, not one. A single base answers "does it help HERE", which is not the question.
    The decision this feeds is whether the final tuning must carry the balancer as a dimension at
    all, and that only holds if the verdict survives a change of base. The bases are the top of
    stage A's ranking, which are statistically tied with one another, so they span the region any
    adopted configuration will come from.

    Run against a patched source tree through SRC_OVERRIDE: the flag is not in any container yet.
    Both arms share that tree, so the comparison does not depend on how it differs from the image.
    """
    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    for i, base in enumerate(bases):
        point = parse_point(base)
        bad = validate_point(point)
        if bad:
            skipped.append(f"{base}: {bad}")
            continue
        for enabled in (False, True):
            label = f"bal{i}{'on' if enabled else 'off'}"
            rows += expand(label, dict(point, training__learnable_loss_balancer=enabled), seeds(n_seeds))
    return rows, skipped


def stage_finals(prefix: str, winners: list[str], n_seeds: int, include_anchors: bool,
                 base_point: dict | None = None):
    """The decisive step: promoted configurations re-measured at many seeds.

    v1's ranking failed here and said so honestly — its top three were 1.5-1.8% apart while three
    seeds could only resolve 5.8%, so it reported a three-way tie. Twenty-five seeds resolve ~2%.
    This is where surplus compute is worth the most (PLAN §7.1); the anchors ride along so every
    margin is quoted against the untuned point measured under identical conditions.
    """
    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    for w in winners:
        # base_point carries the axes the label does not: a stage-B head label says nothing about
        # the encoder or scheduler it was measured on.
        point = dict(base_point or {}, **parse_point(w))
        bad = validate_point(point)
        label = f"{prefix}_" + w.split("_", 1)[1] if "_" in w else f"{prefix}_{w}"
        if bad:
            skipped.append(f"{label}: {bad}")
            continue
        rows += expand(label, point, seeds(n_seeds))
    if include_anchors:
        rows += expand(f"{prefix}_base", {}, seeds(n_seeds))
        rows += expand(
            f"{prefix}_v1enc",
            {"model__latent_dim": 384, "training__encoder_lr": 1e-3},
            seeds(n_seeds),
        )
    return rows, skipped


# --- stage B': multi-task JOINT head tuning ---------------------------------------------------
#
# Not per-task. v1 tuned all 24 heads individually and measured the result: only 2 of 24 gains
# survived seed repetition and 5 tasks got WORSE, while its joint-tuning control arm worked. The
# lesson v1 paid for is that tuning has to happen in the regime the model is deployed in, so v2
# tunes one shared head configuration on the multi-task probe and does not re-run the refuted
# design. material_type is left at defaults: v1 measured it as insensitive (+/-0.005).

B_HEAD_HIDDEN = [[64], [256, 128], [512, 256, 128]]
B_HEAD_LRS = [1e-3, 5e-3]
B_KR_X_HIDDEN = [[128, 64], [256, 128, 64]]
B_KR_LRS = [1e-4, 5e-4]


def stage_b(base_point: dict, n_seeds: int) -> tuple[list[tuple[str, str]], list[str]]:
    """Head capacity x head LR x KR-branch capacity x KR LR, on the adopted A' base.

    Every point inherits the A'-adopted encoder AND scheduler; a head tuned on a different base is
    a head tuned in a different regime, which is the failure this stage is designed around.
    """
    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    for hidden in B_HEAD_HIDDEN:
        for head_lr in B_HEAD_LRS:
            for kr_x in B_KR_X_HIDDEN:
                for kr_lr in B_KR_LRS:
                    point = dict(
                        base_point,
                        model__head_hidden_dims=hidden,
                        training__head_lr=head_lr,
                        model__kr_x_hidden_dims=kr_x,
                        training__kr_lr=kr_lr,
                    )
                    label = (
                        f"b_H{tag(hidden)}_HL{tag(head_lr)}_X{tag(kr_x)}_KL{tag(kr_lr)}"
                    )
                    bad = validate_point(point)
                    if bad:
                        skipped.append(f"{label}: {bad}")
                        continue
                    rows += expand(label, point, seeds(n_seeds))
    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["smoke", "s0", "a1", "a1r", "a1b", "a2", "a3", "a4", "bal", "balx", "single", "xfer", "b", "b3"])
    ap.add_argument("--winner", help="runid of the winning A' point (a2)")
    ap.add_argument("--winners", nargs="+", default=[], help="promoted runids (a1b / a3 / b3)")
    ap.add_argument("--seeds-n", type=int, default=None, help="seeds per point (stage default otherwise)")
    ap.add_argument("--tasks", nargs="+", default=PROBE6, help="single: which tasks to run alone")
    ap.add_argument("--orders", type=int, default=3, help="xfer: random orders per task")
    ap.add_argument("--prefix", default="st", help="single: runid prefix, so two bases do not collide")
    ap.add_argument("--points", type=int, default=200, help="a1r: number of random points")
    ap.add_argument("--rng-seed", type=int, default=20260827, help="a1r: sampling seed")
    ap.add_argument("--lrs", type=float, nargs="+", default=[], help="a1b: extra encoder LRs")
    ap.add_argument("--min-lrs", type=float, nargs="+", default=[], help="a1b: extra min_lrs")
    ap.add_argument("--base", help="b: adopted A' runid whose settings every head point inherits")
    ap.add_argument(
        "--no-anchors",
        action="store_true",
        help="finals: omit the untuned/v1 anchor arms (they are normally required)",
    )
    args = ap.parse_args()

    n = args.seeds_n
    skipped: list[str] = []

    if args.stage == "smoke":
        rows = smoke()
    elif args.stage == "s0":
        rows = stage0(n or 9)
    elif args.stage == "a1":
        rows, skipped = stage_a1(n or 5)
    elif args.stage == "a1r":
        rows, skipped = stage_a1r(args.points, n or 3, args.rng_seed)
    elif args.stage == "a1b":
        if not args.winners:
            raise SystemExit("a1b needs --winners (the points sitting on the edge)")
        if not (args.lrs or args.min_lrs):
            raise SystemExit("a1b needs --lrs and/or --min-lrs (which edge to reopen)")
        rows, skipped = stage_a1b(args.winners, args.lrs, args.min_lrs, n or 5)
    elif args.stage == "xfer":
        if not args.winners:
            raise SystemExit("xfer needs --winners (the adopted base)")
        rows, skipped = stage_xfer(args.winners[0], args.orders, args.rng_seed,
                                   args.tasks if args.tasks != PROBE6 else ALL24)
    elif args.stage == "single":
        if not args.winners:
            raise SystemExit("single needs --winners (one adopted base)")
        rows, skipped = stage_single(args.winners[0], n or 5, args.tasks, args.prefix)
    elif args.stage == "balx":
        if not args.winners:
            raise SystemExit("balx needs --winners")
        rows, skipped = stage_balx(args.winners, n or 5)
    elif args.stage == "bal":
        if not args.winners:
            raise SystemExit("bal needs --winners (the bases to test the balancer on)")
        rows, skipped = stage_bal(args.winners, n or 5)
    elif args.stage == "a4":
        rows, skipped = stage_a4(n or 5)
    elif args.stage == "a2":
        if not args.winner:
            raise SystemExit("a2 needs --winner")
        rows, skipped = stage_a2(args.winner, n or 5)
    elif args.stage == "a3":
        if not args.winners:
            raise SystemExit("a3 needs --winners (the promoted short list)")
        rows, skipped = stage_finals("a3", args.winners, n or 25, not args.no_anchors)
    elif args.stage == "b":
        if not args.base:
            raise SystemExit("b needs --base (the adopted A' runid)")
        rows, skipped = stage_b(parse_point(args.base), n or 5)
    elif args.stage == "b3":
        if not args.winners:
            raise SystemExit("b3 needs --winners")
        if not args.base:
            raise SystemExit(
                "b3 needs --base (the adopted A' runid): a stage-B label encodes head settings "
                "only, so without it every arm would collapse to the untuned baseline"
            )
        rows, skipped = stage_finals("b3", args.winners, n or 25, not args.no_anchors,
                                     base_point=parse_point(args.base))

    write(args.stage, rows, skipped)


if __name__ == "__main__":
    main()
