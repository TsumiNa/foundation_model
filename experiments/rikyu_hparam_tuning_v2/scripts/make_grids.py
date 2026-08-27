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


def parse_point(runid: str) -> dict:
    """Recover the settings a stage-A runid encodes (inverse of ``point_label``)."""
    point: dict = {}
    for bit in runid.split("_"):
        if bit.startswith("s") and bit[1:].isdigit():
            continue  # seed suffix
        key, _, raw = bit[0], None, bit[1:]
        val = raw.replace("p", ".")
        if key == "L":
            point["model__latent_dim"] = int(raw)
        elif key == "E":
            point["training__encoder_lr"] = float(val)
        elif key == "M":
            point["training__scheduler__min_lr"] = float(val)
        elif key == "P" and raw.isdigit():
            point["training__scheduler__patience"] = int(raw)
        elif key == "F":
            point["training__scheduler__factor"] = float(val)
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


def stage_finals(prefix: str, winners: list[str], n_seeds: int, include_anchors: bool):
    """The decisive step: promoted configurations re-measured at many seeds.

    v1's ranking failed here and said so honestly — its top three were 1.5-1.8% apart while three
    seeds could only resolve 5.8%, so it reported a three-way tie. Twenty-five seeds resolve ~2%.
    This is where surplus compute is worth the most (PLAN §7.1); the anchors ride along so every
    margin is quoted against the untuned point measured under identical conditions.
    """
    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    for w in winners:
        point = parse_point(w)
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
    ap.add_argument("stage", choices=["smoke", "s0", "a1", "a1r", "a1b", "a2", "a3", "a4", "b", "b3"])
    ap.add_argument("--winner", help="runid of the winning A' point (a2)")
    ap.add_argument("--winners", nargs="+", default=[], help="promoted runids (a1b / a3 / b3)")
    ap.add_argument("--seeds-n", type=int, default=None, help="seeds per point (stage default otherwise)")
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
        rows, skipped = stage_finals("b3", args.winners, n or 25, not args.no_anchors)

    write(args.stage, rows, skipped)


if __name__ == "__main__":
    main()
