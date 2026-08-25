#!/usr/bin/env python3
"""Emit the grid files consumed by ``scripts/fm_array.sbatch``.

One line per array task: ``<runid>\t<shell-quoted fm overrides>``. Every knob a grid point
changes is written explicitly, so a run's identity is fully recoverable from its runid and the
grid file is the experiment's registry of what was actually tried.

Usage
-----
    python experiments/rikyu_hparam_tuning/scripts/make_grids.py a1
    python experiments/rikyu_hparam_tuning/scripts/make_grids.py a2 --winners a1_L128_H256_E5e-3 ...

Stage A2/A3/A4/A5 and all of stage B depend on the previous stage's winner, so they take the
winning setting on the command line rather than hard-coding a guess.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

# --- baseline (= configs/single_task.toml, = the untuned hybrid_full24 architecture) ---
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
    "training.kr_weight_decay": 5e-5,
    "data.batch_size": 256,
    "descriptor.n_grids": 8,
}

# Stage A probes the encoder with the 3-task sequence in configs/probe3.toml — one task per
# REPORT_20260809 size group (formation_energy 23180 / tc 7207 / magnetization 1160). The task
# sequence lives in that config, so stage-A grid points override architecture only.
ENC_TASKS = ["formation_energy", "tc", "magnetization"]
REG_TASKS = ["formation_energy", "volume", "final_energy"]
KR_TASKS = ["seebeck", "dos_density"]
CLF_TASK = "material_type"


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
    """Shell-quoted ``--set`` string for the given dotted keys."""
    parts: list[str] = []
    for key, value in kw.items():
        parts += ["--set", shlex.quote(f"{key.replace('__', '.')}={fmt(value)}")]
    return " ".join(parts)


def task_override(task: str) -> str:
    return overrides(**{"pretrain__task_sequence": [task]})


def write(name: str, rows: list[tuple[str, str]]) -> Path:
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
    return path


# --- stage A: encoder / shared trunk --------------------------------------------------------

# Measured on RIKYU: one full-data single-task probe is ~5 min on a GB200, so the grid is
# sized for coverage rather than for cost. 4 x 5 x 4 = 80 runs ~ 8 GPU-h.
LATENTS = [64, 128, 256, 384]
ENC_HIDDEN = [[256], [512], [512, 256], [1024, 512], [1024, 512, 256]]
ENC_LRS = [1e-3, 2e-3, 5e-3, 1e-2]


def stage_a1() -> list[tuple[str, str]]:
    rows = []
    for latent in LATENTS:
        for hidden in ENC_HIDDEN:
            for lr in ENC_LRS:
                runid = f"a1_L{latent}_H{tag(hidden)}_E{tag(lr)}"
                ov = overrides(
                    model__latent_dim=latent,
                    model__encoder_hidden_dims=hidden,
                    training__encoder_lr=lr,
                )
                rows.append((runid, ov))
    return rows


def parse_a1(runid: str) -> dict:
    """Recover the encoder settings a stage-A1 runid encodes."""
    _, latent, hidden, lr = runid.split("_", 3)
    return {
        "model__latent_dim": int(latent[1:]),
        "model__encoder_hidden_dims": [int(v) for v in hidden[1:].split("-")],
        "training__encoder_lr": float(lr[1:].replace("p", ".")),
    }


def stage_a2(winners: list[str]) -> list[tuple[str, str]]:
    """Batch-size scan on the stage-A1 short list (256 already measured in A1)."""
    rows = []
    for w in winners:
        enc = parse_a1(w)
        for bs in (512, 1024):
            runid = f"a2_{w[3:]}_B{bs}"
            ov = overrides(**enc, data__batch_size=bs)
            rows.append((runid, ov))
    return rows


def stage_a3(winner: str, batch_size: int) -> list[tuple[str, str]]:
    """Descriptor-resolution scan on the stage-A winner (n_grids 8 already measured)."""
    rows = []
    enc = parse_a1(winner)
    for n_grids in (4, 16):
        runid = f"a3_{winner[3:]}_B{batch_size}_G{n_grids}"
        ov = overrides(**enc, data__batch_size=batch_size, descriptor__n_grids=n_grids)
        rows.append((runid, ov))
    return rows


def stage_a4(short_list: list[str], batch_size: int) -> list[tuple[str, str]]:
    """Seed repeats for the whole A1 short list plus the untuned baseline.

    Ranking 80 single-seed points invites reading noise, so every candidate that could be adopted
    is re-measured at two further seeds. A config is only preferred over another when its margin
    survives this spread.
    """
    rows = []
    arms = [(w[3:], parse_a1(w), batch_size) for w in short_list]
    arms.append(("base", {}, BASE["data.batch_size"]))
    for label, enc, bs in arms:
        for seed in (2026, 2027):
            runid = f"a4_{label}_s{seed}"
            ov = overrides(**enc, data__batch_size=bs)
            rows.append((runid, ov + f" --seed {seed}"))
    return rows


def stage_a6(winner: str, batch_size: int) -> list[tuple[str, str]]:
    """The autoencoder head always trains and its gradients reach the shared trunk, so its LR is
    an encoder-side knob. 1-D scan on the winner (5e-3 already measured)."""
    rows = []
    enc = parse_a1(winner)
    for ae_lr in (1e-3, 1e-2):
        runid = f"a6_{winner[3:]}_B{batch_size}_A{tag(ae_lr)}"
        ov = overrides(**enc, data__batch_size=batch_size, training__ae_lr=ae_lr)
        rows.append((runid, ov))
    return rows


# --- stage B: task heads (encoder frozen at the stage-A winner) ------------------------------

HEAD_HIDDEN = [[64], [128, 64], [256, 128], [256, 128, 64], [512, 256, 128]]
HEAD_LRS = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
KR_N_KERNEL = [15, 32, 64, 128]
KR_X_HIDDEN = [[128, 64], [256, 128, 64]]
KR_LRS = [2e-4, 5e-4, 1e-3, 2e-3]
KR_T_HIDDEN = [[16, 8], [32, 16], [64, 32, 16]]
KR_WD = [5e-5, 5e-4]
CLF_HIDDEN = [[64], [128, 64], [256, 128], [256, 128, 64]]
CLF_LRS = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


def stage_b_reg(enc: dict) -> list[tuple[str, str]]:
    rows = []
    for hidden in HEAD_HIDDEN:
        for lr in HEAD_LRS:
            for task in REG_TASKS:
                runid = f"breg_H{tag(hidden)}_L{tag(lr)}_{task}"
                ov = task_override(task) + " " + overrides(
                    **enc, model__head_hidden_dims=hidden, training__head_lr=lr
                )
                rows.append((runid, ov))
    return rows


def stage_b_kr(enc: dict) -> list[tuple[str, str]]:
    rows = []
    for n_kernel in KR_N_KERNEL:
        for x_hidden in KR_X_HIDDEN:
            for lr in KR_LRS:
                for task in KR_TASKS:
                    runid = f"bkr_K{n_kernel}_X{tag(x_hidden)}_L{tag(lr)}_{task}"
                    ov = task_override(task) + " " + overrides(
                        **enc,
                        model__n_kernel=n_kernel,
                        model__kr_x_hidden_dims=x_hidden,
                        training__kr_lr=lr,
                    )
                    rows.append((runid, ov))
    return rows


def stage_b_kr2(enc: dict, n_kernel: int, x_hidden: list[int], kr_lr: float) -> list[tuple[str, str]]:
    """Coordinate-branch width and weight decay, conditioned on the stage-B-KR winner."""
    rows = []
    for t_hidden in KR_T_HIDDEN:
        for wd in KR_WD:
            if t_hidden == BASE["model.kr_t_hidden_dims"] and wd == BASE["training.kr_weight_decay"]:
                continue  # identical to the stage-B-KR winner, already measured
            for task in KR_TASKS:
                runid = f"bkr2_T{tag(t_hidden)}_W{tag(wd)}_{task}"
                ov = task_override(task) + " " + overrides(
                    **enc,
                    model__n_kernel=n_kernel,
                    model__kr_x_hidden_dims=x_hidden,
                    model__kr_t_hidden_dims=t_hidden,
                    training__kr_lr=kr_lr,
                    training__kr_weight_decay=wd,
                )
                rows.append((runid, ov))
    return rows


def stage_b_clf(enc: dict) -> list[tuple[str, str]]:
    """material_type. Its head LR is set per task (``[[tasks]].lr``) because [training].head_lr
    is shared with every regression head and stage B tunes them independently."""
    rows = []
    for hidden in CLF_HIDDEN:
        for lr in CLF_LRS:
            runid = f"bclf_H{tag(hidden)}_L{tag(lr)}"
            ov = task_override(CLF_TASK) + " " + overrides(
                **enc,
                model__head_hidden_dims=hidden,
                training__head_lr=lr,
            )
            rows.append((runid, ov))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["a1", "a2", "a3", "a4", "a6", "breg", "bkr", "bkr2", "bclf"])
    ap.add_argument("--winner", help="stage-A1 runid of the winning encoder (a1_L..._H..._E...)")
    ap.add_argument("--winners", nargs="+", help="stage-A1 short list (a2 / a5)")
    ap.add_argument("--batch-size", type=int, default=BASE["data.batch_size"])
    ap.add_argument("--n-grids", type=int, default=BASE["descriptor.n_grids"])
    ap.add_argument("--n-kernel", type=int, default=BASE["model.n_kernel"])
    ap.add_argument("--kr-x-hidden", type=int, nargs="+", default=BASE["model.kr_x_hidden_dims"])
    ap.add_argument("--kr-lr", type=float, default=BASE["training.kr_lr"])
    args = ap.parse_args()

    def enc_settings() -> dict:
        if not args.winner:
            raise SystemExit(f"stage {args.stage} needs --winner")
        settings = parse_a1(args.winner)
        settings["data__batch_size"] = args.batch_size
        settings["descriptor__n_grids"] = args.n_grids
        return settings

    if args.stage == "a1":
        write("a1", stage_a1())
    elif args.stage == "a2":
        write("a2", stage_a2(args.winners or []))
    elif args.stage == "a3":
        write("a3", stage_a3(args.winner, args.batch_size))
    elif args.stage == "a4":
        write("a4", stage_a4(args.winners or [], args.batch_size))
    elif args.stage == "a6":
        write("a6", stage_a6(args.winner, args.batch_size))
    elif args.stage == "breg":
        write("breg", stage_b_reg(enc_settings()))
    elif args.stage == "bkr":
        write("bkr", stage_b_kr(enc_settings()))
    elif args.stage == "bkr2":
        write("bkr2", stage_b_kr2(enc_settings(), args.n_kernel, args.kr_x_hidden, args.kr_lr))
    elif args.stage == "bclf":
        write("bclf", stage_b_clf(enc_settings()))


if __name__ == "__main__":
    main()
