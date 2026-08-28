#!/usr/bin/env python3
"""Write the B'-winning head parameters into the three stage-C' tuned configs.

The three `final_hybrid_c2top{1,2,3}.toml` files already carry the top-three A' points — encoder
and scheduler. This adds the head block, and adds the SAME head block to all three.

That is the whole design of the stage. c2_top1/2/3 exist to test whether the probe's ranking of A'
configurations survives at 24 tasks, so the three arms must differ in the A' axes and nothing else;
giving each its own head parameters would confound the ranking test with a second variable and
leave no way to attribute a reordering. B' tuned heads on the A'-ADOPTED base, so the winning head
block belongs to that base and is applied unchanged to all three.

    python scripts/apply_b_winner.py b_H256-128_HL0p001_X128-64_KL0p0001

The runid is parsed rather than passed as four flags because that string is the identifier the B'
summary ranks and the run directories carry; retyping its four values by hand is exactly the step
that silently applies the wrong winner.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CONFIGS = ["final_hybrid_c2top1.toml", "final_hybrid_c2top2.toml", "final_hybrid_c2top3.toml"]

# `tag()` in make_grids.py renders 1e-3 as "0p001" and [256,128] as "256-128"; this inverts it.
RUNID = re.compile(r"^b_H(?P<hidden>[\d-]+)_HL(?P<head_lr>[\dpe.+-]+)_X(?P<kr_x>[\d-]+)_KL(?P<kr_lr>[\dpe.+-]+)$")


def untag_num(text: str) -> float:
    return float(text.replace("p", "."))


def untag_dims(text: str) -> list[int]:
    return [int(v) for v in text.split("-")]


def parse_runid(runid: str) -> dict:
    runid = re.sub(r"_s\d+$", "", runid)  # tolerate a seeded run directory name
    m = RUNID.match(runid)
    if not m:
        raise SystemExit(f"not a stage-B runid: {runid!r}")
    return {
        "head_hidden_dims": untag_dims(m.group("hidden")),
        "head_lr": untag_num(m.group("head_lr")),
        "kr_x_hidden_dims": untag_dims(m.group("kr_x")),
        "kr_lr": untag_num(m.group("kr_lr")),
    }


def patch(path: Path, params: dict, runid: str) -> list[str]:
    """Replace the four head keys in place, and report every line actually changed.

    Replacement is line-anchored on the exact key rather than a blanket regex: `head_lr` is a
    substring of nothing here, but `kr_lr` would also match inside a hypothetical `kr_lr_min`, and
    a config edit that silently matches the wrong key produces a run that looks correct and is not.
    """
    lines = path.read_text().splitlines()
    changed, seen = [], set()
    for i, line in enumerate(lines):
        key = line.split("=")[0].strip()
        if key in params and key not in seen:
            value = params[key]
            rendered = str(value) if isinstance(value, list) else repr(value)
            lines[i] = f"{key} = {rendered}"
            changed.append(f"{key}: {line.strip()}  ->  {lines[i]}")
            seen.add(key)
    missing = set(params) - seen
    if missing:
        raise SystemExit(f"{path.name}: keys not found, refusing to write: {sorted(missing)}")
    header = f"# head block from the B' winner {runid} (applied identically to all three arms)"
    text = "\n".join(lines) + "\n"
    if header not in text:
        text = text.replace("[model]\n", f"[model]\n{header}\n", 1)
    path.write_text(text)
    return changed


def flatten(section: dict, prefix: str) -> dict[str, object]:
    """{'model.latent_dim': 384, ...} for the scalar/list leaves under one TOML table."""
    out: dict[str, object] = {}
    for key, value in section.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            out |= flatten(value, path)
        else:
            out[path] = value
    return out


def render(value) -> str:
    """A --set value the CLI parses back to what the TOML held."""
    if isinstance(value, list):
        return "[" + ",".join(render(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_consolidation_grid(configs: Path, runid: str) -> Path:
    """Emit grid_c2con.txt, deriving the tuned arm's overrides by DIFFING the two configs.

    `fm finetune` rebuilds every head before loading the checkpoint, so a consolidation run whose
    architecture does not match the checkpoint it warm-starts from is not a slightly different
    run — it is a wrong one. final_consolidate_v2.toml carries the UNTUNED [model]/[training]
    baseline (shared with the control arm), so the tuned line has to restate every value
    final_hybrid_c2top1.toml changed.

    Restating them by hand is the failure this function exists to prevent: miss one and the run
    still starts. So the override list is computed from the two files rather than typed, and any
    later edit to c2top1 propagates here automatically.
    """
    import tomllib

    tuned = tomllib.loads((configs / "final_hybrid_c2top1.toml").read_text())
    base = tomllib.loads((configs / "final_consolidate_v2.toml").read_text())
    diff = {}
    for table in ("model", "training"):
        a = flatten(tuned.get(table, {}), table)
        b = flatten(base.get(table, {}), table)
        for key, value in a.items():
            # Keys present only in the tuned file are also overrides — [training.scheduler] is
            # exactly that case, and dropping it would consolidate at the wrong annealing floor.
            if key not in b or b[key] != value:
                diff[key] = value
    # early_stopping / logging / max_epochs belong to the finetune régime, not the checkpoint's
    # architecture, and the consolidation config sets them deliberately.
    drop = {"training.max_epochs", "training.seed"}
    diff = {k: v for k, v in diff.items() if k not in drop and not k.startswith("training.early_stopping")
            and not k.startswith("training.logging")}

    sets = " ".join(f"--set {k}={render(v)}" for k, v in sorted(diff.items()))
    lines = [
        "c2base_consolidated\t--checkpoint /out/c2base/training/final_model.pt",
        f"c2top1_consolidated\t--checkpoint /out/c2top1/training/final_model.pt {sets}",
    ]
    path = configs / "grid_c2con.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"\ngrid_c2con.txt: {len(diff)} overrides derived by diffing c2top1 against the "
          f"consolidation baseline")
    for k, v in sorted(diff.items()):
        print(f"  --set {k}={render(v)}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runid", help="the winning stage-B runid, e.g. b_H256-128_HL0p001_X128-64_KL0p0001")
    ap.add_argument("--configs", type=Path, default=Path(__file__).resolve().parent.parent / "configs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    params = parse_runid(args.runid)
    print(f"winner {args.runid} decodes to:")
    for k, v in params.items():
        print(f"  {k} = {v}")
    if args.dry_run:
        return
    for name in CONFIGS:
        path = args.configs / name
        if not path.exists():
            sys.exit(f"missing {path}")
        print(f"\n{name}:")
        for line in patch(path, params, args.runid):
            print(f"  {line}")
    write_consolidation_grid(args.configs, args.runid)


if __name__ == "__main__":
    main()
