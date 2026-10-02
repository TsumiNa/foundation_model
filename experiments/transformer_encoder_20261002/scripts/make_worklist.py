"""Generate disjoint encoder/seed lanes; the Slurm launcher packs adjacent lanes."""

import argparse
from pathlib import Path

from benchmark import load_protocol


def write_worklist(protocol: Path, output: Path, arms: list[str] | None, seeds: list[int] | None) -> int:
    config, _, registered = load_protocol(protocol)
    arms = list(registered) if arms is None else arms
    seeds = config.seeds if seeds is None else seeds
    if not arms or not seeds or len(set(arms)) != len(arms) or len(set(seeds)) != len(seeds):
        raise ValueError("Worklist selections must be nonempty and unique")
    if not set(arms) <= set(registered) or not set(seeds) <= set(config.seeds):
        raise ValueError("Every encoder and seed must be registered")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{arm}\t{seed}\n" for arm in arms for seed in seeds))
    return len(arms) * len(seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    print(write_worklist(args.protocol, args.output, args.arms, args.seeds))
