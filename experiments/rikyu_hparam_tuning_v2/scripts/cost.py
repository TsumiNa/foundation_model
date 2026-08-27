#!/usr/bin/env python3
"""GPU-hour accounting for the campaign, and what packing actually saved.

Two numbers per stage, and the difference between them is the point:

* **run-hours** — the wall clock every run consumed, summed. This is what the campaign would have
  cost at one run per GPU, and it is what a naive read of ``_timing.tsv`` gives you.
* **card-hours** — what was actually billed. A packed task holds ONE GPU while N runs share it, so
  its N runs cost the card once, not N times. This is the number that matters, because the cluster
  bills GPUs and nothing else (``TRESBillingWeights = CPU=0.0, Mem=0.0, GRES/gpu=1.0``).

The worker records the pack size as a fifth column, so the split is read from the data rather than
inferred from which stages were launched with ``--pack``.

    python scripts/cost.py                                     # this campaign
    python scripts/cost.py --also /path/to/another/outbase     # alongside another

No throughput ratio is printed. ``card_s`` is DEFINED as ``run_s / pack_size``, so dividing one by
the other returns the pack size whatever the runs did — an accounting identity that looks like a
measurement. The real speed-up comes from a calibration that runs the same grid points both ways;
for this workload that is 7.1x at eight per card, not the 8.0x the identity would suggest.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

PACKED = re.compile(r"^packed(\d+)$")


@dataclass
class Usage:
    run_s: float = 0.0  # wall clock summed over runs
    card_s: float = 0.0  # GPU-seconds actually held
    n: int = 0
    packed_run_s: float = 0.0
    packed_card_s: float = 0.0
    n_packed: int = 0
    failed_s: float = 0.0
    n_failed: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self


def read_stage(tsv: Path) -> Usage:
    u = Usage()
    for line in tsv.read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        try:
            secs = float(parts[1])
        except ValueError:
            continue
        if parts[2] != "0":
            # Failed runs still held a card. Counted apart so they cannot quietly inflate the
            # per-run averages the campaign is sized from.
            u.failed_s += secs
            u.n_failed += 1
            continue
        u.n += 1
        u.run_s += secs
        m = PACKED.match(parts[4]) if len(parts) > 4 else None
        if m:
            share = secs / int(m.group(1))
            u.card_s += share
            u.packed_run_s += secs
            u.packed_card_s += share
            u.n_packed += 1
        else:
            u.card_s += secs
    return u


def report(root: Path) -> None:
    tsvs = sorted(root.glob("*/_timing.tsv"))
    if not tsvs:
        print(f"{root}: no timing logs")
        return
    print(f"\n=== {root.name} ===")
    print(f"{'stage':12s} {'runs':>6s} {'packed':>7s} {'run-h':>9s} {'card-h':>9s} {'saved':>9s}")
    total = Usage()
    for tsv in tsvs:
        u = read_stage(tsv)
        if not u.n and not u.n_failed:
            continue
        print(f"{tsv.parent.name:12s} {u.n:6d} {u.n_packed:7d} {u.run_s / 3600:9.1f} "
              f"{u.card_s / 3600:9.1f} {(u.run_s - u.card_s) / 3600:9.1f}")
        total += u
    print(f"{'TOTAL':12s} {total.n:6d} {total.n_packed:7d} {total.run_s / 3600:9.1f} "
          f"{total.card_s / 3600:9.1f} {(total.run_s - total.card_s) / 3600:9.1f}")

    if total.n_failed:
        print(f"  {total.failed_s / 3600:.1f} card-h burned by {total.n_failed} failed run(s)")
    if total.n_packed and total.packed_card_s:
        # No throughput ratio here on purpose. card_s is DEFINED as run_s / pack_size, so
        # run_s / card_s returns the pack size by construction and measures nothing — it would
        # print "8.0x" however the packed runs actually behaved. The real speed-up has to come
        # from a calibration that re-runs the same points both ways (7.1x at eight per card).
        print(f"  packed: {total.n_packed} of {total.n} runs, "
              f"{total.packed_run_s / 3600:.0f} run-h billed as {total.packed_card_s / 3600:.0f} "
              f"card-h (an accounting split, not a measured speed-up)")
    unpacked_n = total.n - total.n_packed
    if unpacked_n:
        unpacked_card = (total.card_s - total.packed_card_s) / 3600
        print(f"  unpacked: {unpacked_n} runs at {unpacked_card:.0f} card-h — "
              f"about {unpacked_card / 8:.0f} card-h had they been packed 8 to a GPU")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outbase", type=Path,
                    default=Path("/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning_v2"))
    ap.add_argument("--also", type=Path, action="append", default=[])
    args = ap.parse_args()
    for root in [args.outbase, *args.also]:
        report(root)


if __name__ == "__main__":
    main()
