"""Write units.txt: one run_experiment.py argument line per unit (plan 3.4 steps 4-6)."""

import random
from itertools import product
from pathlib import Path

SEEDS = range(5)
COMBOS = list(product(("AB", "ABX3"), ("L1", "L2"), ("P1", "P2")))
MAIN = {("AB", "L2", "P1"), ("ABX3", "L2", "P1")}

lines = []
for (cls, lab, part), prot, seed in product(COMBOS, "AB", SEEDS):
    base = f"--cls {cls} --label {lab} --part {part} --protocol {prot} --seed {seed}"
    lines.append(f"{base} --engine tabpfn --device cuda")
    if (cls, lab, part) in MAIN:
        lines.append(f"{base} --engine tabicl --device cuda")
    lines.append(f"{base} --engine catboost")
random.Random(0).shuffle(lines)  # balance heavy and light units across array tasks
Path(__file__).with_name("units.txt").write_text("\n".join(lines) + "\n")
print(len(lines), "units")
