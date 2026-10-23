"""Write units_followup.txt: the three follow-up experiments (README, Follow-ups)."""

import random
from pathlib import Path

SEEDS = range(5)
lines = []
for seed in SEEDS:
    # 1. retrieval
    for cls in ("AB", "ABX3"):
        for eng in ("tabpfn", "tabicl"):
            lines.append(f"--cls {cls} --label L2 --part P1 --protocol R --seed {seed} --engine {eng} --device cuda")
    # 2. intermetallics split
    for lab in ("L1", "L2"):
        for prot in ("A", "B"):
            base = f"--cls AB --label {lab} --part P1x --protocol {prot} --seed {seed}"
            lines.append(f"{base} --engine tabpfn --device cuda")
            lines.append(f"{base} --engine catboost")
            if lab == "L2":
                lines.append(f"{base} --engine tabicl --device cuda")
    # 3. mixed context
    for cls, part, engines in (
        ("AB", "P1", ("tabpfn", "tabicl")),
        ("ABX3", "P1", ("tabpfn", "tabicl")),
        ("AB", "P1x", ("tabpfn",)),
    ):
        for eng in engines:
            lines.append(
                f"--cls {cls} --label L2 --part {part} --protocol M --seed {seed} --engine {eng} --device cuda"
            )
random.Random(0).shuffle(lines)
Path(__file__).with_name("units_followup.txt").write_text("\n".join(lines) + "\n")
print(len(lines), "units")
