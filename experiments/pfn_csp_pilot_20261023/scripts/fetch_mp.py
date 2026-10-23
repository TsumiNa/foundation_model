"""Fetch the experimentally matched (theoretical=False) AB and ABX3 entries from Materials Project.

Plan section 2.1 step 1. Every entry whose anonymous formula is AB or ABC3 is downloaded with its
structure; per-composition selection and filtering happen in build_dataset.py.

The API key is read from the environment variable MP_API_KEY and never written anywhere.

    MP_API_KEY=... uv run python scripts/fetch_mp.py
"""

from __future__ import annotations

import gzip
import json
import os
from datetime import date
from pathlib import Path

from mp_api.client import MPRester

EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "artifacts" / "data"
FIELDS = [
    "material_id",
    "formula_pretty",
    "formula_anonymous",
    "nelements",
    "energy_above_hull",
    "theoretical",
    "deprecated",
    "database_IDs",
    "structure",
]
ANONYMOUS = {"AB": "AB", "ABX3": "ABC3"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    key = os.environ["MP_API_KEY"]
    with MPRester(key) as mpr:
        db_version = mpr.get_database_version()
        meta = {"db_version": db_version, "fetched": date.today().isoformat(), "counts": {}}
        for cls, anon in ANONYMOUS.items():
            docs = mpr.materials.summary.search(formula=anon, theoretical=False, deprecated=False, fields=FIELDS)
            rows = []
            for d in docs:
                rows.append(
                    {
                        "material_id": str(d.material_id),
                        "formula_pretty": d.formula_pretty,
                        "formula_anonymous": d.formula_anonymous,
                        "nelements": d.nelements,
                        "energy_above_hull": d.energy_above_hull,
                        "theoretical": d.theoretical,
                        "deprecated": d.deprecated,
                        "icsd_ids": list((d.database_IDs or {}).get("icsd", [])),
                        "structure": d.structure.as_dict(),
                    }
                )
            with gzip.open(OUT / f"mp_raw_{cls}.json.gz", "wt") as fh:
                json.dump(rows, fh)
            meta["counts"][cls] = len(rows)
            print(cls, len(rows))
    (OUT / "mp_fetch_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
