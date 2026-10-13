#!/usr/bin/env python3
"""Fetch the MamunHighT2019 reactions from the Catalysis-Hub GraphQL API, page by page.

Source. Mamun, O., Winther, K. T., Boes, J. R. & Bligaard, T. High-throughput calculations of
catalytic properties of bimetallic alloy surfaces. Sci. Data 6, 76 (2019);
https://www.catalysis-hub.org/publications/MamunHighT2019 (pubId MamunHighT2019).
API: https://api-catalysis-hub.slac.stanford.edu/graphql, key in the X-API-Key header.

The key is read from the CATHUB_API_KEY environment variable and never written anywhere.
Limits: 10 requests / minute, accounts above 500 requests / day are suspended — so this script
sleeps between requests, stops at --max-requests, and saves every page to
data/CatalysisHub_MamunHighT2019/pages/ so a rerun resumes from the last saved cursor.

    CATHUB_API_KEY=... uv run python data/scripts/fetch_cathub_mamun2019.py --probe
    CATHUB_API_KEY=... uv run python data/scripts/fetch_cathub_mamun2019.py --page-size 500 --max-requests 150
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

ENDPOINT = "https://api-catalysis-hub.slac.stanford.edu/graphql"
OUT = Path(__file__).resolve().parent.parent / "CatalysisHub_MamunHighT2019" / "pages"
FIELDS = "id Equation chemicalComposition surfaceComposition facet sites coverages reactants products reactionEnergy activationEnergy dftCode dftFunctional"
SLEEP_S = 7.0  # ~8.5 requests / minute, under the 10 / minute limit


def post(query: str, key: str) -> dict:
    r = requests.post(ENDPOINT, json={"query": query}, headers={"X-API-Key": key}, timeout=300)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise SystemExit(f"API error: {data['errors']}")
    return data["data"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", help="one request: totalCount and the first 2 rows")
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--max-requests", type=int, default=100)
    a = ap.parse_args()
    key = os.environ.get("CATHUB_API_KEY")
    if not key:
        raise SystemExit("set CATHUB_API_KEY in the environment")
    if a.probe:
        d = post(
            '{reactions(pubId:"MamunHighT2019", first:2){totalCount pageInfo{hasNextPage endCursor} edges{node{%s}}}}'
            % FIELDS,
            key,
        )
        print(json.dumps(d, indent=1)[:4000])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    pages = sorted(OUT.glob("page_*.json"))
    cursor = json.loads(pages[-1].read_text())["pageInfo"]["endCursor"] if pages else None
    if pages and not json.loads(pages[-1].read_text())["pageInfo"]["hasNextPage"]:
        print("already complete:", len(pages), "pages")
        return
    for i in range(a.max_requests):
        after = f', after:"{cursor}"' if cursor else ""
        d = post(
            '{reactions(pubId:"MamunHighT2019", first:%d%s){totalCount pageInfo{hasNextPage endCursor} edges{node{%s}}}}'
            % (a.page_size, after, FIELDS),
            key,
        )["reactions"]
        n = len(pages) + i
        (OUT / f"page_{n:05d}.json").write_text(json.dumps(d))
        got = len(d["edges"])
        cursor = d["pageInfo"]["endCursor"]
        print(f"page {n}: {got} rows (total {d['totalCount']}), hasNext {d['pageInfo']['hasNextPage']}", flush=True)
        if not d["pageInfo"]["hasNextPage"]:
            print("complete")
            return
        time.sleep(SLEEP_S)
    print("stopped at --max-requests; rerun to continue")


if __name__ == "__main__":
    main()
