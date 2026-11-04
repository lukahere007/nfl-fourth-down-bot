"""Build the slim derived tables in data/derived/ from data/raw/.

Usage: python scripts/build_dataset.py [--start 2014 --end 2025]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fourth_down.data import build_tables, load_raw, shrink

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2014)
    p.add_argument("--end", type=int, default=2025)
    a = p.parse_args()
    raw = load_raw(ROOT / "data" / "raw", list(range(a.start, a.end + 1)))
    out = ROOT / "data" / "derived"
    out.mkdir(parents=True, exist_ok=True)
    for name, t in build_tables(raw).items():
        t = shrink(t)
        if name == "wp":  # nflverse reference columns are only needed for evaluation
            t = t.drop(columns=["play_id"])
        t.to_parquet(out / f"{name}.parquet", index=False, compression="zstd")
        print(f"{name:11s} {len(t):>8,d} rows  {(out / f'{name}.parquet').stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
