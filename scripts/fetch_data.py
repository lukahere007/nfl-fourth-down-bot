"""Download nflverse play-by-play parquet files into data/raw/.

Usage:
    python scripts/fetch_data.py                 # 2014-2025
    python scripts/fetch_data.py --start 2018 --end 2025

Source: https://github.com/nflverse/nflverse-data/releases (tag ``pbp``),
licensed CC-BY 4.0 by the nflverse project. Files are only downloaded, never
executed. Existing files are skipped unless ``--force`` is passed.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(season: int, out_dir: Path, force: bool = False) -> Path:
    """Download one season of play-by-play to ``out_dir`` and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"play_by_play_{season}.parquet"
    if dest.exists() and not force:
        print(f"[skip] {dest.name} already present")
        return dest
    tmp = dest.with_suffix(".part")
    print(f"[get ] {season} ...", flush=True)
    with urllib.request.urlopen(URL.format(season=season), timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    print(f"[done] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", type=int, default=2014)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    for season in range(args.start, args.end + 1):
        download(season, RAW_DIR, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
