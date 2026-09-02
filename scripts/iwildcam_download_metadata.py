#!/usr/bin/env python3
"""
Download only the iWildCam v2.0 metadata CSV (~30 MB).

The full WILDS release is a ~12 GB image archive. Everything in the exploration
and mini-dataset stages needs only the metadata, so this fetches the CSV alone
and creates the directory stub WILDS expects -- letting the dataset be opened
with ``download=False`` without touching the images.

    python scripts/iwildcam_download_metadata.py
    python scripts/iwildcam_download_metadata.py --data_dir /scratch/$USER/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

BUNDLE_URL = (
    "https://worksheets.codalab.org/rest/bundles/"
    "0x6313da2b204647e79a14b468131fcd64/contents/blob/"
)
METADATA_URL = BUNDLE_URL + "metadata.csv"

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch the iWildCam metadata CSV")
    p.add_argument("--data_dir", default=str(_REPO_ROOT / "data"),
                   help="Dataset root (default: <repo>/data)")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if metadata.csv is already present")
    return p.parse_args()


def download_file(url: str, dest: Path) -> None:
    """Stream a URL to disk, printing progress."""
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.0f}%)",
                          end="")
    print()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir) / "iwildcam_v2.0"
    data_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_dir / "metadata.csv"
    release_path  = data_dir / "RELEASE_v2.0.txt"

    if metadata_path.exists() and not args.force:
        size_mb = metadata_path.stat().st_size / 1e6
        print(f"metadata.csv already present at {metadata_path} ({size_mb:.1f} MB) -- skipping.")
        print("Pass --force to re-download.")
    else:
        download_file(METADATA_URL, metadata_path)
        print(f"Saved metadata.csv ({metadata_path.stat().st_size / 1e6:.1f} MB)")

    # WILDS refuses to open a dataset directory without a release marker.
    if not release_path.exists():
        release_path.write_text(
            "metadata-only download for exploration (images not present)\n",
            encoding="utf-8",
        )
        print(f"Created {release_path}")

    print("\nNext:")
    print("  python scripts/iwildcam_explore_metadata.py      # dataset EDA, no images")
    print("  python scripts/iwildcam_build_mini_dataset.py    # build the mini subset")


if __name__ == "__main__":
    main()
