"""
Downloads only the metadata.csv for iWildCam v2.0 (~30 MB).
Creates the directory structure expected by WILDS so the dataset can be
loaded with download=False for EDA without touching the ~12 GB image archive.

Usage:
    python download_metadata.py
"""
import os
import sys
import requests
from pathlib import Path

BUNDLE_URL = (
    "https://worksheets.codalab.org/rest/bundles/"
    "0x6313da2b204647e79a14b468131fcd64/contents/blob/"
)
METADATA_URL  = BUNDLE_URL + "metadata.csv"
DATA_DIR      = Path(__file__).parent.parent / "data" / "iwildcam_v2.0"
METADATA_PATH = DATA_DIR / "metadata.csv"
RELEASE_PATH  = DATA_DIR / "RELEASE_v2.0.txt"


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    print(f"  → {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB  ({pct:.0f}%)", end="")
    print()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if METADATA_PATH.exists():
        print(f"metadata.csv already present at {METADATA_PATH} — skipping download.")
    else:
        download_file(METADATA_URL, METADATA_PATH)
        print(f"Saved metadata.csv ({METADATA_PATH.stat().st_size / 1e6:.1f} MB)")

    if not RELEASE_PATH.exists():
        RELEASE_PATH.write_text(
            "metadata-only download for EDA (images not present)\n"
        )
        print(f"Created {RELEASE_PATH}")

    print("\nDone. You can now run the EDA notebook with download=False.")


if __name__ == "__main__":
    main()
