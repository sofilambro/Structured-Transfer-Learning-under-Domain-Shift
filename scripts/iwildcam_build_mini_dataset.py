#!/usr/bin/env python3
"""
Build mini-iWildCam: a ~50k-image subset for laptop-scale iteration (was notebook 02).

The full dataset is 203k images and needs a GPU-hours budget per configuration.
The mini subset exists so the pipeline can be exercised end to end locally. It is
**not** a scientific artifact: every archived result used ``dataset_mode="full"``,
and mini numbers are not comparable to the leaderboard.

Two stages:

``--part metadata``  filter classes and subsample rows. Needs only metadata.csv;
                     runs in seconds, no network.
``--part images``    fetch just the selected images. Streams the 12 GB archive
                     over the network and writes only files in the subset, so
                     peak disk stays near the ~3 GB of output. The stock WILDS
                     downloader instead saves the archive *and* extracts it,
                     needing ~32 GB of free space.

    python scripts/iwildcam_build_mini_dataset.py --part metadata
    python scripts/iwildcam_build_mini_dataset.py --part images
    python scripts/iwildcam_build_mini_dataset.py --part all
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from io import RawIOBase
from pathlib import Path

import pandas as pd

BUNDLE_URL = (
    "https://worksheets.codalab.org/rest/bundles/"
    "0x6313da2b204647e79a14b468131fcd64/contents/blob/"
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Selection criteria. Classes too rare to learn *or* too rare to evaluate OOD are
# dropped outright -- keeping a class with three OOD images adds noise to macro
# metrics without adding signal.
MIN_TRAIN_IMAGES    = 100
MIN_OOD_TEST_IMAGES = 20
TARGET_TOTAL        = 50_000
# Per-class sampling weight is count^ALPHA. 1.0 preserves the original imbalance
# exactly, 0.0 fully balances. 0.8 compresses the tail mildly, keeping the
# distribution's shape realistic while making rare classes learnable at 50k.
ALPHA = 0.8
SEED  = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the mini-iWildCam subset")
    p.add_argument("--part", choices=("metadata", "images", "all"), default="metadata")
    p.add_argument("--data_dir", default=str(_REPO_ROOT / "data"))
    p.add_argument("--target_total", type=int, default=TARGET_TOTAL)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def subsample_split(group: pd.DataFrame, target: int, alpha: float, seed: int) -> pd.DataFrame:
    """
    Sample about ``target`` rows from one split, compressing class imbalance.

    Per-class quota is proportional to ``count ** alpha``, floored at 1 so no
    surviving class is sampled out of existence.
    """
    if len(group) <= target:
        return group

    class_counts = group.groupby("y").size()
    weights      = class_counts.pow(alpha)
    quotas       = (weights / weights.sum() * target).round().clip(lower=1).astype(int)

    parts = []
    for cls, quota in quotas.items():
        members = group[group["y"] == cls]
        parts.append(members.sample(n=min(quota, len(members)), random_state=seed))
    return pd.concat(parts)


def build_metadata(data_dir: Path, target_total: int, seed: int) -> pd.DataFrame:
    meta_full = data_dir / "iwildcam_v2.0" / "metadata.csv"
    mini_dir  = data_dir / "iwildcam_mini"
    mini_dir.mkdir(parents=True, exist_ok=True)

    if not meta_full.exists():
        raise SystemExit(
            f"Full metadata not found at {meta_full}.\n"
            f"Run: python scripts/iwildcam_download_metadata.py --data_dir {data_dir}"
        )

    df = pd.read_csv(meta_full)
    print(f"Full dataset: {len(df):,} rows, {df['y'].nunique()} classes")

    train_counts    = df[df["split"] == "train"].groupby("y").size()
    ood_test_counts = df[df["split"] == "test"].groupby("y").size()
    valid_classes = (
        set(train_counts[train_counts >= MIN_TRAIN_IMAGES].index)
        & set(ood_test_counts[ood_test_counts >= MIN_OOD_TEST_IMAGES].index)
    )
    filtered = df[df["y"].isin(valid_classes)].copy()

    print(f"\nClass filter (>= {MIN_TRAIN_IMAGES} train, >= {MIN_OOD_TEST_IMAGES} OOD test):")
    print(f"  classes : {df['y'].nunique()} -> {len(valid_classes)}")
    print(f"  images  : {len(df):,} -> {len(filtered):,}")

    # Allocate the budget across splits in proportion to their filtered sizes, so
    # the mini set keeps the original train/val/test balance.
    split_sizes   = filtered.groupby("split").size()
    split_targets = (split_sizes / split_sizes.sum() * target_total).round().astype(int)

    print("\nPer-split targets:")
    for split, target in split_targets.items():
        actual = split_sizes[split]
        action = f"keep all {actual:,}" if actual <= target else f"sample {target:,} of {actual:,}"
        print(f"  {split:<10} {action}")

    parts = [
        subsample_split(group, int(split_targets[split]), ALPHA, seed)
        for split, group in filtered.groupby("split")
    ]
    mini = pd.concat(parts).reset_index(drop=True)

    mini_meta = mini_dir / "metadata.csv"
    mini.to_csv(mini_meta, index=False)
    print(f"\nMini dataset: {len(mini):,} images, {mini['y'].nunique()} classes")
    print(f"Saved -> {mini_meta}  ({mini_meta.stat().st_size / 1e6:.1f} MB)")

    # The mini subset is only useful if it preserves the *shift*: OOD splits must
    # still share no camera location with train.
    train_locs = set(mini[mini["split"] == "train"]["location_remapped"])
    print("\nLocation overlap with train (OOD splits must be 0):")
    for raw, name in [("id_val", "id_val"), ("val", "ood_val"),
                      ("id_test", "id_test"), ("test", "ood_test")]:
        locs = set(mini[mini["split"] == raw]["location_remapped"])
        print(f"  {name:<10} {len(locs):>4} locations, {len(locs & train_locs):>4} shared")

    return mini


class _IterStream(RawIOBase):
    """
    Adapt a ``requests`` chunk iterator to a file-like object for ``tarfile``.

    This is what lets the archive be read as a stream: ``tarfile`` in ``r|gz``
    mode reads strictly forward, so members can be inspected and selectively
    extracted as bytes arrive, without the archive ever landing on disk.
    """

    def __init__(self, iterator):
        self._iter = iterator
        self._buf  = b""

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        n = len(b)
        while len(self._buf) < n:
            try:
                self._buf += next(self._iter)
            except StopIteration:
                break
        out, self._buf = self._buf[:n], self._buf[n:]
        b[:len(out)] = out
        return len(out)


def download_images(data_dir: Path) -> None:
    import requests

    mini_dir  = data_dir / "iwildcam_mini"
    mini_meta = mini_dir / "metadata.csv"
    img_dir   = mini_dir / "train"

    if not mini_meta.exists():
        raise SystemExit(
            f"Mini metadata not found at {mini_meta}.\n"
            f"Run this script with --part metadata first."
        )

    img_dir.mkdir(parents=True, exist_ok=True)
    mini = pd.read_csv(mini_meta)

    needed    = set(mini["filename"].values)
    on_disk   = {p.name for p in img_dir.glob("*.jpg")}
    to_fetch  = needed - on_disk

    print(f"Images needed   : {len(needed):,}")
    print(f"Already on disk : {len(on_disk):,}")
    print(f"To download     : {len(to_fetch):,}")
    if not to_fetch:
        print("Nothing to do.")
        return

    free_gb = shutil.disk_usage(data_dir).free / 1e9
    print(f"Free disk space : {free_gb:.1f} GB")
    if free_gb < 4:
        raise SystemExit("Need at least ~4 GB free for the mini image set. Aborting.")

    print("\nStreaming the archive (~12 GB transfer, only selected files are written).")
    print("Expect 30-90 minutes depending on the connection.\n")

    saved = skipped = 0
    with requests.get(BUNDLE_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        stream = _IterStream(response.iter_content(chunk_size=4 << 20))
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if name not in to_fetch:
                    skipped += 1
                    continue
                handle = tar.extractfile(member)
                if handle:
                    (img_dir / name).write_bytes(handle.read())
                    saved += 1
                    if saved % 500 == 0:
                        print(f"\r  saved {saved:,} / {len(to_fetch):,}", end="")

    used_gb = sum(p.stat().st_size for p in img_dir.glob("*.jpg")) / 1e9
    print(f"\n\nDone. Saved {saved:,}, skipped {skipped:,}.")
    print(f"Images on disk: {used_gb:.2f} GB")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.part in ("metadata", "all"):
        build_metadata(data_dir, args.target_total, args.seed)
    if args.part in ("images", "all"):
        print("\n" + "=" * 68)
        download_images(data_dir)

    if args.part == "metadata":
        print("\nNext: python scripts/iwildcam_build_mini_dataset.py --part images")


if __name__ == "__main__":
    main()
