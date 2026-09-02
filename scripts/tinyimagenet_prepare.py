#!/usr/bin/env python3
"""
Prepare the TinyImageNet A/B class split and the corrupted validation sets.

Run once before any Experiment-1 training. Three stages:

``--stage split``       derive the six semantic groups, split the 200 classes into
                        disjoint halves A and B, and write them to
                        ``data/tinyimagenet_splits/``.
``--stage corrupt``     build the TinyImageNet-C-style corrupted validation set
                        for each subset (5,000 images each).
``--stage all``         both.

The split files are small and worth committing: the WordNet grouping depends on
which corpus version is installed, so pinning the derived split is what makes the
experiment reproducible across machines. See the caveat in
``structured_transfer.tinyimagenet.splits``.

    python scripts/tinyimagenet_prepare.py --stage all
    python scripts/tinyimagenet_prepare.py --stage split --dump-groups
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from structured_transfer.tinyimagenet.config import CONFIG
from structured_transfer.tinyimagenet.corruptions import CORRUPTIONS, build_corrupted_set
from structured_transfer.tinyimagenet.data import (
    TinyImageNetSubset,
    read_wnids,
    read_words,
    save_subset_wnids,
)
from structured_transfer.tinyimagenet.splits import (
    describe_split,
    make_class_split,
    semantic_groups,
    semantic_groups_from_csv,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare TinyImageNet for Experiment 1")
    p.add_argument("--stage", choices=("split", "corrupt", "all"), default="all")
    p.add_argument("--data_dir", default=str(_REPO_ROOT / "data"))
    p.add_argument("--split_seed", type=int, default=CONFIG["split_seed"],
                   help="Split seed; the paper's 'split 1' is seed 1.")
    p.add_argument("--corruption_seed", type=int, default=CONFIG["corruption_seed"])
    p.add_argument("--groups_csv", default=None,
                   help="Load a committed wnid->group mapping instead of rederiving it.")
    p.add_argument("--dump-groups", action="store_true",
                   help="Write the derived grouping to class_groups.csv for committing.")
    return p.parse_args()


def resolve_root(data_dir: Path) -> Path:
    """Locate the tiny-imagenet-200 directory, with a helpful error if absent."""
    root = data_dir / "tiny-imagenet-200"
    if not (root / "wnids.txt").exists():
        raise SystemExit(
            f"TinyImageNet not found at {root}.\n\n"
            f"Download and unzip it there:\n"
            f"  curl -O {DOWNLOAD_URL}\n"
            f"  unzip tiny-imagenet-200.zip -d {data_dir}\n\n"
            f"Expected layout: {root}/{{wnids.txt, words.txt, train/, val/}}"
        )
    return root


def stage_split(root: Path, out_dir: Path, args: argparse.Namespace) -> None:
    wnids  = read_wnids(root)
    labels = read_words(root)
    print(f"Read {len(wnids)} classes from {root / 'wnids.txt'}")

    if args.groups_csv:
        groups = semantic_groups_from_csv(args.groups_csv)
        print(f"Loaded semantic groups from {args.groups_csv}")
    else:
        groups = semantic_groups(wnids, labels)
        print("Derived semantic groups from WordNet")

    subset_a, subset_b = make_class_split(wnids, groups, seed=args.split_seed)

    print()
    print(describe_split(subset_a, subset_b, groups))

    out_dir.mkdir(parents=True, exist_ok=True)
    save_subset_wnids(subset_a, out_dir / f"split{args.split_seed}_A.txt")
    save_subset_wnids(subset_b, out_dir / f"split{args.split_seed}_B.txt")
    print(f"\nWrote subset files to {out_dir}")

    if args.dump_groups:
        groups_path = out_dir / "class_groups.csv"
        with open(groups_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["wnid", "label", "group"])
            for wnid in sorted(wnids):
                writer.writerow([wnid, labels.get(wnid, ""), groups.get(wnid, "")])
        print(f"Wrote {groups_path} -- commit this to pin the grouping.")


def stage_corrupt(root: Path, splits_dir: Path, out_root: Path,
                  args: argparse.Namespace) -> None:
    from structured_transfer.tinyimagenet.data import load_subset_wnids

    total_severities = sum(CORRUPTIONS.values())
    print(f"Corruptions: {', '.join(f'{k} (1-{v})' for k, v in CORRUPTIONS.items())}")
    print(f"Each image gets exactly one corruption at one severity "
          f"({total_severities} severity levels across 6 corruptions).\n")

    for subset_name in ("A", "B"):
        subset_path = splits_dir / f"split{args.split_seed}_{subset_name}.txt"
        if not subset_path.exists():
            raise SystemExit(
                f"Missing {subset_path}. Run with --stage split first."
            )

        wnids   = load_subset_wnids(subset_path)
        dataset = TinyImageNetSubset(root, wnids, split="val", transform=None)
        print(f"Subset {subset_name}: {len(dataset)} clean validation images")

        # build_corrupted_set works on (name, class) pairs plus a loader, so it
        # stays independent of how the dataset indexes files.
        path_by_name = {path.name: path for path, _ in dataset.samples}
        images = [(path.name, label) for path, label in dataset.samples]

        out_dir = out_root / f"split{args.split_seed}_{subset_name}"
        manifest = build_corrupted_set(
            images=images,
            load_image=lambda name: Image.open(path_by_name[name]).convert("RGB"),
            out_dir=out_dir,
            seed=args.corruption_seed,
        )
        print(f"  -> {manifest.parent}  (manifest: {manifest.name})\n")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    root = resolve_root(data_dir)

    splits_dir  = data_dir / "tinyimagenet_splits"
    corrupt_dir = data_dir / "tinyimagenet_corrupted"

    if args.stage in ("split", "all"):
        print("=" * 68)
        print("  Stage 1: semantic A/B class split")
        print("=" * 68)
        stage_split(root, splits_dir, args)

    if args.stage in ("corrupt", "all"):
        print("\n" + "=" * 68)
        print("  Stage 2: TinyImageNet-C-style corrupted validation sets")
        print("=" * 68)
        stage_corrupt(root, splits_dir, corrupt_dir, args)

    print("\nNext: train the two source networks, then the transfer runs:")
    print("  python scripts/tinyimagenet_run.py --config SSSS --direction AtoA")
    print("  python scripts/tinyimagenet_run.py --config SSSS --direction BtoB")
    print("  python scripts/tinyimagenet_run.py --all")


if __name__ == "__main__":
    main()
