#!/usr/bin/env python3
"""
Exploratory analysis of the iWildCam metadata (formerly notebook 01).

Metadata only -- no images are read, so this runs on a laptop in seconds after
``iwildcam_download_metadata.py``. It answers the questions that motivate the
experiment's design choices:

* How imbalanced is the label distribution?  (justifies balanced accuracy and
  macro-F1 over raw accuracy, and the inverse-frequency training sampler)
* Do the OOD splits really use unseen camera locations?  (the shift the study
  measures -- if OOD locations overlapped train, there would be no shift)
* Do the OOD splits contain classes absent from training?

    python scripts/iwildcam_explore_metadata.py
    python scripts/iwildcam_explore_metadata.py --data_dir /scratch/$USER/data --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structured_transfer.analysis import style  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Metadata split names -> our canonical names. "val"/"test" are the OOD splits;
# the CSV gives them no marker, which is easy to misread.
SPLIT_NAMES = {
    "train":   ("train",    "ID"),
    "id_val":  ("id_val",   "ID"),
    "val":     ("ood_val",  "OOD"),
    "id_test": ("id_test",  "ID"),
    "test":    ("ood_test", "OOD"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="iWildCam metadata EDA")
    p.add_argument("--data_dir", default=str(_REPO_ROOT / "data"))
    p.add_argument("--out_dir", default=str(_REPO_ROOT / "results" / "iwildcam" / "eda"))
    p.add_argument("--no-plots", action="store_true", help="Print the tables only.")
    return p.parse_args()


def split_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw_split, group in df.groupby("split"):
        name, domain = SPLIT_NAMES.get(raw_split, (raw_split, "?"))
        rows.append({
            "split":            name,
            "domain":           domain,
            "samples":          len(group),
            "unique_classes":   group["y"].nunique(),
            "unique_locations": group["location_remapped"].nunique(),
        })
    return pd.DataFrame(rows).set_index("split").sort_values("samples", ascending=False)


def report_imbalance(train: pd.DataFrame) -> pd.Series:
    counts = train["y"].value_counts().sort_values(ascending=False)
    tiny = (counts < 10).sum()

    print("\n== Class imbalance (train) ==")
    print(f"  Training samples      : {len(train):,}")
    print(f"  Classes present       : {len(counts)}")
    print(f"  Most common class     : {counts.iloc[0]:,} images")
    print(f"  Median class size     : {int(counts.median()):,} images")
    print(f"  Rarest class          : {counts.iloc[-1]:,} images")
    print(f"  Imbalance ratio       : {counts.iloc[0] / counts.iloc[-1]:.0f}x")
    print(f"  Classes with <10 imgs : {tiny} ({100 * tiny / len(counts):.0f}%)")
    print("  -> raw accuracy is dominated by the head of this distribution, which")
    print("     is why the study ranks configurations by balanced accuracy.")
    return counts


def report_domain_overlap(df: pd.DataFrame) -> None:
    """The crux: OOD splits must share no camera location with training."""
    train_locs = set(df[df["split"] == "train"]["location_remapped"])

    print("\n== Location overlap with training ==")
    print(f"  {'split':<10} {'locations':>10} {'shared':>8}")
    for raw_split, (name, domain) in SPLIT_NAMES.items():
        if raw_split == "train":
            continue
        locs = set(df[df["split"] == raw_split]["location_remapped"])
        shared = len(locs & train_locs)
        flag = ""
        if domain == "OOD" and shared > 0:
            flag = "  <-- UNEXPECTED: OOD split shares locations with train"
        if domain == "ID" and shared == 0:
            flag = "  <-- UNEXPECTED: ID split shares no locations with train"
        print(f"  {name:<10} {len(locs):>10} {shared:>8}{flag}")

    print("\n== Class overlap with training ==")
    train_classes = set(df[df["split"] == "train"]["y"])
    print(f"  {'split':<10} {'classes':>8} {'shared':>8} {'unseen':>8}")
    for raw_split, (name, _) in SPLIT_NAMES.items():
        if raw_split == "train":
            continue
        classes = set(df[df["split"] == raw_split]["y"])
        print(f"  {name:<10} {len(classes):>8} {len(classes & train_classes):>8} "
              f"{len(classes - train_classes):>8}")
    print("  -> classes unseen in training can never be predicted correctly and")
    print("     drag macro metrics down; this is a property of the benchmark.")


def make_plots(df: pd.DataFrame, train: pd.DataFrame, counts: pd.Series,
               out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    style.apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Class imbalance ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    axes[0].bar(range(len(counts)), counts.values, width=1.0,
                color=style.SERIES_BLUE)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Class rank")
    axes[0].set_ylabel("Images (log scale)")
    axes[0].set_title("Class frequency, sorted", loc="left")

    axes[1].hist(counts.values, bins=40, color=style.SERIES_BLUE)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Images per class")
    axes[1].set_ylabel("Number of classes (log)")
    axes[1].set_title("Distribution of class sizes", loc="left")

    fig.suptitle("iWildCam is long-tailed: the largest class is "
                 f"{counts.iloc[0] / counts.iloc[-1]:.0f}x the smallest",
                 x=0.008, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "class_imbalance.png")
    plt.close(fig)

    # ── Location distribution ────────────────────────────────────────────────
    loc_counts = train["location_remapped"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    ax.bar(range(len(loc_counts)), loc_counts.values, width=1.0,
           color=style.SERIES_ORANGE)
    ax.set_yscale("log")
    ax.set_xlabel("Location rank")
    ax.set_ylabel("Images (log scale)")
    ax.set_title(f"Images per camera location, training split "
                 f"({len(loc_counts)} locations)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "location_distribution.png")
    plt.close(fig)

    # ── Species richness ─────────────────────────────────────────────────────
    richness = train.groupby("location_remapped")["y"].nunique()
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.hist(richness.values, bins=30, color=style.SERIES_AQUA)
    ax.set_xlabel("Distinct species observed at a location")
    ax.set_ylabel("Number of locations")
    ax.set_title(f"Species richness per camera location "
                 f"(mean {richness.mean():.1f}, max {richness.max()})",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "species_richness.png")
    plt.close(fig)

    # ── Temporal distribution ────────────────────────────────────────────────
    if "datetime" in df.columns:
        parsed = pd.to_datetime(df["datetime"], errors="coerce", format="mixed")
        frame = pd.DataFrame({
            "split": df["split"], "month": parsed.dt.month, "hour": parsed.dt.hour,
        })
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
        for raw_split, label, color in [
            ("train", "Train (ID)",  style.SERIES_BLUE),
            ("test",  "OOD test",    style.SERIES_ORANGE),
        ]:
            subset = frame[frame["split"] == raw_split]
            axes[0].hist(subset["month"].dropna(), bins=12, range=(1, 13),
                         alpha=0.65, label=label, color=color, density=True)
            axes[1].hist(subset["hour"].dropna(), bins=24, range=(0, 24),
                         alpha=0.65, label=label, color=color, density=True)
        for ax, xlabel, title in zip(
            axes, ["Month", "Hour of day"],
            ["Capture month", "Capture hour"],
        ):
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.set_title(title, loc="left")
            ax.legend()
        fig.suptitle("Location shift comes with temporal shift too",
                     x=0.008, ha="left", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "temporal_distribution.png")
        plt.close(fig)

    print(f"\nPlots written to {out_dir}")


def main() -> None:
    args = parse_args()
    meta_path = Path(args.data_dir) / "iwildcam_v2.0" / "metadata.csv"

    if not meta_path.exists():
        raise SystemExit(
            f"Metadata not found at {meta_path}.\n"
            f"Run: python scripts/iwildcam_download_metadata.py --data_dir {args.data_dir}"
        )

    df = pd.read_csv(meta_path)
    print(f"Loaded {len(df):,} rows from {meta_path}")

    print("\n== Split overview ==")
    print(split_overview(df).to_string())

    train = df[df["split"] == "train"]
    counts = report_imbalance(train)
    report_domain_overlap(df)

    if not args.no_plots:
        make_plots(df, train, counts, Path(args.out_dir))


if __name__ == "__main__":
    main()
