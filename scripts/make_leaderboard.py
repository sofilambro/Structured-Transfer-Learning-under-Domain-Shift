#!/usr/bin/env python3
"""
Rebuild ``results/iwildcam/leaderboard.csv`` from the archived run artifacts.

The committed ``leaderboard.csv`` is generated from these artifacts, so
regenerating it on a clean checkout must reproduce the same table -- that is the
check that the loading code and the artifacts still agree. ``--check`` compares
values numerically plus row order, rather than comparing bytes, so a difference
in float formatting is not treated as a failure.

It also diffs the result against ``leaderboard_paper.csv`` (the transcription of
the paper's Table 3). Twenty of twenty-one configurations match exactly. The one
that does not is T5S1, whose archived run was cut short by the wall-clock budget;
see ``results/iwildcam/README.md``.

    python scripts/make_leaderboard.py
    python scripts/make_leaderboard.py --check       # verify, write nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from structured_transfer.analysis.load import RESULTS_DIR, load_leaderboard, load_runs

#: Columns of the committed CSV, in order.
COLUMNS = [
    "frozen_depth", "tuned_depth", "scratch_depth", "scratch_start",
    "trainable_pct", "epochs_completed", "train_hours",
    "inference_gflops", "train_gflops",
    "id_acc", "id_bal_acc", "id_f1",
    "ood_acc", "ood_bal_acc", "ood_f1", "gpu",
]

#: Decimal places per column, matching the committed file.
ROUNDING = {
    "trainable_pct": 1, "epochs_completed": 2, "train_hours": 2,
    "inference_gflops": 3, "train_gflops": 3,
    "id_acc": 1, "id_bal_acc": 1, "id_f1": 1,
    "ood_acc": 1, "ood_bal_acc": 1, "ood_f1": 1,
}

#: Metrics compared against the paper table.
COMPARED = ["trainable_pct", "id_acc", "id_bal_acc", "id_f1",
            "ood_acc", "ood_bal_acc", "ood_f1"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild the iWildCam leaderboard")
    p.add_argument("--runs_dir", default=None,
                   help="Run artifacts directory. Default: results/iwildcam/runs.")
    p.add_argument("--out", default=None,
                   help="Output CSV. Default: results/iwildcam/leaderboard.csv.")
    p.add_argument("--check", action="store_true",
                   help="Compare against the committed file without overwriting it.")
    return p.parse_args()


def build(runs_dir: Path | None) -> pd.DataFrame:
    """Load the runs and shape them into the committed CSV's schema."""
    df = load_runs(runs_dir)
    df = df.sort_values(["ood_bal_acc", "ood_acc"], ascending=False)
    df = df[COLUMNS].round(ROUNDING)
    df.index.name = "config"
    return df


def compare_to_paper(df: pd.DataFrame) -> int:
    """Print a per-configuration diff against Table 3. Returns the mismatch count."""
    paper = load_leaderboard(prefer="paper")

    print("\nComparison against the paper's Table 3:")
    mismatched = 0
    for config in df.index:
        if config not in paper.index:
            print(f"  {config}: not present in the paper table")
            mismatched += 1
            continue
        diffs = [
            f"{col} {df.loc[config, col]:.1f} vs {paper.loc[config, col]:.1f}"
            for col in COMPARED
            if abs(df.loc[config, col] - paper.loc[config, col]) > 0.051
        ]
        if diffs:
            mismatched += 1
            print(f"  {config:<8} differs: {'; '.join(diffs)}")

    matched = len(df) - mismatched
    print(f"\n  {matched}/{len(df)} configurations match the paper exactly.")
    if mismatched:
        print("  See results/iwildcam/README.md for why T5S1 differs "
              "(its archived run hit the wall-clock ceiling at 19.6 epochs).")
    return mismatched


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else RESULTS_DIR / "iwildcam" / "leaderboard.csv"

    df = build(Path(args.runs_dir) if args.runs_dir else None)
    print(f"Built leaderboard from {len(df)} run artifacts.\n")
    print(df[["trainable_pct", "epochs_completed", "id_acc", "id_bal_acc",
              "ood_acc", "ood_bal_acc", "ood_f1"]].to_string())

    compare_to_paper(df)

    if args.check:
        if not out_path.exists():
            raise SystemExit(f"\n--check: {out_path} does not exist yet.")
        committed = pd.read_csv(out_path).set_index("config")
        # Reindex before comparing: row order is part of the file, so compare
        # the values on a shared index and the ordering separately.
        same_order = list(committed.index) == list(df.index)
        aligned = committed.reindex(df.index)
        numeric = [c for c in COLUMNS if c != "gpu"]
        close = ((aligned[numeric] - df[numeric]).abs() < 1e-9).all().all()
        if same_order and close:
            print(f"\n--check: {out_path.name} is up to date.")
        else:
            raise SystemExit(
                f"\n--check FAILED: {out_path.name} differs from the rebuilt table"
                f"{' (row order differs)' if not same_order else ''}. "
                f"Re-run without --check to regenerate."
            )
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, lineterminator="\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
