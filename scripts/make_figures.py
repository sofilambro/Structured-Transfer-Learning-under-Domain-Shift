#!/usr/bin/env python3
"""
Regenerate every figure from the committed result data.

Figures read either the archived run artifacts or the reference result tables;
any figure whose inputs are unavailable is skipped with a printed explanation.

Output is written to ``results/iwildcam/figures/`` and
``results/tinyimagenet/figures/``, both git-ignored: the figures are derived
artifacts, and the data behind them is what the repository commits.

    python scripts/make_figures.py                 # reference tables (default)
    python scripts/make_figures.py --source runs   # archived run logs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structured_transfer.analysis.figures import generate_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate the paper's figures")
    p.add_argument(
        "--source", choices=("paper", "runs"), default="paper",
        help="Data source for the iWildCam figures. 'paper' (default) uses the "
             "reference leaderboard, so figures agree with the write-up; 'runs' "
             "uses the archived run logs.",
    )
    p.add_argument("--results_dir", default=None,
                   help="Results tree root. Default: <repo>/results.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    written = generate_all(results_dir=args.results_dir, prefer=args.source)
    if not written:
        raise SystemExit("No figures were produced -- check the messages above.")


if __name__ == "__main__":
    main()
