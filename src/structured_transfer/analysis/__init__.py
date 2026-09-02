"""
Loading result artifacts and regenerating the paper's figures.

``load`` has no Torch dependency, so the leaderboard and the tables can be
inspected on a laptop with nothing but pandas installed. ``figures`` additionally
needs matplotlib.
"""

from .load import (
    load_leaderboard,
    load_run_curves,
    load_runs,
    load_tinyimagenet_runs,
    load_tinyimagenet_tables,
)

__all__ = [
    "load_leaderboard",
    "load_runs",
    "load_run_curves",
    "load_tinyimagenet_tables",
    "load_tinyimagenet_runs",
]
