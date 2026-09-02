"""
Load experiment artifacts into tidy DataFrames.

Two kinds of source exist, and the distinction matters when reading results:

**Run artifacts** -- ``results/iwildcam/runs/timing_*.json``, one per
configuration, written by ``scripts/iwildcam_run.py``. These carry the raw
per-epoch evaluation logs alongside the final metrics.

**Reference tables** -- ``leaderboard_paper.csv``, ``selfer_paper.csv``,
``transfer_paper.csv``. These are canonical for any figure quoted in the
write-up.

``load_leaderboard(prefer="paper")`` is the default, so figures agree with the
reference tables. Pass ``prefer="runs"`` to plot the archived logs instead; the
two agree on 20 of the 21 iWildCam configurations, and
``results/iwildcam/README.md`` documents the exception.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..partitions import depths

#: Repository root, resolved from this file's location.
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"


def load_runs(runs_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load every ``timing_*.json`` in a directory into one row per configuration.

    Returns a DataFrame indexed by ``config`` with the structural axes
    (``frozen_depth``, ``tuned_depth``, ``scratch_depth``, ``scratch_start``),
    cost columns, and ID/OOD metrics as percentages.

    Metrics are read from ``final_results``, i.e. the end-of-run evaluation, and
    are scaled to percent so they match the paper's tables directly.
    """
    runs_dir = Path(runs_dir or RESULTS_DIR / "iwildcam" / "runs")
    paths = sorted(runs_dir.glob("timing_*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No timing_*.json found in {runs_dir}. Run scripts/iwildcam_run.py "
            f"--all, or point at the archived runs directory."
        )

    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            run = json.load(handle)

        partition = tuple(run["config"]["partition"])
        row = {
            "config":           run["label"],
            **depths(partition),
            "trainable_params": run["n_trainable"],
            "total_params":     run["n_total"],
            "trainable_pct":    100 * run["n_trainable"] / run["n_total"],
            "epochs_completed": run["epochs_completed"],
            "train_hours":      run["train_elapsed_s"] / 3600,
            "secs_per_epoch":   run["secs_per_epoch"],
            "inference_gflops": run["inference_flops"] / 1e9,
            "train_gflops":     run["training_flops_per_sample"] / 1e9,
            "gpu":              run.get("gpu_name", "unknown"),
        }

        # final_results keys are split names; flatten to <split>_<metric> in %.
        for split, metrics in run["final_results"].items():
            prefix = {"id_val": "id", "ood_val": "ood"}.get(split, split)
            for metric, value in metrics.items():
                if metric == "n" or value is None:
                    continue
                name = {"balanced_acc": "bal_acc", "acc": "acc", "f1": "f1"}.get(metric)
                if name is None:
                    continue
                row[f"{prefix}_{name}"] = 100 * value

        rows.append(row)

    df = pd.DataFrame(rows).set_index("config")
    return df.sort_values("ood_bal_acc", ascending=False)


def load_run_curves(runs_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Long-format per-epoch training curves: one row per (config, evaluation).

    Only the run artifacts carry these -- the paper reports endpoints only -- so
    any curve plot necessarily uses ``prefer="runs"`` data.
    """
    runs_dir = Path(runs_dir or RESULTS_DIR / "iwildcam" / "runs")
    rows = []
    for path in sorted(runs_dir.glob("timing_*.json")):
        with open(path, encoding="utf-8") as handle:
            run = json.load(handle)
        for entry in run.get("eval_log", []):
            rows.append({
                "config":      run["label"],
                "epoch":       entry.get("epoch"),
                "step":        entry.get("step"),
                "elapsed_s":   entry.get("elapsed_s"),
                "train_loss":  entry.get("train_loss"),
                "train_acc":   entry.get("train_acc"),
                "id_acc":      _pct(entry.get("id_val_acc")),
                "id_bal_acc":  _pct(entry.get("id_val_balanced_acc")),
                "id_f1":       _pct(entry.get("id_val_f1")),
                "ood_acc":     _pct(entry.get("ood_val_acc")),
                "ood_bal_acc": _pct(entry.get("ood_val_balanced_acc")),
                "ood_f1":      _pct(entry.get("ood_val_f1")),
            })
    return pd.DataFrame(rows)


def load_leaderboard(
    prefer: str = "paper",
    results_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    The iWildCam leaderboard, from either the paper table or the run artifacts.

    Args:
        prefer: ``"paper"`` reads the transcribed Table 3 -- canonical, and what
                the figures use by default. ``"runs"`` derives it from
                ``runs/*.json``. The two agree on 20 of 21 configurations; T5S1
                differs because its archived run was cut short by the wall-clock
                budget.

    Returns a DataFrame indexed by ``config``, sorted by OOD balanced accuracy.
    """
    results_dir = Path(results_dir or RESULTS_DIR) / "iwildcam"

    if prefer == "runs":
        return load_runs(results_dir / "runs")
    if prefer != "paper":
        raise ValueError(f"prefer must be 'paper' or 'runs', got {prefer!r}")

    df = pd.read_csv(results_dir / "leaderboard_paper.csv").set_index("config")
    return df.sort_values("ood_bal_acc", ascending=False)


def load_tinyimagenet_tables(
    results_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The TinyImageNet selfer and transfer boards (paper Tables 4 and 5).

    Returns ``(selfer, transfer)``, each indexed by ``config``.

    These are transcriptions, not run output: Experiment 1's artifacts were not
    preserved. Once ``scripts/tinyimagenet_run.py`` has produced real runs,
    :func:`load_tinyimagenet_runs` supersedes this.
    """
    results_dir = Path(results_dir or RESULTS_DIR) / "tinyimagenet"
    selfer   = pd.read_csv(results_dir / "selfer_paper.csv").set_index("config")
    transfer = pd.read_csv(results_dir / "transfer_paper.csv").set_index("config")
    return selfer, transfer


def load_tinyimagenet_runs(runs_dir: str | Path | None = None) -> pd.DataFrame | None:
    """
    Per-run Experiment 1 artifacts, if any exist yet.

    Returns ``None`` when the directory is absent or empty, which is the state
    of a fresh clone. Two figures depend on this and are skipped until it
    returns data: the A/B symmetry check (paper Figure 2) needs per-*direction*
    results, and the ID/OOD-C rank correlation (Figure 6) needs all 56
    individual runs -- neither is recoverable from the aggregated tables.
    """
    runs_dir = Path(runs_dir or RESULTS_DIR / "tinyimagenet" / "runs")
    paths = sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []
    if not paths:
        return None

    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            run = json.load(handle)
        rows.append({
            "config":      run["label"],
            "direction":   run["direction"],
            "is_selfer":   run["direction"] in ("AtoA", "BtoB"),
            "clean_acc":   100 * run["final_results"]["clean_acc"],
            "final_acc":   100 * run["final_results"].get("final_acc", float("nan")),
            "corrupt_acc": 100 * run["final_results"].get("corrupt_acc", float("nan")),
            "best_epoch":  run["final_results"].get("best_epoch"),
        })
    return pd.DataFrame(rows)


def _pct(value):
    """Scale a fraction to percent, tolerating None."""
    return None if value is None else 100 * value
