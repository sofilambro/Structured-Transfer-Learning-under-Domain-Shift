#!/usr/bin/env python3
"""
Run one or more F/T/S configurations on iWildCam (paper, Experiment 2).

This is the sweep driver. The original version hardcoded the T5S1 partition in
its source, so the 21-run sweep behind the paper was produced by hand-editing the
file between submissions; the partition is now a command-line argument.

The JSON output schema is unchanged, so files written here load alongside the
archived runs in ``results/iwildcam/runs/``.

Examples
--------
    # One configuration
    python scripts/iwildcam_run.py --partition F2T3S1

    # The full 21-configuration sweep, in the paper's order
    python scripts/iwildcam_run.py --all

    # Pipeline check: 5 representative partitions, 2 epochs, mini dataset
    python scripts/iwildcam_run.py --smoke

    # Cluster: one array task per configuration (see slurm/iwildcam_sweep.sbatch)
    python scripts/iwildcam_run.py --index $SLURM_ARRAY_TASK_ID \\
        --data_dir /scratch/$USER/data --max_time_min 180
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from structured_transfer.budget import EpochTimeBudgetTracker
from structured_transfer.iwildcam.config import CONFIG
from structured_transfer.iwildcam.evaluate import Evaluator
from structured_transfer.iwildcam.models import build_model
from structured_transfer.iwildcam.train import train
from structured_transfer.partitions import (
    RESNET50_BLOCKS,
    config_label,
    enumerate_configs,
    parse_label,
)
from structured_transfer.utils import estimate_flops, jsonify

#: Five partitions spanning the design space, for the smoke test.
SMOKE_PARTITIONS = ("F5S1", "F4T1S1", "F2T2S2", "T5S1", "S6")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="iWildCam F/T/S transfer experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    selection = p.add_mutually_exclusive_group()
    selection.add_argument(
        "--partition", metavar="LABEL",
        help="Partition label, e.g. F2T3S1, T5S1, S6. Default: the config default (T5S1).",
    )
    selection.add_argument(
        "--all", action="store_true",
        help="Run all 21 monotonic partitions sequentially.",
    )
    selection.add_argument(
        "--index", type=int, metavar="N",
        help="Run partition N of the 21 (0-based). For SLURM array jobs.",
    )
    selection.add_argument(
        "--smoke", action="store_true",
        help=f"Pipeline check: {', '.join(SMOKE_PARTITIONS)} for 2 epochs on the mini dataset.",
    )

    p.add_argument("--data_dir", default=None, help="Override the data directory.")
    p.add_argument("--dataset_mode", choices=("mini", "full"), default=None,
                   help="Override dataset size. Archived runs used 'full'.")
    p.add_argument("--out_dir", default=None,
                   help="Where result JSON files are written. Default: results/iwildcam/runs.")
    p.add_argument("--max_epochs", type=int, default=100,
                   help="Epoch ceiling (default: 100).")
    p.add_argument("--max_time_min", type=float, default=185.0,
                   help="Wall-clock training ceiling per run, in minutes (default: 185, "
                        "matching the archived sweep).")
    p.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    p.add_argument("--save_model", action="store_true",
                   help="Also save final model weights (~95 MB per run).")
    p.add_argument("--list", action="store_true",
                   help="Print the 21 partitions with their indices and exit.")
    return p.parse_args()


def resolve_partitions(args: argparse.Namespace) -> list[tuple[str, ...]]:
    """Turn the CLI selection flags into a concrete list of partitions."""
    all_configs = enumerate_configs(RESNET50_BLOCKS)

    if args.all:
        return all_configs
    if args.smoke:
        return [parse_label(label) for label in SMOKE_PARTITIONS]
    if args.index is not None:
        if not 0 <= args.index < len(all_configs):
            raise SystemExit(
                f"--index must be in [0, {len(all_configs) - 1}]; got {args.index}. "
                f"Use --list to see the mapping."
            )
        return [all_configs[args.index]]
    if args.partition:
        return [parse_label(args.partition)]
    return [tuple(CONFIG["partition"])]


def build_config(args: argparse.Namespace, partition: tuple[str, ...]) -> dict:
    """Apply CLI overrides on top of the archived default protocol."""
    cfg = dict(CONFIG)
    cfg["partition"] = partition
    cfg["epochs"]    = args.max_epochs

    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.dataset_mode:
        cfg["dataset_mode"] = args.dataset_mode
    if args.seed is not None:
        cfg["seed"] = args.seed

    if args.smoke:
        # Small and fast: this checks that the pipeline runs, not that a
        # partition is any good. Two epochs on 50k images proves nothing about
        # transfer, and the results are not comparable to the archived sweep.
        cfg["dataset_mode"] = args.dataset_mode or "mini"
        cfg["epochs"]       = 2

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Mini-dataset runs go to a separate directory. Their result files would
        # otherwise land next to the archived full-dataset sweep, where the
        # analysis layer globs everything in sight -- a two-epoch smoke test
        # would silently pollute, and could overwrite, the real leaderboard.
        out_dir = Path(cfg["log_dir"])
        if cfg["dataset_mode"] != "full":
            out_dir = out_dir.parent / f"runs_{cfg['dataset_mode']}"

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["log_dir"] = str(out_dir)
    return cfg


def run_one(cfg: dict, args: argparse.Namespace, device: torch.device) -> Path:
    """Train and evaluate one partition; write its result JSON. Returns the path."""
    partition = tuple(cfg["partition"])
    label     = config_label(partition)
    out_dir   = Path(cfg["log_dir"])
    gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    print("=" * 68)
    print(f"  iWildCam F/T/S transfer  --  {label}   {partition}")
    print(f"  dataset : {cfg['dataset_mode']}  |  batch {cfg['batch_size']}  "
          f"|  workers {cfg['num_workers']}")
    print(f"  budget  : up to {cfg['epochs']} epochs OR "
          f"{args.max_time_min:.0f} min, whichever first")
    print(f"  device  : {device}  ({gpu_name})")
    print(f"  data    : {cfg['data_dir']}")
    print("=" * 68)

    wall_start = time.perf_counter()

    print("\n[1/3] Building model ...")
    model, n_train, n_total = build_model(partition, cfg)
    flops = estimate_flops(model, cfg["image_size"], n_train, n_total)
    print(f"  Trainable params : {n_train:,} / {n_total:,}  "
          f"({100 * n_train / n_total:.1f}%)")
    print(f"  Inference FLOPs  : {flops['inference_flops'] / 1e9:.2f} G")
    print(f"  Training FLOPs   : {flops['training_flops_per_sample'] / 1e9:.2f} G / sample")

    print("\n[2/3] Training ...")
    evaluator = Evaluator(cfg, device)
    tracker   = EpochTimeBudgetTracker(
        max_epochs=cfg["epochs"],
        max_time_s=args.max_time_min * 60,
    )

    # Test splits are never touched: model selection and reporting both use the
    # validation splits, so the test sets stay clean for any future work.
    eval_log, final_results = train(
        model, cfg, evaluator, tracker,
        eval_splits=["id_val", "ood_val"],
        final_splits=["train", "id_val", "ood_val"],
        checkpoint_dir=None,
    )

    snapshot         = tracker.get_snapshot()
    train_elapsed    = snapshot["elapsed_s"]
    epochs_completed = snapshot["epoch"]
    secs_per_epoch   = train_elapsed / max(epochs_completed, 1e-6)

    print("\n[3/3] Saving results ...")
    if args.save_model:
        model_path = out_dir / f"model_{label}_final.pt"
        torch.save(model.state_dict(), model_path)
        print(f"  Model  -> {model_path}  ({model_path.stat().st_size / 1e6:.0f} MB)")

    wall_elapsed = time.perf_counter() - wall_start
    print(f"\n  Training elapsed : {train_elapsed / 60:.1f} min")
    print(f"  Total wall time  : {wall_elapsed / 60:.1f} min")
    print(f"  Epochs completed : {epochs_completed:.2f}")
    print(f"  Time per epoch   : {secs_per_epoch / 60:.2f} min")

    if epochs_completed < cfg["epochs"]:
        print(
            f"  NOTE: stopped on the {args.max_time_min:.0f} min wall-clock ceiling, "
            f"not the epoch ceiling. Slow partitions complete fewer epochs than "
            f"fast ones -- account for this when comparing runs."
        )

    output = {
        "label":     label,
        "gpu_name":  gpu_name,
        "config":    jsonify(cfg),
        "n_trainable": n_train,
        "n_total":     n_total,
        "inference_flops":           flops["inference_flops"],
        "training_flops_per_sample": flops["training_flops_per_sample"],
        "epochs_completed": epochs_completed,
        "train_elapsed_s":  train_elapsed,
        "wall_elapsed_s":   wall_elapsed,
        "secs_per_epoch":   secs_per_epoch,
        "epochs_per_hour":  epochs_completed / max(train_elapsed / 3600, 1e-9),
        "eval_log":         jsonify(eval_log),
        "final_results":    jsonify(final_results),
    }

    # The suffix names the dataset mode, matching the archived files
    # (timing_T5S1_full.json). A mini run is therefore never confusable with a
    # full one, even if both end up in the same directory.
    out_path = out_dir / f"timing_{label}_{cfg['dataset_mode']}.json"
    if out_path.exists():
        print(f"  NOTE: overwriting existing {out_path.name}")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"  Results -> {out_path}")
    return out_path


def main() -> None:
    args = parse_args()

    if args.list:
        print("idx  label     partition")
        for i, partition in enumerate(enumerate_configs(RESNET50_BLOCKS)):
            print(f"{i:>3}  {config_label(partition):<9} {partition}")
        return

    partitions = resolve_partitions(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        print(
            "WARNING: no CUDA device found. A full-dataset run takes roughly "
            "3 GPU-hours; on CPU it is not practical. Use --smoke to check the "
            "pipeline instead.\n"
        )

    if args.smoke:
        print(
            "SMOKE TEST -- 2 epochs on the mini dataset. This verifies the "
            "pipeline end to end. The numbers are NOT comparable to the archived "
            "sweep and should not be read as results.\n"
        )

    written = []
    for i, partition in enumerate(partitions, start=1):
        if len(partitions) > 1:
            print(f"\n\n########  Run {i}/{len(partitions)}  ########")
        cfg = build_config(args, partition)
        written.append(run_one(cfg, args, device))

    print(f"\nDone. {len(written)} run(s) written.")
    if len(written) > 1:
        print("Rebuild the leaderboard with: python scripts/make_leaderboard.py")


if __name__ == "__main__":
    main()
