#!/usr/bin/env python3
"""
Run one F/T/S configuration in one source/target direction (paper, Experiment 1).

The full protocol is 58 models: two source networks trained from scratch (the
all-scratch ``SSSS`` baseline on subset A and on subset B), then each of the 14
source-dependent partitions trained in all four directions
``A->A, B->B, A->B, B->A``.

The direction matters more than it looks. ``A->A`` and ``B->B`` are the *selfer*
controls: source and target label spaces are identical, so any degradation is
optimization or co-adaptation, never domain mismatch. ``A->B`` and ``B->A`` are
the cross-domain runs, which add feature specialization on top. Comparing the two
is what separates the effects.

Per-run outputs are written to ``results/tinyimagenet/runs/``; the reference
tables in that directory are read-only inputs to the analysis layer.

Examples
--------
    # The two source networks (must exist before any transfer run)
    python scripts/tinyimagenet_run.py --config SSSS --direction AtoA
    python scripts/tinyimagenet_run.py --config SSSS --direction BtoB

    # One transfer run
    python scripts/tinyimagenet_run.py --config FFTS --direction AtoB

    # Everything (long: 56 runs x 100 epochs)
    python scripts/tinyimagenet_run.py --all

    # Cluster array task
    python scripts/tinyimagenet_run.py --index $SLURM_ARRAY_TASK_ID
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from structured_transfer.partitions import (
    WRN2810_BLOCKS,
    enumerate_configs,
    parse_label,
    validate_partition,
)
from structured_transfer.tinyimagenet.config import CONFIG, DIRECTIONS
from structured_transfer.tinyimagenet.data import (
    CorruptedValSet,
    TinyImageNetSubset,
    get_dataloader,
    get_transforms,
    load_subset_wnids,
)
from structured_transfer.tinyimagenet.models import build_model
from structured_transfer.tinyimagenet.train import train
from structured_transfer.utils import jsonify

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The all-scratch baseline. Source-independent, so it is trained once per
#: subset rather than once per direction.
BASELINE_LABEL = "SSSS"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TinyImageNet controlled F/T/S transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--config", metavar="LABEL",
                           help="4-letter backbone partition, e.g. FFTS, TTTS, FFFF.")
    selection.add_argument("--all", action="store_true",
                           help="Run the complete 58-model protocol sequentially.")
    selection.add_argument("--index", type=int, metavar="N",
                           help="Run job N of the protocol (0-based). For SLURM arrays.")

    p.add_argument("--direction", choices=DIRECTIONS, default="AtoA",
                   help="Source->target direction (default: AtoA).")
    p.add_argument("--data_dir", default=str(_REPO_ROOT / "data"))
    p.add_argument("--out_dir", default=None,
                   help="Result JSON destination. Default: results/tinyimagenet/runs.")
    p.add_argument("--ckpt_dir", default=None,
                   help="Where source checkpoints live. Default: checkpoints/tinyimagenet.")
    p.add_argument("--max_epochs", type=int, default=CONFIG["epochs"],
                   help=f"Epochs (default: {CONFIG['epochs']}, the paper's protocol).")
    p.add_argument("--split_seed", type=int, default=CONFIG["split_seed"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--no-corrupt", action="store_true",
                   help="Skip corrupted-set evaluation (useful for a quick check).")
    p.add_argument("--list", action="store_true",
                   help="Print the protocol's job list with indices and exit.")
    return p.parse_args()


def build_protocol() -> list[tuple[str, str]]:
    """
    The full job list as ``(backbone_label, direction)`` pairs.

    Two baselines plus 14 partitions x 4 directions = 58 jobs, matching the
    paper's count. ``SSSS`` copies nothing from a source network, so running it
    in all four directions would train the same model four times; it appears
    once per target subset instead.

    Labels are the 4-letter backbone form (``FFTS``) used by Tables 4 and 5, not
    the counted form ``config_label`` produces (``F2T1S2``) -- the two must not
    be mixed here. ``config_label`` of the all-scratch partition is ``S5``, which
    never equals ``SSSS``, so mixing them would schedule the baseline four times
    and yield 62 jobs instead of 58.
    """
    jobs: list[tuple[str, str]] = [(BASELINE_LABEL, "AtoA"), (BASELINE_LABEL, "BtoB")]
    for partition in enumerate_configs(WRN2810_BLOCKS):
        label = "".join(partition[:-1])      # drop the always-scratch head slot
        if label == BASELINE_LABEL:
            continue
        for direction in DIRECTIONS:
            jobs.append((label, direction))
    return jobs


def backbone_label(label: str) -> str:
    """
    Accept both the 4-letter backbone form and the head-inclusive form.

    Table 4 and 5 name configurations by backbone only (``FFTS``), while
    :func:`config_label` counts the head (``F2T1S2``). Users type the former.
    """
    partition = _partition_from_label(label)
    return "".join(partition[:-1])


def _partition_from_label(label: str) -> tuple[str, ...]:
    """Parse either ``FFTS`` (letters) or ``F2T1S2`` (counted) into a partition."""
    text = label.strip().upper()
    if any(ch.isdigit() for ch in text):
        return parse_label(text, WRN2810_BLOCKS)
    if len(text) != len(WRN2810_BLOCKS) - 1:
        raise SystemExit(
            f"Backbone label {label!r} must have {len(WRN2810_BLOCKS) - 1} letters "
            f"(one per block: conv1, group1, group2, group3), e.g. 'FFTS'."
        )
    partition = tuple(text) + ("S",)   # the head is always scratch
    validate_partition(partition, WRN2810_BLOCKS)
    return partition


def source_checkpoint_path(ckpt_dir: Path, subset: str, split_seed: int) -> Path:
    """Where the from-scratch source network for a subset is stored."""
    return ckpt_dir / f"base{subset}_split{split_seed}.pt"


def run_one(label: str, direction: str, args: argparse.Namespace,
            device: torch.device) -> Path:
    """Train and evaluate one (configuration, direction) job."""
    partition = _partition_from_label(label)
    bb_label  = backbone_label(label)
    source, target = direction[0], direction[-1]     # "AtoB" -> ("A", "B")

    data_dir    = Path(args.data_dir)
    root        = data_dir / "tiny-imagenet-200"
    splits_dir  = data_dir / "tinyimagenet_splits"
    corrupt_dir = data_dir / "tinyimagenet_corrupted" / f"split{args.split_seed}_{target}"

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(CONFIG["save_dir"])
    out_dir  = Path(args.out_dir) if args.out_dir else Path(CONFIG["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(CONFIG)
    cfg.update({
        "partition":  partition,
        "epochs":     args.max_epochs,
        "seed":       args.seed,
        "split_seed": args.split_seed,
        "data_dir":   str(data_dir),
    })

    print("=" * 68)
    print(f"  TinyImageNet F/T/S  --  {bb_label}   {direction}   {partition}")
    print(f"  source: subset {source}   target: subset {target}   "
          f"({'selfer' if source == target else 'transfer'})")
    print(f"  epochs: {cfg['epochs']}   batch: {cfg['batch_size']}   device: {device}")
    print("=" * 68)

    target_wnids = load_subset_wnids(splits_dir / f"split{args.split_seed}_{target}.txt")
    train_set = TinyImageNetSubset(root, target_wnids, "train",
                                   get_transforms("train", cfg["image_size"]))
    val_set   = TinyImageNetSubset(root, target_wnids, "val",
                                   get_transforms("val", cfg["image_size"]))
    print(f"\nTarget subset {target}: {len(train_set):,} train / {len(val_set):,} val "
          f"across {len(target_wnids)} classes")

    train_loader = get_dataloader(train_set, cfg, shuffle=True)
    val_loader   = get_dataloader(val_set, cfg, shuffle=False)

    corrupt_loader = None
    manifest = corrupt_dir / "manifest.csv"
    if not args.no_corrupt:
        if manifest.exists():
            corrupt_set = CorruptedValSet(manifest, get_transforms("val", cfg["image_size"]))
            corrupt_loader = get_dataloader(corrupt_set, cfg, shuffle=False)
            print(f"Corrupted validation set: {len(corrupt_set):,} images")
        else:
            print(f"No corrupted set at {manifest} -- skipping corruption evaluation. "
                  f"Build it with scripts/tinyimagenet_prepare.py --stage corrupt")

    # Source weights, needed by every partition that has an F or a T block.
    source_state = None
    if any(mode in ("F", "T") for mode in partition[:-1]):
        source_path = source_checkpoint_path(ckpt_dir, source, args.split_seed)
        if not source_path.exists():
            raise SystemExit(
                f"Source network not found at {source_path}.\n"
                f"Train it first:\n"
                f"  python scripts/tinyimagenet_run.py --config {BASELINE_LABEL} "
                f"--direction {source}to{source}"
            )
        source_state = torch.load(source_path, map_location="cpu")
        print(f"Loaded source weights from {source_path}")

    model, n_train, n_total = build_model(partition, cfg, source_state)
    print(f"Trainable: {n_train:,} / {n_total:,}  ({100 * n_train / n_total:.1f}%)\n")

    wall_start = time.perf_counter()
    epoch_log, final_results = train(model, cfg, train_loader, val_loader,
                                     corrupt_loader, device)
    wall_elapsed = time.perf_counter() - wall_start

    # An all-scratch run on subset X *is* the source network baseX; save it so
    # the transfer runs have something to copy from.
    if bb_label == BASELINE_LABEL and source == target:
        source_path = source_checkpoint_path(ckpt_dir, target, args.split_seed)
        torch.save(model.state_dict(), source_path)
        print(f"\nSaved source network -> {source_path}")

    output = {
        "label":       bb_label,
        "direction":   direction,
        "partition":   list(partition),
        "config":      jsonify(cfg),
        "n_trainable": n_train,
        "n_total":     n_total,
        "wall_elapsed_s": wall_elapsed,
        "epoch_log":     jsonify(epoch_log),
        "final_results": jsonify(final_results),
    }

    out_path = out_dir / f"{bb_label}_{direction}_split{args.split_seed}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Results -> {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    protocol = build_protocol()

    if args.list:
        print(f"{len(protocol)} jobs in the protocol:")
        print("idx  config  direction")
        for i, (label, direction) in enumerate(protocol):
            print(f"{i:>3}  {label:<7} {direction}")
        return

    if args.all:
        jobs = protocol
    elif args.index is not None:
        if not 0 <= args.index < len(protocol):
            raise SystemExit(
                f"--index must be in [0, {len(protocol) - 1}]; got {args.index}. "
                f"Use --list to see the mapping."
            )
        jobs = [protocol[args.index]]
    else:
        jobs = [(args.config or BASELINE_LABEL, args.direction)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA device. WRN-28-10 for 100 epochs on CPU is not "
              "practical; use --max_epochs 1 only to check the pipeline runs.\n")

    for i, (label, direction) in enumerate(jobs, start=1):
        if len(jobs) > 1:
            print(f"\n\n########  Job {i}/{len(jobs)}: {label} {direction}  ########")
        run_one(label, direction, args, device)

    print(f"\nDone. {len(jobs)} run(s) complete.")


if __name__ == "__main__":
    main()
