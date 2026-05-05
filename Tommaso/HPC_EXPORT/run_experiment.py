#!/usr/bin/env python3
"""
HPC timing benchmark: T5S1 full fine-tuning on iWildCam (full dataset).

Pipeline: build model → train (per-epoch eval, time ceiling) → final eval → save results + model.

Usage:
    python run_experiment.py
    python run_experiment.py --data_dir /path/to/data --max_epochs 20 --max_time_min 45
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "iwildcam"))

import torch
from config import CONFIG
from models import build_model, config_label
from evaluate import Evaluator
from budget import EpochTimeBudgetTracker
from train import train
from utils import estimate_flops


def parse_args():
    p = argparse.ArgumentParser(description="iWildCam T5S1 timing benchmark")
    p.add_argument("--data_dir",     default=None,  help="Override data directory")
    p.add_argument("--max_epochs",   type=int,   default=50,   help="Epoch ceiling (default: 50)")
    p.add_argument("--max_time_min", type=float, default=45.0, help="Wall-clock training ceiling in minutes (default: 45)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = dict(CONFIG)
    cfg["dataset_mode"] = "full"
    cfg["partition"]    = ("T", "T", "T", "T", "T", "S")   # T5S1
    cfg["epochs"]       = args.max_epochs
    cfg["use_wandb"]    = False

    if args.data_dir:
        cfg["data_dir"] = args.data_dir

    label = config_label(cfg["partition"])   # "T5S1"

    script_dir      = Path(__file__).parent
    cfg["save_dir"] = str(script_dir / "checkpoints")
    cfg["log_dir"]  = str(script_dir / "results")
    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print(f"  iWildCam timing benchmark — {label} (full fine-tune)")
    print(f"  dataset : full  |  batch: {cfg['batch_size']}  |  workers: {cfg['num_workers']}")
    print(f"  budget  : up to {args.max_epochs} epochs  OR  {args.max_time_min:.0f} min (whichever first)")
    print(f"  device  : {device}  ({gpu_name})")
    print(f"  data    : {cfg['data_dir']}")
    print("=" * 65)

    wall_start = time.perf_counter()

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n[1/3] Building model …")
    model, n_train, n_total = build_model(cfg["partition"], cfg)
    flops = estimate_flops(model, cfg["image_size"], n_train, n_total)
    print(f"  Trainable params : {n_train:,} / {n_total:,}  ({100*n_train/n_total:.1f}%)")
    print(f"  Inference FLOPs  : {flops['inference_flops']  / 1e9:.2f} GFLOPs")
    print(f"  Training FLOPs/s : {flops['training_flops_per_sample'] / 1e9:.2f} GFLOPs")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[2/3] Training …")
    evaluator = Evaluator(cfg, device)
    tracker   = EpochTimeBudgetTracker(
        max_epochs = args.max_epochs,
        max_time_s = args.max_time_min * 60,
    )

    # Eval splits: id_val and ood_val after every epoch; no test sets used.
    eval_splits  = ["id_val", "ood_val"]
    final_splits = ["train", "id_val", "ood_val"]

    eval_log, final_results = train(
        model, cfg, evaluator, tracker,
        eval_splits   = eval_splits,
        final_splits  = final_splits,
        checkpoint_dir = None,   # no per-epoch checkpoints; final model saved below
    )

    train_elapsed    = tracker.get_snapshot()["elapsed_s"]
    epochs_completed = tracker.get_snapshot()["epoch"]
    secs_per_epoch   = train_elapsed / max(epochs_completed, 1e-6)

    # ── Save model weights ────────────────────────────────────────────────────
    print("\n[3/3] Saving results …")
    model_path = Path(cfg["log_dir"]) / f"model_{label}_final.pt"
    torch.save(model.state_dict(), model_path)
    model_mb = model_path.stat().st_size / 1e6
    print(f"  Model  → {model_path}  ({model_mb:.0f} MB)")

    wall_elapsed = time.perf_counter() - wall_start

    print(f"\n  Training elapsed : {train_elapsed/60:.1f} min")
    print(f"  Total wall time  : {wall_elapsed/60:.1f} min")
    print(f"  Epochs completed : {epochs_completed:.2f}")
    print(f"  Time per epoch   : {secs_per_epoch/60:.2f} min")
    print(f"  Throughput       : {epochs_completed / max(train_elapsed/3600, 1e-9):.1f} epochs/hour")

    # ── Save JSON results ─────────────────────────────────────────────────────
    def _jsonify(v):
        if isinstance(v, (int, float, bool, str, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return [_jsonify(x) for x in v]
        if isinstance(v, dict):
            return {k: _jsonify(vv) for k, vv in v.items()}
        return str(v)

    output = {
        "label"        : label,
        "gpu_name"     : gpu_name,
        "config"       : _jsonify(cfg),
        "n_trainable"  : n_train,
        "n_total"      : n_total,
        "inference_flops"           : flops["inference_flops"],
        "training_flops_per_sample" : flops["training_flops_per_sample"],
        "epochs_completed"  : epochs_completed,
        "train_elapsed_s"   : train_elapsed,
        "wall_elapsed_s"    : wall_elapsed,
        "secs_per_epoch"    : secs_per_epoch,
        "epochs_per_hour"   : epochs_completed / max(train_elapsed / 3600, 1e-9),
        "eval_log"          : _jsonify(eval_log),      # one entry per epoch: id_val + ood_val metrics
        "final_results"     : _jsonify(final_results), # id_val + ood_val after training ends
    }

    out_path = Path(cfg["log_dir"]) / f"timing_{label}_full.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results → {out_path}")
    print(f"\nDone. Total wall time: {wall_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
