"""
Default configuration for the iWildCam transfer study (paper, Experiment 2).

These values are the ones the archived 21-configuration sweep ran under; they are
reproduced in Appendix A.2 of the paper. Do not change them casually -- every run
in ``results/iwildcam/runs/`` shares this protocol, and the comparability of the
leaderboard depends on it. Override per-run through the CLI instead
(``scripts/iwildcam_run.py --data_dir ... --max_epochs ...``).
"""

from __future__ import annotations

import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_PACKAGE_DIR, "..", "..", ".."))

CONFIG = {
    # ── Data ─────────────────────────────────────────────────────────────────
    "data_dir":     os.path.join(_REPO_ROOT, "data"),
    "dataset":      "iwildcam",
    # "mini" is the ~50k-image subset built by scripts/iwildcam_build_mini_dataset.py,
    # for laptop-scale iteration. "full" is the 203k-image WILDS release and is
    # what every archived result used.
    "dataset_mode": "full",
    "num_classes":  182,       # max(y)+1; identical for mini and full
    "image_size":   224,

    # ── Model ────────────────────────────────────────────────────────────────
    "backbone":     "resnet50",

    # F/T/S partition over (stem, stage1, stage2, stage3, stage4, head).
    # The head must always be 'S'. Default is full fine-tuning, T5S1, which is
    # the paper's best OOD configuration.
    # Use structured_transfer.partitions.enumerate_configs() for all 21.
    "partition":    ("T", "T", "T", "T", "T", "S"),

    # ── Training ─────────────────────────────────────────────────────────────
    "batch_size":   32,
    # Ceiling only. Runs are time-budgeted (see EpochTimeBudgetTracker), and in
    # practice stopped at roughly 40 epochs.
    "epochs":       100,
    "seed":         42,
    "num_workers":  4,
    "device":       "cuda",    # falls back to CPU in train.py

    # ── Optimizer ────────────────────────────────────────────────────────────
    # AdamW. The head gets a 10x larger LR than the backbone: it is randomly
    # initialized and must cover a new 182-class label space, while T blocks only
    # need to drift from their ImageNet solution.
    "lr":               1e-4,   # trainable backbone blocks (T and S)
    "lr_head":          1e-3,   # classifier head (always S)
    "weight_decay":     1e-3,
    "scheduler":        "cosine",   # cosine | step | none

    # ── Regularization ───────────────────────────────────────────────────────
    # Added after the first full-dataset pass overfit the head badly. Both are
    # part of the archived protocol.
    "label_smoothing":  0.1,    # 0.0 disables
    "head_dropout":     0.3,    # 0.0 disables

    # ── Frozen-block BatchNorm ───────────────────────────────────────────────
    # When True, BatchNorm layers inside F blocks are held in eval() mode so
    # their running statistics keep the source (ImageNet) values -- the strict
    # reading of "frozen", and what the TinyImageNet experiment does.
    # Kept False here because the archived 21-run sweep did NOT do this; turning
    # it on changes the protocol, so new runs would not be directly comparable to
    # results/iwildcam/runs/.
    "freeze_frozen_bn": False,

    # ── Class imbalance ──────────────────────────────────────────────────────
    # iWildCam is long-tailed (the rarest classes have a handful of images), so
    # training draws with inverse-frequency weights. Evaluation is never
    # reweighted -- macro metrics handle imbalance on that side.
    "use_weighted_sampler": True,

    # ── Logging ──────────────────────────────────────────────────────────────
    "save_dir":     os.path.join(_REPO_ROOT, "checkpoints"),
    "log_dir":      os.path.join(_REPO_ROOT, "results", "iwildcam", "runs"),
    "project_name": "structured-transfer-iwildcam",
}
