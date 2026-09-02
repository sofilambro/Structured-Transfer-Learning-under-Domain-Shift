"""
Default configuration for the controlled TinyImageNet study (paper, Experiment 1).

Values follow Appendix A.1. Unlike Experiment 2 this protocol is fixed by
*epochs*, not wall clock: every configuration gets exactly 100 epochs, so runs
are directly comparable without a time ceiling truncating the slow ones.
"""

from __future__ import annotations

import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_PACKAGE_DIR, "..", "..", ".."))

CONFIG = {
    # ── Data ─────────────────────────────────────────────────────────────────
    "data_dir":    os.path.join(_REPO_ROOT, "data"),
    "dataset":     "tinyimagenet",
    "image_size":  64,
    # Each half of the 200-class pool. 500 train / 50 val images per class, so a
    # subset holds 50,000 train and 5,000 val images.
    "num_classes": 100,
    "split_seed":  1,          # "split 1" in the paper

    # ── Model ────────────────────────────────────────────────────────────────
    "backbone":     "wrn-28-10",
    "depth":        28,
    "widen_factor": 10,

    # F/T/S partition over (conv1, group1, group2, group3, head).
    # The head must always be 'S'. 15 valid partitions; SSSS(+S head) is the
    # source-free all-scratch baseline, i.e. the base networks themselves.
    "partition":    ("S", "S", "S", "S", "S"),

    # ── Training ─────────────────────────────────────────────────────────────
    "epochs":       100,
    "batch_size":   128,
    "seed":         42,
    "num_workers":  4,
    "device":       "cuda",

    # ── Optimizer ────────────────────────────────────────────────────────────
    # SGD with momentum, cosine schedule. Two learning rates split by
    # initialization, not by depth: scratch blocks and the head take large steps,
    # copied fine-tuned blocks take small ones.
    "optimizer":    "sgd",
    "lr_scratch":   0.1,
    "lr_finetune":  0.01,
    "momentum":     0.9,
    "weight_decay": 5e-4,
    "scheduler":    "cosine",
    "nesterov":     False,

    # ── Frozen-block BatchNorm ───────────────────────────────────────────────
    # On, per Appendix A.1: frozen blocks keep source-domain running statistics.
    # Turning this off makes "frozen" mean only "weights held fixed" and lets the
    # block's outputs drift with the target data.
    "freeze_frozen_bn": True,

    # ── Corruption evaluation ────────────────────────────────────────────────
    # TinyImageNet-C-style synthetic stress test. A secondary robustness probe,
    # not a substitute for the real location shift measured in Experiment 2.
    "corruption_seed":  0,
    "corruption_split": "val",

    # ── Logging ──────────────────────────────────────────────────────────────
    "save_dir": os.path.join(_REPO_ROOT, "checkpoints", "tinyimagenet"),
    "log_dir":  os.path.join(_REPO_ROOT, "results", "tinyimagenet", "runs"),
}

#: The four source/target directions of the transfer protocol (Appendix A.1).
#: A->A and B->B are the "selfer" controls, where source and target label spaces
#: match, so any degradation is co-adaptation or optimization rather than domain
#: mismatch. A->B and B->A are the cross-domain "transfer" runs, which
#: additionally expose feature specialization.
DIRECTIONS: tuple[str, ...] = ("AtoA", "BtoB", "AtoB", "BtoA")
