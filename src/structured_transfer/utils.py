"""
Shared utilities: seeding, parameter accounting, FLOP estimation, checkpointing.

Used by both experiments. Nothing here is experiment-specific.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy and Torch, and put cuDNN in deterministic mode.

    Determinism costs throughput (``benchmark=False`` disables kernel
    autotuning), which is a deliberate trade: the study compares partitions to
    each other, so run-to-run reproducibility matters more than raw speed.

    Note this does not make results *bit*-reproducible across GPU models --
    the archived runs span A100 and other cards.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Return ``(n_trainable, n_total)``.

    ``n_trainable / n_total`` is the ``Train.`` column of the paper's Table 3.
    Because ResNet50's Stage4 alone holds 62.7% of the parameters, this ratio is
    a poor proxy for actual cost -- earlier stages are cheap in parameters but
    expensive in FLOPs, since they run at larger spatial resolutions. Use
    :func:`estimate_flops` when reasoning about compute.
    """
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    return n_trainable, n_total


def estimate_flops(
    model: torch.nn.Module,
    image_size: int = 224,
    n_trainable: int | None = None,
    n_total: int | None = None,
) -> dict:
    """
    Estimate inference and per-sample training FLOPs.

    Inference cost comes from ``fvcore.nn.FlopCountAnalysis`` when available; it
    is identical across partitions, since freezing changes the backward pass, not
    the forward one. On ResNet50 at 224x224 this returns ~4.11 G, matching the
    paper's Table 2 total of 4.1098 G. Without fvcore we fall back to a published
    constant, which is coarser -- install fvcore for the numbers in the paper.

    Training cost uses the standard approximation that a backward pass costs
    roughly twice a forward pass, but only through the *trainable* portion:

        total ~= forward x (1 + 2 x trainable_ratio)

    This is what produces the paper's efficiency claim. Fine-tuning only Stage4
    (F4T1S1) has ``trainable_ratio = 0.642``, giving ``1 + 2(0.642) = 2.284``
    against ``3.0`` for full fine-tuning -- i.e. **76%** of the full fine-tuning
    FLOPs, which is the figure quoted in Section 4.1 and Appendix D.2. The
    archived run files confirm it: 9.39 G against 12.33 G.

    The approximation is deliberately crude. It attributes cost by parameter
    share rather than by per-layer activation size, so it understates the cost of
    early stages. Wall-clock time is logged alongside it for exactly that reason.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.zeros(1, 3, image_size, image_size)
        with torch.no_grad():
            fa = FlopCountAnalysis(model.eval(), dummy)
            fa.unsupported_ops_warnings(False)
            fa.uncalled_modules_warnings(False)
            inference_flops = int(fa.total())
    except Exception:
        # ResNet50 @ 224x224: ~4.1 GMACs. Only a fallback -- prefer fvcore.
        inference_flops = 8_200_000_000

    if n_trainable is None or n_total is None:
        n_trainable, n_total = count_parameters(model)

    trainable_ratio = n_trainable / max(n_total, 1)
    training_flops_per_sample = int(inference_flops * (1.0 + 2.0 * trainable_ratio))

    return {
        "inference_flops":           inference_flops,
        "training_flops_per_sample": training_flops_per_sample,
    }


class Timer:
    """Lightweight wall-clock timer, usable standalone or as a context manager."""

    def __init__(self):
        self._start: float | None = None
        self.elapsed_s: float = 0.0

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start

    def reset(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.elapsed_s = self.elapsed()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str | Path,
    extra: dict | None = None,
) -> None:
    """Save model + optimizer state, plus any extra scalars (step, metrics, ...)."""
    state = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load a checkpoint in place and return the saved epoch number."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("epoch", 0)


def jsonify(value):
    """
    Coerce a value into something ``json.dump`` accepts.

    Run configs contain ``Path`` objects and partition tuples; this flattens them
    without losing information, so the archived result schema stays stable.
    """
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    return str(value)
