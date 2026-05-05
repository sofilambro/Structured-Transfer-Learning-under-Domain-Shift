import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (n_trainable, n_total) parameter counts."""
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

    Uses fvcore.nn.FlopCountAnalysis if available; falls back to the known
    ResNet50-on-224×224 constant (~8.2 GFLOPs = 2× 4.1 GMACs).

    Training FLOPs approximation:
        forward (same for all configs) + backward (only through trainable layers)
        backward ≈ 2× forward for the trainable portion of the network.
        total ≈ forward × (1 + 2 × trainable_ratio)

    Returns:
        {
            "inference_flops":           int,   # FLOPs for one forward pass
            "training_flops_per_sample": int,   # approx FLOPs for one fwd+bwd
        }
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
        # ResNet50 224×224: ~4.1 GMACs → ~8.2 GFLOPs
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
    """Load checkpoint and return the saved epoch number."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("epoch", 0)
