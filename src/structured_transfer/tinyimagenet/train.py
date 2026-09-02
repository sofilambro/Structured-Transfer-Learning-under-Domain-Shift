"""
Training loop for the controlled TinyImageNet study (paper, Experiment 1).

Protocol (Appendix A.1), identical for all 58 models:

    100 epochs, SGD momentum 0.9, weight decay 5e-4, cosine schedule, batch 128
    lr 0.1   for scratch backbone blocks and the classifier head
    lr 0.01  for copied, fine-tuned blocks
    frozen blocks excluded from the optimizer, their BatchNorm held in eval mode

Fixing epochs rather than wall clock is what makes this the *controlled* half of
the study: no configuration is truncated for being slow, so a difference in the
results reflects a difference in what the partition can learn.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..utils import set_seed
from .evaluate import BestTracker, evaluate
from .models import freeze_frozen_batchnorm, get_param_groups


def train(
    model: nn.Module,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    corrupt_loader: DataLoader | None = None,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """
    Train one F/T/S configuration on one target subset.

    Args:
        model:          built by :func:`.models.build_model`
        config:         project config; must contain ``partition``
        train_loader:   target subset, train split
        val_loader:     target subset, clean validation split
        corrupt_loader: corrupted validation split; evaluated once at the end,
                        since building it is the expensive part and it is a
                        stress test rather than a model-selection signal
        device:         defaults to CUDA when available

    Returns:
        ``(epoch_log, final_results)``. ``final_results`` carries ``clean_acc``
        (the *best* validation top-1, per Tables 4 and 5), ``final_acc`` (the
        last epoch), ``best_epoch``, and ``corrupt_acc`` when a corrupted loader
        was supplied.
    """
    set_seed(config["seed"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model.to(device)

    partition = config["partition"]
    epochs    = config["epochs"]

    optimizer = SGD(
        get_param_groups(model, partition, config),
        momentum     = config.get("momentum", 0.9),
        weight_decay = config.get("weight_decay", 5e-4),
        nesterov     = config.get("nesterov", False),
    )
    # Per-step cosine annealing over the full nominal horizon. Both parameter
    # groups anneal proportionally, so the 10x gap between the scratch and
    # fine-tune learning rates is preserved for the whole run.
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
        if config.get("scheduler", "cosine") == "cosine"
        else None
    )
    criterion = nn.CrossEntropyLoss()
    freeze_bn = config.get("freeze_frozen_bn", True)

    epoch_log: list[dict] = []
    best = BestTracker()

    for epoch in range(epochs):
        model.train()
        if freeze_bn:
            # model.train() just reset every submodule, including the frozen
            # blocks' BatchNorm. Re-freeze before the first batch of the epoch.
            freeze_frozen_batchnorm(model, partition)

        epoch_loss  = 0.0
        n_correct   = 0
        n_total     = 0
        epoch_start = time.perf_counter()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item() * labels.numel()
            n_correct  += (logits.argmax(dim=1) == labels).sum().item()
            n_total    += labels.numel()

        val = evaluate(model, val_loader, device)
        is_best = best.update(val["acc"], epoch)

        entry = {
            "epoch":      epoch + 1,
            "train_loss": epoch_loss / max(n_total, 1),
            "train_acc":  n_correct / max(n_total, 1),
            "val_acc":    val["acc"],
            "val_loss":   val["loss"],
            "lr":         optimizer.param_groups[0]["lr"],
            "epoch_time_s": time.perf_counter() - epoch_start,
        }
        epoch_log.append(entry)

        if verbose:
            print(
                f"Epoch {epoch + 1:3d}/{epochs}  "
                f"train_loss={entry['train_loss']:.4f}  "
                f"train_acc={entry['train_acc']:.3f}  "
                f"val_acc={entry['val_acc']:.4f}"
                f"{'  *best' if is_best else ''}"
            )

    final_results = {
        # Tables 4 and 5 report best validation top-1, not the final epoch.
        "clean_acc":  best.best_acc,
        "final_acc":  epoch_log[-1]["val_acc"] if epoch_log else float("nan"),
        "best_epoch": best.best_epoch + 1,
    }

    if corrupt_loader is not None:
        if verbose:
            print("\nEvaluating TinyImageNet-C-style corrupted validation set ...")
        corrupt = evaluate(model, corrupt_loader, device)
        final_results["corrupt_acc"] = corrupt["acc"]
        final_results["corrupt_n"]   = corrupt["n"]

    if verbose:
        print(
            f"\nBest clean val top-1: {final_results['clean_acc']:.4f} "
            f"(epoch {final_results['best_epoch']})"
            + (f"   corrupt top-1: {final_results['corrupt_acc']:.4f}"
               if "corrupt_acc" in final_results else "")
        )

    return epoch_log, final_results
