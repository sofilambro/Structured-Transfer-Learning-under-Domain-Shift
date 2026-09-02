"""
Evaluation for the controlled TinyImageNet study (paper, Experiment 1).

Two numbers per run:

``clean``    top-1 accuracy on the target subset's 5,000 clean validation images.
             The paper reports the *best* validation top-1 seen during training,
             not the last -- see :func:`BestTracker`.
``corrupt``  top-1 accuracy on the TinyImageNet-C-style corrupted validation set.

Unlike iWildCam, the class distribution here is uniform by construction (500
train / 50 val images per class), so plain top-1 accuracy is the right metric and
balanced accuracy would be identical up to sampling noise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> dict:
    """
    Top-1 accuracy and mean cross-entropy over a loader.

    Returns ``{"acc", "loss", "n"}``.
    """
    device = torch.device(device)
    model.eval()
    model.to(device)

    n_correct  = 0
    n_total    = 0
    total_loss = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        n_correct  += (logits.argmax(dim=1) == labels).sum().item()
        n_total    += labels.numel()

    return {
        "acc":  n_correct / max(n_total, 1),
        "loss": total_loss / max(n_total, 1),
        "n":    n_total,
    }


@torch.no_grad()
def evaluate_per_corruption(
    model: torch.nn.Module,
    dataset,
    device: torch.device | str,
    batch_size: int = 128,
    num_workers: int = 4,
) -> dict[str, dict]:
    """
    Break corruption accuracy down by corruption type and severity.

    Uses the manifest carried on :class:`~.data.CorruptedValSet`, so no second
    pass over the images is needed. Not reported in the paper -- the aggregate is
    what Tables 4 and 5 use -- but it is the natural next question once a
    configuration looks unusually fragile.

    Returns ``{corruption_name: {"acc", "n", "by_severity": {severity: acc}}}``.
    """
    device = torch.device(device)
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    correct: list[bool] = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct.extend((preds == labels).cpu().tolist())

    # dataset.meta rows are in the same order as dataset.samples, which the
    # loader iterates with shuffle=False.
    results: dict[str, dict] = {}
    for row, is_correct in zip(dataset.meta, correct):
        name     = row["corruption"]
        severity = int(row["severity"])
        entry = results.setdefault(name, {"hits": 0, "n": 0, "_sev": {}})
        entry["hits"] += int(is_correct)
        entry["n"]    += 1
        sev = entry["_sev"].setdefault(severity, {"hits": 0, "n": 0})
        sev["hits"] += int(is_correct)
        sev["n"]    += 1

    return {
        name: {
            "acc": entry["hits"] / max(entry["n"], 1),
            "n":   entry["n"],
            "by_severity": {
                s: v["hits"] / max(v["n"], 1)
                for s, v in sorted(entry["_sev"].items())
            },
        }
        for name, entry in results.items()
    }


class BestTracker:
    """
    Remembers the best validation accuracy seen so far, and its epoch.

    Tables 4 and 5 report "best validation top-1", not the final-epoch value.
    That choice matters for this experiment specifically: the unstable
    configurations (a large scratch ``group3`` above a fine-tuned ``group2``)
    oscillate, so a last-epoch reading would confound instability with a bad
    final sample. Reporting the best value isolates the actual ceiling each
    partition reaches.
    """

    def __init__(self):
        self.best_acc: float = 0.0
        self.best_epoch: int = -1

    def update(self, acc: float, epoch: int) -> bool:
        """Record ``acc``; return True if it is a new best."""
        if acc > self.best_acc:
            self.best_acc   = acc
            self.best_epoch = epoch
            return True
        return False
