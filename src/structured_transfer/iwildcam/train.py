"""
Training loop for the iWildCam F/T/S experiments (paper, Experiment 2).

The loop itself is partition-agnostic: which blocks train and at what learning
rate is decided in :func:`models.build_model` and :func:`models.get_param_groups`,
and when to evaluate or stop is decided by the injected
:class:`~structured_transfer.budget.BudgetTracker`. That separation is what lets
all 21 configurations run under one identical protocol.

Typical use -- see ``scripts/iwildcam_run.py`` for the full CLI:

    from structured_transfer.iwildcam.config   import CONFIG
    from structured_transfer.iwildcam.models   import build_model
    from structured_transfer.iwildcam.evaluate import Evaluator
    from structured_transfer.iwildcam.train    import train
    from structured_transfer.budget            import EpochTimeBudgetTracker

    partition = ("F", "F", "T", "T", "S", "S")
    model, n_train, n_total = build_model(partition, CONFIG)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(CONFIG, device)
    tracker   = EpochTimeBudgetTracker(max_epochs=100, max_time_s=45 * 60)

    eval_log, final_results = train(model, CONFIG, evaluator, tracker)
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from ..budget import BudgetTracker
from ..utils import save_checkpoint, set_seed
from .data import get_dataloader
from .evaluate import DEFAULT_EVAL_SPLITS, Evaluator


def train(
    model: nn.Module,
    config: dict,
    evaluator: Evaluator,
    budget_tracker: BudgetTracker,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    eval_splits: list[str] | None = None,
    final_splits: list[str] | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[list[dict], dict]:
    """
    Run the training loop.

    Args:
        model:           model to train (moved to the device internally)
        config:          project config; must contain ``partition``
        evaluator:       Evaluator instance (shares DataLoaders across calls)
        budget_tracker:  controls evaluation frequency and stopping
        optimizer:       pre-built optimizer; built from config if None
        scheduler:       pre-built LR scheduler; built from config if None
        eval_splits:     splits evaluated mid-training (default: id_val + ood_val)
        final_splits:    splits evaluated once at the end
        checkpoint_dir:  if set, save a checkpoint after each evaluation

    Returns:
        ``(eval_log, final_results)`` -- one eval_log entry per mid-training
        evaluation, and the evaluator output on ``final_splits``.
    """
    set_seed(config["seed"])

    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    if eval_splits is None:
        eval_splits = DEFAULT_EVAL_SPLITS
    if final_splits is None:
        final_splits = ["train", "id_val", "ood_val", "id_test", "ood_test"]

    partition = config.get("partition", ("T", "T", "T", "T", "T", "S"))

    use_weighted = config.get("use_weighted_sampler", False)
    train_loader = get_dataloader("train", config, weighted_sampler=use_weighted)

    # Label smoothing both regularizes and softens the long tail: with 182 highly
    # imbalanced classes, hard targets push the head to be overconfident on the
    # few common species.
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.get("label_smoothing", 0.0)
    )

    if optimizer is None:
        from .models import get_param_groups
        param_groups = get_param_groups(model, partition, config)
        optimizer    = AdamW(param_groups, weight_decay=config.get("weight_decay", 1e-3))

    if scheduler is None:
        scheduler = _build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Opt-in, default off. The archived 21-run sweep did NOT hold frozen BN in
    # eval mode, so enabling this changes the protocol and makes new runs not
    # directly comparable to results/iwildcam/runs/. It is exposed because it is
    # the stricter reading of "frozen" -- see models.freeze_frozen_batchnorm.
    freeze_bn = config.get("freeze_frozen_bn", False)

    eval_log:    list[dict] = []
    global_step: int        = 0
    stopped:     bool       = False

    for epoch in range(config["epochs"]):
        model.train()
        if freeze_bn:
            from .models import freeze_frozen_batchnorm
            freeze_frozen_batchnorm(model, partition)

        epoch_loss  = 0.0
        n_correct   = 0
        n_total     = 0
        epoch_start = time.perf_counter()

        for batch_idx, (imgs, labels, _) in enumerate(train_loader):
            t0 = time.perf_counter()

            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()   # per-step cosine schedule

            batch_time   = time.perf_counter() - t0
            global_step += 1
            frac_epoch   = epoch + batch_idx / len(train_loader)

            epoch_loss += loss.item() * len(labels)
            n_correct  += (logits.argmax(1) == labels).sum().item()
            n_total    += len(labels)

            budget_tracker.step(batch_time, global_step, frac_epoch)

            if budget_tracker.should_eval():
                entry = _run_eval(
                    model, evaluator, budget_tracker, eval_splits,
                    train_loss=epoch_loss / n_total,
                    train_acc=n_correct / n_total,
                )
                eval_log.append(entry)
                _print_eval(entry, eval_splits)

                if checkpoint_dir is not None:
                    ckpt = Path(checkpoint_dir) / f"ckpt_step{global_step:07d}.pt"
                    save_checkpoint(model, optimizer, epoch, ckpt,
                                    extra={"step": global_step})

                # Evaluator left the model in eval mode; restore training mode
                # (and re-apply the BN freeze, which model.train() just undid).
                model.train()
                if freeze_bn:
                    from .models import freeze_frozen_batchnorm
                    freeze_frozen_batchnorm(model, partition)

            if budget_tracker.should_stop():
                print(f"Budget exhausted at step {global_step} "
                      f"(epoch {frac_epoch:.2f}). Stopping.")
                stopped = True
                break

        _print_epoch(epoch, config["epochs"], epoch_loss / max(n_total, 1),
                     n_correct / max(n_total, 1),
                     time.perf_counter() - epoch_start)

        if stopped:
            break

    print("\nRunning final evaluation ...")
    final_results = evaluator.run(model, splits=final_splits)
    _print_final(final_results)

    return eval_log, final_results


def _build_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """
    Cosine annealing over the *nominal* horizon (``epochs x steps_per_epoch``).

    Under a wall-clock budget a run may stop before the schedule completes, so
    the LR does not necessarily reach its floor. That is intentional: every
    configuration then follows an identical LR trajectory for as long as it runs,
    rather than each getting a different schedule shape.
    """
    sched  = config.get("scheduler", "cosine")
    epochs = config["epochs"]
    if sched == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
    if sched == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    return None  # "none"


def _run_eval(
    model, evaluator, budget_tracker, eval_splits,
    train_loss: float, train_acc: float,
) -> dict:
    """Flatten one evaluation into a single log row stamped with cost metrics."""
    snapshot = budget_tracker.get_snapshot()
    eval_res = evaluator.run(model, splits=eval_splits)

    entry: dict = {**snapshot, "train_loss": train_loss, "train_acc": train_acc}
    for split, metrics in eval_res.items():
        for k, v in metrics.items():
            entry[f"{split}_{k}"] = v
    return entry


def _print_eval(entry: dict, eval_splits: list[str]) -> None:
    header = (f"step {entry['step']:>6}  epoch {entry['epoch']:.2f}  "
              f"elapsed {entry['elapsed_s']:.0f}s")
    lines = [header]
    for split in eval_splits:
        if f"{split}_acc" not in entry:
            continue
        lines.append(
            f"  {split:8s}  "
            f"acc={entry[f'{split}_acc']:.3f}  "
            f"bal={entry.get(f'{split}_balanced_acc', float('nan')):.3f}  "
            f"f1={entry.get(f'{split}_f1', float('nan')):.3f}  "
            f"loss={entry.get(f'{split}_loss', float('nan')):.4f}"
        )
    print("\n".join(lines))


def _print_epoch(epoch: int, total: int, loss: float, acc: float, elapsed_s: float) -> None:
    print(f"Epoch {epoch + 1}/{total}  train_loss={loss:.4f}  "
          f"train_acc={acc:.3f}  epoch_time={elapsed_s:.0f}s")


def _print_final(results: dict) -> None:
    print("\n-- Final results " + "-" * 46)
    for split, metrics in results.items():
        acc     = metrics.get("acc",          float("nan"))
        bal_acc = metrics.get("balanced_acc", float("nan"))
        f1      = metrics.get("f1",           float("nan"))
        loss    = metrics.get("loss",         float("nan"))
        print(f"  {split:12s}  acc={acc:.3f}  bal_acc={bal_acc:.3f}  "
              f"f1={f1:.3f}  loss={loss:.4f}")
