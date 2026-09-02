"""
ResNet50 assembled according to an F/T/S partition (paper, Experiment 2).

Block decomposition (Appendix A.2), with the parameter and forward-FLOP shares
from Table 2:

    Block   Parameters              Forward FLOPs
    stem         9,536  ( 0.04%)     0.1196 G ( 2.91%)   conv1 + bn1
    stage1     215,808  ( 0.90%)     0.6768 G (16.47%)   layer1
    stage2   1,219,584  ( 5.11%)     1.0338 G (25.15%)   layer2
    stage3   7,098,368  (29.72%)     1.4687 G (35.74%)   layer3
    stage4  14,964,736  (62.66%)     0.8105 G (19.72%)   layer4
    head       372,918  ( 1.56%)     0.0005 G ( 0.01%)   fc (182 classes)
    total   23,880,950  (100.0%)     4.1098 G (100.0%)

Note how badly parameters and FLOPs disagree: Stage4 holds 63% of the weights but
only 20% of the compute, because it runs at 7x7 resolution. That mismatch is why
the paper reports a FLOP-based efficiency frontier rather than a
trainable-parameter one.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from ..partitions import RESNET50_BLOCKS as BLOCKS
from ..partitions import MODES, config_label, enumerate_configs, validate_partition
from ..utils import count_parameters

__all__ = [
    "BLOCKS", "MODES", "build_model", "get_param_groups",
    "config_label", "enumerate_configs",
]


def build_model(
    partition: tuple | list,
    config: dict,
) -> tuple[nn.Module, int, int]:
    """
    Build a ResNet50 configured for the given F/T/S block partition.

    Args:
        partition: 6-element sequence aligned with :data:`BLOCKS`
                   ``(stem, stage1, stage2, stage3, stage4, head)``, each one of:
                     ``'F'`` frozen pretrained weights
                     ``'T'`` trainable pretrained weights
                     ``'S'`` reinitialized (scratch) and trainable
                   The head must always be ``'S'``.
        config:    project config; reads ``head_dropout`` and, as a fallback,
                   ``num_classes``.

    Returns:
        ``(model on CPU, n_trainable, n_total)``. The caller moves it to a device.
    """
    validate_partition(partition, BLOCKS)

    # Prefer the class count implied by the metadata on disk; fall back to the
    # config when the dataset is not present (e.g. running the unit tests).
    try:
        from .data import get_num_classes
        num_classes = get_num_classes(config)
    except Exception:
        num_classes = config.get("num_classes", 182)

    # ImageNet weights are only downloaded when at least one backbone block
    # actually reuses them. For the all-scratch baseline S6 nothing is loaded, so
    # it is a genuine no-pretraining control rather than a reinitialized copy.
    needs_pretrained = any(m in ("F", "T") for m in partition[:-1])
    weights = "IMAGENET1K_V1" if needs_pretrained else None
    model = models.resnet50(weights=weights)

    # Fresh head. Always S: the source label space (ImageNet-1k) and the target
    # label space (182 iWildCam species) share nothing, so copied classifier
    # weights would be noise.
    head = nn.Linear(model.fc.in_features, num_classes)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    dropout_p = float(config.get("head_dropout", 0.0))
    model.fc = nn.Sequential(nn.Dropout(p=dropout_p), head) if dropout_p > 0 else head

    block_modules = _get_block_modules(model)
    for block_name, mode in zip(BLOCKS, partition):
        modules = block_modules[block_name]
        if mode == "F":
            _set_grad(modules, False)
        elif mode == "T":
            _set_grad(modules, True)
        else:  # S
            if block_name != "head":
                _reinitialize(modules)   # the head was just built fresh above
            _set_grad(modules, True)

    n_trainable, n_total = count_parameters(model)
    return model, n_trainable, n_total


def get_param_groups(
    model: nn.Module,
    partition: tuple | list,
    config: dict,
) -> list[dict]:
    """
    Two parameter groups: trainable backbone at ``lr``, head at ``lr_head``.

    Frozen parameters are excluded entirely rather than given a zero learning
    rate, so the optimizer allocates no state for them. For the fully frozen
    F5S1 configuration no backbone group exists at all and this degenerates to a
    linear probe over fixed ImageNet features.
    """
    lr      = config["lr"]
    lr_head = config.get("lr_head", lr)

    head_ids           = {id(p) for p in model.fc.parameters()}
    backbone_trainable = [p for p in model.parameters()
                          if p.requires_grad and id(p) not in head_ids]
    head_trainable     = [p for p in model.fc.parameters() if p.requires_grad]

    groups = []
    if backbone_trainable:
        groups.append({"params": backbone_trainable, "lr": lr})
    if head_trainable:
        groups.append({"params": head_trainable, "lr": lr_head})
    return groups


def freeze_frozen_batchnorm(model: nn.Module, partition: tuple | list) -> None:
    """
    Hold BatchNorm layers inside ``F`` blocks in eval mode.

    Call this after ``model.train()`` on every epoch. Without it, a "frozen"
    block is not actually frozen: its affine weights stay fixed, but the running
    mean and variance keep updating from target-domain batches, so the block's
    output drifts anyway. Keeping the source statistics is what makes F mean
    *reuse* (paper, Appendix A.1).
    """
    block_modules = _get_block_modules(model)
    for block_name, mode in zip(BLOCKS, partition):
        if mode != "F":
            continue
        for module in block_modules[block_name]:
            for submodule in module.modules():
                if isinstance(submodule, nn.modules.batchnorm._BatchNorm):
                    submodule.eval()


def _get_block_modules(model: nn.Module) -> dict[str, list[nn.Module]]:
    """Map block names to the submodules carrying their parameters."""
    return {
        "stem":   [model.conv1, model.bn1],
        "stage1": [model.layer1],
        "stage2": [model.layer2],
        "stage3": [model.layer3],
        "stage4": [model.layer4],
        "head":   [model.fc],
    }


def _set_grad(modules: list[nn.Module], value: bool) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad = value


def _reinitialize(modules: list[nn.Module]) -> None:
    """Kaiming/Xavier reinitialization for conv, norm and linear layers."""
    for m in modules:
        for subm in m.modules():
            if isinstance(subm, nn.Conv2d):
                nn.init.kaiming_normal_(subm.weight, mode="fan_out", nonlinearity="relu")
                if subm.bias is not None:
                    nn.init.zeros_(subm.bias)
            elif isinstance(subm, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(subm.weight)
                nn.init.zeros_(subm.bias)
            elif isinstance(subm, nn.Linear):
                nn.init.xavier_uniform_(subm.weight)
                if subm.bias is not None:
                    nn.init.zeros_(subm.bias)
