"""
WRN-28-10 assembled according to an F/T/S partition (paper, Experiment 1).

The architecture reproduces the reported parameter and forward-FLOP budget
exactly; ``tests/test_models.py`` asserts the per-block counts against it.

Architecture
------------
Standard WideResNet (Zagoruyko & Komodakis) with depth 28 and widening factor 10:
an initial 3x3 convolution followed by three residual groups of
``n = (28 - 4) / 6 = 4`` basic blocks each, with widths ``16 -> 160 -> 320 -> 640``.
Applied directly to 64x64 TinyImageNet input, so every spatial map is twice the
size of the usual CIFAR configuration.

Four controlled blocks, matching Table 1:

    Block        Parameters                Forward FLOPs        Output
    conv1               432  ( 0.001%)      0.0018 G ( 0.007%)   16 x 64 x 64
    group1        1,640,672  ( 4.49%)       6.7202 G (28.21%)   160 x 64 x 64
    group2        6,968,000  (19.07%)       8.5538 G (35.90%)   320 x 32 x 32
    group3       27,862,400  (76.26%)       8.5498 G (35.88%)   640 x 16 x 16
    head             65,380  ( 0.18%)       0.0006 G ( 0.002%)  100 logits
    total        36,536,884  (100.0%)      23.8261 G (100.0%)

The distribution is the mirror image of ResNet50's: here the deepest group holds
**76%** of the parameters while carrying only a third of the FLOPs. That is what
makes ``group3`` the decisive block in this experiment -- both the FFFF
cross-domain collapse and the G2->G3 scratch-after-fine-tune instability are
about how ``group3`` is treated.

Head composition (Table 1's 65,380): final BatchNorm affine ``2 x 640 = 1,280``
plus the linear classifier ``640 x 100 + 100 = 64,100``. The paper's head is the
final normalization/ReLU, global average pooling, and a linear layer, so the
trailing BN belongs to the head rather than to ``group3``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..partitions import WRN2810_BLOCKS as BLOCKS
from ..partitions import MODES, config_label, enumerate_configs, validate_partition
from ..utils import count_parameters

__all__ = [
    "BLOCKS", "MODES", "WideResNet", "build_model", "get_param_groups",
    "freeze_frozen_batchnorm", "config_label", "enumerate_configs",
]


class _BasicBlock(nn.Module):
    """
    Pre-activation wide residual block: BN-ReLU-Conv-BN-ReLU-Conv, plus shortcut.

    The 1x1 shortcut convolution appears only when the block changes shape (first
    block of a group). It is intentionally counted as part of its group, which is
    what makes the parameter totals match Table 1.
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        # Pre-activation: when the block projects, the shortcut branches off the
        # activated tensor, not the raw input.
        identity = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + identity


class WideResNet(nn.Module):
    """
    WRN-depth-widen with its four controlled blocks exposed as named attributes.

    Attributes ``conv1``, ``group1``, ``group2``, ``group3`` and ``head`` line up
    one-to-one with :data:`BLOCKS`, so partition logic can address them directly.
    """

    def __init__(
        self,
        num_classes: int = 100,
        depth: int = 28,
        widen_factor: int = 10,
    ):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WRN depth must satisfy (depth - 4) % 6 == 0, got {depth}.")
        n = (depth - 4) // 6                       # blocks per group; 4 for WRN-28
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # No stride and no max-pool at the stem: TinyImageNet is already small at
        # 64x64, so downsampling immediately would discard most of the signal.
        # Resolution therefore runs 64 -> 64 -> 32 -> 16, as Appendix A.1 states.
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.group1 = self._make_group(widths[0], widths[1], n, stride=1)
        self.group2 = self._make_group(widths[1], widths[2], n, stride=2)
        self.group3 = self._make_group(widths[2], widths[3], n, stride=2)

        # Final normalization + ReLU + global average pooling + linear classifier.
        self.head = _WRNHead(widths[3], num_classes)

        self._init_weights()

    @staticmethod
    def _make_group(in_planes: int, out_planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks = [_BasicBlock(in_planes, out_planes, stride)]
        blocks += [_BasicBlock(out_planes, out_planes, 1) for _ in range(n_blocks - 1)]
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        return self.head(x)


class _WRNHead(nn.Module):
    """Final BN + ReLU + global average pooling + linear classifier."""

    def __init__(self, in_planes: int, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def build_model(
    partition: tuple | list,
    config: dict,
    source_state_dict: dict | None = None,
) -> tuple[nn.Module, int, int]:
    """
    Build a WRN-28-10 configured for the given F/T/S block partition.

    Args:
        partition: 5-element sequence aligned with :data:`BLOCKS`
                   ``(conv1, group1, group2, group3, head)``. Head must be ``'S'``.
        config:    project config; reads ``num_classes``, ``depth``, ``widen_factor``.
        source_state_dict:
                   weights of the source network (``baseA`` or ``baseB``). Required
                   whenever any block is ``F`` or ``T``. Unlike Experiment 2 there
                   is no external pretrained checkpoint -- the source networks are
                   trained from scratch by this same pipeline, which is what makes
                   the selfer/transfer comparison controlled.

    Returns:
        ``(model on CPU, n_trainable, n_total)``.
    """
    validate_partition(partition, BLOCKS)

    model = WideResNet(
        num_classes  = config.get("num_classes", 100),
        depth        = config.get("depth", 28),
        widen_factor = config.get("widen_factor", 10),
    )

    needs_source = any(m in ("F", "T") for m in partition[:-1])
    if needs_source and source_state_dict is None:
        raise ValueError(
            f"Partition {config_label(partition)} copies source weights into at "
            f"least one block, so source_state_dict is required. Train a base "
            f"network first: scripts/tinyimagenet_run.py --config SSSS --subset A"
        )

    if needs_source:
        _load_source_blocks(model, source_state_dict, partition)

    # The head is always S. It is already randomly initialized by the constructor
    # and never copied: source and target use disjoint 100-class subsets, so the
    # source classifier's label space is meaningless on the target.
    for block_name, mode in zip(BLOCKS, partition):
        module = getattr(model, block_name)
        if mode == "F":
            _set_grad(module, False)
        else:                       # 'T' and 'S' are both trainable
            _set_grad(module, True)

    n_trainable, n_total = count_parameters(model)
    return model, n_trainable, n_total


def _load_source_blocks(
    model: nn.Module,
    source_state_dict: dict,
    partition: tuple | list,
) -> None:
    """
    Copy source weights into every ``F`` and ``T`` block; leave ``S`` blocks random.

    Copying block-by-block rather than loading the whole state dict and
    re-randomizing afterwards keeps the S blocks at exactly the initialization
    the constructor produced, so a run's scratch blocks depend only on the seed
    and not on which source network was loaded.
    """
    for block_name, mode in zip(BLOCKS, partition):
        if mode not in ("F", "T"):
            continue
        prefix = f"{block_name}."
        block_weights = {
            key[len(prefix):]: value
            for key, value in source_state_dict.items()
            if key.startswith(prefix)
        }
        if not block_weights:
            raise KeyError(
                f"Source checkpoint has no parameters under {prefix!r}. "
                f"Expected keys for blocks {BLOCKS}."
            )
        getattr(model, block_name).load_state_dict(block_weights)


def get_param_groups(
    model: nn.Module,
    partition: tuple | list,
    config: dict,
) -> list[dict]:
    """
    Two learning rates, split by *initialization* rather than by depth.

    Per Appendix A.1: scratch backbone blocks and the classifier head train at
    ``lr_scratch`` (0.1), while copied fine-tuned blocks train at ``lr_finetune``
    (0.01). Frozen blocks are excluded from the optimizer entirely.

    The 10x gap is the point: a randomly initialized block needs a large step to
    learn anything in 100 epochs, whereas a transferred block only needs to drift
    from an already-good solution -- and would be destroyed by lr 0.1. This gap is
    also the mechanism behind the paper's G2->G3 instability: a large scratch
    ``group3`` takes big steps while its input distribution is still moving,
    because the fine-tuned ``group2`` beneath it is changing at the same time.
    """
    lr_scratch  = config.get("lr_scratch", 0.1)
    lr_finetune = config.get("lr_finetune", 0.01)

    scratch_params:  list[nn.Parameter] = []
    finetune_params: list[nn.Parameter] = []

    for block_name, mode in zip(BLOCKS, partition):
        module = getattr(model, block_name)
        params = [p for p in module.parameters() if p.requires_grad]
        if mode == "T":
            finetune_params.extend(params)
        elif mode == "S":
            scratch_params.extend(params)
        # 'F' blocks contribute nothing: requires_grad is already False.

    groups = []
    if finetune_params:
        groups.append({"params": finetune_params, "lr": lr_finetune})
    if scratch_params:
        groups.append({"params": scratch_params, "lr": lr_scratch})
    return groups


def freeze_frozen_batchnorm(model: nn.Module, partition: tuple | list) -> None:
    """
    Hold BatchNorm layers inside ``F`` blocks in eval mode.

    Required by Appendix A.1: "Batch-normalization layers whose affine parameters
    are frozen are kept in evaluation mode, preserving source-domain running
    statistics in frozen blocks." Without this a frozen block is only
    half-frozen -- its weights hold still, but its running mean and variance keep
    absorbing target-domain batches, so its outputs drift and the FFFF
    cross-domain collapse would be measuring the wrong thing.

    Call after every ``model.train()``, which otherwise resets all submodules.
    """
    for block_name, mode in zip(BLOCKS, partition):
        if mode != "F":
            continue
        for submodule in getattr(model, block_name).modules():
            if isinstance(submodule, nn.modules.batchnorm._BatchNorm):
                submodule.eval()


def block_parameter_table(model: nn.Module | None = None) -> dict[str, int]:
    """
    Parameter count per controlled block -- reproduces the paper's Table 1.

    Used by the tests to assert the architecture matches the published budget.
    """
    if model is None:
        model = WideResNet(num_classes=100)
    return {
        name: sum(p.numel() for p in getattr(model, name).parameters())
        for name in BLOCKS
    }


def _set_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value
