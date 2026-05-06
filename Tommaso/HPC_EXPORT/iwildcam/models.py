import torch.nn as nn
from torchvision import models

from utils import count_parameters

BLOCKS = ("stem", "stage1", "stage2", "stage3", "stage4", "head")
MODES  = ("F", "T", "S")


def enumerate_configs() -> list[tuple[str, ...]]:
    """
    Return all 21 valid monotonic partitions as 6-tuples over BLOCKS.
    Backbone (stem–stage4) follows F^a T^b S^c with a+b+c=5; head is always S.
    """
    configs = []
    n = len(BLOCKS) - 1  # 5 backbone blocks
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            configs.append(tuple(["F"] * a + ["T"] * b + ["S"] * c + ["S"]))
    return configs


def config_label(partition: tuple | list) -> str:
    """
    Compact human-readable label for a partition.
    e.g. ("F","F","F","F","F","S") -> "F5S1"
         ("F","F","T","T","S","S") -> "F2T2S2"
         ("S","S","S","S","S","S") -> "S6"
    """
    counts = {m: 0 for m in MODES}
    for m in partition:
        counts[m] += 1
    return "".join(f"{m}{counts[m]}" for m in MODES if counts[m] > 0)


def build_model(
    partition: tuple | list,
    config: dict,
) -> tuple[nn.Module, int, int]:
    """
    Build a ResNet50 configured for the given F/T/S block partition.

    partition: 6-element sequence aligned with BLOCKS = (stem, stage1, stage2,
               stage3, stage4, head). Each element must be one of:
                 'F' – frozen pretrained weights
                 'T' – trainable pretrained weights
                 'S' – reinitialized (scratch) and trainable
               The head (index 5) must always be 'S'.

    Returns: (model on CPU, n_trainable, n_total)
    """
    _validate_partition(partition)

    try:
        from data import get_num_classes
        num_classes = get_num_classes(config)
    except Exception:
        num_classes = config.get("num_classes", 182)

    needs_pretrained = any(m in ("F", "T") for m in partition[:-1])
    weights = "IMAGENET1K_V1" if needs_pretrained else None
    model = models.resnet50(weights=weights)

    # Fresh head (always S: Xavier init, fully trainable)
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
                _reinitialize(modules)
            _set_grad(modules, True)

    n_trainable, n_total = count_parameters(model)
    return model, n_trainable, n_total


def get_param_groups(
    model: nn.Module,
    partition: tuple | list,
    config: dict,
) -> list[dict]:
    """
    Two parameter groups: backbone trainable (lr) and head (lr_head).
    If no backbone parameters are trainable (pure linear probe), only the head
    group is returned.
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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate_partition(partition: tuple | list) -> None:
    if len(partition) != len(BLOCKS):
        raise ValueError(
            f"partition must have {len(BLOCKS)} elements (one per block), "
            f"got {len(partition)}. Blocks: {BLOCKS}"
        )
    for block, mode in zip(BLOCKS, partition):
        if mode not in MODES:
            raise ValueError(
                f"Block {block!r}: invalid mode {mode!r}. Use 'F', 'T', or 'S'."
            )
    if partition[-1] != "S":
        raise ValueError(
            f"Head (last block) must always be 'S', got {partition[-1]!r}."
        )


def _get_block_modules(model: nn.Module) -> dict[str, list[nn.Module]]:
    """Map block names to the list of submodules that carry their parameters."""
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
    """Kaiming/Xavier reinit for conv, BN, and linear layers."""
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
