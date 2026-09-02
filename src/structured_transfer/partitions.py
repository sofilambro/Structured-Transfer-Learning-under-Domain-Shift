"""
Monotonic Frozen / Fine-tuned / Scratch (F/T/S) backbone partitions.

This is the formal core of the study (paper, Section 3). A convolutional backbone
is decomposed into ``L`` ordered functional blocks ``B_1, ..., B_L``, from early
feature extractors to deeper task-specific stages. Each block is assigned a state

    z_l in {F, T, S},    l = 1, ..., L

where

    F  copied source weights, held fixed (reuse)
    T  copied source weights, optimized  (adapt)
    S  randomly reinitialized, optimized (relearn)

and the search is restricted to *monotonic* configurations

    F^a T^b S^c,    a + b + c = L

so frozen blocks form a lower prefix, fine-tuned blocks form an adaptation region,
and scratch blocks sit closest to the task-specific head.

Why monotonic only (paper, Section 3): a reinitialized block placed *below* copied
blocks would feed them feature distributions they were never trained to process,
and interleaved patterns would introduce several representation shifts along depth,
making transfer failure indistinguishable from architecture-induced mismatch.
Confining S to the top keeps relearning inside task-specific stages, and lets T act
as a transition region between fixed source features and new target features.

The number of valid partitions is the (L+1)-th triangular number:

    (L + 1)(L + 2) / 2

which gives **15** for the four-block WRN-28-10 study and **21** for the five-block
ResNet50 study.

Notation caveat carried over from the experiment logs
-----------------------------------------------------
The classifier head is always ``S`` (the target label space differs from the
source label space) and is *not* counted in ``L`` in the paper's notation. The
iWildCam run logs, however, append the head explicitly as a trailing ``S1``:
``T5S1`` means five fine-tuned backbone blocks plus a scratch head, and ``S6``
means five scratch backbone blocks plus a scratch head. This module follows the
log convention -- partitions include the head slot -- because that is what the
archived result files use. Use :func:`backbone_of` to drop the head when you want
the paper's block-level notation.
"""

from __future__ import annotations

import re

#: The three block states. Order matters: labels are emitted F, then T, then S.
MODES: tuple[str, ...] = ("F", "T", "S")

#: Controlled blocks of the ImageNet-pretrained ResNet50 (paper, Appendix A.2).
#: ``stage1``-``stage4`` are torchvision's ``layer1``-``layer4``.
RESNET50_BLOCKS: tuple[str, ...] = (
    "stem", "stage1", "stage2", "stage3", "stage4", "head",
)

#: Controlled blocks of WRN-28-10 on TinyImageNet (paper, Appendix A.1).
WRN2810_BLOCKS: tuple[str, ...] = ("conv1", "group1", "group2", "group3", "head")

#: Backwards-compatible alias. The iWildCam modules historically imported
#: ``BLOCKS`` from ``models``; keep the name pointing at the ResNet50 layout so
#: the archived run files and older call sites keep working.
BLOCKS = RESNET50_BLOCKS

_LABEL_RE = re.compile(r"^(?:F(\d+))?(?:T(\d+))?(?:S(\d+))?$")


def enumerate_configs(blocks: tuple[str, ...] = RESNET50_BLOCKS) -> list[tuple[str, ...]]:
    """
    All valid monotonic partitions for a backbone, as tuples aligned with ``blocks``.

    The backbone (everything but the trailing head slot) follows ``F^a T^b S^c``
    with ``a + b + c == len(blocks) - 1``; the head is always appended as ``S``.

    Returns ``(n+1)(n+2)/2`` configurations for ``n`` backbone blocks: 21 for
    ResNet50 (n=5), 15 for WRN-28-10 (n=4). Ordering is by increasing frozen
    depth ``a``, then increasing fine-tuned depth ``b``, which is the order the
    paper's Appendix B lists them in.
    """
    configs: list[tuple[str, ...]] = []
    n = len(blocks) - 1  # backbone blocks; the head slot is not partitioned
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            configs.append(tuple(["F"] * a + ["T"] * b + ["S"] * c + ["S"]))
    return configs


def config_label(partition: tuple | list) -> str:
    """
    Compact label for a partition, counting the head slot.

    >>> config_label(("F", "F", "F", "F", "F", "S"))
    'F5S1'
    >>> config_label(("F", "F", "T", "T", "S", "S"))
    'F2T2S2'
    >>> config_label(("S", "S", "S", "S", "S", "S"))
    'S6'

    Modes contributing zero blocks are omitted, so ``T5S1`` -- not ``F0T5S1``.
    """
    counts = {m: 0 for m in MODES}
    for m in partition:
        counts[m] += 1
    return "".join(f"{m}{counts[m]}" for m in MODES if counts[m] > 0)


def parse_label(label: str, blocks: tuple[str, ...] = RESNET50_BLOCKS) -> tuple[str, ...]:
    """
    Inverse of :func:`config_label`: turn ``"F2T3S1"`` back into a partition tuple.

    This is what the command-line runners use, so a sweep can be driven by label
    (``--partition F2T3S1``) instead of by editing a tuple in a source file.

    >>> parse_label("F2T3S1")
    ('F', 'F', 'T', 'T', 'T', 'S')

    Raises ``ValueError`` if the label is malformed, or if its block count does
    not match ``blocks``.
    """
    match = _LABEL_RE.match(label.strip().upper())
    if not match or not label.strip():
        raise ValueError(
            f"Malformed partition label {label!r}. Expected e.g. 'F2T3S1', 'T5S1', 'S6'."
        )

    counts = [int(g) if g else 0 for g in match.groups()]
    partition = tuple(
        mode for mode, count in zip(MODES, counts) for _ in range(count)
    )

    if len(partition) != len(blocks):
        raise ValueError(
            f"Label {label!r} describes {len(partition)} blocks, but this backbone "
            f"has {len(blocks)} ({', '.join(blocks)}). Remember the trailing head "
            f"slot: five ResNet50 backbone blocks plus a scratch head is 'T5S1', not 'T5'."
        )
    validate_partition(partition, blocks)
    return partition


def validate_partition(
    partition: tuple | list,
    blocks: tuple[str, ...] = RESNET50_BLOCKS,
) -> None:
    """
    Assert that ``partition`` is a usable monotonic F/T/S assignment.

    Checks, in order: correct length, valid mode letters, head is ``S``, and
    monotonicity (no F after T or S, no T after S). Raises ``ValueError`` with a
    message naming the offending block; returns ``None`` on success.
    """
    if len(partition) != len(blocks):
        raise ValueError(
            f"partition must have {len(blocks)} elements (one per block), "
            f"got {len(partition)}. Blocks: {blocks}"
        )

    for block, mode in zip(blocks, partition):
        if mode not in MODES:
            raise ValueError(
                f"Block {block!r}: invalid mode {mode!r}. Use 'F', 'T', or 'S'."
            )

    # The head is relearned unconditionally: the target label space differs from
    # the source label space, so copied classifier weights are meaningless.
    if partition[-1] != "S":
        raise ValueError(
            f"Head (last block) must always be 'S', got {partition[-1]!r}."
        )

    # Monotonicity: the mode index must never decrease along depth.
    rank = {mode: i for i, mode in enumerate(MODES)}
    for i in range(1, len(partition)):
        if rank[partition[i]] < rank[partition[i - 1]]:
            raise ValueError(
                f"Non-monotonic partition {config_label(partition)}: block "
                f"{blocks[i]!r} is {partition[i]!r} but sits above "
                f"{blocks[i - 1]!r} which is {partition[i - 1]!r}. Only F^a T^b S^c "
                f"orderings are studied -- see structured_transfer.partitions."
            )


def backbone_of(partition: tuple | list) -> tuple[str, ...]:
    """Drop the trailing head slot, giving the paper's block-level notation."""
    return tuple(partition[:-1])


def depths(partition: tuple | list) -> dict[str, int]:
    """
    Structural coordinates of a partition, used as analysis axes.

    Returns ``frozen_depth`` (a), ``tuned_depth`` (b), ``scratch_depth`` (c) over
    the *backbone only*, plus ``scratch_start`` (a+b) -- the index of the first
    backbone block trained from scratch. ``scratch_start`` is the axis that
    exposes the paper's key iWildCam finding: whether Stage4 is relearned or
    adapted dominates OOD performance.
    """
    backbone = backbone_of(partition)
    a = backbone.count("F")
    b = backbone.count("T")
    c = backbone.count("S")
    return {
        "frozen_depth":  a,
        "tuned_depth":   b,
        "scratch_depth": c,
        "scratch_start": a + b,
    }
