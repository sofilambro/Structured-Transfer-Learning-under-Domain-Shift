"""
Structured transfer learning under domain shift.

Code for the study *"Beyond Freeze-or-Fine-Tune: Structured Transfer Learning
under Domain Shift"*, which asks which depths of a convolutional backbone should
be **reused**, **adapted**, or **relearned**, by sweeping monotonic
Frozen/Fine-tuned/Scratch partitions ``F^a T^b S^c``.

Layout
------
``partitions``    the F/T/S formalism, shared by both experiments
``budget``        compute-budget tracking (fixed-protocol experiments)
``utils``         seeding, parameter accounting, FLOPs, checkpointing
``iwildcam``      Experiment 2 -- ImageNet-pretrained ResNet50 -> iWildCam (WILDS)
``tinyimagenet``  Experiment 1 -- WRN-28-10 on disjoint TinyImageNet class splits
``analysis``      loading run artifacts and regenerating the paper's figures

Submodules are not imported eagerly: ``partitions`` and ``analysis.load`` are
useful without a GPU or a Torch install, and importing the package should not
drag in torchvision.
"""

__version__ = "1.0.0"

__all__ = ["partitions", "budget", "utils", "iwildcam", "tinyimagenet", "analysis"]
