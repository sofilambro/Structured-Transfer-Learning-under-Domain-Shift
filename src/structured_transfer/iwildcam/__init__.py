"""
Experiment 2 -- ImageNet-pretrained ResNet50 adapted to iWildCam (WILDS).

Real distribution shift: OOD examples come from camera-trap locations never seen
during training, which changes background, illumination, geography, viewpoint,
class priors and animal appearance simultaneously. The backbone is split into
five controlled blocks (stem, stage1-stage4) plus an always-scratch head, giving
21 valid monotonic partitions.

Headline result: the fully frozen backbone F5S1 is *worse than training from
scratch* (22.8% vs 30.3% raw OOD accuracy), while full fine-tuning T5S1 is best.
Under shift this large, reuse alone is not enough.
"""

from .config import CONFIG

__all__ = ["CONFIG"]
