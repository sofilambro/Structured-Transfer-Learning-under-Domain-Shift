"""
iWildCam data loading.

Deliberately CSV-based rather than built on the ``wilds`` dataset object. The
WILDS loader expects the full archive layout on disk, which makes the mini subset
awkward; reading ``metadata.csv`` plus a flat image directory means the mini and
full datasets take the exact same code path, and switching between them is one
config flag. The official WILDS metrics remain reachable for sanity checks via
``evaluate.Evaluator.run_wilds_official``.

Split semantics (paper, Appendix A.2): OOD splits come from camera-trap locations
never seen in training, which shifts background, illumination, geography,
viewpoint, class priors and animal appearance all at once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# Our canonical split names -> the names used in the WILDS metadata CSV.
# The CSV calls the OOD splits "val"/"test" with no marker, which reads as if
# they were the ordinary validation and test sets; renaming them at the boundary
# keeps the distinction explicit everywhere else in the codebase.
_SPLIT_MAP = {
    "train":    "train",
    "id_val":   "id_val",
    "ood_val":  "val",
    "id_test":  "id_test",
    "ood_test": "test",
}
VALID_SPLITS = list(_SPLIT_MAP.keys())

# ImageNet statistics -- required, since the backbone is ImageNet-pretrained and
# frozen blocks expect inputs in that distribution.
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    Augmentation for ``train``; deterministic resize + center crop otherwise.

    Any split name other than ``"train"`` yields the evaluation transform, so
    ``get_transforms("val")`` is a valid way to ask for it.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _get_paths(config: dict) -> tuple[Path, Path]:
    """Return ``(meta_path, img_dir)`` for the configured dataset mode."""
    data_dir = Path(config["data_dir"])
    mode = config.get("dataset_mode", "full")
    if mode == "mini":
        return (
            data_dir / "iwildcam_mini" / "metadata.csv",
            data_dir / "iwildcam_mini" / "train",
        )
    if mode == "full":
        return (
            data_dir / "iwildcam_v2.0" / "metadata.csv",
            data_dir / "iwildcam_v2.0" / "train",
        )
    raise ValueError(f"Unknown dataset_mode {mode!r}. Use 'mini' or 'full'.")


def get_num_classes(config: dict) -> int:
    """
    Read ``max(y) + 1`` from the metadata rather than trusting the config.

    Label ids are global across splits, so this is 182 for both mini and full
    even though the mini subset only *contains* 53 classes. Building the head
    from the metadata keeps checkpoints interchangeable between modes.
    """
    meta_path, _ = _get_paths(config)
    df = pd.read_csv(meta_path, usecols=["y"])
    return int(df["y"].max()) + 1


class IWildCamDataset(Dataset):
    """
    iWildCam images from a metadata CSV plus a flat image directory.

    Identical for mini and full -- only the paths differ.
    Each item is ``(image_tensor, label, location_id)``; the location id is
    carried through so per-domain analyses stay possible downstream, even though
    the training loop ignores it.
    """

    def __init__(
        self,
        meta_path: Path,
        img_dir: Path,
        split: str,
        transform=None,
    ):
        if split not in _SPLIT_MAP:
            raise ValueError(f"Unknown split {split!r}. Valid: {VALID_SPLITS}")

        df = pd.read_csv(meta_path)
        self.df        = df[df["split"] == _SPLIT_MAP[split]].reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.split     = split

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["filename"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["y"]), int(row["location_remapped"])

    @property
    def labels(self) -> np.ndarray:
        return self.df["y"].to_numpy()

    @property
    def locations(self) -> np.ndarray:
        return self.df["location_remapped"].to_numpy()


def get_dataloader(
    split: str,
    config: dict,
    transform=None,
    weighted_sampler: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for one split.

    Args:
        split:            one of :data:`VALID_SPLITS`
        config:           project config (uses dataset_mode, batch_size, ...)
        transform:        override the default split transform
        weighted_sampler: inverse-frequency class sampling; train split only

    ``drop_last`` is on for training so BatchNorm never sees a size-1 batch.
    """
    meta_path, img_dir = _get_paths(config)

    if transform is None:
        transform = get_transforms(split, config.get("image_size", 224))

    dataset = IWildCamDataset(meta_path, img_dir, split, transform)

    # A sampler and shuffle=True are mutually exclusive in PyTorch.
    shuffle = split == "train" and not weighted_sampler
    sampler = (
        _make_weighted_sampler(dataset)
        if (weighted_sampler and split == "train")
        else None
    )

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=split == "train",
    )


def _make_weighted_sampler(dataset: IWildCamDataset) -> WeightedRandomSampler:
    """
    Inverse-frequency sampling over classes.

    Each epoch draws ``len(dataset)`` samples with replacement, so an epoch keeps
    its nominal length but rare species appear far more often than their natural
    rate. ``clip(min=1)`` guards against a zero count for a label id that exists
    globally but is absent from this split.
    """
    labels       = dataset.labels
    class_counts = np.bincount(labels, minlength=labels.max() + 1).clip(min=1)
    weights      = (1.0 / class_counts)[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(labels),
        replacement=True,
    )
