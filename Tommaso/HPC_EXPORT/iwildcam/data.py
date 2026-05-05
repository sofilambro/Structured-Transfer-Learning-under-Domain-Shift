from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# ── Split name mapping ────────────────────────────────────────────────────────
# Our canonical names  →  names used in the metadata CSV
_SPLIT_MAP = {
    "train":    "train",
    "id_val":   "id_val",
    "ood_val":  "val",
    "id_test":  "id_test",
    "ood_test": "test",
}
VALID_SPLITS = list(_SPLIT_MAP.keys())

# ── ImageNet normalisation ────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
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


# ── Path resolution ───────────────────────────────────────────────────────────
def _get_paths(config: dict) -> tuple[Path, Path]:
    """Return (meta_path, img_dir) for the configured dataset mode."""
    data_dir = Path(config["data_dir"])
    mode = config.get("dataset_mode", "mini")
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
    """Read the actual max(y)+1 from metadata (may differ from config value)."""
    meta_path, _ = _get_paths(config)
    df = pd.read_csv(meta_path, usecols=["y"])
    return int(df["y"].max()) + 1


# ── Dataset ───────────────────────────────────────────────────────────────────
class IWildCamDataset(Dataset):
    """
    Loads iWildCam images from a metadata CSV + flat image directory.
    Works identically for the mini and full dataset — only the paths differ.

    Each item: (image_tensor, label, location_id)
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

    # -- core interface --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["filename"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["y"]), int(row["location_remapped"])

    # -- convenience -----------------------------------------------------------
    @property
    def labels(self) -> np.ndarray:
        return self.df["y"].to_numpy()

    @property
    def locations(self) -> np.ndarray:
        return self.df["location_remapped"].to_numpy()


# ── DataLoader factory ────────────────────────────────────────────────────────
def get_dataloader(
    split: str,
    config: dict,
    transform=None,
    weighted_sampler: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for the given split.

    Args:
        split:            one of VALID_SPLITS
        config:           project config dict (uses dataset_mode, batch_size, …)
        transform:        override the default split transform
        weighted_sampler: inverse-frequency class sampling (train only)
    """
    meta_path, img_dir = _get_paths(config)

    if transform is None:
        transform = get_transforms(split, config.get("image_size", 224))

    dataset = IWildCamDataset(meta_path, img_dir, split, transform)

    shuffle = split == "train" and not weighted_sampler
    sampler = _make_weighted_sampler(dataset) if (weighted_sampler and split == "train") else None

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
    labels       = dataset.labels
    class_counts = np.bincount(labels, minlength=labels.max() + 1).clip(min=1)
    weights      = (1.0 / class_counts)[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(labels),
        replacement=True,
    )
