"""
TinyImageNet data loading, restricted to a 100-class subset (paper, Experiment 1).

TinyImageNet ships as 200 classes of 64x64 images, 500 train and 50 validation
each, laid out as ::

    tiny-imagenet-200/
      wnids.txt                     200 WordNet ids, one per line
      words.txt                     wnid <TAB> human-readable label
      train/<wnid>/images/*.JPEG
      val/images/*.JPEG
      val/val_annotations.txt       filename <TAB> wnid <TAB> bbox...

The experiment never trains on all 200 at once: a run works on subset A or B (see
:mod:`.splits`), so a dataset here always carries exactly 100 classes and remaps
their wnids to contiguous indices ``0..99``. That remapping is why a source
network's classifier is meaningless on the target subset, and hence why the head
is always ``S``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

#: TinyImageNet channel statistics. Computed over the full 200-class training
#: set, so both subsets share one normalization and a transferred block never
#: sees a shifted input scale.
_MEAN = [0.4802, 0.4481, 0.3975]
_STD  = [0.2770, 0.2691, 0.2821]


def get_transforms(split: str, image_size: int = 64) -> transforms.Compose:
    """
    Random crop with reflect padding plus horizontal flip for training.

    The standard small-image recipe. No resizing: 64x64 is the native resolution
    and the WRN is configured for it directly.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=8, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def read_wnids(root: str | Path) -> list[str]:
    """Read the 200 class ids from ``wnids.txt``, in file order."""
    with open(Path(root) / "wnids.txt", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_words(root: str | Path) -> dict[str, str]:
    """Read ``words.txt`` into ``{wnid: human-readable label}``."""
    labels: dict[str, str] = {}
    with open(Path(root) / "words.txt", encoding="utf-8") as handle:
        for line in handle:
            if "\t" not in line:
                continue
            wnid, name = line.rstrip("\n").split("\t", 1)
            labels[wnid] = name
    return labels


class TinyImageNetSubset(Dataset):
    """
    A 100-class subset of TinyImageNet.

    Args:
        root:   the ``tiny-imagenet-200`` directory.
        wnids:  the 100 class ids in this subset. Sorted internally, so label
                indices depend only on the id set and not on argument order --
                a run is reproducible regardless of how the subset was built.
        split:  ``"train"`` or ``"val"``.
        transform: torchvision transform applied to each image.

    Each item is ``(image_tensor, label)`` with ``label`` in ``0..99``.
    """

    def __init__(
        self,
        root: str | Path,
        wnids: list[str],
        split: str = "train",
        transform=None,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"Unknown split {split!r}. Use 'train' or 'val'.")

        self.root      = Path(root)
        self.split     = split
        self.transform = transform
        self.wnids     = sorted(wnids)
        self.wnid_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}

        self.samples: list[tuple[Path, int]] = (
            self._index_train() if split == "train" else self._index_val()
        )
        if not self.samples:
            raise RuntimeError(
                f"No images found for split={split!r} under {self.root}. "
                f"Expected the tiny-imagenet-200 layout -- run "
                f"scripts/tinyimagenet_prepare.py to download and verify it."
            )

    def _index_train(self) -> list[tuple[Path, int]]:
        samples = []
        for wnid in self.wnids:
            img_dir = self.root / "train" / wnid / "images"
            for path in sorted(img_dir.glob("*.JPEG")):
                samples.append((path, self.wnid_to_idx[wnid]))
        return samples

    def _index_val(self) -> list[tuple[Path, int]]:
        """
        Index the validation split via ``val_annotations.txt``.

        Validation images live in one flat directory, so the annotation file is
        the only source of labels. Images belonging to the *other* subset are
        skipped, which is what makes a 100-class subset's validation set 5,000
        images rather than 10,000.
        """
        annotations = self.root / "val" / "val_annotations.txt"
        samples = []
        with open(annotations, encoding="utf-8") as handle:
            for line in handle:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                filename, wnid = parts[0], parts[1]
                if wnid not in self.wnid_to_idx:
                    continue          # belongs to the complementary subset
                samples.append(
                    (self.root / "val" / "images" / filename, self.wnid_to_idx[wnid])
                )
        return sorted(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        # Some TinyImageNet files are greyscale; convert unconditionally so every
        # tensor is 3-channel.
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def labels(self) -> np.ndarray:
        return np.array([label for _, label in self.samples])


class CorruptedValSet(Dataset):
    """
    A TinyImageNet-C-style corrupted validation set, read from its manifest.

    Built by :func:`..corruptions.build_corrupted_set`. Reads the same
    ``(image, label)`` interface as :class:`TinyImageNetSubset`, so the evaluator
    treats it as just another split.
    """

    def __init__(self, manifest_path: str | Path, transform=None):
        import csv

        self.manifest_path = Path(manifest_path)
        self.image_dir     = self.manifest_path.parent
        self.transform     = transform

        with open(self.manifest_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.samples = [
            (self.image_dir / row["corrupted_filename"], int(row["class"]))
            for row in rows
        ]
        self.meta = rows   # corruption name and severity, for per-slice analysis

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloader(
    dataset: Dataset,
    config: dict,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap a dataset in a DataLoader using the config's batch/worker settings."""
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        # Training only: keeps BatchNorm away from a size-1 final batch.
        drop_last=shuffle,
    )


def load_subset_wnids(path: str | Path) -> list[str]:
    """Read a subset's class ids from the text file written by the prepare script."""
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def save_subset_wnids(wnids: list[str], path: str | Path) -> None:
    """
    Persist a subset's class ids, one per line.

    Committing these makes the A/B split reproducible independently of the
    WordNet version installed -- see the caveat in :mod:`.splits`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(wnids)) + "\n", encoding="utf-8")


def compute_channel_stats(dataset: Dataset, max_images: int = 10_000) -> tuple[list, list]:
    """
    Recompute per-channel mean/std, for verifying the constants above.

    Not used at training time -- :data:`_MEAN` and :data:`_STD` are fixed so that
    every run and both subsets share one normalization.
    """
    total = torch.zeros(3)
    total_sq = torch.zeros(3)
    n = 0
    to_tensor = transforms.ToTensor()
    for i in range(min(len(dataset), max_images)):
        path, _ = dataset.samples[i]
        px = to_tensor(Image.open(path).convert("RGB"))
        total    += px.mean(dim=(1, 2))
        total_sq += (px ** 2).mean(dim=(1, 2))
        n += 1
    mean = total / n
    std  = (total_sq / n - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()
