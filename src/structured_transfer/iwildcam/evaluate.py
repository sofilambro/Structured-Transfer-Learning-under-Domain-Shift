"""
Evaluation for the iWildCam experiment.

Raw accuracy is not sufficient here: iWildCam is long-tailed, so a model that
only ever predicts the handful of common species still scores well on it. The
paper therefore ranks partitions by **OOD balanced accuracy** (equivalently macro
recall) and reports **OOD macro-F1** as the secondary check, since macro-F1 is
the standard WILDS/iWildCam metric and is the one most sensitive to rare classes.
Raw accuracy is kept for continuity with the WILDS leaderboard.

Each split result:

    {
        "acc":          float,   # top-1 accuracy
        "balanced_acc": float,   # mean per-class recall -- the primary metric
        "f1":           float,   # macro-averaged F1
        "loss":         float,   # mean cross-entropy
        "n":            int,     # examples evaluated
    }
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

from .data import VALID_SPLITS, get_dataloader, get_transforms

#: Mid-training we only evaluate the two validation splits; the test splits are
#: held back so repeated peeking cannot influence protocol decisions.
DEFAULT_EVAL_SPLITS = ["id_val", "ood_val"]


class Evaluator:
    """
    Evaluates a model on named splits, caching DataLoaders across calls.

    The cache matters: ``train()`` calls ``run()`` after every epoch, and
    rebuilding a DataLoader each time would re-read the 203k-row metadata CSV and
    respawn worker processes on every evaluation.
    """

    def __init__(self, config: dict, device: torch.device | str):
        self.config   = config
        self.device   = torch.device(device)
        self._loaders: dict = {}

    def run(
        self,
        model: torch.nn.Module,
        splits: list[str] | None = None,
    ) -> dict[str, dict]:
        """
        Evaluate ``model`` on the requested splits.

        Returns ``{split: {"acc", "balanced_acc", "f1", "loss", "n"}}``.
        """
        if splits is None:
            splits = DEFAULT_EVAL_SPLITS

        unknown = [s for s in splits if s not in VALID_SPLITS]
        if unknown:
            raise ValueError(f"Unknown split(s): {unknown}. Valid: {VALID_SPLITS}")

        model.eval()
        model.to(self.device)
        results = {}
        with torch.no_grad():
            for split in splits:
                results[split] = self._eval_split(model, split)
        return results

    def run_wilds_official(
        self,
        model: torch.nn.Module,
        splits: list[str] | None = None,
    ) -> dict[str, dict]:
        """
        Evaluate through the official WILDS ``eval()`` method.

        A cross-check that our CSV-based pipeline agrees with the published
        benchmark, not part of the main protocol. Requires ``dataset_mode='full'``
        and the ``wilds`` package.
        """
        if self.config.get("dataset_mode") != "full":
            raise RuntimeError(
                "run_wilds_official() requires dataset_mode='full'. "
                "Switch config['dataset_mode'] to 'full' and ensure the "
                "full dataset is downloaded."
            )
        if splits is None:
            splits = DEFAULT_EVAL_SPLITS

        from wilds import get_dataset as _wilds_get_dataset
        from wilds.common.data_loaders import get_eval_loader

        wilds_ds = _wilds_get_dataset(
            dataset="iwildcam", download=False, root_dir=self.config["data_dir"]
        )
        val_transform = get_transforms("val", self.config.get("image_size", 224))

        _to_wilds = {
            "train":    "train",
            "id_val":   "id_val",
            "ood_val":  "val",
            "id_test":  "id_test",
            "ood_test": "test",
        }

        model.eval()
        model.to(self.device)
        results = {}

        with torch.no_grad():
            for split in splits:
                subset = wilds_ds.get_subset(_to_wilds[split], transform=val_transform)
                loader = get_eval_loader(
                    "standard", subset, batch_size=self.config["batch_size"]
                )
                all_preds, all_labels, all_meta = [], [], []
                for x, y, meta in loader:
                    preds = model(x.to(self.device)).argmax(dim=1).cpu()
                    all_preds.append(preds)
                    all_labels.append(y)
                    all_meta.append(meta)

                y_pred   = torch.cat(all_preds)
                y_true   = torch.cat(all_labels)
                metadata = torch.cat(all_meta)
                wilds_out, _ = wilds_ds.eval(y_pred, y_true, metadata)

                results[split] = {
                    "acc":          wilds_out.get("acc_avg"),
                    "balanced_acc": None,     # not reported by WILDS
                    "f1":           wilds_out.get("F1-macro_all"),
                    "loss":         None,
                    "n":            len(y_true),
                    "wilds":        wilds_out,
                }

        return results

    def _eval_split(self, model: torch.nn.Module, split: str) -> dict:
        loader = self._get_loader(split)
        all_preds:  list[int] = []
        all_labels: list[int] = []
        total_loss = 0.0

        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = model(imgs)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        n   = len(all_labels)
        acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))

        # Macro metrics are averaged only over classes actually present in this
        # split. Label ids run to 182 globally, but no single split contains all
        # of them -- averaging over absent classes would inject spurious zeros
        # and make splits with different class coverage incomparable.
        true_classes = sorted(set(all_labels))
        bal = float(balanced_accuracy_score(all_labels, all_preds))
        f1  = float(f1_score(all_labels, all_preds, average="macro",
                             labels=true_classes, zero_division=0))

        return {"acc": acc, "balanced_acc": bal, "f1": f1,
                "loss": total_loss / n, "n": n}

    def _get_loader(self, split: str):
        if split not in self._loaders:
            # Always the deterministic transform, even for the train split: this
            # measures the model, not the augmentation pipeline.
            transform = get_transforms("val", self.config.get("image_size", 224))
            self._loaders[split] = get_dataloader(
                split, self.config, transform=transform, weighted_sampler=False
            )
        return self._loaders[split]
