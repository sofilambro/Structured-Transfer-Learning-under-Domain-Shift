"""
Evaluator — standard evaluation pipeline for iWildCam experiments.

Usage (from a notebook or train.py):

    evaluator = Evaluator(config, device)

    # default: id_val + ood_val
    results = evaluator.run(model)

    # custom splits
    results = evaluator.run(model, splits=["train", "id_val", "ood_val", "id_test", "ood_test"])

    # sanity-check against WILDS official metrics (full dataset only)
    wilds_results = evaluator.run_wilds_official(model, splits=["id_val", "ood_val"])

Each split result:
    {
        "acc":          float,   # top-1 accuracy
        "balanced_acc": float,   # mean per-class recall (preferred for imbalanced data)
        "f1":           float,   # macro-averaged F1
        "loss":         float,
        "n":            int,
    }
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score

from data import get_dataloader, get_transforms, VALID_SPLITS

DEFAULT_EVAL_SPLITS = ["id_val", "ood_val"]


class Evaluator:
    def __init__(self, config: dict, device: torch.device | str):
        self.config  = config
        self.device  = torch.device(device)
        self._loaders: dict = {}   # lazy-loaded, reused across calls

    # ── Primary evaluation ────────────────────────────────────────────────────
    def run(
        self,
        model: torch.nn.Module,
        splits: list[str] | None = None,
    ) -> dict[str, dict]:
        """
        Evaluate model on the requested splits.

        Returns:
            {split: {"acc", "balanced_acc", "f1", "loss", "n"}}
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

    # ── WILDS official sanity check ───────────────────────────────────────────
    def run_wilds_official(
        self,
        model: torch.nn.Module,
        splits: list[str] | None = None,
    ) -> dict[str, dict]:
        """
        Evaluate using the WILDS official eval() method.
        Requires dataset_mode='full' and the full dataset on disk.
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
                    "balanced_acc": None,
                    "f1":           wilds_out.get("F1-macro_all"),
                    "loss":         None,
                    "n":            len(y_true),
                    "wilds":        wilds_out,
                }

        return results

    # ── Internal helpers ──────────────────────────────────────────────────────
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

        n            = len(all_labels)
        acc          = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        true_classes = sorted(set(all_labels))   # restrict to classes present in this split
        bal = float(balanced_accuracy_score(all_labels, all_preds))
        f1  = float(f1_score(all_labels, all_preds, average="macro",
                             labels=true_classes, zero_division=0))
        return {"acc": acc, "balanced_acc": bal, "f1": f1, "loss": total_loss / n, "n": n}

    def _get_loader(self, split: str):
        if split not in self._loaders:
            transform = get_transforms("val", self.config.get("image_size", 224))
            self._loaders[split] = get_dataloader(
                split, self.config, transform=transform, weighted_sampler=False
            )
        return self._loaders[split]
