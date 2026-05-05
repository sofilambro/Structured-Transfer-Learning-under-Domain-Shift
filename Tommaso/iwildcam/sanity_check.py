"""
Lightweight sanity check for the iWildCam pipeline.

Every check is bounded to a single batch — no full-dataset iteration.
Safe to run on a laptop.

Run from the project root:
    python iwildcam/sanity_check.py
"""

import sys
import traceback
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from config  import CONFIG
from data    import get_dataloader, get_num_classes, VALID_SPLITS, _get_paths
from models  import build_model, VALID_STRATEGIES
from budget  import TimeBudgetTracker, EpochBudgetTracker, StepBudgetTracker
from utils   import set_seed, Timer, save_checkpoint, load_checkpoint

# ── Test config: mini only, tiny batch, no background workers ─────────────────
CFG = {
    **CONFIG,
    "dataset_mode": "mini",
    "batch_size":   4,
    "num_workers":  0,       # no subprocess workers — safe on all machines
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"


def check(name: str, fn):
    try:
        note = fn()
        tag  = f"{GREEN}PASS{RESET}"
        print(f"  [{tag}] {name}" + (f"  — {note}" if note else ""))
        return True
    except Exception as e:
        tag = f"{RED}FAIL{RESET}"
        print(f"  [{tag}] {name}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_files():
    meta, img_dir = _get_paths(CFG)
    assert meta.exists(),    f"Missing: {meta}"
    assert img_dir.exists(), f"Missing: {img_dir}"
    import pandas as pd
    n_rows = len(pd.read_csv(meta))
    n_imgs = len(list(img_dir.glob("*.jpg")))
    return f"{n_rows:,} metadata rows, {n_imgs:,} images"


def check_num_classes():
    n = get_num_classes(CFG)
    assert n > 0
    return f"max(y)+1 = {n}"


def check_dataloader(split: str):
    loader = get_dataloader(split, CFG)
    imgs, labels, locs = next(iter(loader))
    assert imgs.shape   == (CFG["batch_size"], 3, 224, 224), imgs.shape
    assert labels.shape == (CFG["batch_size"],)
    assert locs.shape   == (CFG["batch_size"],)
    return f"batch {tuple(imgs.shape)}"


def check_weighted_sampler():
    loader = get_dataloader("train", CFG, weighted_sampler=True)
    imgs, labels, _ = next(iter(loader))
    assert imgs.shape[0] == CFG["batch_size"]
    return "sampler OK"


def check_build_model(strategy: str):
    model, n_train, n_total = build_model(strategy, CFG)
    assert n_total > 0
    if strategy == "linear_probe":
        assert n_train < n_total
    elif strategy == "scratch":
        assert n_train == n_total
    pct = 100 * n_train / n_total
    return f"{n_train:,} / {n_total:,} trainable ({pct:.1f}%)"


def check_forward(strategy: str):
    model, _, _ = build_model(strategy, CFG)
    model.to(DEVICE).eval()
    loader = get_dataloader("train", CFG)
    imgs, labels, _ = next(iter(loader))
    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
    with torch.no_grad():
        logits = model(imgs)
    assert logits.shape == (CFG["batch_size"], CFG["num_classes"])
    loss = F.cross_entropy(logits, labels)
    assert loss.item() > 0
    return f"loss={loss.item():.4f}"


def check_single_batch_eval():
    """Evaluates ONE batch from id_val and ood_val — no full-split iteration."""
    from sklearn.metrics import f1_score
    import numpy as np

    model, _, _ = build_model("linear_probe", CFG)
    model.to(DEVICE).eval()

    results = {}
    for split in ["id_val", "ood_val"]:
        loader = get_dataloader(split, CFG)
        imgs, labels, _ = next(iter(loader))
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            logits = model(imgs)
        preds  = logits.argmax(1).cpu().tolist()
        truths = labels.cpu().tolist()
        acc = float(np.mean(np.array(preds) == np.array(truths)))
        results[split] = acc

    return "  ".join(f"{s}: acc={a:.2f}" for s, a in results.items())


def check_budget_trackers():
    t = TimeBudgetTracker(eval_every_s=0.0, max_time_s=999)
    t.step(0.01, 1, 0.1)
    assert t.should_eval()
    assert not t.should_stop()
    assert {"elapsed_s", "step", "epoch"} <= t.get_snapshot().keys()

    e = EpochBudgetTracker(eval_every_n=1, max_epochs=3)
    e.step(0.01, 10, 1.0)
    assert e.should_eval()
    e.step(0.01, 20, 3.0)
    assert e.should_stop()

    s = StepBudgetTracker(eval_every_n=5, max_steps=10)
    s.step(0.01, 5, 0.5)
    assert s.should_eval()
    s.step(0.01, 10, 1.0)
    assert s.should_stop()

    return "Time / Epoch / Step all OK"


def check_checkpoint():
    model, _, _ = build_model("linear_probe", CFG)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(model, optim, epoch=5, path=path)
        assert path.exists()
        model2, _, _ = build_model("linear_probe", CFG)
        epoch = load_checkpoint(path, model2)
        assert epoch == 5
        for (_, p1), (_, p2) in zip(model.named_parameters(),
                                     model2.named_parameters()):
            assert torch.allclose(p1, p2)
    return "save + load roundtrip OK"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    print(f"\n{'='*58}")
    print(f"  iWildCam pipeline — sanity check")
    print(f"  device={DEVICE}  mode={CFG['dataset_mode']}  "
          f"batch_size={CFG['batch_size']}  num_workers={CFG['num_workers']}")
    print(f"{'='*58}\n")

    passed = failed = 0

    def run(name, fn):
        nonlocal passed, failed
        if check(name, fn):
            passed += 1
        else:
            failed += 1

    print("1. Files")
    run("metadata.csv + image directory exists", check_files)
    run("get_num_classes() from metadata",       check_num_classes)

    print("\n2. DataLoaders  (1 batch each)")
    for split in VALID_SPLITS:
        run(f"split='{split}'", lambda s=split: check_dataloader(s))
    run("train  weighted_sampler=True", check_weighted_sampler)

    print("\n3. Model builds")
    for strategy in VALID_STRATEGIES:
        run(f"strategy='{strategy}'", lambda s=strategy: check_build_model(s))

    print("\n4. Forward pass  (1 batch, train split)")
    for strategy in VALID_STRATEGIES:
        run(f"strategy='{strategy}'", lambda s=strategy: check_forward(s))

    print("\n5. Single-batch eval  (id_val + ood_val, 1 batch each)")
    run("Evaluator — 1 batch per split", check_single_batch_eval)

    print("\n6. BudgetTrackers")
    run("Time / Epoch / Step", check_budget_trackers)

    print("\n7. Checkpoint")
    run("save + load roundtrip", check_checkpoint)

    print(f"\n{'='*58}")
    print(f"  {passed} passed   {failed} failed")
    print(f"{'='*58}\n")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
