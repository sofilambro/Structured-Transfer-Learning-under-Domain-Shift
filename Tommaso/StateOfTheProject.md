# State of the Project — iWildCam Transfer Learning Study

## What is this project?

A systematic comparison of four CNN fine-tuning strategies for out-of-distribution (OOD) generalisation on the iWildCam (WILDS) dataset. The core question: **how much to fine-tune, and at what cost?**

See `ROADMAP.md` for full objectives, hypotheses, and analysis plan.

---

## What has been done

### 1. Environment
- Python 3.12.9 virtual environment at `venv_ood/`
- All required packages installed: PyTorch 2.12 + CUDA 12.8, torchvision, wilds 2.0.0, pandas, matplotlib, seaborn, scikit-learn, wandb

### 2. Dataset preparation
- `iwildcam/download_metadata.py` — downloads only the iWildCam metadata CSV (~30 MB) from Codalab, bypassing the full 12 GB image archive. Creates the directory stub expected by WILDS so the full dataset can be loaded later without re-downloading.
- `iwildcam/notebooks/01_data_exploration.ipynb` — metadata-only EDA (no images required). Covers split overview, class imbalance, location distribution, ID/OOD overlap, species richness, temporal patterns.
- `iwildcam/notebooks/02_build_mini_dataset.ipynb` — builds a curated mini dataset from metadata (Part A, runs now) and streams only the needed images from the archive (Part B, ~12 GB network / ~3 GB disk). Mini dataset spec: classes with ≥100 train images and ≥20 OOD test images, subsampled to ~50 000 total images with mild imbalance compression (alpha=0.8).

### 3. Core pipeline scripts
All scripts live in `iwildcam/` and work identically for both the mini and full dataset. Switch with one config flag.

| File | Role |
|------|------|
| `config.py` | Central config dict. `dataset_mode: "mini" \| "full"` is the only switch needed to change datasets. |
| `data.py` | `IWildCamDataset` + `get_dataloader()`. Reads from metadata CSV + image folder. Supports inverse-frequency weighted sampling. |
| `models.py` | `build_model(strategy, config)` — builds ResNet-18/50 with the chosen freeze strategy. `get_param_groups()` for differential LRs (head vs backbone). |
| `evaluate.py` | `Evaluator` class. `run(model, splits)` returns `{split: {acc, f1, loss, n}}`. `run_wilds_official()` for sanity checks against the WILDS benchmark (full dataset only). |
| `budget.py` | `BudgetTracker` ABC + three concrete implementations: `TimeBudgetTracker`, `EpochBudgetTracker`, `StepBudgetTracker`. Plug-in point for compute-cost tracking. |
| `train.py` | `train(model, config, evaluator, budget_tracker)` — full training loop with mid-training evaluation hooks driven by the budget tracker. Returns `(eval_log, final_results)`. |
| `utils.py` | `set_seed`, `count_parameters`, `Timer`, `save_checkpoint`, `load_checkpoint`. |

---

## Folder structure

```
TRANSFER LEARNING/
│
├── ROADMAP.md                        project objectives and hypotheses
├── README.md                         this file
│
├── venv_ood/                         Python virtual environment (do not commit)
│
├── data/                             all datasets land here (do not commit)
│   ├── iwildcam_v2.0/
│   │   ├── metadata.csv              downloaded by download_metadata.py
│   │   ├── RELEASE_v2.0.txt          stub created by download_metadata.py
│   │   └── train/                    full images (not downloaded yet)
│   └── iwildcam_mini/
│       ├── metadata.csv              built by notebook 02, Part A
│       └── train/                    mini images (built by notebook 02, Part B)
│
├── iwildcam/                         project source
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── evaluate.py
│   ├── budget.py
│   ├── train.py
│   ├── utils.py
│   ├── download_metadata.py
│   ├── results/                      plots saved here by notebooks
│   ├── checkpoints/                  model checkpoints saved here by train.py
│   └── notebooks/
│       ├── 01_data_exploration.ipynb
│       └── 02_build_mini_dataset.ipynb
│
└── poc_cifar10/                      earlier CIFAR-10 proof-of-concept (reference only)
```

---

## How to run (in order)

```powershell
# Activate environment
& "venv_ood\Scripts\Activate.ps1"

# 1. Download metadata (~30 MB, one-time)
python iwildcam\download_metadata.py

# 2. Run EDA notebook
#    Open iwildcam/notebooks/01_data_exploration.ipynb, select venv_ood kernel, Run All

# 3. Build mini dataset
#    Open iwildcam/notebooks/02_build_mini_dataset.ipynb
#    Part A (metadata + plots): run now
#    Part B (image download, ~12 GB stream → ~3 GB disk): run when ready

# 4. Run an experiment (example)
python - <<'EOF'
import torch
from iwildcam.config import CONFIG
from iwildcam.models import build_model
from iwildcam.evaluate import Evaluator
from iwildcam.budget import EpochBudgetTracker
from iwildcam.train import train

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, n_train, n_total = build_model("full_ft", CONFIG)
evaluator = Evaluator(CONFIG, device)
tracker   = EpochBudgetTracker(eval_every_n=1, max_epochs=CONFIG["epochs"])
log, results = train(model, CONFIG, evaluator, tracker)
EOF
```

---

## Key design decisions (rationale)

| Decision | Rationale |
|----------|-----------|
| CSV-based data loading (not WILDS dataset object) | Works identically for mini and full; WILDS object requires full archive structure |
| Single `dataset_mode` flag in config | One-line switch, no code changes needed anywhere else |
| `BudgetTracker` as abstract interface | Training loop is decoupled from cost-tracking logic; future trackers (FLOPs, memory) slot in without touching `train.py` |
| WILDS official eval as optional sanity check | Ensures comparability with published results; kept separate so it doesn't block mini-dataset work |
| Mild imbalance compression (alpha=0.8) in mini dataset | Reduces extreme imbalance without fully equalising; preserves realistic distribution shape |
