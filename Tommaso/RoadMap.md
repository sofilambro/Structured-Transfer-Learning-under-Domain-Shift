PROJECT SUMMARY: Structured Transfer Learning Study (CNNs – ResNet50)

1. OBJECTIVE
Systematically study transfer learning strategies using frozen (F), fine-tuned (T), and
scratch (S) layer partitions on ResNet50. Evaluate impact on:
- In-domain (ID) performance
- Out-of-domain (OOD) generalization
- Computational cost (FLOPs, trainable parameters)

Two-stage design:
  Stage A — controlled dataset (CIFAR100 or ImageNet subset): isolate the effect of
             F/T/S configurations from dataset-specific noise (mechanism discovery).
  Stage B — iWildCam (WILDS): stress-test the same configurations under real-world
             distribution shift (robustness validation).

--------------------------------------------------

2. CORE RESEARCH QUESTION
How does the choice of F/T/S partition (monotonic configurations of ResNet50 blocks)
affect the trade-off between ID accuracy, OOD robustness, and training cost, across
controlled and real-world distribution-shift settings?

--------------------------------------------------

3. HYPOTHESES

H1  (Early layers generality): Freezing early layers preserves useful low-level features
    with minimal performance loss.
H2  (Top layers specificity): Reinitializing top layers from scratch is necessary when
    the label space changes.
H3  (Middle layers plasticity): Fine-tuning middle layers yields the largest marginal gains.
H4  (Bridging with fine-tuning): A fine-tuned buffer between frozen and scratch regions
    stabilizes optimization (mitigates co-adaptation fragility).
H5  (ID performance): Properly configured transfer learning improves ID accuracy over
    scratch training, especially in low/medium data regimes.
H6  (OOD performance): Transfer learning improves OOD generalization via reuse of
    robust pretrained features.
H7  (Efficiency): Comparable or better performance can be achieved at lower FLOPs /
    fewer trainable parameters.
H8  (Fragility): Poorly chosen F/T/S partitions degrade performance; the method is
    sensitive to configuration.
H9  (Regime dependence): Optimal configurations depend on data size and domain shift;
    more fine-tuning is favored as shift increases.
H10 (Diminishing returns): Extensive fine-tuning of early layers yields limited gains
    relative to its cost.
H11 (Scalability across depth): Yosinski et al. (2014) results on shallow networks are
    expected to hold for ResNet50.
H12 (Large-scale regime): With the optimal F/T/S partition, transfer learning can still
    improve performance even in larger data regimes.

--------------------------------------------------

4. DATASETS

--- Stage A: Controlled dataset (mechanism discovery) ---
Dataset: CIFAR100 or ImageNet subset (TBD)
Purpose: isolate architectural effects; minimize dataset-specific confounders.

--- Stage B: iWildCam (WILDS) (robustness validation) ---
Task:        Animal species classification
Input:       Camera trap images
Domain:      Camera location
Key property: Strong real-world distribution shift across locations
Splits:
  - ID (train / val): seen camera locations
  - OOD (test):       unseen camera locations

Potential confounders (iWildCam-specific):
  - Class imbalance / long-tail distribution (N3)
  - Multiple overlapping shifts: camera, location, illumination (N2)
  - ImageNet pretraining bias toward natural images (N1)
Mitigations: balanced accuracy / macro F1; complement with Stage A results.

--------------------------------------------------

5. MODEL SETUP

Backbone:        ResNet50 pretrained on ImageNet
Head:            Always S (reinitialized + trained); global pooling + FC layer
Constraint:      Same backbone, same head, same preprocessing pipeline across all configs

Block decomposition (6 blocks):
  Stem   — initial conv + pooling; low-level features (edges, textures)
  Stage1 — early residual blocks; simple patterns and local structures
  Stage2 — mid-level feature extraction; primitives into motifs and parts
  Stage3 — higher-level representations; abstract and semantic features
  Stage4 — top residual blocks; highly task-specific features
  Head   — global pooling + FC; maps features to class logits (always S)

--------------------------------------------------

6. CONFIGURATION SPACE

Each block ∈ {F (frozen), T (fine-tuned), S (scratch)}.

Constraint: monotonic configurations of the form  F^a  T^b  S^c
with Head fixed as S.  →  21 valid configurations total.

Rationale for monotonic ordering (F → T → S):
  - Aligns with CNN feature hierarchy (general → specific).
  - Prevents scratch blocks from feeding into frozen pretrained blocks
    (avoids out-of-distribution inputs to frozen layers).
  - Fine-tuned blocks act as a transition buffer between frozen and scratch regions.

Excluded (and why):
  - Inverse monotonic (S → T → F): feature incompatibility; unstable training.
  - S below pretrained layers: breaks pretrained mapping assumptions.
  - Interleaved / non-monotonic: repeated distribution misalignment; confounded effects.

--------------------------------------------------

7. EVALUATION METRICS

Performance:
  M1 ID Accuracy   — top-1 accuracy on in-domain data (balanced accuracy preferred)
  M2 OOD Accuracy  — top-1 accuracy on shifted data (balanced accuracy preferred)
  M3 ID F1         — macro-averaged F1 on in-domain data
  M4 OOD F1        — macro-averaged F1 on shifted data

Compute:
  M5 Training FLOPs   — forward + backward over full training set
  M6 Inference FLOPs  — forward pass cost on test set
  Extra: trainable parameter count (contextualizes cost and flexibility)

--------------------------------------------------

8. EXPERIMENTAL DESIGN CONSTRAINTS

- Same backbone architecture (ResNet50) across all 21 configurations
- Same number of training epochs / steps
- Same optimizer and learning rate schedule
- Same data augmentations
- Hyperparameters tuned once on a reference configuration; fixed across all others (N6)
- Multiple random seeds on key configurations to assess stability (N7)

--------------------------------------------------

9. ANALYSIS PLAN

- Compare ID / OOD accuracy and F1 across all 21 F/T/S configurations
- Evaluate performance vs compute (FLOPs / trainable params) trade-off
- Identify:
  - whether more fine-tuning improves ID at the expense of OOD
  - whether a fine-tuned buffer (T block) stabilizes training vs direct F→S transitions
  - optimal configurations per data-size regime (V2) and domain-shift level (V3)
- Cross-validate Stage A trends on iWildCam (Stage B)

Plots:
  - ID vs OOD scatter (one point per configuration)
  - Performance vs training FLOPs
  - Heatmap: configuration × metric across dataset sizes
  - Per-class OOD breakdown (do rare classes suffer more?)

--------------------------------------------------

10. EXPECTED CHALLENGES / NOISY FACTORS

  N1 ImageNet pretraining bias — include no-pretraining baseline
  N2 iWildCam overlapping shifts — rely on Stage A to separate effects
  N3 Class imbalance (long-tail) — use balanced accuracy and macro F1
  N4 Task specificity — scope conclusions to vision classification
  N5 Architecture choice (ResNet50 only) — justify as representative baseline
  N6 Hyperparameter sensitivity — fix protocol; tune once on reference config
  N7 Stochasticity — multiple seeds; report mean ± std on key configs
  N8 Small effect sizes at scale — use statistical testing; focus on consistent trends

--------------------------------------------------

11. DELIVERABLES

- Comparative evaluation of all 21 F/T/S configurations
- Quantitative analysis of trade-offs (performance, compute, OOD robustness)
- Conclusions on:
  - which configurations are Pareto-optimal
  - when and how much to fine-tune
  - how findings scale from controlled to real-world settings

--------------------------------------------------

12. ONE-LINE SUMMARY

This project systematically evaluates all 21 monotonic F/T/S partitions of ResNet50
to understand how freezing, fine-tuning, and reinitialization affect in-domain accuracy,
OOD generalization, and computational cost, validated on both a controlled dataset and
iWildCam.

--------------------------------------------------

13. TODO LIST

Status legend:  [x] done   [~] in progress   [ ] not started

--- Phase 0: Design & Refactor (new) ---
[ ] Finalize controlled dataset choice (CIFAR100 vs ImageNet subset) for Stage A
[ ] Update models.py — replace 4-strategy build_model() with F/T/S partition API
    (accepts a config vector over [Stem, Stage1, Stage2, Stage3, Stage4, Head])
[ ] Update config.py — add dataset_mode for controlled dataset + 21 config enumeration
[ ] Update data.py — add loader for controlled dataset (Stage A)
[ ] Update evaluate.py / train.py — ensure metrics include F1 and FLOPs logging

--- Phase 1: Data & Environment ---
[x] Set up Python environment (venv_ood, PyTorch + CUDA, wilds, torchvision)
[x] Write download_metadata.py (metadata-only download, ~30 MB)
[x] Write 01_data_exploration.ipynb (metadata EDA, no images)
[x] Write 02_build_mini_dataset.ipynb (class filter + subsample + streaming image download)
[ ] RUN: download_metadata.py
[ ] RUN: 01_data_exploration.ipynb — inspect class imbalance, domain structure, temporal patterns
[ ] RUN: 02_build_mini_dataset.ipynb Part A — build and save iwildcam_mini/metadata.csv
[ ] RUN: 02_build_mini_dataset.ipynb Part B — stream and save mini images (~3 GB disk)

--- Phase 2: Pipeline ---
[x] config.py — central config with dataset_mode switch (mini | full)
[x] data.py — IWildCamDataset + get_dataloader (CSV-based, same API for mini and full)
[x] models.py — build_model() for all 4 strategies + get_param_groups()
[x] evaluate.py — Evaluator with run() + run_wilds_official() sanity check
[x] budget.py — BudgetTracker interface + Time / Epoch / Step implementations
[x] train.py — training loop with mid-training eval hooks
[x] utils.py — seed, timer, param count, checkpointing
[ ] Smoke-test pipeline on controlled dataset (1 epoch, sample of configs, check no crashes)
[ ] Verify eval_log and final_results shapes include F1 and FLOPs fields

--- Phase 3: Stage A — Controlled Dataset Experiments ---
[ ] Decide fixed compute budget (same epochs for all 21 configurations)
[ ] Run all 21 F/T/S configurations on controlled dataset
[ ] Run no-pretraining baseline (all S, random init) — reference for H5, H6
[ ] Collect eval_log + final_results (ID/OOD accuracy, F1, FLOPs, trainable params)
[ ] Sanity-check: reproduce expected trend (more F → lower cost; more S → higher cost)
[ ] Identify Pareto-optimal configs on Stage A data

--- Phase 4: Stage B — iWildCam Experiments ---
[ ] RUN: 02_build_mini_dataset.ipynb Part B (or use cloud/cluster for full dataset)
[ ] Re-run top configurations + no-pretraining baseline on iWildCam
[ ] Re-run all 21 configurations on iWildCam (if compute allows; otherwise top-K)
[ ] Verify Stage A trends are consistent with iWildCam results

--- Phase 5: Analysis & Plots ---
[ ] ID vs OOD scatter plot (one point per configuration, both stages)
[ ] Performance vs training FLOPs curves
[ ] Heatmap: configuration × metric (ID acc, OOD acc, ID F1, OOD F1)
[ ] Trainable parameters vs OOD accuracy
[ ] Per-class breakdown: does OOD drop hit rare classes harder?
[ ] Write 03_results_analysis.ipynb

--- Phase 6: Conclusions ---
[ ] Evaluate each hypothesis (H1–H12) against results
[ ] Write summary of optimal configurations per regime (data size, domain shift)
[ ] Final report / write-up
