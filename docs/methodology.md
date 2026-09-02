# Methodology

The design rationale behind the F/T/S study. This merges the project's original
planning documents (`MindMap.md`, `RoadMap.md`, `StateOfTheProject.md`) into one
current record, updated to describe the code that actually shipped. For the
formal statement see Section 3 of the paper; for hypothesis-by-hypothesis results
see [hypotheses.md](hypotheses.md).

---

## 1. The question

Transfer learning is usually discussed as a binary: freeze the backbone, or
fine-tune it. That framing hides a more informative question. Convolutional
networks build hierarchical representations — edges, then motifs, then parts,
then class-specific detectors — so different depths plausibly want different
treatment. Which depths should be **reused**, which **adapted**, and which
**relearned**?

We make that question answerable by assigning every backbone block one of three
states and sweeping the space exhaustively.

| State | Meaning | Weights | Gradient |
|-------|---------|---------|----------|
| **F** — Frozen | reuse | copied from source | none |
| **T** — Fine-tuned | adapt | copied from source | yes |
| **S** — Scratch | relearn | randomly reinitialized | yes |

## 2. Why monotonic partitions only

We restrict the space to **monotonic** configurations `F^a T^b S^c` with
`a + b + c = L`: frozen blocks form a lower prefix, fine-tuned blocks an
adaptation region, scratch blocks a suffix beneath the head.

This is a control, not a convenience. The excluded orderings are excluded for
reasons:

| Excluded pattern | Why |
|---|---|
| Inverse monotonic (`S → T → F`) | Scratch blocks feed pretrained ones, so frozen layers receive inputs from a distribution they were never trained to process. Training is unstable for reasons that have nothing to do with transfer. |
| Any `S` below an `F`/`T` | Same failure, localized: it breaks the assumption that a pretrained mapping operates on compatible features. |
| Interleaved (`F → S → T`, `T → F → S`) | Multiple representation shifts along depth. A drop can no longer be attributed to transfer rather than to architecture-induced mismatch. |
| Reinit islands (`… T … S … T …`) | Localized distribution breaks that the surrounding fine-tuned blocks must reconcile — confounded and unstable. |

The monotonic form also gives each region a clean interpretation: the frozen
prefix measures how much source representation is reusable, the fine-tuned region
measures how much adaptation is needed, the scratch suffix measures whether
high-level features should be relearned — and `T` doubles as a transition buffer
between fixed source features and new target features. That buffer role is
itself testable, and is what hypothesis H4 asks about.

The space has size `(L+1)(L+2)/2`: **15** configurations for the four-block
WRN-28-10, **21** for the five-block ResNet50.

## 3. Why blocks are the right unit

Cuts are made at **stage boundaries**, not at arbitrary layer counts, and not
inside residual groups.

Each stage boundary is a genuine functional boundary: spatial resolution drops,
channel width doubles, and the abstraction level shifts. Yosinski et al. (2014)
showed transferability degrades with depth as a property of what features *are* —
edges → parts → class detectors — not as a function of how many weights encode
them. Cutting at stage boundaries therefore has a clear interpretation; cutting
inside a stage would sever residual co-adaptation with no semantic justification.

This has a direct consequence for how results should be read. The parameter
distribution is severely skewed — ResNet50's Stage4 alone holds 63% of the
weights — so **trainable parameter count is a poor proxy for configuration
identity**. Freezing the first two blocks changes the trainable share by under
1%, yet is a completely different experiment from freezing four. Worse, the two
backbones skew in *opposite* directions relative to compute:

| | Parameters | Forward FLOPs |
|---|---|---|
| ResNet50 Stage4 | 62.7% | 19.7% |
| WRN-28-10 Group3 | 76.3% | 35.9% |

Early stages are cheap in parameters but expensive in FLOPs, because they run at
large spatial resolutions. So the primary structural variable is the **block
assignment tuple**; `trainable_pct` is reported as context only, and compute is
reported as FLOPs and wall-clock time.

## 4. Two experiments, two regimes

The study deliberately separates *mechanism discovery* from *stress testing*.

### Experiment 1 — controlled (TinyImageNet, WRN-28-10)

Both domains are carved from one dataset and **both source networks are trained
by us from scratch**. That control is what lets two normally-tangled effects be
separated:

- **Co-adaptation fragility** — visible in the *selfer* runs (A→A, B→B), where
  the source and target label spaces are identical. Any degradation there cannot
  be domain mismatch; it is optimization.
- **Feature specialization** — visible only in the *transfer* runs (A→B, B→A),
  where deep features are copied across disjoint label sets.

The A/B class split is semantically stratified across six WordNet-derived groups
rather than drawn uniformly, so the two halves are comparable in difficulty. A
uniform draw could land most of the animals in one half and make "transfer
degradation" an artifact of one subset simply being harder. The symmetry check
(paper Figure 2; Pearson r = 0.996 transfer, 0.971 selfer) is what confirms the
stratification worked.

### Experiment 2 — realistic (iWildCam, ImageNet-pretrained ResNet50)

An external pretrained source, a visually and semantically different target, and
real location shift: OOD examples come from camera-trap sites never seen in
training, which changes background, illumination, geography, viewpoint, class
priors and animal appearance simultaneously.

### What the pairing buys

Neither experiment is sufficient alone. Experiment 1 can isolate mechanisms but
its two domains come from the same 200-class pool, so its shift is mild.
Experiment 2 has a realistic shift but tangles several shifts together, making
causes hard to isolate. Read together, they show the *direction* of the effect:
as source–target mismatch grows, the preferred strategy moves from reuse toward
adaptation.

## 5. Design constraints

Fixed across every configuration within an experiment, so that a difference in
the leaderboard is attributable to the partition:

- same backbone architecture and same block decomposition
- same optimizer, learning-rate schedule and augmentation
- same evaluation splits and metrics
- hyperparameters tuned once on a reference configuration, then frozen
- the classifier head is **always** `S` — the target label space differs from the
  source, so copied classifier weights would be noise

The budget differs by design between the two:

|  | Experiment 1 | Experiment 2 |
|---|---|---|
| Budget | 100 epochs, fixed | wall clock, ~185 min |
| Consequence | No run is truncated; fully controlled | Slow partitions complete fewer epochs |

The iWildCam choice is a concession to a shared cluster: equal wall clock is what
makes 21 jobs schedulable. Its cost is visible in the results — T5S1 is the
slowest partition at ~592 s/epoch and reached only 19.6 epochs, against 35–44
for cheaper partitions. See [`results/iwildcam/README.md`](../results/iwildcam/README.md).

## 6. Metrics, and why not raw accuracy

| ID | Metric | Role |
|----|--------|------|
| M1 | ID accuracy | in-domain task performance |
| M2 | OOD accuracy | performance under shift |
| M3 | ID macro-F1 | per-class performance, in-domain |
| M4 | OOD macro-F1 | per-class performance under shift |
| M5 | Training FLOPs | hardware-independent training cost |
| M6 | Inference FLOPs | deployment cost (constant across partitions) |
| — | Trainable parameters | context only, see §3 |

**iWildCam rankings use OOD balanced accuracy** (equivalently macro recall).
iWildCam is long-tailed: a model that only ever predicts the common species still
scores respectably on raw accuracy. Macro-F1 is reported alongside as the
standard WILDS-comparable metric, most sensitive to rare classes. Raw accuracy is
kept for continuity with the published leaderboard.

**TinyImageNet uses plain top-1**, because the subsets are uniform by
construction — 500 train / 50 validation images per class — so balanced accuracy
would be identical up to sampling noise.

Training FLOPs are estimated as `forward × (1 + 2 × trainable_ratio)`: a backward
pass costs roughly twice a forward pass, but only through the trainable portion.
The approximation is crude — it attributes cost by parameter share rather than by
activation size, so it understates early stages — which is why wall-clock time is
logged beside it.

## 7. Known confounders and what was done about them

| | Confounder | Mitigation |
|---|---|---|
| N1 | ImageNet pretraining is biased toward natural images | S6, the no-pretraining baseline, is in the sweep |
| N2 | iWildCam tangles camera, location and illumination shift | Experiment 1 separates the mechanisms |
| N3 | Long-tailed class distribution | Balanced accuracy and macro-F1; inverse-frequency training sampler |
| N4 | Vision classification only | Conclusions scoped explicitly |
| N5 | Two CNN families only | Acknowledged; ViT / self-supervised backbones are future work |
| N6 | Hyperparameter sensitivity | One protocol, tuned once on a reference configuration |
| N7 | Seed and split variance | Not mitigated: a single split and a single seed. The principal limitation of the study |
| N8 | Small effect sizes | Focus on consistent trends across both experiments, not single-run gains |

N7 is the one that matters most. The A→B / B→A symmetry check gives some
confidence that the split is not pathological, but it is not a substitute for
seed-to-seed variance. Differences of one or two points in the leaderboard should
not be over-read.

## 8. Design decisions worth recording

| Decision | Rationale |
|---|---|
| CSV-based iWildCam loading, not the WILDS dataset object | Works identically for the mini and full datasets; the WILDS object needs the full archive layout on disk |
| One `dataset_mode` flag switches mini/full | No code changes anywhere else |
| `BudgetTracker` as an abstract interface | The training loop is decoupled from cost policy, so epoch-budget and time-budget protocols share one loop |
| WILDS official eval kept as an optional cross-check | Confirms comparability with the published benchmark without blocking work on the mini subset |
| Mild imbalance compression (α = 0.8) in the mini subset | Reduces extreme imbalance without fully equalizing; preserves a realistic distribution shape |
| Frozen BatchNorm held in `eval()` mode | Otherwise a "frozen" block is only half-frozen: weights hold still but running statistics keep absorbing target batches, so its outputs drift anyway |

The BatchNorm decision is protocol-relevant and differs between the experiments:
Experiment 1 does this (Appendix A.1), while the archived Experiment 2 sweep did
**not**. The option exists in the iWildCam config as `freeze_frozen_bn`,
defaulting to `False` so new runs stay comparable to the archived ones.

## 9. Underlying assumptions

| | Assumption | Source |
|---|---|---|
| A1 | CNN representations are hierarchical: early layers general, deep layers task-specific | Yosinski et al., NeurIPS 2014 |
| A2 | Large-scale pretrained features are broadly reusable | Donahue et al., ICML 2014 (DeCAF) |
| A3 | Transferability decreases with depth | Kornblith et al., CVPR 2019 |
| A4 | Stable training requires adjacent blocks to operate on compatible feature distributions | Yosinski et al., NeurIPS 2014 |
| A5 | Updating pretrained weights helps when domain or task mismatch is large | Girshick et al., CVPR 2014 |
| A6 | Partial fine-tuning can be competitive at lower cost | efficiency/performance trade-off literature |
| A7 | Pretrained models generalize better under shift than scratch models | Hendrycks et al., NeurIPS 2019 |
| A8 | Transfer performance is sensitive to *how* layers are frozen or fine-tuned | Neyshabur et al., NeurIPS 2020 |

A4 is the assumption the monotonicity constraint operationalizes. A7 is the one
Experiment 2 partially **contradicts**: a fully frozen ImageNet backbone (F5S1)
performs *worse* under shift than training from scratch (S6). Pretrained features
help only when they remain adaptable.
