# Structured Transfer Learning Study (CNNs – ResNet50)

## 1. Objective

Systematically study transfer learning strategies using frozen, fine-tuned, and scratch layers. Evaluate impact on in-domain (ID) performance, out-of-domain (OOD) generalization, and computational cost (FLOPs, ????). 

## 2. Experiment 1

Partition the network into three contiguous regions:

[FROZEN] → [FINE-TUNED] → [SCRATCH]

Rationale for monotonic order (F → T → S):

* This ordering is a **strong default**, not the only valid one. It aligns with the typical hierarchy of CNN features (general → specific) and tends to preserve useful pretrained representations while adapting higher-level features.
* The main motivation is to **limit feature mismatch**: reinitialized layers (S) are placed at the top so they do not feed into frozen pretrained blocks, which would otherwise receive out-of-distribution inputs.
* Fine-tuned layers (T) provide a **transition region** that adapts feature distributions between frozen and scratch parts, improving stability.

Why monotonic (F → T → S) and exclusion of alternatives:

* **Monotonic (F → T → S)**: preserves pretrained features (F), introduces a controlled adaptation buffer (T), and confines reinitialization (S) to the top, avoiding upstream disruption.

* **Inverse monotonic (S → T → F)**: scratch blocks feed into pretrained layers → **feature incompatibility** (pretrained layers receive out-of-distribution inputs) → unstable or ineffective training.

* **S below pretrained (… S … F/T …)**: any placement of S under F/T breaks the assumption that pretrained mappings operate on compatible features → **mismatch + gradient conflict**.

* **Interleaved / non-monotonic (e.g., F → S → T, T → F → S)**: multiple distribution shifts along depth → **repeated misalignment** and hard-to-interpret effects.

* **Early fine-tune then freeze (T → F → S)**: adapted features must pass through a fixed downstream mapping → **bottlenecked adaptation**; gains are constrained and sensitive to initialization.

* **All fine-tuned with local reinit islands (… T … S … T …)**: islands of S create **localized distribution breaks** that upstream/downstream T must reconcile → unstable and confounded effects.

Underlying assumptions:

* Pretrained features are useful and should not be perturbed by random transformations below them.
* **Feature compatibility across adjacent blocks** is required for stable optimization.

Scope decision:

* To maximize **control and interpretability**, we restrict to monotonic configurations and exclude alternatives that introduce feature mismatch or confounded adaptation patterns.

## 3. Model

Backbone: ResNet50 pretrained on ImageNet

Rationale for backbone choice:

* The goal of this study is not to achieve state-of-the-art performance, but to obtain results that are interpretable and generalizable.
* ResNet is a standard and widely adopted architecture, making results easier to compare with existing literature.
* ResNet50 provides a balanced tradeoff between shallow models (e.g., ResNet18) and deeper ones (e.g., ResNet101), offering sufficient representational capacity without excessive computational cost.
* Its clear stage-wise structure makes it particularly suitable for controlled manipulation of frozen, fine-tuned, and scratch blocks.

Rationale for ImageNet pretraining:

* ImageNet provides large-scale, diverse supervision, enabling the model to learn general-purpose visual features.
* Pretrained representations significantly improve sample efficiency in low- and medium-data regimes.
* Using a standard pretraining dataset ensures comparability with prior work in transfer learning.
* It allows the study to focus on transfer strategies rather than feature learning from scratch.

Block decomposition:

* **Stem**: initial convolution + pooling; extracts low-level features (edges, textures) and reduces spatial resolution.
* **Stage1**: early residual blocks; captures simple patterns and local structures.
* **Stage2**: mid-level feature extraction; combines primitives into motifs and parts.
* **Stage3**: higher-level representations; more abstract and semantically meaningful features.
* **Stage4**: top residual blocks; highly task-specific features before classification.
* **Head**: global pooling + fully connected layer; maps features to class logits.

Parameter distribution per block (ResNet50, 178-class head):

| Block  | Parameters | % of total |
|--------|------------|------------|
| Stem   |      9,536 |       0.0% |
| Stage1 |    215,808 |       0.9% |
| Stage2 |  1,219,584 |       5.1% |
| Stage3 |  7,098,368 |      29.7% |
| Stage4 | 14,964,736 |      62.7% |
| Head   |    364,722 |       1.5% |
| **Total** | **23,872,754** | **100%** |

**Why blocks are the right unit for F/T/S cuts (features, not parameters):**

The parameter distribution is heavily skewed toward deeper, wider stages (Stage4 alone holds ~63%). This means *trainable parameter count is a poor proxy for configuration identity*: freezing the first two blocks changes the parameter count by less than 1%, yet it has a well-defined functional meaning. The correct lens is the **feature hierarchy**, not parameter mass.

Each stage boundary is a functional boundary: spatial resolution changes, channel width doubles, and the representational abstraction level shifts. Yosinski et al. (2014) showed that transferability degrades with depth as a property of what features *are* (edges → parts → class detectors), not as a function of how many weights encode them. Cuts at stage boundaries therefore have a clear theoretical interpretation aligned with H1–H3.

Cuts *within* a stage would break residual co-adaptation without a clean semantic justification, and sub-stage granularity is listed as V6 (optional follow-up) precisely because it adds complexity without proportional interpretive value at this stage of the study.

**Implication for metrics:** `trainable_pct` should be reported as context only, not used as a primary axis of analysis. The primary structural variable is the block assignment tuple; the primary compute metric is wall-clock training time (`elapsed_s`).

## 4. Configuration Space

Each block is assigned one of:

* F (Frozen): weights fixed
* T (Fine-tuned): pretrained weights updated
* S (Scratch): weights reinitialized and trained

Constraint: monotonic configurations of the form

F^a T^b S^c

with Head fixed as S.

Total valid configurations: 21

## 5. Datasets

Dataset: iWildCam

Rationale for dataset choice:

* iWildCam provides a realistic distribution shift (camera traps, locations, environments), suitable for OOD evaluation.
* It reflects real-world conditions (imbalance, noise, domain variability).
* As part of WILDS, it is a standard benchmark for robustness under shift.

Goal: evaluate transfer strategies under realistic distribution shift.

Potential limitation and mitigation:

* iWildCam introduces dataset-specific challenges (class imbalance, long-tail distribution, camera/location bias) that can confound general conclusions.

* To preserve interpretability, run the core study first on a controlled dataset (e.g., CIFAR100 or ImageNet subset) to isolate the effect of transfer configurations.

* Then validate the same configurations on iWildCam as a robustness test under real-world distribution shift.

* This two-stage design separates *mechanism discovery* (controlled setting) from *stress testing* (iWildCam).

## 6. Underling Assumptions (mostly AIslop)

* **A1 (Feature hierarchy)**: CNN representations are hierarchical, with early layers learning general features and deeper layers learning task-specific ones.

  * Supported by: Yosinski et al., *“How transferable are features in deep neural networks?”*, NeurIPS 2014.

* **A2 (Transferability of pretrained features)**: Features learned on large-scale datasets (ImageNet) are broadly reusable across tasks and domains.

  * Supported by: Donahue et al., *“DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition”*, ICML 2014.

* **A3 (Layer-wise specialization)**: Transferability decreases with depth; deeper layers require more adaptation.

  * Supported by: Kornblith et al., *“Do Better ImageNet Models Transfer Better?”*, CVPR 2019.

* **A4 (Feature compatibility constraint)**: Stable training requires that adjacent layers operate on compatible feature distributions.

  * Supported by: Yosinski et al. (2014) — co-adaptation effects between layers.

* **A5 (Benefit of fine-tuning)**: Updating pretrained weights improves performance when there is domain or task mismatch.

  * Supported by: Girshick et al., *“Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation (R-CNN)”*, CVPR 2014.

* **A6 (Cost-performance tradeoff)**: Partial fine-tuning can achieve competitive performance with reduced computational cost.

  * Supported by: Tan et al., *“EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks”*, ICML 2019 (efficiency-performance tradeoffs).

* **A7 (Robustness of pretrained features)**: Pretrained models tend to generalize better under distribution shift compared to models trained from scratch.

  * Supported by: Hendrycks et al., *“Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty”*, NeurIPS 2019.

* **A8 (Sensitivity to configuration)**: Transfer learning performance is highly sensitive to how layers are frozen or fine-tuned.

  * Supported by: Neyshabur et al., *“What is being transferred in transfer learning?”*, NeurIPS 2020.

## 7. Variables / possible experiments (i have to distinguish better which are variable of the experiment, which are fixed and noisy and which can be motivation for other experiments)

* **V1 Transfer configuration**: compare the 21 valid F/T/S partitions of ResNet50.
* **V2 Dataset size**: small, medium, large subsets to study sample-efficiency.
* **V3 Domain shift**: low-shift vs high-shift settings, depending on how strongly target data differs from the source.
* **V4 Training budget**: fixed protocol to ensure fair comparison across configurations.
* **V5 Pretraining vs no pretraining**: compare ImageNet initialization against training from scratch.
* **V6 Stage granularity**: optionally test whether treating whole stages as blocks is sufficient, or whether finer partitions change conclusions.
* **V7 Robustness across seeds**: repeat key experiments with multiple random seeds to check stability.
* **V8 Controlled-to-real transfer**: first run on a controlled dataset, then validate the same configurations on iWildCam.

## 8. Metrics

* **M1 ID Accuracy**: top-1 accuracy on in-domain (training-like) data; primary measure of task performance *(balanced accuracy may be preferred to reduce noise due to class imbalance)*.
* **M2 OOD Accuracy**: top-1 accuracy on shifted data (e.g., different camera traps / locations); measures robustness to distribution shift *(balanced accuracy may be preferred to reduce noise due to class imbalance)*.
* **M3 ID F1**: macro-averaged F1 score on in-domain data; captures performance across all classes, especially useful under class imbalance by combining precision and recall per class.
* **M4 OOD F1**: macro-averaged F1 score on shifted data; evaluates robustness while accounting for imbalance and per-class variability.
* **M5 Training FLOPs**: total computational cost of training (forward + backward over the full training set); hardware-independent proxy for training cost.
* **M6 Inference FLOPs**: computational cost of a forward pass on the test set; proxy for deployment efficiency.
* **Extra – Trainable parameters**: number of parameters updated during training; auxiliary metric used to contextualize cost and model flexibility. or whatever is need to descibe computational costs

## 9. Hypotheses

* **H1 (Early layers generality)**: Early layers encode low-level, transferable features; freezing them preserves useful representations with minimal loss.
* **H2 (Top layers specificity)**: Top layers are highly task-specific; reinitializing and training them from scratch is necessary when label space changes.
* **H3 (Middle layers plasticity)**: Middle layers balance generality and specificity; fine-tuning yields the largest marginal gains.
* **H4 (Bridging with fine-tuning)**: Inserting a fine-tuned block between frozen and scratch regions mitigates fragile co-adaptation and stabilizes optimization.
* **H5 (ID performance)**: Properly configured transfer learning improves in-domain accuracy over training from scratch, especially in low/medium data regimes.
* **H6 (OOD performance)**: Transfer learning improves OOD generalization due to reuse of robust pretrained features.
* **H7 (Efficiency)**: Comparable or better performance can be achieved at lower computational cost (fewer trainable parameters / FLOPs).
* **H8 (Fragility)**: Poorly chosen F/T/S partitions can degrade performance; the method is sensitive to configuration.
* **H9 (Regime dependence)**: Optimal configurations depend on data size and domain shift; more fine-tuning is favored as shift increases.
* **H10 (Diminishing returns)**: Extensive fine-tuning of early layers yields limited gains relative to its cost.
* **H11 (Scalability across depth)**: The qualitative results reported in *“How transferable are features in deep neural networks?” (Yosinski et al., 2014)*—obtained on relatively shallow architectures (~8 layers)—are expected to hold for deeper networks such as ResNet50, despite increased depth and representational capacity.
* **H12 (Large-scale regime)**: While *“How transferable are features in deep neural networks?” (Yosinski et al., 2014)* suggests diminishing benefits of transfer learning as dataset size increases, we hypothesize that, when properly configured (optimal F/T/S partition), transfer learning can still improve performance even in larger data regimes.
* H13: 

### 10. POSSIBLE WEAKNESSES and NOISY FACTORS:

* **N1: ImageNet pretraining bias**

  * Issue: features are biased toward natural images and may not transfer optimally to wildlife data.
  * Mitigation: include a "no pretraining" baseline and analyze performance gaps.

* **N2: iWildCam domain shift complexity**

  * Issue: multiple overlapping shifts (camera, location, illumination) make it hard to isolate causes.
  * Mitigation: complement with controlled datasets to separate architectural effects from dataset effects.

* **N3: Class imbalance (long-tail)**

  * Issue: dominant classes bias accuracy and obscure true performance differences.
  * Mitigation: report per-class metrics or balanced accuracy; optionally use reweighting.

* **N4: Task specificity (image classification only)**

  * Issue: conclusions may not generalize to other modalities or tasks.
  * Mitigation: explicitly scope conclusions to vision classification and discuss limits.

* **N5: Architecture choice (ResNet50 only)**

  * Issue: results may depend on architectural inductive biases.
  * Mitigation: justify ResNet50 as a representative baseline; optionally validate on a second architecture if time permits.

* **N6: Hyperparameter sensitivity**

  * Issue: learning rates, schedulers, and optimizers can dominate results.
  * Mitigation: fix a standard protocol and tune only once on a reference configuration.

* **N7: Random splits / stochasticity**

  * Issue: variance due to initialization and data splits can mask real effects.
  * Mitigation: run multiple seeds and report mean ± variance.

* **N8: Measurement noise in large-data regime**

  * Issue: performance differences become small and harder to detect as dataset size increases.
  * Mitigation: use larger evaluation sets, statistical testing, and focus on consistent trends rather than single-run gains.

##
