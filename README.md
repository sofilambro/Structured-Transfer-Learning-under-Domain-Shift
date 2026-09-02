# Beyond Freeze-or-Fine-Tune: Structured Transfer Learning under Domain Shift

A study of which depths of a convolutional backbone should be reused, adapted, or
relearned under domain shift.

Contemporary transfer learning offers a wide range of adaptation strategies, from
frozen-feature baselines to full and parameter-efficient fine-tuning. This work
revisits that design space through a deliberately simple and interpretable
formulation. Given a backbone decomposed into contiguous blocks, each block is
assigned one of three states:

| | State | Weights | Optimized |
|---|---|---|---|
| **F** | Frozen — reuse | copied from source | no |
| **T** | Fine-tuned — adapt | copied from source | yes |
| **S** | Scratch — relearn | randomly reinitialized | yes |

The search is restricted to *monotonic* configurations `F^a T^b S^c`, in which
frozen blocks form a lower prefix, fine-tuned blocks provide an adaptation
region, and scratch blocks sit closest to the always-scratch classifier head.
This yields `(L+1)(L+2)/2` valid partitions: **15** for a four-block WRN-28-10 and
**21** for a five-block ResNet50. The formulation extends the classical
layer-transfer setup of Yosinski et al. (2014) from binary transferred prefixes to
a ternary reuse–adapt–relearn partition.

The study has two parts:

1. **A controlled experiment.** WRN-28-10 on two disjoint 100-class subsets of
   TinyImageNet, with both source networks trained from scratch. Because source
   and target are drawn from a single dataset, this setting separates two effects
   that are ordinarily entangled: co-adaptation fragility and feature
   specialization.
2. **A real-world experiment.** ImageNet-pretrained ResNet50 adapted to
   **iWildCam** (WILDS), where out-of-distribution examples come from camera-trap
   locations unseen during training.

The full write-up is `Report_Transfer_Learning.pdf` at the repository root.
30562 — Machine Learning and Artificial Intelligence, Bocconi University.

---

## Principal findings

**Frozen-feature transfer fails under large distribution shift.** Freezing all
five ResNet50 blocks and training only a new classifier head (F5S1) attains
**22.8%** raw OOD accuracy, below the **30.3%** obtained by training the entire
network from scratch (S6). Under shift of this magnitude, a fixed ImageNet
backbone with a new head is too rigid to be useful.

**The final residual stage governs transfer performance.** Replacing a scratch
Stage4 with a fine-tuned one (T4S2 to T5S1) yields **+11.2 percentage points** of
raw OOD accuracy, the largest single improvement observed in the sweep. When
frozen, the model is locked into source-task semantics; when trained from
scratch, it must relearn high-level representations from camera-trap data alone.
Fine-tuning is the only assignment that combines pretrained structure with target
adaptation.

**Deep frozen features are reusable only when the label spaces match.** In the
controlled experiment, freezing all four WRN blocks attains **73.5%** when source
and target class subsets coincide, and **53.1%** when they do not — a 20.4-point
gap attributable to the task-specificity of the deepest residual group.

**The ternary sweep exposes a failure mode invisible to a binary design.**
Whenever Group2 is fine-tuned while Group3 is trained from scratch (TTTS, FTTS,
FFTS), accuracy falls to 67–69%, in both same-domain and cross-domain runs. The
symmetry across conditions rules out a domain-mismatch explanation. Group3
contains 76% of the model's parameters and is optimized at ten times the learning
rate of the transferred blocks while its input distribution is still drifting,
which is consistent with a moving-target optimization problem rather than a
transfer effect.

**Mixed partitions expose an efficiency frontier rather than an accuracy gain.**
Fine-tuning Stage4 alone (F4T1S1) reaches 89% of full fine-tuning's OOD accuracy
at approximately **76%** of its training FLOPs. No mixed partition surpasses full
fine-tuning outright; their value is diagnostic, indicating where adaptation is
necessary and where freezing remains safe.

**Required adaptation increases with source–target mismatch.** A consistent
pattern emerges across the two experiments:

| Setting | Shift | Best configuration |
|---|---|---|
| TinyImageNet, same-domain | none | large frozen prefixes remain safe |
| TinyImageNet, cross-domain | disjoint label spaces | fine-tune most later blocks |
| ImageNet to iWildCam | real location shift | full fine-tuning |

---

## Repository layout

```
├── README.md
├── pyproject.toml                pip install -e ".[dev]"
├── requirements.txt              cluster-friendly alternative
│
├── docs/
│   ├── methodology.md            design rationale: monotonicity, block granularity, metrics
│   ├── hypotheses.md             H1–H12, with verdicts and supporting evidence
│   └── hpc.md                    cluster protocol, SLURM submission, measured costs
│
├── src/structured_transfer/
│   ├── partitions.py             the F/T/S formalism, shared by both experiments
│   ├── budget.py                 compute-budget policies (epoch, wall clock, step)
│   ├── utils.py                  seeding, parameter accounting, FLOPs, checkpointing
│   ├── iwildcam/                 Experiment 2: ResNet50 to iWildCam
│   ├── tinyimagenet/             Experiment 1: WRN-28-10 on TinyImageNet
│   └── analysis/                 artifact loading and figure generation
│
├── scripts/                      command-line entry points, one per task
├── slurm/                        array-job submission scripts
├── results/
│   ├── iwildcam/                 21 run artifacts, leaderboards, exploratory plots
│   └── tinyimagenet/             per-configuration result tables
└── tests/                        CPU-only, no dataset required
```

---

## Installation and use

```bash
git clone https://github.com/sofilambro/Transfer_Learning_AI_project
cd Transfer_Learning_AI_project

# Install a torch build matching the local CUDA version first, e.g. CUDA 12.4:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,flops]"

pytest -q                             # partition and architecture tests
python scripts/make_leaderboard.py    # rebuild the leaderboard from run artifacts
python scripts/make_figures.py        # regenerate all figures
```

`make_leaderboard.py` requires only the committed result files, so it runs on a
fresh clone without a dataset or GPU.

### iWildCam experiment

```bash
python scripts/iwildcam_download_metadata.py              # ~30 MB, metadata only
python scripts/iwildcam_explore_metadata.py               # dataset characterization
python scripts/iwildcam_build_mini_dataset.py --part all  # ~50k-image subset, ~3 GB

python scripts/iwildcam_run.py --smoke              # pipeline check: 5 configs, 2 epochs
python scripts/iwildcam_run.py --partition F2T3S1   # a single configuration
python scripts/iwildcam_run.py --all                # all 21 (approximately 65 GPU-hours)
```

The mini subset supports local iteration. All reported results use the full
203k-image dataset; results obtained on the mini subset are not comparable to the
leaderboard.

### TinyImageNet experiment

```bash
# tiny-imagenet-200 must be present under data/
python scripts/tinyimagenet_prepare.py --stage all    # class split, corrupted validation sets

python scripts/tinyimagenet_run.py --config SSSS --direction AtoA   # source network A
python scripts/tinyimagenet_run.py --config SSSS --direction BtoB   # source network B
python scripts/tinyimagenet_run.py --config FFTS --direction AtoB   # a transfer run
python scripts/tinyimagenet_run.py --all                            # the full 58-model protocol
```

The two source networks must be trained before any transfer run.

### Cluster execution

```bash
DATA_DIR=$SCRATCH/tl/data sbatch slurm/iwildcam_sweep.sbatch     # 21-task array
```

See [`docs/hpc.md`](docs/hpc.md) for environment setup, resource sizing, and the
time-budget protocol.

---

## Configuration spaces

**ResNet50 on iWildCam** — five backbone blocks, always-scratch head, 21 partitions:

```
                                                        S6
                                                 F1S5,  T1S5
                                          F2S4,  F1T1S4,  T2S4
                                   F3S3,  F2T1S3,  F1T2S3,  T3S3
                            F4S2,  F3T1S2,  F2T2S2,  F1T3S2,  T4S2
                     F5S1,  F4T1S1,  F3T2S1,  F2T3S1,  F1T4S1,  T5S1
```

**WRN-28-10 on TinyImageNet** — four backbone blocks, 15 partitions, in the
four-letter backbone notation used throughout the write-up (the head is omitted
because it is always scratch):

```
                                    SSSS
                                FSSS,  TSSS
                            FFSS,  FTSS,  TTSS
                      FFFS,  FFTS,  FTTS,  TTTS
                FFFF,  FFFT,  FFTT,  FTTT,  TTTT
```

### Block decomposition

The two backbones distribute parameters and computation in opposite directions
relative to depth, which is why compute is reported in FLOPs rather than as a
trainable-parameter share.

**ResNet50 on iWildCam** (182-class head):

| Block | Parameters | % | Forward FLOPs | % |
|---|---:|---:|---:|---:|
| Stem | 9,536 | 0.04 | 0.1196 G | 2.91 |
| Stage1 | 215,808 | 0.90 | 0.6768 G | 16.47 |
| Stage2 | 1,219,584 | 5.11 | 1.0338 G | 25.15 |
| Stage3 | 7,098,368 | 29.72 | 1.4687 G | 35.74 |
| Stage4 | 14,964,736 | 62.66 | 0.8105 G | 19.72 |
| Head | 372,918 | 1.56 | 0.0005 G | 0.01 |
| **Total** | **23,880,950** | 100 | **4.1098 G** | 100 |

**WRN-28-10 on TinyImageNet** (100-class head, 64×64 input):

| Block | Parameters | % | Forward FLOPs | % |
|---|---:|---:|---:|---:|
| Conv1 | 432 | 0.001 | 0.0018 G | 0.007 |
| Group1 | 1,640,672 | 4.49 | 6.7202 G | 28.21 |
| Group2 | 6,968,000 | 19.07 | 8.5538 G | 35.90 |
| Group3 | 27,862,400 | 76.26 | 8.5498 G | 35.88 |
| Head | 65,380 | 0.18 | 0.0006 G | 0.002 |
| **Total** | **36,536,884** | 100 | **23.8261 G** | 100 |

Stage4 accounts for 63% of ResNet50's parameters but only 20% of its forward
computation, while earlier stages are inexpensive in parameters and costly in
FLOPs because they operate at larger spatial resolutions. Consequently
`trainable_pct` is reported as contextual information only: F4T1S1 optimizes 64%
of the parameters yet requires 76% of the FLOPs of full fine-tuning.

---

## iWildCam results

All 21 configurations, ranked by OOD balanced accuracy (equivalently macro
recall), which is the primary metric because iWildCam is long-tailed. Macro-F1 is
reported as the WILDS-comparable secondary metric. The machine-readable copy is
`results/iwildcam/leaderboard_paper.csv`.

| Config | Train. % | ID Acc | ID Bal | ID F1 | OOD Acc | OOD Bal | OOD F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **T5S1** | 100.0 | 73.5 | 41.8 | 35.5 | **55.3** | **31.3** | **29.5** |
| F3T2S1 | 93.9 | 71.2 | 41.9 | 35.9 | 50.8 | 31.1 | 26.8 |
| F1T4S1 | 100.0 | 70.8 | 42.5 | 36.0 | 51.7 | 30.9 | 26.3 |
| F2T3S1 | 99.1 | 73.2 | 42.7 | 36.2 | 51.7 | 30.8 | 26.9 |
| F4T1S1 | 64.2 | 69.5 | 41.4 | 34.5 | 49.3 | 29.8 | 23.5 |
| F2T2S2 | 99.1 | 68.9 | 41.2 | 34.6 | 48.8 | 29.6 | 25.0 |
| F4S2 | 64.2 | 67.1 | 40.1 | 33.7 | 47.1 | 29.1 | 23.9 |
| T4S2 | 100.0 | 68.7 | 41.4 | 34.8 | 44.1 | 28.8 | 24.6 |
| F3T1S2 | 93.9 | 67.7 | 41.0 | 33.2 | 47.0 | 28.2 | 23.1 |
| F1T3S2 | 100.0 | 67.9 | 40.4 | 32.4 | 47.5 | 28.2 | 22.7 |
| F3S3 | 93.9 | 64.1 | 38.4 | 31.7 | 41.2 | 27.7 | 21.4 |
| F2T1S3 | 99.1 | 66.1 | 38.1 | 31.0 | 41.9 | 26.6 | 21.3 |
| F1T2S3 | 100.0 | 64.2 | 37.3 | 30.1 | 44.2 | 26.3 | 20.7 |
| T3S3 | 100.0 | 65.8 | 37.4 | 29.9 | 40.4 | 25.2 | 18.8 |
| F1T1S4 | 100.0 | 64.3 | 33.9 | 28.0 | 40.8 | 23.0 | 19.6 |
| T2S4 | 100.0 | 62.8 | 35.1 | 28.4 | 39.6 | 22.5 | 18.1 |
| **F5S1** | **1.6** | 42.8 | 35.3 | 26.3 | **22.8** | 21.9 | 15.2 |
| F2S4 | 99.1 | 59.0 | 34.1 | 28.0 | 39.8 | 21.3 | 16.7 |
| T1S5 | 100.0 | 58.4 | 34.2 | 26.7 | 33.3 | 20.7 | 14.8 |
| F1S5 | 100.0 | 57.5 | 32.3 | 25.9 | 33.0 | 17.6 | 15.0 |
| **S6** | 100.0 | 56.5 | 30.7 | 24.1 | **30.3** | 17.5 | 14.1 |

The two emphasized rows near the foot of the table are the central result: the
fully frozen backbone F5S1 ranks below S6, which uses no pretraining at all.

Across the 21 configurations, ID and OOD metrics are strongly aligned. The sweep
does not expose a regime in which a configuration improves in-domain performance
at the cost of OOD generalization; weak partitions fail under both conditions,
indicating that the F/T/S assignment governs representation quality rather than
an overfitting–robustness trade-off.

## TinyImageNet results

Clean accuracy is the best validation top-1 attained during training; corrupted
accuracy is top-1 on a TinyImageNet-C-style corrupted validation set. Δ is
relative to the source-free SSSS baseline. Machine-readable copies are in
`results/tinyimagenet/`.

| | Selfer (A→A, B→B) | | Transfer (A→B, B→A) | |
|---|---:|---:|---:|---:|
| **Config** | **Clean** | **Δ** | **Clean** | **Δ** |
| SSSS | 73.7 | +0.0 | 73.7 | +0.0 |
| FSSS | 73.8 | +0.1 | 74.0 | +0.3 |
| FFSS | 73.0 | −0.7 | 73.3 | −0.4 |
| FFFS | 71.8 | −1.9 | 71.4 | −2.3 |
| FFFF | 73.5 | −0.2 | **53.1** | **−20.6** |
| TSSS | 73.3 | −0.4 | 74.2 | +0.5 |
| TTSS | 73.6 | −0.1 | 73.6 | −0.1 |
| TTTS | **68.0** | **−5.7** | **69.0** | **−4.7** |
| TTTT | 73.6 | −0.1 | 74.1 | +0.4 |
| FTSS | 73.2 | −0.5 | 73.1 | −0.6 |
| FFTS | **66.9** | **−6.8** | **67.8** | **−5.9** |
| FTTS | **68.2** | **−5.5** | **69.2** | **−4.5** |
| FFFT | 73.9 | +0.2 | 71.9 | −1.8 |
| FFTT | 73.6 | −0.1 | 73.8 | +0.1 |
| FTTT | 73.3 | −0.4 | 74.2 | +0.5 |

Two failure modes are visible, and they are qualitatively distinct. FFFF degrades
only in the cross-domain column, which identifies feature specialization. TTTS,
FFTS and FTTS degrade comparably in both columns, which identifies the
Group2-to-Group3 optimization instability and excludes a domain explanation.

Corruption accuracy tracks clean accuracy closely across the 56 source-dependent
runs (Pearson r = 0.980), so this synthetic evaluation does not reveal a separate
clean–robustness trade-off. It should be interpreted as a stress test rather than
as a substitute for the real distribution shift measured on iWildCam.

---

## Limitations

**Seeds and splits.** The controlled experiment uses a single semantically
stratified split into two 100-class subsets, and the main experiments are not
repeated over multiple random seeds. The two transfer directions A→B and B→A
provide a useful symmetry check, but do not replace an estimate of split-to-split
or seed-to-seed variance. Differences of one to two points should not be
over-interpreted.

**Dataset scale.** TinyImageNet is smaller and lower-resolution than ImageNet,
and its two class subsets are drawn from the same 200-class pool, so the
source–target shift is limited relative to more realistic transfer scenarios.
This likely reduces the magnitude of the observed generalization benefit.

**Architecture granularity.** Both experiments use coarse block-level
decompositions: four WRN-28-10 blocks and five ResNet50 blocks. This keeps the
sweep interpretable and computationally feasible, but cannot identify finer
transition points within residual groups or bottleneck layers.

**Model family and adaptation methods.** The experiments focus on convolutional
backbones and do not evaluate Vision Transformers, ConvNeXt-style models, or
self-supervised backbones. They also do not compare against parameter-efficient
adaptation methods such as adapters, low-rank updates, or visual prompt tuning.
These constitute a natural extension of the F/T/S diagnostic framework rather
than a competing formulation.

---

## Citation

```bibtex
@misc{lambro2026structured,
  title  = {Beyond Freeze-or-Fine-Tune: Structured Transfer Learning under Domain Shift},
  author = {Lambro, Sofia and Lombardi, Alessandro and Riva, Marco
            and Russo, Katia and Tarantino, Tommaso},
  year   = {2026},
  note   = {30562 --- Machine Learning and Artificial Intelligence,
            Bocconi University. Course project report.}
}
```

## Authors

Sofia Lambro, Alessandro Lombardi, Marco Riva, Katia Russo, Tommaso Tarantino —
Bocconi University.

## References

- Yosinski, J., Clune, J., Bengio, Y., and Lipson, H. *How transferable are features in deep neural networks?* NeurIPS, 2014.
- Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., et al. *WILDS: A benchmark of in-the-wild distribution shifts.* ICML, 2021.
- Donahue, J., Jia, Y., Vinyals, O., Hoffman, J., Zhang, N., Tzeng, E., and Darrell, T. *DeCAF: A deep convolutional activation feature for generic visual recognition.* ICML, 2014.
- Kornblith, S., Shlens, J., and Le, Q. V. *Do better ImageNet models transfer better?* CVPR, 2019.
- Hendrycks, D. and Dietterich, T. *Benchmarking neural network robustness to common corruptions and perturbations.* ICLR, 2019.
- He, K., Zhang, X., Ren, S., and Sun, J. *Deep residual learning for image recognition.* CVPR, 2016.
- Neyshabur, B., Sedghi, H., and Zhang, C. *What is being transferred in transfer learning?* NeurIPS, 2020.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
