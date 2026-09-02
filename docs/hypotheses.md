# Hypotheses and verdicts

The project was framed around twelve hypotheses before the experiments were run.
This document records what each predicted, what the evidence shows, and which
figure or table bears on it.

The hypothesis numbering is preserved in the figure filenames — `fig7`
corresponds to H1, `fig8` to H2, `fig9` to H4 — which is why the mapping is worth
recording explicitly.

---

## Summary

| | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Early layers are general; freezing them costs little | Supported | `fig7`, `fig3` |
| H2 | Top layers are task-specific and must be relearned when the label space changes | Partial | `fig8`, `fig4` |
| H3 | Fine-tuning middle layers yields the largest marginal gains | Contradicted | Table 3 |
| H4 | A fine-tuned buffer between F and S stabilizes optimization | Supported | `fig9`, `fig5` |
| H5 | Transfer improves in-domain accuracy over scratch | Regime-dependent | Tables 3, 5 |
| H6 | Transfer improves OOD generalization | Partial | Table 3 |
| H7 | Comparable performance at lower compute | Partial | Table 3 |
| H8 | Poor partitions degrade performance; the method is configuration-sensitive | Supported | Tables 3–5 |
| H9 | More fine-tuning is favoured as shift increases | Supported | cross-experiment |
| H10 | Fine-tuning early layers gives limited gains relative to cost | Supported | `fig7` |
| H11 | Yosinski's shallow-network results hold for deeper networks | Partial | `fig1a` |
| H12 | Well-configured transfer still helps in larger-data regimes | Not tested | — |

---

## H1 — Early layers generality

> Early layers encode low-level, transferable features; freezing them preserves
> useful representations with minimal loss.

**Supported, with an important boundary.**

On iWildCam, pushing the frozen prefix deeper while keeping later stages adapted
barely moves the result: T5S1 → F1T4S1 → F2T3S1 → F3T2S1 span 55.3% → 51.7% →
51.7% → 50.8% raw OOD accuracy. Freezing three of five blocks costs about 4.5
points and saves real compute.

On TinyImageNet the same holds even more cleanly: FSSS *beats* the scratch
baseline (74.0 vs 73.7 selfer, and 74.0 transfer).

The boundary: this only holds while the **later** stages remain free to adapt.
The pure-frozen series shows the other side — F5S1 reaches 22.8% OOD accuracy,
worse than the 30.3% of training from scratch. "Freezing early layers is cheap"
is not "freezing is cheap."

*Figures:* `fig7_early_layer_generality`, `fig3_coadaptation_fragility`

## H2 — Top layers specificity

> Top layers are highly task-specific; reinitializing and training them from
> scratch is necessary when the label space changes.

**Partially supported: the diagnosis holds, the prescription does not.**

The diagnosis is confirmed. Deep features are task-specific: frozen FFFF falls to
53.1% cross-domain against 73.5% same-domain, a 20.4-point gap attributable to the
specificity of the deepest block, and on iWildCam freezing Stage4 (F5S1) yields
the weakest configuration in the sweep.

The prescription does not follow. Relearning from scratch is not the appropriate
remedy; fine-tuning is. Replacing a scratch Stage4 with a fine-tuned one
(T4S2 → T5S1) yields +11.2 percentage points of raw OOD accuracy, the largest
single improvement in the sweep. Training from scratch discards 63% of the
pretrained parameters and must recover high-level structure from camera-trap data
alone, whereas fine-tuning retains that structure and adapts it.

The deepest stage is therefore the most consequential block, but the appropriate
treatment is adaptation rather than reinitialization.

*Figures:* `fig8_stage4_specificity`, `fig4_specificity_collapse`

## H3 — Middle layers plasticity

> Middle layers balance generality and specificity; fine-tuning them yields the
> largest marginal gains.

**Contradicted.** The largest marginal gain is at the *last* stage, not the
middle. Compare the two transitions in Table 3:

| Transition | What changes | Raw OOD gain |
|---|---|---|
| T4S2 → T5S1 | Stage4: scratch → fine-tuned | **+11.2 pp** |
| F3T1S2 → F3T2S1 | a middle stage gains fine-tuning | +3.8 pp |

Middle-layer fine-tuning helps, but Stage4 dominates.

## H4 — Bridging with fine-tuning

> Inserting a fine-tuned block between frozen and scratch regions mitigates
> fragile co-adaptation and stabilizes optimization.

**Supported in both experiments, and it is the clearest structural effect
found.**

On iWildCam, at every frozen depth, adding T blocks between the frozen prefix and
the scratch suffix improves OOD accuracy monotonically — e.g. at `a = 4`,
F4S2 → F4T1S1 gains +2.2 points.

TinyImageNet shows the converse, more sharply: the configurations *without* a
buffer where one is most needed — a large scratch Group3 sitting directly above a
fine-tuned Group2 — are exactly the ones that fail. TTTS, FTTS and FFTS all fall
to 67–69%, in **both** selfer and transfer runs, which rules out a domain
explanation. Group3 holds 76% of WRN's parameters and trains at 10× the learning
rate while its input distribution is still moving underneath it. Moving Group3
out of scratch restores near-baseline performance every time.

*Figures:* `fig9_t_buffer_benefit`, `fig5_finetuning_recovery`

## H5 — ID performance

> Properly configured transfer improves in-domain accuracy over scratch,
> especially in low/medium data regimes.

**Regime-dependent.**

Clearly supported on iWildCam: T5S1 reaches 73.5% ID accuracy against 56.5% for
S6, a 17-point margin.

Marginal on TinyImageNet, where the best transfer configurations reach 74.2%
against a 73.7% scratch baseline, a difference of half a point. This is the
expected pattern: the TinyImageNet target subsets provide 50,000 training images
across 100 classes, sufficient to learn good representations without transfer.
The benefit of transfer is largest where target data is scarce relative to task
difficulty.

## H6 — OOD performance

> Transfer learning improves OOD generalization through reuse of robust
> pretrained features.

**True only with adaptation — the mechanism in the hypothesis is wrong.**

Transfer does improve OOD: T5S1 reaches 55.3% raw OOD accuracy against 30.3% for
S6. But the stated mechanism, *reuse*, is not what delivers it. Pure reuse (F5S1,
1.6% of parameters trainable) gives 22.8% — **worse than scratch**. The gain
comes from pretrained initialization plus adaptation, not from holding features
fixed.

This is the paper's most direct practical warning, and it contradicts assumption
A7 as stated.

## H7 — Efficiency

> Comparable or better performance at lower computational cost.

**A real but modest frontier, and mixed partitions do not dominate.**

Fine-tuning only Stage4 (F4T1S1) reaches 49.3% raw OOD accuracy at about **76%**
of full fine-tuning's training FLOPs — 89% of T5S1's performance for roughly
three-quarters of the cost. That is a genuine operating point.

But the frontier is sharply shaped. F3T2S1 and F2T3S1 come closer to full
fine-tuning while needing 94–99% of the compute, so most of the *cheap* gain
comes from adapting Stage4 and the last few points cost nearly full price. No
mixed partition beats T5S1 on OOD balanced accuracy.

This is also where trainable-parameter share misleads: F4T1S1 trains 64% of the
parameters but needs 76% of the FLOPs, because the frozen early stages still run
their forward pass at large spatial resolution.

## H8 — Fragility

> Poorly chosen F/T/S partitions degrade performance; the method is sensitive to
> configuration.

**Strongly supported.** The iWildCam sweep spans 17.5% to 31.3% OOD balanced
accuracy — nearly a factor of two — across configurations of one identical
architecture on one identical dataset under one identical protocol. TinyImageNet
adds two distinct failure *modes*: the FFFF cross-domain collapse (−20.6 points)
and the G2→G3 scratch-after-fine-tune instability (−5 to −7 points).

The second of these is invisible to a binary frozen/scratch design, and finding
it is the main thing the ternary sweep bought.

## H9 — Regime dependence

> Optimal configurations depend on data size and domain shift; more fine-tuning
> is favoured as shift increases.

**Supported — the cleanest cross-experiment trend.**

| Setting | Shift | Best configuration |
|---|---|---|
| TinyImageNet selfer (A→A, B→B) | none | large frozen prefixes remain safe; FFFT at 73.9% |
| TinyImageNet transfer (A→B, B→A) | disjoint label sets | fine-tune most later blocks; FTTT / TSSS at 74.2% |
| ImageNet → iWildCam | real location shift | full fine-tuning, T5S1 |

As source–target mismatch grows, the optimum slides monotonically from reuse
toward adaptation.

## H10 — Diminishing returns on early layers

> Extensive fine-tuning of early layers yields limited gains relative to cost.

**Supported.** F3T2S1 reaches 31.1% OOD balanced accuracy against T5S1's 31.3%
— 0.2 points apart — while freezing the stem, Stage1 and Stage2. Those three
blocks are 6% of the parameters but a large share of the FLOPs, so fine-tuning
them buys almost nothing for real compute.

## H11 — Scalability across depth

> Yosinski et al.'s findings on ~8-layer networks hold for deeper networks.

**Qualitatively yes, quantitatively attenuated.**

The structure replicates: a co-adaptation dip in the selfer frozen curve, a
specialization collapse in the transfer frozen curve, recovery under fine-tuning.
The *magnitudes* differ. Co-adaptation fragility is much weaker here — 73.7% →
71.8%, about 2 points, where AlexNet showed considerably more. WRN's residual
connections plausibly soften it: a frozen intermediate block is easier to route
around when there is a skip path.

Feature specialization, by contrast, is if anything stronger: a 20-point collapse
at FFFF.

*Figure:* `fig1a_yosinski_replication`

## H12 — Large-scale regime

> With an optimal partition, transfer still helps in larger-data regimes.

**Not tested.** This requires a data-size sweep — the same configurations at
several target-set sizes — which falls outside the present protocol. It remains
open.

---

## Findings not anticipated by the hypotheses

Two results were not predicted by any of H1–H12, and both emerge from the ternary
sweep rather than from a binary frozen/scratch design:

1. **The Group2→Group3 instability.** A large scratch block trained immediately
   above a fine-tuned block constitutes a moving-target optimization problem,
   independent of domain. H4 anticipated that a fine-tuned buffer would
   "stabilize optimization", but attributed the effect to co-adaptation; the
   operative mechanism is that the scratch block is optimized against a
   representation that is still shifting beneath it.

2. **Frozen transfer can underperform no transfer.** H6 assumed that reuse is
   beneficial under shift. F5S1 at 22.8% against S6 at 30.3% shows that a fixed
   backbone from a mismatched source domain is worse than no pretrained backbone
   at all.

Both sharpen the same conclusion: the value of deep features derives not from
their having been pretrained, but from their remaining compatible with the
target — either because the task coincides, or because they retain the freedom to
adapt.
