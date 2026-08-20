# SolarGuard — Experimental Evaluation Report

This report documents the reasoning behind SolarGuard's experiments: what was asked, how each
comparison was controlled, what the evidence supports, and — equally important — what it does not.
Two of the three investigations produced inconclusive or negative results. Those are recorded here
with the same care as anything that worked.

All figures are read from saved artifacts under `experiments/`. Nothing is estimated.

---

## 1. Research Question

The baseline model reaches 0.7534 validation macro-F1, but its performance is highly uneven across
classes: Electrical-damage scores 0.889 F1 while Bird-drop scores 0.545. The question driving this
work was therefore not *"how do we raise the average"* but:

> **Which specific mechanism limits performance on the weak classes — the loss function, the
> augmentation pipeline, or the data itself — and can any single change be shown to improve
> matters given the evidence available?**

Two factors were investigated, chosen in the order the evidence justified:

1. **The loss function (Experiment 2).** Class imbalance is already handled by inverse-frequency
   weighted cross-entropy. But the weakest class, Bird-drop, is *not* the rarest — it has 87
   training images and a near-neutral class weight of 1.0345. Frequency-based weighting cannot
   help a class that is difficult rather than scarce. Focal loss modulates by prediction
   confidence instead, so it was the natural next hypothesis.
2. **The augmentation pipeline (Experiment 3).** After Experiment 2 failed, attention moved to
   whether training data itself was being degraded. Direct measurement of the augmentation
   pipeline found a defect, described in §6.

A third question — whether results were reproducible at all — had to be settled first (§4), and
answering it uncovered a bug that invalidated earlier comparisons.

---

## 2. Baseline

The locked reference model, frozen and never modified:

| Property | Value |
|---|---|
| Architecture | `BaselineCNN` — 4 conv blocks (16/32/64/128) → GlobalAvgPool → Dropout(0.5) → Linear(128, 6) |
| Trainable parameters | **98,454** |
| Seed | 42 |
| Best checkpoint | **epoch 23** |
| Validation samples | **117** |
| Accuracy | 0.7692 |
| Macro precision | 0.7639 |
| Macro recall | 0.7504 |
| **Macro F1** | **0.7534** |
| Weighted F1 | 0.7744 |
| Validation loss | 0.8091 |

### Why macro-F1 is the primary metric

The six classes are not equally represented. In the 540-image training split, Clean and Dusty
together account for 46.1% of the data, while Physical-Damage accounts for 8.5% — an imbalance
ratio of **2.73:1**.

Macro-F1 averages per-class F1 **without weighting by class frequency**, so every class
contributes equally regardless of size. Accuracy and weighted-F1 both weight by support, which
means a model can score well on either while failing the rarest classes entirely. This is not
hypothetical for SolarGuard: the baseline's weighted-F1 (0.7744) exceeds its macro-F1 (0.7534) by
0.0210, and that gap *is* the majority-class skew made visible.

The operational argument points the same way. A missed Physical-Damage or Electrical-damage panel
matters more than a missed Clean one, so the metric used to select models should not discount
exactly the classes the system exists to catch. Accuracy and weighted-F1 are reported throughout
for completeness but are **never** used for checkpoint selection or experiment decisions.

---

## 3. Experimental Design

**Locked baseline.** One reference run is frozen. Every subsequent experiment is compared against
it and against a 3-seed baseline arm, so no comparison depends on re-running the reference.

**Single-variable changes.** Each experiment alters exactly one thing, and that claim is verified
mechanically rather than asserted. For Experiment 3, all 28 configuration keys were parsed and
diffed to confirm a single functional difference, and the resolved transform pipelines were
compared object-by-object. Dataset, split, architecture, optimizer, learning rate, scheduler,
batch size, image resolution, and selection rule are held constant across all arms.

**Multi-seed evaluation.** Every arm runs **seeds 42, 123, 456**, reported as mean ± standard
deviation. This was not a stylistic choice: the baseline's own seed-to-seed macro-F1 standard
deviation is **0.0381**, which is the noise floor any claimed effect must clear.

**Checkpoint selection.** Highest validation macro-F1 across epochs, with `min_delta = 1e-4` so
floating-point noise cannot register as improvement, ties broken by lowest validation loss. Early
stopping fires after 15 epochs without improvement.

**Reproducibility.** `set_seed()` covers Python `random`, NumPy, PyTorch CPU/CUDA, and sets
`cudnn.deterministic=True`, `benchmark=False`. It is called **before model construction** so weight
initialization is genuinely seed-controlled. DataLoaders receive a seeded generator and a
`worker_init_fn`.

**Metric recomputation.** Reported per-class metrics come from **reloading the selected best
checkpoint and re-running inference**, not from the training loop's in-memory values. This
doubles as a check that checkpoints round-trip faithfully.

**Test data untouched.** The 115-image test split has never been loaded or evaluated, and has not
informed any decision — architecture, hyperparameters, loss, augmentation, or checkpoint
selection. This is enforced structurally: `build_train_val_dataloaders()` contains no code path
capable of constructing a test loader. Had the test set been used to choose between Experiments 2
and 3, it would have become a second validation set, and any later "test performance" figure would
be optimistically biased by exactly the amount of selection pressure applied to it.

### Statistical limitations imposed by n = 117

The validation set has 117 images. Three consequences follow, and they bound every conclusion in
this report:

- **Metric granularity.** One flipped prediction changes macro-F1 by roughly **0.0085**. Nothing
  finer than that is measurable at all.
- **Noise floor.** The baseline's seed-to-seed standard deviation is **0.0381** — about 4.5
  validation samples wide.
- **Detectable effect size.** With 3 seeds per arm, the standard error of a difference between
  arms is roughly 0.025, so only effects of approximately **0.06 macro-F1 or larger** can be
  separated from noise. Both experiments produced effects far below that threshold.

Per-class resolution is worse still. Physical-Damage has 10 validation images, so its F1 moves by
about 0.10 per single prediction — its per-class score is close to a measurement artifact.

---

## 4. Experiment 1 — Baseline Reproducibility

**Question.** Before comparing anything, could an identical configuration reproduce itself?

**Finding: initially, no.** Two runs of the same configuration, both nominally seeded at 42,
produced macro-F1 of 0.7844 and 0.8108 — a spread of 0.0264. Investigation traced this to
initialization order: the model was being constructed **before** `set_seed()` was called, so weight
initialization was governed by whatever RNG state the process happened to be in, not by the seed.
The diagnosis was confirmed directly — two `BaselineCNN()` constructions in the same unseeded
process produce different weights, while two each preceded by `set_seed(42)` produce bit-identical
weights.

**Fix.** `set_seed(config.seed)` moved to execute immediately before model construction.

**Verification after the fix.** Two independent full runs of the identical configuration were
compared exhaustively:

| Check | Result |
|---|---|
| All 83 epochs of `history.csv` (train/val loss, accuracy, macro-F1, weighted-F1, LR) | **identical** |
| Confusion matrices | **identical** |
| `classification_report.json` | **identical** |
| Best-checkpoint model weights, tensor by tensor | **bit-identical** |
| Wall-clock time per epoch | differs (expected — not part of the computation) |

**What this establishes.** Reproducibility is a verified property on this hardware, not an
aspiration. It also means that any difference observed between arms is attributable to the changed
variable plus seed effects — not to uncontrolled nondeterminism.

**What it does not establish.** Reproducibility across *different* GPU architectures. cuDNN may
select different algorithms on different hardware, so bit-identical results are guaranteed only on
matching hardware.

**Why this matters more than it looks.** This bug was silent. It produced plausible numbers, no
warnings, and no crash. Had it gone unnoticed, Experiments 2 and 3 would have compared arms whose
initializations were uncontrolled — and any conclusion drawn from them would have been unsound.

---

## 5. Experiment 2 — Focal Loss

### Hypothesis

Inverse-frequency class weighting corrects for **rarity**. The baseline's weakest class,
Bird-drop (F1 0.545), is not rare: 87 training images, 19 validation images, and a class weight of
1.0345 — essentially neutral. Its errors scatter across four other classes, which is the signature
of a **confusable** class, not a starved one. Focal loss modulates the loss by prediction
confidence via a `(1 − p_t)^γ` term, concentrating gradient on hard examples regardless of class
frequency. The hypothesis was that this would specifically improve Bird-drop.

### Configuration

Class-weighted focal loss, **γ = 2**, retaining the existing inverse-frequency weights as the α
term. Reduction matches `nn.CrossEntropyLoss(weight=…)` exactly: `Σ(wᵢ·lᵢ) / Σ(wᵢ)`.

The implementation was verified to be a strict single-knob extension of the baseline: at **γ = 0**
it reproduces weighted cross-entropy in both loss value and **bit-identical gradients**
(max |Δgrad| = 0.000e+00), while at γ = 2 gradients genuinely differ (max |Δgrad| = 9.61e-03).
Per-sample behaviour was confirmed to match the theory — an easy, confident example retains
0.0000 of its cross-entropy loss, a hard one retains 0.4954. Nothing else changed.

### Results (seeds 42, 123, 456)

| seed | baseline macro-F1 | focal macro-F1 | best epoch (base → focal) |
|---|--:|--:|---|
| 42 | 0.7844 | 0.7331 | 68 → 30 |
| 123 | 0.7189 | 0.7301 | 27 → 27 |
| 456 | 0.7855 | 0.7690 | 43 → 24 |
| **mean ± std** | **0.7629 ± 0.0381** | **0.7441 ± 0.0216** | 46.0 → 27.0 |

### Effect size and statistical interpretation

| Quantity | Value |
|---|---|
| Absolute macro-F1 change | **−0.0189** |
| Baseline seed-to-seed std | 0.0381 |
| Change vs noise floor | **smaller than noise** (0.0189 < 0.0381) |
| Cohen's d | −0.608 |
| 95% CI on the difference | **[−0.0685, +0.0308]** — contains zero |
| Seeds on which focal won | **1 of 3** |

The confidence interval spans both a substantial degradation and a small improvement. **No
reliable effect was demonstrated in either direction**, and the point estimate favours the
baseline.

### Decision: rejected

**Focal loss did not demonstrate a reliable macro-F1 improvement over the weighted cross-entropy
baseline on this dataset.** The baseline configuration was retained.

### What was learned

This is a useful negative result, not a wasted experiment.

- **The predicted mechanism did not occur.** Bird-drop, the explicit target, got **worse**
  (0.6195 → 0.5786, −0.0410). The only class that improved meaningfully was Physical-Damage
  (0.5516 → 0.6083, +0.0567) — the *rarest* class, which is the opposite of the confidence-based
  mechanism proposed. The hypothesis was not merely unsupported; its stated causal story was
  contradicted.
- **A genuine secondary effect appeared.** Focal loss halved Physical-Damage's cross-seed
  variance (std 0.137 → 0.068), stabilising the smallest class even without a significant mean
  gain.
- **An unresolved confound remains.** Focal loss caused systematically earlier stopping (mean best
  epoch 46.0 → 27.0; every focal run stopped earlier than its baseline counterpart). Because focal
  suppresses loss on easy examples, gradient magnitude shrinks as the model improves, so validation
  macro-F1 plateaus sooner and patience-15 fires early. **The focal models may be undertrained
  rather than inferior.** Testing this would require a longer patience or an adjusted learning rate
  for the focal arm — which would break single-variable control, so it was not done.
- **Validation loss dropped sharply** (0.7414 → 0.4398) and this is **not** evidence of
  improvement. Focal loss is a different objective; its values are not comparable to
  cross-entropy's. This is recorded only so the number is not later mistaken for a result.

---

## 6. Experiment 3 — Removing Rotation Augmentation

### Hypothesis and how it was formed

Experiment 2's failure prompted a direct inspection of the training pipeline rather than another
guess at the loss function. Two findings emerged from measurement:

1. **The model is not overfitting.** In all three baseline runs, training loss sits *above*
   validation loss at both the best and final epoch (gaps +0.026 to +0.189). With 98,454
   parameters and 540 images, the model never memorises the training set — augmentation already
   makes training harder than evaluation. This is direct evidence that adding *more* augmentation
   is the wrong direction, and it is consistent with Experiment 2's failure (focal loss also makes
   the objective harder to satisfy).
2. **One augmentation was measurably corrupting training images.** `RandomRotation(15)` defaults to
   `fill=0`. Direct measurement over 200 augmented samples found it blacks out a mean of **5.53%
   of pixels (max 10.08%)**, affecting **~91% of training images**. Validation images contain
   **0%** black pixels.

The second finding is a train/eval distribution gap created by augmentation itself: the model
spends capacity learning to ignore black wedges that never appear at inference, while ~5% of real
panel content is discarded per image.

A competing hypothesis was tested and **rejected before this one was chosen**. An apparent
train/eval geometry mismatch — training crops at aspect ratios 0.75–1.333 while evaluation squashes
full images up to 3.6:1, affecting 90.6% of validation images — looked compelling. Running
inference with all three baseline checkpoints showed squashed images are classified *better*, not
worse (0.8095 vs 0.7500; correlation +0.061). The data contradicted the theory, so it was not
pursued.

### Exact change

A single boolean, in a **separate config file** (`configs/preprocessing_norot.yaml`) so the
baseline arm remains reproducible from the repository unchanged:

```yaml
rotation:
    enabled: false        # baseline: true
```

Verified as the only difference: **28 configuration keys compared, 1 functional difference**. The
resolved training transform lists differ by exactly one element:

```
baseline: [RandomResizedCrop, Lambda, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize]
norot:    [RandomResizedCrop, Lambda, RandomHorizontalFlip,                 ColorJitter, ToTensor, Normalize]
```

Loss, optimizer, learning rate, scheduler, architecture, split, and seeds unchanged.

### Results (seeds 42, 123, 456)

| seed | baseline macro-F1 | no-rotation macro-F1 | best epoch (base → norot) |
|---|--:|--:|---|
| 42 | 0.7844 | 0.7877 | 68 → 75 |
| 123 | **0.7189** | **0.7863** | 27 → 80 |
| 456 | 0.7855 | 0.7447 | 43 → 23 |
| **mean ± std** | **0.7629 ± 0.0381** | **0.7729 ± 0.0244** | 46.0 → 59.3 |

| Quantity | Value |
|---|---|
| Absolute macro-F1 change | **+0.0100** |
| Baseline seed-to-seed std | 0.0381 |
| Change vs noise floor | **smaller than noise** (0.0100 ≈ ¼ of 0.0381) |
| Cohen's d | +0.312 |
| 95% CI on the difference | **[−0.0413, +0.0612]** — contains zero |
| Seeds on which no-rotation won | 2 of 3 |
| **Std reduction** | **0.0381 → 0.0244 (~36%)** |
| Worst-seed macro-F1 | 0.7189 → **0.7447** (+0.0258) |

> **The +0.0100 macro-F1 shift is NOT statistically established.** It is roughly one quarter of the
> seed-to-seed noise floor, and the confidence interval comfortably contains zero. This experiment
> does **not** demonstrate improved generalization.

### Class-level changes

| Class | baseline F1 | no-rotation F1 | Δ |
|---|--:|--:|--:|
| Bird-drop | 0.6195 | 0.6523 | **+0.0328** |
| Physical-Damage | 0.5516 | 0.5803 | **+0.0286** |
| Dusty | 0.8441 | 0.8689 | +0.0248 |
| Snow-Covered | 0.8944 | 0.8890 | −0.0054 |
| Electrical-damage | 0.8457 | 0.8359 | −0.0099 |
| Clean | 0.8221 | 0.8110 | −0.0111 |

The two weakest baseline classes improved most, but three classes got slightly worse — the effect
is less targeted than the hypothesis predicted.

### Worst-seed behaviour

The most interesting result is not the mean. Seed 123 was the baseline's failure case: macro-F1
0.7189, best epoch 27, early-stopped at 42. Without rotation, the same seed reached **0.7863 at
epoch 80**. Removing rotation appears to eliminate a failure mode in which training stalled early —
which is what the ~36% variance reduction reflects.

### Decision: provisionally retained

Retained, explicitly **not** on the strength of the mean, for three reasons:

1. **It removes a measured defect.** Training on images where ~5.53% of pixels have been blacked
   out by an artifact absent from all validation data is worse-specified than not doing so. Even at
   exactly zero performance benefit, the cleaner pipeline is the better-defined one. Experiment 2
   had no equivalent standing argument — focal loss had to *win* to justify itself, and did not.
2. **Variance fell ~36%, with the worst seed improving most.** A model that sometimes collapses is
   worse than one that does not, independently of average performance.
3. **All four aggregate metrics moved in the same direction**, unlike Experiment 2 which degraded
   three of four.

The defensible claim is **"a better-specified preprocessing pipeline that did no harm and reduced
variance"** — not "an improvement in generalization."

**Retained cost, invisible to this experiment.** Real user photographs are often tilted. Removing
rotation removes tilt invariance, and the validation set — which is not tilted — cannot measure
that loss. If tilt robustness matters later, the correct follow-up is rotation with edge-replicate
fill rather than black fill.

---

## 7. Comparative Results

All arms: seeds 42, 123, 456; identical dataset, split, architecture, optimizer, learning rate,
scheduler, batch size, and selection rule.

| Metric | Baseline (weighted CE) | Exp 2 — Focal (γ=2) | Exp 3 — No rotation |
|---|---|---|---|
| **Macro-F1** | **0.7629 ± 0.0381** | 0.7441 ± 0.0216 | **0.7729 ± 0.0244** |
| Accuracy | 0.7892 ± 0.0261 | 0.7635 ± 0.0197 | 0.7977 ± 0.0300 |
| Weighted-F1 | 0.7872 ± 0.0278 | 0.7625 ± 0.0203 | 0.7958 ± 0.0254 |
| Validation loss | 0.7414 ± 0.0443 | 0.4398 ± 0.0418 † | 0.7424 ± 0.1092 |
| Macro-F1 Δ vs baseline | — | **−0.0189** | **+0.0100** |
| Cohen's d | — | −0.608 | +0.312 |
| 95% CI on Δ | — | [−0.0685, +0.0308] | [−0.0413, +0.0612] |
| Seeds won vs baseline | — | 1 / 3 | 2 / 3 |
| Mean best epoch | 46.0 | 27.0 | 59.3 |
| **Status** | reference | **rejected** | **provisionally retained** |

† Focal validation loss is computed under a **different objective** and is not comparable to the
cross-entropy columns. It is shown for completeness only.

Per-seed macro-F1:

| seed | Baseline | Focal | No rotation |
|---|--:|--:|--:|
| 42 | 0.7844 | 0.7331 | 0.7877 |
| 123 | 0.7189 | 0.7301 | 0.7863 |
| 456 | 0.7855 | 0.7690 | 0.7447 |

**Neither experiment produced a statistically established effect.** Both confidence intervals
contain zero and both effect sizes are smaller than the baseline's own seed noise.

---

## 8. Error Analysis

Aggregated confusion matrix across the 3-seed baseline arm — 351 predictions (117 validation
images × 3 seeds). Rows = true class, columns = predicted:

| true / pred | Bird-drop | Clean | Dusty | Electrical | Physical | Snow |
|---|--:|--:|--:|--:|--:|--:|
| **Bird-drop** | **38** | 7 | 5 | 4 | 0 | 3 |
| **Clean** | 7 | **67** | 4 | 0 | 2 | 1 |
| **Dusty** | 5 | 3 | **65** | 1 | 1 | 3 |
| **Electrical-damage** | 3 | 1 | 1 | **33** | 1 | 0 |
| **Physical-Damage** | 10 | 3 | 1 | 1 | **14** | 1 |
| **Snow-Covered** | 3 | 1 | 0 | 0 | 2 | **60** |

Per-class figures (baseline arm, mean of 3 seeds; validation support is per single run, n=117):

| Class | Precision | Recall | F1 | False positives received | Val support |
|---|--:|--:|--:|--:|--:|
| Snow-Covered | 0.8833 | 0.9091 | 0.8944 | 8 | 22 |
| Electrical-damage | 0.8500 | 0.8462 | 0.8457 | 6 | 13 |
| Dusty | 0.8554 | 0.8333 | 0.8441 | 11 | 26 |
| Clean | 0.8268 | 0.8272 | 0.8221 | 15 | 27 |
| Bird-drop | 0.5840 | 0.6667 | 0.6195 | **28** | 19 |
| Physical-Damage | 0.6881 | **0.4667** | 0.5516 | 6 | 10 |

**Bird-drop** — the weakest class by F1, and the only one whose difficulty is not explained by
scarcity. It has 87 training and 19 validation images, more than either fault class, and a
near-neutral class weight (1.0345), so frequency-based rebalancing offers it no assistance. It
accumulates **28 false positives**, the highest of any class, while achieving only 66.7% recall —
the model over-predicts Bird-drop relative to its support. High false-positive volume combined
with low recall indicates poor inter-class separability rather than a well-defined decision
boundary. Its own errors scatter across four classes (Clean 7, Dusty 5, Electrical-damage 4,
Snow 3), consistent with a category the model has not learned to delimit.

**Physical-Damage** — the lowest recall in the matrix at **46.7%**, and the origin of the single
dominant error mode: **10 of 30 instances (33%) are predicted as Bird-drop**, more than twice any
other confusion. Both classes present as small, localized, high-contrast irregularities on an
otherwise uniform panel surface; at 224×224 the distinction between a crack and a dropping is
fine-grained. Its precision (0.6881) substantially exceeds its recall (0.4667): when the model
does predict Physical-Damage it is usually right, but it misses over half of them. **With only 10
validation images, a single prediction moves this class's F1 by ~0.10** — these figures should be
read as indicative, not precise.

**Dusty** (0.8441 F1) and **Clean** (0.8221 F1) — the two largest classes, both performing solidly
and symmetrically (precision ≈ recall in each). Their errors are mutual and modest: Clean → Dusty 4,
Dusty → Clean 3. Clean receives 15 false positives, the second-highest count, mostly from Bird-drop
(7) and Dusty (3), which is consistent with light or partial surface contamination being genuinely
ambiguous against a clean panel.

**Electrical-damage** (0.8457 F1) — the strongest fault class despite having only 13 validation
images, with balanced precision (0.8500) and recall (0.8462) and the joint-fewest false positives
(6). Its visual signature appears well separated from the surface-contamination classes.

**Snow-Covered** (0.8944 F1, 90.9% recall) — the best-performing class overall. It receives 8 false
positives and loses only 6 of 66 instances.

**The pattern across all six.** The three highest-recall classes — Snow-Covered (90.9%),
Electrical-damage (84.6%), Dusty (83.3%) — all alter the appearance of the **whole panel**. The two
weakest — Bird-drop (66.7%) and Physical-Damage (46.7%) — are both **small, localized** defects.
This is consistent with a 98,454-parameter network using global average pooling, which aggregates
spatially and is structurally better suited to global appearance changes than to small local
anomalies.

---

## 9. Scientific Interpretation

### What the experiments establish

- **Training is reproducible on fixed hardware.** Verified to bit-identical model weights across
  independent runs (§4).
- **A real seeding defect existed and was fixed.** Weight initialization was not seed-controlled;
  this is confirmed by direct test, not inference.
- **A real augmentation defect existed and was measured.** `RandomRotation(15)` with default
  `fill=0` blacks out ~5.53% of pixels in ~91% of training images, an artifact present in 0% of
  validation images.
- **Removing rotation reduced cross-seed variance by ~36%** (0.0381 → 0.0244), with the worst seed
  improving by +0.0258.
- **The model is not overfitting.** Training loss exceeds validation loss in every run.
- **Focal loss's predicted mechanism did not occur.** Bird-drop, its explicit target, got worse.

### What the experiments do **not** establish

- **That focal loss is worse than weighted cross-entropy.** The CI [−0.0685, +0.0308] contains
  zero, and an undertraining confound (mean best epoch 46 → 27) is unresolved.
- **That removing rotation improves generalization.** The +0.0100 shift is ~¼ of the noise floor
  and its CI contains zero.
- **Any test-set performance whatsoever.** The test split has never been evaluated. No claim about
  unseen-data performance is made anywhere in this project.
- **That the retained configuration is optimal.** Only three configurations were compared.
- **That tilt invariance is unnecessary.** Removing rotation discards it; the validation set cannot
  measure that cost.

### Why seed variance matters

A single training run conflates the effect being tested with the effect of initialization. Here
that conflation would have been decisive: baseline runs ranged from 0.7189 to 0.7855 — a 0.0666
spread — purely from seed choice. An experiment reporting one run at 0.7855 against one baseline
run at 0.7189 could have claimed a "+0.067 improvement" from a change that did nothing at all.
Reporting mean ± std across shared seeds is what makes any comparison interpretable.

### Why validation-set size limits conclusions

With 117 validation images, one prediction is worth 0.0085 macro-F1, and 3 seeds per arm give a
standard error on the between-arm difference of ~0.025. Only effects around **0.06 macro-F1 or
larger** are detectable. Both experiments produced effects well below that. This is a property of
the measurement apparatus, not of the interventions — a real 0.02 improvement would be
indistinguishable from noise under this design, and honest reporting requires saying so rather
than reading a favourable point estimate as success.

### Why removing a known artifact can be justified without a significant mean improvement

The two experiments were held to deliberately different standards, and the asymmetry is
principled.

Focal loss **adds** a mechanism. Its justification depends entirely on that mechanism producing a
measurable benefit. It did not, so it was rejected.

Removing rotation **subtracts** a measured defect. Training on images where ~5% of pixels have been
replaced by an artifact that never appears at evaluation is worse-specified than not doing so,
independently of the metric. Here the burden of proof runs the other way: keeping the artifact
requires evidence that it *helps*, and no such evidence exists — the model is not overfitting, so
the regularization argument for rotation is weak, while the corruption is directly measured.
"Does no harm and removes a known defect" is a sufficient standard for a subtractive change,
combined with the 36% variance reduction as supporting evidence. This is why the decision is
labelled **provisional** rather than confirmed.

---

## 10. Engineering Lessons

1. **Reproducibility must be verified, not assumed.** The seeding bug produced plausible numbers,
   no warning, and no crash. It was found only by explicitly testing whether two identical runs
   agreed — and it had already silently invalidated earlier comparisons.
2. **Validate checkpoints by reloading them.** Reported metrics are recomputed from the reloaded
   best checkpoint rather than trusted from training-loop memory, which simultaneously verifies
   that serialization round-trips correctly.
3. **Audit data integrity before modeling.** The raw dataset was 77% duplicated. Training on it
   naively would have leaked near-identical images across splits and produced inflated,
   meaningless validation scores.
4. **Control one variable, and verify the control mechanically.** For Experiment 3 all 28 config
   keys were diffed and the resolved transform pipelines compared object-by-object. This caught a
   real error: an early version of the sweep runner would have trained the no-rotation arm with
   **focal loss**, confounding two experiments. The bug was caught before any GPU time was spent.
5. **Negative results are results.** Focal loss is documented as rejected, with its predicted
   mechanism explicitly contradicted. Quietly dropping it would have hidden the most informative
   finding of that experiment.
6. **Inspect augmentation pipelines by measuring them.** The rotation artifact was invisible in the
   configuration file, which described rotation as "simulates handheld/drone camera tilt." Only
   direct pixel measurement revealed the black-corner side effect and the `fill=0` default.
7. **Establish the noise floor before interpreting effects.** Knowing that seed std was 0.0381
   *before* evaluating either experiment is what made honest reporting straightforward rather than
   a judgment call after the fact.
8. **Avoid metric cherry-picking.** Macro-F1 was fixed as the primary metric in advance and
   enforced in code — `TrainingConfig` rejects `accuracy` as a selection metric outright. Focal
   loss's large validation-loss drop (0.7414 → 0.4398) looks like a win in isolation but is
   measured under a different objective and is explicitly excluded from the comparison.

---

## 11. Final Model Decision

**Currently preferred configuration:**

| Component | Setting |
|---|---|
| Architecture | `BaselineCNN`, 98,454 parameters |
| Loss | Class-weighted cross-entropy (inverse frequency, computed from `train.csv` only) |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Scheduler | ReduceLROnPlateau on val macro-F1 (factor 0.5, patience 7) |
| Augmentation | Horizontal flip, ±0 rotation (**disabled**), colour jitter, random-resized-crop (0.9–1.0) |
| Preprocessing | 224×224, bilinear, ImageNet normalization |
| Selection | Highest val macro-F1, `min_delta` 1e-4, ties → lowest val loss |
| Reference performance | 0.7729 ± 0.0244 macro-F1 across seeds 42/123/456 |

**Why:** focal loss was rejected on evidence; rotation removal was retained provisionally because
it eliminates a measured data-corruption artifact, reduced cross-seed variance by ~36%, and did no
harm to the mean. All other components are unchanged from the locked baseline, which remains frozen
for reference.

**This configuration is not production-ready, and no such claim is made.** It has never been
evaluated on the test set. Two of six classes score below 0.60 F1. Cross-seed variance remains
0.0244 — roughly three validation samples wide. A local inference application was added after
these experiments concluded (`app/streamlit_app.py`); no containerization or public deployment
exists. The preference expressed here is a *research* preference among three compared
configurations, nothing more.

---

## 12. Limitations

- **Dataset size.** 540 training images across six classes; the rarest has 46. Small for image
  classification and likely the binding constraint on performance.
- **Class imbalance.** 2.73:1 between largest and smallest class. Handled with weighted
  cross-entropy, but weighting cannot manufacture information 46 images do not contain.
- **Validation-set size caps statistical resolution.** 117 images; one prediction = 0.0085
  macro-F1. Physical-Damage has 10 validation images, moving ~0.10 F1 per sample.
- **No statistically established result.** Both experiments produced effects smaller than seed
  noise; three seeds per arm can resolve only large effects.
- **Dataset provenance is imperfect.** A community-uploaded Kaggle aggregation of images from
  multiple public web sources. The uploader's CC BY-NC-SA 4.0 licence reflects their right to
  license the compilation; it does not establish that every underlying photograph's rights-holder
  consented. No image is claimed as original work, and none is redistributed.
- **Label quality is not independently verified.** 18 images were excluded for sitting in
  near-duplicate clusters with conflicting labels. The true label-error rate across the 772
  manifest paths is unknown — duplicate-based detection cannot catch a uniquely-photographed
  mislabeled image.
- **No source/panel grouping metadata exists.** The split is grouped by near-duplicate cluster,
  the strongest available substitute, but two genuinely different photographs of the same physical
  panel would not be grouped and could leak across splits undetected.
- **Domain shift.** Validation images come from the same aggregated dataset as training. Real user
  photographs differ in camera, lighting, angle, and framing — and are often tilted, which the
  retained configuration is now less equipped to handle.
- **Unresolved confound in Experiment 2.** Focal loss stopped systematically earlier (46 → 27
  epochs); those models may be undertrained rather than inferior.
- **Test set never evaluated.** By design. No generalization claim beyond validation is made.

---

## 13. Future Experiments

**None of the following have been performed.** They are listed in the order the evidence justifies.

1. **NOT YET DONE — Transfer learning (MobileNetV2 / EfficientNet-B0).** The strongest available
   lever. The evidence indicates a capacity- and data-limited model rather than an overfitting one,
   and pretrained ImageNet features address both. The training loop is already
   architecture-agnostic and would require no modification.
2. **NOT YET DONE — Increase seeds per arm to 5–10.** At n=3 only effects ≳0.06 macro-F1 are
   detectable. More seeds would let realistic effect sizes be resolved, and would allow the
   Experiment 3 result to be confirmed or withdrawn.
3. **NOT YET DONE — Rotation with edge-replicate fill.** Experiment 3 removed rotation entirely,
   discarding tilt invariance along with the black-corner artifact. Replicate-padding would
   plausibly retain the benefit without the corruption.
4. **NOT YET DONE — Resolve the focal-loss undertraining confound.** Re-run the focal arm with
   longer early-stopping patience to separate "worse objective" from "stopped too early."
5. **NOT YET DONE — Target the Physical-Damage / Bird-drop confusion.** The dominant error mode
   (33% of Physical-Damage instances). Higher input resolution or targeted data collection are the
   obvious candidates.
6. **NOT YET DONE — Manual review of the 18 excluded label-conflict images**, plus a broader label
   audit of the retained 772.
7. **NOT YET DONE — Grad-CAM explainability**, to check whether the model attends to actual defect
   regions rather than background or panel framing.
8. **NOT YET DONE — Single, final test-set evaluation**, once all modeling decisions are frozen.
   This may be done **exactly once**; any iteration afterwards would invalidate it.
9. **NOT YET DONE — Containerization and public deployment.** A local inference application
   already exists (`app/streamlit_app.py`); hosting it publicly should follow the test-set
   evaluation above, not precede it.

---

*All figures in this report are read from saved artifacts under `experiments/`. Per-run history,
metrics, confusion matrices, classification reports, configurations, and checkpoints are stored per
seed and per arm. See [`../README.md`](../README.md) for the project overview and
[`../DATASET.md`](../DATASET.md) for dataset provenance.*
