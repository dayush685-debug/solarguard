# SolarGuard — Model Card

**Status:** research / portfolio model. Not production-ready. Test set not evaluated. A local
inference application exists; no public or production deployment exists.

All figures below are read from saved artifacts under `experiments/` or from source code under
`src/`. Where a property is not documented in the repository, this card says so rather than
estimating it.

---

## 1. Model Summary

SolarGuard performs **six-class single-label image classification** on photographs of solar
panels, assigning each image to one visual condition.

**Classes** (exactly as defined in `data/splits/class_mapping.json`, with their integer indices):

| Index | Class |
|--:|---|
| 0 | Bird-drop |
| 1 | Clean |
| 2 | Dusty |
| 3 | Electrical-damage |
| 4 | Physical-Damage |
| 5 | Snow-Covered |

**Architecture:** `BaselineCNN` — a from-scratch convolutional network defined in
`src/solarguard/models/baseline_cnn.py`, with **98,454 trainable parameters**. It is not a
pretrained or fine-tuned model; no transfer-learning architecture is implemented in this
repository.

**Input:** RGB image, resized to 224 × 224, ImageNet-normalized.
**Output:** 6 raw logits (no softmax applied inside the model).

---

## 2. Intended Use

This model exists to support **research and portfolio demonstration** of an end-to-end,
reproducible machine-learning pipeline: dataset auditing, leakage-safe splitting, controlled
single-variable experimentation, and honest reporting of inconclusive results.

**Appropriate uses:**

- Classifying solar-panel imagery drawn from a distribution similar to the training dataset into
  the six categories above, in a research or educational setting.
- Serving as a documented baseline against which future modeling work (for example transfer
  learning) can be compared.
- Illustrating experimental methodology — reproducibility verification, multi-seed evaluation,
  effect sizes relative to seed variance.

**The distinction from industrial inspection matters and is not cosmetic.** An industrial
inspection system would require evaluation on held-out data (not done here), characterization of
performance under real operating conditions such as varied cameras, lighting, angles, weather and
panel types (not done here), calibrated confidence estimates (not done here), and defined
failure-handling behaviour (not defined here). This model has none of those properties. Its
measured performance describes one validation split of one aggregated public dataset, and nothing
beyond it.

---

## 3. Out-of-Scope Use

This model should **not** be treated as:

- **A certified industrial inspection system.** No certification, standards conformance, or
  external validation of any kind has been performed or claimed.
- **A safety-critical decision system.** It has no calibrated confidence, no abstention mechanism,
  no defined behaviour on out-of-distribution input, and no evaluated error bounds.
- **A replacement for expert inspection.** Two of six classes score below 0.56 F1 on validation
  (Bird-drop 0.545, Physical-Damage 0.556). Physical-Damage recall across the 3-seed baseline arm
  is 46.7% — under half of true instances are found.
- **Evidence of real-world deployment performance.** The test split has never been evaluated, and
  no data outside the source dataset has ever been passed through the model. Validation figures
  are not deployment figures.

No regulatory, compliance, or certification claims are made anywhere in this repository.

---

## 4. Dataset

**Source:** [PV Panel Defect Dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)
(Kaggle). **Licence: CC BY-NC-SA 4.0 — non-commercial use only.** The dataset is not redistributed
in this repository; `data/` is gitignored except for image-free metadata. Full provenance notes are
in [`../DATASET.md`](../DATASET.md).

### Total dataset size vs. experimental manifest

These are two different quantities and are kept distinct throughout:

| Quantity | Value | Meaning |
|---|--:|---|
| **Total dataset images** | **1,574** | Images present on disk in the downloaded dataset |
| **Controlled-experiment manifest paths** | **772** | Verified-unique image paths referenced by `data/splits/manifest.csv` and used for training and validation |

**772 is not the dataset size.** All 1,574 files remain present and unmodified; the 772 are a
selected subset. The reduction comes from the raw download being **77% exact byte-level
duplicates** (509 duplicate groups, independently verified), plus removal of near-duplicates and
18 images sitting in unresolved label-conflict clusters. The 1,574 count is never treated as 1,574
pieces of independent evidence.

### Split

Constructed with seed 42, stratified by class and grouped by near-duplicate cluster so an image
and its near-copy cannot land on opposite sides of the split:

| Split | Images |
|---|--:|
| Train | 540 |
| Validation | 117 |
| Test | 115 — **never loaded or evaluated** |

### Class distribution (training split, n = 540)

| Class | Train n | % of train | Inverse-frequency class weight |
|---|--:|--:|--:|
| Clean | 126 | 23.3% | 0.7143 |
| Dusty | 123 | 22.8% | 0.7317 |
| Snow-Covered | 100 | 18.5% | 0.9000 |
| Bird-drop | 87 | 16.1% | 1.0345 |
| Electrical-damage | 58 | 10.7% | 1.5517 |
| Physical-Damage | 46 | 8.5% | 1.9565 |

**Imbalance ratio: 2.73:1** (Clean vs Physical-Damage).

Validation support per class (n = 117): Clean 27, Dusty 26, Snow-Covered 22, Bird-drop 19,
Electrical-damage 13, Physical-Damage 10.

**Verified image properties:** 0 corrupt files, 453 distinct resolutions, median 720 × 632 px,
aspect ratios 1.0 to 3.6.

---

## 5. Model Architecture

Read directly from `src/solarguard/models/baseline_cnn.py`.

Each of the four convolutional blocks is identical in structure, differing only in channel count:

```
Conv2d(in, out, kernel_size=3, padding=1, bias=False)
BatchNorm2d(out)
ReLU(inplace=True)
MaxPool2d(kernel_size=2)
```

`bias=False` on the convolutions because BatchNorm's own shift parameter makes a separate conv bias
redundant.

| Stage | Composition | Output shape | Parameters |
|---|---|---|--:|
| Input | — | 3 × 224 × 224 | — |
| Block 1 | Conv(3→16) → BN → ReLU → MaxPool | 16 × 112 × 112 | 464 |
| Block 2 | Conv(16→32) → BN → ReLU → MaxPool | 32 × 56 × 56 | 4,672 |
| Block 3 | Conv(32→64) → BN → ReLU → MaxPool | 64 × 28 × 28 | 18,560 |
| Block 4 | Conv(64→128) → BN → ReLU → MaxPool | 128 × 14 × 14 | 73,984 |
| Pooling | `AdaptiveAvgPool2d(1)` → flatten | 128 | 0 |
| Regularization | `Dropout(p=0.5)` | 128 | 0 |
| Output | `Linear(128, 6)` | 6 logits | 774 |
| **Total** | | | **98,454** |

- **Activation:** ReLU throughout the convolutional blocks. No activation is applied after the
  output layer.
- **Pooling:** `MaxPool2d(kernel_size=2)` after each block (four times, 224 → 14 spatially), then
  **global average pooling is used** — `AdaptiveAvgPool2d(1)` reduces the 128 × 14 × 14 feature map
  to a 128-vector. There is no `Flatten` + large fully-connected head.
- **Output layer:** a single `Linear(128, 6)` producing **raw logits**. Softmax is not applied
  inside the model; the loss applies `log_softmax` internally.

---

## 6. Training Configuration

Values below are read from a saved experiment artifact
(`experiments/baseline_3seed/seed_42/config.yaml`) and from the locked checkpoint's embedded
configuration — not from source defaults.

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| LR scheduler | `ReduceLROnPlateau` on validation macro-F1, factor 0.5, patience 7 |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping | Patience 15 epochs on validation macro-F1 |
| Seed | 42 (locked baseline); experiments additionally use 123 and 456 |
| Loss | Class-weighted cross-entropy |
| Class weighting | Inverse frequency, `n_train / (n_classes × count[class])`, computed from `train.csv` only — never from validation or test labels |
| Input resolution | 224 × 224, bilinear interpolation |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Checkpoint selection | Highest validation macro-F1, `min_delta = 1e-4`, ties broken by lowest validation loss |
| `num_workers` | 0 |
| Device | CUDA |

### Preprocessing and augmentation

**Training pipeline** (locked baseline, `configs/preprocessing.yaml`), as resolved from the code:

```
RandomResizedCrop(224, scale=(0.9, 1.0)) → convert RGB → RandomHorizontalFlip(p=0.5)
→ RandomRotation(15) → ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0)
→ ToTensor → Normalize
```

**Training pipeline** (Experiment 3 variant, `configs/preprocessing_norot.yaml`) — identical except
`RandomRotation` is absent:

```
RandomResizedCrop(224, scale=(0.9, 1.0)) → convert RGB → RandomHorizontalFlip(p=0.5)
→ ColorJitter(...) → ToTensor → Normalize
```

**Validation pipeline** (both configs, identical): `Resize(224, 224) → convert RGB → ToTensor →
Normalize`. No augmentation is applied at evaluation time. Vertical flip and random erasing are
disabled in both configurations.

### Not documented here

- **Total wall-clock training cost across all experiments** — per-run times are recorded in each
  run's `metrics.json`, but no aggregate figure is stated in this repository.
- **Hardware used for the locked baseline run** specifically — the checkpoint records `device:
  cuda` but not the GPU model.
- **Any hyperparameter search.** None was performed; all values above are the initial choices,
  never tuned.

---

## 7. Evaluation

### Locked baseline — validation results

Single reference run, seed 42, best checkpoint at **epoch 23**, evaluated on the **117-image
validation split**:

| Metric | Value |
|---|--:|
| Accuracy | 0.7692 |
| Macro precision | 0.7639 |
| Macro recall | 0.7504 |
| **Macro F1** | **0.7534** |
| Weighted F1 | 0.7744 |
| Validation loss | 0.8091 |

### Why macro-F1 is the emphasized metric

The six classes are not equally represented — Clean and Dusty together are 46.1% of the training
split, while Physical-Damage is 8.5%, an imbalance ratio of 2.73:1. Macro-F1 averages per-class F1
**without weighting by class frequency**, so every class contributes equally regardless of size.
Accuracy and weighted-F1 both weight by support, meaning a model can score well on either while
performing poorly on the rarest classes.

This is directly observable here: the baseline's weighted-F1 (0.7744) exceeds its macro-F1 (0.7534)
by 0.0210, and that gap is the majority-class skew made visible. Macro-F1 is fixed as the primary
metric in code — `TrainingConfig` rejects `accuracy` as a selection metric outright — so it cannot
be swapped after seeing results.

### Class-level metrics — locked baseline

| Class | F1 | Validation n |
|---|--:|--:|
| Electrical-damage | 0.889 | 13 |
| Snow-Covered | 0.884 | 22 |
| Clean | 0.863 | 27 |
| Dusty | 0.784 | 26 |
| Physical-Damage | 0.556 | 10 |
| Bird-drop | 0.545 | 19 |

### Class-level metrics — 3-seed baseline arm (mean of seeds 42, 123, 456)

Reported separately because they come from a different set of runs and include precision and
recall:

| Class | Precision | Recall | F1 |
|---|--:|--:|--:|
| Snow-Covered | 0.8833 | 0.9091 | 0.8944 |
| Electrical-damage | 0.8500 | 0.8462 | 0.8457 |
| Dusty | 0.8554 | 0.8333 | 0.8441 |
| Clean | 0.8268 | 0.8272 | 0.8221 |
| Bird-drop | 0.5840 | 0.6667 | 0.6195 |
| Physical-Damage | 0.6881 | 0.4667 | 0.5516 |

**No test-set evaluation has been performed.** Every figure on this card is a validation figure.

---

## 8. Experimental Findings

Two controlled single-variable experiments were completed, each run across seeds 42, 123, 456 with
dataset, split, architecture, optimizer, learning rate, scheduler, batch size, and selection rule
held constant. Full reasoning is in [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md).

The baseline's own seed-to-seed macro-F1 standard deviation is **0.0381**, which is the noise floor
any claimed effect must exceed.

### Focal Loss (γ = 2)

Weighted focal loss replacing weighted cross-entropy, retaining the same class weights as the α
term. Verified to reduce to the baseline loss exactly at γ = 0 (bit-identical gradients).

| Quantity | Value |
|---|---|
| Baseline macro-F1 | 0.7629 ± 0.0381 |
| Focal macro-F1 | 0.7441 ± 0.0216 |
| Absolute change | −0.0189 |
| **95% CI on the difference** | **[-0.0685, +0.0308]** — contains zero |
| Seeds won | 1 of 3 |

**This experiment did not establish a statistically reliable macro-F1 improvement over the
weighted cross-entropy baseline.** The point estimate favours the baseline, the confidence interval
spans zero, and the change is smaller than the seed-to-seed noise floor. No improvement is claimed.
The configuration was **rejected**; the baseline loss was retained.

The predicted mechanism was also contradicted: focal loss was expected to help Bird-drop (a
confusable rather than rare class), but Bird-drop's F1 fell from 0.6195 to 0.5786. One confound
remains unresolved — focal loss stopped systematically earlier (mean best epoch 46.0 → 27.0), so
those models may be undertrained rather than inferior.

### No Rotation Augmentation

`RandomRotation(15)` removed from the training pipeline; every other setting unchanged (verified
by parsing and diffing all 28 configuration keys — exactly one functional difference).

| Quantity | Value |
|---|---|
| Baseline macro-F1 | 0.7629 ± 0.0381 |
| No-rotation macro-F1 | 0.7729 ± 0.0244 |
| Mean macro-F1 shift | **≈ +0.0100** |
| **95% CI on the difference** | **[-0.0413, +0.0612]** — contains zero |
| Seed standard deviation | **0.0381 → 0.0244** (≈ **36% reduction**) |
| Worst-seed macro-F1 | 0.7189 → 0.7447 |
| Seeds won | 2 of 3 |

**The +0.0100 shift is not statistically established.** It is approximately one quarter of the
seed-to-seed noise floor and its confidence interval contains zero.

**Measured rotation artifact.** `RandomRotation` defaults to `fill=0`. Direct measurement over 200
augmented samples found it blacks out a mean of **≈5.53% of pixels (max 10.08%)**, affecting
**≈91% of training images**. Validation images contain **0%** black pixels — the artifact exists
only in training.

**Decision: provisionally retained**, on the grounds that it removes a measured data-corruption
artifact and reduced cross-seed variance by ≈36%, not on the strength of the mean.

> **This is evidence, not proof of improved real-world generalization.** No claim is made that
> removing rotation improves performance on unseen data. A cost is also incurred that this
> validation set cannot measure: real photographs are often tilted, and removing rotation removes
> tilt invariance.

---

## 9. Limitations

- **Limited validation sample size.** 117 images. One prediction changes macro-F1 by ≈0.0085.
  Physical-Damage has only 10 validation images, so its F1 moves ≈0.10 per single prediction — that
  class's score should be read as indicative, not precise.
- **Class imbalance.** 2.73:1 between largest and smallest class. Handled with inverse-frequency
  weighted cross-entropy, but weighting cannot create information that 46 training images do not
  contain.
- **Dataset quantity.** 540 training images across six classes; the rarest has 46. This is small
  for image classification and is likely the binding constraint on performance.
- **Label quality is not independently verified.** 18 images were excluded for sitting in
  near-duplicate clusters with conflicting labels. The true label-error rate across the 772
  manifest paths is unknown — duplicate-based detection cannot catch a uniquely-photographed
  mislabeled image.
- **Dataset provenance is imperfect.** A community-uploaded aggregation of images from multiple
  public web sources. The uploader's CC BY-NC-SA 4.0 licence reflects their right to license the
  compilation; it does not establish that every underlying photograph's rights-holder consented.
- **No source/panel grouping metadata exists.** The split is grouped by near-duplicate cluster, the
  strongest available substitute, but two genuinely different photographs of the same physical
  panel would not be grouped and could leak across splits undetected.
- **Domain shift.** Validation images are drawn from the same aggregated dataset as training. Real
  photographs will differ in camera, lighting, angle, framing, weather, panel type, and tilt. The
  validation split does not represent the full range of real-world imaging conditions.
- **Statistical uncertainty.** Neither completed experiment produced a statistically established
  effect; both confidence intervals contain zero. With three seeds per arm, only effects of roughly
  0.06 macro-F1 or larger are detectable.
- **No test-set evaluation at this project stage.** The 115-image test split has never been loaded
  or evaluated, by design. No generalization claim beyond validation is made.
- **Unresolved confound in the focal-loss experiment.** Systematically earlier stopping means those
  models may be undertrained rather than genuinely worse.

---

## 10. Bias / Failure Modes

Based on the aggregated confusion matrix across the 3-seed baseline arm (351 predictions = 117
validation images × 3 seeds):

| true / pred | Bird-drop | Clean | Dusty | Electrical | Physical | Snow |
|---|--:|--:|--:|--:|--:|--:|
| **Bird-drop** | **38** | 7 | 5 | 4 | 0 | 3 |
| **Clean** | 7 | **67** | 4 | 0 | 2 | 1 |
| **Dusty** | 5 | 3 | **65** | 1 | 1 | 3 |
| **Electrical-damage** | 3 | 1 | 1 | **33** | 1 | 0 |
| **Physical-Damage** | 10 | 3 | 1 | 1 | **14** | 1 |
| **Snow-Covered** | 3 | 1 | 0 | 0 | 2 | **60** |

### Bird-drop — highest false-positive rate

Bird-drop accumulates **28 false positives**, the most of any class, while achieving only **66.7%
recall**. The model over-predicts Bird-drop relative to its support. This combination of high
false-positive volume and low recall indicates poor inter-class separability rather than a
well-defined decision boundary. Its own errors scatter across four classes (Clean 7, Dusty 5,
Electrical-damage 4, Snow-Covered 3).

Notably, Bird-drop is **not** among the rarest classes — 87 training and 19 validation images, with
a near-neutral class weight of 1.0345. Its weakness is therefore not explained by scarcity, and
frequency-based rebalancing offers it no assistance.

### Physical-Damage — lowest recall, dominant single error mode

Physical-Damage has the lowest recall in the matrix at **46.7%** — under half of true instances are
identified. The dominant error mode across the entire matrix is **Physical-Damage predicted as
Bird-drop: 10 of 30 instances (33%)**, more than twice any other confusion.

Its precision (0.6881) substantially exceeds its recall (0.4667): when the model does predict
Physical-Damage it is usually correct, but it misses most of them. **Operationally this is the
least desirable direction of error for a defect-detection task** — physical damage is more likely
to be missed than falsely reported.

With only 10 validation images, these figures carry substantial measurement uncertainty.

### Systematic pattern

The three highest-recall classes — Snow-Covered (90.9%), Electrical-damage (84.6%), Dusty (83.3%) —
all alter the appearance of the **whole panel**. The two weakest — Bird-drop (66.7%) and
Physical-Damage (46.7%) — are both **small, localized** defects. This is consistent with a
98,454-parameter network using global average pooling, which aggregates spatially and is
structurally better suited to global appearance changes than to small local anomalies.

**Bias note.** No demographic or protected-attribute bias analysis applies to this model — the
inputs are photographs of equipment, not people. The bias present is **class-level performance
disparity**: the model is substantially less reliable on Bird-drop and Physical-Damage than on the
other four classes, and any use would inherit that disparity.

---

## 11. Reproducibility

**Verified, not assumed.** Two independent runs of an identical configuration produced
byte-identical results across all 83 epochs of `history.csv`, identical confusion matrices,
identical classification reports, and **bit-identical model weights** in the saved checkpoint.

**Seeds.** `set_seed()` in `src/solarguard/training/train.py` covers Python `random`, NumPy,
PyTorch CPU and CUDA, and sets `cudnn.deterministic=True`, `cudnn.benchmark=False`. It is called
**before model construction**, so weight initialization is genuinely seed-controlled. DataLoaders
receive a seeded generator and a `worker_init_fn`. Seeds used: **42** (locked baseline),
**42, 123, 456** (all controlled experiments).

**Saved artifacts** — per run, under `experiments/<arm>/seed_<n>/`:

```
checkpoint_best.pt          model + optimizer state, epoch, config, class mapping, seed
history.csv                 per-epoch train/val loss, accuracy, macro-F1, weighted-F1, LR, time
metrics.json                headline + per-class precision/recall/F1 from the best checkpoint
classification_report.json  full per-class breakdown
confusion_matrix.csv        raw counts
config.yaml                 complete training configuration
class_mapping.json          class-to-index mapping
training_curves.png         loss and macro-F1 curves with the best epoch marked
```

Per-arm summaries at `experiments/<arm>/summary.csv`.

**Experiment arms:**

| Arm | Directory |
|---|---|
| Locked baseline | `experiments/baseline_cnn/colab_run_20260818_170845/` |
| 3-seed baseline | `experiments/baseline_3seed/` |
| Experiment 2 — focal loss | `experiments/exp2_focal_gamma2/` |
| Experiment 3 — no rotation | `experiments/exp3_no_rotation/` |

**Configuration files:** `configs/preprocessing.yaml` (baseline),
`configs/preprocessing_norot.yaml` (Experiment 3 variant). Split definition and seed:
`data/splits/split_config.json`.

**Evaluation procedure.** Reported per-class metrics are produced by **reloading the selected best
checkpoint and re-running inference on the validation set** — not carried over from the training
loop's in-memory state. This also verifies that checkpoints round-trip faithfully. Evaluation uses
`build_eval_transform()`, which applies no augmentation and is shared with the serving path in
`src/solarguard/inference/predictor.py`, so the two cannot drift apart.

**To reproduce:**

```bash
pip install -e ".[dev]"
pip install torch torchvision   # not pinned in pyproject.toml: Colab ships its own CUDA build
# obtain the dataset per DATASET.md into data/candidates/PV_Panel_Defect_Dataset/
python scripts/build_manifest.py        # deduplicated, leakage-safe split (seed 42)
python scripts/compute_statistics.py    # dataset statistics + verification
python -m pytest tests/                 # 113 tests
python scripts/run_seed_sweep.py --arm baseline   # or: focal | norot
```

**Honest limit:** reproducibility is verified on identical hardware. Results are not guaranteed to
match bit-for-bit across different GPU architectures, since cuDNN may select different algorithms.

---

## 12. Deployment Considerations

**A local inference application exists. There is no public or production deployment.**

### What exists and has been verified

| Component | Path | Status |
|---|---|---|
| Deployment checkpoint | `models/solarguard_baseline_v1.pt` | 406 KB; weights verified bit-identical to the locked baseline |
| Export script | `scripts/export_deployment_checkpoint.py` | strips `optimizer_state_dict` (1.21 MB → 406 KB); source opened read-only, checksum-verified unmodified |
| Inference pipeline | `src/solarguard/inference/predictor.py` | single-image prediction with top-k and full distribution |
| Local application | `app/streamlit_app.py` | verified to start and serve **HTTP 200**, no errors logged |
| Tests | `tests/test_inference.py` | 23 inference tests; **113/113 pass** repository-wide |

Two correctness properties are enforced by test rather than convention:

1. **Serving preprocessing is identical to validation preprocessing.** The predictor imports
   `build_eval_transform()` — the same function behind every validation metric on this card —
   and a test asserts both produce `torch.equal` tensors.
2. **Class labels come from the checkpoint**, which is self-describing, rather than from a
   separate file that could silently diverge and mislabel every prediction.

The served artifact records its own provenance: source checkpoint, trained epoch (23), seed (42),
and validation macro-F1 (0.7534). A spot check on 12 real validation images returned 10/12
correct, with both failures being Physical-Damage predicted as Bird-drop — the documented
dominant error mode.

**No public endpoint, hosted instance, container configuration, or cloud deployment exists.**
The model is **not production-ready** and no such claim is made. The application surfaces the
limitations below in a persistent panel rather than hiding them.

### What would need validating before any deployment

- **Test-set evaluation.** The 115-image test split must be evaluated **once**, after all modeling
  decisions are frozen, to obtain an unbiased performance estimate. Until then no
  generalization figure exists at all.
- **Real-world evaluation.** Performance must be measured on photographs actually representative of
  the deployment setting — not on a held-out slice of the same aggregated public dataset.
- **Domain shift characterization.** Validation images share their distribution with training
  images. Real inputs will differ in camera hardware, resolution, lighting, weather, angle, panel
  type, and framing. Additionally, the currently preferred configuration has **rotation
  augmentation removed**, so it has less tilt invariance — and real photographs are frequently
  tilted. That cost is invisible to this validation set.
- **Confidence interpretation.** The model outputs raw logits. Softmax over them yields values that
  sum to one, but these are **not calibrated probabilities** — no calibration analysis has been
  performed. They should not be presented to a user as confidence percentages without calibration.
- **Failure-mode handling.** Physical-Damage recall is 46.7%, so a naive deployment would miss over
  half of physically damaged panels. Any real system would need an abstention or
  escalate-to-human path, which does not exist here.
- **Out-of-distribution behaviour.** The model will confidently assign one of six labels to any
  input, including images that are not solar panels at all. There is no rejection mechanism.

### Why current metrics are not production performance

The reported figures describe **one validation split of one aggregated public dataset**, measured
on 117 images, with cross-seed variance of 0.0244–0.0381 macro-F1. They were computed on data drawn
from the same distribution as training, after model selection was performed against that same
validation set. They are appropriate for comparing configurations against each other, which is what
they were used for. They are not an estimate of performance on new data from a different source,
and should not be quoted as one.

---

## 13. Ethical / Safety Considerations

- **No personal data.** The dataset consists of photographs of equipment. No people, faces, or
  personally identifying information are involved, and no demographic bias analysis applies.
- **Licence compliance.** The dataset is CC BY-NC-SA 4.0 — **non-commercial use only**. This
  project is non-commercial and educational. The dataset is not redistributed here, and any
  downstream use of the trained weights inherits the non-commercial restriction.
- **Provenance honesty.** The source is a community aggregation of web images whose original
  rights-holders cannot all be verified. No image is claimed as original work by this project. This
  limitation is stated rather than glossed over.
- **Risk of over-trust.** The main foreseeable harm is someone treating validation metrics as
  deployment performance and relying on the model to find defects it demonstrably misses —
  Physical-Damage recall is 46.7%. This card exists partly to make that misuse harder.
- **No safety, medical, or regulatory claims** are made. This model has no certification, meets no
  standard, and has not been externally validated.

---

## 14. Model Card Status

| Property | Status |
|---|---|
| **Model type** | Research / portfolio |
| **Production-ready** | **No** |
| **Test set evaluated** | **No** — 115 images held out, never loaded |
| **Local application** | **Yes** — `app/streamlit_app.py`, verified serving HTTP 200 |
| **Public / production deployment** | **No** — no endpoint, hosted instance, or container exists |
| **External validation** | None |
| **Certification / regulatory approval** | None claimed |
| Locked baseline macro-F1 (validation) | 0.7534 |
| Currently preferred configuration | Baseline architecture and loss, rotation augmentation removed (provisional) |
| Statistically established improvements | **None** — both experiments produced effects smaller than seed noise |

This card describes a model built to demonstrate reproducible experimental methodology. Its value
lies in the documented process — dataset auditing, leakage-safe splitting, controlled
single-variable experiments, multi-seed evaluation, and honest reporting of negative and
inconclusive results — rather than in its absolute accuracy.

---

*Related documentation: [`../README.md`](../README.md) (project overview),
[`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) (full experimental reasoning),
[`../DATASET.md`](../DATASET.md) (dataset provenance).*
