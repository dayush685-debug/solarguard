# SolarGuard Dataset — Provenance & Download

**This repository does not contain or redistribute the dataset.** `data/` is gitignored in full.
Anyone reproducing this project must download the source data themselves, per the instructions
below.

## Source

- **Official listing:** [kaggle.com/datasets/alicjalena/pv-panel-defect-dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)
- **License:** CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike) — confirmed directly on the
  Kaggle listing. **Non-commercial use only.** This project is a non-commercial educational
  portfolio piece, consistent with that restriction.
- **Raw size:** 1,574 images, 6 classes (Bird-drop, Clean, Dusty, Electrical-damage,
  Physical-Damage, Snow-Covered), pre-split into train/val/test by the original uploader.

## Provenance limitation — stated plainly, not glossed over

This is a Kaggle community-uploaded dataset aggregating images from multiple public web sources.
**The uploader's CC BY-NC-SA 4.0 license reflects their right to license the compilation; it does
not by itself establish that every underlying photograph's original rights-holder consented to
this specific redistribution.** This is a common, largely unavoidable characteristic of publicly
available web-aggregated computer-vision datasets in this niche — it was found to apply, to varying
degrees, to every comparable RGB solar-defect dataset investigated during dataset selection (see
`PLANNING.md`). No image in this dataset is claimed as original work by this project, and none of
it is redistributed here.

## Why 1,574 raw files but 772 used

The raw download is **77% duplicated** (509 exact-duplicate groups, verified independently — see
`PLANNING.md`'s Phase 3 dataset-selection study). After removing exact/near-duplicates and 18
images sitting in unresolved label-conflict clusters, **772 verified-unique images** remain. This
project treats those 772 as the entire modeling population — the raw 1,574 count is never used as
if each file were independent evidence.

## Reproducing the data setup

1. Download the dataset from the Kaggle link above (requires a free Kaggle account).
2. Extract it so the structure is: `data/candidates/PV_Panel_Defect_Dataset/{train,val,test}/{class}/*.jpg`
3. Run `PYTHONPATH=src python scripts/build_manifest.py` — reproduces the deduplicated,
   leakage-safe manifest at `data/splits/manifest.csv` (seed 42, deterministic).
4. Run `PYTHONPATH=src python scripts/compute_statistics.py` — reproduces
   `data/final/dataset_statistics.json` and runs the leakage/label/duplicate verification checks.
5. `PYTHONPATH=src python -m pytest tests/` — confirms the reproduced artifacts match what's
   documented in `PLANNING.md`.

Full methodology, scoring against alternative datasets, and the complete audit trail are in
`PLANNING.md` — this file intentionally contains only what's needed to obtain and verify the data,
kept separate from the pipeline code in `src/` and `scripts/`.
