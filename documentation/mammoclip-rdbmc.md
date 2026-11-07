# Mammo-CLIP transfer learning for axillary adenopathy on the Romanian Dense Breast Mammography Collection

**Aim.** Test whether the pretrained Mammo-CLIP image encoder (EfficientNet-B5) can predict, from a single 2D mammogram, whether axillary adenopathy is listed for a breast, using a public dataset. 

## Data

- **Source:** Kaggle `bajcsiadel96/romanian-dense-breast-mammography-collection` (RDBMC, CC BY 4.0). Digital mammography from dense breasts only (ACR composition C and D). Per the README: 251 patients, 450 breasts, 2934 images (2D mammography plus tomosynthesis), with per-breast structured annotations in `metadata.csv`.
- **Images used:** only the **2D MLO mammogram** of each breast (`*_MLO_MAM.png`), because axillary nodes are visible on the MLO view. Tomosynthesis files and CC views were not used. The raw PNGs are about 4096 x 3328 pixels.
- **Size:** `metadata.csv` has 448 breasts from 244 patients. 7 breasts had no downloadable MLO 2D PNG, leaving **441 breasts from 242 patients**.
- **Label:** a breast is positive if its `Mass_Associated features` field lists axillary adenopathy. This gives **49 positive breasts (11.1%) from 47 patients**. All positives have a mass, and 251 of the 441 breasts have a mass.
- **Split:** no fixed train/test split, because positives are few. Instead, **5-fold cross-validation grouped by patient** (stratified on the label), so both breasts of one patient are always in the same fold. All metrics use out-of-fold predictions, so every breast is scored by a model that never saw its patient.
- **Metadata use:** metadata is used to build the label and to group patients.

## Experiment

**Preprocessing.** Grayscale read, Gaussian blur, Otsu threshold, crop to the largest contour's bounding box, resize to 1024 x 640 (H x W). Pixels scaled to 0-1, normalized with mean 0.3089279 / std 0.25053555408335154 (the defaults in the Mammo-CLIP repo `train_classifier.py`), channel repeated 3 times.

**Model.** The Mammo-CLIP EfficientNet-B5 encoder (`Pre-trained-checkpoints/b5-model-best-epoch-7.tar`, Hugging Face `shawn24/Mammo-CLIP`), loaded with `efficientnet_pytorch`. Binary output (adenopathy listed or not).

**Three model runs**
1. **Linear probe, Mammo-CLIP:** frozen encoder; the 2048-number feature vector of each image feeds a logistic regression. 5-fold cross-validation repeated with 3 seeds; predictions averaged.
2. **Linear probe, ImageNet baseline:** identical to run 1 with ImageNet-pretrained B5 weights.
3. **Fine-tuned Mammo-CLIP:** in each fold, the last 2 encoder blocks and the final conv layer are unfrozen (lr 2e-5), a new head learns at lr 1e-3, batch-norm statistics frozen, 8 epochs with cosine decay, class-weighted BCE loss, light augmentation (flip, small shift/rotation, brightness), batch size 2 x 4 accumulation, mixed precision.

**No-image baselines.** Two rules that never look at the image: "the breast has a mass" and "patient age". A model must beat these to show it learned something from the image.

**Metrics.**
- **ROC AUC:** the chance that a random positive breast is scored above a random negative one (0.5 = chance, 1.0 = perfect), with a 95% interval from resampling patients (1000 resamples).
- **AUC among breasts with a mass:** positives only occur in breasts with a mass, so the all-breasts AUC can be inflated by simply detecting a mass. Restricting to breasts that all have a mass is the better test.
- **AP (average precision):** how clean the top of the ranked list is. Chance equals the share of positives (0.111).

## Results

Cross-validated, out-of-fold, 441 breasts (49 positive). Brackets are 95% intervals.

| Model | AUC, all breasts | AUC, breasts with a mass (n=251, 49 positive) | AP (chance 0.111) |
|---|---|---|---|
| Mammo-CLIP, frozen (linear probe) | 0.838 [0.773, 0.895] | 0.787 [0.709, 0.857] | 0.475 |
| Mammo-CLIP, fine-tuned (last blocks, 1 seed) | 0.786 [0.722, 0.850] | 0.714 [0.636, 0.793] | 0.384 |
| ImageNet, frozen (linear probe) | 0.743 [0.669, 0.812] | 0.711 [0.631, 0.790] | 0.272 |
| No image: has a mass | 0.747 | not defined (same score for every breast) | n/a |
| No image: patient age | 0.467 | not reported | n/a |

Across the 3 cross-validation seeds, the all-breasts AUC of the linear probes ranged from 0.815 to 0.844 (Mammo-CLIP) and from 0.708 to 0.757 (ImageNet).

## Interpretation

1. **The frozen Mammo-CLIP linear probe is the best model on every measure.** It clears chance by a wide margin and also clears the no-image "has a mass" rule (0.838 against 0.747), although the lower end of its interval (0.773) sits only just above that rule.
2. **The ImageNet probe adds essentially nothing beyond detecting a mass.** Its AUC of 0.743 equals the 0.747 of the mass-only rule.
3. **Among breasts that all have a mass, Mammo-CLIP still separates the groups** (0.787, interval above 0.5), while ImageNet and the fine-tuned model are clearly lower (about 0.71).
4. **Partial fine-tuning did not help.** It scored lower than the frozen probe on every measure. With 49 positive breasts, overfitting is a plausible reason.
