## Data
 
- **Source:** Kaggle `orvile/cbis-ddsm-dataset`. Full mammograms (`*_FULL_PRE.png`), e.g. 4955 x 4955 pixels.
- **Size:** 882 annotated images; 881 readable and processed.
- **Labels** come from per-image tags: pathology (`malignant` 362, `benign` 413, `benign_without_callback` 107) and lesion type (`mass`, `calcification`).
- **Split:** by **participant** (`P_#####` from the filename), because each participant has several image/patient IDs (eg left/right). Official train and test were pooled and re-split 70/15/15; no participant appears in two splits.

## Experiment

**Preprocessing.** Grayscale read, Gaussian blur, Otsu threshold, crop to the largest contour's bounding box, resize to 1024 x 640 (H x W). Pixels scaled to 0-1, normalized with mean 0.3089279 / std 0.25053555408335154 (the defaults in the Mammo-CLIP repo `train_classifier.py`), channel repeated 3 times.
 
**Model.** The Mammo-CLIP EfficientNet-B5 encoder (`Pre-trained-checkpoints/b5-model-best-epoch-7.tar`, Hugging Face `shawn24/Mammo-CLIP`), loaded with `efficientnet_pytorch`. A linear head (2048 -> 3, multi-label) with class-weighted BCE loss.
 
**Three runs**
1. **Linear probe, Mammo-CLIP:** frozen encoder, AdamW lr 1e-3, 15 epochs.
2. **Fine-tuned Mammo-CLIP:** last 2 blocks + head conv unfrozen (lr 1e-5), head lr 1e-4, batch-norm statistics frozen, 10 epochs.
3. **Linear probe, ImageNet baseline:** same as run 1 with ImageNet-pretrained B5 weights.
Batch size 4 x 2 accumulation (2 x 4 for fine-tuning)
 
**Metric.** ROC AUC: the chance that a random positive image is scored above a random negative one (0.5 = chance, 1.0 = perfect).

## Results
 
### Test set (128 images, 85 participants)
 
| Model | Malignant | Mass | Calc |
|---|---|---|---|
| Mammo-CLIP, frozen | 0.676 | 0.678 | 0.680 |
| Mammo-CLIP, fine-tuned | 0.673 | 0.688 | 0.686 |
| ImageNet, frozen | 0.653 | 0.626 | 0.631 |