# Eval results

Run on 2026-11-03 15:27 with `python evals/run_evals.py`. Images: DMID `1024/TIFF` copies (converted to grayscale and JPEG q92 for transfer, so a rerun on the original PNGs can differ in the third decimal).

Ground truth: DMID metadata (B/M class per lesion; NORM = no abnormality). Expected answers should be reviewed with the challenge owner.

A case passes when the AI priority is in the accepted set for its ground truth: malignant → HIGH or MEDIUM, benign → MEDIUM or LOW, normal → LOW.

| Case | Image | View | Density | Truth | Expected | Mock v1.2 (random, Milestone 2) | GMIC (NYU) ensemble of 5 |
|---|---|---|---|---|---|---|---|
| C01 | IMG471 | MLOLT | G | M | HIGH or MEDIUM (flag for early read) | HIGH (0.410) pass | MEDIUM (0.037) pass |
| C02 | IMG458 | CCLT | - | M | HIGH or MEDIUM (flag for early read) | HIGH (0.805) pass | LOW (0.009) FAIL |
| C03 | IMG161 | CCRT | G | M | HIGH or MEDIUM (flag for early read) | HIGH (0.946) pass | LOW (0.023) FAIL |
| C04 | IMG491 | MLOLT | G | M | HIGH or MEDIUM (flag for early read) | HIGH (0.325) pass | HIGH (0.202) pass |
| C05 | IMG287 | MLOLT | G | M | HIGH or MEDIUM (flag for early read) | HIGH (0.919) pass | MEDIUM (0.040) pass |
| C06 | IMG370 | MLORT | G | M | HIGH or MEDIUM (flag for early read) | HIGH (0.349) pass | LOW (0.020) FAIL |
| C07 | IMG445 | CCRT | G | M | HIGH or MEDIUM (flag for early read) | HIGH (0.277) pass | LOW (0.023) FAIL |
| C08 | IMG355 | CCLT | G | B | MEDIUM or LOW (not urgent) | HIGH (0.537) FAIL | LOW (0.012) pass |
| C09 | IMG219 | CCLT | G | B | MEDIUM or LOW (not urgent) | MEDIUM (0.096) pass | LOW (0.013) pass |
| C10 | IMG158 | MLORT | G | B | MEDIUM or LOW (not urgent) | HIGH (0.672) FAIL | LOW (0.021) pass |
| C11 | IMG286 | CCRT | G | B | MEDIUM or LOW (not urgent) | HIGH (0.777) FAIL | HIGH (0.132) FAIL |
| C12 | IMG432 | CCRT | D | B | MEDIUM or LOW (not urgent) | HIGH (0.854) FAIL | LOW (0.025) pass |
| C13 | IMG272 | CCRT | G | B | MEDIUM or LOW (not urgent) | HIGH (0.839) FAIL | MEDIUM (0.042) pass |
| C14 | IMG426 | MLOLT | G | B | MEDIUM or LOW (not urgent) | HIGH (0.707) FAIL | MEDIUM (0.042) pass |
| C15 | IMG177 | MLORT | G | NORM | LOW (likely normal) | HIGH (0.713) FAIL | LOW (0.008) pass |
| C16 | IMG152 | CCRT | G | NORM | LOW (likely normal) | HIGH (0.663) FAIL | LOW (0.018) pass |
| C17 | IMG123 | CCLT | G | NORM | LOW (likely normal) | HIGH (0.128) FAIL | MEDIUM (0.034) FAIL |
| C18 | IMG240 | CCRT | G | NORM | LOW (likely normal) | HIGH (0.422) FAIL | LOW (0.015) pass |
| C19 | IMG397 | MLOLT | G | NORM | LOW (likely normal) | HIGH (0.397) FAIL | LOW (0.021) pass |
| C20 | IMG509 | CCLT | F | NORM | LOW (likely normal) | HIGH (0.681) FAIL | LOW (0.009) pass |
| C21 | IMG229 | MLOLT | G | NORM | LOW (likely normal) | HIGH (0.831) FAIL | LOW (0.028) pass |
| C22 | IMG006 | MLORT | F | NORM | LOW (likely normal) | HIGH (0.566) FAIL | LOW (0.010) pass |

## Summary

| Metric | Mock v1.2 (random, Milestone 2) | GMIC (NYU) ensemble of 5 |
|---|---|---|
| Cases passed | 8 / 22 | 16 / 22 |
| Malignant flagged (HIGH or MEDIUM) | 7 / 7 | 3 / 7 |
| Normals placed LOW | 0 / 8 | 7 / 8 |
| AUC, malignant vs rest | 0.49 | 0.62 |
| AUC, any lesion vs normal | 0.58 | 0.75 |
| Mean time per image (s) | 0.0 | 6.6 |
