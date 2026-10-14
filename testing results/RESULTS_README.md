# ECA-LDNet reviewer follow-up results

This folder contains the cleaned reviewer-evaluation outputs for the three full-model seeds (17, 43, and 73).

## Included paired evaluations

All included paired CSV files use the fixed `512x512` evaluation protocol and contain per-image PSNR, SSIM, MAE, MSE, RMSE, MS-SSIM, and DeltaE2000 values. Bootstrap summaries and run manifests are stored beside each per-image table.

| Dataset | Pairs per seed | Seed 17 PSNR / SSIM / DeltaE2000 | Seed 43 PSNR / SSIM / DeltaE2000 | Seed 73 PSNR / SSIM / DeltaE2000 |
|---|---:|---:|---:|---:|
| I-HAZE | 5 | 16.567 / 0.7417 / 12.727 | 16.613 / 0.7233 / 12.987 | 16.345 / 0.7410 / 13.194 |
| NH-HAZE | 5 | 11.352 / 0.4191 / 23.544 | 11.219 / 0.4240 / 24.084 | 11.409 / 0.4160 / 23.690 |
| O-HAZE | 45 | 15.423 / 0.6312 / 16.702 | 14.937 / 0.6318 / 17.455 | 15.353 / 0.6366 / 17.258 |
| RESIDE-6K-Test | 1000 | 28.882 / 0.9531 / 2.954 | 28.826 / 0.9564 / 3.006 | 28.774 / 0.9509 / 2.960 |
| SOTS-Indoor | 500 | 28.853 / 0.9491 / 2.743 | 28.722 / 0.9550 / 2.767 | 28.705 / 0.9508 / 2.770 |
| SOTS-Outdoor | 500 | 29.135 / 0.9618 / 3.036 | 29.390 / 0.9636 / 3.041 | 29.214 / 0.9563 / 2.975 |

## Validation blend sweep

The validation-only sweep evaluates blend weights 0, 0.02, 0.05, 0.08, 0.10, and 0.20 for each seed. For observed hazy input, weight 0.08 gives the highest validation PSNR for all three seeds. Seed 17 and seed 43 also have their highest validation SSIM at 0.08; seed 73 has a marginally higher SSIM at 0.10. The complete per-weight values are in `validation_blend_sweep_all_seeds.csv`, with the plot in `validation_blend_sweep.png`.

## Qualitative outputs

`qualitative_figures/selected_predictions/` contains the saved prediction examples for all included datasets and all three seeds. The output names retain the source image stem so that each prediction can be matched to its paired CSV row and reference image.

## Reproducibility

`reviewer_followup_clean.ipynb` is the cleaned rerun notebook. It keeps the validation sweep, paired evaluation, qualitative selection, and cross-dataset figure code for the datasets present in this folder. The notebook is syntax-valid and has no references to excluded datasets.

The incomplete native-resolution rerun output was not included. The paired tables in this folder therefore represent the fixed-square protocol; any resolution-sensitivity claim must be labelled accordingly unless a complete native-aspect-ratio rerun is added later.

