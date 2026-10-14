# Evaluation results index: seeds 17 and 73

Updated 2026-10-04. The complete per-image CSVs and prediction images remain in the two run folders listed below. `EVALUATION_SUMMARY_SEEDS17_73.csv` collects all PSNR, SSIM, and MS-SSIM rows with sample counts, bootstrap intervals, and checkpoint paths where recorded.

## Run folders

- `test_eval_seed17_seed73/test_eval/`: earlier `radiometric` evaluation; original run did not save an evaluation manifest, so its exact checkpoint paths are not recorded.
- `test_eval_submitted_seed17_seed73/`: `submitted` preprocessing, 256x256, no TTA; manifest records the checkpoint and dataset paths.

## Protocol and interpretation

The revised seed-17/seed-73 checkpoints were trained with `radiometric` preprocessing. Therefore, the `submitted` evaluation is a preprocessing-mismatch diagnostic for these checkpoints, not the primary score to report for the revised training protocol. Keep both runs; do not combine their means or treat them as direct model comparisons.

The submitted-protocol manifest used `stage4_best.keras` for seed 17 and `stage4_last.keras` for seed 73. For a final paper table, use a checkpoint selected only by validation for each seed, use the same preprocessing as training, and save an evaluation manifest. The earlier radiometric run lacks a manifest, so rerun it on the chosen checkpoints with manifest enabled before citing it.

## PSNR results (dB)

| Dataset | Seed 17 radiometric | Seed 73 radiometric | Seed 17 submitted diagnostic | Seed 73 submitted diagnostic |
|---|---:|---:|---:|---:|
| RESIDE-6K | 29.845 [29.666, 30.029] | 29.410 [29.226, 29.593] | 28.269 [28.074, 28.469] | 28.072 [27.870, 28.275] |
| SOTS-Indoor | 29.884 [29.661, 30.114] | 29.490 [29.261, 29.723] | 28.562 [28.311, 28.817] | 28.126 [27.858, 28.399] |
| SOTS-Outdoor | 29.659 [29.385, 29.933] | 29.057 [28.785, 29.329] | 27.969 [27.672, 28.260] | 27.982 [27.682, 28.269] |

Values are image means with 95% image-bootstrap confidence intervals; n=1,000 for RESIDE-6K and n=500 for each SOTS split.

## Next steps

1. Confirm the validation-best Stage 4 checkpoint for each existing seed, then rerun the primary evaluation with `radiometric` preprocessing and record checkpoint paths in the manifest.
2. The reviewer explicitly requests at least three independent seeds for the main results and ablations. Seed 43 is a reasonable third seed (the number itself has no special significance), but run it only after freezing the training/evaluation protocol. Train the full planned curriculum with the same configuration as seeds 17 and 73, then evaluate with `radiometric` and report mean ± standard deviation across seeds.
3. Keep the `submitted` run as a diagnostic. Do not use it to replace the main score for these radiometric-trained checkpoints.
4. After the three-seed results, prioritize the reviewer-requested ablations, blend/epsilon sensitivity, real/near-real evaluation, and controlled comparisons. Seed 43 alone does not complete the revision.
