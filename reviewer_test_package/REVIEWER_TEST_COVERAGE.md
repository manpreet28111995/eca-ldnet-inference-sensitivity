# Test package coverage and remaining reviewer experiments

This ZIP is a repeatable **evaluation harness**, not proof that the reviewer response is complete. It contains the three full-model seeds and the four ablation families that already have finished Stage 4 checkpoints.

## Run in the Kaggle notebook

- Evaluate all 15 checkpoints: full seeds 17/43/73 and `no_eca`, `eca_only`, `no_attention`, `no_physics` for those same seeds.
- Paired data: RESIDE-6K test only if a distinct test pair folder exists; SOTS Indoor/Outdoor; O-HAZE, I-HAZE, NH-HAZE and DNH-HAZE when attached.
- Unpaired data: RTTS when attached.
- Paired metrics: PSNR, SSIM, MS-SSIM, MAE/MSE/RMSE, CIEDE2000, and optional LPIPS/DISTS; per-image rows and image-bootstrap 95% intervals.
- Unpaired proxies: NIQE, BRISQUE, MUSIQ, MANIQA, CLIP-IQA, optional FADE, with hazy/dehazed scores and bootstrap intervals.
- Save every valid evaluated prediction under `tested_sample_<dataset>`, plus run JSON, checkpoint SHA-256, pair counts, per-model CSVs, and across-seed means/sample SD.
- Benchmark batch-1 latency/throughput/GPU peak memory at 256, 512, 1280×720, and 1920×1080 on the same Kaggle GPU. An OOM or unsupported size is recorded as a failure; larger sizes are not marked successful.

## Reviewer requests that still need experiments or manuscript work

These are not answered by evaluating the existing checkpoint set:

1. Train matched SE/CBAM, spatial-only, attention-order/parallel, and capacity-matched controls on the same split and schedule. Only the four named ablation families above currently have packaged three-seed checkpoints.
2. Run the validation-only ASM blend/epsilon sweep (0, 0.02, 0.05, 0.08, 0.10, 0.20; epsilon 0.05/0.10/0.20) and a constrained learnable-blend comparison. Do not select from the held-out test sets.
3. Run controlled training comparisons that isolate primary vs polish loss, EMA, and late-LR changes. Stage histories alone cannot establish causal effects.
4. Probe atmospheric-light/transmission outputs against ground truth only if the attached data really contain aligned A/t labels. Otherwise describe them as uncalibrated latent guidance and narrow physical claims.
5. Revise the manuscript and point-by-point response: novelty/related work, claims/limitations, architecture and qualitative figures, English, and separate Ethics/Consent declarations. The notebook does not make those edits.

The ZIP includes the existing training/evaluation code and the blend/capacity scripts as starting points. A missing experiment must remain labeled missing until its training run, history, checkpoint, and evaluation output are produced.
