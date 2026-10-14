# ECA-LDNet: Engineering Characterization of a Single-Image Dehazing Network

This repository contains code and experiment artifacts for an engineering characterization of ECA-LDNet. The study examines how inference-time settings affect outputs from three fixed, validation-selected checkpoints. It does not claim a new attention operator, state-of-the-art performance, or deployment readiness.

ECA-LDNet is a depthwise-separable U-Net with established Efficient Channel Attention (ECA) and pixel-attention blocks, plus a low-weight Atmospheric Scattering Model (ASM)-inspired correction branch. The network has **1,481,871 parameters** and **2.171 GMACs (4.343 GFLOPs) at 256 × 256**. Its atmospheric-light-like and transmission-like predictions are not physically calibrated estimates.

## Main finding

The three Stage 4 checkpoints were evaluated with their weights fixed while the ASM blend and denominator stabilizer `ε` varied:

- On the reused grouped validation split of 2,066 pairs, blend `0.08` and `ε = 0.1` gave the highest tested mean PSNR. At fixed blend `0.08`, `ε = 0.2` gave the highest tested mean SSIM.
- In nested five-fold selection over all 45 O-HAZE pairs, every fold selected blend `0.20` and `ε = 0.01`. The pooled out-of-fold change against `0.08 / 0.1` was **+0.686 dB PSNR** and **+0.0171 SSIM**.

These are inference-only output recombinations. The validation sweep reused the checkpoint-selection split; O-HAZE had been evaluated previously, and the nested selections landed on tested grid boundaries. Results therefore describe sensitivity for these checkpoints, not a transferable optimum or an isolated synthetic-to-real domain effect.

## Evaluation scope

Training used RESIDE-6K training data and the Indoor Training Set (ITS). A shared 90/10 split was grouped by SHA-256 hashes of clean targets (split seed 42), which reduces exact-target leakage but cannot ensure scene-level independence.

The primary evaluation used SOTS-Indoor and SOTS-Outdoor (500 pairs each, resized to 256 × 256), O-HAZE (45 pairs at native resolution), and five-pair test subsets from I-HAZE and NH-HAZE (resized to 256 × 256). Follow-up analyses include the 1,000-pair RESIDE-6K-Test partition, all 55 pairs in the Dense-Haze release, and all labeled I-HAZE and NH-HAZE partitions. Dense-Haze results pool development/training, validation, and challenge-test images; they are not an official challenge-test score. Each I-HAZE and NH-HAZE test partition contains five pairs.

The batch-one timing profile used sequential calls on a Kaggle runtime with two Tesla T4 GPUs visible. End-to-end throughput was 2.94 images/s at 256 × 256 and 2.90 images/s at 512 × 512. These measurements describe that runtime and do not establish deployment performance.

## Repository layout

- `eca_ldnet/` and `scripts/`: current model, training/evaluation code, and supporting command-line tools.
- `tests/`, `pyproject.toml`, and `requirements.txt`: test suite and package/dependency metadata.
- `source_snapshots/`: only files that differ in the historical ablation and reviewer-evaluator code snapshots; see [source snapshot map](docs/SOURCE_SNAPSHOTS.md).
- `notebooks/`: revision-training and reviewer-test notebooks.
- `docs/`: reproducibility notes, source-snapshot map, and laptop/Kaggle run guides, including per-seed ablation guides.
- `datasets/README.md`: dataset sources, expected layouts, and access notes. Raw benchmark images are not included.
- `reproducibility/reviewer_test_package/`: standalone reviewer package with code, 15 Stage 4 best checkpoints, split metadata, requirements, and Kaggle notebook. Checkpoint hashes are recorded in `package_manifest.json`.
- `results/full_model_training/`: full-model histories, stage manifests, split manifests, and canonical checkpoints for seeds 17, 43, and 73; see its README for checkpoint curation details.
- `results/ablation_training_history/`: ablation run configurations and stage histories for four variants and three seeds.
- `results/reviewer_evaluation/`: reviewer evaluation summaries, per-image paired results, bootstrap summaries, and training-history outputs.
- `testing results/`: fixed-512 paired evaluations and revised blend sweeps; see [results guide](testing%20results/RESULTS_README.md).
- `real_paired_evaluation_results/`: real-haze paired evaluation outputs for four datasets and three full-model seeds.
- `Interferences/`: analysis notebooks with their result tables and run manifests.
- `SHA256SUMS.txt`: checksums for packaged repository files.

## Running the code

Raw benchmark images are not included. Dataset sources, layouts, environment setup, and run commands are documented with each source snapshot:

- [Dataset instructions](datasets/README.md)
- [Reproducibility checklist](docs/REPRODUCIBILITY.md)
- [Laptop run guide](docs/laptop_how_to_run/README.md)
- [Kaggle run guide](docs/kaggle_how_to_run/README.md)
- [Reviewer evaluation package guide](reproducibility/reviewer_test_package/REVIEWER_TEST_PACKAGE_README.md)

The code, model weights, training configurations, and evaluation scripts are also listed at the [project repository](https://github.com/manpreet28111995/Image-Dehazing-using-ECA-LDNet).

## Interpretation limits

- I-HAZE and NH-HAZE test subsets contain five pairs each; these are limited-sample measurements.
- Dense-Haze pools all 55 release pairs, including non-test partitions.
- Ablation variants used a 40-epoch screening schedule; the full model used a planned 140-epoch schedule. These screens do not estimate full-budget component effects.
- Cross-method comparisons use different training data or pipelines and do not establish a controlled ranking.
- Runtime measurements exclude checkpoint loading, output serialization, metric computation, concurrent requests, and controlled cold-storage reads.

Dataset and pretrained-weight terms remain those of their providers. The code snapshots include their own license files.
