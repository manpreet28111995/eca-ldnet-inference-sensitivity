# ECA-LDNet Code and Reviewer Package

This repository contains ECA-LDNet source snapshots and a standalone package for reviewer evaluation. All trained models and experiment results are available at the [ECA-LDNet Hugging Face repository](https://huggingface.co/manpreet28111995/eca-ldnet). This includes model files, result tables, per-image records, training outputs, and supporting analyses.

## Repository contents

| Path | Contents |
|---|---|
| [`code/full_training_source/`](code/full_training_source/) | Full-model implementation, training and evaluation scripts, notebooks, and environment guides. Start with its [README](code/full_training_source/README.md). |
| [`code/ablation_revision_source/`](code/ablation_revision_source/) | Revised ablation implementation, scripts, tests, and run guides. Start with its [README](code/ablation_revision_source/README.md). |
| [`code/reviewer_evaluator_snapshot/`](code/reviewer_evaluator_snapshot/) | Snapshot of the evaluator code used for reviewer-oriented evaluation. |
| [`reviewer_test_package/`](reviewer_test_package/) | Standalone inference-evaluation package with selected checkpoints, code, run metadata, and a Kaggle notebook. Start with the [reviewer package guide](reviewer_test_package/REVIEWER_TEST_PACKAGE_README.md). |

## Reviewer evaluation

The reviewer package includes 15 Stage 4 validation-best checkpoints for the full model and four available ablations across seeds 17, 43, and 73. It does not include benchmark images. Follow `reviewer_test_package/REVIEWER_TEST_PACKAGE_README.md` to prepare dataset inputs and run the Kaggle notebook. That guide describes package coverage, required inputs, outputs, and limitations.

## Training and ablation workflows

Use the run guides within each source snapshot for its supported laptop or Kaggle workflow. Install the dependencies listed by the chosen source snapshot, obtain datasets from their providers, and follow that snapshot's documented directory layout. The snapshots preserve different training and evaluation workflows; read their manifests and documentation before comparing outputs across them.

## Model size and compute

The full ECA-LDNet architecture has **1,481,871 parameters** and requires **2.171 GMACs (4.343 GFLOPs) at 256 × 256**, according to the released layer-wise counter. These are architecture-level counts; runtime depends on hardware and execution settings.

## Results and versioning

All models and results are hosted in the [ECA-LDNet Hugging Face repository](https://huggingface.co/manpreet28111995/eca-ldnet). Use its version or revision when citing or comparing files.

## Data and licensing

Benchmark image datasets are not redistributed in this repository. Obtain them from their providers and follow the corresponding access terms. Source snapshots include their own license files where applicable; see each license before reusing code or packaged weights.
