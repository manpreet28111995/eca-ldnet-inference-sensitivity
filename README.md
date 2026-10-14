# ECA-LDNet reproducibility bundle

This repository organizes source, trained checkpoints, analysis evidence, and the manuscript for the ECA-LDNet dehazing study. It supports inference/evaluation of the reported fixed checkpoints and documents the training recipe. Raw benchmark datasets are external.

## Layout

- `eca_ldnet/`, `scripts/`, `tests/`: canonical training/evaluation source copied from `code/full_training_source`.
- `source_snapshots/`: only files that differ in the ablation and reviewer-evaluator snapshots; see `docs/SOURCE_SNAPSHOTS.md`.
- `reproducibility/reviewer_test_package/`: standalone evaluation bundle with 15 Stage 4 best checkpoints, split metadata, histories, run configs, and Kaggle notebook. Checkpoint SHA-256 values are in its `package_manifest.json`.
- `results/`, `testing results/`, `real_paired_evaluation_results/`, `Interferences/`: numeric run evidence, manifests, and analysis notebooks. Generated prediction images and plot copies omitted.
- `datasets/README.md`: download sources and expected layouts; no raw images included.

## Start

Install from repository root with a compatible Python environment:

```bash
pip install -e ".[dev]"
```

Dataset layout is in `datasets/README.md`; training guides are in `docs/ablations_kaggle.md` and the `docs/kaggle_how_to_run/` and `docs/laptop_how_to_run/` folders. For fixed-checkpoint evaluation, use `reproducibility/reviewer_test_package/REVIEWER_TEST_PACKAGE_README.md` and its notebook. Analysis notebooks in `Interferences/` reproduce saved numerical analyses when required datasets/checkpoints are mounted.

## Reproducibility limits to resolve

The manuscript records Python 3.13.15 and TensorFlow 2.20.0 for reported evaluation runs. Current package metadata instead limits Python to `<3.13` and TensorFlow to `<2.20`; dependency ranges are not a lockfile. Pin separate training/evaluation environments and optional metric dependencies before claiming an exact rerun. Some saved manifests contain Kaggle or workstation absolute paths; rebase paths while preserving hashes.

The study uses externally hosted datasets. Follow each dataset's terms; do not commit images unless redistribution is allowed. Report the exact partitions and caveats in the paper rather than treating pooled/reused data as independent test data.

## License

Project code uses the included MIT license. Dataset and pretrained-weight terms remain those of their providers.
