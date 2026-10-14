# Reproduction checklist

1. Obtain ITS and RESIDE-6K training data plus the specific evaluation partitions listed in `datasets/README.md`. Record versions, access dates, licenses, pair counts, and file hashes.
2. Use the seed/split manifests in `reproducibility/reviewer_test_package/metadata/`; current CSV paths are Kaggle-specific and must be rebased locally.
3. For exact reported inference, use the matching Stage 4 best checkpoint for seeds 17, 43, and 73. Four ablation variants are also included. Verify SHA-256 against `package_manifest.json`.
4. Preserve the paper's preprocessing, image size, tile/overlap settings, blend/epsilon, pair matching, and metric implementation from each run manifest. Do not substitute one evaluation protocol for another.
5. Run the analysis notebook that matches each result in `Interferences/`; use `paper/Interferences_results_used.md` to map evidence to manuscript sections/figures.
6. Pin the actual Python, TensorFlow, Keras, OpenCV, scikit-image, and optional IQA packages, plus CUDA/cuDNN, GPU, precision, and VGG ImageNet weight provenance. The current metadata/environment mismatch is documented in the root README.

Stage 5 is not part of the reported Stage 4 checkpoint results and its recipe is not established by the current source.
