# Kaggle reviewer-evaluation package

This folder is the reviewer-evaluation package. To run its Kaggle notebook, zip this folder and attach it as a Dataset or Model input. It contains evaluation code, 15 validation-best checkpoints, training histories/configuration, split manifests, and a Kaggle notebook; raw image datasets are external.

## Included checkpoints

- Full ECA-LDNet: seeds 17, 43, 73.
- Existing completed ablations, all at Stage 4 validation-best: `no_eca`, `eca_only`, `no_attention`, and `no_physics`, for seeds 17, 43, 73.
- Histories and run configuration for Stages 1–4 of each ablation. The source Stage 4 ZIPs repeat earlier stage weights; this test archive retains the histories/configs and only the final Stage 4 best checkpoint per run.
- Each full-model seed also retains all four stage histories/manifests. Full-model Stage 4 latest/periodic and earlier-stage checkpoints are excluded because this is an inference test package, not a resume-training package.

The notebook evaluates all selected checkpoints on every paired/unpaired dataset path you enable. Paired outputs include PSNR, SSIM, MS-SSIM, MAE/MSE/RMSE, CIEDE2000 when `scikit-image` is installed, LPIPS/DISTS when `pyiqa` is installed, per-image CSVs, summary CSVs, bootstrap intervals, checkpoint hashes, run JSON, and dehazed images. Every image that successfully pairs, reads, and scores is saved under `tested_sample_<dataset>`; unmatched or unreadable inputs are counted in the run manifest. No image is selected by metric rank, so the saved outputs do not cherry-pick favorable examples.

The package supports the trained full model and the four available ablations. It does not contain trained checkpoints for the additional reviewer-requested SE/CBAM/capacity-matched/order/parallel controls, blend/epsilon sweeps, or EMA/loss/LR factorial runs. Those require separate experiments and must not be reported as completed from these inference tests.

## Dataset inputs

Attach only the evaluation datasets in Kaggle's **Add Input** panel. The two training sets (ITS and RESIDE-6K train) are not required for inference.

| Dataset | Public source | Evaluation use |
|---|---|---|
| ITS (training only) | [Kaggle ITS RESIDE-Standard](https://www.kaggle.com/datasets/balraj98/indoor-training-set-its-residestandard) or [official RESIDE-Standard](https://sites.google.com/view/reside-dehaze-datasets/reside-standard) | Not needed for the inference notebook; attach only if training additional reviewer ablations |
| SOTS Indoor/Outdoor | [Kaggle SOTS RESIDE](https://www.kaggle.com/datasets/balraj98/synthetic-objective-testing-set-sots-reside) or [RESIDE-Standard](https://sites.google.com/view/reside-dehaze-datasets/reside-standard) | Paired synthetic, 256×256 |
| RESIDE-6K test | [Kaggle RESIDE-6K](https://www.kaggle.com/datasets/kmljts/reside-6k) | Paired synthetic only if a separate hazy/GT test split is actually present; never substitute training data |
| O-HAZE | [Official NTIRE 2018 page](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/) | Paired real, native resolution |
| I-HAZE | [Official NTIRE 2018 page](https://data.vision.ee.ethz.ch/cvl/ntire18/i-haze/) | Paired real, native resolution |
| NH-HAZE | [Official NTIRE 2020 page](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/) | Paired non-homogeneous real haze, native resolution |
| DNH-HAZE | [NTIRE 2024 challenge page](https://cvlai.net/ntire/2024/) and [challenge portal](https://codalab.lisn.upsaclay.fr/competitions/17529) | Paired dense/non-homogeneous real haze, native resolution |
| RTTS | [RESIDE-β official page](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2) | Unpaired real-world evaluation with NR-IQA/FADE proxies |

The Kaggle notebook starts with the previously used SOTS and RESIDE-6K mounts. Edit the dataset-path cell for any attached input whose mount or folder layout differs. Real paired sets and RTTS are auto-skipped until their paths exist. Confirm pair counts and preserve the dataset split name in the outputs.

## Run

1. Add the package ZIP as a Kaggle input and open `notebooks/reviewer_kaggle_test.ipynb` (also included at the archive root).
2. Enable a GPU and Internet in the notebook settings, then run the dependency/setup cell. Internet is needed for the optional metric packages and their pretrained weights.
3. Run package discovery, then edit the dataset paths and enable any real datasets you attached.
4. Run the paired/unpaired evaluation cell. It runs all 15 checkpoints by default and writes progress after each model/dataset task so a restarted session can resume.
5. Download `/kaggle/working/reviewer_evaluation_results.zip` from notebook output. Each result folder contains CSV/JSON evidence and `tested_sample_<dataset>` images.

For SOTS and other paired synthetic datasets, use the predeclared 256×256 radiometric input protocol. For paired real sets, use native resolution if the loaded model supports it. Record failures/OOMs instead of silently resizing. No-reference IQA is a perceptual proxy, not ground-truth dehazing accuracy.
