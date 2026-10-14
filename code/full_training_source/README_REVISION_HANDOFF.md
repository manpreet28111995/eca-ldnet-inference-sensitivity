# ECA-LDNet reviewer revision package handoff

This package contains the current model/training source, explicit Kaggle ablation guides, and compact numerical evidence. Upload and share only `eca_ld_net_revision_ablations_.zip` from `kaggle_revision/`. The Markdown guides work with Kaggle's automatically mounted/extracted model input; they do not require a separate notebook or a manual code-ZIP extraction step.

## What is in this ZIP

- `eca_ldnet/`: full staged training, model, data, losses, callbacks, inference, and evaluation implementation.
- `scripts/` and `tests/`: evaluation/training utilities, local dataset validation, stage ZIP save/restore helpers, and source checks.
- `ablations_kaggle.md`: index for 12 Kaggle guides and 12 matching laptop guides, covering four ablations × seeds 17, 73, and 43 at a 40-epoch budget.
- `kaggle_how_to_run/`: 12 complete Kaggle cell-by-cell guides, grouped by seed.
- `laptop_how_to_run/`: 12 matching local terminal guides, grouped by seed, with full stage commands and ZIP restore/save commands.
- `datasets/`: empty RESIDE-6K and ITS folder placeholders plus exact placement instructions. Dataset images must be downloaded separately and are not included.
- `revision_evidence/`: two-seed stage histories, split audit, all saved per-image and summary test CSVs, legacy 8% blend sweep, and legacy ablation CSV.

## Immediate run plan

1. The friend runs four variants (`no_eca`, `eca_only`, `pixel_only`, `no_attention`) at seeds 17, 73, and 43, each with 40 epochs across four stages: 18/8/6/8. Each ablation starts from fresh weights and does not require a full-model checkpoint.
2. Keep stage batches 8/16/8/16, split seed 42, and the shared full-model architecture/objective/preprocessing/physics settings. The configured full model uses 60/30/20/30 epochs with early stopping enabled; these guides use 18/8/6/8 with early stopping disabled, which shortens each learning-rate schedule. Report these as 40-epoch screening ablations, not full-budget protocol-matched replications. Stage 5 is not included.
3. Every run has a unique `ablation_runs/<variant>/seed_<seed>/` folder with a `models/` directory, per-stage history CSVs, `run_config.json`, and `split_manifest.csv`. Checkpoints include seed/variant/stage in their filename. Each stage ZIP includes that run folder and the next-stage guide prints when it has restored `eca_ldnet_final.keras`.
4. Your existing full seed-17 run used batch 16 in Stage 3. To correct that full-model comparison, resume from its Stage 2 checkpoint, rerun Stage 3 with batch 8, then Stage 4 with batch 16 from the new Stage 3 result. Seed 73 already used Stage 3 batch 8 and Stage 4 batch 16.
5. Existing test CSVs are preserved, but the radiometric run lacks exact checkpoint paths. Rerun selected validation-best checkpoints with an evaluation manifest before treating those numbers as final. The submitted-preprocessing evaluation is diagnostic only.

## Important bundle limits

This ZIP intentionally excludes datasets and trained `.keras` weights. It also excludes 4,000 rendered prediction PNGs (roughly 452 MB); all available per-image numerical metrics are included as CSV. Original checkpoints and PNGs remain in their source folders on this workstation/Kaggle outputs.
