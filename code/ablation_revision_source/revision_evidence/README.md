# Revision evidence included in this package

The package includes compact histories, manifests, and numerical results, but excludes trained `.keras` checkpoints and rendered PNG predictions.

## Full-model seeds 17 and 73

`training_two_seeds/histories/` contains archived histories. Historical seed17 Stage 3 is the earlier batch-16 run; the corrected continuation is under `training_two_seeds/corrected_seed17_stage3_batch8/`: 20/20 epochs, batch 8, from Stage 2 with the matching split. Best validation PSNR is 29.58713 dB and SSIM is 0.93565. Corrected Stage 4 is pending. Seed73 used batch map 8/16/8/16 and Stage 3 early-stopped at 16 epochs.

## Existing evaluation and legacy results

Radiometric test CSVs for seeds 17/73 lack exact checkpoint paths in a manifest; treat as preliminary and rerun selected validation-best checkpoints with a manifest before final reporting. Submitted-preprocessing evaluation is diagnostic and must stay separate.

`legacy_submitted_ablation_seed42/ablation_results.csv` has five rows: one full-model baseline plus four component variants (no ECA, no Pixel Attention, no physics guidance, and no attention). Those are historical single-run results without verified seed/split evidence; they are not substitutes for the new controlled runs.

## Current distinct ablation plan

The full model and `no_eca` (PA only), `eca_only` (ECA only), `no_attention` (neither) form a complete 2×2 attention-component comparison. `no_physics` keeps both attention branches and disables the physics-guidance branch. The CLI alias `pixel_only` is identical to `no_eca`; its duplicate guide is omitted.

Each of the four unique ablations is documented at seeds 17, 73, and 43, starts from fresh weights, and needs no full-model checkpoint. They use 18/8/6/8 epochs, batches 8/16/8/16, split seed 42, and shortened cosine learning-rate schedules. Report as screening ablations, not full-budget replications.
