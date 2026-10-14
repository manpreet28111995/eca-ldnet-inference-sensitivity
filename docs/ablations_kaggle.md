# Kaggle ablation run index

The package has 12 self-contained Kaggle guides and 12 matching laptop guides: four ablations at each seed (17, 73, and 43). Kaggle guides include input checks, four stage cells, stage restore checks, and ZIP exports. Laptop guides include local terminal commands for each stage and commands to package/restore stage ZIPs.

## 40-epoch budget

Each run uses four stages and 40 epochs total:

| Stage | Epochs | Batch |
|---|---:|---:|
| Stage 1 | 18 | 8 |
| Stage 2 | 8 | 16 |
| Stage 3 | 6 | 8 |
| Stage 4 | 8 | 16 |

The architecture, stage objectives, optimizer family, preprocessing, split, physics settings, batch mapping, and model source remain the same as the shared full-model code. The configured full-model budget is 60/30/20/30 epochs with early stopping enabled; these ablations use 18/8/6/8 epochs with early stopping disabled. This also shortens each stage's learning-rate schedule. Report these as 40-epoch screening ablations, not full-budget protocol-matched replications.

## Dataset placement on a laptop

The ZIP contains empty placeholders under `datasets/`. Download RESIDE-6K and ITS separately and place their images under `datasets/RESIDE-6K/train/hazy`, `datasets/RESIDE-6K/train/GT`, `datasets/ITS/hazy`, and `datasets/ITS/clear`. See [the dataset folder instructions](datasets/README.md). Kaggle runs attach the datasets through Kaggle inputs instead.

## Save and continue between stages

Each run writes to `/kaggle/working/ablation_runs/<variant>/seed_<seed>/`. For example, seed 73 `no_eca` writes to `/kaggle/working/ablation_runs/no_eca/seed_73/`. `models/` contains names such as `seed73_no_eca_stage1_best.keras`, `seed73_no_eca_stage1_latest.keras`, and `seed73_no_eca_stage1_epoch018.keras`. `history/` contains a seed/variant/stage-tagged CSV for every stage. `run_config.json` records the run settings and completed stage settings, and `split_manifest.csv` records the grouped split.

After a stage, save its ZIP under `save_here/seed<seed>_<variant>/`. Kaggle users attach that ZIP as a Kaggle input and run the next restore cell. Laptop users run `scripts/restore_ablation_stage.py` with the prior stage ZIP. Both methods restore the same seed/variant folder, check `eca_ldnet_final.keras`, and confirm the next stage is ready.

## Seed 73

- [no_eca — Kaggle](kaggle_how_to_run/seed73/no_eca.md) · [Laptop](laptop_how_to_run/seed73/no_eca.md)
- [eca_only — Kaggle](kaggle_how_to_run/seed73/eca_only.md) · [Laptop](laptop_how_to_run/seed73/eca_only.md)
- [pixel_only — Kaggle](kaggle_how_to_run/seed73/pixel_only.md) · [Laptop](laptop_how_to_run/seed73/pixel_only.md)
- [no_attention — Kaggle](kaggle_how_to_run/seed73/no_attention.md) · [Laptop](laptop_how_to_run/seed73/no_attention.md)

## Seed 17

- [no_eca — Kaggle](kaggle_how_to_run/seed17/no_eca.md) · [Laptop](laptop_how_to_run/seed17/no_eca.md)
- [eca_only — Kaggle](kaggle_how_to_run/seed17/eca_only.md) · [Laptop](laptop_how_to_run/seed17/eca_only.md)
- [pixel_only — Kaggle](kaggle_how_to_run/seed17/pixel_only.md) · [Laptop](laptop_how_to_run/seed17/pixel_only.md)
- [no_attention — Kaggle](kaggle_how_to_run/seed17/no_attention.md) · [Laptop](laptop_how_to_run/seed17/no_attention.md)

## Seed 43

- [no_eca — Kaggle](kaggle_how_to_run/seed43/no_eca.md) · [Laptop](laptop_how_to_run/seed43/no_eca.md)
- [eca_only — Kaggle](kaggle_how_to_run/seed43/eca_only.md) · [Laptop](laptop_how_to_run/seed43/eca_only.md)
- [pixel_only — Kaggle](kaggle_how_to_run/seed43/pixel_only.md) · [Laptop](laptop_how_to_run/seed43/pixel_only.md)
- [no_attention — Kaggle](kaggle_how_to_run/seed43/no_attention.md) · [Laptop](laptop_how_to_run/seed43/no_attention.md)

The fifth `no_physics` variant is postponed. The 12 jobs above are independent; no job needs a full-model checkpoint. For comparable full-model rows, remember that the saved seed-17 full run used batch 16 in Stage 3 and needs its separate Stage 3–4 correction. The saved seed-73 full run used the canonical batches, but its early-stopped Stage 3 ran 16 epochs; it is still a different budget from these 40-epoch ablations.

Each run uses split seed 42, batches `8,16,8,16`, radiometric preprocessing, physics blend 0.08, physics epsilon 0.1, and the ImageNet perceptual protocol.
