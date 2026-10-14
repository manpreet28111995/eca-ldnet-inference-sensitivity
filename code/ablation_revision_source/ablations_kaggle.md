# Ablation run index

The package contains 12 Kaggle guides and 12 laptop guides: four unique ablations at each seed (17, 73, and 43).

## Distinct conditions

The source maps `no_eca` and `pixel_only` to exactly the same attention flags (`use_eca=False`, `use_pa=True`). That is one condition; the `pixel_only` guides are omitted so it is not trained twice.

| Condition | ECA | Pixel Attention | Physics guidance |
|---|---:|---:|---:|
| Full model reference | On | On | On |
| `no_eca` | Off | On | On |
| `eca_only` | On | Off | On |
| `no_attention` | Off | Off | On |
| `no_physics` | On | On | Off |

The first four rows make a 2×2 attention-component comparison. `no_physics` separately tests the ASM-inspired guidance branch. Each ablation starts from fresh weights and requires no full-model checkpoint.

## Budget and protocol caveat

Each ablation uses 40 epochs total: stages 1–4 use 18/8/6/8 epochs and batches 8/16/8/16. Shared stage objectives, preprocessing, split seed 42, other training settings, and stage order are retained. The full-model schedule is 60/30/20/30 with early stopping enabled; these shorter ablation schedules disable early stopping and recalculate the cosine learning-rate schedules. Report these as 40-epoch screening ablations, not full-budget replications.

## Full-model seed17 correction

Corrected seed17 Stage 3 evidence is under `revision_evidence/training_two_seeds/corrected_seed17_stage3_batch8/`: 20/20 epochs, batch 8, continuing from Stage 2 with the matching split. Corrected Stage 4 is pending. Seed73 used batch map 8/16/8/16; its Stage 3 early-stopped at 16 epochs.

## Dataset placement

Laptop training data go under `datasets/RESIDE-6K/train/{hazy,GT}` and `datasets/ITS/{hazy,clear}`. Kaggle runs attach both datasets as inputs. ITS `trans/` and RESIDE `test/` are not used for training.

## Guides by seed

For each seed `17`, `73`, and `43`, run these four guides: `no_eca`, `eca_only`, `no_attention`, and `no_physics`.

- Kaggle: `kaggle_how_to_run/seed<seed>/<variant>.md`
- Laptop: `laptop_how_to_run/seed<seed>/<variant>.md`

The optional `pixel_only` CLI alias is not a separate ablation. Stage 5 is not used.
