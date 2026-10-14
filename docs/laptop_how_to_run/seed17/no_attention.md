# No attention — seed 17 laptop run

This guide runs the `no_attention` ablation at seed 17 on the laptop. It starts from fresh weights and does not need a full-model checkpoint. Follow the environment and dataset setup in `laptop_how_to_run/README.md` first. Run every command from the extracted project root, with the local datasets under `datasets/`.

This is a 40-epoch shortened-budget screening run. It uses the same model source, stage objectives, split seed, batch schedule, preprocessing, and physics settings as the shared code. The shorter run also shortens the learning-rate schedules; report it as a 40-epoch ablation, not a full-budget replication.

Output folder:
`ablation_runs/no_attention/seed_17/`

Stage batches are `8, 16, 8, 16`; split seed is `42`. Each stage saves latest, best, and every-epoch model checkpoints plus a stage history CSV. The run folder also holds `run_config.json` and `split_manifest.csv`.

## Stage 1 — 18 epochs

Run the full training command:

```text
python -m eca_ldnet.train --reside6k "datasets/RESIDE-6K" --its "datasets/ITS" --out "ablation_runs" --stages 1 --max-epochs 18 --disable-early-stopping --stage-batch-sizes 8 16 8 16 --seed 17 --split-seed 42 --variant no_attention --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet --save-every-epoch
```

After it completes, create and check the stage ZIP:

```text
python scripts/package_ablation_stage.py --out-root ablation_runs --save-root save_here --seed 17 --variant no_attention --stage 1
```

The resulting archive is `save_here/seed17_no_attention/seed17_no_attention_stage1.zip`. Save it before starting the next stage. For Stage 2, restore this ZIP first as shown in the following section.

## Restore Stage 1 before Stage 2

Run this command in a fresh terminal/session, or to verify the prior checkpoint is present. It restores only this seed and ablation and prints a success message.

```text
python scripts/restore_ablation_stage.py --archive "save_here/seed17_no_attention/seed17_no_attention_stage1.zip" --out-root ablation_runs --seed 17 --variant no_attention --stage 1
```

## Stage 2 — 8 epochs

Run the full training command:

```text
python -m eca_ldnet.train --reside6k "datasets/RESIDE-6K" --its "datasets/ITS" --out "ablation_runs" --stages 2 --resume "ablation_runs/no_attention/seed_17/models/eca_ldnet_final.keras" --max-epochs 8 --disable-early-stopping --stage-batch-sizes 8 16 8 16 --seed 17 --split-seed 42 --variant no_attention --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet --save-every-epoch
```

After it completes, create and check the stage ZIP:

```text
python scripts/package_ablation_stage.py --out-root ablation_runs --save-root save_here --seed 17 --variant no_attention --stage 2
```

The resulting archive is `save_here/seed17_no_attention/seed17_no_attention_stage2.zip`. Save it before starting the next stage. For Stage 3, restore this ZIP first as shown in the following section.

## Restore Stage 2 before Stage 3

Run this command in a fresh terminal/session, or to verify the prior checkpoint is present. It restores only this seed and ablation and prints a success message.

```text
python scripts/restore_ablation_stage.py --archive "save_here/seed17_no_attention/seed17_no_attention_stage2.zip" --out-root ablation_runs --seed 17 --variant no_attention --stage 2
```

## Stage 3 — 6 epochs

Run the full training command:

```text
python -m eca_ldnet.train --reside6k "datasets/RESIDE-6K" --its "datasets/ITS" --out "ablation_runs" --stages 3 --resume "ablation_runs/no_attention/seed_17/models/eca_ldnet_final.keras" --max-epochs 6 --disable-early-stopping --stage-batch-sizes 8 16 8 16 --seed 17 --split-seed 42 --variant no_attention --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet --save-every-epoch
```

After it completes, create and check the stage ZIP:

```text
python scripts/package_ablation_stage.py --out-root ablation_runs --save-root save_here --seed 17 --variant no_attention --stage 3
```

The resulting archive is `save_here/seed17_no_attention/seed17_no_attention_stage3.zip`. This is the final stage ZIP; keep it with your ablation results.

## Restore Stage 3 before Stage 4

Run this command in a fresh terminal/session, or to verify the prior checkpoint is present. It restores only this seed and ablation and prints a success message.

```text
python scripts/restore_ablation_stage.py --archive "save_here/seed17_no_attention/seed17_no_attention_stage3.zip" --out-root ablation_runs --seed 17 --variant no_attention --stage 3
```

## Stage 4 — 8 epochs

Run the full training command:

```text
python -m eca_ldnet.train --reside6k "datasets/RESIDE-6K" --its "datasets/ITS" --out "ablation_runs" --stages 4 --resume "ablation_runs/no_attention/seed_17/models/eca_ldnet_final.keras" --max-epochs 8 --disable-early-stopping --stage-batch-sizes 8 16 8 16 --seed 17 --split-seed 42 --variant no_attention --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet --save-every-epoch
```

After it completes, create and check the stage ZIP:

```text
python scripts/package_ablation_stage.py --out-root ablation_runs --save-root save_here --seed 17 --variant no_attention --stage 4
```

The resulting archive is `save_here/seed17_no_attention/seed17_no_attention_stage4.zip`. This is the final stage ZIP; keep it with your ablation results.

## Expected files after Stage 4

The run folder includes `models/seed17_no_attention_stage4_best.keras`, `models/seed17_no_attention_stage4_latest.keras`, per-epoch checkpoints for each stage, `models/eca_ldnet_final.keras`, four CSV files under `history/`, `run_config.json`, and `split_manifest.csv`.
