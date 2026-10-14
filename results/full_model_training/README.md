# Full model training results

Organized by seed and stage. Each stage contains its history, stage manifest, and canonical checkpoints; each seed has its split manifest. Stage manifests record original source mappings, optimizer iteration counters, and aliases omitted during curation.

## Completion

- Seeds 17 and 43: Stages 1–4 completed (60, 30, 20, 30 epochs).
- Seed 73: Stages 1, 2, and 4 completed; Stage 3 early-stopped after 16 of 20 epochs.
- Batch sizes: Stage 1 = 8, Stage 2 = 16, Stage 3 = 8, Stage 4 = 16. Split seed = 42.

Each original checkpoint was compared to its stage-specific source bundle. Stage-named checkpoints from one stage are not placed in another stage folder. Snapshots with identical model tensors and identical optimizer tensors are represented once; omitted names and the retained canonical file are listed in the stage manifest. Checkpoints with identical model weights but different optimizer states are both retained for resumability.

For Seeds 17 and 73, `stage2_best.keras` is sourced from the before-resume checkpoint because its optimizer iteration aligns with the history's best-epoch row. The later conflicting file previously named `stage2_best.keras` is excluded and documented in the manifest.
