# Kaggle ablation guides

Choose a guide by seed and ablation. Each guide includes Kaggle input setup, dataset checks, four stage cells, restore checks before Stages 2–4, and a ZIP export after every stage.

Attach the shared model ZIP as a Kaggle Model input and attach RESIDE-6K and ITS separately. Training reads ITS `hazy/` and `clear/`, and RESIDE `RESIDE-6K/train/hazy/` and `RESIDE-6K/train/GT/`. It does not use ITS `trans/` or RESIDE `test/` for training. The guide handles automatic model-input extraction.

## Four distinct ablations per seed

The code audit found `no_eca` and `pixel_only` are the same condition (ECA off, Pixel Attention on); do not run both. Use these distinct variants:

- `no_eca`: ECA off, Pixel Attention on.
- `eca_only`: ECA on, Pixel Attention off.
- `no_attention`: both attention branches off.
- `no_physics`: keep both attentions on and disable the ASM-inspired physics-guidance branch.

Together with the full model, these cover the attention component comparisons plus the separate physics-branch comparison. The legacy `pixel_only` alias remains in code for compatibility, but its run guide is intentionally omitted.

Each uses 40 epochs (18/8/6/8), batches 8/16/8/16, and split seed 42. The shorter epoch budget also shortens the learning-rate schedules, so report these as screening ablations. Save/download the stage ZIP after each stage, attach it for the next-stage restore cell, and continue through Stage 4.

Seeds: `seed17/`, `seed73/`, and `seed43/`.
