# ECA-LDNet reviewer revision package handoff

Share `eca_ld_net_revision_ablations_.zip`. It contains the training source, complete Kaggle/laptop guides, dataset layout notes, and compact reviewer evidence.

Four distinct ablations are guided for seeds 17, 73, and 43: `no_eca`, `eca_only`, `no_attention`, and `no_physics`. The code maps `pixel_only` to exactly the same switches as `no_eca`; its run guide is omitted to prevent duplicate training. The full model plus the first three ablations form the 2×2 ECA/Pixel Attention comparison; `no_physics` separately tests the ASM-inspired branch. Every ablation starts from scratch and needs no full-model checkpoint.

These are 40-epoch screening ablations (18/8/6/8) with batches 8/16/8/16. The full training schedule is 60/30/20/30 with early stopping. The shorter stages recalculate shorter cosine learning-rate schedules and disable early stopping, so report them as shortened-budget ablations rather than full-budget replications.

The corrected seed17 Stage 3 is included as history/config/split evidence (20/20 epochs, batch 8, same split as Stage 2). Corrected Stage 4 is pending and its checkpoint weights are not in this compact bundle. Final manuscript responses also need the other requested evidence such as blend sensitivity, training-recipe controls, and test/real-image evaluation; this ZIP alone does not complete all reviewer revisions.
