# Source snapshot map

Root `eca_ldnet/` and `scripts/` are the full training source snapshot. Historical experiment runs also used two related code snapshots; only the files differing from the root snapshot are retained here. To reconstruct a historical run, replace the matching root file with the snapshot file before running it.

- `source_snapshots/ablation_revision/`: revised data/evaluation/blend-sweep files associated with the ablation revision source.
- `source_snapshots/reviewer_evaluator/`: evaluator files used by the reviewer-metric snapshot.

Do not combine these files silently. Record which snapshot produced each result.
