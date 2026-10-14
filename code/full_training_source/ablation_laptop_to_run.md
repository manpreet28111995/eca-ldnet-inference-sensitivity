# ECA-LDNet ablations: laptop and Kaggle runbook

This runbook contains commands for **future controlled ablation runs**. The full-model training is already complete and archived for seeds 17, 43, and 73. For final full-model inference evaluation and current evidence status, use [REVIEWER_EVALUATION_RUNBOOK.md](REVIEWER_EVALUATION_RUNBOOK.md). This file is for new ablation experiments only; do not use it to launch or repair full-model training. Use the revised staged trainer for new ablations; do **not** use `scripts/run_ablation.py` or `scripts/run_ablation_v2.py` for reviewer-grade results.

For the first four parallel Kaggle jobs, use [ablations_kaggle.md](ablations_kaggle.md). It contains four complete seed-73 cells for `no_eca`, `eca_only`, `pixel_only`, and `no_attention`, with `--save-every-epoch`. The fifth `no_physics` run is postponed.

## Decision after checking the existing runs

There are two different ablation sets in this repository:

| Evidence | What is saved | Can it serve as the new controlled ablation table? |
|---|---|---|
| Older submitted ablations | Full, no ECA, no pixel attention, no physics guidance, and no attention. Checkpoints exist through epoch 28 and `ablation_results.csv` has one result row for each variant. | No. The results file does not record the seed or a split manifest. The old runner defaults to seed 42, so 42 is plausible but not proven. That runner uses a random pair split and a single shortened 28-epoch training recipe, not the revised scene-grouped four-stage protocol. Keep these as historical results; do not blend them with the new runs. |
| Revised full model | Seeds 17, 43, and 73 have curated Stage 1–4 histories/checkpoints under `full_model_training/`; all use the common split seed 42 and the verified 8/16/8/16 stage batch schedule. Seed 73 Stage 3 early-stopped at epoch 16; seeds 17 and 43 reached all configured stage caps. | The revised full model is archived and protocol-aligned. The final paired/unpaired evaluation is still pending; use the validation-best `stage4_best.keras` checkpoints and save manifests before citing test metrics. |

The seed-42 legacy ablations are **not wasted**: retain them as historical evidence, but they cannot replace controlled runs because their seed/split are unrecorded and their protocol differs. Run each selected core ablation from scratch at training seeds **17, 43, and 73**, with grouped split seed 42 and stage batches 8/16/8/16. Ablation runs are separate experiments and do not load full-model weights. Prioritize a smaller, fully controlled set if compute is limited; do not present legacy and revised runs as one table.

Keep the archived full-model histories unchanged. Run new ablations independently at **training seeds 17, 43, and 73**, always with **split seed 42**. Every ablation command should use stage batches **8, 16, 8, 16** (S1=8, S2=16, S3=8, S4=16), the same datasets, four stages, radiometric preprocessing, physics blend 0.08, physics epsilon 0.1, and ImageNet perceptual protocol.

## Revised protocol used for the commands below

The commands use `eca_ldnet.train`, the same trainer as the revised full model. They set the measured stage batch mapping explicitly: **S1 8, S2 16, S3 8, S4 16**. This mapping comes from the saved LR-schedule step counts and 17,924 training pairs. Do not replace the mapping with a global `--batch-size`.

Each command runs one complete variant/seed from fresh weights through Stages 1–4. Results are isolated at:

```text
$RUN_ROOT/<variant>/seed_<seed>/
  split_manifest.csv
  history/stage1_history.csv ... stage4_history.csv
  models/stageN_latest.keras       # saved every epoch for recovery
  models/stageN_best.keras         # best validation checkpoint for that stage
  models/stageN_epochNNN.keras     # periodic full checkpoints every 10 epochs
  models/eca_ldnet_final.keras     # selected Stage 4 result after completion
```

The CSVs record training/validation loss, PSNR, SSIM, MAE, and learning rate per epoch. Do not use test images to select a checkpoint or variant.

## Runtime estimate before launching jobs

No laptop or Kaggle wall-clock duration is recorded for the full four-stage jobs, so a fixed hours estimate would be guesswork. The canonical schedule is about 246,540 optimizer steps per complete run (S1 60×2,241; S2 30×1,121; S3 20×2,241; S4 30×1,121, from 17,924 training pairs). Run the documented pilot on the target GPU and record seconds per step for each stage. Estimate one full run as `2,241×60×S1_step_seconds + 1,121×30×S2_step_seconds + 2,241×20×S3_step_seconds + 1,121×30×S4_step_seconds`, plus validation/checkpoint overhead. Multiply that measured estimate by 15 for the friend’s five core ablations × three seeds. Those 15 ablations are separate full training runs. Save epoch checkpoints and resume from the matching `stageN_latest.keras` if interrupted.

## Laptop GPU setup: use WSL2, not native Windows Python

This repository requires TensorFlow 2.16–2.19 and Keras 3.9+. Native Windows TensorFlow GPU support ended after TensorFlow 2.10, so the RTX 5060 training path is WSL2/Linux. Install/update the NVIDIA driver on Windows and WSL2 itself; do not install a Linux NVIDIA display driver inside WSL. Official references: [TensorFlow pip installation](https://www.tensorflow.org/install/pip), [TensorFlow Windows source/install notes](https://www.tensorflow.org/install/source_windows), [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

In Windows PowerShell, check the Windows driver and install WSL2/Ubuntu if needed:

```powershell
nvidia-smi
wsl --status
wsl --update
wsl --install -d Ubuntu-24.04
```

Restart Windows if WSL asks for it. Open Ubuntu from the Start menu, then run these Linux commands. The repo is on `D:` and is visible to WSL as `/mnt/d/Image-Dehazing-using-ECA-LDNet`.

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip
cd /mnt/d/Image-Dehazing-using-ECA-LDNet
python3.12 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
python -m pip install 'tensorflow[and-cuda]>=2.16,<2.20' 'keras>=3.9,<4' 'numpy>=1.23,<2.0' opencv-python scikit-learn pandas matplotlib
python -m pip install -e . --no-deps
nvidia-smi
python -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU"))'
```

Proceed only if the TensorFlow command lists a GPU. The current Windows Python 3.14 and the checked-in `.venv` are not the training environment; do not run these jobs from Windows PowerShell. Kaggle T4 remains a valid alternative and uses the same run commands after the package/data paths are set.

## Set local dataset paths once

Put the actual dataset roots in these two values. The RESIDE root must contain its train split; ITS must contain paired hazy/clear images. If your datasets are on another drive, use their corresponding `/mnt/<drive>/...` paths.

```bash
export PROJECT=/mnt/d/Image-Dehazing-using-ECA-LDNet
export RESIDE6K_ROOT=/mnt/d/REPLACE_WITH_YOUR_DATA_PATH/RESIDE-6K
export ITS_ROOT=/mnt/d/REPLACE_WITH_YOUR_DATA_PATH/ITS
export RUN_ROOT=$PROJECT/revision_ablation_runs_2026_10
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
source .venv-wsl/bin/activate
set -o pipefail
```

Replace the two dataset paths **once** with the real locations on your laptop. Check the data roots before starting:

```bash
test -d "$RESIDE6K_ROOT" && test -d "$ITS_ROOT" && echo 'Dataset roots exist'
```

## Exact training commands

The commands below are for **new controlled ablation experiments only**. Full-model training for seeds 17, 43, and 73 is complete and archived. Do not run any full-model seed-43-from-scratch command or seed-17 Stage 3–4 repair command from an older copy of this runbook. For full-model evidence and evaluation, use [REVIEWER_EVALUATION_RUNBOOK.md](REVIEWER_EVALUATION_RUNBOOK.md).

### No ECA



```bash
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 17 --split-seed 42 --variant no_eca --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/no_eca_seed17.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 73 --split-seed 42 --variant no_eca --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/no_eca_seed73.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant no_eca --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/no_eca_seed43.log"
```
### ECA only

```bash
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 17 --split-seed 42 --variant eca_only --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/eca_only_seed17.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 73 --split-seed 42 --variant eca_only --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/eca_only_seed73.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant eca_only --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/eca_only_seed43.log"
```
### Pixel attention only

```bash
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 17 --split-seed 42 --variant pixel_only --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/pixel_only_seed17.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 73 --split-seed 42 --variant pixel_only --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/pixel_only_seed73.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant pixel_only --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/pixel_only_seed43.log"
```
### No attention

```bash
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 17 --split-seed 42 --variant no_attention --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/no_attention_seed17.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 73 --split-seed 42 --variant no_attention --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/no_attention_seed73.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant no_attention --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/no_attention_seed43.log"
```
### No physics guidance

```bash
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 17 --split-seed 42 --variant no_physics --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/no_physics_seed17.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 73 --split-seed 42 --variant no_physics --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/no_physics_seed73.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant no_physics --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/no_physics_seed43.log"
```

Run each line once. The order does not affect another run because each variant/seed has an independent model and output directory. To estimate the laptop runtime, run the GPU/data pilot in the revision notebook first; there is no verified laptop epoch timing yet.

## If a run is interrupted: exact recovery method

Do not repeat its four-stage command from Stage 1. Recover the interrupted stage from its per-epoch `stageN_latest.keras`; use the row count in its `stageN_history.csv` as the next epoch index. Example below recovers `no_eca`, seed 43, Stage 2. It computes the next epoch directly from the saved CSV and preserves the optimizer/schedule serialized in the latest checkpoint.

```bash
export HISTORY="$RUN_ROOT/no_eca/seed_43/history/stage2_history.csv"
INITIAL_EPOCH=$(python -c 'import csv,os; rows=list(csv.DictReader(open(os.environ["HISTORY"], newline=""))); print(max((int(r["epoch"]) for r in rows if int(r["stage"])==2), default=-1)+1)')
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 2 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant no_eca --resume "$RUN_ROOT/no_eca/seed_43/models/stage2_latest.keras" --initial-epoch "$INITIAL_EPOCH" --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee -a "$RUN_ROOT/logs/no_eca_seed43.log"
```

For another interrupted job, run the same recovery procedure with that job's explicit variant, seed, stage history file, and `stageN_latest.keras`. Do not use a Stage 4 checkpoint to start Stage 1 of an ablation. Recovery retains the model/optimizer checkpoint and metric history; Stage 4's custom EMA shadow averages restart from the recovered weights, so an interrupted EMA stage may differ slightly from an uninterrupted run. The saved validation-best checkpoint remains the model to evaluate.

## Other reviewer ablations after the core set

The five component variants above directly reproduce the older component comparisons under the revised protocol. Next, screen the training-strategy factors at seed 43 in separate output roots so the files cannot overwrite the normal full run. These use the same full architecture and disable one factor at a time:

```bash
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/factorial_no_ema" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --disable-ema --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/factorial_no_ema_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/factorial_no_polish" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --disable-polish-loss --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/factorial_no_polish_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/factorial_no_late_lr" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --disable-late-lr --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/factorial_no_late_lr_seed43.log"
```

These three are useful screening contrasts, but they are not a full 2×2×2 factorial. If results show material effects and compute allows, run the pairwise and triple combinations with unique `--out` roots, the core variants already run at seeds 17, 73, and 43 provide three-seed comparisons for those components. Do not attribute the old Stage 3→4 improvement to a single factor.

For the complete seed-43 factorial screen, the baseline cell is the `full` seed-43 command above. Run the remaining seven cells with their own output roots:

```bash
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/factorial_no_ema_no_polish" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --disable-ema --disable-polish-loss --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/factorial_no_ema_no_polish_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/factorial_no_ema_no_late_lr" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --disable-ema --disable-late-lr --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/factorial_no_ema_no_late_lr_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/factorial_no_polish_no_late_lr" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --disable-polish-loss --disable-late-lr --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/factorial_no_polish_no_late_lr_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/factorial_no_ema_no_polish_no_late_lr" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --disable-ema --disable-polish-loss --disable-late-lr --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/factorial_no_ema_no_polish_no_late_lr_seed43.log"
```
### Capacity-matched architecture screens

Run the capacity search once, before training these candidates. The search now includes SE, CBAM, PA→ECA, parallel attention, standard/group convolution, and learned residual. It measures parameter counts and selects width multipliers; it does not train models.

```bash
python scripts/match_model_capacity.py --target-params 1481871 --output "$RUN_ROOT/model_capacity_matches.csv"
export CAPACITY_CSV="$RUN_ROOT/model_capacity_matches.csv"
```

Each command below retrieves the width multiplier from that CSV and uses a unique output folder. These are seed-43 screens; repeat selected candidates with seeds 17 and 73 before making a multi-seed claim.

```bash
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant se --width-multiplier "$(python -c 'import csv,os; print(next(float(r["width_multiplier"]) for r in csv.DictReader(open(os.environ["CAPACITY_CSV"])) if r["variant"]=="dws_se"))')" --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/se_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant cbam --width-multiplier "$(python -c 'import csv,os; print(next(float(r["width_multiplier"]) for r in csv.DictReader(open(os.environ["CAPACITY_CSV"])) if r["variant"]=="dws_cbam"))')" --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/cbam_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant pa_eca --width-multiplier "$(python -c 'import csv,os; print(next(float(r["width_multiplier"]) for r in csv.DictReader(open(os.environ["CAPACITY_CSV"])) if r["variant"]=="dws_pa_eca"))')" --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/pa_eca_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant parallel --width-multiplier "$(python -c 'import csv,os; print(next(float(r["width_multiplier"]) for r in csv.DictReader(open(os.environ["CAPACITY_CSV"])) if r["variant"]=="dws_parallel"))')" --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/parallel_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant standard_unet --width-multiplier "$(python -c 'import csv,os; print(next(float(r["width_multiplier"]) for r in csv.DictReader(open(os.environ["CAPACITY_CSV"])) if r["variant"]=="standard_unet"))')" --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/standard_unet_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant group_unet --width-multiplier "$(python -c 'import csv,os; print(next(float(r["width_multiplier"]) for r in csv.DictReader(open(os.environ["CAPACITY_CSV"])) if r["variant"]=="group_unet"))')" --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/group_unet_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant learned_residual --width-multiplier "$(python -c 'import csv,os; print(next(float(r["width_multiplier"]) for r in csv.DictReader(open(os.environ["CAPACITY_CSV"])) if r["variant"]=="learned_residual"))')" --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/learned_residual_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant learnable_blend --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/learnable_blend_seed43.log"
```

### Loss-term screens

Run these only after the core component table. They screen one loss-term removal at a time for seed 43; the `--loss-ablation` flag removes the named term only in stages whose objective contains that term. Every screen has a separate output folder.

```bash
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/loss_no_charbonnier" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --loss-ablation charbonnier --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/loss_no_charbonnier_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/loss_no_ssim" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --loss-ablation ssim --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/loss_no_ssim_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/loss_no_edge" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --loss-ablation edge --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/loss_no_edge_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/loss_no_frequency" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --loss-ablation frequency --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/loss_no_frequency_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/loss_no_perceptual" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --loss-ablation perceptual --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/loss_no_perceptual_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/loss_s4_no_mse" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --loss-ablation mse --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/loss_s4_no_mse_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/loss_s4_no_ms_ssim" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --loss-ablation ms_ssim --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/loss_s4_no_ms_ssim_seed43.log"
python -m eca_ldnet.train --reside6k "$RESIDE6K_ROOT" --its "$ITS_ROOT" --out "$RUN_ROOT/loss_s4_no_vgg19" --stages 1 2 3 4 --stage-batch-sizes 8 16 8 16 --seed 43 --split-seed 42 --variant full --loss-ablation vgg19_perceptual --preprocessing radiometric --physics-blend 0.08 --physics-eps 0.1 --perceptual-protocol imagenet 2>&1 | tee "$RUN_ROOT/logs/loss_s4_no_vgg19_seed43.log"
```

These are screening runs, not multi-seed conclusions. Repeat loss removals that materially change validation behavior with seeds 17 and 73 before reporting an effect.

### Revised-checkpoint blend/epsilon sweep

After the full seed-43 model completes, sweep its fixed validation manifest only. This is inference-only and does not retrain the model:

```bash
python scripts/sweep_asm_blend.py --model "$RUN_ROOT/full/seed_43/models/eca_ldnet_final.keras" --manifest "$RUN_ROOT/full/seed_43/split_manifest.csv" --split validation --output "$RUN_ROOT/blend_sweep_full_seed43" --preprocessing radiometric --weights 0 0.02 0.05 0.08 0.10 0.20 --epsilons 0.05 0.1 0.2
```

Do not select settings using SOTS or another final test set. Train a learnable-blend model only as its separate `learnable_blend` run above.

## Evaluation and reporting

After training, evaluate the selected validation-best `models/stage4_best.keras` for every full/ablation seed on the same untouched test sets with `radiometric` preprocessing at 256×256. Save an evaluation manifest with exact checkpoint paths, dataset paths, preprocessing, resolution, and TTA setting; save summary and per-image CSVs. The existing radiometric CSVs for seed 17/73 are useful records, but have no manifest and cannot prove which checkpoint produced them. A separate `submitted` evaluation exists and is a preprocessing-mismatch diagnostic for radiometric-trained models. Do not combine it with the primary scores. All three archived full-model seeds use the audited 8/16/8/16 batch schedule. The full-model and ablation results are comparable only when evaluation settings match. The older epoch-28 ablations and seed-42 8% sweep are legacy results; do not present them as revised-protocol outcomes.
