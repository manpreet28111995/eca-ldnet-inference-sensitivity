# Pixel attention only — seed 43, 40 epochs

Use this guide for the **SEED43** Kaggle job. This run trains the `pixel_only` ablation from fresh weights. It does not load or require a full-model checkpoint.

This is a 40-epoch shortened-budget screening experiment. The model architecture, stage objectives, optimizer family, data split, preprocessing, physics settings, and batch sizes follow the shared full-model code. The 18/8/6/8 budget also shortens each stage learning-rate schedule, so report it as a 40-epoch ablation, not a full-budget replication. Stage 5 is not used.

## Kaggle inputs and code setup

Attach the automatically extracted Kaggle model input containing the uploaded ZIP, plus RESIDE-6K and ITS. The example model path below is the one you supplied. If your Kaggle owner or model slug differs, edit only `MODEL_INPUT` to that account’s mounted input path. No manual ZIP extraction is used here.

Run this setup cell first:

```python
from pathlib import Path
import shutil, sys, subprocess
import tensorflow as tf

MODEL_INPUT = Path("/kaggle/input/models/lucky998s/eca-ldnet-revision/pytorch/default/1")  # change only if your mounted model path differs
CODE_ROOT = Path("/kaggle/working/eca_ldnet_revision")
if (MODEL_INPUT / "eca_ldnet" / "train.py").is_file():
    MODEL_SOURCE = MODEL_INPUT
else:
    matches = list(MODEL_INPUT.rglob("eca_ldnet/train.py"))
    assert matches, f"The input must contain eca_ldnet/train.py: {MODEL_INPUT}"
    MODEL_SOURCE = matches[0].parent.parent
if not (CODE_ROOT / "eca_ldnet" / "train.py").is_file():
    shutil.copytree(MODEL_SOURCE, CODE_ROOT, dirs_exist_ok=True)
assert (CODE_ROOT / "eca_ldnet" / "train.py").is_file(), CODE_ROOT
sys.path.insert(0, str(CODE_ROOT))

RESIDE_ROOT = Path("/kaggle/input/datasets/kmljts/reside-6k")
ITS_ROOT = Path("/kaggle/input/datasets/balraj98/indoor-training-set-its-residestandard")
assert RESIDE_ROOT.is_dir(), f"Missing dataset: {RESIDE_ROOT}"
assert ITS_ROOT.is_dir(), f"Missing dataset: {ITS_ROOT}"
GPUS = tf.config.list_physical_devices("GPU")
assert GPUS, "Enable a Kaggle GPU accelerator before training."
from eca_ldnet.data import find_dir, collect_pairs
RESIDE_HAZY = find_dir(str(RESIDE_ROOT), ["train/hazy", "Train/hazy"])
RESIDE_GT = find_dir(str(RESIDE_ROOT), ["train/GT", "train/gt", "train/clear"])
ITS_HAZY = find_dir(str(ITS_ROOT), ["hazy", "Hazy", "train/hazy"])
ITS_GT = find_dir(str(ITS_ROOT), ["clear", "GT", "gt", "train/clear", "train/GT"])
RESIDE_PAIRS = collect_pairs(RESIDE_HAZY, RESIDE_GT, "RESIDE-6K check")
ITS_PAIRS = collect_pairs(ITS_HAZY, ITS_GT, "ITS check")
assert RESIDE_PAIRS and ITS_PAIRS, "No paired training data found; verify attached dataset directories."

SEED = 43
VARIANT = "pixel_only"
OUT_ROOT = Path("/kaggle/working/ablation_runs")
RUN_DIR = OUT_ROOT / VARIANT / f"seed_{SEED}"
EXPORT_ROOT = Path("/kaggle/working/save_here") / f"seed{SEED}_{VARIANT}"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
print("Code:", CODE_ROOT)
print("Run output:", RUN_DIR)
print("RESIDE:", RESIDE_ROOT)
print("ITS:", ITS_ROOT)
print("GPU:", GPUS)
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

Protocol for every stage: split seed 42, batches 8/16/8/16, radiometric preprocessing, physics blend 0.08, epsilon 0.1, ImageNet perceptual protocol. Early stopping is disabled so the run completes exactly the stated 40 epochs.

Each training cell below is followed by its own export cell. For a new Kaggle session, download the ZIP from the previous stage and save it on your computer under `save_here/seed<SEED>_<VARIANT>/`. Add that ZIP as a private Kaggle dataset (or a new version of your private continuation dataset), attach it as a Kaggle input, then run the next stage restore cell. The cell searches `/kaggle/input`, restores into the same seed/variant folder, and prints `Previous Stage N model restored successfully; ready to run Stage N+1`. No manual extraction is required. If the session stays active, just run the restore cell; it detects the checkpoint already present.

## Stage 1 — 18 epochs

Run this Stage 1 cell:

```python
import subprocess, sys

cmd = [
    sys.executable, "-m", "eca_ldnet.train",
    "--reside6k", str(RESIDE_ROOT), "--its", str(ITS_ROOT),
    "--out", str(OUT_ROOT), "--stages", "1",
    "--max-epochs", "18", "--disable-early-stopping",
    "--stage-batch-sizes", "8", "16", "8", "16",
    "--seed", str(SEED), "--split-seed", "42", "--variant", VARIANT,
    "--preprocessing", "radiometric", "--physics-blend", "0.08",
    "--physics-eps", "0.1", "--perceptual-protocol", "imagenet",
    "--save-every-epoch",
]
subprocess.run(cmd, cwd=CODE_ROOT, check=True)

STAGE = 1
MODELS = RUN_DIR / "models"
HISTORY = RUN_DIR / "history" / f"seed{SEED}_{VARIANT}_stage{STAGE}_history.csv"
CONFIG = RUN_DIR / "run_config.json"
SPLIT_MANIFEST = RUN_DIR / "split_manifest.csv"
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_latest.keras").is_file()
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_best.keras").is_file()
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_epoch018.keras").is_file()
assert HISTORY.is_file()
assert CONFIG.is_file(), f"Run config missing: {CONFIG}"
assert SPLIT_MANIFEST.is_file(), f"Split manifest missing: {SPLIT_MANIFEST}"
print("Stage", STAGE, "complete; history:", HISTORY)
print("Run config:", CONFIG)
print("Split manifest:", SPLIT_MANIFEST)
print("Best checkpoint:", MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_best.keras")
```

### Export after Stage 1

Run this separate cell and download the ZIP from the Kaggle notebook output:

```python
import shutil
ARCHIVE_BASE = EXPORT_ROOT / f"seed{SEED}_{VARIANT}_stage1"
ARCHIVE_PATH = Path(shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=str(OUT_ROOT), base_dir=str(Path(VARIANT) / f"seed_{SEED}")))
assert ARCHIVE_PATH.is_file()
print("Download this stage ZIP to your computer’s save_here folder:", ARCHIVE_PATH)
print("For a new Kaggle session, add this ZIP as a private Kaggle input before running the next stage restore cell.")
```

## Stage 2 — 8 epochs

Restore the previous-stage ZIP automatically if this is a new Kaggle session:

```python
import shutil
PRIOR_STAGE = 1
STAGE = 2
PRIOR_CHECKPOINT = RUN_DIR / "models" / "eca_ldnet_final.keras"
PRIOR_ARCHIVE_NAME = f"seed{SEED}_{VARIANT}_stage{PRIOR_STAGE}.zip"
if not PRIOR_CHECKPOINT.is_file():
    prior_archives = list(Path("/kaggle/input").rglob(PRIOR_ARCHIVE_NAME))
    assert len(prior_archives) == 1, f"Expected exactly one attached {PRIOR_ARCHIVE_NAME}; found {len(prior_archives)}. Attach only the matching previous-stage ZIP."
    print(f"Restoring previous Stage {PRIOR_STAGE} archive:", prior_archives[0])
    shutil.unpack_archive(str(prior_archives[0]), str(OUT_ROOT))
else:
    print(f"Previous Stage {PRIOR_STAGE} checkpoint is already present:", PRIOR_CHECKPOINT)
assert PRIOR_CHECKPOINT.is_file(), f"Previous stage model missing after restore: {PRIOR_CHECKPOINT}"
print(f"Previous Stage {PRIOR_STAGE} model restored successfully; ready to run Stage {STAGE}:", PRIOR_CHECKPOINT)
```

Run this Stage 2 cell:

```python
import subprocess, sys

cmd = [
    sys.executable, "-m", "eca_ldnet.train",
    "--reside6k", str(RESIDE_ROOT), "--its", str(ITS_ROOT),
    "--out", str(OUT_ROOT), "--stages", "2",
    "--resume", str(PRIOR_CHECKPOINT),
    "--max-epochs", "8", "--disable-early-stopping",
    "--stage-batch-sizes", "8", "16", "8", "16",
    "--seed", str(SEED), "--split-seed", "42", "--variant", VARIANT,
    "--preprocessing", "radiometric", "--physics-blend", "0.08",
    "--physics-eps", "0.1", "--perceptual-protocol", "imagenet",
    "--save-every-epoch",
]
subprocess.run(cmd, cwd=CODE_ROOT, check=True)

STAGE = 2
MODELS = RUN_DIR / "models"
HISTORY = RUN_DIR / "history" / f"seed{SEED}_{VARIANT}_stage{STAGE}_history.csv"
CONFIG = RUN_DIR / "run_config.json"
SPLIT_MANIFEST = RUN_DIR / "split_manifest.csv"
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_latest.keras").is_file()
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_best.keras").is_file()
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_epoch008.keras").is_file()
assert HISTORY.is_file()
assert CONFIG.is_file(), f"Run config missing: {CONFIG}"
assert SPLIT_MANIFEST.is_file(), f"Split manifest missing: {SPLIT_MANIFEST}"
print("Stage", STAGE, "complete; history:", HISTORY)
print("Run config:", CONFIG)
print("Split manifest:", SPLIT_MANIFEST)
print("Best checkpoint:", MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_best.keras")
```

### Export after Stage 2

Run this separate cell and download the ZIP from the Kaggle notebook output:

```python
import shutil
ARCHIVE_BASE = EXPORT_ROOT / f"seed{SEED}_{VARIANT}_stage2"
ARCHIVE_PATH = Path(shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=str(OUT_ROOT), base_dir=str(Path(VARIANT) / f"seed_{SEED}")))
assert ARCHIVE_PATH.is_file()
print("Download this stage ZIP to your computer’s save_here folder:", ARCHIVE_PATH)
print("For a new Kaggle session, add this ZIP as a private Kaggle input before running the next stage restore cell.")
```

## Stage 3 — 6 epochs

Restore the previous-stage ZIP automatically if this is a new Kaggle session:

```python
import shutil
PRIOR_STAGE = 2
STAGE = 3
PRIOR_CHECKPOINT = RUN_DIR / "models" / "eca_ldnet_final.keras"
PRIOR_ARCHIVE_NAME = f"seed{SEED}_{VARIANT}_stage{PRIOR_STAGE}.zip"
if not PRIOR_CHECKPOINT.is_file():
    prior_archives = list(Path("/kaggle/input").rglob(PRIOR_ARCHIVE_NAME))
    assert len(prior_archives) == 1, f"Expected exactly one attached {PRIOR_ARCHIVE_NAME}; found {len(prior_archives)}. Attach only the matching previous-stage ZIP."
    print(f"Restoring previous Stage {PRIOR_STAGE} archive:", prior_archives[0])
    shutil.unpack_archive(str(prior_archives[0]), str(OUT_ROOT))
else:
    print(f"Previous Stage {PRIOR_STAGE} checkpoint is already present:", PRIOR_CHECKPOINT)
assert PRIOR_CHECKPOINT.is_file(), f"Previous stage model missing after restore: {PRIOR_CHECKPOINT}"
print(f"Previous Stage {PRIOR_STAGE} model restored successfully; ready to run Stage {STAGE}:", PRIOR_CHECKPOINT)
```

Run this Stage 3 cell:

```python
import subprocess, sys

cmd = [
    sys.executable, "-m", "eca_ldnet.train",
    "--reside6k", str(RESIDE_ROOT), "--its", str(ITS_ROOT),
    "--out", str(OUT_ROOT), "--stages", "3",
    "--resume", str(PRIOR_CHECKPOINT),
    "--max-epochs", "6", "--disable-early-stopping",
    "--stage-batch-sizes", "8", "16", "8", "16",
    "--seed", str(SEED), "--split-seed", "42", "--variant", VARIANT,
    "--preprocessing", "radiometric", "--physics-blend", "0.08",
    "--physics-eps", "0.1", "--perceptual-protocol", "imagenet",
    "--save-every-epoch",
]
subprocess.run(cmd, cwd=CODE_ROOT, check=True)

STAGE = 3
MODELS = RUN_DIR / "models"
HISTORY = RUN_DIR / "history" / f"seed{SEED}_{VARIANT}_stage{STAGE}_history.csv"
CONFIG = RUN_DIR / "run_config.json"
SPLIT_MANIFEST = RUN_DIR / "split_manifest.csv"
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_latest.keras").is_file()
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_best.keras").is_file()
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_epoch006.keras").is_file()
assert HISTORY.is_file()
assert CONFIG.is_file(), f"Run config missing: {CONFIG}"
assert SPLIT_MANIFEST.is_file(), f"Split manifest missing: {SPLIT_MANIFEST}"
print("Stage", STAGE, "complete; history:", HISTORY)
print("Run config:", CONFIG)
print("Split manifest:", SPLIT_MANIFEST)
print("Best checkpoint:", MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_best.keras")
```

### Export after Stage 3

Run this separate cell and download the ZIP from the Kaggle notebook output:

```python
import shutil
ARCHIVE_BASE = EXPORT_ROOT / f"seed{SEED}_{VARIANT}_stage3"
ARCHIVE_PATH = Path(shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=str(OUT_ROOT), base_dir=str(Path(VARIANT) / f"seed_{SEED}")))
assert ARCHIVE_PATH.is_file()
print("Download this stage ZIP to your computer’s save_here folder:", ARCHIVE_PATH)
print("For a new Kaggle session, add this ZIP as a private Kaggle input before running the next stage restore cell.")
```

## Stage 4 — 8 epochs

Restore the previous-stage ZIP automatically if this is a new Kaggle session:

```python
import shutil
PRIOR_STAGE = 3
STAGE = 4
PRIOR_CHECKPOINT = RUN_DIR / "models" / "eca_ldnet_final.keras"
PRIOR_ARCHIVE_NAME = f"seed{SEED}_{VARIANT}_stage{PRIOR_STAGE}.zip"
if not PRIOR_CHECKPOINT.is_file():
    prior_archives = list(Path("/kaggle/input").rglob(PRIOR_ARCHIVE_NAME))
    assert len(prior_archives) == 1, f"Expected exactly one attached {PRIOR_ARCHIVE_NAME}; found {len(prior_archives)}. Attach only the matching previous-stage ZIP."
    print(f"Restoring previous Stage {PRIOR_STAGE} archive:", prior_archives[0])
    shutil.unpack_archive(str(prior_archives[0]), str(OUT_ROOT))
else:
    print(f"Previous Stage {PRIOR_STAGE} checkpoint is already present:", PRIOR_CHECKPOINT)
assert PRIOR_CHECKPOINT.is_file(), f"Previous stage model missing after restore: {PRIOR_CHECKPOINT}"
print(f"Previous Stage {PRIOR_STAGE} model restored successfully; ready to run Stage {STAGE}:", PRIOR_CHECKPOINT)
```

Run this Stage 4 cell:

```python
import subprocess, sys

cmd = [
    sys.executable, "-m", "eca_ldnet.train",
    "--reside6k", str(RESIDE_ROOT), "--its", str(ITS_ROOT),
    "--out", str(OUT_ROOT), "--stages", "4",
    "--resume", str(PRIOR_CHECKPOINT),
    "--max-epochs", "8", "--disable-early-stopping",
    "--stage-batch-sizes", "8", "16", "8", "16",
    "--seed", str(SEED), "--split-seed", "42", "--variant", VARIANT,
    "--preprocessing", "radiometric", "--physics-blend", "0.08",
    "--physics-eps", "0.1", "--perceptual-protocol", "imagenet",
    "--save-every-epoch",
]
subprocess.run(cmd, cwd=CODE_ROOT, check=True)

STAGE = 4
MODELS = RUN_DIR / "models"
HISTORY = RUN_DIR / "history" / f"seed{SEED}_{VARIANT}_stage{STAGE}_history.csv"
CONFIG = RUN_DIR / "run_config.json"
SPLIT_MANIFEST = RUN_DIR / "split_manifest.csv"
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_latest.keras").is_file()
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_best.keras").is_file()
assert (MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_epoch008.keras").is_file()
assert HISTORY.is_file()
assert CONFIG.is_file(), f"Run config missing: {CONFIG}"
assert SPLIT_MANIFEST.is_file(), f"Split manifest missing: {SPLIT_MANIFEST}"
print("Stage", STAGE, "complete; history:", HISTORY)
print("Run config:", CONFIG)
print("Split manifest:", SPLIT_MANIFEST)
print("Best checkpoint:", MODELS / f"seed{SEED}_{VARIANT}_stage{STAGE}_best.keras")
```

### Export after Stage 4

Run this separate cell and download the ZIP from the Kaggle notebook output:

```python
import shutil
ARCHIVE_BASE = EXPORT_ROOT / f"seed{SEED}_{VARIANT}_stage4"
ARCHIVE_PATH = Path(shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=str(OUT_ROOT), base_dir=str(Path(VARIANT) / f"seed_{SEED}")))
assert ARCHIVE_PATH.is_file()
print("Download this stage ZIP to your computer’s save_here folder:", ARCHIVE_PATH)
print("For a new Kaggle session, add this ZIP as a private Kaggle input before running the next stage restore cell.")
```

## Final files after Stage 4

Run output directory: `/kaggle/working/ablation_runs/pixel_only/seed_43/`

The run folder contains `split_manifest.csv`, `run_config.json`, all four seed/variant-tagged epoch-history CSVs, explicit seed/variant/stage `*_latest.keras`, `*_best.keras`, and numbered per-epoch checkpoints. `models/eca_ldnet_final.keras` is the stage handoff checkpoint; after Stage 4 it is selected from the Stage 4 validation-best weights. Each stage ZIP contains the complete run folder, so restoring it cannot mix another seed or ablation.

The four accounts can train different variants in parallel. Each needs its own Kaggle GPU session and attached datasets. The code and commands do not rely on your full-model checkpoint.
