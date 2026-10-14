# Laptop ablation guides and dataset placement

Choose one guide from `seed73/`, `seed17/`, or `seed43/`. Each seed folder has a full guide for each of the four variants. Commands are local terminal commands and each one is written out for its exact seed, variant, and stage.

## 1. Extract and open the package

Extract the shared ZIP on the laptop. Open a terminal in the extracted project root—the folder containing `eca_ldnet/`, `scripts/`, and `datasets/`. Keep using this same project path between stages so the saved run settings continue to match.

## 2. Download and place datasets

The ZIP contains empty dataset folders only; it does not contain dataset images. Download RESIDE-6K and ITS separately and place their training images like this:

```text
datasets/
├── RESIDE-6K/
│   └── train/
│       ├── hazy/    # hazy training images
│       └── GT/      # matching clear images
└── ITS/
    ├── hazy/        # hazy training images
    └── clear/       # matching clear images
```

The placeholder directories are already in the ZIP. Keep each hazy image paired with its clear target using the dataset's matching filenames. See `datasets/README.md` for the same layout.

## 3. Set up Python

The project supports Python 3.10–3.12. Python 3.11 is a good choice.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Linux or macOS terminal:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Check that TensorFlow imports and check GPU visibility:

```text
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

An empty GPU list means it will train on CPU, which can be much slower. A working local GPU requires a TensorFlow-supported driver/runtime setup for that laptop; use the platform’s compatible TensorFlow GPU setup before beginning a long run.

## 4. Check dataset pairs

From the project root, run:

```text
python scripts/check_local_datasets.py --reside6k datasets/RESIDE-6K --its datasets/ITS
```

Continue only if it reports matched pairs for both datasets.

## 5. Run and save stages

Open one guide, then copy its Stage 1 training command and packaging command. Each stage writes its checkpoints, history, `run_config.json`, and `split_manifest.csv` under `ablation_runs/<variant>/seed_<seed>/`. The packaging command saves a ZIP under `save_here/seed<seed>_<variant>/`.

Before a later stage, run the guide's restore command for the immediately previous ZIP. It prints `Previous Stage N model restored successfully; ready to run Stage N+1` when the checkpoint is present. Then run that stage's full training command and packaging command. The restore helper only accepts a ZIP with the expected seed, variant, and stage name.

Example ZIP names for seed 73 `no_eca`:

```text
save_here/seed73_no_eca/seed73_no_eca_stage1.zip
save_here/seed73_no_eca/seed73_no_eca_stage2.zip
save_here/seed73_no_eca/seed73_no_eca_stage3.zip
save_here/seed73_no_eca/seed73_no_eca_stage4.zip
```

Keep this laptop's dataset directory at the same project path between stages; the saved configuration records the dataset paths. Kaggle stage ZIPs use Kaggle dataset paths and should not be resumed on the laptop. Laptop ZIPs can be resumed on the same laptop after re-extraction at the same project path.

The 40-epoch schedule is 18/8/6/8 with batches 8/16/8/16. Because the epoch budget is shortened, report these as screening ablations rather than full-budget replications.
