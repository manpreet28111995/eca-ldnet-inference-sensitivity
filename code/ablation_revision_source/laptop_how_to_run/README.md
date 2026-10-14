# Laptop ablation guides and dataset placement

Choose a seed folder (`seed17/`, `seed73/`, or `seed43/`). Each has full terminal guides for four distinct ablations: `no_eca`, `eca_only`, `no_attention`, and `no_physics`. The full model is the comparison reference.

`no_eca` and `pixel_only` are duplicate flags in this code (`use_eca=False`, `use_pa=True`); do not run both. `no_eca` is retained as the one guide for that condition.

## 1. Extract and open the package

Extract the ZIP and open a terminal in the project root containing `eca_ldnet/`, `scripts/`, and `datasets/`. Keep using the same project path between stages.

## 2. Download and place training datasets

The ZIP contains empty placeholders only. Download RESIDE-6K and ITS separately:

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

Keep hazy and clear images paired by matching filenames. ITS `trans/` and RESIDE `test/` are not training inputs.

## 3. Install Python dependencies

Python 3.10–3.12 is supported; Python 3.11 is recommended.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Check TensorFlow/GPU visibility:

```text
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

## 4. Verify dataset pairs

From the project root:

```text
python scripts/check_local_datasets.py --reside6k datasets/RESIDE-6K --its datasets/ITS
```

Proceed only if both datasets report matched pairs.

## 5. Run and resume

Use the exact seed/variant guide. Each stage saves checkpoints, history, `run_config.json`, and `split_manifest.csv` under `ablation_runs/<variant>/seed_<seed>/`. Package each completed stage and restore the previous stage ZIP before continuing. The guide commands are complete for the specified seed and variant.

The budget is 40 epochs (18/8/6/8), batches 8/16/8/16. It shortens the full-model learning-rate schedules; report it as screening ablations.
