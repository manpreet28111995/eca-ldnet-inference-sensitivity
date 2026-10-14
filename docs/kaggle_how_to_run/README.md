# Kaggle ablation guides

Choose one guide by seed and ablation. Every guide has the Kaggle input setup, dataset checks, four stage cells, a restore cell before Stages 2–4, and a ZIP export cell after each completed stage.

Attach the shared model ZIP as a Kaggle Model input and attach RESIDE-6K and ITS as separate dataset inputs. The expected dataset tree is:

```text
/kaggle/input/datasets/balraj98/indoor-training-set-its-residestandard/
├── clear/
├── hazy/
├── trans/
└── metadata.csv

/kaggle/input/datasets/kmljts/reside-6k/
└── RESIDE-6K/
    ├── test/
    │   ├── GT/
    │   └── hazy/
    └── train/
        ├── GT/
        └── hazy/
```

Training reads ITS `hazy/` and `clear/`, and RESIDE-6K `RESIDE-6K/train/hazy/` and `RESIDE-6K/train/GT/`. It does not use ITS `trans/` or RESIDE `test/` for training. Change the example `MODEL_INPUT` path only if Kaggle mounted your model under a different account or slug. The model input is mounted automatically; the guide does not ask you to unzip the code ZIP manually.

At the end of each stage, download `seed<seed>_<variant>_stage<stage>.zip`. To continue in a fresh Kaggle session, attach that ZIP as an input, run the setup cell, then run the next stage restore cell. The restore cell handles either the ZIP file or its extracted run folder in `/kaggle/input`, and confirms that `eca_ldnet_final.keras` is ready before training continues.

The guides use 40 total epochs (18/8/6/8), batch sizes 8/16/8/16, split seed 42, and four variants: `no_eca`, `eca_only`, `pixel_only`, and `no_attention`. These are shortened-budget screening ablations; the shorter epoch budget also shortens the learning-rate schedules.

- Seed 73: `seed73/`
- Seed 17: `seed17/`
- Seed 43: `seed43/`
