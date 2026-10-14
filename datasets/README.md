# Dataset setup

Raw benchmark images are not included. Obtain each dataset from its cited source and follow its access and redistribution terms. Record source version, download date, file counts, and SHA-256 hashes for the exact files used.

## Training inputs

- ITS: Kaggle `balraj98/indoor-training-set-its-residestandard` or official RESIDE-Standard. Use `clear/` and `hazy/`.
- RESIDE-6K: Kaggle `kmljts/reside-6k`. Use `RESIDE-6K/train/GT/` and `RESIDE-6K/train/hazy/`.

Expected local layout:

```text
datasets/
├── ITS/{clear,hazy}/
└── RESIDE-6K/RESIDE-6K/train/{GT,hazy}/
```

## Evaluation inputs

Use the exact named partitions from the paper: SOTS-Indoor/Outdoor, O-HAZE, I-HAZE, NH-HAZE, Dense-Haze (NTIRE 2019 release), and RESIDE-6K-Test where listed. The manuscript reports five-pair I-HAZE and NH-HAZE test subsets in its primary protocol; its separate follow-up uses labeled release partitions. Dense-Haze follow-up pools 55 release pairs, not just the challenge-test subset. See `paper/main.tex`, `paper/supplementary.tex`, and run manifests for counts and preprocessing.

Dataset images must not be committed unless their terms permit redistribution. Split manifests currently retain original Kaggle absolute paths; rebase them to local dataset roots before local execution while preserving filenames and content hashes.
