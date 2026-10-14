# ECA-LDNet: Efficient Channel-Attention Lightweight Dehazing Network

A **1.48M-parameter**, physics-guided, dual-attention network for single-image
dehazing. ECA-LDNet pairs **Efficient Channel Attention (ECA)** with
**Pixel (spatial) Attention** inside a depthwise-separable U-Net, and embeds an
**Atmospheric-Scattering-Model (ASM) physics-correction layer** that estimates
atmospheric light `A` and a transmission map `t` to refine the output. It reaches
**32.53 dB / 0.972 SSIM on SOTS-Indoor** at a verified **2.171 GMACs** (4.343 GFLOPs,
256×256). It is positioned as a compact, component-analyzed design, **not** as a
state-of-the-art or lowest-compute model: FFA-Net and the DehazeFormer family reach
higher fidelity at greater cost.

> This repository is a cleaned, installable, and tested refactor of the original
> research notebooks. The model code now lives in the `eca_ldnet/` package, with
> the original notebooks preserved under `notebooks/` for full reproducibility.

---

## Highlights

- **Lightweight:** 1,481,871 parameters (5.65 MB), depthwise-separable convolutions throughout.
- **Dual attention:** ECA (channel) + Pixel Attention (spatial) in every residual block.
- **ASM-inspired guidance:** transmission-derived decoder gating and a stabilized
  correction blended into the CNN result. New models apply the correction to the
  observed hazy input; released checkpoints preserve the legacy raw-prediction input.
- **Interactive-rate:** ~72–82 ms / 256×256 image (≈12–14 FPS) on an NVIDIA Tesla P100; device-specific, not an edge-device or universal real-time claim.
- **Verified compute:** 2.171 GMACs / 4.343 GFLOPs at 256×256 from a layer-wise counter (`scripts/count_macs.py`).
- **Reproducible:** multi-stage progressive training (Charbonnier + SSIM + edge + FFT + VGG perceptual).

## Reviewer ablation package

Use [`kaggle_how_to_run/`](kaggle_how_to_run/README.md) for Kaggle cell-by-cell guides and [`laptop_how_to_run/`](laptop_how_to_run/README.md) for local terminal commands. The package has guides for four unique ablations (`no_eca`, `eca_only`, `no_attention`, and `no_physics`) at seeds 17, 73, and 43. The CLI alias `pixel_only` duplicates `no_eca` and is omitted to prevent duplicate training. See [`ablations_kaggle.md`](ablations_kaggle.md) for the index.

For Kaggle, attach ITS and RESIDE-6K as separate dataset inputs. Their mounted folders should look like this:

```text
/kaggle/input/datasets/balraj98/indoor-training-set-its-residestandard/
├── clear/
├── hazy/
├── trans/
└── metadata.csv

/kaggle/input/datasets/kmljts/reside-6k/
└── RESIDE-6K/
    ├── train/
    │   ├── GT/
    │   └── hazy/
    └── test/
```

Training reads ITS `clear/` and `hazy/`, and RESIDE-6K `RESIDE-6K/train/GT/` and `RESIDE-6K/train/hazy/`. It does not use ITS `trans/` or RESIDE `test/` for training. For laptop runs, download the dataset files separately and put them in the empty `datasets/` layout described in [`datasets/README.md`](datasets/README.md).

## Benchmark Results

Baseline figures are quoted from the cited literature and use ITS-only training,
whereas ECA-LDNet uses a mixed RESIDE-6K + ITS corpus, so this comparison is
indicative rather than controlled (see the paper, Table I).

| Method | Year | PSNR (SOTS-Indoor) | SSIM | Params |
|---|---|---|---|---|
| DCP | 2009 | 16.62 | 0.8546 | — |
| AOD-Net | 2017 | 20.51 | 0.8160 | 0.002M |
| GridDehazeNet | 2019 | 32.16 | 0.9836 | 0.956M |
| FFA-Net | 2020 | 36.39 | 0.9886 | 4.456M |
| DehazeFormer-S | 2023 | 36.82 | 0.9920 | 1.283M |
| DehazeFormer-B | 2023 | 37.84 | 0.9940 | 2.514M |
| DehazeFormer-L | 2023 | 40.05 | 0.9960 | 25.44M |
| **ECA-LDNet (Ours)** | **2026** | **32.53** | **0.9717** | **1.482M** |

ECA-LDNet does not lead SOTS-Indoor; FFA-Net and the DehazeFormer family are higher.
The following are the **submitted legacy-checkpoint results**. They are not results
for the corrected observed-input revision model; report new revision scores only
after controlled retraining/evaluation.

| Dataset | PSNR (dB) | SSIM |
|---|---|---|
| SOTS-Indoor | 32.53 | 0.9717 |
| SOTS-Outdoor | 31.94 | 0.9769 |
| RESIDE-6K | 30.20 | 0.9641 |

See [`results/final_evaluation/final_report.txt`](results/final_evaluation/final_report.txt)
and [`results/model_architecture_summary.txt`](results/model_architecture_summary.txt).

## Architecture

```
Hazy (256×256×3)
   │
   ├─ Encoder: 4× [Residual(DWSConv×2 → ECA → PixelAttention) → MaxPool]  (32→64→128→256)
   │
   ├─ Bottleneck (512) ──► Physics head:  A (atm. light, 3)   t (transmission, 16×16×1)
   │
   ├─ Decoder: 4× [Up → Concat skip → Residual], gated by (1 − t) haze-density maps
   │
   └─ Refine → raw sigmoid (256×256×3)
              └─ PhysicsCorrectionLayer(raw, A, t) ──► Dehazed (256×256×3)
```

## Installation

```bash
git clone https://github.com/manpreet28111995/Image-Dehazing-using-ECA-LDNet.git
cd Image-Dehazing-using-ECA-LDNet
pip install -e .          # or: pip install -r requirements.txt
```

Python 3.10–3.12, TensorFlow 2.16–2.19, **Keras ≥ 3.9** (required to load the released
`.keras` checkpoints — earlier Keras 3 releases reject their serialized
`quantization_config` field).

## Quick Start

**Dehaze a single image** with a released checkpoint:

```bash
python -m eca_ldnet.inference \
    --model checkpoints/stage5_final_ema_model.keras \
    --input path/to/hazy.png \
    --output dehazed.png
```

**Build the model in Python:**

```python
from eca_ldnet import build_eca_ldnet
model = build_eca_ldnet()
model.summary()                 # 1,481,871 params
```

**Evaluate on benchmarks:**

```bash
python -m eca_ldnet.evaluate \
    --model checkpoints/stage5_final_ema_model.keras \
    --sots /path/to/SOTS --reside6k /path/to/RESIDE-6K
```

**Train from scratch (recovered S1-S4 stages with grouped validation):**

```bash
python -m eca_ldnet.train --reside6k /data/RESIDE-6K --its /data/ITS \
    --out ./runs --stages 1 2 3 4 --seed 42 --variant full \
    --preprocessing radiometric --perceptual-protocol imagenet
```

Stage 1-2 use the submitted primary loss; Stage 3 uses its distinct reweighted
loss; Stage 4 uses the recovered VGG19 plus multi-scale SSIM objective and EMA.
The source repository does not yet establish the exact Stage 5 recipe, so the
CLI does not include Stage 5 in its default run. Do not describe Stage 5 as an
exact reproduction until that recipe is verified. `--batch-size` can be set
after the GPU pilot. Stage checkpoints ending `_best.keras` are selected by
validation PSNR; `_latest.keras` is for recovery and reload starts a fresh
optimizer schedule.

The package targets Python 3.10–3.12 with TensorFlow 2.16–2.19 and Keras 3.9+.
The current workstation interpreter is Python 3.14 and has no TensorFlow, so
use Kaggle for this revision unless you first set up a compatible Python and
working TensorFlow GPU runtime in WSL2.

Controlled variants include `no_eca`, `eca_only`, `no_attention` (the CLI alias `pixel_only` is identical to `no_eca`),
`no_physics`, `se`, `cbam`, `pa_eca`, `parallel`, `standard_unet`, `group_unet`,
`learned_residual`, and `learnable_blend` (a trainable scalar constrained to
[0, 0.2], initialized at 0.08). Use `scripts/match_model_capacity.py` to find width
multipliers that closely match parameter counts, then pass the selected value as
`--width-multiplier`. Keep the same split seed,
preprocessing, stage budget, and data for every comparison.

**Screen blend weights on a validation manifest, not on SOTS test images:**

```bash
python scripts/sweep_asm_blend.py --model Models/stage5_final_ema_model.keras \
    --manifest ./runs/full/seed_42/split_manifest.csv \
    --output ./runs/blend_sweep --preprocessing submitted
```

Native-resolution evaluation is available with `--native-resolution`; the model
pads images to a multiple of 16 and crops predictions back to the original size.

## Datasets

| Name | Use | Source |
|---|---|---|
| RESIDE-6K | train + test | `kmljts/reside-6k` |
| ITS (indoor) | train | `balraj98/indoor-training-set-its-residestandard` |
| SOTS (indoor/outdoor) | test | `balraj98/synthetic-objective-testing-set-sots-reside` |

## Repository Layout

```
eca_ldnet/                  Installable package
  layers.py                 ECABlock, PixelAttention, PhysicsCorrectionLayer
  blocks.py                 DWS conv, residual, encoder/decoder blocks
  model.py                  build_eca_ldnet() + load_pretrained()
  losses.py                 Charbonnier/SSIM/edge/FFT/VGG losses + metrics
  data.py                   Dataset discovery, preprocessing, tf.data pipeline
  callbacks.py              Warmup-cosine LR schedule + logging callbacks
  train.py                  Five-stage training/ablation CLI with grouped splits
  evaluate.py               Benchmark PSNR/SSIM + latency/MACs CLI
  inference.py              Single-image dehazing CLI
tests/                      Architecture, protocol, loss, and serialization tests
notebooks/                  Original and reviewer-revision Kaggle notebooks
checkpoints/                Released stage-5 models (.keras)
results/                    Per-stage histories, plots, final evaluation report
docs/                       Project report
```

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

Tests verify the exact parameter count (1,481,871), I/O shapes, output range,
presence of the dual-attention and physics modules, save/load round-trip, and
loss/metric correctness. They skip automatically if TensorFlow is not installed.

## Citation

```bibtex
@software{2026ecaldnet,
  author = {Manpreet Singh, Sai Deekshith Lekkalla, Rohith Reddy Bellibatlu, Manmeet Singh Kapoor},
  title  = {ECA-LDNet: Efficient Channel-Attention Lightweight Dehazing Network},
  year   = {2026},
  url    = {https://github.com/manpreet28111995/Image-Dehazing-using-ECA-LDNet}
}
```

## License

Released under the [MIT License](LICENSE).
