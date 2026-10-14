"""Benchmark evaluation: PSNR/SSIM over SOTS / RESIDE-6K + latency/MACs + CLI.

Example
-------
    python -m eca_ldnet.evaluate --model checkpoints/stage5_final_ema_model.keras \\
        --sots /path/to/SOTS --reside6k /path/to/RESIDE-6K --output ./eval_out
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import tensorflow as tf

from .config import IMG_SIZE
from .data import find_dir, get_gt_path, load_image_pair


def _psnr(p, g):
    return float(tf.image.psnr(p[np.newaxis], g[np.newaxis], 1.0).numpy()[0])


def _ssim(p, g):
    return float(tf.image.ssim(p[np.newaxis], g[np.newaxis], 1.0).numpy()[0])


def predict_full(model, img):
    h, w = img.shape[:2]
    pad_h, pad_w = (-h) % 16, (-w) % 16
    if pad_h or pad_w:
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        padded = img
    pred = model.predict(padded[np.newaxis], verbose=0)[0]
    return np.clip(pred[:h, :w], 0, 1).astype(np.float32)


def evaluate_dataset(
    model, hazy_dir, gt_dir, name="", limit=None,
    preprocessing="radiometric", native_resolution=False,
):
    """Mean PSNR / SSIM over a dataset directory; returns (psnr, ssim) or None."""
    if not hazy_dir or not gt_dir:
        return None
    files = sorted(f for f in os.listdir(hazy_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if limit:
        files = files[:limit]
    psnrs, ssims = [], []
    for fname in files:
        gp = get_gt_path(fname, gt_dir)
        if not gp:
            continue
        hazy, gt = load_image_pair(
            os.path.join(hazy_dir, fname), gp,
            size=None if native_resolution else IMG_SIZE,
            preprocessing=preprocessing,
        )
        if hazy is None:
            continue
        pred = predict_full(model, hazy)
        psnrs.append(_psnr(pred, gt))
        ssims.append(_ssim(pred, gt))
    if not psnrs:
        return None
    print(f"  {name:<14}: PSNR={np.mean(psnrs):.4f}  SSIM={np.mean(ssims):.4f}  (n={len(psnrs)})")
    return float(np.mean(psnrs)), float(np.mean(ssims))


def _layer_macs(layer):
    """Analytic multiply-accumulate count for one Keras layer from its config."""
    cls = layer.__class__.__name__
    try:
        out = layer.output.shape
    except Exception:
        return 0

    def spatial(shape):
        dims = [d for d in shape[1:-1] if d is not None]
        p = 1
        for d in dims:
            p *= int(d)
        return p

    if cls == "Conv2D":
        kh, kw = layer.kernel_size
        cin = int(layer.input.shape[-1]); cout = int(out[-1])
        g = getattr(layer, "groups", 1) or 1
        return spatial(out) * cout * (cin // g) * kh * kw
    if cls == "DepthwiseConv2D":
        kh, kw = layer.kernel_size
        cin = int(layer.input.shape[-1]); dm = getattr(layer, "depth_multiplier", 1)
        return spatial(out) * cin * dm * kh * kw
    if cls == "SeparableConv2D":
        kh, kw = layer.kernel_size
        cin = int(layer.input.shape[-1]); cout = int(out[-1])
        return spatial(out) * cin * kh * kw + spatial(out) * cout * cin
    if cls == "Conv1D":
        (k,) = layer.kernel_size
        cin = int(layer.input.shape[-1]); cout = int(out[-1])
        steps = int(out[1]) if out[1] is not None else 1
        return steps * cout * cin * k
    if cls == "Dense":
        cin = int(layer.input.shape[-1]); units = int(out[-1])
        lead = 1
        for d in out[1:-1]:
            if d is not None:
                lead *= int(d)
        return lead * cin * units
    return 0


def analytic_macs(model, resolution=IMG_SIZE):
    """Count MACs at a declared resolution, even for dynamic-input checkpoints."""
    spatial = model.input_shape[1:3]
    counted_model = model
    if any(dim is None for dim in spatial):
        # Dynamic H/W graphs otherwise have no concrete output maps and the
        # analytic counter would silently undercount every convolution.
        fixed_input = tf.keras.Input(shape=(resolution, resolution, 3), name="mac_count_input")
        counted_model = tf.keras.models.clone_model(model, input_tensors=fixed_input)
    return sum(_layer_macs(layer) for layer in counted_model.layers)


def measure_latency_and_flops(model, runs=100, warmup=20):
    """Wall-clock latency / FPS plus a valid layer-wise MAC count for the model."""
    dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    for _ in range(warmup):
        model.predict(dummy, verbose=0)
    t0 = time.time()
    for _ in range(runs):
        model.predict(dummy, verbose=0)
    avg_ms = ((time.time() - t0) / runs) * 1000.0
    params_m = model.count_params() / 1e6
    macs_g = analytic_macs(model, resolution=IMG_SIZE) / 1e9
    return {"avg_ms": avg_ms, "fps": 1000.0 / avg_ms, "macs_g": macs_g, "params_m": params_m}


def main():
    p = argparse.ArgumentParser(description="ECA-LDNet benchmark evaluation")
    p.add_argument("--model", required=True)
    p.add_argument("--sots", default=None, help="SOTS root (indoor/ + outdoor/)")
    p.add_argument("--reside6k", default=None, help="RESIDE-6K root")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default="./eval_out")
    p.add_argument("--preprocessing", choices=["radiometric", "submitted"], default="radiometric")
    p.add_argument("--native-resolution", action="store_true", help="Evaluate full images with dynamic spatial inputs")
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    from .model import load_pretrained

    model = load_pretrained(args.model)
    if args.native_resolution:
        from .model import build_eca_ldnet

        legacy_input = len(model.get_layer("physics_output").input) == 3
        dynamic_model = build_eca_ldnet(
            img_size=None, physics_uses_observed_input=not legacy_input
        )
        dynamic_model.set_weights(model.get_weights())
        model = dynamic_model
    print(f"Model: {model.count_params():,} params")

    lat = measure_latency_and_flops(model)
    print(f"Latency: {lat['avg_ms']:.2f} ms  ({lat['fps']:.2f} FPS)  MACs~{lat['macs_g']:.2f}G")

    if args.sots:
        evaluate_dataset(
            model,
            find_dir(args.sots, ["indoor/hazy", "SOTS/indoor/hazy"]),
            find_dir(args.sots, ["indoor/clear", "indoor/gt", "indoor/GT"]),
            "SOTS-Indoor",
            args.limit,
            args.preprocessing,
            args.native_resolution,
        )
        evaluate_dataset(
            model,
            find_dir(args.sots, ["outdoor/hazy", "SOTS/outdoor/hazy"]),
            find_dir(args.sots, ["outdoor/clear", "outdoor/Clear", "outdoor/GT"]),
            "SOTS-Outdoor",
            args.limit,
            args.preprocessing,
            args.native_resolution,
        )
    if args.reside6k:
        evaluate_dataset(
            model,
            find_dir(args.reside6k, ["test/hazy", "Test/hazy"]),
            find_dir(args.reside6k, ["test/GT", "test/gt", "test/clear"]),
            "RESIDE-6K",
            args.limit,
            args.preprocessing,
            args.native_resolution,
        )


if __name__ == "__main__":
    main()
