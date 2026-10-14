"""Paired dehazing evaluation with per-image scores and bootstrap intervals.

LPIPS/DISTS are optional and use pyiqa/PyTorch on CPU to avoid competing with
TensorFlow for GPU memory. PSNR, SSIM, and MS-SSIM are computed at the declared
evaluation resolution. Native mode requires a dynamic-spatial model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Initialize CUDA contexts conservatively before Keras creates any variables.
# This is required on some WSL/RTX 50-series combinations; without it the
# first random initializer can fail with CUDA_ERROR_INVALID_HANDLE.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eca_ldnet.data import get_gt_path, load_image_pair
from eca_ldnet.evaluate import predict_full
from eca_ldnet.model import load_pretrained


def _bootstrap_ci(values, *, samples=5000, seed=42):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float(values.mean()), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[ids].mean(axis=1)
    return float(values.mean()), float(np.std(values, ddof=1)), *map(
        float, np.quantile(means, [0.025, 0.975])
    )


def _make_pyiqa_metrics(names):
    if not names:
        return {}, None
    try:
        import pyiqa
        import torch
    except ImportError as exc:
        raise SystemExit("Install optional pyiqa and torch to compute LPIPS/DISTS: pip install pyiqa") from exc
    metrics = {name: pyiqa.create_metric(name, device="cpu") for name in names}
    return metrics, torch


def _perceptual_scores(metrics, torch, pred, gt):
    if not metrics:
        return {}
    pred_t = torch.from_numpy(np.ascontiguousarray(pred.transpose(2, 0, 1)[None])).float()
    gt_t = torch.from_numpy(np.ascontiguousarray(gt.transpose(2, 0, 1)[None])).float()
    with torch.inference_mode():
        return {name: float(metric(pred_t, gt_t).detach().cpu().reshape(-1).mean())
                for name, metric in metrics.items()}


VARIANT_FLAGS = {
    "full": {},
    "no_eca": {"use_eca": False},
    "eca_only": {"use_eca": True, "use_pa": False},
    "no_attention": {"use_eca": False, "use_pa": False},
    "no_physics": {"use_physics": False},
}


def _load_eval_model(path, native_resolution, variant="full"):
    # Load/build on CPU first.  RTX 50-series + WSL can raise
    # CUDA_ERROR_INVALID_HANDLE during Keras variable initializers even when
    # the checkpoint itself is valid. Prediction remains eligible for GPU
    # placement after construction.
    with tf.device("/CPU:0"):
        model = load_pretrained(path, compile=False)
    if native_resolution and any(d is not None for d in model.input_shape[1:3]):
        from eca_ldnet.model import build_eca_ldnet

        # Rebuild the exact model wiring at dynamic H/W, preserving whether
        # the checkpoint uses the legacy raw-output ASM graph or observed input.
        flags = VARIANT_FLAGS[variant]
        has_physics = "physics_output" in {layer.name for layer in model.layers}
        observed_input = len(model.get_layer("physics_output").input) == 4 if has_physics else True
        # Build the variable-shape graph on CPU.  On WSL with RTX 50-series
        # GPUs, Keras' random variable initializer can otherwise fail with
        # CUDA_ERROR_INVALID_HANDLE before inference starts.  The resulting
        # model is still placed on the GPU for prediction by TensorFlow.
        with tf.device("/CPU:0"):
            dynamic = build_eca_ldnet(
                img_size=None, physics_uses_observed_input=observed_input, **flags
            )
            dynamic.set_weights(model.get_weights())
        model = dynamic
    return model


def main():
    p = argparse.ArgumentParser(description="Paired full-reference evaluation")
    p.add_argument("--model", required=True)
    p.add_argument("--variant", choices=sorted(VARIANT_FLAGS), default="full")
    p.add_argument("--hazy-dir", required=True)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--dataset", required=True, help="Dataset label recorded in outputs")
    p.add_argument("--output", required=True)
    p.add_argument("--preprocessing", choices=["radiometric", "submitted"], default="radiometric")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--native-resolution", action="store_true")
    p.add_argument("--perceptual", nargs="*", choices=["lpips", "dists"], default=[])
    p.add_argument("--tested-sample-dir", help="Directory for predictions of valid matched/readable pairs")
    p.add_argument("--save-prediction-samples", type=int, default=None,
                   help="Score every matched image but save predictions for at most this many uniformly sampled pairs")
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model = _load_eval_model(args.model, args.native_resolution, args.variant)
    fixed_hw = model.input_shape[1:3]
    if args.native_resolution and any(d is not None for d in fixed_hw):
        raise SystemExit(
            f"Native evaluation needs a dynamic model; checkpoint input is {model.input_shape}. "
            "Use --size 256 for this submitted fixed-resolution checkpoint."
        )
    metrics, torch = _make_pyiqa_metrics(args.perceptual)
    os.makedirs(args.output, exist_ok=True)
    image_out = args.tested_sample_dir or os.path.join(
        args.output, f"tested_sample_{re.sub(r'[^A-Za-z0-9._-]+', '_', args.dataset).strip('_')}"
    )
    os.makedirs(image_out, exist_ok=True)

    rows = []
    files = sorted(f for f in os.listdir(args.hazy_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    if args.save_prediction_samples is not None and args.save_prediction_samples < 0:
        raise SystemExit("--save-prediction-samples must be non-negative")
    if args.save_prediction_samples is None:
        save_names = set(files)
    else:
        valid_names = [f for f in files if get_gt_path(f, args.gt_dir)]
        rng = np.random.default_rng(args.seed)
        count = min(args.save_prediction_samples, len(valid_names))
        save_names = set(rng.choice(valid_names, size=count, replace=False).tolist()) if count else set()
    saved_names = []
    unmatched_gt, unreadable, shape_mismatch = [], [], []
    for filename in files:
        gt_path = get_gt_path(filename, args.gt_dir)
        if not gt_path:
            unmatched_gt.append(filename)
            continue
        hazy, gt = load_image_pair(
            os.path.join(args.hazy_dir, filename), gt_path,
            size=None if args.native_resolution else args.size,
            preprocessing=args.preprocessing,
        )
        if hazy is None:
            unreadable.append(filename)
            continue
        if hazy.shape != gt.shape:
            print(f"Skip {filename}: hazy/GT shape mismatch {hazy.shape} vs {gt.shape}")
            shape_mismatch.append(filename)
            continue
        pred = predict_full(model, hazy)
        diff = pred.astype(np.float64) - gt.astype(np.float64)
        row = {
            "dataset": args.dataset,
            "image": filename,
            "height": int(gt.shape[0]),
            "width": int(gt.shape[1]),
            "psnr": float(tf.image.psnr(pred[None], gt[None], 1.0).numpy()[0]),
            "ssim": float(tf.image.ssim(pred[None], gt[None], 1.0).numpy()[0]),
            "mae": float(np.mean(np.abs(diff))),
            "mse": float(np.mean(np.square(diff))),
            "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        }
        try:
            row["ms_ssim"] = float(tf.image.ssim_multiscale(pred[None], gt[None], 1.0).numpy()[0])
        except tf.errors.InvalidArgumentError:
            row["ms_ssim"] = float("nan")
        try:
            from skimage.color import deltaE_ciede2000, rgb2lab
            row["deltaE2000"] = float(np.mean(deltaE_ciede2000(
                rgb2lab(np.clip(pred, 0, 1)), rgb2lab(np.clip(gt, 0, 1))
            )))
        except ImportError:
            row["deltaE2000"] = float("nan")
        row.update(_perceptual_scores(metrics, torch, pred, gt))
        rows.append(row)
        if filename in save_names:
            pred8 = np.rint(np.clip(pred, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(image_out, os.path.splitext(filename)[0] + ".png"),
                        cv2.cvtColor(pred8, cv2.COLOR_RGB2BGR))
            saved_names.append(filename)
        print(filename, "PSNR", f"{row['psnr']:.3f}", "SSIM", f"{row['ssim']:.5f}")

    if not rows:
        raise SystemExit("No valid hazy/GT pairs found. Check directory structure and filename matching.")
    detail = os.path.join(args.output, "paired_per_image.csv")
    with open(detail, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = os.path.join(args.output, "paired_summary.csv")
    fields = ["dataset", "metric", "n", "mean", "std", "bootstrap_ci95_low", "bootstrap_ci95_high"]
    with open(summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for metric in [k for k in rows[0] if k not in {"dataset", "image", "height", "width"}]:
            vals = [r[metric] for r in rows if np.isfinite(r.get(metric, np.nan))]
            if vals:
                mean, std, low, high = _bootstrap_ci(vals, samples=args.bootstrap_samples, seed=args.seed)
                writer.writerow({"dataset": args.dataset, "metric": metric, "n": len(vals),
                                 "mean": mean, "std": std,
                                 "bootstrap_ci95_low": low, "bootstrap_ci95_high": high})
    print(f"Paired results: {detail}\nSummary and image-level bootstrap CIs: {summary}")
    if not np.isfinite(rows[0].get("deltaE2000", np.nan)):
        print("Install scikit-image to calculate CIEDE2000 (pip install scikit-image).")
    model_path = Path(args.model).resolve()
    run_manifest = {
        "script": str(Path(__file__).resolve()), "command": [sys.executable, *sys.argv],
        "model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_input_shape": str(model.input_shape), "model_parameters": int(model.count_params()),
        "dataset": args.dataset, "hazy_dir": str(Path(args.hazy_dir).resolve()),
        "gt_dir": str(Path(args.gt_dir).resolve()), "preprocessing": args.preprocessing,
        "resolution": "native" if args.native_resolution else int(args.size),
        "tested_sample_dir": str(Path(image_out).resolve()),
        "saved_tested_predictions": len(saved_names), "saved_prediction_files": saved_names,
        "prediction_sampling": ("all scored pairs" if args.save_prediction_samples is None else
                                f"uniform sample capped at {args.save_prediction_samples}; all matched pairs scored"),
        "perceptual_metrics": list(args.perceptual), "image_files_seen": len(files),
        "scored_pairs": len(rows), "unmatched_ground_truth": unmatched_gt,
        "unreadable_pairs": unreadable, "shape_mismatch": shape_mismatch,
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
    }
    manifest_path = os.path.join(args.output, "paired_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(run_manifest, stream, indent=2)
    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
