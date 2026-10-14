"""Paired dehazing evaluation with per-image scores and bootstrap intervals.

LPIPS/DISTS are optional and use pyiqa/PyTorch on CPU to avoid competing with
TensorFlow for GPU memory. PSNR, SSIM, and MS-SSIM are computed at the declared
evaluation resolution. Native mode requires a dynamic-spatial model.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

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


def main():
    p = argparse.ArgumentParser(description="Paired full-reference evaluation")
    p.add_argument("--model", required=True)
    p.add_argument("--hazy-dir", required=True)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--dataset", required=True, help="Dataset label recorded in outputs")
    p.add_argument("--output", required=True)
    p.add_argument("--preprocessing", choices=["radiometric", "submitted"], default="radiometric")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--native-resolution", action="store_true")
    p.add_argument("--perceptual", nargs="*", choices=["lpips", "dists"], default=[])
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model = load_pretrained(args.model)
    fixed_hw = model.input_shape[1:3]
    if args.native_resolution and any(d is not None for d in fixed_hw):
        raise SystemExit(
            f"Native evaluation needs a dynamic model; checkpoint input is {model.input_shape}. "
            "Use --size 256 for this submitted fixed-resolution checkpoint."
        )
    metrics, torch = _make_pyiqa_metrics(args.perceptual)
    os.makedirs(args.output, exist_ok=True)
    image_out = os.path.join(args.output, "predictions")
    os.makedirs(image_out, exist_ok=True)

    rows = []
    files = sorted(f for f in os.listdir(args.hazy_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    for filename in files:
        gt_path = get_gt_path(filename, args.gt_dir)
        if not gt_path:
            continue
        hazy, gt = load_image_pair(
            os.path.join(args.hazy_dir, filename), gt_path,
            size=None if args.native_resolution else args.size,
            preprocessing=args.preprocessing,
        )
        if hazy is None:
            continue
        if hazy.shape != gt.shape:
            print(f"Skip {filename}: hazy/GT shape mismatch {hazy.shape} vs {gt.shape}")
            continue
        pred = predict_full(model, hazy)
        row = {
            "dataset": args.dataset,
            "image": filename,
            "height": int(gt.shape[0]),
            "width": int(gt.shape[1]),
            "psnr": float(tf.image.psnr(pred[None], gt[None], 1.0).numpy()[0]),
            "ssim": float(tf.image.ssim(pred[None], gt[None], 1.0).numpy()[0]),
        }
        try:
            row["ms_ssim"] = float(tf.image.ssim_multiscale(pred[None], gt[None], 1.0).numpy()[0])
        except tf.errors.InvalidArgumentError:
            row["ms_ssim"] = float("nan")
        row.update(_perceptual_scores(metrics, torch, pred, gt))
        rows.append(row)
        pred8 = np.rint(np.clip(pred, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(image_out, os.path.splitext(filename)[0] + ".png"),
                    cv2.cvtColor(pred8, cv2.COLOR_RGB2BGR))
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


if __name__ == "__main__":
    main()
