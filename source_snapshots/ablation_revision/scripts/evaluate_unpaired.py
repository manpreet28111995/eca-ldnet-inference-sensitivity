"""Native-resolution unpaired real-haze evaluation with NR-IQA proxies.

No-reference scores compare image statistics/perceptual priors; they do not
measure recovered ground truth or prove real-world dehazing correctness.
Metrics use pyiqa/PyTorch on CPU to keep TensorFlow inference GPU memory clear.
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

from eca_ldnet.data import preprocess_hazy
from eca_ldnet.evaluate import predict_full
from eca_ldnet.model import load_pretrained


def main():
    p = argparse.ArgumentParser(description="Unpaired real-haze evaluation")
    p.add_argument("--model", required=True)
    p.add_argument("--hazy-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--preprocessing", choices=["radiometric", "submitted"], default="radiometric")
    p.add_argument("--metrics", nargs="+", default=["niqe", "brisque", "musiq", "maniqa"])
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model = load_pretrained(args.model)
    if any(d is not None for d in model.input_shape[1:3]):
        raise SystemExit("Native-resolution evaluation requires a dynamic-spatial checkpoint.")
    try:
        import pyiqa
        import torch
    except ImportError as exc:
        raise SystemExit("Install optional pyiqa and torch in the notebook before NR-IQA: pip install pyiqa") from exc
    metrics = {name: pyiqa.create_metric(name, device="cpu") for name in args.metrics}

    os.makedirs(args.output, exist_ok=True)
    out_dir = os.path.join(args.output, "dehazed_native")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    files = sorted(f for f in os.listdir(args.hazy_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    for filename in files:
        path = os.path.join(args.hazy_dir, filename)
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hazy = preprocess_hazy(rgb, protocol=args.preprocessing)
        pred = predict_full(model, hazy)
        out_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".png")
        out8 = np.rint(np.clip(pred, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(out_path, cv2.cvtColor(out8, cv2.COLOR_RGB2BGR))

        hazy_t = torch.from_numpy(np.ascontiguousarray(hazy.transpose(2, 0, 1)[None])).float()
        pred_t = torch.from_numpy(np.ascontiguousarray(pred.transpose(2, 0, 1)[None])).float()
        row = {"dataset": args.dataset, "image": filename,
               "height": int(rgb.shape[0]), "width": int(rgb.shape[1])}
        with torch.inference_mode():
            for name, metric in metrics.items():
                row[f"{name}_hazy"] = float(metric(hazy_t).detach().cpu().reshape(-1).mean())
                row[f"{name}_dehazed"] = float(metric(pred_t).detach().cpu().reshape(-1).mean())
        rows.append(row)
        print(filename, "->", {k: round(v, 4) for k, v in row.items() if k.endswith("_dehazed")})

    if not rows:
        raise SystemExit("No readable images found in --hazy-dir")
    detail_path = os.path.join(args.output, "unpaired_per_image.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    # Bootstrap intervals are descriptive within this dataset; do not pool
    # different datasets or interpret NR-IQA as reference fidelity.
    rng = np.random.default_rng(args.seed)
    summary_path = os.path.join(args.output, "unpaired_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "metric", "n", "hazy_mean", "dehazed_mean", "delta_mean",
                  "delta_ci95_low", "delta_ci95_high"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for name in metrics:
            h = np.asarray([r[f"{name}_hazy"] for r in rows], dtype=np.float64)
            d = np.asarray([r[f"{name}_dehazed"] for r in rows], dtype=np.float64)
            delta = d - h
            if len(delta) > 1:
                ids = rng.integers(0, len(delta), size=(args.bootstrap_samples, len(delta)))
                low, high = np.quantile(delta[ids].mean(axis=1), [0.025, 0.975])
            else:
                low = high = float("nan")
            writer.writerow({"dataset": args.dataset, "metric": name, "n": len(rows),
                             "hazy_mean": h.mean(), "dehazed_mean": d.mean(),
                             "delta_mean": delta.mean(), "delta_ci95_low": low,
                             "delta_ci95_high": high})
    print(f"Unpaired results: {detail_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
