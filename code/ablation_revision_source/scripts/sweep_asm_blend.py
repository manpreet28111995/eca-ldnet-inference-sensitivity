"""Inference-only sensitivity sweep for the saved ECA-LDNet checkpoint.

This evaluates both the checkpoint's legacy raw-prediction correction and
the paper's observed-hazy-input equation using the same saved raw/A/t heads.
It is a screening experiment, not a substitute for retraining each blend.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eca_ldnet.data import get_gt_path, load_image_pair
from eca_ldnet.layers import CUSTOM_OBJECTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hazy-dir", default=None)
    parser.add_argument("--gt-dir", default=None)
    parser.add_argument("--manifest", default=None, help="CSV split manifest with hazy_path, clean_path, split columns")
    parser.add_argument("--split", default="validation", help="Manifest split to evaluate")
    parser.add_argument("--output", default="./blend_sweep")
    parser.add_argument("--preprocessing", choices=["submitted", "radiometric"], default="submitted")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--weights", type=float, nargs="+", default=[0, .02, .05, .08, .10, .20])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[.05, .1, .2])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model = tf.keras.models.load_model(args.model, custom_objects=CUSTOM_OBJECTS, compile=False)
    heads = tf.keras.Model(
        model.input,
        [model.get_layer("raw_sigmoid").output,
         model.get_layer("atm_light").output,
         model.get_layer("t_full").output],
    )
    if args.manifest:
        with open(args.manifest, newline="", encoding="utf-8") as stream:
            pairs = [
                (row["hazy_path"], row["clean_path"])
                for row in csv.DictReader(stream)
                if row.get("split") == args.split
            ]
        if args.limit:
            pairs = pairs[:args.limit]
    else:
        if not args.hazy_dir or not args.gt_dir:
            raise SystemExit("Provide --manifest or both --hazy-dir and --gt-dir")
        files = sorted(
            f for f in os.listdir(args.hazy_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        )
        if args.limit:
            files = files[:args.limit]
        pairs = [
            (os.path.join(args.hazy_dir, filename), get_gt_path(filename, args.gt_dir))
            for filename in files
        ]

    rows = []
    for hazy_path, gt_path in pairs:
        if not gt_path:
            continue
        filename = os.path.basename(hazy_path)
        hazy, gt = load_image_pair(
            hazy_path, gt_path,
            preprocessing=args.preprocessing,
        )
        if hazy is None:
            continue
        raw_b, a_b, t_b = heads.predict(hazy[None], verbose=0)
        # Keep all image tensors rank-3 before adding the metric batch axis.
        # A (1, 1, 1, 3) atmospheric-light tensor broadcasts an extra batch
        # dimension into j_aux and makes tf.image.psnr return a non-scalar.
        raw, a, t = raw_b[0], a_b[0].reshape(1, 1, 3), t_b[0]
        for source, observed in (("legacy_raw", raw), ("observed_hazy", hazy)):
            for eps in args.epsilons:
                j_aux = np.clip((observed - a) / (t + eps) + a, 0, 1)
                for weight in args.weights:
                    pred = np.clip(raw * (1 - weight) + j_aux * weight, 0, 1).astype(np.float32)
                    rows.append({
                        "image": filename,
                        "source": source,
                        "blend": weight,
                        "epsilon": eps,
                        "psnr": float(np.asarray(tf.image.psnr(pred[None], gt[None], 1.0).numpy()).reshape(-1)[0]),
                        "ssim": float(np.asarray(tf.image.ssim(pred[None], gt[None], 1.0).numpy()).reshape(-1)[0]),
                    })
        print(f"Evaluated {filename}")

    if not rows:
        raise SystemExit("No valid hazy/clean image pairs found. Check --hazy-dir and --gt-dir.")
    detail_path = os.path.join(args.output, "blend_sweep_per_image.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    keys = sorted({(r["source"], r["blend"], r["epsilon"]) for r in rows})
    summary_path = os.path.join(args.output, "blend_sweep_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as stream:
        fields = ["source", "blend", "epsilon", "n", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source, blend, eps in keys:
            selected = [r for r in rows if (r["source"], r["blend"], r["epsilon"]) == (source, blend, eps)]
            psnr = np.asarray([r["psnr"] for r in selected])
            ssim = np.asarray([r["ssim"] for r in selected])
            writer.writerow({
                "source": source, "blend": blend, "epsilon": eps, "n": len(selected),
                "psnr_mean": psnr.mean(), "psnr_std": psnr.std(ddof=1) if len(psnr) > 1 else 0.0,
                "ssim_mean": ssim.mean(), "ssim_std": ssim.std(ddof=1) if len(ssim) > 1 else 0.0,
            })
    print(f"Saved per-image results: {detail_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
