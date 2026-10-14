"""Native-resolution unpaired real-haze evaluation with NR-IQA proxies.

No-reference scores compare image statistics/perceptual priors; they do not
measure recovered ground truth or prove real-world dehazing correctness.
Metrics use pyiqa/PyTorch on CPU to keep TensorFlow inference GPU memory clear.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eca_ldnet.data import preprocess_hazy
from eca_ldnet.evaluate import predict_full
from eca_ldnet.model import load_pretrained


VARIANT_FLAGS = {
    "full": {},
    "no_eca": {"use_eca": False},
    "eca_only": {"use_eca": True, "use_pa": False},
    "no_attention": {"use_eca": False, "use_pa": False},
    "no_physics": {"use_physics": False},
}


def _load_eval_model(path, variant="full"):
    model = load_pretrained(path, compile=False)
    if any(d is not None for d in model.input_shape[1:3]):
        from eca_ldnet.model import build_eca_ldnet
        flags = VARIANT_FLAGS[variant]
        has_physics = "physics_output" in {layer.name for layer in model.layers}
        observed_input = len(model.get_layer("physics_output").input) == 4 if has_physics else True
        dynamic = build_eca_ldnet(
            img_size=None, physics_uses_observed_input=observed_input, **flags
        )
        dynamic.set_weights(model.get_weights())
        model = dynamic
    return model


def main():
    p = argparse.ArgumentParser(description="Unpaired real-haze evaluation")
    p.add_argument("--model", required=True)
    p.add_argument("--variant", choices=sorted(VARIANT_FLAGS), default="full")
    p.add_argument("--hazy-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--preprocessing", choices=["radiometric", "submitted"], default="radiometric")
    p.add_argument("--metrics", nargs="+", default=["niqe", "brisque", "musiq", "maniqa", "clipiqa"])
    p.add_argument("--tested-sample-dir", help="Directory for all readable dehazed test inputs")
    p.add_argument("--fade", action="store_true",
                   help="also compute FADE for hazy/dehazed images (requires pip install fade-python)")
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model = _load_eval_model(args.model, args.variant)
    try:
        import pyiqa
        import torch
    except ImportError as exc:
        raise SystemExit("Install optional pyiqa and torch in the notebook before NR-IQA: pip install pyiqa") from exc
    metrics = {name: pyiqa.create_metric(name, device="cpu") for name in args.metrics}
    fade_metric = None
    if args.fade:
        try:
            from pyfade import fade as fade_metric
        except ImportError as exc:
            raise SystemExit("Install FADE support with: pip install fade-python") from exc

    os.makedirs(args.output, exist_ok=True)
    safe_dataset = re.sub(r"[^A-Za-z0-9._-]+", "_", args.dataset).strip("_")
    out_dir = args.tested_sample_dir or os.path.join(args.output, f"tested_sample_{safe_dataset}")
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
        if fade_metric is not None:
            row["fade_hazy"] = float(fade_metric(hazy))
            row["fade_dehazed"] = float(fade_metric(pred))
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
        metric_names = list(metrics) + (["fade"] if fade_metric is not None else [])
        for name in metric_names:
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
    model_path = Path(args.model).resolve()
    manifest = {
        "script": str(Path(__file__).resolve()), "command": [sys.executable, *sys.argv],
        "model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_input_shape": str(model.input_shape), "model_parameters": int(model.count_params()),
        "dataset": args.dataset, "hazy_dir": str(Path(args.hazy_dir).resolve()),
        "preprocessing": args.preprocessing, "resolution": "native",
        "no_reference_metrics": list(args.metrics) + (["fade"] if fade_metric is not None else []),
        "image_files_seen": len(files), "scored_images": len(rows),
        "tested_sample_dir": str(Path(out_dir).resolve()),
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
        "interpretation": "NR-IQA/FADE are proxies; they do not establish reference fidelity.",
    }
    manifest_path = os.path.join(args.output, "unpaired_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
