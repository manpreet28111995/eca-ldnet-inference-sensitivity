"""Measure batch-1 latency and peak GPU memory for packaged checkpoints."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from eca_ldnet.model import build_eca_ldnet, load_pretrained


VARIANT_FLAGS = {
    "full": {},
    "no_eca": {"use_eca": False},
    "eca_only": {"use_eca": True, "use_pa": False},
    "no_attention": {"use_eca": False, "use_pa": False},
    "no_physics": {"use_physics": False},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", nargs="+", default=["256x256", "512x512", "1280x720", "1920x1080"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--model-ids", nargs="*", default=None)
    args = parser.parse_args()
    if not tf.config.list_physical_devices("GPU"):
        raise SystemExit("GPU required for comparable latency/memory benchmarking.")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    package = args.package_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((package / "package_manifest.json").read_text())
    models = manifest["models"]
    if args.model_ids:
        requested = set(args.model_ids)
        models = [m for m in models if m["model_id"] in requested]
        if requested != {m["model_id"] for m in models}:
            raise SystemExit("Some requested model IDs are missing from package_manifest.json")
    sizes = []
    for value in args.resolutions:
        w, h = map(int, value.lower().split("x"))
        sizes.append((w, h))

    fields = ["model_id", "family", "variant", "seed", "target_width", "target_height",
              "padded_width", "padded_height", "batch_size", "precision", "warmup_runs",
              "measured_runs", "parameters", "device", "gpu_peak_mb", "latency_mean_ms",
              "latency_p50_ms", "latency_p95_ms", "throughput_images_s", "status", "error"]
    rows = []
    for spec in models:
        ckpt = package / spec["checkpoint"]
        fixed = None
        dynamic = None
        call = None
        tensor = None
        try:
            fixed = load_pretrained(str(ckpt), compile=False)
            if spec["variant"] not in VARIANT_FLAGS:
                raise ValueError(f"No dynamic rebuild flags for variant {spec['variant']}")
            observed = "physics_output" in [layer.name for layer in fixed.layers] and len(fixed.get_layer("physics_output").input) == 4
            dynamic = build_eca_ldnet(img_size=None, physics_uses_observed_input=observed,
                                      **VARIANT_FLAGS[spec["variant"]])
            dynamic.set_weights(fixed.get_weights())
            del fixed
            fixed = None
            params = int(dynamic.count_params())
            for width, height in sizes:
                padded_w = ((width + 15) // 16) * 16
                padded_h = ((height + 15) // 16) * 16
                row = {"model_id": spec["model_id"], "family": spec["family"],
                       "variant": spec["variant"], "seed": spec["seed"],
                       "target_width": width, "target_height": height,
                       "padded_width": padded_w, "padded_height": padded_h,
                       "batch_size": 1, "precision": "float32", "warmup_runs": args.warmup,
                       "measured_runs": args.runs, "parameters": params,
                       "device": tf.config.list_physical_devices("GPU")[0].name,
                       "gpu_peak_mb": "", "latency_mean_ms": "", "latency_p50_ms": "",
                       "latency_p95_ms": "", "throughput_images_s": "", "status": "failed", "error": ""}
                try:
                    tensor = tf.zeros((1, padded_h, padded_w, 3), dtype=tf.float32)
                    call = lambda: dynamic(tensor, training=False)[..., :height, :width, :]
                    for _ in range(args.warmup):
                        call().numpy()
                    tf.config.experimental.reset_memory_stats("GPU:0")
                    times = []
                    for _ in range(args.runs):
                        start = time.perf_counter()
                        call().numpy()
                        times.append((time.perf_counter() - start) * 1000.0)
                    peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1024**2
                    mean = statistics.mean(times)
                    row.update({"gpu_peak_mb": peak, "latency_mean_ms": mean,
                                "latency_p50_ms": float(np.percentile(times, 50)),
                                "latency_p95_ms": float(np.percentile(times, 95)),
                                "throughput_images_s": 1000.0 / mean, "status": "ok"})
                except (tf.errors.ResourceExhaustedError, Exception) as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"[:500]
                    row["status"] = "oom" if isinstance(exc, tf.errors.ResourceExhaustedError) else "inference_failed"
                    print("BENCHMARK FAILURE", spec["model_id"], f"{width}x{height}", row["error"])
                rows.append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader(); writer.writerows(rows)
                print(spec["model_id"], f"{width}x{height}", row["status"], row["latency_p50_ms"])
                if row["status"] in {"oom", "inference_failed"}:
                    # Preserve the failure and mark larger sizes as unrun; do not imply those sizes succeeded.
                    for later_w, later_h in sizes[sizes.index((width, height)) + 1:]:
                        later_pw = ((later_w + 15) // 16) * 16
                        later_ph = ((later_h + 15) // 16) * 16
                        rows.append({"model_id": spec["model_id"], "family": spec["family"],
                                     "variant": spec["variant"], "seed": spec["seed"],
                                     "target_width": later_w, "target_height": later_h,
                                     "padded_width": later_pw, "padded_height": later_ph,
                                     "batch_size": 1, "precision": "float32", "warmup_runs": args.warmup,
                                     "measured_runs": 0, "parameters": params,
                                     "device": row["device"], "status": "not_run_after_failure",
                                     "error": "Skipped after a prior size failed; check the preceding benchmark row."})
                    break
        except Exception as exc:
            print("MODEL LOAD FAILURE", spec["model_id"], repr(exc))
            for width, height in sizes:
                rows.append({"model_id": spec["model_id"], "family": spec["family"],
                             "variant": spec["variant"], "seed": spec["seed"],
                             "target_width": width, "target_height": height,
                             "batch_size": 1, "precision": "float32", "warmup_runs": args.warmup,
                             "measured_runs": args.runs, "device": "", "status": "model_load_failed",
                             "error": f"{type(exc).__name__}: {exc}"[:500]})
        finally:
            del fixed, dynamic, call, tensor
            tf.keras.backend.clear_session()
            gc.collect()
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    meta = {"python": platform.python_version(), "tensorflow": tf.__version__,
            "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
            "batch_size": 1, "precision": "float32", "warmup_runs": args.warmup,
            "measured_runs": args.runs, "resolutions_width_height": sizes,
            "interpretation": "Measurements apply only to the recorded Kaggle hardware/software and include no host-side preprocessing.",
            "csv": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print("Saved benchmark CSV and metadata:", args.output)


if __name__ == "__main__":
    main()
