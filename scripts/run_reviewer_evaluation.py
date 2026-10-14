"""Run the frozen full/ablation checkpoints on user-attached Kaggle datasets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def reset_test_images(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"Refusing to clear predictions outside the run output: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True,
                        help="JSON with paired and unpaired dataset directory mappings")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/reviewer_evaluation"))
    parser.add_argument("--model-ids", nargs="*", default=None)
    parser.add_argument("--dataset-names", nargs="*", default=None)
    parser.add_argument("--perceptual", nargs="*", choices=["lpips", "dists"], default=[])
    parser.add_argument("--nr-metrics", nargs="*", default=["niqe", "brisque", "musiq", "maniqa", "clipiqa"])
    parser.add_argument("--fade", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip tasks with complete manifests")
    args = parser.parse_args()

    package = args.package_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code = package / "code"
    manifest_path = package / "package_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Package manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest["models"]
    if args.model_ids:
        selected = set(args.model_ids)
        models = [m for m in models if m["model_id"] in selected]
        missing = selected - {m["model_id"] for m in models}
        if missing:
            raise SystemExit(f"Unknown model IDs: {sorted(missing)}")
    config = json.loads(args.dataset_config.read_text(encoding="utf-8"))
    paired = [d for d in config.get("paired", []) if d.get("enabled", True)]
    unpaired = [d for d in config.get("unpaired", []) if d.get("enabled", True)]
    if args.dataset_names:
        requested = set(args.dataset_names)
        paired = [d for d in paired if d["name"] in requested]
        unpaired = [d for d in unpaired if d["name"] in requested]
        found = {d["name"] for d in paired + unpaired}
        if requested - found:
            raise SystemExit(f"Dataset names are not enabled/present in config: {sorted(requested-found)}")
    if not paired and not unpaired:
        raise SystemExit("No datasets are enabled in the dataset config.")

    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text()) if args.resume and progress_path.exists() else {"complete": []}
    done = set(progress.get("complete", []))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code)
    paired_script = code / "scripts" / "evaluate_paired.py"
    unpaired_script = code / "scripts" / "evaluate_unpaired.py"
    for dataset in paired:
        hazy, gt = Path(dataset["hazy_dir"]), Path(dataset["gt_dir"])
        if not hazy.is_dir() or not gt.is_dir():
            raise SystemExit(f"Dataset {dataset['name']} paths do not exist: {hazy} | {gt}")
        files = images(hazy)
        # Use the package matcher, so this preflight and each evaluator count the same pairs.
        sys.path.insert(0, str(code))
        from eca_ldnet.data import get_gt_path
        pairs = [(f, get_gt_path(f.name, str(gt))) for f in files]
        matched = [(f, target) for f, target in pairs if target]
        print(f"PAIR PREFLIGHT {dataset['name']}: hazy={len(files)} matched={len(matched)} unmatched={len(files)-len(matched)}")
        if not matched:
            raise SystemExit(f"No hazy/GT pairs matched for {dataset['name']}; fix paths/names first.")
        if dataset.get("expected_pairs") and len(matched) != int(dataset["expected_pairs"]):
            print(f"WARNING: expected_pairs={dataset['expected_pairs']}, found={len(matched)}; record the actual split.")

    for model in models:
        checkpoint = package / model["checkpoint"]
        if not checkpoint.is_file() or sha256(checkpoint) != model["checkpoint_sha256"]:
            raise SystemExit(f"Checkpoint missing or SHA-256 mismatch: {checkpoint}")
        for dataset in paired:
            key = f"paired::{model['model_id']}::{dataset['name']}"
            dest = output / "paired" / model["model_id"] / slug(dataset["name"])
            if args.resume and key in done and (dest / "paired_run_manifest.json").is_file():
                print("SKIP complete", key)
                continue
            tested_samples = dest / f"tested_sample_{slug(dataset['name'])}"
            reset_test_images(tested_samples, dest)
            command = [sys.executable, str(paired_script), "--model", str(checkpoint),
                       "--variant", model["variant"],
                       "--hazy-dir", dataset["hazy_dir"], "--gt-dir", dataset["gt_dir"],
                       "--dataset", dataset["name"], "--output", str(dest),
                       "--preprocessing", dataset.get("preprocessing", "radiometric"),
                       "--bootstrap-samples", str(args.bootstrap_samples), "--seed", str(args.seed),
                       "--tested-sample-dir", str(tested_samples)]
            if dataset.get("native_resolution", False):
                command.append("--native-resolution")
            else:
                command += ["--size", str(int(dataset.get("size", 256)))]
            if args.perceptual:
                command += ["--perceptual", *args.perceptual]
            print("RUN", key)
            subprocess.run(command, cwd=code, env=env, check=True)
            done.add(key)
            progress_path.write_text(json.dumps({"complete": sorted(done), "updated_unix": time.time()}, indent=2))

        for dataset in unpaired:
            key = f"unpaired::{model['model_id']}::{dataset['name']}"
            dest = output / "unpaired" / model["model_id"] / slug(dataset["name"])
            if args.resume and key in done and (dest / "unpaired_run_manifest.json").is_file():
                print("SKIP complete", key)
                continue
            hazy = Path(dataset["hazy_dir"])
            if not hazy.is_dir():
                raise SystemExit(f"Dataset {dataset['name']} hazy path does not exist: {hazy}")
            reset_test_images(dest / f"tested_sample_{slug(dataset['name'])}", dest)
            command = [sys.executable, str(unpaired_script), "--model", str(checkpoint),
                       "--variant", model["variant"],
                       "--hazy-dir", str(hazy), "--dataset", dataset["name"],
                       "--output", str(dest), "--preprocessing", dataset.get("preprocessing", "radiometric"),
                       "--metrics", *args.nr_metrics, "--bootstrap-samples", str(args.bootstrap_samples),
                       "--seed", str(args.seed)]
            if args.fade:
                command.append("--fade")
            print("RUN", key)
            subprocess.run(command, cwd=code, env=env, check=True)
            done.add(key)
            progress_path.write_text(json.dumps({"complete": sorted(done), "updated_unix": time.time()}, indent=2))

    # One row per model seed and dataset; aggregate seed means without pooling image observations.
    detail = []
    for model in models:
        for dataset in paired:
            summary_path = output / "paired" / model["model_id"] / slug(dataset["name"]) / "paired_summary.csv"
            if not summary_path.is_file():
                continue
            for row in read_csv_rows(summary_path):
                detail.append({"family": model["family"], "variant": model["variant"],
                               "model_id": model["model_id"], "seed": model["seed"],
                               "dataset": dataset["name"], **row})
    if detail:
        detail_path = output / "per_model_dataset_summary.csv"
        with detail_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(detail[0]))
            writer.writeheader(); writer.writerows(detail)
        groups = {}
        for row in detail:
            key = (row["family"], row["variant"], row["dataset"], row["metric"])
            groups.setdefault(key, []).append(float(row["mean"]))
        aggregate = []
        import statistics
        for (family, variant, dataset_name, metric), values in sorted(groups.items()):
            aggregate.append({"family": family, "variant": variant, "dataset": dataset_name,
                              "metric": metric, "seed_count": len(values),
                              "mean_of_seed_means": statistics.mean(values),
                              "sample_sd_across_seed_means": statistics.stdev(values) if len(values) > 1 else ""})
        with (output / "across_seed_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
            writer.writeheader(); writer.writerows(aggregate)

    overall = {"package_manifest_sha256": sha256(manifest_path), "models_run": [m["model_id"] for m in models],
               "paired_datasets": [{"name": d["name"], "hazy_dir": d["hazy_dir"], "gt_dir": d["gt_dir"],
                                    "native_resolution": d.get("native_resolution", False), "size": d.get("size", 256)} for d in paired],
               "unpaired_datasets": [{"name": d["name"], "hazy_dir": d["hazy_dir"]} for d in unpaired],
               "preprocessing": "dataset-config per entry; default radiometric", "perceptual_metrics": args.perceptual,
               "nr_metrics": args.nr_metrics, "fade": args.fade, "completed_tasks": sorted(done),
               "interpretation": "Use per-image metrics and per-seed summaries; NR-IQA values are proxies, not ground-truth fidelity."}
    (output / "evaluation_manifest.json").write_text(json.dumps(overall, indent=2))
    archive_base = output.parent / "reviewer_evaluation_results"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output))
    print(f"Evaluation complete. Results ZIP: {archive_path}")
    print(f"All run outputs and tested_sample_<dataset> images: {output}")


if __name__ == "__main__":
    main()
