#!/usr/bin/env python3
"""Build a compact, auditable handoff ZIP for paper analysis.

The bundle contains source snapshots, curated full-model training records and
checkpoints for seeds 17/43/73, plus the organized reviewer evaluation tree.
Raw datasets, virtual environments, caches, and intermediate ZIPs are excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ARCHIVE = ROOT / "kaggle_revision" / "eca_ldnet_full_model_training_final.zip"
ABLATION_ARCHIVE = ROOT / "kaggle_revision" / "eca_ld_net_revision_ablations_.zip"
REVIEWER_CODE_ARCHIVE = ROOT / "kaggle_revision" / "eca_ldnet_reviewer_eval_code.zip"
REVIEWER_PACKAGE_ARCHIVE = ROOT / "kaggle_revision" / "eca_ldnet_reviewer_test_package.zip"
TRAIN_RESULTS = ROOT / "full_model_training"
EVAL_RESULTS = ROOT / "evaluation_all_models_revisions" / "reviewer_evaluation"
OUT_DIR = ROOT / "paper_handoff"
OUT_ZIP = OUT_DIR / "ECA_LDNet_Reviewer_Paper_Handoff_2026-10-13.zip"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def refresh_evaluation_indexes() -> dict:
    """Reconcile the top-level evaluation indexes from per-run evidence."""
    paired = EVAL_RESULTS / "paired"
    rows: list[dict] = []
    completed: set[str] = set()
    model_names: set[str] = set()
    datasets: dict[str, dict] = {}
    for model_dir in sorted(paired.iterdir()):
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name
        model_names.add(model_id)
        bits = model_id.rsplit("_seed", 1)
        if len(bits) != 2 or not bits[1].isdigit():
            continue
        variant, seed = bits[0], int(bits[1])
        family = "full" if variant == "full" else "ablation"
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            run_path = dataset_dir / "paired_run_manifest.json"
            summary_path = dataset_dir / "paired_summary.csv"
            image_csv = dataset_dir / "paired_per_image.csv"
            if not (run_path.is_file() and summary_path.is_file() and image_csv.is_file()):
                continue
            run = read_json(run_path)
            completed.add(f"paired::{model_id}::{dataset}")
            datasets.setdefault(dataset, {
                "name": dataset,
                "hazy_dir": run.get("hazy_dir"),
                "gt_dir": run.get("gt_dir"),
                "native_resolution": "--native-resolution" in run.get("command", []),
                "size": run.get("resolution"),
            })
            with summary_path.open("r", encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    rows.append({
                        "family": family,
                        "variant": variant,
                        "model_id": model_id,
                        "seed": seed,
                        "dataset": row["dataset"],
                        "metric": row["metric"],
                        "n": int(row["n"]),
                        "mean": float(row["mean"]),
                        "std": float(row["std"]),
                        "bootstrap_ci95_low": float(row["bootstrap_ci95_low"]),
                        "bootstrap_ci95_high": float(row["bootstrap_ci95_high"]),
                    })

    if not rows:
        raise RuntimeError(f"No completed paired evaluation runs found under {paired}")

    columns = ["family", "variant", "model_id", "seed", "dataset", "metric", "n", "mean", "std", "bootstrap_ci95_low", "bootstrap_ci95_high"]
    with (EVAL_RESULTS / "per_model_dataset_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    aggregate: list[dict] = []
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["family"], row["variant"], row["dataset"], row["metric"])
        groups.setdefault(key, []).append(row)
    for key, group in sorted(groups.items()):
        seed_means = [r["mean"] for r in group]
        aggregate.append({
            "family": key[0],
            "variant": key[1],
            "dataset": key[2],
            "metric": key[3],
            "seed_count": len(seed_means),
            "mean_of_seed_means": statistics.fmean(seed_means),
            "sample_sd_across_seed_means": statistics.stdev(seed_means) if len(seed_means) > 1 else "",
        })
    aggregate_columns = ["family", "variant", "dataset", "metric", "seed_count", "mean_of_seed_means", "sample_sd_across_seed_means"]
    with (EVAL_RESULTS / "across_seed_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_columns)
        writer.writeheader()
        writer.writerows(aggregate)

    manifest_path = EVAL_RESULTS / "evaluation_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    old_datasets = {d["name"]: d for d in manifest.get("paired_datasets", [])}
    old_datasets.update(datasets)
    manifest["models_run"] = sorted(model_names)
    manifest["paired_datasets"] = [old_datasets[name] for name in sorted(old_datasets)]
    manifest["completed_tasks"] = sorted(set(manifest.get("completed_tasks", [])) | completed)
    manifest["unpaired_datasets"] = manifest.get("unpaired_datasets", [])
    write_json(manifest_path, manifest)

    progress_path = EVAL_RESULTS / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {}
    progress["complete"] = sorted(set(progress.get("complete", [])) | completed)
    progress["updated_unix"] = int(time.time())
    write_json(progress_path, progress)
    return {"models": len(model_names), "paired_runs": len(completed), "summary_rows": len(rows), "datasets": sorted(datasets)}


def compressed_suffix(path: str) -> bool:
    return Path(path).suffix.lower() in {".keras", ".png", ".jpg", ".jpeg", ".webp", ".zip", ".pdf"}


def add_file(bundle: zipfile.ZipFile, path: Path, arcname: str, records: list[dict]) -> None:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as src, bundle.open(arcname, "w", force_zip64=True) as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            dst.write(chunk)
    # ZipFile.open(w) uses the archive default compression; rebuild non-text
    # files through a per-entry writer in a second pass only where needed.
    records.append({"path": arcname, "bytes": size, "sha256": digest.hexdigest()})


def add_tree(bundle: zipfile.ZipFile, source: Path, prefix: str, records: list[dict]) -> int:
    count = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        arcname = (Path(prefix) / path.relative_to(source)).as_posix()
        add_file(bundle, path, arcname, records)
        count += 1
    return count


def add_code_sources(bundle: zipfile.ZipFile, records: list[dict]) -> int:
    count = 0
    source_files = [
        "README.md", "README_REVISION_HANDOFF.md", "REVIEWER_EVALUATION_RUNBOOK.md",
        "REVIEW_RESPONSE_TRACKER.md", "REVISION_PLAN.md", "LICENSE", "pyproject.toml",
        "requirements.txt", "ablation_results.csv", "ablations_kaggle.md", "ablation_laptop_to_run.md",
    ]
    for rel in source_files:
        path = ROOT / rel
        if path.is_file():
            add_file(bundle, path, f"code/full_training_source/{rel}", records)
            count += 1
    for rel in ["eca_ldnet", "scripts", "tests", "docs", "kaggle_how_to_run", "laptop_how_to_run", "seed17_how_to_run", "seed73_how_to_run", "seed_43_how-to_run"]:
        source = ROOT / rel
        if source.is_dir():
            count += add_tree(bundle, source, f"code/full_training_source/{rel}", records)
    for rel in ["notebooks/revision_kaggle_train.ipynb", "notebooks/reviewer_kaggle_test.ipynb"]:
        path = ROOT / rel
        if path.is_file():
            add_file(bundle, path, f"code/full_training_source/{rel}", records)
            count += 1

    with zipfile.ZipFile(ABLATION_ARCHIVE, "r") as ablations:
        for info in ablations.infolist():
            if info.is_dir():
                continue
            arcname = f"code/ablation_revision_source/{info.filename}"
            digest = hashlib.sha256()
            size = 0
            with ablations.open(info, "r") as src, bundle.open(arcname, "w", force_zip64=True) as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                    dst.write(chunk)
            records.append({"path": arcname, "bytes": size, "sha256": digest.hexdigest()})
            count += 1
    return count


def add_zip_snapshot(bundle: zipfile.ZipFile, archive: Path, prefix: str, records: list[dict]) -> int:
    count = 0
    with zipfile.ZipFile(archive, "r") as source_zip:
        names = [info.filename for info in source_zip.infolist() if not info.is_dir()]
        top_levels = {name.split("/", 1)[0] for name in names if "/" in name}
        wrapper = (next(iter(top_levels)) + "/") if len(top_levels) == 1 else ""
        strip_wrapper = bool(wrapper) and all(name.startswith(wrapper) for name in names)
        for info in source_zip.infolist():
            if info.is_dir():
                continue
            source_name = info.filename[len(wrapper):] if strip_wrapper else info.filename
            arcname = f"{prefix}/{source_name}"
            digest = hashlib.sha256()
            size = 0
            with source_zip.open(info, "r") as src, bundle.open(arcname, "w", force_zip64=True) as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                    dst.write(chunk)
            records.append({"path": arcname, "bytes": size, "sha256": digest.hexdigest()})
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_ZIP)
    args = parser.parse_args()
    output: Path = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in [TRAIN_ARCHIVE, ABLATION_ARCHIVE, REVIEWER_CODE_ARCHIVE, REVIEWER_PACKAGE_ARCHIVE, TRAIN_RESULTS, EVAL_RESULTS]:
        if not path.exists():
            raise FileNotFoundError(path)

    # Keep the locally curated training tree tied to the supplied final archive.
    with zipfile.ZipFile(TRAIN_ARCHIVE, "r") as source_zip:
        expected = {i.filename for i in source_zip.infolist() if not i.is_dir()}
    actual = {f"full_model_training/{p.relative_to(TRAIN_RESULTS).as_posix()}" for p in TRAIN_RESULTS.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    if expected != actual:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise RuntimeError(f"Curated training files differ from source ZIP. Missing={missing[:8]} Extra={extra[:8]}")

    evaluation_audit = refresh_evaluation_indexes()
    records: list[dict] = []
    with zipfile.ZipFile(output, "w", allowZip64=True) as bundle:
        bundle.comment = b"ECA-LDNet final paper analysis handoff; see README_FOR_ANALYST.md and PACKAGE_MANIFEST.json."
        code_count = add_code_sources(bundle, records)
        reviewer_code_count = add_zip_snapshot(bundle, REVIEWER_CODE_ARCHIVE, "code/reviewer_evaluator_snapshot", records)
        reviewer_package_count = add_zip_snapshot(bundle, REVIEWER_PACKAGE_ARCHIVE, "reproducibility/reviewer_test_package", records)
        training_count = add_tree(bundle, TRAIN_RESULTS, "results/full_model_training", records)
        evaluation_count = add_tree(bundle, EVAL_RESULTS, "results/reviewer_evaluation", records)

        README = f"""# ECA-LDNet paper analysis handoff

This archive consolidates the code snapshots, training histories/checkpoints, and reviewer evaluation evidence for paper analysis.

## Contents

- `code/full_training_source/`: project training/evaluation source and run guides.
- `code/ablation_revision_source/`: contents of the supplied ablation revision archive.
- `code/reviewer_evaluator_snapshot/`: exact evaluator source snapshot used for reviewer metrics.
- `reproducibility/reviewer_test_package/`: checkpoint package and manifest for all evaluated full and ablation variants.
- `results/full_model_training/seed17`, `seed43`, `seed73`: all four stage histories, stage manifests, shared split manifests, curated model checkpoints, and seed manifests.
- `results/reviewer_evaluation/`: organized per-model/per-dataset files, per-image metrics, prediction examples, summaries, and evaluation manifests.
- `PACKAGE_MANIFEST.json`: SHA-256 and byte size for every bundled file except this manifest and README.

## Full model seed checkpoints

For inference comparisons, use the validation-best stage 4 checkpoint:

- Seed 17: `results/full_model_training/seed17/stage4/models/stage4_best.keras`
- Seed 43: `results/full_model_training/seed43/stage4/models/stage4_best.keras`
- Seed 73: `results/full_model_training/seed73/stage4/models/stage4_best.keras`

Each seed's stage 4 folder also contains its final/latest training checkpoint where present. The full stage histories and retained checkpoint snapshots are included for audit and continuation analysis. The seed manifests specify the exact included filenames and optimizer iterations. All three use the same split hash. Seed 73 stage 3 stopped early at 16 of 20 epochs; this is recorded in its manifests and history.

## Evaluation scope and cautions

The reviewer evaluation includes 15 model variants (the full model and four ablations for seeds 17, 43, and 73) across O-HAZE, SOTS-Indoor, SOTS-Outdoor, I-HAZE, and NH-HAZE where run evidence exists. The local I-HAZE and NH-HAZE test folders each contain five paired images; report those sample counts and do not describe them as complete 35/55-image benchmark evaluations. Their run manifests record 256-pixel inference and saved predictions. O-HAZE/SOTS protocols and pair counts are recorded in their own run manifests.

Use per-image CSVs and per-seed summaries for claims. `across_seed_summary.csv` reports mean and standard deviation across seed means; it is not a pooled per-image confidence interval. No raw datasets are included.

## Integrity and provenance

Training files come from `kaggle_revision/eca_ldnet_full_model_training_final.zip` and are checked against its file list before packaging. Ablation source contents come from `kaggle_revision/eca_ld_net_revision_ablations_.zip`. The evaluation tree is the organized local reviewer result folder at packaging time.
"""
        bundle.writestr("README_FOR_ANALYST.md", README)
        manifest = {
            "package_name": output.name,
            "created_utc_epoch": int(time.time()),
            "sources": {
                "full_training_archive": {"name": TRAIN_ARCHIVE.name, "sha256": file_sha256(TRAIN_ARCHIVE)},
                "ablation_archive": {"name": ABLATION_ARCHIVE.name, "sha256": file_sha256(ABLATION_ARCHIVE)},
                "reviewer_eval_code_archive": {"name": REVIEWER_CODE_ARCHIVE.name, "sha256": file_sha256(REVIEWER_CODE_ARCHIVE)},
                "reviewer_test_package_archive": {"name": REVIEWER_PACKAGE_ARCHIVE.name, "sha256": file_sha256(REVIEWER_PACKAGE_ARCHIVE)},
                "training_collection_manifest_sha256": file_sha256(TRAIN_RESULTS / "collection_manifest.json"),
            },
            "evaluation_audit": evaluation_audit,
            "included_file_counts": {"project_and_ablation_source_files": code_count, "reviewer_evaluator_source_files": reviewer_code_count, "reviewer_checkpoint_package_files": reviewer_package_count, "training_result_files": training_count, "reviewer_evaluation_files": evaluation_count},
            "files": records,
        }
        bundle.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({
        "output": str(output),
        "size_bytes": output.stat().st_size,
        "size_gib": round(output.stat().st_size / (1024 ** 3), 3),
        "included_file_counts": manifest["included_file_counts"],
        "evaluation_audit": evaluation_audit,
        "manifest_entries": len(records),
    }, indent=2))


if __name__ == "__main__":
    main()
