"""Build a compact, self-contained Kaggle package for reviewer evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle_revision" / "eca_ldnet_reviewer_test_package.zip"
SEEDS = (17, 43, 73)
VARIANTS = {
    "eca_only": "eca_only",
    "no_attention": "no_attention",
    "no_eca": "no_eca",
    "no_physics": "no_physics",
}
SOURCE_DIRS = {"seed73": {"no_attention": "no_attension"}}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_bytes(archive: zipfile.ZipFile, name: str, data: bytes, *, stored=False) -> None:
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def write_file(archive: zipfile.ZipFile, name: str, path: Path, *, stored=False) -> str:
    digest = hashlib.sha256()
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with path.open("rb") as src, archive.open(info, "w") as dst:
        while block := src.read(1024 * 1024):
            digest.update(block)
            dst.write(block)
    return digest.hexdigest()


def source_ablation_zip(seed: int, variant: str) -> Path:
    folder = SOURCE_DIRS.get(f"seed{seed}", {}).get(variant, variant)
    path = ROOT / "new_results_springer_revision" / "ablations" / f"seed{seed}" / folder / f"seed{seed}_{variant}_stage4.zip"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def find_ablation_entries(zf: zipfile.ZipFile, variant: str, seed: int):
    entries = zf.namelist()
    run_config = [n for n in entries if n.endswith("/run_config.json")]
    split = [n for n in entries if n.endswith("/split_manifest.csv")]
    best = [n for n in entries if n.lower().endswith("stage4_best.keras")]
    if len(run_config) != 1 or len(split) != 1 or len(best) != 1:
        raise ValueError(f"Expected exactly one config/split/Stage-4-best in {zf.filename}: "
                         f"config={len(run_config)} split={len(split)} best={len(best)}")
    cfg = json.loads(zf.read(run_config[0]))
    identity = cfg.get("identity", {})
    if int(identity.get("seed", -1)) != seed:
        raise ValueError(f"Seed mismatch in {zf.filename}: {identity}")
    cfg_variant = str(identity.get("variant", "")).lower().replace(" ", "_")
    if cfg_variant and cfg_variant != variant:
        raise ValueError(f"Variant mismatch in {zf.filename}: {identity}")
    histories = {}
    for n in entries:
        match = re.search(r"stage([1-4])_history\.csv$", n, re.IGNORECASE)
        if match:
            stage = int(match.group(1))
            if stage in histories:
                raise ValueError(f"Duplicate stage {stage} history in {zf.filename}")
            histories[stage] = n
    if set(histories) != {1, 2, 3, 4}:
        raise ValueError(f"Missing ablation stage history in {zf.filename}: {histories}")
    return run_config[0], split[0], best[0], histories


def verify_keras(data: bytes, label: str) -> dict:
    import io
    with zipfile.ZipFile(io.BytesIO(data)) as keras:
        names = set(keras.namelist())
        required = {"config.json", "metadata.json", "model.weights.h5"}
        missing = required - names
        if missing:
            raise ValueError(f"{label} is not a complete Keras v3 archive; missing {sorted(missing)}")
        bad = keras.testzip()
        if bad:
            raise ValueError(f"Corrupt Keras member {bad} in {label}")
        return {"entries": len(names), "keras_format": json.loads(keras.read("metadata.json")).get("keras_version")}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUT.parent).free
    if free < 500 * 1024**2:
        raise SystemExit(f"Need at least 500 MiB free to build package; only {free/1024**2:.0f} MiB free")
    fd, tmp_name = tempfile.mkstemp(prefix=OUT.stem + ".", suffix=".partial", dir=OUT.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    manifest = {"package": OUT.name, "schema_version": 1, "models": [], "files": {},
                "deduplicated_split_manifests": {}, "omitted": {
                    "all_epoch_checkpoints": "Final inference package; histories and validation-best weights are retained.",
                    "dataset_images": "Attach public datasets separately in Kaggle.",
                }}
    try:
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as out:
            # Current source tree is authoritative; include only inference code and direct dependencies.
            code_files = sorted((ROOT / "eca_ldnet").glob("*.py"))
            code_files += [ROOT / "scripts" / n for n in (
                "evaluate_paired.py", "evaluate_unpaired.py", "plot_stage_histories.py",
                "count_macs.py", "run_reviewer_evaluation.py", "benchmark_reviewer_models.py",
                "sweep_asm_blend.py", "match_model_capacity.py", "run_ablation.py", "run_ablation_v2.py",
            ) if (ROOT / "scripts" / n).is_file()]
            for path in code_files:
                name = "reviewer_test_package/code/" + path.relative_to(ROOT).as_posix()
                manifest["files"][name] = write_file(out, name, path)
            for rel in ("requirements.txt", "LICENSE"):
                path = ROOT / rel
                if path.is_file():
                    name = f"reviewer_test_package/{rel}"
                    manifest["files"][name] = write_file(out, name, path)
            for rel in ("notebooks/reviewer_kaggle_test.ipynb", "kaggle_revision/REVIEWER_TEST_PACKAGE_README.md",
                        "kaggle_revision/REVIEWER_TEST_COVERAGE.md"):
                path = ROOT / rel
                name = "reviewer_test_package/" + Path(rel).name
                manifest["files"][name] = write_file(out, name, path)

            # Full-model validation-best checkpoints and their stage histories/manifests.
            split_digests = {}
            for seed in SEEDS:
                base = ROOT / "full_model_training" / f"seed{seed}"
                model_path = base / "stage4" / "models" / "stage4_best.keras"
                if not model_path.is_file():
                    raise FileNotFoundError(model_path)
                model_name = f"reviewer_test_package/models/full/seed{seed}/stage4_best.keras"
                digest = write_file(out, model_name, model_path, stored=True)
                model_data = model_path.read_bytes()
                keras_info = verify_keras(model_data, str(model_path))
                manifest["files"][model_name] = digest
                split_bytes = (base / "split_manifest.csv").read_bytes()
                split_hash = sha256_bytes(split_bytes)
                if split_hash not in split_digests:
                    split_name = f"reviewer_test_package/metadata/splits/{split_hash}.csv"
                    write_bytes(out, split_name, split_bytes)
                    split_digests[split_hash] = split_name
                    manifest["files"][split_name] = split_hash
                seed_manifest = base / "seed_manifest.json"
                sname = f"reviewer_test_package/metadata/full/seed{seed}/seed_manifest.json"
                manifest["files"][sname] = write_file(out, sname, seed_manifest)
                histories = {}
                for stage in range(1, 5):
                    stage_dir = base / f"stage{stage}"
                    for filename in ("history.csv", "stage_manifest.json"):
                        p = stage_dir / filename
                        name = f"reviewer_test_package/metadata/full/seed{seed}/stage{stage}/{filename}"
                        manifest["files"][name] = write_file(out, name, p)
                        if filename == "history.csv":
                            with p.open(newline="", encoding="utf-8-sig") as f:
                                histories[str(stage)] = sum(1 for _ in csv.DictReader(f))
                manifest["models"].append({"model_id": f"full_seed{seed}", "family": "full", "variant": "full",
                    "seed": seed, "checkpoint": model_name.removeprefix("reviewer_test_package/"), "checkpoint_sha256": digest,
                    "split_sha256": split_hash, "split_manifest": split_digests[split_hash].removeprefix("reviewer_test_package/"),
                    "source_checkpoint": str(model_path.relative_to(ROOT)), "stage_epochs": histories,
                    "checkpoint_format": keras_info})

            # Each final ablation archive redundantly contains all earlier snapshots.
            # Keep only its Stage 4 best model plus four histories, config, and split.
            for seed in SEEDS:
                for variant in VARIANTS:
                    src_zip = source_ablation_zip(seed, variant)
                    with zipfile.ZipFile(src_zip) as source:
                        config_entry, split_entry, best_entry, history_entries = find_ablation_entries(source, variant, seed)
                        split_bytes = source.read(split_entry)
                        split_hash = sha256_bytes(split_bytes)
                        if split_hash not in split_digests:
                            split_name = f"reviewer_test_package/metadata/splits/{split_hash}.csv"
                            write_bytes(out, split_name, split_bytes)
                            split_digests[split_hash] = split_name
                            manifest["files"][split_name] = split_hash
                        key = f"{variant}/seed{seed}"
                        config_name = f"reviewer_test_package/metadata/ablations/{key}/run_config.json"
                        manifest["files"][config_name] = sha256_bytes(source.read(config_entry))
                        write_bytes(out, config_name, source.read(config_entry))
                        hist_epochs = {}
                        for stage, entry in history_entries.items():
                            data = source.read(entry)
                            hist_name = f"reviewer_test_package/metadata/ablations/{key}/stage{stage}_history.csv"
                            manifest["files"][hist_name] = sha256_bytes(data)
                            write_bytes(out, hist_name, data)
                            hist_epochs[str(stage)] = sum(1 for _ in csv.DictReader(data.decode("utf-8-sig").splitlines()))
                        # Stream the single selected checkpoint from the outer archive into the package.
                        model_name = f"reviewer_test_package/models/ablations/{variant}/seed{seed}/stage4_best.keras"
                        info = zipfile.ZipInfo(model_name)
                        info.compress_type = zipfile.ZIP_STORED
                        info.external_attr = 0o644 << 16
                        h = hashlib.sha256()
                        import io
                        buffer = io.BytesIO()
                        with source.open(best_entry) as inp:
                            while block := inp.read(1024 * 1024):
                                h.update(block)
                                buffer.write(block)
                        model_data = buffer.getvalue()
                        keras_info = verify_keras(model_data, f"{src_zip}!{best_entry}")
                        out.writestr(info, model_data)
                        digest = h.hexdigest()
                        manifest["files"][model_name] = digest
                        manifest["models"].append({"model_id": f"{variant}_seed{seed}", "family": "ablation",
                            "variant": variant, "seed": seed, "checkpoint": model_name.removeprefix("reviewer_test_package/"),
                            "checkpoint_sha256": digest, "split_sha256": split_hash,
                            "split_manifest": split_digests[split_hash].removeprefix("reviewer_test_package/"), "source_archive": str(src_zip.relative_to(ROOT)),
                            "source_member": best_entry, "source_archive_sha256": sha256_file(src_zip),
                            "stage_epochs": hist_epochs, "checkpoint_format": keras_info})

            manifest["deduplicated_split_manifests"] = split_digests
            manifest["counts"] = {"models": len(manifest["models"]), "full_models": len(SEEDS),
                                   "ablation_models": len(SEEDS) * len(VARIANTS), "unique_splits": len(split_digests)}
            manifest["expected_ablations"] = sorted(VARIANTS)
            manifest["source_ablation_variants"] = VARIANTS
            manifest["required_reviewer_studies_not_in_this_checkpoint_set"] = [
                "SE/CBAM and parameter-matched attention baselines",
                "attention order/parallel comparisons",
                "blend and epsilon sensitivity sweeps",
                "EMA/loss/LR factorial controls",
                "runtime, memory, and resolution benchmark",
            ]
            manifest_name = "reviewer_test_package/package_manifest.json"
            manifest_data = json.dumps(manifest, indent=2).encode()
            write_bytes(out, manifest_name, manifest_data)
            out.comment = b"ECA-LDNet reviewer test package; see README and notebook."
        # Reopen the finished archive and verify its directory, CRCs, and all packaged files.
        with zipfile.ZipFile(tmp) as check:
            bad = check.testzip()
            if bad:
                raise ValueError(f"Outer package CRC failure: {bad}")
            if len(check.namelist()) != len(set(check.namelist())):
                raise ValueError("Package contains duplicate ZIP paths")
            packaged_manifest = json.loads(check.read("reviewer_test_package/package_manifest.json"))
            if packaged_manifest["counts"]["models"] != 15:
                raise ValueError("Expected 15 canonical checkpoints")
        os.replace(tmp, OUT)
        print(f"Created and CRC-verified: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
        print(json.dumps(manifest["counts"], indent=2))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
