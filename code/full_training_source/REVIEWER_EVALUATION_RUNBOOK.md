# Reviewer evaluation runbook

This document separates work that already exists from experiments that still need to run. Do not report a planned test as a completed result.

## Current verified local files

- There are Stage 1–4 history CSVs and `stage4_best.keras` checkpoints for seeds 17, 43, and 73 under `full_model_training/`.
- History lengths: seeds 17 and 43 have 60/30/20/30 epochs in Stages 1–4; seed 73 has 60/30/16/30. Seed 73 Stage 3 stopped early after 16 epochs; its Stage 4 history has 30 epochs.
- Per-seed loss/metric curves and a history summary were generated at `full_model_training/training_history_report/`.
- Existing test metrics in `revision_evidence/test_evaluation/` are preliminary. The radiometric rows for 17/73 have no checkpoint manifest; the submitted-preprocessing diagnostic is mismatched to the revised training protocol and used seed 73 `stage4_last.keras`. Do not merge these into the final table.
- The next primary synthetic evaluation must use each seed's Stage 4 `stage4_best.keras`, the same radiometric preprocessing, the same image sizes, and an output manifest. The three seeds are independent runs; report their dataset means as mean ± sample SD across seeds, while also showing each seed and its per-image bootstrap 95% CI.

## Stage objectives and schedule to report

| Stage | Epoch cap | Batch | Peak LR | Objective | Other change |
|---|---:|---:|---:|---|---|
| 1 | 60 | 8 | 1e-3 | 0.35 Charbonnier + 0.20 SSIM + 0.10 Sobel edge + 0.10 log-FFT + 0.25 VGG16 perceptual | 5-epoch warmup/cosine schedule |
| 2 | 30 | 16 | 1e-4 | Same primary objective as Stage 1 | 2-epoch warmup/cosine fine-tune |
| 3 | 20 | 8 | 5e-5 | 0.40 Charbonnier + 0.20 SSIM + 0.10 Sobel edge + 0.05 log-FFT + 0.25 VGG16 perceptual | Reweighted objective; seed 73 early stopped at 16 |
| 4 | 30 | 16 | 1e-5 | 0.40 MSE + 0.40 Charbonnier + 0.05 SSIM + 0.10 five-scale MS-SSIM + 0.05 VGG19 perceptual | EMA 0.999; checkpoint selection uses validation PSNR |

Stage 1 and Stage 2 peak LRs use warmup/cosine schedules (5 and 2 warmup epochs); Stage 3 and Stage 4 use cosine schedules from their listed peak LRs. The CSV records actual per-epoch LR values. `loss` values across stages are not directly comparable because their formulas differ. The generated plots are descriptive; they do not isolate the causal effect of EMA, loss changes, or LR. Reviewer 2 explicitly asks for separate/factorial controls of those changes.

### What the names mean and which metrics to report

`ECA` means Efficient Channel Attention: it is an **architecture component**, not a loss term or a final quality metric. `MS-SSIM` (multi-scale structural similarity; if “MSEA” was intended to mean this) is a **Stage 4 loss term** with weight 0.10 and also a reported full-reference metric. It is not a separate model module. The reviewer letter asks for PSNR/SSIM/MS-SSIM, LPIPS/DISTS, and real-image no-reference quality measures; it does not ask for a metric named MSEA.

| Evaluation data | Report these metrics | Interpretation/output |
|---|---|---|
| Paired synthetic and paired real (RESIDE-6K Test only if truly held out; SOTS Indoor/Outdoor; O-/I-/NH-/DNH-HAZE) | PSNR, SSIM, MS-SSIM, LPIPS, DISTS, MAE, MSE, RMSE, mean CIEDE2000 | Per-image values plus dataset mean, SD and image-bootstrap 95% CI. LPIPS/DISTS need `pyiqa`; CIEDE2000 needs `scikit-image`. Keep synthetic 256×256 and real native-resolution results as separately labeled protocols. |
| Unpaired RTTS and real hazy inputs | NIQE, BRISQUE, MUSIQ, MANIQA, CLIP-IQA, FADE | Score hazy and dehazed images, report per-image deltas and bootstrap intervals. These are proxies, not ground-truth accuracy. |
| Three model seeds, per dataset | Each seed’s dataset mean, then mean ± **sample** SD across seeds 17/43/73 | Do not pool seed-image pairs as independent seeds. Keep image-bootstrap CIs within each seed/dataset. |
| Runtime/scale test | Batch-1 mean/p50/p95 latency, throughput, peak GPU memory, parameters, device, precision and success/OOM | Test 256, 512, 720p and 1080p for each seed on one fixed device/software stack. Record failures; graph support alone is not evidence of high-resolution performance. |


## Exact evaluation order

1. Freeze the evaluation checkpoint and protocol: `stage4_best.keras` for seeds 17, 43, 73; `radiometric`; no TTA; no test-set checkpoint selection.
2. Run synthetic paired evaluation at 256×256 on RESIDE-6K Test, SOTS Indoor, and SOTS Outdoor. Save predictions, image-level CSVs, summary CSVs, a per-run manifest (checkpoint SHA-256, inputs, command, protocol, versions, counts), and the overall manifest.
3. Run real paired evaluation on O-HAZE, I-HAZE, NH-HAZE, and DNH-HAZE when the official data are available. Try native image dimensions using the dynamic model rebuild. Record actual input dimensions and pair counts. If GPU memory prevents a dataset, resize proportionally with the same documented maximum side for every model and label it resized; do not call that native-resolution evaluation.
4. Run RTTS as unpaired real-world evaluation: save outputs and paired hazy/dehazed MUSIQ, MANIQA, CLIP-IQA, NIQE, BRISQUE, and FADE values with per-image deltas and bootstrap intervals. No-reference IQA scores are proxies, not ground-truth dehazing accuracy; a lower FADE indicates lower predicted fog density but can also result from over-enhancement.
5. Report per-dataset results separately. For each metric, show seed 17/43/73, mean ± sample SD across the 3 seed-level means, and the image-bootstrap CI within each seed. Do not pool all images from all seeds as if independent model seeds.
6. Inspect visual outputs with fixed crops for sky, dense haze, edges, bright objects, and color-sensitive regions; include failure cases and no cherry-picking.
7. Benchmark the same three models on one declared device/software stack at 256, 512, 1280×720, and 1920×1080. Record warm-up, batch=1 latency distribution, peak memory, throughput, precision, and device. If 1080p fails from OOM, report the failure and peak size that succeeded.
8. Only after this locked primary evaluation, run the reviewer-requested ablations and sensitivity studies listed below. Do not tune a model on these test sets.

### Keep final testing separate from the next ablation phase

The final testing run evaluates the already-trained Stage 4 checkpoints for seeds 17/43/73. It does not test why a component or training stage helped. `MS-SSIM` is both a reported image-quality metric and one weighted term in the Stage 4 polish objective; report it as a metric in this evaluation, and test its loss contribution later as an ablation. The reviewer-requested training study should separately compare the primary objective with the Stage 4 polish objective, remove each active loss term in controlled runs (Charbonnier, SSIM/MS-SSIM, Sobel edge, FFT, and VGG perceptual), and isolate EMA, loss reweighting, and late-LR schedule effects. ECA, pixel attention, ASM blend/epsilon, and matched SE/CBAM/order controls are architectural/sensitivity experiments, also performed after the locked test evaluation. Existing stage loss curves describe optimization but cannot attribute causal gains to any one term.

## Kaggle notebook setup

Upload these two local ZIPs as Kaggle inputs using your existing upload workflow: `kaggle_revision/eca_ldnet_reviewer_eval_code.zip` and `kaggle_revision/eca_ldnet_full_model_training_final.zip`. If Kaggle exposes them as ZIP files, Cell 1 extracts them; if your model upload flow already unpacks them into folders, Cell 1 uses those folders directly. Attach RESIDE-6K and SOTS separately. The cells write to `/kaggle/working`; the upload inputs stay read-only.

**Run Cell 3 (dependency install) first in a fresh Kaggle session**, before Cell 1 imports TensorFlow. If installation changes core framework versions, restart the session and rerun Cell 3 before proceeding. Then run Cells 1, 2, optional 2B, 4, optional 5, 5B, and finally 6.

### Cell 1 — paths and checkpoint verification

```python
from pathlib import Path
import os, sys, json, csv, subprocess, platform, zipfile

INPUTS_ROOT = Path("/kaggle/input")
def extract_if_needed(zip_name, dest):
    hits = list(INPUTS_ROOT.rglob(zip_name))
    if len(hits) == 1:
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(hits[0]) as zf:
            bad = zf.testzip()
            assert bad is None, f"Corrupt member in {hits[0]}: {bad}"
            zf.extractall(dest)
        return dest
    assert len(hits) == 0, f"Found duplicate {zip_name} inputs: {hits}"
    return None

CODE_ROOT = extract_if_needed(
    "eca_ldnet_reviewer_eval_code.zip", Path("/kaggle/working/reviewer_eval_code")
)
if CODE_ROOT is None:
    code_candidates = [p.parent.parent for p in INPUTS_ROOT.rglob("evaluate_paired.py")
                       if p.parent.name == "scripts" and (p.parent.parent / "eca_ldnet").is_dir()
                       and "_load_eval_model" in p.read_text(encoding="utf-8")]
    assert len(code_candidates) == 1, f"Expected one uploaded evaluation code root; found {code_candidates}"
    CODE_ROOT = code_candidates[0]

MODEL_BUNDLE_ROOT = extract_if_needed(
    "eca_ldnet_full_model_training_final.zip", Path("/kaggle/working/reviewer_eval_bundle")
)
if MODEL_BUNDLE_ROOT is None:
    model_roots = [p for p in INPUTS_ROOT.rglob("full_model_training")
                   if p.is_dir() and (p / "seed17/stage4/models/stage4_best.keras").is_file()]
    assert len(model_roots) == 1, f"Expected one full_model_training folder; found {model_roots}"
    MODEL_ROOT = model_roots[0]
else:
    MODEL_ROOT = MODEL_BUNDLE_ROOT / "full_model_training"
OUT_ROOT = Path("/kaggle/working/reviewer_evaluation")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
# To continue in a fresh Kaggle session, attach the most recent incremental
# results ZIP as an input. It contains all seeds completed before that archive.
prior_result_zips = list(INPUTS_ROOT.rglob("reviewer_evaluation_after_seed*.zip"))
assert len(prior_result_zips) <= 1, f"Attach at most one cumulative prior-results ZIP: {prior_result_zips}"
if prior_result_zips:
    with zipfile.ZipFile(prior_result_zips[0]) as zf:
        bad = zf.testzip()
        assert bad is None, f"Corrupt prior result archive member: {bad}"
        zf.extractall(OUT_ROOT)
    print("Restored previous seed results:", prior_result_zips[0])
assert (CODE_ROOT / "eca_ldnet").is_dir(), CODE_ROOT
assert (CODE_ROOT / "scripts/evaluate_paired.py").is_file(), CODE_ROOT
assert "--fade" in (CODE_ROOT / "scripts/evaluate_unpaired.py").read_text(encoding="utf-8")
for seed in (17, 43, 73):
    ckpt = MODEL_ROOT / f"seed{seed}" / "stage4" / "models" / "stage4_best.keras"
    assert ckpt.is_file(), f"Missing validation-best checkpoint: {ckpt}"
    print(seed, ckpt, f"{ckpt.stat().st_size / 1e6:.1f} MB")
sys.path.insert(0, str(CODE_ROOT))
os.environ["PYTHONPATH"] = str(CODE_ROOT)
import tensorflow as tf
print("Python", sys.version)
print("TensorFlow", tf.__version__)
print("GPU", tf.config.list_physical_devices("GPU"))
assert tf.config.list_physical_devices("GPU"), "Enable a Kaggle GPU"
from eca_ldnet.model import load_pretrained, build_eca_ldnet
import gc, numpy as np
for seed in (17, 43, 73):
    ckpt = MODEL_ROOT / f"seed{seed}/stage4/models/stage4_best.keras"
    fixed = load_pretrained(str(ckpt), compile=False)
    observed_input = len(fixed.get_layer("physics_output").input) == 4
    dynamic = build_eca_ldnet(img_size=None, physics_uses_observed_input=observed_input)
    dynamic.set_weights(fixed.get_weights())
    probe = dynamic(np.zeros((1, 272, 304, 3), dtype=np.float32), training=False)
    assert tuple(probe.shape) == (1, 272, 304, 3), (seed, probe.shape)
    print("Dynamic checkpoint transfer OK:", seed, tuple(probe.shape), "observed-input ASM:", observed_input)
    del fixed, dynamic, probe
    tf.keras.backend.clear_session()
    gc.collect()
history_plot_script = CODE_ROOT / "scripts/plot_stage_histories.py"
subprocess.run([sys.executable, str(history_plot_script), "--training-root", str(MODEL_ROOT),
                "--output", str(OUT_ROOT / "training_history_report")],
               cwd=CODE_ROOT, check=True)
```

### Cell 2 — dataset paths and pair-count check

```python
RESIDE_TEST = Path("/kaggle/input/datasets/kmljts/reside-6k/RESIDE-6K/test")
SOTS_ROOT = Path("/kaggle/input/datasets/balraj98/synthetic-objective-testing-set-sots-reside")
datasets = {}
if (RESIDE_TEST / "hazy").is_dir() and (RESIDE_TEST / "GT").is_dir():
    datasets["RESIDE-6K"] = (RESIDE_TEST / "hazy", RESIDE_TEST / "GT", False)
else:
    print("No separate RESIDE-6K test folders found at", RESIDE_TEST,
          "— do not substitute training pairs; SOTS remains the held-out synthetic test.")
datasets.update({
    "SOTS-Indoor": (SOTS_ROOT / "indoor/hazy", SOTS_ROOT / "indoor/clear", False),
    "SOTS-Outdoor": (SOTS_ROOT / "outdoor/hazy", SOTS_ROOT / "outdoor/clear", False),
})
assert (SOTS_ROOT / "indoor/hazy").is_dir() and (SOTS_ROOT / "outdoor/hazy").is_dir(), \
       f"SOTS folders not found under {SOTS_ROOT}; inspect /kaggle/input and set SOTS_ROOT correctly."
from eca_ldnet.data import get_gt_path
image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
for name, (hazy, gt, native) in datasets.items():
    assert hazy.is_dir() and gt.is_dir(), (name, hazy, gt)
    hazy_files = sorted(p.name for p in hazy.iterdir() if p.suffix.lower() in image_exts)
    gt_files = sorted(p.name for p in gt.iterdir() if p.suffix.lower() in image_exts)
    matched = [name for name in hazy_files if get_gt_path(name, str(gt))]
    missing = sorted(set(hazy_files) - set(matched))
    print(name, "hazy:", len(hazy_files), "GT:", len(gt_files), "matched:", len(matched), "unmatched:", missing[:5])
    assert len(matched) == len(hazy_files), f"Fix pair naming before evaluation: {name}, missing={len(missing)}"
```

### Cell 2B — add real paired datasets after attaching/extracting them

For each real paired dataset you actually attached, set its hazy and GT paths below to the exact extracted folders. Add only datasets whose two folders exist; O-HAZE/I-HAZE may need author-provided archives because the old official links are unavailable. NH-HAZE is sufficient to run a real paired evaluation while the other sources are being resolved. Rerun Cell 2's pair-count preflight after this cell.

```python
real_candidates = {
    "O-HAZE": (Path("/kaggle/input/REPLACE/O-HAZE/hazy"), Path("/kaggle/input/REPLACE/O-HAZE/GT"), True),
    "I-HAZE": (Path("/kaggle/input/REPLACE/I-HAZE/hazy"), Path("/kaggle/input/REPLACE/I-HAZE/GT"), True),
    "NH-HAZE": (Path("/kaggle/input/REPLACE/NH-HAZE/hazy"), Path("/kaggle/input/REPLACE/NH-HAZE/GT"), True),
    "DNH-HAZE": (Path("/kaggle/input/REPLACE/DNH-HAZE/hazy"), Path("/kaggle/input/REPLACE/DNH-HAZE/GT"), True),
}
for name, (hazy, gt, native) in real_candidates.items():
    if hazy.is_dir() and gt.is_dir():
        datasets[name] = (hazy, gt, native)
        print("Added real paired set:", name, hazy, gt)
    else:
        print("Not attached; skipping for now:", name, hazy, gt)
```

Use expected counts 45/35/55 only with the full corresponding official split. Some hosted copies are challenge train/test partitions; if the counts differ, record the exact partition and don't label it as the full set.

### Cell 3 — install optional perceptual metrics

```python
%pip -q install pyiqa scikit-image fade-python
```

If the package install changes TensorFlow/PyTorch versions or errors, restart the session and install in a clean Kaggle session. Do not silently omit LPIPS/DISTS from the final table; write down any metric that could not be run.

### Cell 4 — paired synthetic and real datasets

```python
import sys, shutil
eval_script = CODE_ROOT / "scripts/evaluate_paired.py"
env = os.environ.copy()
env["PYTHONPATH"] = str(CODE_ROOT)
# Run one seed at a time to preserve a downloadable checkpoint if Kaggle stops.
# Change this to [43], then [73], and rerun the cell for the remaining seeds.
# If starting a new Kaggle session, attach the previous cumulative ZIP first;
# Cell 1 restores it before this cell runs.
SEEDS_TO_RUN = [17]
for seed in SEEDS_TO_RUN:
    model = MODEL_ROOT / f"seed{seed}/stage4/models/stage4_best.keras"
    for name, (hazy, gt, native) in datasets.items():
        out = OUT_ROOT / f"seed{seed}" / name
        cmd = [sys.executable, str(eval_script), "--model", str(model),
               "--hazy-dir", str(hazy), "--gt-dir", str(gt), "--dataset", name,
               "--output", str(out), "--preprocessing", "radiometric",
               "--bootstrap-samples", "5000", "--seed", "42",
               "--perceptual", "lpips", "dists"]
        if native:
            cmd.append("--native-resolution")
        else:
            cmd += ["--size", "256"]
        print("RUN", " ".join(cmd))
        subprocess.run(cmd, cwd=CODE_ROOT, env=env, check=True)
    archive_base = f"/kaggle/working/reviewer_evaluation_after_seed{seed}"
    shutil.make_archive(archive_base, "zip", root_dir=OUT_ROOT)
    print("Incremental results archive:", archive_base + ".zip")
    from IPython.display import FileLink, display
    display(FileLink(archive_base + ".zip"))
```

For real paired data, add each dataset only after checking its exact folder layout and hazy/GT name matching. Typical counts for the full benchmark releases are O-HAZE 45, I-HAZE 35, and NH-HAZE 55; confirm DNH-HAZE partition/count from the official challenge release; evaluator output counts must agree or be explained. Preserve RGB/GT alignment; do not independently crop or resize either side.

### Cell 5 — real-image no-reference metrics and unpaired RTTS

Attach/extract RTTS so that `RTTS_ROOT/JPEGImages` contains its test hazy images. The official RESIDE-Beta page describes RTTS as a VOC-style set; it has no paired clean targets.

```python
# Set this only if RTTS has been attached and extracted. Paired real datasets
# are pulled from `datasets` added in Cell 2B and get the same NR-IQA metrics.
RTTS_HAZY = None  # e.g. Path("/kaggle/input/.../RTTS/JPEGImages")
if RTTS_HAZY is not None:
    RTTS_HAZY = Path(RTTS_HAZY)
    assert RTTS_HAZY.is_dir(), RTTS_HAZY

unpaired_script = CODE_ROOT / "scripts/evaluate_unpaired.py"
nr_datasets = {
    name: (hazy, gt)
    for name, (hazy, gt, _native) in datasets.items()
    if name in {"O-HAZE", "I-HAZE", "NH-HAZE", "DNH-HAZE"}
}
if RTTS_HAZY is not None:
    nr_datasets["RTTS"] = (RTTS_HAZY, None)
assert nr_datasets, "Add paired real datasets in Cell 2B or set RTTS_HAZY."

# Run one seed at a time. Change to [43], then [73], to avoid losing progress
# or rerunning completed seeds if the Kaggle session is interrupted.
NR_SEEDS_TO_RUN = [17]
for seed in NR_SEEDS_TO_RUN:
    model = MODEL_ROOT / f"seed{seed}/stage4/models/stage4_best.keras"
    for name, (hazy, _gt) in nr_datasets.items():
        out = OUT_ROOT / f"seed{seed}" / name / "nr_iqa"
        cmd = [sys.executable, str(unpaired_script), "--model", str(model),
               "--hazy-dir", str(hazy), "--dataset", name,
               "--output", str(out), "--preprocessing", "radiometric",
               "--metrics", "niqe", "brisque", "musiq", "maniqa", "clipiqa",
               "--fade", "--bootstrap-samples", "5000", "--seed", "42"]
        subprocess.run(cmd, cwd=CODE_ROOT, env=env, check=True)
    archive_base = f"/kaggle/working/reviewer_evaluation_after_seed{seed}"
    shutil.make_archive(archive_base, "zip", root_dir=OUT_ROOT)
    print("Incremental results archive:", archive_base + ".zip")
    from IPython.display import FileLink, display
    display(FileLink(archive_base + ".zip"))
```

For O-HAZE/I-HAZE/NH-HAZE, the no-reference rows score the hazy input and dehazed output, while Cell 4 reports full-reference scores against the aligned clear target. Keep these tables separate: NR-IQA scores are perceptual proxies and can move in a favorable direction even when the image is over-enhanced. If RTTS is unavailable, record it as unavailable; do not substitute a paired set and call it unpaired.

### Cell 5B — batch-1 resolution latency and peak memory

Run this on the same Kaggle GPU for all three checkpoints. This records actual inference timings at the target scales; it does not claim that larger resolutions improve quality. Cell 4 supplies full-reference native-resolution quality scores on any attached paired real datasets.

```python
import csv, gc, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from eca_ldnet.model import load_pretrained, build_eca_ldnet

benchmark_path = OUT_ROOT / "resolution_latency_memory.csv"
benchmark_fields = ["seed", "target_h", "target_w", "padded_h", "padded_w", "batch",
                    "precision", "device", "parameters", "status", "error", "gpu_peak_mb",
                    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "throughput_images_s"]
benchmark_rows = list(csv.DictReader(benchmark_path.open(encoding="utf-8"))) if benchmark_path.is_file() else []
completed_resolutions = {(int(r["seed"]), int(r["target_h"]), int(r["target_w"])) for r in benchmark_rows}
def save_benchmark_rows():
    with benchmark_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=benchmark_fields)
        writer.writeheader(); writer.writerows(benchmark_rows)

gpu_name = tf.config.list_physical_devices("GPU")[0].name
target_sizes = [(256, 256), (512, 512), (720, 1280), (1080, 1920)]
for seed in (17, 43, 73):
    if all((seed, h, w) in completed_resolutions for h, w in target_sizes):
        print("Already benchmarked seed:", seed)
        continue
    fixed = load_pretrained(
        str(MODEL_ROOT / f"seed{seed}/stage4/models/stage4_best.keras"), compile=False
    )
    observed_input = len(fixed.get_layer("physics_output").input) == 4
    model = build_eca_ldnet(img_size=None, physics_uses_observed_input=observed_input)
    model.set_weights(fixed.get_weights())
    params = int(model.count_params())
    del fixed
    infer = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    for h, w in target_sizes:
        if (seed, h, w) in completed_resolutions:
            continue
        ph, pw = ((h + 15) // 16) * 16, ((w + 15) // 16) * 16
        x = tf.random.uniform((1, ph, pw, 3), dtype=tf.float32, seed=42)
        row = {"seed": seed, "target_h": h, "target_w": w,
               "padded_h": ph, "padded_w": pw, "batch": 1,
               "precision": "float32", "device": gpu_name, "parameters": params,
               "status": "ok", "error": ""}
        try:
            for _ in range(3):
                y = infer(x)
                _ = y[:, :h, :w, :].numpy()
            try:
                tf.config.experimental.reset_memory_stats("GPU:0")
            except (ValueError, RuntimeError):
                pass
            times_ms = []
            for _ in range(20):
                t0 = time.perf_counter()
                y = infer(x)
                _ = y[:, :h, :w, :].numpy()  # synchronize device before stopping timer
                times_ms.append((time.perf_counter() - t0) * 1000)
            try:
                row["gpu_peak_mb"] = tf.config.experimental.get_memory_info("GPU:0")["peak"] / (1024 ** 2)
            except (ValueError, RuntimeError):
                row["gpu_peak_mb"] = float("nan")
            row.update({"latency_mean_ms": float(np.mean(times_ms)),
                        "latency_p50_ms": float(np.percentile(times_ms, 50)),
                        "latency_p95_ms": float(np.percentile(times_ms, 95)),
                        "throughput_images_s": 1000.0 / float(np.mean(times_ms))})
        except tf.errors.ResourceExhaustedError as exc:
            row.update({"status": "OOM", "error": str(exc)[:500],
                        "gpu_peak_mb": float("nan"), "latency_mean_ms": float("nan"),
                        "latency_p50_ms": float("nan"), "latency_p95_ms": float("nan"),
                        "throughput_images_s": float("nan")})
            print("OOM at", seed, h, w, "— retained in benchmark table")
        benchmark_rows.append(row)
        completed_resolutions.add((seed, h, w))
        save_benchmark_rows()
        del x
        if "y" in locals(): del y
        gc.collect()
    del infer, model
    tf.keras.backend.clear_session()
    gc.collect()
    archive_base = f"/kaggle/working/reviewer_evaluation_after_seed{seed}"
    shutil.make_archive(archive_base, "zip", root_dir=OUT_ROOT)
    from IPython.display import FileLink, display
    display(FileLink(archive_base + ".zip"))
print("Saved resolution benchmark:", benchmark_path)
```

If a target scale OOMs, retain the row and report that failure. Repeat runs only if the first timing shows instability, and do not compare timings from different GPU types or precision modes as if they were matched.

### Cell 6 — aggregate across seeds and archive results

```python
import pandas as pd
summary_rows = []
for seed in (17, 43, 73):
    for dataset_name in datasets:
        p = OUT_ROOT / f"seed{seed}" / dataset_name / "paired_summary.csv"
        d = pd.read_csv(p)
        d["seed"] = seed
        summary_rows.append(d)
all_seed = pd.concat(summary_rows, ignore_index=True)
all_seed.to_csv(OUT_ROOT / "all_seed_level_results.csv", index=False)
across = (all_seed.groupby(["dataset", "metric"])["mean"]
          .agg(seed_mean="mean", seed_sd="std", seed_count="count").reset_index())
across.to_csv(OUT_ROOT / "three_seed_mean_sd.csv", index=False)
nr_datasets = globals().get("nr_datasets", {})
nr_rows = []
for seed in (17, 43, 73):
    for dataset_name in nr_datasets:
        p = OUT_ROOT / f"seed{seed}" / dataset_name / "nr_iqa" / "unpaired_summary.csv"
        if p.is_file():
            d = pd.read_csv(p)
            d["seed"] = seed
            nr_rows.append(d)
if nr_rows:
    all_nr = pd.concat(nr_rows, ignore_index=True)
    all_nr.to_csv(OUT_ROOT / "all_seed_nr_iqa_results.csv", index=False)
    nr_across = (all_nr.groupby(["dataset", "metric"])
                 .agg(seed_count=("seed", "nunique"),
                      hazy_mean_across_seeds=("hazy_mean", "mean"),
                      hazy_seed_sd=("hazy_mean", "std"),
                      dehazed_mean_across_seeds=("dehazed_mean", "mean"),
                      dehazed_seed_sd=("dehazed_mean", "std"),
                      delta_mean_across_seeds=("delta_mean", "mean"),
                      delta_seed_sd=("delta_mean", "std")).reset_index())
    nr_across.to_csv(OUT_ROOT / "three_seed_nr_iqa_mean_sd.csv", index=False)
manifest = {
    "preprocessing": "radiometric", "synthetic_resolution": "256x256",
    "real_paired_resolution": "native if memory permits; record actual dimensions",
    "checkpoint_rule": "stage4_best.keras selected by validation PSNR",
    "seeds": [17, 43, 73], "python": sys.version,
    "paired_datasets_evaluated": list(datasets),
    "no_reference_datasets_evaluated": list(nr_datasets),
    "tensorflow": tf.__version__, "device": [str(x) for x in tf.config.list_physical_devices("GPU")],
    "checkpoints": {str(s): str(MODEL_ROOT / f"seed{s}/stage4/models/stage4_best.keras") for s in (17,43,73)},
}
(OUT_ROOT / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2))
import shutil
shutil.make_archive("/kaggle/working/reviewer_evaluation", "zip", OUT_ROOT)
print("Created /kaggle/working/reviewer_evaluation.zip")
from IPython.display import FileLink, display
display(FileLink("/kaggle/working/reviewer_evaluation.zip"))
```

This aggregation calculates mean ± sample SD across the three seed-level means. Paired image-bootstrap intervals remain in each seed/dataset `paired_summary.csv`; NR-IQA paired hazy/dehazed deltas and their per-image bootstrap intervals remain in each `unpaired_summary.csv`, with the across-seed summary saved separately. Do not combine paired and no-reference tables.

## What to download on the laptop

The project-local destination should be `datasets/review_test/`. Kaggle CLI commands below keep the downloaded archives in `downloads` and extract alongside them; run them from PowerShell in the repository root. They require Kaggle CLI authentication. RESIDE-6K Test and SOTS are synthetic paired sets; O-/I-/NH-/DNH-HAZE are real-haze paired sets; RTTS is unpaired.

```powershell
$ProjectRoot = 'D:\Image-Dehazing-using-ECA-LDNet'
Set-Location $ProjectRoot
py -m pip install --user kaggle
$UserScripts = py -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='nt_user'))"
$env:Path = "$UserScripts;$env:Path"
kaggle auth login
.\scripts\download_review_data.ps1
```

The install/auth commands follow the current Kaggle CLI instructions; OAuth login avoids placing a raw API key in this project. The CLI uses your Kaggle account's access to download the datasets.

That project script keeps both Kaggle archives and extracted datasets under `datasets/review_test/`; it also downloads the official NH-HAZE archive. The explicit commands it runs are:

```powershell
$Project = 'D:\Image-Dehazing-using-ECA-LDNet'
$DataRoot = Join-Path $Project 'datasets\review_test'
$Downloads = Join-Path $DataRoot 'downloads'
New-Item -ItemType Directory -Force $Downloads | Out-Null
kaggle datasets download -d kmljts/reside-6k -p $Downloads
kaggle datasets download -d balraj98/synthetic-objective-testing-set-sots-reside -p $Downloads
Get-ChildItem $Downloads -Filter '*.zip' | ForEach-Object {
    $dest = Join-Path $DataRoot $_.BaseName
    New-Item -ItemType Directory -Force $dest | Out-Null
    Expand-Archive -LiteralPath $_.FullName -DestinationPath $dest -Force
}
```

For O-HAZE, I-HAZE, NH-HAZE, and DNH-HAZE, download from their official dataset pages and save the original archives under `datasets/review_test/downloads/`, then extract each archive into its own folder. The old O-/I-HAZE page links currently resolve to unavailable URLs, so use the dataset page/download link if restored or an author-provided official mirror; verify the number of pairs and cite the exact source/version. NH-HAZE has an official 330 MB archive. Do not use an unverified mirror in a paper. RTTS is available from the official RESIDE-β page via its download link; the page provides Dropbox/Baidu options, so browser download may be required. These datasets are not in the local training `datasets/` folder now.

NH-HAZE's currently listed official archive can be downloaded with this additional PowerShell command:

```powershell
$NhhazeZip = Join-Path $Downloads 'NH-HAZE.zip'
Invoke-WebRequest 'https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip' -OutFile $NhhazeZip
Expand-Archive -LiteralPath $NhhazeZip -DestinationPath (Join-Path $DataRoot 'NH-HAZE') -Force
```

The ETH O-/I-HAZE pages list the datasets and sizes (547 MB and 312 MB), but their linked archives currently return 404. Open the official pages to obtain a working official link or request it from the dataset authors; do not replace them with an unverified copy. If you download an official archive in the browser, save it to `$Downloads` and rerun `scripts/download_review_data.ps1` to extract it there.

```powershell
Start-Process 'https://data.vision.ee.ethz.ch/cvl/vision2/ntire18/o-haze/'
Start-Process 'https://data.vision.ee.ethz.ch/cvl/vision2/ntire18/i-haze/'
Start-Process 'https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/'
Start-Process 'https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2/'
```

Those pages describe O-HAZE's 45 and I-HAZE's 35 real hazy/clear pairs; NH-HAZE has 55 non-homogeneous real pairs and explicitly permits use/publication with citation. The official RESIDE-β page lists RTTS and its VOC-style `JPEGImages` layout. Its RTTS download link redirects to Box; browser download may be needed, then save the archive under `$Downloads` and rerun the script.

## Remaining reviewer experiments (after final-model evaluation)

1. **Blending/epsilon:** freeze a validation protocol; blend 0/2/5/8/10/20%, learnable weight, and epsilon sensitivity; report paired PSNR/SSIM/LPIPS/DISTS and real no-reference scores. Select on validation; test only once after selection.
2. **Component/matched-capacity ablations:** ECA only, pixel attention only, ECA+pixel, no physics, SE, CBAM, simple spatial gate, residual refinement, standard/group/depthwise convolution, attention order and parallel. Same data/split/training budget; at least 3 seeds for primary comparisons; report parameters and MACs.
3. **Loss and training-factor studies:** individual loss terms and Stage 1/2 vs Stage 3 vs Stage 4 objective; a controlled factorial for EMA on/off, loss reweighting on/off, and late LR schedule. Do not claim Stage 4 gain is caused by one factor from the current curves.
4. **ASM interpretation:** compare latent A/t to synthetic ground-truth if metadata is available; otherwise call them uncalibrated latent variables and soften physical claims. Include randomized/detached/feature-level auxiliary controls if retaining the physics-guidance claim.
5. **Fair compact baselines and deployment:** retrain relevant compact models with the same split, preprocessing, resolution, augmentation, and budget. Measure all models on the same modern GPU and, if feasible, CPU/Jetson, including 256/512/720p/1080p latency and memory.
6. **Manuscript/rebuttal:** clarify engineering design-point novelty, avoid SOTA/real-world claims unsupported by results, update related work, figures/crops/failure cases, English, limitations, and a point-by-point response. Add the editor-required `Declarations` section with separate Ethical approval, Consent to participate, Consent to publish headings and “Not Applicable” only if true for all authors.

### Full Reviewer 2 subsection crosswalk

- **Opening / Abstract / §§1–2:** describe an engineering operating point, not a new dehazing paradigm; avoid leadership claims; specify the P100/FP32/256/batch-1 context for the old speed figure; update related work on real-domain adaptation, realistic haze synthesis, non-homogeneous haze, and high-resolution dehazing; reduce leaderboard-style prose.
- **§3 and §3.3–3.4:** train capacity-matched depthwise, standard and group-convolution baselines; compare SE, CBAM, simple spatial gate, ECA-only, pixel-attention-only, SE+PA, PA→ECA, ECA→PA, parallel ECA+PA, and residual-refinement controls. Plot attention maps and state the PSNR versus SSIM/MS-SSIM trade-off.
- **§§3.1–3.2:** run and report actual 512/720p/1080p resolution and latency/memory tests; distinguish graph compatibility from validated scale-up; discuss homogeneous ASM mismatch and non-homogeneous haze.
- **§§3.5–3.7:** analyze the bottleneck A/t-like outputs; if synthetic A/t labels are accessible, compute MAE/RMSE/SSIM; compare randomized, detached, and feature-level auxiliary controls or narrow the claim. Test blend 0/2/5/8/10/20%, a constrained learnable blend, and epsilon sensitivity, with paired and real-image metrics.
- **§3.8 and §4.10:** remove individual loss terms and compare primary versus Stage 4 polish losses; independently test EMA, loss reweighting, and LR schedule changes (factorial/controlled contrasts).
- **§§4.1–4.3:** keep the scene-grouped split; evaluate real paired and unpaired sets, native dimensions where feasible, LPIPS/DISTS and NR-IQA/FADE; repeat the main model and key ablations over at least three seeds and report mean ± SD plus paired uncertainty.
- **§§4.4–4.8:** retrain fair compact baselines under the same training protocol; do not claim superiority from unmatched published tables or the 22-image case set; state trade-offs and limitations.
- **§4.9:** compare latency, memory, and throughput on the same modern device/software stack across stated resolutions; include CPU/Jetson/mobile only if measured.
- **§4.11 and §§5–6:** use standardized zooms and failure cases, include attention maps and A/t visuals only with clear uncalibrated labels, discuss synthetic-to-real limitations, and revise the discussion/conclusion to match actual evidence.

## Reviewer status (not a completion percentage)

- **Editor:** experiment families and manuscript tasks are identified; real-data evaluation and manuscript/rebuttal changes remain.
- **Reviewer 1:** novelty explanation, blend sensitivity, stage-factor analysis, and upgraded figures remain.
- **Reviewer 2:** most extensive remaining work: controlled compact baselines, resolution/efficiency runs, real/near-real tests, perceptual/NR-IQA metrics, blend/epsilon, loss/EMA/LR factorial, A/t validation or narrower claims, and manuscript reframing.
- **Reviewer 3:** multi-seed main result is present for 17/43/73, but clean unified evaluation/mean±SD still must be generated; novelty and blend sensitivity plus English edits remain.

The three seeds satisfy the count of three complete Stage 4 history files, but the final three-seed evaluation table has not been generated yet. Reviewer requests are not “completed” until the corresponding results and manuscript response are finished.
