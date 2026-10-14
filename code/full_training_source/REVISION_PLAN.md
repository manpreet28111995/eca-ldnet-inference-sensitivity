# Springer Major Revision: ECA-LDNet

Prepared from the submitted 28-page manuscript, the decision/reviewer letter, and the current repository. This is an execution plan; it does not edit the submitted PDF or make claims about experiments that have not been run.

## What the decision means

The editor invited a **major revision** and asks for a point-by-point response. Reviewer 2 recommends rejection, but that is one reviewer recommendation; the editor's decision is the operative instruction in the letter. Treat this as a real opportunity to revise, with a substantial risk that the editor may reject if the central evidence remains weak. The letter recommends return within 10 days. Check the live submission portal and, if that window has passed or is infeasible, ask the corresponding author to contact the editor immediately with the submission ID and a realistic extension request.

The paper already admits most of the reviewers' central weaknesses: the novelty is integration of established modules; the 8% blend is untested; the ablations are single-seed and shortened; training/validation split is pair-random; most evidence is synthetic; baseline comparisons are not controlled; and all compute/latency evidence is at 256 x 256 on a P100. The revision must replace these limitations with data, or explicitly narrow the claims where evidence cannot be produced.

## Immediate repository findings to resolve before training

1. `eca_ldnet/train.py` does a random 90/10 split of hazy/clean pairs. It does not group variants by source clean scene, so it cannot rule out scene leakage.
2. `eca_ldnet/data.py::preprocess_hazy` applies per-channel min-max normalization to each hazy image, while the clean target is only resized and divided by 255. This changes the input's radiometry without applying the same mapping to its paired target, and can break the relation assumed by the ASM-derived branch. Freeze and document the submitted protocol for a faithful reproduction, then run a separate corrected, paired-consistent preprocessing protocol as a controlled experiment. Do not silently mix these results.
3. **Confirmed critical ASM wiring mismatch:** inspection of both `Models/stage5_final_ema_model.keras` and `Models/stage5_best.keras` shows `physics_output` receives `raw_sigmoid`, `atm_light`, and `t_full`. It therefore computes `(J_raw-A)/(t+eps)+A`, not the paper's displayed `(I_hazy-A)/(t+eps)+A`. The original training notebook has the same call. Revise the method description to match the saved model, and test a corrected observed-input formulation only as a distinct model variant. Never describe the old checkpoint as implementing the displayed equation.
4. The source notebooks recover distinct S1-S3 losses and a Stage 4 MSE/Charbonnier/SSIM/MS-SSIM/VGG19 loss. The exact Stage 5 recipe is not present in the checked-in notebooks and remains unresolved; do not claim a faithful S5 rerun until it is recovered.
5. Final `.keras` checkpoints exist under `Models/` (including `stage5_final_ema_model.keras`), so start with checkpoint/graph forensics and inexpensive inference-only sweeps before deciding how much must be retrained.
6. The active Windows interpreter is Python 3.14.5 with no TensorFlow; the repo's `.venv` has no Windows `Scripts` directory and its Linux `bin/python` is not callable from this PowerShell session. The pinned TensorFlow/Keras stack targets Python 3.10–3.12, so use Kaggle now; do not try this current Windows interpreter for model training. A local GPU route would require a working WSL2/NVIDIA setup and a compatible Python environment.
7. The architecture figure is readable only when enlarged; the common-image montage is very small and has no zoom crops; the single-model figure has only four examples and no detailed crops. Rebuild these as vector/high-resolution figures from original outputs, not screenshots of notebook cells.

## Revision implementation status (29 September 2026)

The repository now contains a revision training/evaluation path, but it is **not yet experimentally validated**; the Windows interpreter in this workspace has no TensorFlow installation. The Kaggle T4 notebook is intended to run syntax/import/model/pilot checks before any multi-day experiment. Source-notebook inspection corrected the stage objective mapping: S3 is reweighted; S4 uses MSE 0.40, Charbonnier 0.40, SSIM 0.05, five-scale MS-SSIM 0.10, and VGG19 0.05. The exact S5 recipe is not in the repository and must not be invented.

Implemented in the working tree:

- Grouped train/validation splitting by SHA-256 of clean targets, with an exported per-pair split manifest and separate split/training seeds.
- Explicit `submitted` versus `radiometric` preprocessing modes. New observed-input ASM runs default to radiometric inputs; legacy checkpoint inference uses the submitted preprocessing setting.
- An explicit observed-hazy-input correction graph for new models and compatibility with the legacy three-input correction graph. The submitted checkpoint is never relabeled as corrected.
- Dynamic spatial input dimensions, native-resolution evaluation with multiple-of-16 padding, multiple attention/convolution controls, loss-term switches, and EMA stages.
- A Kaggle notebook with a one-epoch pilot, fixed-validation blend/epsilon screening, capacity matching, staged training, and result export. Epoch-level `stage*_latest.keras` recovery checkpoints are saved separately from validation-selected `stage*_best.keras` checkpoints.

Still required before claiming results: run Kaggle runtime tests and pilot; verify the actual Kaggle dataset directory/pairing and clean-scene grouping; verify checkpoint loading, EMA checkpoint semantics, uninterrupted/resumed training behavior, and the real measured step time; then conduct and report only the experiments completed. The `latest` recovery checkpoint is a recovery point; restarting from it initializes a fresh stage optimizer/schedule, so it is not an exact optimizer-state resume.

## Work order and stop/go gates

## Seven-day execution target (29 Sep–6 Oct 2026)

This is a deadline-driven target, not a promise that every requested experiment will finish. At the end of Day 1, use measured pilot throughput and actual Kaggle quota to cut scope if needed.

- **29 Sep:** corresponding-author extension request; checkpoint/code audit (the ASM mismatch is confirmed); identify exact Kaggle datasets and source-scene IDs; verify GPU/session and run a one-epoch plus checkpoint-reload pilot.
- **30 Sep:** implement/validate scene-grouped manifest and metric pipeline; extract checkpoint intermediates and finish inference-only blend sweep; freeze the experimental protocol.
- **1–3 Oct:** train only the highest-priority controlled runs that fit the measured compute budget; start with repeated seeds for the full model and the most informative ablations. In parallel, evaluate the frozen checkpoint on paired and unpaired real-haze sets.
- **4 Oct:** finish same-hardware benchmarking, tables, zoom-crop figures, and result uncertainty; stop launching experiments that cannot finish and be analyzed before the deadline.
- **5 Oct:** revise manuscript and point-by-point response; author check of the ASM correction description, declarations, claims, and all results.
- **6 Oct:** final consistency check and submission-ready package; submit by this date to keep the requested two-day buffer before the 8 Oct deadline.

If the pilot indicates that even the core multi-seed runs cannot finish by 3 Oct, do not fabricate or overstate evidence: prioritize the inference-only blend analysis, one correctly described checkpoint, strongest real-data evaluation, a limited controlled baseline, and explicitly narrowed claims, while pursuing an extension.

### Phase 0 — today: secure the revision and lock data

- Confirm the revision deadline/status in the Springer portal. The letter's recommended 10-day window is urgent; request an extension through the corresponding author if needed.
- Preserve the exact submitted manuscript, checkpoints, configs, and original Kaggle notebooks. Create a revision branch and a new `revision/` results directory; never overwrite the submitted run.
- Inventory the exact train/validation/test files and source-scene IDs for RESIDE-6K and ITS. Generate a manifest with hazy path, clean path, source-scene ID, split, dimensions, and hashes. Fail on unmatched or duplicate pairs.
- Split by clean-scene identity, not by hazy filename. Keep SOTS test sets untouched. Report the split counts and publish the manifest-generation script.
- Decide one primary preprocessing protocol after a direct side-by-side check of the paper's notebook and current package. Record channel order, resize/crop, input scaling, target scaling, and augmentation separately. Validate that transformations preserve hazy/clean alignment.
- Checkpoint graph tracing is complete: the saved final and best checkpoints both use `J_raw` as the image input to the correction. The revision manuscript must disclose and correctly describe that. Decide with the authors whether to retain the checkpoint-defined correction under narrower “ASM-parameterized correction” language or add the observed-input variant; any changed variant must be separately trained/evaluated.
- Make one short pilot epoch and checkpoint-resume smoke test before any full run; record step time, peak VRAM, and whether the exact checkpoint can be restored.

**Gate:** if the data cannot be grouped reliably by source scene, disclose that limitation and avoid claiming leakage-free validation. Do not run the full seed matrix until pairing and checkpoint serialization work.

### Phase 1 — core experiments (highest priority)

Keep the training data, scene split, resolution, preprocessing, augmentations, epoch/update budget, loss, optimizer, and checkpoint-selection rule identical across all variants. Use three independent seeds at minimum for the key comparisons (reviewer 2 asks 3–5). Report mean +/- standard deviation and per-seed values; include paired bootstrap confidence intervals on test images where feasible.

1. **Full model multi-seed:** run ECA-LDNet with seeds 17, 42, and 73 on the scene-grouped split. If budget allows, extend to five seeds. Keep a separate untouched SOTS test evaluation.
2. **Blend sensitivity:** evaluate blend weights 0, 0.02, 0.05, 0.08, 0.10, and 0.20, plus a learnable scalar constrained to [0, 0.2]. First do inference-only sweeps on the same saved raw-CNN output and auxiliary output to isolate the direct blend effect. Then retrain the finalists (0, best fixed weight, 0.08, learnable) with at least three seeds because changing blend during training can change learned representations. Sweep epsilon (0.05, 0.1, 0.2) for the finalist(s), not the entire grid. Report PSNR, SSIM, LPIPS, and paired uncertainty; on real images report NR metrics separately and do not call them ground-truth fidelity.
3. **Architecture ablations:** prioritize a plain depthwise U-Net; ECA-only; pixel-attention-only; ECA+pixel attention; and no-attention. Add PA->ECA vs ECA->PA and parallel ECA/PA if time permits. Add SE and CBAM-style channel/spatial blocks only if capacity can be matched and the implementations can be trained under the same budget. Report exact parameters and measured MACs for every variant. State explicitly that these tests establish this design's component behavior, not a new attention operator.
4. **Training-strategy factorial:** hold architecture and split fixed; separately toggle EMA, polish-loss reweighting, and the late LR schedule (2 x 2 x 2 factorial, one seed for screening, then repeat the decisive contrasts with three seeds). Report S1-S4 checkpoints and validation curves. Keep the unresolved Stage 5 recipe out of reproduction claims. Do not attribute the old S3->S4 gain to any one factor.
5. **Loss ablation:** do not try every combination at full multi-seed cost first. Screen removing each term from the primary loss (Charbonnier, SSIM, edge, FFT, VGG) with the same shortened budget, then repeat only material effects at full budget. Explain the screen/full distinction and report it transparently.

### Phase 2 — real/near-real validation and fair comparisons

- Evaluate paired real-haze datasets such as O-HAZE and NH-HAZE at native resolution where the method supports it; report PSNR/SSIM plus LPIPS or DISTS, with dataset-specific protocols and aligned crops. Include I-HAZE if licensing/download and protocol permit. Do not resize every image to 256 x 256 as the only result.
- Evaluate unpaired real scenes such as RTTS separately. Report FADE and a small, fixed set of NR-IQA metrics (for example MUSIQ and MANIQA) as perceptual proxies, not as objective proof of better dehazing. Keep original images, outputs, metric versions, and per-image scores.
- Retrain at least the plain depthwise U-Net and one relevant lightweight attention baseline (SE or CBAM) under the exact same split and recipe. If feasible, include a compact published dehazer whose official code and data recipe can be reproduced. Clearly separate these controlled results from published literature numbers.
- Add paired image-level bootstrap confidence intervals and per-dataset tables. The existing 22-image pretrained-model comparison remains an illustrative case study only; do not use its 0.03 dB difference as evidence of superiority.
- Visualize hazy input, output, reference (when available), and fixed zoom crops for dense haze, sky/bright objects, edges/textures, and color-sensitive areas. Include representative failures and the transmission-like / atmospheric-light-like outputs with synthetic labels only when true parameter metadata is available.

### Phase 3 — practical efficiency and manuscript

- Benchmark the final checkpoint on the RTX 5060 laptop at 256, 512, and at least one native 720p/1080p image. Record exact GPU name, driver/runtime, framework, precision, batch size 1, warm-up, timed iterations, median and p90 latency, peak allocated/reserved VRAM, and whether tiling is used. Benchmark the controlled baseline(s) on that same machine. Keep P100 numbers only as historical, device-specific context.
- Do not claim edge/mobile real-time performance from an RTX laptop measurement. Jetson/CPU/mobile deployment is optional unless available; absent hardware should be named as a limitation.
- Redraw Fig. 1 with legible feature dimensions and unambiguous arrows. Rebuild accuracy/compute plots from revision results, increase qualitative figure resolution, add crops, consistent color management, and cite image sources. Proof all captions and English (including the phrase flagged as “parameter economy”).
- Rewrite the framing: the contribution is a compact, transparent engineering design and controlled component analysis, not a new attention mechanism, validated physical estimator, SOTA model, or proven edge system. Call the branch “ASM-inspired” unless latent A/t are validated against known synthetic parameters.
- Update related work with verified, relevant recent work (including CoA, HazeFlow, HazeGen/DiffDehaze, and Inf-Dehaze where appropriate); cite primary papers, describe their distinct scope fairly, and avoid a citation dump. The reviewer-mentioned methods are not all direct comparable baselines.
- Declarations: retain a separate **Declarations** section and use the exact individual subheadings requested: **Ethical approval**, **Consent to participate**, and **Consent to publish**, each followed by **Not Applicable** (confirm this is truthful with all authors). Keep data/code/funding/conflict statements separately as applicable.
- Correct terminology and reporting: 12–14 FPS is tied to the measured Tesla P100, FP32, batch 1, 256 x 256 setup; MACs are resolution-specific; the 8% value is empirical; real NR-IQA does not establish reference fidelity; and a fully convolutional graph alone does not prove high-resolution scalability.

## Compute plan: laptop versus Kaggle

- **First choice for repeatable training:** Kaggle GPU sessions, because this project is TensorFlow/Keras and Kaggle provides a preconfigured GPU environment. The exact available accelerator and quota vary; check the account and active session before planning. Save a recovery checkpoint and CSV history every epoch; understand that reloading `stage*_latest.keras` starts a fresh optimizer/schedule rather than restoring optimizer state.
- **Use the RTX 5060 laptop for:** environment verification, pilot throughput/VRAM tests, inference, native-resolution evaluation, figure generation, and possibly training once TensorFlow sees the GPU. This repo requires TensorFlow >=2.16. TensorFlow documentation says native-Windows GPU support ends at TF 2.10; for this repo's versions use WSL2/Linux GPU support, or convert the training stack to PyTorch only if there is enough time to validate parity. Do not downgrade this research stack to TF 2.10 as a quick fix.
- On either device, use mixed precision only after checking output/metric stability against FP32; the FFT, VGG, and image losses need particular care. Start with batch 4 or 8 and gradient accumulation only if needed; reduce batch before reducing image size, because the paper's reported protocol is 256 x 256.
- Estimate full-run cost empirically from the pilot: `steps_per_epoch x measured_seconds_per_step x epochs x seeds`. The paper's original training history and checked-in source recipes do not fully establish Stage 5. Budget only for the recovered S1-S4 recipe and high-priority controlled runs; report any schedule differences. Avoid promising completion from Kaggle's nominal quota before this estimate.

## Minimum credible revision if time/compute is tight

Do these first: (1) correct/justify the split and exact preprocessing; (2) three-seed main result plus key ablations; (3) blend sweep with epsilon check; (4) at least one real paired dataset and one unpaired real set; (5) same-protocol lightweight baseline; (6) split apart EMA/loss/LR effects; (7) figures, declarations, limitations, and full point-by-point rebuttal. Defer a MOS study, broad 2025-method reproduction, Jetson/mobile tests, and a full loss-factorial if time is limited; state the boundary honestly.

## Submission package checklist

- Revised manuscript with tracked changes/highlights if Springer requests them.
- Detailed response letter: paste each editor/reviewer comment verbatim, give a concise response, identify the exact new experiment and manuscript page/line, and explain respectfully where a requested experiment was infeasible and how claims were narrowed.
- New tables/figures and supplementary material; code/config/data split manifest; checkpoint IDs/hashes; per-seed logs; scripts to regenerate metrics and plots.
- Final consistency pass: no number differs between abstract, tables, figures, conclusion, response letter, and released artifacts. Never report a planned or failed run as a result.

## Selected primary/official references to check for the revision

- CoA (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Ma_CoA_Towards_Real_Image_Dehazing_via_Compression-and-Adaptation_CVPR_2025_paper.html
- HazeFlow (ICCV 2025): https://junsung6140.github.io/hazeflow/
- HazeGen / DiffDehaze (arXiv): https://arxiv.org/abs/2503.19262
- Inf-Dehaze (CVPR 2026 Findings): https://openaccess.thecvf.com/content/CVPR2026F/papers/Yan_Inf-Dehaze_Beyond_GPU_Memory_Constraints_for_Ultra-High-Resolution_Image_Dehazing_CVPRF_2026_paper.pdf
- TensorFlow GPU installation and Windows/WSL2 support: https://www.tensorflow.org/install/pip
- Kaggle GPU notebook setup/resource notes: https://www.kaggle.com/docs/notebooks and https://www.kaggle.com/docs/efficient-gpu-usage
