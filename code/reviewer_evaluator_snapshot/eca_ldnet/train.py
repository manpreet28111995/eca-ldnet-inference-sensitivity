"""Multi-stage training entry point for ECA-LDNet.

Reproduces the 3-stage progressive schedule from the paper:

* Stage 1 - full training, LR 1e-3, warmup-cosine, 60 epochs.
* Stage 2 - fine-tune,    LR 1e-4, 30 epochs.
* Stage 3 - Stage-3 polish objective, LR 5e-5, 20 epochs, batch 8.
* Stage 4 - recovered Stage-4 MSE/Charbonnier/SSIM/MS-SSIM/VGG19 objective,
  LR 1e-5, EMA 0.999, 30 epochs by default. Stage 5 is an explicitly
  provisional continuation because its original recipe was not recovered.

Example
-------
    python -m eca_ldnet.train --reside6k /data/RESIDE-6K --its /data/ITS \\
        --out ./runs --stages 1 2 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from . import config as C
from .callbacks import WarmupCosineDecay, make_callbacks
from .data import build_dataset, clean_scene_id, collect_pairs, find_dir, set_global_seed
from .losses import LOSS_PHASE_WEIGHTS, make_loss, mae_metric, psnr_metric, ssim_metric
from .model import build_eca_ldnet, load_pretrained


def _gather_training_pairs(reside6k, its):
    pairs = []
    if reside6k:
        pairs += collect_pairs(
            find_dir(reside6k, ["train/hazy", "Train/hazy"]),
            find_dir(reside6k, ["train/GT", "train/gt", "train/clear"]),
            "RESIDE-6K train",
        )
    if its:
        pairs += collect_pairs(
            find_dir(its, ["hazy", "Hazy", "train/hazy"]),
            find_dir(its, ["clear", "GT", "gt", "train/clear", "train/GT"]),
            "ITS indoor",
        )
    return pairs


def _split_grouped_pairs(pairs, val_fraction=0.10, seed=C.SEED):
    """Split pairs by clean-target content hash to prevent scene leakage."""
    if not pairs:
        raise ValueError("Cannot split an empty pair list")
    hazy = [pair[0] for pair in pairs]
    clean = [pair[1] for pair in pairs]
    groups = np.asarray([clean_scene_id(path) for path in clean])
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(hazy, groups=groups))
    if set(groups[train_idx]) & set(groups[val_idx]):
        raise RuntimeError("Clean-scene leakage detected between train and validation")
    return train_idx, val_idx, groups


def train(args):
    import tensorflow as tf

    set_global_seed(args.seed)
    run_dir = os.path.join(args.out, args.variant, f"seed_{args.seed}")
    model_dir = os.path.join(run_dir, "models")
    history_dir = os.path.join(run_dir, "history")
    for d in (model_dir, history_dir):
        os.makedirs(d, exist_ok=True)

    pairs = _gather_training_pairs(args.reside6k, args.its)
    if not pairs:
        raise SystemExit("No training pairs found. Check --reside6k / --its paths.")
    hazy = [p[0] for p in pairs]
    gt = [p[1] for p in pairs]
    train_idx, val_idx, groups = _split_grouped_pairs(
        pairs, val_fraction=args.val_fraction, seed=args.split_seed
    )
    tr_h, va_h = [hazy[i] for i in train_idx], [hazy[i] for i in val_idx]
    tr_g, va_g = [gt[i] for i in train_idx], [gt[i] for i in val_idx]
    train_idx_set = set(int(i) for i in train_idx)
    with open(os.path.join(run_dir, "split_manifest.csv"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["hazy_path", "clean_path", "clean_sha256", "split"])
        for i, (hp, gp, scene) in enumerate(zip(hazy, gt, groups)):
            writer.writerow([hp, gp, scene, "train" if i in train_idx_set else "validation"])
    print(f"Train: {len(tr_h)}  Val: {len(va_h)}")

    exact_resume = bool(args.resume and args.initial_epoch > 0)
    if args.resume:
        # A nonzero initial epoch means an interrupted copy of the same stage.
        # Load the compiled .keras object so AdamW iterations, slots, and its
        # warmup/cosine schedule continue. Stage transitions use initial_epoch=0
        # and intentionally receive a fresh optimizer for the next objective.
        model = load_pretrained(args.resume, compile=exact_resume)
        if exact_resume and model.optimizer is None:
            raise RuntimeError(
                "Exact resume requested, but the checkpoint has no optimizer state. "
                "Use a checkpoint saved by the revised trainer."
            )
    else:
        variant_args = {
            "full": {},
            "no_eca": {"use_eca": False},
            "pixel_only": {"use_eca": False, "use_pa": True},
            "eca_only": {"use_eca": True, "use_pa": False},
            "no_attention": {"use_eca": False, "use_pa": False},
            "no_physics": {"use_physics": False},
            "learnable_blend": {"learnable_physics_blend": True},
            "se": {"attention_variant": "se"},
            "cbam": {"attention_variant": "cbam"},
            "pa_eca": {"attention_variant": "pa_eca"},
            "parallel": {"attention_variant": "parallel"},
            "standard_unet": {"conv_type": "standard"},
            "group_unet": {"conv_type": "group"},
            "learned_residual": {"correction_mode": "learned_residual"},
        }
        model = build_eca_ldnet(
            **variant_args[args.variant],
            physics_blend=args.physics_blend,
            physics_eps=args.physics_eps,
            physics_uses_observed_input=not args.legacy_asm_input,
            width_multiplier=args.width_multiplier,
        )
        if args.resume_weights:
            model.load_weights(args.resume_weights)
    print(f"ECA-LDNet: {model.count_params():,} params")

    stage_cfgs = {1: C.STAGE1, 2: C.STAGE2, 3: C.STAGE3, 4: C.STAGE4}
    if args.provisional_stage5:
        # The original Stage 5 recipe is absent from the repository. This
        # opt-in repeats the recovered Stage 4 recipe and must be reported as
        # a new continuation, never as a faithful reproduction.
        stage_cfgs[5] = C.STAGE5
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.stage_batch_sizes is not None and any(size < 1 for size in args.stage_batch_sizes):
        raise SystemExit("Every value in --stage-batch-sizes must be positive")
    if args.batch_size is not None and args.stage_batch_sizes is not None:
        raise SystemExit("Use either --batch-size or --stage-batch-sizes, not both")
    batch_schedule = list(args.stage_batch_sizes or [
        C.STAGE1.batch_size, C.STAGE2.batch_size, C.STAGE3.batch_size, C.STAGE4.batch_size
    ])
    identity = {
        "variant": args.variant,
        "seed": args.seed,
        "reside6k_root": os.path.abspath(args.reside6k) if args.reside6k else None,
        "its_root": os.path.abspath(args.its) if args.its else None,
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "stage_batch_sizes": batch_schedule,
        "preprocessing": args.preprocessing,
        "physics_blend": args.physics_blend,
        "physics_eps": args.physics_eps,
        "perceptual_protocol": args.perceptual_protocol,
        "legacy_asm_input": args.legacy_asm_input,
        "width_multiplier": args.width_multiplier,
        "disable_ema": args.disable_ema,
        "disable_polish_loss": args.disable_polish_loss,
        "disable_late_lr": args.disable_late_lr,
        "loss_ablation": args.loss_ablation,
    }
    config_path = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as stream:
            run_config = json.load(stream)
        if run_config.get("identity") != identity:
            raise RuntimeError(
                f"Run settings differ from the saved {config_path}; use a new --out folder "
                "instead of mixing runs."
            )
    else:
        run_config = {"identity": identity, "stages": {}}
    with open(config_path, "w", encoding="utf-8") as stream:
        json.dump(run_config, stream, indent=2)
    for s in args.stages:
        if s not in stage_cfgs:
            if s == 5:
                raise SystemExit("Stage 5 source recipe is unresolved; pass --provisional-stage5 only to repeat Stage 4 as a new continuation.")
            raise SystemExit(f"Unsupported stage: {s}. Choose from 1, 2, 3, 4.")
        cfg = stage_cfgs[s]
        if args.stage_batch_sizes is not None:
            cfg = replace(cfg, batch_size=args.stage_batch_sizes[s - 1])
        elif args.batch_size is not None:
            cfg = replace(cfg, batch_size=args.batch_size)
        if s >= 4 and args.disable_late_lr:
            cfg = replace(cfg, lr=C.LR_S3)
        if args.max_epochs:
            cfg = replace(
                cfg,
                epochs=args.max_epochs,
                warmup_epochs=min(cfg.warmup_epochs, max(0, args.max_epochs - 1)),
            )
        checkpoint_stem = (
            f"seed{args.seed}_{args.variant}_stage{s}"
            if args.variant != "full" else f"stage{s}"
        )
        stage_record = {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.lr,
            "warmup_epochs": cfg.warmup_epochs,
        }
        previous_record = run_config["stages"].get(str(s))
        if previous_record is not None and previous_record != stage_record:
            raise RuntimeError(
                f"Stage {s} settings differ from saved {config_path}; use a new --out folder "
                "to prevent mixing or duplicating experiments."
            )
        run_config["stages"][str(s)] = stage_record
        with open(config_path, "w", encoding="utf-8") as stream:
            json.dump(run_config, stream, indent=2)
        latest_path = os.path.join(model_dir, f"{checkpoint_stem}_latest.keras")
        exact_same_stage_resume = bool(
            args.initial_epoch > 0 and args.resume
            and os.path.isfile(latest_path)
            and os.path.realpath(args.resume) == os.path.realpath(latest_path)
        )
        if os.path.isfile(latest_path) and not exact_same_stage_resume:
            raise RuntimeError(
                f"{latest_path} already exists. This stage has already started; "
                "use its latest checkpoint with --initial-epoch for same-stage recovery, "
                "or choose a new --out folder for an independent run."
            )
        if args.initial_epoch > 0 and not exact_same_stage_resume:
            raise RuntimeError(
                "Same-stage recovery requires --resume to point to this stage's "
                f"latest checkpoint: {latest_path}"
            )
        if s <= 2:
            phase = "primary"
        elif s == 3 or args.disable_polish_loss:
            phase = "stage3"
        else:
            phase = "stage4"
        excluded = (
            [args.loss_ablation]
            if args.loss_ablation in LOSS_PHASE_WEIGHTS[phase] else []
        )
        loss_fn = make_loss(phase, exclude=excluded, perceptual_protocol=args.perceptual_protocol)
        train_ds = build_dataset(
            tr_h, tr_g, augment=True, batch=cfg.batch_size, seed=args.seed,
            preprocessing=args.preprocessing,
        )
        val_ds = build_dataset(
            va_h, va_g, augment=False, batch=cfg.batch_size, shuffle=False,
            cache_ds=True, seed=args.seed, preprocessing=args.preprocessing,
        )
        steps = max(1, len(tr_h) // cfg.batch_size)
        lr_sched = WarmupCosineDecay(cfg.lr, cfg.warmup_epochs * steps, cfg.epochs * steps)
        weight_decay = {1: 1e-4, 2: 5e-5, 3: 2.5e-5, 4: 1e-5, 5: 1e-5}[s]
        clipnorm = 0.5 if s == 3 else 1.0
        if not (exact_resume and model.optimizer is not None):
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(
                    learning_rate=lr_sched, weight_decay=weight_decay, clipnorm=clipnorm
                ),
                loss=loss_fn,
                metrics=[psnr_metric, ssim_metric, mae_metric],
            )
        else:
            print(
                "Exact same-stage resume: preserving checkpoint optimizer, "
                "iterations, and learning-rate schedule."
            )
        print(f"\n=== Stage {s} | LR={cfg.lr} | {cfg.epochs} epochs | batch {cfg.batch_size} ===")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs,
            callbacks=make_callbacks(
                s, model_dir, history_dir,
                patience=8 if s >= 4 else C.PATIENCE_ES,
                use_ema=s >= 4 and not args.disable_ema,
                initial_epoch=args.initial_epoch,
                early_stopping=not args.disable_early_stopping,
                save_every_epoch=args.save_every_epoch,
                checkpoint_prefix=checkpoint_stem,
            ),
            verbose=1,
            initial_epoch=args.initial_epoch,
        )
        # On a resumed stage, EarlyStopping only observes the current process.
        # The stage*_best checkpoint and callback threshold span prior sessions,
        # so restore that globally best model before entering the next stage or
        # writing eca_ldnet_final.keras.
        stage_best_path = os.path.join(model_dir, f"{checkpoint_stem}_best.keras")
        if os.path.isfile(stage_best_path):
            best_model = load_pretrained(stage_best_path, compile=False)
            model.set_weights(best_model.get_weights())
            del best_model
            print(f"Restored best Stage {s} weights from {stage_best_path}")
    model.save(os.path.join(model_dir, "eca_ldnet_final.keras"))
    print(f"Saved final model to {model_dir}/eca_ldnet_final.keras")
    if "physics_output" in {layer.name for layer in model.layers}:
        blend_layer = model.get_layer("physics_output")
        print(f"Final ASM blend: {float(tf.keras.backend.get_value(blend_layer.current_blend)):.6f}")


def main():
    p = argparse.ArgumentParser(description="Train ECA-LDNet")
    p.add_argument("--reside6k", default=None)
    p.add_argument("--its", default=None)
    p.add_argument("--out", default="./runs")
    p.add_argument("--resume", default=None, help="Resume from a .keras checkpoint")
    p.add_argument("--resume-weights", default=None, help="Initialize the selected graph from compatible checkpoint weights")
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--provisional-stage5", action="store_true",
                   help="Opt into Stage 5 by repeating the recovered Stage 4 recipe; not a faithful S5 reproduction")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--split-seed", type=int, default=C.SEED, help="Fixed seed for grouped train/validation split")
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override stage batch size after the T4 memory pilot")
    p.add_argument("--stage-batch-sizes", type=int, nargs=4, default=None,
                   metavar=("S1", "S2", "S3", "S4"),
                   help="Set batch size separately for stages 1–4, e.g. 8 16 8 16")
    p.add_argument("--preprocessing", choices=["radiometric", "submitted"], default="radiometric")
    p.add_argument("--physics-blend", type=float, default=C.PHYS_BLEND)
    p.add_argument("--physics-eps", type=float, default=C.PHYS_EPS)
    p.add_argument("--width-multiplier", type=float, default=1.0, help="Scale backbone width for capacity-matched controls")
    p.add_argument("--legacy-asm-input", action="store_true", help="Reproduce checkpoint correction applied to raw prediction")
    p.add_argument("--max-epochs", type=int, default=None, help="Cap each selected stage (useful for timing pilots)")
    p.add_argument(
        "--initial-epoch", type=int, default=0,
        help="Resume a selected stage at this zero-based epoch; use with --resume and the matching latest checkpoint",
    )
    p.add_argument("--loss-ablation", choices=["none", "mse", "charbonnier", "ssim", "ms_ssim", "edge", "frequency", "perceptual", "vgg19_perceptual", "color_consistency", "gradient"], default="none")
    p.add_argument("--perceptual-protocol", choices=["legacy", "imagenet"], default="imagenet",
                   help="Legacy reproduces the submitted VGG16 input mapping; ImageNet uses canonical VGG preprocessing")
    p.add_argument("--disable-ema", action="store_true", help="Factorial ablation: disable EMA in stages 4-5")
    p.add_argument("--disable-early-stopping", action="store_true",
                   help="Run every requested epoch; useful when completing a fixed stage budget")
    p.add_argument("--disable-polish-loss", action="store_true", help="Factorial ablation: keep primary loss in stages 4-5")
    p.add_argument("--disable-late-lr", action="store_true", help="Factorial ablation: use stage-3 LR for stages 4-5")
    p.add_argument("--save-every-epoch", action="store_true",
                   help="Also save numbered stageN_epochNNN.keras checkpoints after every epoch")
    p.add_argument(
        "--variant",
        choices=["full", "no_eca", "pixel_only", "eca_only", "no_attention", "no_physics", "learnable_blend", "se", "cbam", "pa_eca", "parallel", "standard_unet", "group_unet", "learned_residual"],
        default="full",
    )
    train(p.parse_args())


if __name__ == "__main__":
    main()
