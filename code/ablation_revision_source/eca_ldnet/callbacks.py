"""Learning-rate schedule and training callbacks."""
from __future__ import annotations

import math
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks

from .config import MIN_LR


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to ``min_lr``."""

    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr=MIN_LR):
        self.peak_lr = float(peak_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step / (self.warmup_steps + 1e-8))
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps + 1e-8)
        cos_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * tf.clip_by_value(progress, 0.0, 1.0))
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cos_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


class CSVMetricsLogger(callbacks.Callback):
    """Append per-epoch metrics (incl. current LR) to a CSV file."""

    def __init__(self, path, stage, initial_epoch=0):
        super().__init__()
        self.path = path
        self.stage = stage
        self.records = []
        self.epoch_offset = 0
        self.initial_epoch = int(initial_epoch)

    def on_train_begin(self, logs=None):
        if os.path.isfile(self.path):
            try:
                self.records = pd.read_csv(self.path).to_dict("records")
                previous = [int(r["epoch"]) for r in self.records
                            if int(r.get("stage", -1)) == self.stage]
                # Keras reports the global epoch index when initial_epoch is used.
                # Align that index with the existing CSV instead of duplicating it.
                self.epoch_offset = max(previous, default=-1) + 1 - self.initial_epoch
            except (ValueError, KeyError, pd.errors.EmptyDataError):
                self.records = []
                self.epoch_offset = 0

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": self.epoch_offset + epoch, "stage": self.stage}
        row.update(logs or {})
        try:
            opt = self.model.optimizer
            lr = opt.learning_rate
            row["lr"] = float(lr(opt.iterations) if callable(lr) else lr)
        except Exception:
            row["lr"] = 0.0
        self.records.append(row)
        pd.DataFrame(self.records).to_csv(self.path, index=False)


class ExponentialMovingAverage(callbacks.Callback):
    """Track EMA weights and evaluate/checkpoint with EMA weights per epoch."""

    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = float(decay)
        self.shadow = None
        self.backup = None
        self.model_weights = None

    def on_train_begin(self, logs=None):
        self.model_weights = list(self.model.weights)
        self.shadow = [tf.Variable(w, trainable=False) for w in self.model_weights]
        self.backup = None

    def on_train_batch_end(self, batch, logs=None):
        for avg, current in zip(self.shadow, self.model_weights):
            avg.assign_sub((1.0 - self.decay) * (avg - tf.cast(current, avg.dtype)))

    def on_test_begin(self, logs=None):
        # Keras invokes validation before epoch-end callbacks such as checkpointing.
        # Keep EMA weights active through on_epoch_end so validation and checkpoint
        # selection refer to the same averaged model.
        if self.backup is None:
            self.backup = [tf.identity(v) for v in self.model_weights]
            for variable, averaged in zip(self.model_weights, self.shadow):
                variable.assign(tf.cast(averaged, variable.dtype))

    def on_epoch_begin(self, epoch, logs=None):
        if self.backup is not None:
            for variable, original in zip(self.model_weights, self.backup):
                variable.assign(original)
            self.backup = None

    def on_train_end(self, logs=None):
        # Leave EMA weights installed for saving the final EMA checkpoint.
        if self.backup is not None:
            for variable, averaged in zip(self.model_weights, self.shadow):
                variable.assign(tf.cast(averaged, variable.dtype))
            self.backup = None


def make_callbacks(stage, model_dir, history_dir, monitor="val_psnr_metric", patience=15,
                   use_ema=False, initial_epoch=0, early_stopping=True,
                   save_every_epoch=False, checkpoint_prefix=None):
    """Standard checkpoint + early-stopping + CSV-logging callback set."""
    # When a stage resumes in a later process, keep the best-model threshold
    # from its existing history. Otherwise ModelCheckpoint would treat the
    # first resumed epoch as a new run and could overwrite a better checkpoint
    # from before the interruption.
    checkpoint_stem = checkpoint_prefix or f"stage{stage}"
    history_path = f"{history_dir}/{checkpoint_stem}_history.csv"
    previous_best = None
    if initial_epoch > 0 and os.path.isfile(history_path):
        try:
            history = pd.read_csv(history_path)
            stage_rows = history[pd.to_numeric(history["stage"], errors="coerce") == stage]
            values = pd.to_numeric(stage_rows[monitor], errors="coerce").dropna()
            if not values.empty:
                previous_best = float(values.max())
        except (KeyError, ValueError, pd.errors.EmptyDataError):
            previous_best = None

    items = []
    if use_ema:
        from .config import EMA_DECAY
        items.append(ExponentialMovingAverage(EMA_DECAY))
    items.extend([
        # Keep an epoch-level recovery point. Kaggle sessions can stop before a
        # stage completes; this checkpoint is for recovery, while *_best.keras
        # remains the validation-selected model used for reporting.
        callbacks.ModelCheckpoint(
            f"{model_dir}/{checkpoint_stem}_latest.keras",
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),
        EveryNEpochsCheckpoint(
            model_dir=model_dir,
            checkpoint_stem=checkpoint_stem,
            every=1 if save_every_epoch else 10,
        ),
        callbacks.ModelCheckpoint(
            f"{model_dir}/{checkpoint_stem}_best.keras",
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
            initial_value_threshold=previous_best,
        ),
    ])
    if early_stopping:
        items.append(callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=patience,
            baseline=previous_best,
            restore_best_weights=True,
            verbose=1,
        ))
    items.append(CSVMetricsLogger(
        history_path, stage,
        initial_epoch=initial_epoch,
    ))
    return items


class EveryNEpochsCheckpoint(tf.keras.callbacks.Callback):
    """Save a complete Keras model at fixed epoch boundaries for Kaggle recovery."""

    def __init__(self, model_dir, checkpoint_stem, every=10):
        super().__init__()
        self.model_dir = model_dir
        self.checkpoint_stem = checkpoint_stem
        self.every = every

    def on_epoch_end(self, epoch, logs=None):
        completed = epoch + 1
        if completed % self.every == 0:
            path = os.path.join(
                self.model_dir, f"{self.checkpoint_stem}_epoch{completed:03d}.keras"
            )
            self.model.save(path)
            print(f"Saved periodic checkpoint: {path}")
