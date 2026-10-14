"""Summarize and plot the four training-stage histories for seeds 17/43/73."""
from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--training-root", default=str(Path(__file__).resolve().parents[1] / "full_model_training"))
parser.add_argument("--output", default=None)
args = parser.parse_args()
ROOT = Path(args.training_root)
OUT = Path(args.output) if args.output else ROOT / "training_history_report"
OUT.mkdir(parents=True, exist_ok=True)
STAGES = (1, 2, 3, 4)
SEEDS = (17, 43, 73)
FIELDS = ["seed", "stage", "history_rows", "best_val_psnr", "best_val_psnr_epoch",
          "best_val_ssim", "best_val_ssim_epoch", "final_lr"]
summary = []

for seed in SEEDS:
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"ECA-LDNet training history — seed {seed}\n"
                 "Stage loss scales are not directly comparable because objectives change by stage")
    for col, stage in enumerate(STAGES):
        path = ROOT / f"seed{seed}" / f"stage{stage}" / "history.csv"
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise RuntimeError(f"Empty history: {path}")
        epochs = [int(r["epoch"]) + 1 for r in rows]
        for key, ax, title in (
            ("loss", axes[0, col], f"Stage {stage}: objective loss"),
            ("psnr_metric", axes[1, col], f"Stage {stage}: PSNR (dB)"),
            ("ssim_metric", axes[2, col], f"Stage {stage}: SSIM"),
        ):
            train_key, val_key = key, f"val_{key}"
            ax.plot(epochs, [float(r[train_key]) for r in rows], label="train", lw=1.5)
            ax.plot(epochs, [float(r[val_key]) for r in rows], label="validation", lw=1.5)
            ax.set_title(title)
            ax.set_xlabel("epoch within stage")
            ax.grid(alpha=.25)
            if col == 0:
                ax.legend(fontsize=8)
        psnr_idx = max(range(len(rows)), key=lambda i: float(rows[i]["val_psnr_metric"]))
        ssim_idx = max(range(len(rows)), key=lambda i: float(rows[i]["val_ssim_metric"]))
        summary.append({
            "seed": seed, "stage": stage, "history_rows": len(rows),
            "best_val_psnr": rows[psnr_idx]["val_psnr_metric"],
            "best_val_psnr_epoch": epochs[psnr_idx],
            "best_val_ssim": rows[ssim_idx]["val_ssim_metric"],
            "best_val_ssim_epoch": epochs[ssim_idx],
            "final_lr": rows[-1]["lr"],
        })
    fig.savefig(OUT / f"seed{seed}_stage_histories.png", dpi=180)
    plt.close(fig)

with (OUT / "stage_history_summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(summary)
print(f"Wrote plots and history summary under {OUT}")
