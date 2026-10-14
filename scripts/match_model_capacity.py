"""Find width multipliers that capacity-match reviewer-requested controls."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eca_ldnet.model import build_eca_ldnet


def variants():
    return {
        "full_dws_dual_asm": {},
        "dws_se": {"attention_variant": "se"},
        "dws_cbam": {"attention_variant": "cbam"},
        "dws_pa_eca": {"attention_variant": "pa_eca"},
        "dws_parallel": {"attention_variant": "parallel"},
        "dws_eca": {"use_eca": True, "use_pa": False},
        "dws_pixel": {"use_eca": False, "use_pa": True},
        "dws_spatial": {"attention_variant": "pixel"},
        "standard_unet": {"conv_type": "standard"},
        "group_unet": {"conv_type": "group"},
        "learned_residual": {"correction_mode": "learned_residual"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-params", type=int, default=None)
    parser.add_argument("--min-width", type=float, default=0.4)
    parser.add_argument("--max-width", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--output", default="./model_capacity_matches.csv")
    args = parser.parse_args()

    target = args.target_params or build_eca_ldnet().count_params()
    rows = []
    for name, kwargs in variants().items():
        best = None
        for width in [args.min_width + i * (args.max_width - args.min_width) / (args.steps - 1)
                      for i in range(args.steps)]:
            model = build_eca_ldnet(width_multiplier=width, **kwargs)
            params = model.count_params()
            candidate = (abs(params - target), width, params)
            if best is None or candidate < best:
                best = candidate
            # Releasing each graph avoids retaining hundreds of Keras models
            # in GPU/host memory during this architecture-only search.
            import tensorflow as tf
            del model
            tf.keras.backend.clear_session()
        rows.append({
            "variant": name,
            "target_params": target,
            "width_multiplier": best[1],
            "params": best[2],
            "relative_error": best[0] / target,
        })
        print(f"{name:<22} width={best[1]:.4f} params={best[2]:,} error={best[0]/target:.2%}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
