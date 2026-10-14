"""Check that local RESIDE-6K and ITS training pairs are discoverable."""
from __future__ import annotations

import argparse
from pathlib import Path

from eca_ldnet.data import collect_pairs, find_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reside6k", type=Path, default=Path("datasets/RESIDE-6K"))
    parser.add_argument("--its", type=Path, default=Path("datasets/ITS"))
    args = parser.parse_args()

    reside_hazy = find_dir(str(args.reside6k), ["train/hazy", "Train/hazy"])
    reside_gt = find_dir(str(args.reside6k), ["train/GT", "train/gt", "train/clear"])
    its_hazy = find_dir(str(args.its), ["hazy", "Hazy", "train/hazy"])
    its_gt = find_dir(str(args.its), ["clear", "GT", "gt", "train/clear", "train/GT"])
    if not all((reside_hazy, reside_gt, its_hazy, its_gt)):
        raise SystemExit(
            "Could not find all expected dataset folders. See datasets/README.md.\n"
            f"RESIDE hazy={reside_hazy}, GT={reside_gt}; ITS hazy={its_hazy}, clear={its_gt}"
        )
    reside_pairs = collect_pairs(reside_hazy, reside_gt, "RESIDE-6K local check")
    its_pairs = collect_pairs(its_hazy, its_gt, "ITS local check")
    if not reside_pairs or not its_pairs:
        raise SystemExit("At least one dataset has zero matched image pairs; check filenames and folders.")
    print(f"Dataset check passed: RESIDE-6K={len(reside_pairs)}, ITS={len(its_pairs)} matched pairs.")


if __name__ == "__main__":
    main()
