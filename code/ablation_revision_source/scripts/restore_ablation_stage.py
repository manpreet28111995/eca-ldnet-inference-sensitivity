"""Restore one exact previous-stage ZIP into its seed/variant run directory."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath


VARIANTS = {"no_eca", "eca_only", "pixel_only", "no_attention"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("ablation_runs"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True,
                        help="Previous stage contained in the archive")
    args = parser.parse_args()

    expected_name = f"seed{args.seed}_{args.variant}_stage{args.stage}.zip"
    if args.archive.name != expected_name:
        raise SystemExit(f"Expected archive named {expected_name}; got {args.archive.name}.")
    run_dir = args.out_root / args.variant / f"seed_{args.seed}"
    checkpoint = run_dir / "models" / "eca_ldnet_final.keras"
    if not checkpoint.is_file():
        if not args.archive.is_file():
            raise SystemExit(f"Previous-stage ZIP not found: {args.archive}")
        expected_prefix = f"{args.variant}/seed_{args.seed}/"
        with zipfile.ZipFile(args.archive) as archive:
            for name in archive.namelist():
                member = PurePosixPath(name)
                if member.is_absolute() or ".." in member.parts or not name.startswith(expected_prefix):
                    raise SystemExit(f"Unexpected path in previous-stage ZIP: {name}")
            archive.extractall(args.out_root)

    if not checkpoint.is_file():
        raise SystemExit(f"Restore did not produce the expected handoff checkpoint: {checkpoint}")
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise SystemExit(f"Restored run configuration is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = config.get("identity", {})
    if identity.get("seed") != args.seed or identity.get("variant") != args.variant:
        raise SystemExit("Restored run_config.json does not match the requested seed and ablation.")
    if str(args.stage) not in config.get("stages", {}):
        raise SystemExit(f"Restored ZIP does not record completed Stage {args.stage}.")
    print(f"Previous Stage {args.stage} model restored successfully; ready to run Stage {args.stage + 1}:")
    print(checkpoint.resolve())


if __name__ == "__main__":
    main()
