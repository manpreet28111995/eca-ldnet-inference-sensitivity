"""Package one completed seed/variant stage, its history, split and settings."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


EPOCHS = {1: 18, 2: 8, 3: 6, 4: 8}
VARIANTS = {"no_eca", "eca_only", "pixel_only", "no_attention"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("ablation_runs"))
    parser.add_argument("--save-root", type=Path, default=Path("save_here"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--stage", type=int, choices=sorted(EPOCHS), required=True)
    args = parser.parse_args()

    run_dir = args.out_root / args.variant / f"seed_{args.seed}"
    stem = f"seed{args.seed}_{args.variant}_stage{args.stage}"
    required = [
        run_dir / "run_config.json",
        run_dir / "split_manifest.csv",
        run_dir / "models" / "eca_ldnet_final.keras",
        run_dir / "models" / f"{stem}_best.keras",
        run_dir / "models" / f"{stem}_latest.keras",
        run_dir / "models" / f"{stem}_epoch{EPOCHS[args.stage]:03d}.keras",
        run_dir / "history" / f"{stem}_history.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Stage is not complete; missing required files:\n  " + "\n  ".join(missing))

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    identity = config.get("identity", {})
    if identity.get("seed") != args.seed or identity.get("variant") != args.variant:
        raise SystemExit("Saved run_config.json seed/variant does not match this ZIP request.")
    stage_config = config.get("stages", {}).get(str(args.stage), {})
    if stage_config.get("epochs") != EPOCHS[args.stage]:
        raise SystemExit("Saved stage epoch count does not match the 40-epoch guide.")

    archive_dir = args.save_root / f"seed{args.seed}_{args.variant}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{stem}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
            archive_name = (Path(args.variant) / f"seed_{args.seed}" / path.relative_to(run_dir)).as_posix()
            archive.write(path, archive_name)
    print(f"Stage {args.stage} archive saved: {archive_path.resolve()}")
    print(f"Contains only this run: {args.variant}/seed_{args.seed}/")


if __name__ == "__main__":
    main()
