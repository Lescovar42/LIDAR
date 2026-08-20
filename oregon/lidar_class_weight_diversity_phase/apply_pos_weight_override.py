#!/usr/bin/env python3
"""Patch the current oregon/train_baseline.py with a controlled --pos-weight override.

The patch is intentionally narrow: architecture, optimizer, normalization, split
selection, checkpoint-selection threshold, and loss composition remain unchanged.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (
'''def compute_pos_weight(dataset_dir: Path, rows: list[dict[str, str]], cap: float = 50.0) -> float:
    positive = 0
    negative = 0
    for row in rows:
        with np.load(dataset_dir / row["patch_path"]) as data:
            mask = data["mask"]
        positive += int((mask == 1).sum())
        negative += int((mask == 0).sum())
    if positive == 0:
        raise ValueError("Training split contains zero positive, non-ignored pixels")
    return float(min(cap, max(1.0, negative / positive)))
''',
'''def compute_pos_weight(dataset_dir: Path, rows: list[dict[str, str]], cap: float = 50.0) -> float:
    positive = 0
    negative = 0
    for row in rows:
        with np.load(dataset_dir / row["patch_path"]) as data:
            mask = data["mask"]
        positive += int((mask == 1).sum())
        negative += int((mask == 0).sum())
    if positive == 0:
        raise ValueError("Training split contains zero positive, non-ignored pixels")
    return float(min(cap, max(1.0, negative / positive)))


def resolve_pos_weight(
    dataset_dir: Path,
    rows: list[dict[str, str]],
    requested: str,
) -> tuple[float, float, str]:
    """Return (used_weight, auto_weight, mode) for reproducible ablations."""
    auto_weight = compute_pos_weight(dataset_dir, rows)
    normalized = requested.strip().casefold()
    if normalized == "auto":
        return auto_weight, auto_weight, "auto"
    try:
        fixed_weight = float(requested)
    except ValueError as exc:
        raise ValueError("--pos-weight must be 'auto' or a positive number") from exc
    if not np.isfinite(fixed_weight) or fixed_weight <= 0:
        raise ValueError("--pos-weight must be a finite number greater than zero")
    return fixed_weight, auto_weight, "fixed"
'''
    ),
    (
'''    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
''',
'''    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pos-weight",
        default="auto",
        help=(
            "BCE positive-class weight: 'auto' uses the training-pixel negative/positive ratio; "
            "a positive number (for example 1.0) applies a fixed controlled override."
        ),
    )
'''
    ),
    (
'''    pos_weight = compute_pos_weight(dataset_dir, train_rows)
    print(f"Positive-class weight (ignored pixels excluded): {pos_weight:.3f}")
''',
'''    try:
        pos_weight, auto_pos_weight, pos_weight_mode = resolve_pos_weight(
            dataset_dir, train_rows, args.pos_weight
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        "Positive-class weight (ignored pixels excluded): "
        f"used={pos_weight:.6g}, auto={auto_pos_weight:.6g}, mode={pos_weight_mode}"
    )
    run_config = {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "outdir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "training_region": training_region,
        "require_qc": args.require_qc,
        "device_request": args.device,
        "pos_weight_request": args.pos_weight,
        "pos_weight_mode": pos_weight_mode,
        "pos_weight_used": pos_weight,
        "auto_pos_weight": auto_pos_weight,
        "train_patch_ids": [row["patch_id"] for row in train_rows],
        "validation_patch_ids": [row["patch_id"] for row in validation_rows],
        "test_patch_ids": [row["patch_id"] for row in test_rows],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
'''
    ),
    (
'''                    "validation_metrics": validation_metrics,
                    "training_regions": training_regions,
''',
'''                    "validation_metrics": validation_metrics,
                    "training_regions": training_regions,
                    "pos_weight_mode": pos_weight_mode,
                    "pos_weight_used": pos_weight,
                    "auto_pos_weight": auto_pos_weight,
                    "run_config": run_config,
'''
    ),
    (
'''    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"Training on {device} ({device_name})")
''',
'''    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    run_config.update(
        {
            "device_used": str(device),
            "device_name": device_name,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(f"Training on {device} ({device_name})")
'''
    ),
    (
'''        "excluded_validation_regions": excluded_validation_regions,
        "require_qc": args.require_qc,
''',
'''        "excluded_validation_regions": excluded_validation_regions,
        "require_qc": args.require_qc,
        "pos_weight_mode": pos_weight_mode,
        "pos_weight_used": pos_weight,
        "auto_pos_weight": auto_pos_weight,
        "run_config_path": str(output_dir / "run_config.json"),
'''
    ),
)


def patch_text(source: str) -> str:
    if 'parser.add_argument(\n        "--pos-weight"' in source and "def resolve_pos_weight(" in source:
        raise RuntimeError("The --pos-weight override already appears to be integrated")
    patched = source
    for old, new in REPLACEMENTS:
        occurrences = patched.count(old)
        if occurrences != 1:
            raise RuntimeError(
                f"Expected exactly one current-repository match, found {occurrences}. "
                "The upstream file may have changed; inspect before applying."
            )
        patched = patched.replace(old, new, 1)
    return patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("train_baseline.py"),
        help="Path to the repository's oregon/train_baseline.py",
    )
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    path = args.file.resolve()
    if not path.exists():
        parser.error(f"File not found: {path}")
    original = path.read_text(encoding="utf-8")
    try:
        patched = patch_text(original)
    except RuntimeError as exc:
        parser.error(str(exc))
    if not args.no_backup:
        backup = path.with_suffix(path.suffix + ".pre_pos_weight.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"Backup: {backup}")
    path.write_text(patched, encoding="utf-8")
    print(f"Patched: {path}")
    print("Added --pos-weight auto|<positive number> and run_config/checkpoint provenance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
