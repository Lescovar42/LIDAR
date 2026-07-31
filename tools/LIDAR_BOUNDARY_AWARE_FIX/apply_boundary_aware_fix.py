#!/usr/bin/env python3
"""Install the boundary-aware manifest tool and add --manifest to training."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

START_MARKER = "# BEGIN BOUNDARY-AWARE MANIFEST SUPPORT"
END_MARKER = "# END BOUNDARY-AWARE MANIFEST SUPPORT"


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Cannot apply {description}: expected one exact match, found {count}. "
            "The repository may have changed; no partial edit was written."
        )
    return text.replace(old, new, 1)


def patch_train_baseline(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    if START_MARKER in original:
        return False

    parser_old = '    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_pilot"))\n'
    parser_new = parser_old + f'''    {START_MARKER}\n    parser.add_argument(\n        "--manifest",\n        type=Path,\n        default=None,\n        help=(\n            "Optional patch manifest. Relative paths are resolved inside --dataset-dir. "\n            "Use patches_boundary_aware.csv for training-only interior capping."\n        ),\n    )\n    {END_MARKER}\n'''
    patched = replace_once(original, parser_old, parser_new, "--manifest parser argument")

    manifest_old = '''    qc_manifest_path = dataset_dir / "patches_qc.csv"\n    manifest_path = qc_manifest_path if qc_manifest_path.exists() else dataset_dir / "patches.csv"\n'''
    manifest_new = f'''    {START_MARKER}\n    if args.manifest is not None:\n        manifest_path = args.manifest\n        if not manifest_path.is_absolute():\n            manifest_path = dataset_dir / manifest_path\n        manifest_path = manifest_path.resolve()\n    else:\n        boundary_manifest_path = dataset_dir / "patches_boundary_aware.csv"\n        qc_manifest_path = dataset_dir / "patches_qc.csv"\n        if boundary_manifest_path.exists():\n            manifest_path = boundary_manifest_path\n        else:\n            manifest_path = qc_manifest_path if qc_manifest_path.exists() else dataset_dir / "patches.csv"\n    {END_MARKER}\n'''
    patched = replace_once(patched, manifest_old, manifest_new, "manifest resolution")

    rows_old = "    rows = read_manifest(manifest_path)\n"
    rows_new = rows_old + '    print(f"Manifest: {manifest_path}")\n'
    patched = replace_once(patched, rows_old, rows_new, "manifest reporting")

    backup = path.with_suffix(path.suffix + ".before_boundary_aware_fix")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    oregon = repo_root / "oregon"
    if not oregon.is_dir() and (repo_root / "train_baseline.py").exists():
        # Also support extraction directly inside F:\\LIDAR\\oregon.
        oregon = repo_root
    train = oregon / "train_baseline.py"
    if not train.exists():
        parser.error(f"Could not find train_baseline.py under {oregon}")

    source_dir = Path(__file__).resolve().parent
    tool_source = source_dir / "prepare_boundary_aware_manifest.py"
    if not tool_source.exists():
        parser.error(f"Package is incomplete: {tool_source.name} is missing")
    destination = oregon / "diagnostics" / "prepare_boundary_aware_manifest.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tool_source, destination)

    changed = patch_train_baseline(train)
    print(f"Installed: {destination}")
    print("Patched train_baseline.py" if changed else "train_baseline.py was already patched")
    print("\nNext:")
    print("  cd <repo>\\oregon")
    print("  python .\\diagnostics\\prepare_boundary_aware_manifest.py --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
