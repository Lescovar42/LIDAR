#!/usr/bin/env python3
"""
Materialize the frozen Tillamook 64/16/20 split.

Source:
  dataset_tillamook_expansion_100_train_binary_15m/
    patches.csv
    patches/.../*.npz

Frozen split:
  phase1_tillamook_split_64_16_20/preview_manifest.csv
  phase1_tillamook_split_64_16_20/split_map_64_16_20.csv
  phase1_tillamook_split_64_16_20/split_verification.json

Output:
  dataset_tillamook_100_binary_15m_split641620/
    patches.csv
    channels.json
    failed_tiles.csv
    dataset_summary.json
    split_map_64_16_20.csv
    split_verification.json
    split_fingerprints.json
    TEST_SET_LOCKED.txt
    patches/train/*.npz
    patches/validation/*.npz
    patches/test/*.npz

This script does NOT:
- change masks
- resample terrain
- recompute features
- regenerate the split
- touch external regions
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VALID_SPLITS = ("train", "validation", "test")


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def hash_lines(values):
    payload = "\n".join(sorted(str(v) for v in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size != src.stat().st_size:
            raise RuntimeError(f"Existing destination has wrong size: {dst}")
        return "exists"
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def find_source_patch(source_dataset: Path, row: dict[str, str]) -> Path:
    rel = Path(row["patch_path"])
    candidate = source_dataset / rel
    if candidate.exists():
        return candidate

    # Common fallback when the original dataset had train-only storage.
    patch_id = row["patch_id"]
    matches = list(source_dataset.rglob(f"{patch_id}.npz"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find NPZ for patch_id={patch_id}, original patch_path={rel}"
        )
    raise RuntimeError(
        f"Multiple NPZ candidates for patch_id={patch_id}: {matches}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("dataset_tillamook_expansion_100_train_binary_15m"),
    )
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=Path("phase1_tillamook_split_64_16_20"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("dataset_tillamook_100_binary_15m_split641620"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source_dataset.resolve()
    phase1 = args.phase1_dir.resolve()
    outdir = args.outdir.resolve()

    source_manifest = source / "patches.csv"
    preview_manifest = phase1 / "preview_manifest.csv"
    split_map = phase1 / "split_map_64_16_20.csv"
    split_verification = phase1 / "split_verification.json"

    for required in (source_manifest, preview_manifest, split_map, split_verification):
        if not required.exists():
            raise SystemExit(f"Required input missing: {required}")

    if outdir.exists() and any(outdir.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output directory is not empty: {outdir}\n"
                "Use --overwrite only if you intentionally want to rebuild the materialized dataset."
            )
        shutil.rmtree(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    source_rows, source_fields = read_csv(source_manifest)
    preview_rows, preview_fields = read_csv(preview_manifest)

    if len(source_rows) != len(preview_rows):
        raise RuntimeError(
            f"Manifest row-count mismatch: source={len(source_rows)}, preview={len(preview_rows)}"
        )

    if not source_rows:
        raise RuntimeError("Source manifest is empty")

    source_by_id = {row["patch_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise RuntimeError("Duplicate patch_id values in source manifest")

    preview_ids = [row["patch_id"] for row in preview_rows]
    if len(set(preview_ids)) != len(preview_ids):
        raise RuntimeError("Duplicate patch_id values in preview manifest")

    if set(preview_ids) != set(source_by_id):
        missing_from_preview = sorted(set(source_by_id) - set(preview_ids))
        extra_in_preview = sorted(set(preview_ids) - set(source_by_id))
        raise RuntimeError(
            "Preview/source patch identity mismatch.\n"
            f"Missing from preview: {missing_from_preview[:10]}\n"
            f"Extra in preview: {extra_in_preview[:10]}"
        )

    split_values = {row["split"] for row in preview_rows}
    if split_values != set(VALID_SPLITS):
        raise RuntimeError(f"Expected splits {VALID_SPLITS}; found {sorted(split_values)}")

    # Verify tile-level split consistency.
    tile_to_split = {}
    for row in preview_rows:
        tile = row["tile_name"]
        split = row["split"]
        previous = tile_to_split.setdefault(tile, split)
        if previous != split:
            raise RuntimeError(f"Tile appears in multiple splits: {tile}")

    counts = Counter()
    category_counts = {s: Counter() for s in VALID_SPLITS}
    tile_sets = {s: set() for s in VALID_SPLITS}
    patch_ids = {s: [] for s in VALID_SPLITS}
    polygon_ids = {s: set() for s in VALID_SPLITS}

    final_rows = []
    link_stats = Counter()
    observed_mask_values = set()

    print(f"Materializing {len(preview_rows)} patches...")

    for index, preview in enumerate(preview_rows, start=1):
        patch_id = preview["patch_id"]
        split = preview["split"]
        source_row = source_by_id[patch_id]

        # Only split/path are intentionally changed.
        final = dict(source_row)
        final["split"] = split

        destination_rel = Path("patches") / split / f"{patch_id}.npz"
        final["patch_path"] = str(destination_rel).replace("\\", "/")

        src_npz = find_source_patch(source, source_row)
        dst_npz = outdir / destination_rel

        action = link_or_copy(src_npz, dst_npz)
        link_stats[action] += 1

        with np.load(src_npz) as data:
            if "features" not in data or "mask" not in data:
                raise RuntimeError(f"NPZ missing features/mask arrays: {src_npz}")
            mask_values = set(np.unique(data["mask"]).tolist())
            if not mask_values <= {0, 1}:
                raise RuntimeError(
                    f"Non-binary mask in patch {patch_id}: {sorted(mask_values)}"
                )
            observed_mask_values.update(mask_values)

        counts[split] += 1
        category_counts[split][final.get("category", "")] += 1
        tile_sets[split].add(final["tile_name"])
        patch_ids[split].append(patch_id)

        ids = {
            x.strip()
            for x in str(final.get("landslide_ids_in_tile", "")).split(";")
            if x.strip()
        }
        polygon_ids[split].update(ids)

        final_rows.append(final)

        if index % 500 == 0 or index == len(preview_rows):
            print(f"  {index}/{len(preview_rows)}")

    expected_counts = {
        "train": 3950,
        "validation": 891,
        "test": 1210,
    }
    expected_tiles = {
        "train": 63,
        "validation": 16,
        "test": 19,
    }

    if dict(counts) != expected_counts:
        raise RuntimeError(
            f"Unexpected patch counts. Expected {expected_counts}; got {dict(counts)}"
        )

    actual_tiles = {s: len(tile_sets[s]) for s in VALID_SPLITS}
    if actual_tiles != expected_tiles:
        raise RuntimeError(
            f"Unexpected tile counts. Expected {expected_tiles}; got {actual_tiles}"
        )

    # Cross-split polygon overlap must remain zero.
    overlaps = {}
    for i, a in enumerate(VALID_SPLITS):
        for b in VALID_SPLITS[i + 1:]:
            shared = sorted(polygon_ids[a] & polygon_ids[b])
            overlaps[f"{a}__{b}"] = shared

    if any(overlaps.values()):
        raise RuntimeError(
            "Cross-split landslide polygon overlap detected after materialization: "
            + str({k: len(v) for k, v in overlaps.items()})
        )

    # Keep original field ordering.
    write_csv(outdir / "patches.csv", final_rows, source_fields)

    # Preserve dataset metadata/provenance.
    for name in ("channels.json", "failed_tiles.csv"):
        src = source / name
        if src.exists():
            shutil.copy2(src, outdir / name)

    shutil.copy2(split_map, outdir / "split_map_64_16_20.csv")
    shutil.copy2(split_verification, outdir / "split_verification.json")
    shutil.copy2(preview_manifest, outdir / "phase1_preview_manifest.csv")

    fingerprints = {
        split: {
            "patch_count": counts[split],
            "tile_count": len(tile_sets[split]),
            "patch_id_sha256": hash_lines(patch_ids[split]),
            "tile_name_sha256": hash_lines(tile_sets[split]),
            "landslide_id_sha256": hash_lines(polygon_ids[split]),
            "unique_landslide_ids": len(polygon_ids[split]),
        }
        for split in VALID_SPLITS
    }

    (outdir / "split_fingerprints.json").write_text(
        json.dumps(fingerprints, indent=2),
        encoding="utf-8",
    )

    summary = {
        "ground_truth_policy": "strict_binary_0_1",
        "source_dataset": str(source),
        "phase1_split_dir": str(phase1),
        "total_patches": len(final_rows),
        "usable_tiles": len(tile_to_split),
        "split_counts": dict(counts),
        "split_tile_counts": actual_tiles,
        "split_category_counts": {
            split: dict(category_counts[split])
            for split in VALID_SPLITS
        },
        "unique_polygon_ids": {
            split: len(polygon_ids[split])
            for split in VALID_SPLITS
        },
        "cross_split_polygon_overlap_count": {
            key: len(value)
            for key, value in overlaps.items()
        },
        "mask_values_observed": sorted(observed_mask_values),
        "materialization": dict(link_stats),
        "split_policy": {
            "target": "64/16/20 by spatially grouped tiles",
            "buffer_m": 500,
            "seed": 42,
            "train_fraction_actual_tiles": actual_tiles["train"] / len(tile_to_split),
            "validation_fraction_actual_tiles": actual_tiles["validation"] / len(tile_to_split),
            "test_fraction_actual_tiles": actual_tiles["test"] / len(tile_to_split),
        },
        "test_policy": (
            "Internal test split is locked. Do not use test rows for normalization, "
            "checkpoint selection, threshold selection, architecture selection, "
            "feature selection, or class-weight tuning."
        ),
    }

    (outdir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lock_text = """TILLAMOOK INTERNAL TEST SET — LOCKED

This directory contains the frozen 20%-target internal Tillamook test partition.

Do NOT use test rows for:
- normalization statistics
- positive-class-weight calculation
- checkpoint / epoch selection
- threshold selection
- architecture selection
- feature selection
- hyperparameter tuning
- repeated error-guided model changes

Use TRAIN for fitting.
Use VALIDATION for development/model selection.
Use TEST only after the final Tillamook configuration and validation threshold are frozen.

External Buxton/Vernonia and Oregon City regions remain separate external tests.
"""
    (outdir / "TEST_SET_LOCKED.txt").write_text(lock_text, encoding="utf-8")

    print()
    print("PHASE 2 MATERIALIZATION COMPLETE")
    print(f"Output: {outdir}")
    print(f"Total patches: {len(final_rows)}")
    for split in VALID_SPLITS:
        print(
            f"{split:10s}: patches={counts[split]:4d} "
            f"tiles={len(tile_sets[split]):2d} "
            f"polygons={len(polygon_ids[split]):3d}"
        )
    print(f"Mask values: {sorted(observed_mask_values)}")
    print(f"Link/copy stats: {dict(link_stats)}")
    print("Cross-split polygon overlap: 0")
    print()
    print("NEXT: rerun verify_splits.py on the final patches.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
