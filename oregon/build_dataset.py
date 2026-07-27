#!/usr/bin/env python3
"""Build a reusable Oregon LiDAR/SLIDO patch dataset.

This stage is intentionally separate from training. It reads each LAZ once,
creates terrain derivatives and a filtered SLIDO mask, splits at tile level,
and saves compressed patches plus an auditable CSV manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from terrain_utils import (
    build_feature_stack,
    classify_patch,
    intersecting_ids,
    iter_patch_windows,
    rasterize_slido_mask,
    read_laz_ground_dem,
)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "tile"


def assign_tile_splits(tile_paths: list[Path], seed: int) -> dict[str, str]:
    """Deterministically assign complete tiles, never individual patches."""
    ordered = sorted(tile_paths, key=lambda path: path.name.casefold())
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)

    if n <= 1:
        counts = (n, 0, 0)
    elif n == 2:
        counts = (1, 1, 0)
    else:
        n_val = max(1, round(n * 0.20))
        n_test = max(1, round(n * 0.20))
        if n_val + n_test >= n:
            n_val = 1
            n_test = 1
        n_train = n - n_val - n_test
        counts = (n_train, n_val, n_test)

    n_train, n_val, n_test = counts
    mapping: dict[str, str] = {}
    for path in ordered[:n_train]:
        mapping[path.name] = "train"
    for path in ordered[n_train : n_train + n_val]:
        mapping[path.name] = "validation"
    for path in ordered[n_train + n_val : n_train + n_val + n_test]:
        mapping[path.name] = "test"
    return mapping


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_negative_candidates(
    candidates: list[dict[str, Any]],
    *,
    positive_count: int,
    ratio: float,
    max_per_tile: int,
    seed_text: str,
) -> list[dict[str, Any]]:
    if not candidates or max_per_tile <= 0:
        return []
    desired = max(10, int(math.ceil(max(1, positive_count) * ratio)))
    desired = min(desired, max_per_tile, len(candidates))

    seed = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    hard = [row for row in candidates if row["is_hard_negative"]]
    ordinary = [row for row in candidates if not row["is_hard_negative"]]
    rng.shuffle(hard)
    rng.shuffle(ordinary)

    hard_target = min(len(hard), int(math.ceil(desired * 0.70)))
    selected = hard[:hard_target]
    remainder = desired - len(selected)
    selected.extend(ordinary[:remainder])
    remainder = desired - len(selected)
    if remainder:
        selected.extend(hard[hard_target : hard_target + remainder])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an auditable, tile-split Oregon landslide patch dataset.")
    parser.add_argument("--laz-dir", type=Path, default=Path("lidar_tiles"))
    parser.add_argument("--slido-geojson", type=Path, default=Path("slido_deposits_oregon_city.geojson"))
    parser.add_argument("--outdir", type=Path, default=Path("dataset_pilot"))
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-tiles", type=int, default=10)
    parser.add_argument("--max-cells", type=int, default=80_000_000)
    parser.add_argument("--min-ground-cell-fraction", type=float, default=0.25)
    parser.add_argument("--min-patch-ground-fraction", type=float, default=0.50)
    parser.add_argument("--interior-threshold", type=float, default=0.10)
    parser.add_argument("--boundary-threshold", type=float, default=0.01)
    parser.add_argument(
        "--include-trace-positives",
        action="store_true",
        help="Keep patches with >0 but < boundary-threshold mask fraction. Default drops them.",
    )
    parser.add_argument("--negative-buffer-m", type=float, default=50.0)
    parser.add_argument("--hard-negative-slope", type=float, default=8.0)
    parser.add_argument("--negative-ratio", type=float, default=1.5)
    parser.add_argument("--max-negatives-per-tile", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--all-touched", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.patch_size <= 0 or args.stride <= 0:
        parser.error("--patch-size and --stride must be positive")
    if not 0 <= args.min_ground_cell_fraction <= 1:
        parser.error("--min-ground-cell-fraction must be between 0 and 1")
    if not 0 <= args.min_patch_ground_fraction <= 1:
        parser.error("--min-patch-ground-fraction must be between 0 and 1")
    if not 0 <= args.boundary_threshold <= args.interior_threshold <= 1:
        parser.error("thresholds must satisfy 0 <= boundary <= interior <= 1")

    laz_dir = args.laz_dir.resolve()
    slido_path = args.slido_geojson.resolve()
    outdir = args.outdir.resolve()
    if not laz_dir.exists():
        parser.error(f"LAZ directory does not exist: {laz_dir}")
    if not slido_path.exists():
        parser.error(f"SLIDO GeoJSON does not exist: {slido_path}")

    laz_paths = sorted([*laz_dir.glob("*.laz"), *laz_dir.glob("*.las")])
    if args.max_tiles:
        laz_paths = laz_paths[: args.max_tiles]
    if not laz_paths:
        parser.error(f"No .laz or .las files found in {laz_dir}")

    if outdir.exists() and any(outdir.iterdir()):
        if not args.overwrite:
            parser.error(f"Output directory is not empty: {outdir}. Use --overwrite deliberately.")
        shutil.rmtree(outdir)
    (outdir / "patches").mkdir(parents=True, exist_ok=True)

    split_by_tile = assign_tile_splits(laz_paths, args.seed)
    rows: list[dict[str, Any]] = []
    failed_tiles: list[dict[str, str]] = []
    tile_summaries: list[dict[str, Any]] = []
    feature_names: tuple[str, ...] | None = None

    print(f"Processing {len(laz_paths)} tile(s)")
    print(f"Tile split counts: {dict(Counter(split_by_tile.values()))}")

    for tile_index, laz_path in enumerate(laz_paths, start=1):
        print("\n" + "=" * 72)
        print(f"[{tile_index}/{len(laz_paths)}] {laz_path.name}")
        print("=" * 72)
        try:
            tile = read_laz_ground_dem(
                laz_path,
                cell_size=args.cell_size,
                max_cells=args.max_cells,
            )
            if tile.ground_cell_fraction < args.min_ground_cell_fraction:
                raise RuntimeError(
                    f"ground-cell coverage {tile.ground_cell_fraction:.3f} below "
                    f"minimum {args.min_ground_cell_fraction:.3f}"
                )

            features, current_feature_names = build_feature_stack(tile.dem, cell_size=args.cell_size)
            if feature_names is None:
                feature_names = current_feature_names
            elif feature_names != current_feature_names:
                raise RuntimeError("Feature channel definitions changed between tiles")

            mask, intersecting_records = rasterize_slido_mask(
                slido_path,
                tile,
                description="Landslide",
                all_touched=args.all_touched,
            )
            landslide_ids, ref_ids = intersecting_ids(intersecting_records)
            distance_to_positive = (
                distance_transform_edt(mask == 0) * args.cell_size
                if mask.any()
                else np.full(mask.shape, np.inf, dtype=np.float32)
            )

            positive_candidates: list[dict[str, Any]] = []
            negative_candidates: list[dict[str, Any]] = []
            skipped_low_coverage = 0
            skipped_trace = 0

            for row_offset, col_offset in iter_patch_windows(
                *tile.shape,
                patch_size=args.patch_size,
                stride=args.stride,
            ):
                row_slice = slice(row_offset, row_offset + args.patch_size)
                col_slice = slice(col_offset, col_offset + args.patch_size)
                patch_mask = mask[row_slice, col_slice]
                patch_valid = tile.valid_ground_mask[row_slice, col_slice]
                valid_fraction = float(patch_valid.mean())
                if valid_fraction < args.min_patch_ground_fraction:
                    skipped_low_coverage += 1
                    continue

                positive_fraction = float(patch_mask.mean())
                category = classify_patch(
                    positive_fraction,
                    interior_threshold=args.interior_threshold,
                    boundary_threshold=args.boundary_threshold,
                )
                patch_slope_mean = float(features[1, row_slice, col_slice].mean())
                min_distance = float(distance_to_positive[row_slice, col_slice].min())
                candidate = {
                    "row_offset": row_offset,
                    "col_offset": col_offset,
                    "positive_fraction": positive_fraction,
                    "category": category,
                    "ground_fraction": valid_fraction,
                    "mean_slope_degrees": patch_slope_mean,
                    "distance_to_positive_m": min_distance,
                    "is_hard_negative": category == "negative" and patch_slope_mean >= args.hard_negative_slope,
                }

                if category == "negative":
                    if min_distance >= args.negative_buffer_m:
                        negative_candidates.append(candidate)
                elif category == "positive_trace" and not args.include_trace_positives:
                    skipped_trace += 1
                else:
                    positive_candidates.append(candidate)

            selected_negatives = choose_negative_candidates(
                negative_candidates,
                positive_count=len(positive_candidates),
                ratio=args.negative_ratio,
                max_per_tile=args.max_negatives_per_tile,
                seed_text=f"{args.seed}:{laz_path.name}",
            )
            selected_candidates = positive_candidates + selected_negatives
            selected_candidates.sort(key=lambda row: (row["row_offset"], row["col_offset"]))

            tile_stem = safe_name(laz_path.stem)
            split = split_by_tile[laz_path.name]
            split_dir = outdir / "patches" / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for candidate in selected_candidates:
                row_offset = int(candidate["row_offset"])
                col_offset = int(candidate["col_offset"])
                row_slice = slice(row_offset, row_offset + args.patch_size)
                col_slice = slice(col_offset, col_offset + args.patch_size)
                patch_id = f"{tile_stem}_r{row_offset:06d}_c{col_offset:06d}"
                patch_path = split_dir / f"{patch_id}.npz"

                feature_patch = features[:, row_slice, col_slice].astype(args.feature_dtype)
                mask_patch = mask[row_slice, col_slice].astype(np.uint8)
                np.savez_compressed(patch_path, features=feature_patch, mask=mask_patch)

                x_min = tile.transform.c + col_offset * tile.transform.a
                y_max = tile.transform.f + row_offset * tile.transform.e
                x_max = x_min + args.patch_size * tile.transform.a
                y_min = y_max + args.patch_size * tile.transform.e
                rows.append(
                    {
                        "patch_id": patch_id,
                        "split": split,
                        "category": candidate["category"],
                        "tile_name": laz_path.name,
                        "tile_path": str(laz_path),
                        "patch_path": str(patch_path.relative_to(outdir)),
                        "row_offset": row_offset,
                        "col_offset": col_offset,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "crs": tile.crs.to_string(),
                        "positive_fraction": candidate["positive_fraction"],
                        "ground_fraction": candidate["ground_fraction"],
                        "mean_slope_degrees": candidate["mean_slope_degrees"],
                        "distance_to_positive_m": candidate["distance_to_positive_m"],
                        "is_hard_negative": candidate["is_hard_negative"],
                        "landslide_ids_in_tile": landslide_ids,
                        "slido_ref_ids_in_tile": ref_ids,
                        "qc_status": "",
                        "qc_notes": "",
                    }
                )

            category_counts = Counter(row["category"] for row in selected_candidates)
            tile_summary = {
                "tile_name": laz_path.name,
                "split": split,
                "crs": tile.crs.to_string(),
                "dem_shape": list(tile.shape),
                "ground_point_count": tile.ground_point_count,
                "ground_cell_fraction": tile.ground_cell_fraction,
                "intersecting_slido_polygons": len(intersecting_records),
                "mask_positive_fraction": float(mask.mean()),
                "saved_patches": len(selected_candidates),
                "category_counts": dict(category_counts),
                "skipped_low_ground_coverage": skipped_low_coverage,
                "skipped_trace_positives": skipped_trace,
                "eligible_negative_candidates": len(negative_candidates),
            }
            tile_summaries.append(tile_summary)
            print(json.dumps(tile_summary, indent=2))

        except Exception as exc:
            failed_tiles.append({"tile_name": laz_path.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED: {type(exc).__name__}: {exc}")

    fields = [
        "patch_id", "split", "category", "tile_name", "tile_path", "patch_path",
        "row_offset", "col_offset", "x_min", "y_min", "x_max", "y_max", "crs",
        "positive_fraction", "ground_fraction", "mean_slope_degrees",
        "distance_to_positive_m", "is_hard_negative", "landslide_ids_in_tile",
        "slido_ref_ids_in_tile", "qc_status", "qc_notes",
    ]
    write_csv(outdir / "patches.csv", rows, fields)
    write_csv(outdir / "failed_tiles.csv", failed_tiles, ["tile_name", "error"])

    category_counts = Counter(row["category"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    split_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        split_category_counts[row["split"]][row["category"]] += 1

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "laz_dir": str(laz_dir),
        "slido_geojson": str(slido_path),
        "outdir": str(outdir),
        "processed_tiles": len(tile_summaries),
        "failed_tiles": len(failed_tiles),
        "saved_patches": len(rows),
        "split_counts": dict(split_counts),
        "category_counts": dict(category_counts),
        "split_category_counts": {key: dict(value) for key, value in split_category_counts.items()},
        "feature_names": list(feature_names or []),
        "feature_dtype": args.feature_dtype,
        "cell_size": args.cell_size,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "description_filter": "Landslide",
        "tile_summaries": tile_summaries,
        "parameters": vars(args) | {"laz_dir": str(args.laz_dir), "slido_geojson": str(args.slido_geojson), "outdir": str(args.outdir)},
    }
    (outdir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (outdir / "channels.json").write_text(json.dumps({"feature_names": list(feature_names or [])}, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Saved {len(rows)} patches to {outdir}")
    print(f"Split counts: {dict(split_counts)}")
    print(f"Category counts: {dict(category_counts)}")
    print(f"Failed tiles: {len(failed_tiles)}")
    print("Next: run diagnostics/visualize_dataset.py and fill qc_status in qc_review.csv.")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
