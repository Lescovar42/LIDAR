#!/usr/bin/env python3
"""A22-only Tillamook resolution probe with SLIDO-aware patch diagnostics.

This diagnostic writes metrics only. It does not write training patches and it
never pins a project or cell size automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform, unary_union

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from build_manifest import extract_landslide_date
from slido_utils import load_deposits, normalize_confidence
from terrain_utils import classify_patch, iter_patch_windows, read_laz_ground_dem

DEFAULT_CELL_SIZES = (1.0, 1.5, 2.0)
DEFAULT_PROJECT = "USGS_LPC_OR_WesternWildfires_A22"

_PROJECTED_LABEL_CACHE: dict[tuple[int, str], list[dict[str, Any]]] = {}


def _property_value(properties: Mapping[str, Any], name: str) -> Any:
    wanted = name.casefold()
    for key, value in properties.items():
        if str(key).casefold() == wanted:
            return value
    return None


def record_filename(record: Mapping[str, Any]) -> str:
    url = str(
        record.get("downloadURL")
        or record.get("downloadUrl")
        or record.get("download_url")
        or ""
    )
    name = Path(unquote(urlparse(url).path)).name
    if name:
        return name
    fallback = str(record.get("title") or record.get("tile_id") or "tile.laz")
    return fallback if Path(fallback).suffix else f"{fallback}.laz"


def load_selection(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("A22 probe selection must be a JSON list")
    return [dict(record) for record in raw if isinstance(record, Mapping)]


def load_label_records(path: Path, *, lidar_year: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for geometry, properties in load_deposits(path, description="Landslide"):
        confidence = normalize_confidence(
            _property_value(properties, "confidence_class")
            or _property_value(properties, "CONFIDENCE")
        )
        event_date, event_source = extract_landslide_date(properties)
        event_year = event_date.year if event_date is not None else None
        temporal_excluded = event_year is not None and event_year > lidar_year
        disposition = (
            "positive"
            if confidence in {"high", "moderate"} and not temporal_excluded
            else "ignore"
        )
        landslide_id = (
            _property_value(properties, "UNIQUE_ID")
            or _property_value(properties, "OBJECTID")
            or ""
        )
        key = str(landslide_id) if landslide_id else hashlib.sha1(geometry.wkb).hexdigest()
        output.append(
            {
                "geometry_wgs84": geometry,
                "polygon_key": key,
                "landslide_id": str(landslide_id),
                "confidence_class": confidence,
                "event_year": event_year,
                "event_year_source": event_source or "",
                "temporal_excluded": temporal_excluded,
                "disposition": disposition,
            }
        )
    return output


def project_labels(
    records: Sequence[Mapping[str, Any]],
    *,
    target_crs: CRS,
    tile_bounds: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    cache_key = (id(records), target_crs.to_wkt())
    transformed_records = _PROJECTED_LABEL_CACHE.get(cache_key)
    if transformed_records is None:
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        transformed_records = []
        for record in records:
            copy = dict(record)
            copy["geometry"] = shapely_transform(
                transformer.transform,
                record["geometry_wgs84"],
            )
            transformed_records.append(copy)
        _PROJECTED_LABEL_CACHE[cache_key] = transformed_records

    tile_polygon = box(*tile_bounds)
    projected: list[dict[str, Any]] = []
    for record in transformed_records:
        geometry = record["geometry"]
        if geometry.is_empty or not geometry.intersects(tile_polygon):
            continue
        clipped = geometry.intersection(tile_polygon)
        if clipped.is_empty or clipped.area <= 0:
            continue
        copy = dict(record)
        copy["geometry"] = clipped
        projected.append(copy)
    return projected


def rasterize_labels(
    projected: Sequence[Mapping[str, Any]],
    *,
    shape: tuple[int, int],
    transform: Any,
) -> np.ndarray:
    positive_shapes = [
        (mapping(record["geometry"]), 1)
        for record in projected
        if record["disposition"] == "positive"
    ]
    ignore_shapes = [
        (mapping(record["geometry"]), 1)
        for record in projected
        if record["disposition"] == "ignore"
    ]
    labels = np.zeros(shape, dtype=np.uint8)
    if positive_shapes:
        positive = rasterize(
            positive_shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
            all_touched=False,
            dtype="uint8",
        ).astype(bool)
        labels[positive] = 1
    if ignore_shapes:
        ignored = rasterize(
            ignore_shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
            all_touched=False,
            dtype="uint8",
        ).astype(bool)
        labels[ignored] = 255
    return labels


def patch_box(transform: Any, row: int, col: int, patch_size: int):
    x0, y0 = transform * (col, row)
    x1, y1 = transform * (col + patch_size, row + patch_size)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def summarize_patch_labels(
    valid_ground_mask: np.ndarray,
    labels: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    min_patch_ground_fraction: float,
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    counts: Counter[str] = Counter()
    accepted_windows: list[tuple[int, int]] = []
    ground_fractions: list[float] = []
    positive_fractions: list[float] = []
    ignore_fractions: list[float] = []
    total = 0

    for row, col in iter_patch_windows(
        *valid_ground_mask.shape,
        patch_size=patch_size,
        stride=stride,
    ):
        total += 1
        ground = valid_ground_mask[row : row + patch_size, col : col + patch_size]
        ground_fraction = float(ground.mean())
        ground_fractions.append(ground_fraction)
        if ground_fraction < min_patch_ground_fraction:
            counts["rejected_ground"] += 1
            continue

        accepted_windows.append((row, col))
        patch = labels[row : row + patch_size, col : col + patch_size]
        positive_fraction = float(np.mean(patch == 1))
        ignore_fraction = float(np.mean(patch == 255))
        positive_fractions.append(positive_fraction)
        ignore_fractions.append(ignore_fraction)

        if positive_fraction > 0:
            category = classify_patch(positive_fraction)
            counts[category] += 1
            counts["positive_total"] += 1
            if ignore_fraction > 0:
                counts["positive_with_ignore"] += 1
        elif ignore_fraction > 0:
            counts["ignore_only"] += 1
        else:
            counts["negative"] += 1

    accepted = len(accepted_windows)
    summary = {
        "total": total,
        "accepted": accepted,
        "accepted_fraction": accepted / total if total else 0.0,
        "rejected_ground": counts["rejected_ground"],
        "positive_total": counts["positive_total"],
        "positive_interior": counts["positive_interior"],
        "positive_boundary": counts["positive_boundary"],
        "positive_trace": counts["positive_trace"],
        "positive_with_ignore": counts["positive_with_ignore"],
        "negative": counts["negative"],
        "ignore_only": counts["ignore_only"],
        "ground_fraction": {
            "mean": float(np.mean(ground_fractions)) if ground_fractions else None,
            "min": float(min(ground_fractions)) if ground_fractions else None,
            "max": float(max(ground_fractions)) if ground_fractions else None,
        },
        "accepted_positive_fraction": {
            "mean": float(np.mean(positive_fractions)) if positive_fractions else None,
            "max": float(max(positive_fractions)) if positive_fractions else None,
        },
        "accepted_ignore_fraction": {
            "mean": float(np.mean(ignore_fractions)) if ignore_fractions else None,
            "max": float(max(ignore_fractions)) if ignore_fractions else None,
        },
    }
    return summary, accepted_windows


def summarize_tile(
    path: Path,
    *,
    project: str,
    cell_size: float,
    label_records: Sequence[Mapping[str, Any]],
    patch_size: int,
    stride: int,
    min_patch_ground_fraction: float,
    max_cells: int,
) -> dict[str, Any]:
    tile = read_laz_ground_dem(path, cell_size=cell_size, max_cells=max_cells)
    projected = project_labels(
        label_records,
        target_crs=tile.crs,
        tile_bounds=tile.bounds,
    )
    labels = rasterize_labels(
        projected,
        shape=tile.shape,
        transform=tile.transform,
    )
    patches, accepted_windows = summarize_patch_labels(
        tile.valid_ground_mask,
        labels,
        patch_size=patch_size,
        stride=stride,
        min_patch_ground_fraction=min_patch_ground_fraction,
    )

    accepted_union = (
        unary_union(
            [
                patch_box(tile.transform, row, col, patch_size)
                for row, col in accepted_windows
            ]
        )
        if accepted_windows
        else None
    )
    eligible = {
        str(record["polygon_key"])
        for record in projected
        if record["disposition"] == "positive"
    }
    retained = {
        str(record["polygon_key"])
        for record in projected
        if record["disposition"] == "positive"
        and accepted_union is not None
        and record["geometry"].intersects(accepted_union)
        and record["geometry"].intersection(accepted_union).area > 0
    }
    ignored = {
        str(record["polygon_key"])
        for record in projected
        if record["disposition"] == "ignore"
    }
    temporally_excluded = {
        str(record["polygon_key"])
        for record in projected
        if record["temporal_excluded"]
    }

    return {
        "project": project,
        "tile": path.name,
        "cell_size": cell_size,
        "shape": list(tile.shape),
        "crs": tile.crs.to_string(),
        "ground_point_count": tile.ground_point_count,
        "ground_cell_fraction": tile.ground_cell_fraction,
        "missing_ground_cell_fraction": 1.0 - tile.ground_cell_fraction,
        "patches": patches,
        "eligible_polygon_keys": sorted(eligible),
        "retained_polygon_keys": sorted(retained),
        "ignored_polygon_keys": sorted(ignored),
        "temporally_excluded_polygon_keys": sorted(temporally_excluded),
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for cell_size in sorted({float(row["cell_size"]) for row in rows}):
        group = [row for row in rows if float(row["cell_size"]) == cell_size]
        eligible = {
            key
            for row in group
            for key in row["eligible_polygon_keys"]
        }
        retained = {
            key
            for row in group
            for key in row["retained_polygon_keys"]
        }
        ignored = {
            key
            for row in group
            for key in row["ignored_polygon_keys"]
        }
        temporal = {
            key
            for row in group
            for key in row["temporally_excluded_polygon_keys"]
        }
        total_patches = sum(int(row["patches"]["total"]) for row in group)
        accepted_patches = sum(int(row["patches"]["accepted"]) for row in group)
        summaries.append(
            {
                "cell_size": cell_size,
                "tile_count": len(group),
                "ground_cell_fraction": {
                    "mean": float(np.mean([row["ground_cell_fraction"] for row in group])),
                    "min": float(min(row["ground_cell_fraction"] for row in group)),
                    "max": float(max(row["ground_cell_fraction"] for row in group)),
                },
                "missing_ground_cell_fraction": {
                    "mean": float(
                        np.mean([row["missing_ground_cell_fraction"] for row in group])
                    )
                },
                "patches": {
                    "total": total_patches,
                    "accepted": accepted_patches,
                    "accepted_fraction": (
                        accepted_patches / total_patches if total_patches else 0.0
                    ),
                    "rejected_ground": sum(
                        int(row["patches"]["rejected_ground"]) for row in group
                    ),
                    "positive_total": sum(
                        int(row["patches"]["positive_total"]) for row in group
                    ),
                    "positive_interior": sum(
                        int(row["patches"]["positive_interior"]) for row in group
                    ),
                    "positive_boundary": sum(
                        int(row["patches"]["positive_boundary"]) for row in group
                    ),
                    "positive_trace": sum(
                        int(row["patches"]["positive_trace"]) for row in group
                    ),
                    "positive_with_ignore": sum(
                        int(row["patches"]["positive_with_ignore"]) for row in group
                    ),
                    "negative": sum(
                        int(row["patches"]["negative"]) for row in group
                    ),
                    "ignore_only": sum(
                        int(row["patches"]["ignore_only"]) for row in group
                    ),
                },
                "polygons": {
                    "eligible_unique": len(eligible),
                    "retained_unique": len(retained),
                    "retained_fraction": len(retained) / len(eligible) if eligible else 0.0,
                    "ignored_unique": len(ignored),
                    "temporally_excluded_unique": len(temporal),
                    "retention_definition": (
                        "positive polygon intersects at least one patch passing "
                        "the minimum ground-fraction threshold"
                    ),
                },
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure A22 ground coverage and SLIDO-aware patch survival."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--laz-dir", type=Path, required=True)
    parser.add_argument("--slido", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--lidar-year", type=int, default=2020)
    parser.add_argument("--cell-size", action="append", type=float, dest="cell_sizes")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-patch-ground-fraction", type=float, default=0.5)
    parser.add_argument("--max-cells", type=int, default=80_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.selection.is_file():
        parser.error(f"Missing selection: {args.selection}")
    if not args.laz_dir.is_dir():
        parser.error(f"Missing LAZ directory: {args.laz_dir}")
    if not args.slido.is_file():
        parser.error(f"Missing SLIDO GeoJSON: {args.slido}")
    if args.patch_size <= 0 or args.stride <= 0:
        parser.error("--patch-size and --stride must be positive")
    if not 0 <= args.min_patch_ground_fraction <= 1:
        parser.error("--min-patch-ground-fraction must be between 0 and 1")
    cell_sizes = tuple(args.cell_sizes or DEFAULT_CELL_SIZES)
    if any(value <= 0 for value in cell_sizes):
        parser.error("--cell-size must be positive")

    try:
        selection = load_selection(args.selection)
        labels = load_label_records(args.slido, lidar_year=args.lidar_year)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    paths = [args.laz_dir / record_filename(record) for record in selection]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("Missing selected LAZ file(s): " + ", ".join(missing[:5]))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in paths:
        for cell_size in cell_sizes:
            try:
                row = summarize_tile(
                    path,
                    project=args.project,
                    cell_size=cell_size,
                    label_records=labels,
                    patch_size=args.patch_size,
                    stride=args.stride,
                    min_patch_ground_fraction=args.min_patch_ground_fraction,
                    max_cells=args.max_cells,
                )
                rows.append(row)
                print(
                    f"{path.name} {cell_size:g}m: "
                    f"ground={row['ground_cell_fraction']:.3f}, "
                    f"accepted={row['patches']['accepted']}/{row['patches']['total']}, "
                    f"positive={row['patches']['positive_total']}, "
                    f"negative={row['patches']['negative']}"
                )
            except Exception as exc:
                failures.append(
                    {
                        "project": args.project,
                        "tile": path.name,
                        "cell_size": cell_size,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"FAIL {path.name} {cell_size:g}m: {type(exc).__name__}: {exc}")

    result = {
        "parameters": {
            "selection": str(args.selection.resolve()),
            "laz_dir": str(args.laz_dir.resolve()),
            "slido": str(args.slido.resolve()),
            "project": args.project,
            "lidar_year": args.lidar_year,
            "cell_sizes": list(cell_sizes),
            "patch_size": args.patch_size,
            "stride": args.stride,
            "min_patch_ground_fraction": args.min_patch_ground_fraction,
            "label_rule": (
                "SLIDO Landslide polygons with high/moderate confidence are positive; "
                "low/unknown confidence and known post-LiDAR events are ignore"
            ),
            "decision_status": "not_pinned",
        },
        "tiles": rows,
        "cell_size_summary": aggregate(rows) if rows else [],
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\nCELL-SIZE SUMMARY")
    for summary in result["cell_size_summary"]:
        patches = summary["patches"]
        polygons = summary["polygons"]
        print(
            f"{summary['cell_size']:g}m: "
            f"tiles={summary['tile_count']}, "
            f"ground={summary['ground_cell_fraction']['mean']:.3f}, "
            f"accepted={patches['accepted']}/{patches['total']}, "
            f"positive={patches['positive_total']}, "
            f"negative={patches['negative']}, "
            f"polygons={polygons['retained_unique']}/{polygons['eligible_unique']}"
        )
    print(f"Failures: {len(failures)}")
    print(f"Wrote: {args.output}")
    print("No project or cell size was pinned automatically.")
    return 0 if rows and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
