#!/usr/bin/env python3
"""Verify metric separation between patch bounds in different dataset splits."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

METRIC_CRS = CRS.from_epsg(5070)


@dataclass(frozen=True)
class MetricPatch:
    patch_id: str
    split: str
    region_id: str
    footprint: BaseGeometry


def _required(row: Mapping[str, str], name: str, row_number: int) -> str:
    value = str(row.get(name, "")).strip()
    if not value:
        raise ValueError(f"row {row_number} is missing {name}")
    return value


def metric_patch_from_row(row: Mapping[str, str], row_number: int = 1) -> MetricPatch:
    """Parse and project one manifest patch footprint to the shared metric CRS."""
    patch_id = _required(row, "patch_id", row_number)
    split = _required(row, "split", row_number)
    crs = CRS.from_user_input(_required(row, "crs", row_number))
    try:
        xmin = float(_required(row, "x_min", row_number))
        ymin = float(_required(row, "y_min", row_number))
        xmax = float(_required(row, "x_max", row_number))
        ymax = float(_required(row, "y_max", row_number))
    except ValueError as exc:
        raise ValueError(f"row {row_number} has invalid numeric bounds") from exc
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"row {row_number} has non-positive patch bounds")
    footprint = box(xmin, ymin, xmax, ymax)
    if crs != METRIC_CRS:
        transformer = Transformer.from_crs(crs, METRIC_CRS, always_xy=True)
        footprint = shapely_transform(transformer.transform, footprint)
    return MetricPatch(
        patch_id=patch_id,
        split=split,
        region_id=str(row.get("region_id", "")).strip(),
        footprint=footprint,
    )


def find_split_violations(
    patches: Sequence[MetricPatch], *, buffer_m: float = 500.0
) -> list[dict[str, object]]:
    """Return every unique cross-split pair closer than ``buffer_m``.

    A metric spatial bucket index limits exact distance calculations to nearby
    footprints while preserving exhaustive results.
    """
    if buffer_m < 0:
        raise ValueError("buffer_m must be non-negative")
    if buffer_m == 0:
        return []
    violations: list[dict[str, object]] = []
    ordered = sorted(patches, key=lambda patch: (patch.split, patch.patch_id))
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    def cells(bounds: tuple[float, float, float, float], expand: float = 0.0):
        xmin, ymin, xmax, ymax = bounds
        first_x = math.floor((xmin - expand) / buffer_m)
        last_x = math.floor((xmax + expand) / buffer_m)
        first_y = math.floor((ymin - expand) / buffer_m)
        last_y = math.floor((ymax + expand) / buffer_m)
        for x_index in range(first_x, last_x + 1):
            for y_index in range(first_y, last_y + 1):
                yield x_index, y_index

    for right_index, right in enumerate(ordered):
        candidate_indices = {
            left_index
            for cell in cells(right.footprint.bounds, buffer_m)
            for left_index in buckets.get(cell, [])
        }
        for left_index in sorted(candidate_indices):
            left = ordered[left_index]
            if left.split == right.split:
                continue
            distance = float(left.footprint.distance(right.footprint))
            if distance < buffer_m:
                violations.append(
                    {
                        "patch_a": left.patch_id,
                        "split_a": left.split,
                        "region_a": left.region_id,
                        "patch_b": right.patch_id,
                        "split_b": right.split,
                        "region_b": right.region_id,
                        "distance_m": distance,
                        "required_buffer_m": buffer_m,
                    }
                )
        for cell in cells(right.footprint.bounds):
            buckets[cell].append(right_index)
    return violations


def verify_manifest(path: str | Path, *, buffer_m: float = 500.0) -> dict[str, object]:
    manifest_path = Path(path)
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    patches = [metric_patch_from_row(row, index) for index, row in enumerate(rows, start=2)]
    violations = find_split_violations(patches, buffer_m=buffer_m)
    split_counts: dict[str, int] = {}
    for patch in patches:
        split_counts[patch.split] = split_counts.get(patch.split, 0) + 1
    return {
        "manifest": str(manifest_path.resolve()),
        "buffer_m": buffer_m,
        "patch_count": len(patches),
        "split_counts": dict(sorted(split_counts.items())),
        "violation_count": len(violations),
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check cross-split patch bounds for spatial leakage.")
    parser.add_argument("manifest", nargs="?", type=Path, default=Path("dataset_pilot/patches.csv"))
    parser.add_argument("--buffer-m", type=float, default=500.0)
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    if args.buffer_m < 0:
        parser.error("--buffer-m must be non-negative")
    try:
        report = verify_manifest(args.manifest, buffer_m=args.buffer_m)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 1 if report["violation_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
