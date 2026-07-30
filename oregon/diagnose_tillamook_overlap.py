#!/usr/bin/env python3
"""
Diagnose footprint overlap between two TNM LPC project groups.

This script does not select or download tiles and does not modify regions.json.
It reports:
- project record counts
- footprint-size distributions
- intersecting pair counts
- maximum/quantile IoU
- overlap relative to the smaller footprint
- anchor counts at several thresholds
- top matching record pairs

Run from the oregon directory so local select_tiles.py and tnm_utils.py are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from shapely.strtree import STRtree

try:
    from select_tiles import (
        footprint_iou,
        overlap_fraction,
        tile_footprint,
        tile_id,
        tile_project,
    )
    from tnm_utils import project_matches
except ImportError as exc:
    raise SystemExit(
        "Run this script from the repository's oregon directory, "
        "where select_tiles.py and tnm_utils.py are available."
    ) from exc


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("items", "features", "results", "records", "selected_tiles"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Unsupported TNM JSON structure in {path}")


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(values)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        position = (len(ordered) - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    return {
        "min": float(ordered[0]),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": float(ordered[-1]),
    }


def bounds_summary(geometries) -> dict[str, Any]:
    widths = [float(g.bounds[2] - g.bounds[0]) for g in geometries]
    heights = [float(g.bounds[3] - g.bounds[1]) for g in geometries]
    areas = [float(g.area) for g in geometries]
    return {
        "width": quantiles(widths),
        "height": quantiles(heights),
        "area": quantiles(areas),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure TNM footprint overlap between two LPC projects."
    )
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--left-project", required=True)
    parser.add_argument("--right-project", required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("regions/tillamook_overlap_diagnostic.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("regions/tillamook_overlap_top_pairs.csv"),
    )
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    records = load_records(args.tiles)
    left = [record for record in records if project_matches(record, args.left_project)]
    right = [record for record in records if project_matches(record, args.right_project)]

    if not left:
        raise SystemExit(f"No records matched left project: {args.left_project}")
    if not right:
        raise SystemExit(f"No records matched right project: {args.right_project}")

    left_geometries = [tile_footprint(record) for record in left]
    right_geometries = [tile_footprint(record) for record in right]
    tree = STRtree(right_geometries)

    pairs: list[dict[str, Any]] = []
    left_with_any_intersection: set[int] = set()
    right_with_any_intersection: set[int] = set()

    for left_index, left_geometry in enumerate(left_geometries):
        candidate_indices = tree.query(left_geometry, predicate="intersects")
        for raw_right_index in candidate_indices:
            right_index = int(raw_right_index)
            right_geometry = right_geometries[right_index]
            intersection_area = float(left_geometry.intersection(right_geometry).area)
            if intersection_area <= 0:
                continue

            iou = footprint_iou(left_geometry, right_geometry)
            smaller_overlap = overlap_fraction(left_geometry, right_geometry)
            left_overlap = intersection_area / left_geometry.area
            right_overlap = intersection_area / right_geometry.area

            left_with_any_intersection.add(left_index)
            right_with_any_intersection.add(right_index)

            pairs.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_id": tile_id(left[left_index]),
                    "right_id": tile_id(right[right_index]),
                    "left_project": tile_project(left[left_index]),
                    "right_project": tile_project(right[right_index]),
                    "iou": float(iou),
                    "smaller_overlap": float(smaller_overlap),
                    "left_overlap": float(left_overlap),
                    "right_overlap": float(right_overlap),
                    "intersection_area": intersection_area,
                    "left_area": float(left_geometry.area),
                    "right_area": float(right_geometry.area),
                    "area_ratio_small_to_large": float(
                        min(left_geometry.area, right_geometry.area)
                        / max(left_geometry.area, right_geometry.area)
                    ),
                }
            )

    pairs.sort(
        key=lambda row: (
            -row["iou"],
            -row["smaller_overlap"],
            row["left_id"].casefold(),
            row["right_id"].casefold(),
        )
    )

    thresholds = [0.10, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90]
    threshold_summary: dict[str, Any] = {}
    for threshold in thresholds:
        iou_pairs = [row for row in pairs if row["iou"] >= threshold]
        smaller_pairs = [
            row for row in pairs if row["smaller_overlap"] >= threshold
        ]
        threshold_summary[f"{threshold:.2f}"] = {
            "iou_pair_count": len(iou_pairs),
            "iou_left_anchor_count": len(
                {row["left_index"] for row in iou_pairs}
            ),
            "iou_right_record_count": len(
                {row["right_index"] for row in iou_pairs}
            ),
            "smaller_overlap_pair_count": len(smaller_pairs),
            "smaller_overlap_left_anchor_count": len(
                {row["left_index"] for row in smaller_pairs}
            ),
            "smaller_overlap_right_record_count": len(
                {row["right_index"] for row in smaller_pairs}
            ),
        }

    iou_values = [row["iou"] for row in pairs]
    smaller_values = [row["smaller_overlap"] for row in pairs]
    area_ratios = [row["area_ratio_small_to_large"] for row in pairs]

    max_iou = max(iou_values, default=0.0)
    max_smaller = max(smaller_values, default=0.0)

    interpretation: list[str] = []
    if not pairs:
        interpretation.append(
            "No footprint intersections were found between the two project groups."
        )
    else:
        interpretation.append(
            f"{len(pairs)} intersecting record pairs were found."
        )
        if max_iou < 0.8 and max_smaller >= 0.8:
            interpretation.append(
                "At least one smaller footprint is substantially covered by the "
                "other project, but the footprints differ in size and/or grid, so "
                "IoU >= 0.8 is structurally too strict for record-to-record matching."
            )
        elif max_iou < 0.8:
            interpretation.append(
                "No pair reaches IoU 0.8. Inspect the top pairs and footprint-size "
                "statistics before choosing a different matching rule."
            )
        else:
            interpretation.append(
                "Some pairs reach IoU 0.8; if the selector still reports zero anchors, "
                "inspect project matching, uniqueness, and budget constraints."
            )

    report = {
        "tiles_path": str(args.tiles.resolve()),
        "left_project_requested": args.left_project,
        "right_project_requested": args.right_project,
        "record_counts": {
            "all": len(records),
            "left": len(left),
            "right": len(right),
        },
        "canonical_project_values": {
            "left": sorted({tile_project(record) for record in left}),
            "right": sorted({tile_project(record) for record in right}),
        },
        "footprint_statistics": {
            "left": bounds_summary(left_geometries),
            "right": bounds_summary(right_geometries),
        },
        "intersection_summary": {
            "pair_count": len(pairs),
            "left_records_with_any_intersection": len(left_with_any_intersection),
            "right_records_with_any_intersection": len(right_with_any_intersection),
            "iou": quantiles(iou_values),
            "smaller_overlap": quantiles(smaller_values),
            "area_ratio_small_to_large": quantiles(area_ratios),
        },
        "thresholds": threshold_summary,
        "interpretation": interpretation,
        "top_pairs": pairs[: min(args.top, len(pairs))],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    fieldnames = [
        "left_id",
        "right_id",
        "left_project",
        "right_project",
        "iou",
        "smaller_overlap",
        "left_overlap",
        "right_overlap",
        "intersection_area",
        "left_area",
        "right_area",
        "area_ratio_small_to_large",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pairs[: min(args.top, len(pairs))]:
            writer.writerow({key: row[key] for key in fieldnames})

    print(f"Left records:  {len(left)}")
    print(f"Right records: {len(right)}")
    print(f"Intersecting pairs: {len(pairs)}")
    print(f"Max IoU: {max_iou:.6f}")
    print(f"Max smaller-footprint overlap: {max_smaller:.6f}")
    print()
    print("Threshold summary:")
    for threshold in thresholds:
        values = threshold_summary[f"{threshold:.2f}"]
        print(
            f"  {threshold:.2f}: "
            f"IoU anchors={values['iou_left_anchor_count']}, "
            f"smaller-overlap anchors={values['smaller_overlap_left_anchor_count']}"
        )
    print()
    for line in interpretation:
        print(line)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
