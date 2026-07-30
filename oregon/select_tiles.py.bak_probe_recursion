#!/usr/bin/env python3
"""Select budgeted rural LiDAR tiles without footprint leakage."""
from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

try:
    from .tnm_utils import canonical_project, first_value, project_matches
except ImportError:
    from tnm_utils import canonical_project, first_value, project_matches

SIZE_FIELDS = ("sizeInBytes", "size_bytes", "bytes")
ID_FIELDS = ("tile_id", "tileId", "id", "title", "downloadURL")


def tile_id(tile: Mapping[str, Any], fallback: str = "tile") -> str:
    return str(first_value(tile, ID_FIELDS, fallback))


def tile_project(tile: Mapping[str, Any]) -> str:
    return canonical_project(tile)


def tile_size(tile: Mapping[str, Any]) -> int:
    value = first_value(tile, SIZE_FIELDS, 0)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def tile_footprint(tile: Mapping[str, Any]) -> BaseGeometry:
    geometry = tile.get("geometry") or tile.get("footprint")
    if isinstance(geometry, Mapping):
        result = shape(geometry)
    else:
        bbox = tile.get("boundingBox") or tile.get("bbox")
        if isinstance(bbox, Mapping):
            aliases = {
                "xmin": ("minX", "west", "xmin"),
                "ymin": ("minY", "south", "ymin"),
                "xmax": ("maxX", "east", "xmax"),
                "ymax": ("maxY", "north", "ymax"),
            }
            values = {}
            for name, names in aliases.items():
                values[name] = next((bbox[key] for key in names if key in bbox), None)
            if any(value is None for value in values.values()):
                raise ValueError(f"Incomplete footprint for {tile_id(tile)}")
            result = box(float(values["xmin"]), float(values["ymin"]), float(values["xmax"]), float(values["ymax"]))
        elif isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) == 4:
            result = box(*map(float, bbox))
        else:
            raise ValueError(f"Missing footprint for {tile_id(tile)}")
    if result.is_empty or result.area <= 0:
        raise ValueError(f"Invalid footprint for {tile_id(tile)}")
    return result


def overlap_fraction(left: BaseGeometry, right: BaseGeometry) -> float:
    smaller = min(left.area, right.area)
    return float(left.intersection(right).area / smaller) if smaller > 0 else 0.0


def footprint_iou(left: BaseGeometry, right: BaseGeometry) -> float:
    union_area = left.union(right).area
    return float(left.intersection(right).area / union_area) if union_area > 0 else 0.0


def deduplicate_footprints(
    tiles: Sequence[Mapping[str, Any]], *, overlap_threshold: float = 0.8
) -> list[dict[str, Any]]:
    """Keep one deterministic representative for substantially co-located tiles.

    IoU is deliberately used instead of "any overlap" so adjacent TNM bounding
    envelopes survive small edge overlaps. ``tile_footprint`` prefers exact
    geometry when the record provides it.
    """
    if not 0 < overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in (0, 1]")
    ordered = sorted(
        (dict(tile) for tile in tiles),
        key=lambda item: (
            -float(item.get("_positive_intersection_area", 0.0)),
            tile_project(item).casefold(),
            tile_id(item).casefold(),
        ),
    )
    kept: list[dict[str, Any]] = []
    footprints: list[BaseGeometry] = []
    for tile in ordered:
        footprint = tile_footprint(tile)
        if any(footprint_iou(footprint, prior) >= overlap_threshold for prior in footprints):
            continue
        kept.append(tile)
        footprints.append(footprint)
    return kept


def normalize_confidence(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text.startswith("high"):
        return "high"
    if text.startswith("moderate"):
        return "moderate"
    if text.startswith("low"):
        return "low"
    return "unknown"


def positive_geometries(polygons: Iterable[Mapping[str, Any] | BaseGeometry]) -> list[BaseGeometry]:
    output: list[BaseGeometry] = []
    for feature in polygons:
        if isinstance(feature, BaseGeometry):
            output.append(feature)
            continue
        properties = feature.get("properties") or {}
        confidence = properties.get("confidence_class", properties.get("CONFIDENCE"))
        if normalize_confidence(confidence) not in {"high", "moderate"}:
            continue
        geometry = feature.get("geometry")
        if geometry:
            candidate = shape(geometry)
            if not candidate.is_empty:
                output.append(candidate)
    return output


def feature_geometries(polygons: Iterable[Mapping[str, Any] | BaseGeometry]) -> list[BaseGeometry]:
    output: list[BaseGeometry] = []
    for feature in polygons:
        geometry = feature if isinstance(feature, BaseGeometry) else shape(feature["geometry"]) if feature.get("geometry") else None
        if geometry is not None and not geometry.is_empty:
            output.append(geometry)
    return output


def rank_tiles(
    tiles: Sequence[Mapping[str, Any]], polygons: Iterable[Mapping[str, Any] | BaseGeometry]
) -> list[dict[str, Any]]:
    polygon_records = list(polygons)
    positives = positive_geometries(polygon_records)
    all_deposits = feature_geometries(polygon_records)
    positive_union = unary_union(positives) if positives else None
    deposit_union = unary_union(all_deposits) if all_deposits else None
    ranked: list[dict[str, Any]] = []
    for source in tiles:
        tile = dict(source)
        footprint = tile_footprint(tile)
        area = (
            footprint.intersection(positive_union).area
            if positive_union is not None and footprint.intersects(positive_union)
            else 0.0
        )
        deposit_area = (
            footprint.intersection(deposit_union).area
            if deposit_union is not None and footprint.intersects(deposit_union)
            else 0.0
        )
        tile["_positive_intersection_area"] = float(area)
        tile["_deposit_intersection_area"] = float(deposit_area)
        tile["_is_hard_negative"] = deposit_area <= 0
        tile["_selection_category"] = (
            "positive" if area > 0 else "hard_negative" if deposit_area <= 0 else "ignore_overlap"
        )
        ranked.append(tile)
    return sorted(ranked, key=lambda item: (-float(item["_positive_intersection_area"]), tile_id(item).casefold()))


def _take_with_budget(candidates: Sequence[dict[str, Any]], count: int, remaining_bytes: int) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    if count <= 0:
        return selected, remaining_bytes
    for tile in candidates:
        size = tile_size(tile)
        if size > remaining_bytes:
            continue
        selected.append(tile)
        remaining_bytes -= size
        if len(selected) >= count:
            break
    return selected, remaining_bytes


def select_tiles(
    tiles: Sequence[Mapping[str, Any]],
    polygons: Iterable[Mapping[str, Any] | BaseGeometry],
    *,
    project: str,
    max_tiles: int | None = None,
    max_total_gb: float | None = None,
    negative_quota: float = 0.25,
    overlap_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """Filter, rank and greedily select a project under count and byte limits."""
    if not project:
        raise ValueError("project is required")
    if max_tiles is not None and max_tiles < 0:
        raise ValueError("max_tiles must be non-negative")
    if not 0 <= negative_quota <= 1:
        raise ValueError("negative_quota must be between 0 and 1")
    if max_total_gb is not None and max_total_gb < 0:
        raise ValueError("max_total_gb must be non-negative")

    matching = [dict(tile) for tile in tiles if project_matches(tile, project)]
    if max_total_gb is not None:
        unknown_sizes = [tile_id(tile) for tile in matching if tile_size(tile) <= 0]
        if unknown_sizes:
            preview = ", ".join(unknown_sizes[:5])
            raise ValueError(f"Cannot enforce byte budget; missing size for: {preview}")
    ranked = rank_tiles(matching, polygons)
    ranked = deduplicate_footprints(ranked, overlap_threshold=overlap_threshold)
    eligible = [tile for tile in ranked if tile["_selection_category"] != "ignore_overlap"]
    limit = min(max_tiles if max_tiles is not None else len(eligible), len(eligible))
    if limit == 0:
        return []
    budget = math.floor(max_total_gb * 1_000_000_000) if max_total_gb is not None else 2**63 - 1
    negatives = [tile for tile in eligible if tile["_is_hard_negative"]]
    positives = [tile for tile in eligible if not tile["_is_hard_negative"]]
    negative_target = min(len(negatives), math.ceil(limit * negative_quota))

    selected_negatives, budget = _take_with_budget(negatives, negative_target, budget)
    selected_positives, budget = _take_with_budget(positives, limit - len(selected_negatives), budget)
    selected = selected_positives + selected_negatives
    used_ids = {tile_id(tile) for tile in selected}
    remainder = [tile for tile in eligible if tile_id(tile) not in used_ids]
    extra, _ = _take_with_budget(remainder, limit - len(selected), budget)
    selected.extend(extra)
    return selected


def select_probe_tiles(
    tiles: Sequence[Mapping[str, Any]],
    *,
    projects: Sequence[str],
    count: int,
    overlap_threshold: float = 0.8,
    max_total_gb: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return exactly ``count`` pairwise co-located footprints per project."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if max_total_gb is not None and max_total_gb < 0:
        raise ValueError("max_total_gb must be non-negative")
    if not 0 < overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in (0, 1]")
    unique_projects = list(dict.fromkeys(projects))
    if len(unique_projects) < 2:
        raise ValueError("probe selection requires at least two projects")
    grouped = {
        project: sorted(
            [dict(tile) for tile in tiles if project_matches(tile, project)],
            key=lambda item: tile_id(item).casefold(),
        )
        for project in unique_projects
    }
    budget_bytes = math.floor(max_total_gb * 1_000_000_000) if max_total_gb is not None else 2**63 - 1
    if max_total_gb is not None:
        unknown_sizes = [tile_id(tile) for project_tiles in grouped.values() for tile in project_tiles if tile_size(tile) <= 0]
        if unknown_sizes:
            raise ValueError(f"Cannot enforce probe byte budget; missing size for: {', '.join(unknown_sizes[:5])}")
    if count == 0:
        return {project: [] for project in unique_projects}

    anchor_project = unique_projects[0]
    anchors = grouped[anchor_project]
    geometries = {
        (project, tile_id(tile)): tile_footprint(tile)
        for project, project_tiles in grouped.items()
        for tile in project_tiles
    }
    selected_sets: list[dict[str, dict[str, Any]]] = []
    used = {project: set() for project in unique_projects[1:]}

    def search(anchor_index: int, used_bytes: int) -> bool:
        if len(selected_sets) == count:
            return True
        if len(anchors) - anchor_index < count - len(selected_sets):
            return False
        anchor = anchors[anchor_index]
        anchor_geometry = geometries[(anchor_project, tile_id(anchor))]
        options: list[list[dict[str, Any]]] = []
        for project in unique_projects[1:]:
            matches = [
                tile for tile in grouped[project]
                if tile_id(tile) not in used[project]
                and footprint_iou(anchor_geometry, geometries[(project, tile_id(tile))]) >= overlap_threshold
            ]
            matches.sort(
                key=lambda tile: (-footprint_iou(anchor_geometry, geometries[(project, tile_id(tile))]), tile_id(tile).casefold())
            )
            if not matches:
                options = []
                break
            options.append(matches)

        for combination in product(*options) if options else ():
            candidate_geometries = [anchor_geometry] + [
                geometries[(project, tile_id(tile))]
                for project, tile in zip(unique_projects[1:], combination)
            ]
            if any(
                footprint_iou(candidate_geometries[left], candidate_geometries[right]) < overlap_threshold
                for left in range(len(candidate_geometries))
                for right in range(left + 1, len(candidate_geometries))
            ):
                continue
            set_bytes = tile_size(anchor) + sum(tile_size(tile) for tile in combination)
            if used_bytes + set_bytes > budget_bytes:
                continue
            selected = {anchor_project: anchor} | dict(zip(unique_projects[1:], combination))
            selected_sets.append(selected)
            for project, tile in zip(unique_projects[1:], combination):
                used[project].add(tile_id(tile))
            if search(anchor_index + 1, used_bytes + set_bytes):
                return True
            for project, tile in zip(unique_projects[1:], combination):
                used[project].remove(tile_id(tile))
            selected_sets.pop()
        return search(anchor_index + 1, used_bytes)

    if not search(0, 0):
        available = len(selected_sets)
        raise ValueError(f"Only {available} complete co-located sets available; requested {count}")
    return {
        project: [selected[project] for selected in selected_sets]
        for project in unique_projects
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Select footprint-safe, budgeted rural LiDAR tiles.")
    parser.add_argument("--tiles", type=Path, required=True, help="TNM tile JSON list or FeatureCollection")
    parser.add_argument("--polygons", type=Path, help="SLIDO GeoJSON used to rank high/moderate intersections")
    parser.add_argument("--project", action="append", required=True, help="Project name; repeat for probe comparison")
    parser.add_argument("--probe", type=int, metavar="N", help="Select N co-located footprints for every project")
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--negative-quota", type=float, default=0.25)
    parser.add_argument("--max-total-gb", type=float)
    parser.add_argument(
        "--overlap-threshold", type=float,
        help="Footprint IoU for dedup (default: 0.8) or probe co-location (default: 0.8)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_tiles = _load_json(args.tiles)
    tiles = raw_tiles.get("items", raw_tiles.get("features", [])) if isinstance(raw_tiles, Mapping) else raw_tiles
    if args.probe is not None:
        result: Any = select_probe_tiles(
            tiles, projects=args.project, count=args.probe,
            overlap_threshold=args.overlap_threshold if args.overlap_threshold is not None else 0.8,
            max_total_gb=args.max_total_gb,
        )
    else:
        if len(args.project) != 1:
            parser.error("normal selection accepts exactly one --project")
        polygon_data = _load_json(args.polygons) if args.polygons else {"features": []}
        polygons = polygon_data.get("features", polygon_data) if isinstance(polygon_data, Mapping) else polygon_data
        result = select_tiles(
            tiles, polygons, project=args.project[0], max_tiles=args.max_tiles,
            max_total_gb=args.max_total_gb, negative_quota=args.negative_quota,
            overlap_threshold=args.overlap_threshold if args.overlap_threshold is not None else 0.8,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if isinstance(result, dict):
        print(", ".join(f"{project}: {len(items)}" for project, items in result.items()))
    else:
        print(f"Selected {len(result)} tiles ({sum(tile_size(tile) for tile in result) / 1e9:.3f} GB)")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
