#!/usr/bin/env python3
"""Select budgeted rural LiDAR tiles without footprint leakage."""
from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from shapely.geometry import box, mapping, shape
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
    overlap_metric: str = "iou",
    max_total_gb: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return exactly ``count`` co-located footprint sets per project.

    ``overlap_metric="iou"`` preserves the original behavior. The explicit
    ``"smaller"`` option uses intersection area divided by the smaller source
    footprint. Every selected record is annotated with the common intersection
    geometry so downstream diagnostics can compare identical ground extents.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if max_total_gb is not None and max_total_gb < 0:
        raise ValueError("max_total_gb must be non-negative")
    if not 0 < overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in (0, 1]")
    if overlap_metric not in {"iou", "smaller"}:
        raise ValueError("overlap_metric must be 'iou' or 'smaller'")

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
    missing_projects = [project for project, values in grouped.items() if not values]
    if missing_projects:
        raise ValueError("No TNM records matched project(s): " + ", ".join(missing_projects))

    budget_bytes = (
        math.floor(max_total_gb * 1_000_000_000)
        if max_total_gb is not None else 2**63 - 1
    )
    if max_total_gb is not None:
        unknown_sizes = [
            tile_id(tile)
            for project_tiles in grouped.values()
            for tile in project_tiles
            if tile_size(tile) <= 0
        ]
        if unknown_sizes:
            raise ValueError(
                "Cannot enforce probe byte budget; missing size for: "
                + ", ".join(unknown_sizes[:5])
            )
    if count == 0:
        return {project: [] for project in unique_projects}

    anchor_project = unique_projects[0]
    anchors = grouped[anchor_project]
    geometries = {
        (project, tile_id(tile)): tile_footprint(tile)
        for project, project_tiles in grouped.items()
        for tile in project_tiles
    }

    def score(left: BaseGeometry, right: BaseGeometry) -> float:
        return footprint_iou(left, right) if overlap_metric == "iou" else overlap_fraction(left, right)

    candidate_sets: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_geometry = geometries[(anchor_project, tile_id(anchor))]
        options: list[list[dict[str, Any]]] = []
        for project in unique_projects[1:]:
            scored_matches: list[tuple[float, float, str, dict[str, Any]]] = []
            for tile in grouped[project]:
                geometry = geometries[(project, tile_id(tile))]
                match_score = score(anchor_geometry, geometry)
                if match_score >= overlap_threshold:
                    scored_matches.append((
                        match_score,
                        footprint_iou(anchor_geometry, geometry),
                        tile_id(tile).casefold(),
                        tile,
                    ))
            scored_matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
            matches = [tile for _, _, _, tile in scored_matches]
            if not matches:
                options = []
                break
            options.append(matches)

        for combination in product(*options) if options else ():
            selected = {anchor_project: anchor} | dict(zip(unique_projects[1:], combination))
            selected_geometries = [
                geometries[(project, tile_id(selected[project]))]
                for project in unique_projects
            ]
            pair_scores: list[float] = []
            pair_ious: list[float] = []
            pair_smaller: list[float] = []
            compatible = True
            for left in range(len(selected_geometries)):
                for right in range(left + 1, len(selected_geometries)):
                    lg = selected_geometries[left]
                    rg = selected_geometries[right]
                    pair_score = score(lg, rg)
                    if pair_score < overlap_threshold:
                        compatible = False
                        break
                    pair_scores.append(pair_score)
                    pair_ious.append(footprint_iou(lg, rg))
                    pair_smaller.append(overlap_fraction(lg, rg))
                if not compatible:
                    break
            if not compatible:
                continue

            common_geometry = selected_geometries[0]
            for geometry in selected_geometries[1:]:
                common_geometry = common_geometry.intersection(geometry)
            if common_geometry.is_empty or common_geometry.area <= 0:
                continue

            set_bytes = sum(tile_size(selected[project]) for project in unique_projects)
            if set_bytes > budget_bytes:
                continue
            candidate_sets.append({
                "selected": selected,
                "bytes": set_bytes,
                "score": min(pair_scores),
                "iou": min(pair_ious),
                "smaller_overlap": min(pair_smaller),
                "intersection": common_geometry,
            })

    selected_candidates: list[dict[str, Any]] = []
    used = {project: set() for project in unique_projects}

    def choose(start_index: int, used_bytes: int) -> bool:
        if len(selected_candidates) == count:
            return True
        needed = count - len(selected_candidates)
        if len(candidate_sets) - start_index < needed:
            return False
        for candidate_index in range(start_index, len(candidate_sets)):
            candidate = candidate_sets[candidate_index]
            if used_bytes + candidate["bytes"] > budget_bytes:
                continue
            selected = candidate["selected"]
            selected_ids = {project: tile_id(selected[project]) for project in unique_projects}
            if any(selected_ids[project] in used[project] for project in unique_projects):
                continue
            selected_candidates.append(candidate)
            for project in unique_projects:
                used[project].add(selected_ids[project])
            if choose(candidate_index + 1, used_bytes + candidate["bytes"]):
                return True
            for project in unique_projects:
                used[project].remove(selected_ids[project])
            selected_candidates.pop()
        return False

    if not choose(0, 0):
        complete_anchor_count = len({
            tile_id(candidate["selected"][anchor_project]) for candidate in candidate_sets
        })
        budget_note = f" within {max_total_gb:g} GB" if max_total_gb is not None else ""
        metric_label = "IoU" if overlap_metric == "iou" else "smaller-footprint overlap"
        raise ValueError(
            f"Could not select {count} unique co-located sets at {metric_label} "
            f">= {overlap_threshold:g}{budget_note}; {complete_anchor_count} anchor "
            "footprints had at least one complete project match"
        )

    output = {project: [] for project in unique_projects}
    for pair_index, candidate in enumerate(selected_candidates, 1):
        pair_id = f"probe_{pair_index:03d}"
        intersection_mapping = mapping(candidate["intersection"])
        for project in unique_projects:
            annotated = dict(candidate["selected"][project])
            annotated.update({
                "_probe_pair_id": pair_id,
                "_probe_overlap_metric": overlap_metric,
                "_probe_overlap_threshold": float(overlap_threshold),
                "_probe_pair_score": float(candidate["score"]),
                "_probe_footprint_iou": float(candidate["iou"]),
                "_probe_smaller_overlap": float(candidate["smaller_overlap"]),
                "_probe_intersection_geometry": intersection_mapping,
                "_probe_intersection_crs": "EPSG:4326",
            })
            output[project].append(annotated)
    return output

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
    parser.add_argument(
        "--probe-overlap-metric",
        choices=("iou", "smaller"),
        default="iou",
        help=("Probe matching metric. 'iou' preserves legacy behavior; "
              "'smaller' requires substantial coverage of the smaller footprint."),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_tiles = _load_json(args.tiles)
    tiles = raw_tiles.get("items", raw_tiles.get("features", [])) if isinstance(raw_tiles, Mapping) else raw_tiles
    if args.probe is not None:
        result: Any = select_probe_tiles(
            tiles, projects=args.project, count=args.probe,
            overlap_threshold=args.overlap_threshold if args.overlap_threshold is not None else 0.8,
            overlap_metric=args.probe_overlap_metric,
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
