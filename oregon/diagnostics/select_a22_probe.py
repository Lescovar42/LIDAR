#!/usr/bin/env python3
"""Select a spatially diverse, A22-only Tillamook probe sample.

The output remains a plain JSON list so it can be consumed by download_tiles.py.
Selection annotations are added to each copied TNM record for provenance.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
from shapely.geometry.base import BaseGeometry

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from select_tiles import deduplicate_footprints, rank_tiles, tile_footprint, tile_id, tile_size
from tnm_utils import project_matches

DEFAULT_PROJECT = "USGS_LPC_OR_WesternWildfires_A22"
DEFAULT_BAD_TILE = "USGS_LPC_OR_WesternWildfires_A22_s04380w13950.laz"


def load_tile_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, Mapping):
        records = next(
            (raw[key] for key in ("items", "tiles", "records") if isinstance(raw.get(key), list)),
            None,
        )
        if records is None:
            raise ValueError(f"{path} does not contain a tile-record list")
    else:
        raise ValueError(f"{path} must contain a JSON list or object with items/tiles/records")
    return [dict(record) for record in records if isinstance(record, Mapping)]


def load_geojson_features(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("type") != "FeatureCollection":
        raise ValueError(f"Expected a GeoJSON FeatureCollection: {path}")
    output: list[dict[str, Any]] = []
    for feature in raw.get("features", []):
        if not isinstance(feature, Mapping) or not isinstance(feature.get("geometry"), Mapping):
            continue
        properties = feature.get("properties") or {}
        description = next(
            (
                value
                for key, value in properties.items()
                if str(key).casefold() == "description"
            ),
            None,
        )
        if str(description or "").strip().casefold() != "landslide":
            continue
        output.append(dict(feature))
    return output


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


def load_exclusions(path: Path) -> set[str]:
    excluded = {DEFAULT_BAD_TILE.casefold()}
    if not path.exists():
        return excluded
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            excluded.add(Path(value).name.casefold())
    return excluded


def _centroid_xy(geometry: BaseGeometry) -> tuple[float, float]:
    point = geometry.centroid
    return float(point.x), float(point.y)


def _quality(tile: Mapping[str, Any], *, positive: bool, max_positive_area: float) -> float:
    if positive:
        area = max(0.0, float(tile.get("_positive_intersection_area", 0.0)))
        return math.log1p(area) / math.log1p(max_positive_area) if max_positive_area > 0 else 0.0
    return 1.0


def spread_select(
    candidates: Sequence[dict[str, Any]],
    count: int,
    *,
    already_selected: Sequence[dict[str, Any]] = (),
    remaining_bytes: int = 2**63 - 1,
    positive: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Greedily maximize centroid separation while retaining label relevance."""
    if count <= 0 or not candidates:
        return [], remaining_bytes

    pool = [dict(item) for item in candidates]
    geometries = {tile_id(item): tile_footprint(item) for item in pool}
    reference_geometries = [tile_footprint(item) for item in already_selected]
    all_geometries = list(geometries.values()) + reference_geometries
    minx = min(geometry.bounds[0] for geometry in all_geometries)
    miny = min(geometry.bounds[1] for geometry in all_geometries)
    maxx = max(geometry.bounds[2] for geometry in all_geometries)
    maxy = max(geometry.bounds[3] for geometry in all_geometries)
    diagonal = max(math.hypot(maxx - minx, maxy - miny), 1e-12)
    max_positive_area = max(
        (float(item.get("_positive_intersection_area", 0.0)) for item in pool),
        default=0.0,
    )

    selected: list[dict[str, Any]] = []
    selected_centroids = [_centroid_xy(geometry) for geometry in reference_geometries]

    while pool and len(selected) < count:
        affordable = [item for item in pool if tile_size(item) <= remaining_bytes]
        if not affordable:
            break

        def score(item: Mapping[str, Any]) -> tuple[float, float, str]:
            geometry = geometries[tile_id(item)]
            centroid = _centroid_xy(geometry)
            if selected_centroids:
                spread = min(
                    math.hypot(centroid[0] - x, centroid[1] - y)
                    for x, y in selected_centroids
                ) / diagonal
            else:
                center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)
                spread = math.hypot(centroid[0] - center[0], centroid[1] - center[1]) / diagonal
            quality = _quality(item, positive=positive, max_positive_area=max_positive_area)
            # Spatial spread dominates; positive-overlap area only breaks weak ties.
            combined = 0.80 * spread + 0.20 * quality
            return combined, quality, tile_id(item).casefold()

        chosen = max(affordable, key=score)
        selected.append(chosen)
        remaining_bytes -= tile_size(chosen)
        selected_centroids.append(_centroid_xy(geometries[tile_id(chosen)]))
        pool.remove(chosen)

    return selected, remaining_bytes


def select_representative_tiles(
    tiles: Sequence[Mapping[str, Any]],
    features: Iterable[Mapping[str, Any]],
    *,
    project: str,
    count: int,
    negative_quota: float,
    max_total_gb: float,
    overlap_threshold: float,
    min_footprint_area_ratio: float,
    excluded_filenames: set[str],
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0 <= negative_quota <= 1:
        raise ValueError("negative_quota must be between 0 and 1")
    if max_total_gb <= 0:
        raise ValueError("max_total_gb must be positive")
    if not 0 < overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in (0, 1]")
    if not 0 < min_footprint_area_ratio <= 1:
        raise ValueError("min_footprint_area_ratio must be in (0, 1]")

    matching = [
        dict(tile)
        for tile in tiles
        if project_matches(tile, project)
        and record_filename(tile).casefold() not in excluded_filenames
    ]
    if not matching:
        raise ValueError(f"No non-excluded TNM records matched project {project!r}")

    ranked = rank_tiles(matching, features)
    ranked = deduplicate_footprints(ranked, overlap_threshold=overlap_threshold)
    areas = np.asarray([tile_footprint(tile).area for tile in ranked], dtype=float)
    median_area = float(np.median(areas))
    minimum_area = median_area * min_footprint_area_ratio
    full_footprints = [tile for tile in ranked if tile_footprint(tile).area >= minimum_area]
    edge_filtered = len(ranked) - len(full_footprints)

    eligible = [
        tile
        for tile in full_footprints
        if tile.get("_selection_category") in {"positive", "hard_negative"}
    ]
    positives = [tile for tile in eligible if tile.get("_selection_category") == "positive"]
    negatives = [tile for tile in eligible if tile.get("_selection_category") == "hard_negative"]
    if not positives:
        raise ValueError("No A22 footprints intersect high/moderate SLIDO landslides")

    unknown_sizes = [record_filename(tile) for tile in eligible if tile_size(tile) <= 0]
    if unknown_sizes:
        raise ValueError(
            "Cannot enforce byte budget because size metadata is missing for: "
            + ", ".join(unknown_sizes[:5])
        )

    budget = math.floor(max_total_gb * 1_000_000_000)
    negative_target = min(len(negatives), math.ceil(count * negative_quota))
    positive_target = min(len(positives), count - negative_target)

    selected_positive, budget = spread_select(
        positives,
        positive_target,
        remaining_bytes=budget,
        positive=True,
    )
    selected_negative, budget = spread_select(
        negatives,
        negative_target,
        already_selected=selected_positive,
        remaining_bytes=budget,
        positive=False,
    )
    selected = selected_positive + selected_negative

    selected_ids = {tile_id(tile) for tile in selected}
    remainder = [tile for tile in eligible if tile_id(tile) not in selected_ids]
    # Fill quota shortfalls without changing the label semantics of existing picks.
    fill, budget = spread_select(
        remainder,
        count - len(selected),
        already_selected=selected,
        remaining_bytes=budget,
        positive=False,
    )
    selected.extend(fill)

    if len(selected) < count:
        raise ValueError(
            f"Only {len(selected)} representative A22 tiles fit the filters and "
            f"{max_total_gb:g} GB budget; requested {count}"
        )

    output: list[dict[str, Any]] = []
    for rank, source in enumerate(selected, 1):
        tile = dict(source)
        tile["_probe_selected_rank"] = rank
        tile["_probe_design"] = "A22_only_spatially_diverse"
        tile["_probe_project"] = project
        tile["_probe_minimum_footprint_area"] = minimum_area
        tile["_probe_edge_records_filtered"] = edge_filtered
        output.append(tile)
    return output


def write_aria2_input(
    path: Path,
    selection: Sequence[Mapping[str, Any]],
    *,
    download_dir: Path,
) -> None:
    destination = download_dir.resolve().as_posix()
    lines: list[str] = []
    for record in selection:
        url = str(
            record.get("downloadURL")
            or record.get("downloadUrl")
            or record.get("download_url")
            or ""
        )
        if not url:
            raise ValueError(f"Selected record lacks downloadURL: {tile_id(record)}")
        lines.extend([url, f"  dir={destination}", f"  out={record_filename(record)}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a representative A22-only Tillamook resolution probe."
    )
    parser.add_argument("--tiles", type=Path, default=OREGON_DIR / "regions/tillamook_tnm.json")
    parser.add_argument("--slido", type=Path, default=OREGON_DIR / "slido_tillamook.geojson")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--negative-quota", type=float, default=0.25)
    parser.add_argument("--max-total-gb", type=float, default=8.0)
    parser.add_argument("--overlap-threshold", type=float, default=0.8)
    parser.add_argument("--min-footprint-area-ratio", type=float, default=0.5)
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=OREGON_DIR / "regions/tillamook_a22_probe_exclusions.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OREGON_DIR / "regions/tillamook_a22_probe_selection.json",
    )
    parser.add_argument(
        "--aria2-input",
        type=Path,
        default=OREGON_DIR / "regions/tillamook_a22_probe_aria2.txt",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=OREGON_DIR
        / "probe_lidar/tillamook_probe/USGS_LPC_OR_WesternWildfires_A22",
    )
    args = parser.parse_args()

    if not args.tiles.is_file():
        parser.error(f"Missing TNM records: {args.tiles}")
    if not args.slido.is_file():
        parser.error(f"Missing SLIDO GeoJSON: {args.slido}")

    try:
        selection = select_representative_tiles(
            load_tile_records(args.tiles),
            load_geojson_features(args.slido),
            project=args.project,
            count=args.count,
            negative_quota=args.negative_quota,
            max_total_gb=args.max_total_gb,
            overlap_threshold=args.overlap_threshold,
            min_footprint_area_ratio=args.min_footprint_area_ratio,
            excluded_filenames=load_exclusions(args.exclude_file),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        write_aria2_input(args.aria2_input, selection, download_dir=args.download_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    categories: dict[str, int] = {}
    for tile in selection:
        category = str(tile.get("_selection_category") or "unknown")
        categories[category] = categories.get(category, 0) + 1
    total_bytes = sum(tile_size(tile) for tile in selection)

    print(f"Selected: {len(selection)} A22 tile(s)")
    print(f"Categories: {categories}")
    print(f"Catalog byte total: {total_bytes / 1e9:.3f} GB")
    print(f"Selection: {args.output}")
    print(f"aria2 input: {args.aria2_input}")
    print(f"Download directory: {args.download_dir.resolve()}")
    print("No project or cell size was pinned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
