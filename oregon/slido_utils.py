#!/usr/bin/env python3
"""Utilities for loading and filtering Oregon SLIDO GeoJSON.

The active Oregon pipeline should import this module rather than depending on
scripts under ``archive/``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

try:
    from shapely import make_valid
except ImportError:  # Shapely < 2 fallback
    make_valid = None


def _property_value(properties: Mapping[str, Any], name: str) -> Any:
    """Return a property case-insensitively."""
    if name in properties:
        return properties[name]
    wanted = name.casefold()
    for key, value in properties.items():
        if str(key).casefold() == wanted:
            return value
    return None


def _repair_geometry(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_valid:
        return geometry
    if make_valid is not None:
        repaired = make_valid(geometry)
    else:
        repaired = geometry.buffer(0)
    return repaired


def load_deposits(
    geojson_path: str | Path,
    description: str | None = "Landslide",
    *,
    repair_invalid: bool = True,
) -> list[tuple[BaseGeometry, dict[str, Any]]]:
    """Load SLIDO Polygon/MultiPolygon features.

    Parameters
    ----------
    geojson_path:
        SLIDO GeoJSON FeatureCollection.
    description:
        Value required in the ``DESCRIPTION`` field. The default keeps only
        true ``Landslide`` deposits and excludes ``Fan`` and
        ``Talus-Colluvium``. Pass ``None`` to retain every deposit type.
    repair_invalid:
        Attempt to repair invalid polygon geometry.
    """
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"SLIDO GeoJSON not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Expected GeoJSON FeatureCollection: {path}")

    deposits: list[tuple[BaseGeometry, dict[str, Any]]] = []
    descriptions: Counter[str] = Counter()
    skipped_type = 0
    skipped_description = 0
    skipped_invalid = 0

    for feature in data.get("features", []):
        properties = dict(feature.get("properties") or {})
        actual_description = _property_value(properties, "DESCRIPTION")
        descriptions[str(actual_description or "<missing>")] += 1

        if description is not None:
            if str(actual_description or "").strip().casefold() != description.strip().casefold():
                skipped_description += 1
                continue

        geometry_mapping = feature.get("geometry")
        if not geometry_mapping or geometry_mapping.get("type") not in {"Polygon", "MultiPolygon"}:
            skipped_type += 1
            continue

        try:
            geometry = shape(geometry_mapping)
        except Exception:
            skipped_invalid += 1
            continue

        if geometry.is_empty:
            skipped_invalid += 1
            continue

        if repair_invalid and not geometry.is_valid:
            try:
                geometry = _repair_geometry(geometry)
            except Exception:
                skipped_invalid += 1
                continue

        if geometry.is_empty or not geometry.is_valid:
            skipped_invalid += 1
            continue

        deposits.append((geometry, properties))

    filter_text = description if description is not None else "ALL"
    print(f"Loaded {len(deposits)} SLIDO polygon(s) from {path}")
    print(f"  DESCRIPTION filter: {filter_text}")
    print(f"  Source DESCRIPTION counts: {dict(descriptions)}")
    if skipped_description:
        print(f"  Excluded by DESCRIPTION filter: {skipped_description}")
    if skipped_type:
        print(f"  Excluded non-polygon features: {skipped_type}")
    if skipped_invalid:
        print(f"  Excluded invalid/empty geometries: {skipped_invalid}")

    return deposits


def iter_properties(geojson_path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield feature properties without loading Shapely geometry."""
    with Path(geojson_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for feature in data.get("features", []):
        yield dict(feature.get("properties") or {})


def bbox_from_geojson(geojson_path: str | Path) -> tuple[float, float, float, float]:
    """Return the bounding box of valid polygon features."""
    deposits = load_deposits(geojson_path, description=None)
    if not deposits:
        raise ValueError("No valid polygon geometry in GeoJSON")
    min_x = min(geometry.bounds[0] for geometry, _ in deposits)
    min_y = min(geometry.bounds[1] for geometry, _ in deposits)
    max_x = max(geometry.bounds[2] for geometry, _ in deposits)
    max_y = max(geometry.bounds[3] for geometry, _ in deposits)
    return min_x, min_y, max_x, max_y
