#!/usr/bin/env python3
"""Create a boundary-aware, polygon-group-safe training manifest.

This script does not modify NPZ patches or the original manifest. It:
- recomputes positive/ignore fractions from mask values 1/255;
- detects true in-patch label boundaries without treating patch edges as boundaries;
- intersects each patch footprint with SLIDO polygons to obtain patch-specific IDs;
- removes training rows belonging to landslide polygons also present in validation/test;
- caps redundant near-full/full positive interiors per landslide group in TRAIN only;
- preserves every validation/test row unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pyproj import CRS, Transformer
from scipy.ndimage import binary_erosion
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree

NEW_FIELDS = (
    "patch_landslide_ids",
    "patch_polygon_keys",
    "contains_positive_boundary",
    "boundary_pixel_fraction",
    "boundary_of_positive_fraction",
    "coverage_class",
    "selected_for_manifest",
    "sampling_reason",
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def property_value(properties: dict[str, Any], *names: str) -> Any:
    folded = {str(key).casefold(): value for key, value in properties.items()}
    for name in names:
        value = folded.get(name.casefold())
        if value not in (None, ""):
            return value
    return ""


def stable_polygon_key(geometry: BaseGeometry, properties: dict[str, Any]) -> str:
    explicit = property_value(properties, "UNIQUE_ID", "OBJECTID", "landslide_id")
    if explicit not in (None, ""):
        return str(explicit)
    return hashlib.sha1(geometry.wkb).hexdigest()


@dataclass(frozen=True)
class PolygonRecord:
    geometry_wgs84: BaseGeometry
    polygon_key: str
    landslide_id: str
    confidence_class: str
    event_year: int | None


@dataclass
class SpatialIndex:
    records: list[PolygonRecord]
    geometries: list[BaseGeometry]
    tree: STRtree
    geometry_id_to_index: dict[int, int]

    def query(self, footprint: BaseGeometry) -> list[PolygonRecord]:
        result = self.tree.query(footprint)
        matches: list[PolygonRecord] = []
        for item in result:
            if isinstance(item, (int, np.integer)):
                index = int(item)
            else:
                index = self.geometry_id_to_index.get(id(item), -1)
                if index < 0:
                    # Shapely 1.x may return equivalent but non-identical objects.
                    index = next(
                        (i for i, candidate in enumerate(self.geometries) if candidate.equals(item)),
                        -1,
                    )
            if index >= 0 and self.geometries[index].intersects(footprint):
                matches.append(self.records[index])
        return matches


def normalize_confidence(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if "high" in text:
        return "high"
    if "moderate" in text or "medium" in text:
        return "moderate"
    return "other"


def parse_event_year(properties: dict[str, Any]) -> int | None:
    candidates = (
        "EVENT_YEAR", "event_year", "YEAR", "year", "EVENT_DATE",
        "event_date", "LANDSLIDE_DATE", "landslide_date", "DATE", "date",
    )
    for name in candidates:
        raw = property_value(properties, name)
        if raw in (None, ""):
            continue
        import re
        match = re.search(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)", str(raw))
        if match:
            return int(match.group(0))
    return None


def load_slido(path: Path) -> list[PolygonRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", []) if isinstance(payload, dict) else []
    records: list[PolygonRecord] = []
    for feature in features:
        if not isinstance(feature, dict) or not feature.get("geometry"):
            continue
        properties = feature.get("properties") or {}
        description = str(property_value(properties, "DESCRIPTION", "description")).strip()
        if description and description.casefold() != "landslide":
            continue
        geometry = shape(feature["geometry"])
        if geometry.is_empty:
            continue
        landslide_id = str(property_value(properties, "UNIQUE_ID", "OBJECTID", "landslide_id"))
        records.append(
            PolygonRecord(
                geometry_wgs84=geometry,
                polygon_key=stable_polygon_key(geometry, properties),
                landslide_id=landslide_id,
                confidence_class=normalize_confidence(
                    property_value(properties, "confidence_class", "CONFIDENCE")
                ),
                event_year=parse_event_year(properties),
            )
        )
    if not records:
        raise ValueError(f"No landslide polygons found in {path}")
    return records


def build_spatial_index(records: list[PolygonRecord], target_crs: str) -> SpatialIndex:
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(target_crs), always_xy=True)
    transformed_records: list[PolygonRecord] = []
    geometries: list[BaseGeometry] = []
    for record in records:
        geometry = shapely_transform(transformer.transform, record.geometry_wgs84)
        if geometry.is_empty:
            continue
        transformed_records.append(
            PolygonRecord(
                geometry_wgs84=geometry,
                polygon_key=record.polygon_key,
                landslide_id=record.landslide_id,
                confidence_class=record.confidence_class,
                event_year=record.event_year,
            )
        )
        geometries.append(geometry)
    tree = STRtree(geometries)
    return SpatialIndex(
        records=transformed_records,
        geometries=geometries,
        tree=tree,
        geometry_id_to_index={id(geometry): index for index, geometry in enumerate(geometries)},
    )


def boundary_metrics(mask: np.ndarray) -> tuple[float, float, bool, float, float, str]:
    positive = mask == 1
    ignored = mask == 255
    positive_fraction = float(positive.mean())
    ignore_fraction = float(ignored.mean())

    # border_value=1 prevents the crop edge from being counted as a label boundary.
    eroded = binary_erosion(
        positive,
        structure=np.ones((3, 3), dtype=bool),
        border_value=1,
    )
    boundary = positive & ~eroded
    boundary_pixels = int(boundary.sum())
    positive_pixels = int(positive.sum())
    boundary_pixel_fraction = float(boundary_pixels / mask.size)
    boundary_of_positive_fraction = float(boundary_pixels / positive_pixels) if positive_pixels else 0.0
    contains_boundary = bool(boundary_pixels)

    if positive_fraction == 0:
        coverage_class = "negative"
    elif positive_fraction < 0.01:
        coverage_class = "trace"
    elif positive_fraction < 0.10:
        coverage_class = "low_coverage_positive"
    elif positive_fraction < 0.90:
        coverage_class = "mixed_positive"
    elif positive_fraction < 0.99:
        coverage_class = "near_full_positive"
    else:
        coverage_class = "full_positive"
    return (
        positive_fraction,
        ignore_fraction,
        contains_boundary,
        boundary_pixel_fraction,
        boundary_of_positive_fraction,
        coverage_class,
    )


def deterministic_rank(row: dict[str, Any], seed: int) -> tuple[float, float, float, str]:
    digest = hashlib.sha1(f"{seed}:{row.get('patch_id', '')}".encode("utf-8")).hexdigest()
    # Prefer more contextual near-full patches, then more boundary and ground support.
    return (
        parse_float(row.get("positive_fraction"), 1.0),
        -parse_float(row.get("boundary_pixel_fraction"), 0.0),
        -parse_float(row.get("ground_fraction"), 0.0),
        digest,
    )


def group_key(row: dict[str, Any]) -> str:
    keys = str(row.get("patch_polygon_keys", "")).strip()
    if keys:
        return keys
    tile_name = str(row.get("tile_name", "")).strip()
    return f"tile:{tile_name or '<unknown>'}"


def select_training_rows(
    rows: list[dict[str, Any]],
    *,
    max_near_full_per_group: int,
    max_full_per_group: int,
    seed: int,
    remove_cross_split_polygon_overlap: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_by_polygon: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if parse_float(row.get("positive_fraction"), 0.0) <= 0:
            continue
        for key in str(row.get("patch_polygon_keys", "")).split(";"):
            key = key.strip()
            if key:
                split_by_polygon[key].add(str(row.get("split", "")))

    contaminating_keys = {
        key
        for key, splits in split_by_polygon.items()
        if "train" in splits and any(split != "train" for split in splits)
    }

    selected_ids: set[str] = set()
    reasons: dict[str, str] = {}

    # Preserve every non-training row exactly as an evaluation record.
    for row in rows:
        patch_id = str(row.get("patch_id", ""))
        if str(row.get("split", "")) != "train":
            selected_ids.add(patch_id)
            reasons[patch_id] = "preserved_nontraining_split"

    eligible_train: list[dict[str, Any]] = []
    dropped_cross_split = 0
    for row in rows:
        if str(row.get("split", "")) != "train":
            continue
        patch_id = str(row.get("patch_id", ""))
        row_keys = {key for key in str(row.get("patch_polygon_keys", "")).split(";") if key}
        is_positive = parse_float(row.get("positive_fraction"), 0.0) > 0
        if (
            remove_cross_split_polygon_overlap
            and is_positive
            and row_keys & contaminating_keys
        ):
            reasons[patch_id] = "dropped_train_polygon_present_in_validation_or_test"
            dropped_cross_split += 1
            continue
        eligible_train.append(row)

    ordinary: list[dict[str, Any]] = []
    near_full: dict[str, list[dict[str, Any]]] = defaultdict(list)
    full: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in eligible_train:
        coverage = str(row.get("coverage_class", ""))
        if coverage == "near_full_positive":
            near_full[group_key(row)].append(row)
        elif coverage == "full_positive":
            full[group_key(row)].append(row)
        else:
            ordinary.append(row)

    for row in ordinary:
        patch_id = str(row.get("patch_id", ""))
        selected_ids.add(patch_id)
        reasons[patch_id] = "kept_all_negative_or_boundary_context"

    def take_with_per_polygon_cap(
        groups: dict[str, list[dict[str, Any]]],
        cap: int,
        kept_reason: str,
        dropped_reason: str,
    ) -> None:
        candidates = [row for members in groups.values() for row in members]
        ranked = sorted(candidates, key=lambda row: deterministic_rank(row, seed))
        counts: Counter[str] = Counter()
        for row in ranked:
            patch_id = str(row.get("patch_id", ""))
            keys = [key for key in str(row.get("patch_polygon_keys", "")).split(";") if key]
            if not keys:
                keys = [group_key(row)]
            if cap > 0 and all(counts[key] < cap for key in keys):
                selected_ids.add(patch_id)
                reasons[patch_id] = kept_reason
                for key in keys:
                    counts[key] += 1
            else:
                reasons[patch_id] = dropped_reason

    take_with_per_polygon_cap(
        near_full,
        max_near_full_per_group,
        "kept_near_full_within_group_cap",
        "dropped_redundant_near_full_within_group",
    )
    take_with_per_polygon_cap(
        full,
        max_full_per_group,
        "kept_full_interior_within_group_cap",
        "dropped_redundant_full_interior_within_group",
    )

    selected: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        patch_id = str(copy.get("patch_id", ""))
        copy["selected_for_manifest"] = "true" if patch_id in selected_ids else "false"
        copy["sampling_reason"] = reasons.get(patch_id, "not_selected")
        audit_rows.append(copy)
        if patch_id in selected_ids:
            selected.append(copy)

    report = {
        "input_rows": len(rows),
        "output_rows": len(selected),
        "input_split_counts": dict(Counter(str(row.get("split", "")) for row in rows)),
        "output_split_counts": dict(Counter(str(row.get("split", "")) for row in selected)),
        "input_coverage_counts": dict(Counter(str(row.get("coverage_class", "")) for row in rows)),
        "output_coverage_counts": dict(Counter(str(row.get("coverage_class", "")) for row in selected)),
        "input_train_coverage_counts": dict(
            Counter(str(row.get("coverage_class", "")) for row in rows if row.get("split") == "train")
        ),
        "output_train_coverage_counts": dict(
            Counter(str(row.get("coverage_class", "")) for row in selected if row.get("split") == "train")
        ),
        "cross_split_polygon_keys": sorted(contaminating_keys),
        "cross_split_polygon_key_count": len(contaminating_keys),
        "dropped_train_rows_for_polygon_split_overlap": dropped_cross_split,
        "sampling_reason_counts": dict(Counter(row["sampling_reason"] for row in audit_rows)),
        "parameters": {
            "max_near_full_per_landslide_group": max_near_full_per_group,
            "max_full_per_landslide_group": max_full_per_group,
            "seed": seed,
            "remove_cross_split_polygon_overlap": remove_cross_split_polygon_overlap,
        },
    }
    return selected, {"report": report, "audit_rows": audit_rows}


def resolve_manifest(dataset_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested if requested.is_absolute() else dataset_dir / requested
    qc = dataset_dir / "patches_qc.csv"
    return qc if qc.exists() else dataset_dir / "patches.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--slido", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("patches_boundary_aware.csv"))
    parser.add_argument("--audit-output", type=Path, default=Path("boundary_sampling_audit.csv"))
    parser.add_argument("--report", type=Path, default=Path("boundary_sampling_report.json"))
    parser.add_argument("--max-near-full-per-landslide", type=int, default=4)
    parser.add_argument("--max-full-per-landslide", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cross-split-landslide-overlap",
        action="store_true",
        help="Do not remove training rows from polygons also present in validation/test (not recommended).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_near_full_per_landslide < 0 or args.max_full_per_landslide < 0:
        parser.error("Per-landslide caps must be non-negative")

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = resolve_manifest(dataset_dir, args.manifest).resolve()
    slido_path = args.slido.resolve()
    if not manifest_path.exists():
        parser.error(f"Manifest not found: {manifest_path}")
    if not slido_path.exists():
        parser.error(f"SLIDO file not found: {slido_path}")

    rows, original_fields = read_csv(manifest_path)
    if not rows:
        parser.error(f"Manifest contains no rows: {manifest_path}")
    required = {"patch_id", "patch_path", "split", "x_min", "y_min", "x_max", "y_max", "crs"}
    missing = sorted(required - set(original_fields))
    if missing:
        parser.error("Manifest is missing required columns: " + ", ".join(missing))

    polygons = load_slido(slido_path)
    indexes: dict[str, SpatialIndex] = {}

    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        patch_path = dataset_dir / row["patch_path"]
        if not patch_path.exists():
            raise FileNotFoundError(f"Patch file not found for {row.get('patch_id')}: {patch_path}")
        with np.load(patch_path) as data:
            mask = data["mask"]
        (
            positive_fraction,
            ignore_fraction,
            contains_boundary,
            boundary_pixel_fraction,
            boundary_of_positive_fraction,
            coverage_class,
        ) = boundary_metrics(mask)

        crs = str(row.get("crs", "")).strip()
        if crs not in indexes:
            indexes[crs] = build_spatial_index(polygons, crs)
        footprint = box(
            parse_float(row["x_min"]),
            parse_float(row["y_min"]),
            parse_float(row["x_max"]),
            parse_float(row["y_max"]),
        )
        matches = indexes[crs].query(footprint)
        acquisition_year = int(parse_float(
            row.get("lidar_acquisition_year") or row.get("lidar_year"), 0.0
        )) or None
        eligible_matches = [
            record
            for record in matches
            if record.confidence_class in {"high", "moderate"}
            and not (
                acquisition_year is not None
                and record.event_year is not None
                and record.event_year > acquisition_year
            )
        ]
        positive_pixels = mask == 1
        patch_transform = from_bounds(
            parse_float(row["x_min"]),
            parse_float(row["y_min"]),
            parse_float(row["x_max"]),
            parse_float(row["y_max"]),
            mask.shape[1],
            mask.shape[0],
        )
        positive_matches: list[PolygonRecord] = []
        if positive_pixels.any():
            for record in eligible_matches:
                polygon_pixels = rasterize(
                    [(mapping(record.geometry_wgs84), 1)],
                    out_shape=mask.shape,
                    transform=patch_transform,
                    fill=0,
                    default_value=1,
                    dtype="uint8",
                ).astype(bool)
                if np.any(polygon_pixels & positive_pixels):
                    positive_matches.append(record)
        polygon_keys = sorted({record.polygon_key for record in positive_matches})
        landslide_ids = sorted(
            {record.landslide_id for record in positive_matches if record.landslide_id}
        )

        copy: dict[str, Any] = dict(row)
        # Recomputed values are authoritative for this sampling audit.
        copy["positive_fraction"] = f"{positive_fraction:.12g}"
        copy["ignore_fraction"] = f"{ignore_fraction:.12g}"
        copy["patch_landslide_ids"] = ";".join(landslide_ids)
        copy["patch_polygon_keys"] = ";".join(polygon_keys)
        copy["contains_positive_boundary"] = "true" if contains_boundary else "false"
        copy["boundary_pixel_fraction"] = f"{boundary_pixel_fraction:.12g}"
        copy["boundary_of_positive_fraction"] = f"{boundary_of_positive_fraction:.12g}"
        copy["coverage_class"] = coverage_class
        enriched.append(copy)
        if index % 100 == 0 or index == len(rows):
            print(f"Analyzed {index}/{len(rows)} patches")

    selected, payload = select_training_rows(
        enriched,
        max_near_full_per_group=args.max_near_full_per_landslide,
        max_full_per_group=args.max_full_per_landslide,
        seed=args.seed,
        remove_cross_split_polygon_overlap=not args.allow_cross_split_landslide_overlap,
    )
    audit_rows = payload["audit_rows"]
    report = payload["report"]
    report.update(
        {
            "dataset_dir": str(dataset_dir),
            "input_manifest": str(manifest_path),
            "slido": str(slido_path),
            "slido_polygon_count": len(polygons),
        }
    )

    output = args.output if args.output.is_absolute() else dataset_dir / args.output
    audit_output = args.audit_output if args.audit_output.is_absolute() else dataset_dir / args.audit_output
    report_path = args.report if args.report.is_absolute() else dataset_dir / args.report
    fields = list(original_fields)
    for field in NEW_FIELDS:
        if field not in fields:
            fields.append(field)
    write_csv(output, selected, fields)
    write_csv(audit_output, audit_rows, fields)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nBOUNDARY-AWARE SAMPLING SUMMARY")
    print(json.dumps(report, indent=2))
    print(f"\nTraining manifest: {output}")
    print(f"Audit manifest:    {audit_output}")
    print(f"Report:            {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
