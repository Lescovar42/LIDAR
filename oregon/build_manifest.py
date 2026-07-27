"""
Oregon SLIDO + 3DEP LiDAR dataset orchestrator.

Design goals:
- Oregon-first pilot workflow.
- Metadata/manifests are generated before any large download.
- Downloads are opt-in via --download.
- Publication dates are never treated as acquisition dates by default.
- One manifest row is written per intersecting landslide/tile pair.
- Unknown or same-year temporal relationships are quarantined as "uncertain".

This script expects the existing pipeline stages:
    03_discover_3dep_tiles.py
    04_select_tile_subset.py
    05_download_tile_subset.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ACQUISITION_FIELDS = (
    "acquisitionDate",
    "acquisitionStartDate",
    "acquisitionEndDate",
    "dateAcquired",
    "collectDate",
    "collectionDate",
    "collectionStartDate",
    "collectionEndDate",
    "beginPosition",
    "endPosition",
)

PUBLICATION_FIELDS = ("publicationDate", "dateCreated")

LANDSLIDE_DATE_FIELDS = (
    "DATE_MOVE",
    "EVENT_DATE",
    "LANDSLIDE_DATE",
)

LANDSLIDE_ID_FIELDS = (
    "UNIQUE_ID",
    "UNIQUEID",
    "OBJECTID",
    "FID",
)

DOWNLOAD_URL_FIELDS = (
    "downloadURL",
    "downloadUrl",
    "download_url",
    "url",
    "urls",
)

TILE_ID_FIELDS = (
    "sourceId",
    "id",
    "tileId",
    "title",
)


@dataclass(frozen=True)
class PartialDate:
    value: date
    precision: str  # "day" or "year"
    raw: str

    @property
    def year(self) -> int:
        return self.value.year


@dataclass(frozen=True)
class TemporalDecision:
    status: str  # accepted, rejected, uncertain
    reason: str


def load_stage_module(name: str, path: Path):
    """Load one local pipeline stage without deprecated load_module()."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required pipeline stage not found: {path}\n"
            "Pass --pipeline-dir pointing to the directory containing the three stage scripts."
        )

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_partial_date(value: Any) -> PartialDate | None:
    """Parse a full ISO-like date or a four-digit year.

    This intentionally avoids guessing ambiguous day/month formats.
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "unknown", "nan"}:
        return None

    # Exact year only, including values serialized as 2019.0.
    match = re.fullmatch(r"(18|19|20|21)\d{2}(?:\.0+)?", text)
    if match:
        year = int(text[:4])
        return PartialDate(date(year, 1, 1), "year", text)

    # Accept leading ISO date, including timestamps such as 2019-10-14T00:00:00Z.
    match = re.match(r"^((?:18|19|20|21)\d{2})-(\d{1,2})-(\d{1,2})", text)
    if match:
        try:
            parsed = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return PartialDate(parsed, "day", text)
        except ValueError:
            return None

    # A few metadata services use compact YYYYMMDD.
    match = re.fullmatch(r"((?:18|19|20|21)\d{2})(\d{2})(\d{2})", text)
    if match:
        try:
            parsed = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return PartialDate(parsed, "day", text)
        except ValueError:
            return None

    return None


def first_present(mapping: Mapping[str, Any], fields: Iterable[str]) -> tuple[str | None, Any]:
    for field in fields:
        value = mapping.get(field)
        if value not in (None, "", []):
            return field, value
    return None, None


def extract_lidar_date(
    tile: Mapping[str, Any],
    acquisition_fields: Sequence[str],
    allow_publication_fallback: bool,
) -> tuple[PartialDate | None, str | None, str]:
    """Return the first parseable acquisition date and its provenance."""
    for field in acquisition_fields:
        parsed = parse_partial_date(tile.get(field))
        if parsed:
            return parsed, field, "acquisition"

    # Some APIs nest metadata. Check one level without pretending arbitrary fields are dates.
    for container_name in ("metadata", "properties", "project"):
        nested = tile.get(container_name)
        if isinstance(nested, Mapping):
            for field in acquisition_fields:
                parsed = parse_partial_date(nested.get(field))
                if parsed:
                    return parsed, f"{container_name}.{field}", "acquisition"

    if allow_publication_fallback:
        for field in PUBLICATION_FIELDS:
            parsed = parse_partial_date(tile.get(field))
            if parsed:
                return parsed, field, "publication_fallback"

    return None, None, "missing"


def extract_landslide_date(props: Mapping[str, Any]) -> tuple[PartialDate | None, str | None]:
    for field in LANDSLIDE_DATE_FIELDS:
        parsed = parse_partial_date(props.get(field))
        if parsed:
            return parsed, field

    parsed_year = parse_partial_date(props.get("YEAR"))
    if not parsed_year:
        return None, None

    # Upgrade YEAR to a full date only when numeric MONTH and DAY are both valid.
    try:
        month = int(props.get("MONTH"))
        day = int(props.get("DAY"))
        parsed = date(parsed_year.year, month, day)
        return PartialDate(parsed, "day", f"{parsed.isoformat()}"), "YEAR+MONTH+DAY"
    except (TypeError, ValueError):
        return parsed_year, "YEAR"


def classify_temporal_relation(
    lidar_date: PartialDate | None,
    landslide_date: PartialDate | None,
    lidar_date_kind: str,
) -> TemporalDecision:
    if lidar_date is None:
        return TemporalDecision("uncertain", "missing_lidar_acquisition_date")

    if lidar_date_kind == "publication_fallback":
        return TemporalDecision("uncertain", "publication_date_is_not_acquisition_date")

    if landslide_date is None:
        # For morphology detection this remains a useful candidate, but it requires visual QC.
        return TemporalDecision("uncertain", "missing_landslide_date_visual_qc_required")

    if lidar_date.precision == "day" and landslide_date.precision == "day":
        if lidar_date.value > landslide_date.value:
            return TemporalDecision("accepted", "lidar_acquired_after_landslide")
        return TemporalDecision("rejected", "lidar_not_acquired_after_landslide")

    # With year-only data, equal years cannot establish event ordering.
    if lidar_date.year > landslide_date.year:
        return TemporalDecision("accepted", "lidar_year_after_landslide_year")
    if lidar_date.year < landslide_date.year:
        return TemporalDecision("rejected", "lidar_year_before_landslide_year")
    return TemporalDecision("uncertain", "same_year_order_unknown")


def normalize_bbox(tile: Mapping[str, Any]) -> dict[str, float]:
    bb = tile.get("boundingBox")
    if not isinstance(bb, Mapping):
        raise ValueError("missing_or_invalid_boundingBox")

    aliases = {
        "minX": ("minX", "west", "xmin"),
        "minY": ("minY", "south", "ymin"),
        "maxX": ("maxX", "east", "xmax"),
        "maxY": ("maxY", "north", "ymax"),
    }

    output: dict[str, float] = {}
    for canonical, names in aliases.items():
        value = None
        for name in names:
            if name in bb:
                value = bb[name]
                break
        if value is None:
            raise ValueError(f"boundingBox_missing_{canonical}")
        try:
            output[canonical] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"boundingBox_non_numeric_{canonical}") from exc

    if output["minX"] >= output["maxX"] or output["minY"] >= output["maxY"]:
        raise ValueError("boundingBox_has_invalid_extent")

    return output


def bbox_looks_geographic(bb: Mapping[str, float]) -> bool:
    return (
        -180 <= bb["minX"] <= 180
        and -180 <= bb["maxX"] <= 180
        and -90 <= bb["minY"] <= 90
        and -90 <= bb["maxY"] <= 90
    )


def geometry_looks_geographic(geom: Any) -> bool:
    try:
        min_x, min_y, max_x, max_y = geom.bounds
    except Exception:
        return False
    return -180 <= min_x <= 180 and -180 <= max_x <= 180 and -90 <= min_y <= 90 and -90 <= max_y <= 90


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def extract_identifier(mapping: Mapping[str, Any], fields: Sequence[str], fallback: str) -> str:
    _, value = first_present(mapping, fields)
    return str(value) if value not in (None, "") else fallback


def extract_download_url(tile: Mapping[str, Any]) -> str:
    _, value = first_present(tile, DOWNLOAD_URL_FIELDS)
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if isinstance(value, Mapping):
        for candidate in ("url", "download", "href"):
            if value.get(candidate):
                return str(value[candidate])
    return str(value) if value not in (None, "") else ""


def build_manifests(
    deposits: Sequence[tuple[Any, Mapping[str, Any]]],
    tiles: Sequence[Mapping[str, Any]],
    acquisition_fields: Sequence[str],
    allow_publication_fallback: bool,
    skip_crs_check: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from shapely.geometry import box
    except ImportError as exc:
        raise RuntimeError("This script needs shapely: pip install shapely") from exc

    rows: list[dict[str, Any]] = []
    tile_errors: list[dict[str, Any]] = []

    if not deposits:
        return rows, tile_errors

    for tile_index, tile in enumerate(tiles):
        title = str(tile.get("title") or f"tile_{tile_index}")
        try:
            bb = normalize_bbox(tile)
            tile_box = box(bb["minX"], bb["minY"], bb["maxX"], bb["maxY"])
        except ValueError as exc:
            tile_errors.append({"tile_title": title, "reason": str(exc), "tile": make_json_safe(tile)})
            continue

        lidar_date, lidar_date_field, lidar_date_kind = extract_lidar_date(
            tile,
            acquisition_fields=acquisition_fields,
            allow_publication_fallback=allow_publication_fallback,
        )

        tile_is_geographic = bbox_looks_geographic(bb)
        tile_id = extract_identifier(tile, TILE_ID_FIELDS, f"tile_{tile_index}")
        tile_url = extract_download_url(tile)

        for deposit_index, (geom, props) in enumerate(deposits):
            if geom is None or getattr(geom, "is_empty", True):
                continue

            if not getattr(geom, "is_valid", True):
                try:
                    geom = geom.buffer(0)
                except Exception:
                    continue
                if geom.is_empty or not geom.is_valid:
                    continue

            if not skip_crs_check and tile_is_geographic != geometry_looks_geographic(geom):
                raise ValueError(
                    "CRS mismatch suspected: LiDAR tile bounds and SLIDO geometry appear to use "
                    "different coordinate systems. Reproject both to the same CRS, preferably EPSG:4326, "
                    "or use --skip-crs-check only after verifying them manually."
                )

            if not tile_box.intersects(geom):
                continue

            landslide_date, landslide_date_field = extract_landslide_date(props)
            decision = classify_temporal_relation(lidar_date, landslide_date, lidar_date_kind)
            landslide_id = extract_identifier(props, LANDSLIDE_ID_FIELDS, f"deposit_{deposit_index}")

            intersection_area = tile_box.intersection(geom).area
            rows.append(
                {
                    "tile_id": tile_id,
                    "tile_title": title,
                    "tile_download_url": tile_url,
                    "tile_min_x": bb["minX"],
                    "tile_min_y": bb["minY"],
                    "tile_max_x": bb["maxX"],
                    "tile_max_y": bb["maxY"],
                    "lidar_date": lidar_date.value.isoformat() if lidar_date else "",
                    "lidar_date_precision": lidar_date.precision if lidar_date else "",
                    "lidar_date_field": lidar_date_field or "",
                    "lidar_date_kind": lidar_date_kind,
                    "landslide_id": landslide_id,
                    "slido_ref_id_cod": props.get("REF_ID_COD", ""),
                    "slido_name": props.get("NAME", ""),
                    "slido_age": props.get("AGE", ""),
                    "slido_description": props.get("DESCRIPTION", ""),
                    "slido_date_range": props.get("DATE_RANGE", ""),
                    "slido_year_raw": props.get("YEAR", ""),
                    "slido_month_raw": props.get("MONTH", ""),
                    "slido_day_raw": props.get("DAY", ""),
                    "slido_date_move_raw": props.get("DATE_MOVE", ""),
                    "landslide_date": landslide_date.value.isoformat() if landslide_date else "",
                    "landslide_date_precision": landslide_date.precision if landslide_date else "",
                    "landslide_date_field": landslide_date_field or "",
                    "move_code": props.get("MOVE_CODE", ""),
                    "move_class": props.get("MOVE_CLASS", ""),
                    "type_move": props.get("TYPE_MOVE", ""),
                    "confidence": props.get("CONFIDENCE", ""),
                    "reactivation": props.get("REACTIVATION", ""),
                    "intersection_area_native_units": intersection_area,
                    "temporal_status": decision.status,
                    "temporal_reason": decision.reason,
                }
            )

    return rows, tile_errors


def aggregate_tiles_for_download(
    tiles: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    allowed_statuses: set[str],
    max_tiles: int | None,
) -> list[dict[str, Any]]:
    matches_by_tile: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["temporal_status"] in allowed_statuses:
            matches_by_tile.setdefault(str(row["tile_id"]), []).append(row)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, tile in enumerate(tiles):
        tile_id = extract_identifier(tile, TILE_ID_FIELDS, f"tile_{index}")
        if tile_id in seen or tile_id not in matches_by_tile:
            continue
        seen.add(tile_id)
        matches = matches_by_tile[tile_id]
        selected.append(
            {
                **make_json_safe(tile),
                "_temporal_statuses": sorted({str(m["temporal_status"]) for m in matches}),
                "_landslide_count": len({str(m["landslide_id"]) for m in matches}),
                "_landslide_ids": sorted({str(m["landslide_id"]) for m in matches}),
                "_move_codes": sorted({str(m["move_code"]) for m in matches if m.get("move_code")}),
            }
        )
        if max_tiles is not None and len(selected) >= max_tiles:
            break

    return selected


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "tile_id",
        "tile_title",
        "landslide_id",
        "temporal_status",
        "temporal_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(value), handle, indent=2, ensure_ascii=False)


def parse_statuses(value: str) -> set[str]:
    allowed = {part.strip().lower() for part in value.split(",") if part.strip()}
    invalid = allowed - {"accepted", "uncertain", "rejected"}
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid statuses: {', '.join(sorted(invalid))}")
    if not allowed:
        raise argparse.ArgumentTypeError("At least one status is required")
    return allowed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an auditable Oregon SLIDO + LiDAR candidate manifest; download only when explicitly requested."
    )
    parser.add_argument("--slido-geojson", "--slido_geojson", dest="slido_geojson", required=True)
    parser.add_argument("--pipeline-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--manifest-dir", type=Path, default=Path("./oregon_manifest"))
    parser.add_argument("--outdir", type=Path, default=Path("./oregon_lidar"))
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download LAZ files. Without this flag, the script only writes manifests.",
    )
    parser.add_argument(
        "--download-statuses",
        type=parse_statuses,
        default={"accepted"},
        help="Comma-separated statuses eligible for download (default: accepted).",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Pilot safety cap on the number of unique LiDAR tiles selected/downloaded.",
    )
    parser.add_argument(
        "--max-deposits",
        type=int,
        default=None,
        help="Pilot safety cap on SLIDO deposits evaluated after loading.",
    )
    parser.add_argument(
        "--acquisition-field",
        action="append",
        default=[],
        help="Additional tile metadata field to treat as an acquisition date. Repeatable.",
    )
    parser.add_argument(
        "--allow-publication-date-fallback",
        action="store_true",
        help="Record publication/dateCreated as uncertain when no acquisition date exists. Never marks it accepted.",
    )
    parser.add_argument(
        "--skip-crs-check",
        action="store_true",
        help="Skip coordinate-range CRS sanity checks after verifying CRS manually.",
    )
    args = parser.parse_args()

    if args.max_tiles is not None and args.max_tiles <= 0:
        parser.error("--max-tiles must be greater than zero")
    if args.max_deposits is not None and args.max_deposits <= 0:
        parser.error("--max-deposits must be greater than zero")

    pipeline_dir = args.pipeline_dir.resolve()
    discover_module = load_stage_module("oregon_discover", pipeline_dir / "discover_3dep.py")
    select_module = load_stage_module("oregon_select", pipeline_dir.parent / "archive" / "04_select_tile_subset.py")
    download_module = None
    if args.download:
        download_module = load_stage_module("oregon_download", pipeline_dir / "download_tiles.py")

    slido_path = Path(args.slido_geojson).resolve()
    if not slido_path.exists():
        parser.error(f"SLIDO GeoJSON not found: {slido_path}")

    print(f"\n--- Stage 1: Discovering LiDAR metadata for {slido_path} ---")
    bbox = discover_module.bbox_from_geojson(str(slido_path))
    tiles = list(discover_module.discover_lidar(bbox))
    print(f"Discovered {len(tiles)} LiDAR tile records.")

    print("\n--- Stage 2: Loading SLIDO deposits ---")
    deposits = list(select_module.load_deposits(str(slido_path)))
    if args.max_deposits is not None:
        deposits = deposits[: args.max_deposits]
    print(f"Evaluating {len(deposits)} SLIDO deposits.")

    acquisition_fields = tuple(args.acquisition_field) + DEFAULT_ACQUISITION_FIELDS
    print("\n--- Stage 3: Building landslide/tile manifest ---")
    rows, tile_errors = build_manifests(
        deposits=deposits,
        tiles=tiles,
        acquisition_fields=acquisition_fields,
        allow_publication_fallback=args.allow_publication_date_fallback,
        skip_crs_check=args.skip_crs_check,
    )

    manifest_dir = args.manifest_dir.resolve()
    manifest_csv = manifest_dir / "landslide_tile_manifest.csv"
    manifest_json = manifest_dir / "landslide_tile_manifest.json"
    tile_errors_json = manifest_dir / "tile_errors.json"
    summary_json = manifest_dir / "summary.json"

    write_csv(manifest_csv, rows)
    write_json(manifest_json, rows)
    write_json(tile_errors_json, tile_errors)

    status_counts = {status: sum(1 for row in rows if row["temporal_status"] == status) for status in ("accepted", "uncertain", "rejected")}
    summary = {
        "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "slido_geojson": str(slido_path),
        "discovered_tile_records": len(tiles),
        "evaluated_deposits": len(deposits),
        "intersecting_pairs": len(rows),
        "status_counts": status_counts,
        "tile_metadata_errors": len(tile_errors),
        "download_requested": bool(args.download),
        "download_statuses": sorted(args.download_statuses),
        "max_tiles": args.max_tiles,
    }
    write_json(summary_json, summary)

    print(f"Manifest rows: {len(rows)}")
    print(f"Status counts: {status_counts}")
    print(f"Wrote: {manifest_csv}")
    print(f"Wrote: {summary_json}")

    selected_tiles = aggregate_tiles_for_download(
        tiles=tiles,
        rows=rows,
        allowed_statuses=args.download_statuses,
        max_tiles=args.max_tiles,
    )
    subset_json = manifest_dir / "selected_tiles.json"
    write_json(subset_json, selected_tiles)
    print(f"Selected {len(selected_tiles)} unique tiles for statuses {sorted(args.download_statuses)}.")
    print(f"Wrote: {subset_json}")

    if not args.download:
        print("\nDry run complete. No LiDAR files were downloaded.")
        print("Review the manifest, then rerun with --download and a small --max-tiles pilot cap.")
        return 0

    if not selected_tiles:
        print("\nNo tiles matched the requested download statuses.")
        return 0

    assert download_module is not None
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Stage 4: Downloading {len(selected_tiles)} LiDAR tiles ---")
    download_module.download_tiles(str(subset_json), str(args.outdir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
