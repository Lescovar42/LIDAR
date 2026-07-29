#!/usr/bin/env python3
"""Build SLIDO, TNM LPC, and NAIP candidate-region reports.

Reports normally consume persisted snapshots. ``--refresh`` explicitly queries
TNM and NAIP and saves their metadata before reporting.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import requests
from shapely.geometry import box, shape
from shapely.ops import unary_union

try:
    from .discover_3dep import discover_lidar
    from .fetch_naip_qc import query_records, transform_bounds
    from .region_registry import REGISTRY_PATH, load_registry, resolve_path, resolve_region
    from .slido_utils import property_counts
    from .tnm_utils import canonical_project, project_matches
except ImportError:
    from discover_3dep import discover_lidar
    from fetch_naip_qc import query_records, transform_bounds
    from region_registry import REGISTRY_PATH, load_registry, resolve_path, resolve_region
    from slido_utils import property_counts
    from tnm_utils import canonical_project, project_matches


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("items", "features", "results", "records", "selected_tiles"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def refresh_region_metadata(
    region: Mapping[str, Any],
    *,
    registry_path: str | Path = REGISTRY_PATH,
    tnm_path: Path | None = None,
    naip_path: Path | None = None,
    tnm_fetcher: Callable[[Iterable[float]], list[dict[str, Any]]] | None = None,
    naip_fetcher: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Path]:
    """Query and persist TNM and NAIP metadata for one registry region."""
    paths = {
        "tnm": tnm_path or resolve_path(region, "tnm_records", registry_path),
        "naip": naip_path or resolve_path(region, "naip_records", registry_path),
    }
    tnm_items = (tnm_fetcher or discover_lidar)(tuple(region["bbox"]))
    _write_json(paths["tnm"], {"items": tnm_items})

    bounds_3857 = transform_bounds(tuple(region["bbox"]), "EPSG:4326")
    session = requests.Session()
    try:
        session.headers.update({"User-Agent": "LIDAR-Oregon-Region-Report/1.0"})
        naip_items = (naip_fetcher or query_records)(
            session, bounds_3857=bounds_3857, timeout=60.0, retries=3
        )
    finally:
        session.close()
    _write_json(
        paths["naip"],
        {"features": [{"attributes": item} for item in naip_items]},
    )
    return paths


def project_key(item: Mapping[str, Any]) -> str:
    """Compatibility wrapper for the shared canonical TNM project parser."""
    return canonical_project(item)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _item_geometry(item: Mapping[str, Any]):
    geometry = item.get("geometry") or item.get("footprint")
    if isinstance(geometry, Mapping) and geometry.get("type"):
        try:
            return shape(geometry)
        except Exception:
            pass

    raw_bbox = item.get("bbox") or item.get("boundingBox") or item.get("spatialBoundingBox")
    values: list[Any] | None = None
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        values = list(raw_bbox)
    elif isinstance(raw_bbox, Mapping):
        key_sets = (
            ("minX", "minY", "maxX", "maxY"),
            ("west", "south", "east", "north"),
            ("xmin", "ymin", "xmax", "ymax"),
        )
        for keys in key_sets:
            if all(key in raw_bbox for key in keys):
                values = [raw_bbox[key] for key in keys]
                break
    if values:
        numbers = [_number(value) for value in values]
        if all(value is not None for value in numbers):
            xmin, ymin, xmax, ymax = numbers
            if xmin < xmax and ymin < ymax:
                return box(xmin, ymin, xmax, ymax)
    return None


def group_tnm_projects(
    items: Iterable[Mapping[str, Any]],
    bbox_wgs84: Iterable[float],
    tile_budget: int = 0,
) -> dict[str, dict[str, Any]]:
    """Group TNM records and calculate unioned AOI footprint coverage."""
    aoi = box(*[float(value) for value in bbox_wgs84])
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        grouped.setdefault(project_key(item), []).append(item)

    output: dict[str, dict[str, Any]] = {}
    for name, records in sorted(grouped.items()):
        bytes_total = sum(int(_number(record.get("sizeInBytes") or record.get("size_bytes")) or 0) for record in records)
        footprints = []
        for record in records:
            geometry = _item_geometry(record)
            if geometry is not None and not geometry.is_empty and geometry.intersects(aoi):
                footprints.append(geometry.intersection(aoi))
        coverage = unary_union(footprints).area / aoi.area if footprints and aoi.area else 0.0
        average_bytes = bytes_total / len(records) if records else 0
        output[name] = {
            "tile_count": len(records),
            "summed_bytes": bytes_total,
            "summed_gb": round(bytes_total / 1_000_000_000, 4),
            "aoi_coverage_share": round(min(1.0, coverage), 6),
            "projected_gb_at_tile_budget": round(average_bytes * tile_budget / 1_000_000_000, 4),
        }
    return output


def naip_years(records: Iterable[Mapping[str, Any]]) -> list[int]:
    years: set[int] = set()
    for item in records:
        if isinstance(item.get("attributes"), Mapping):
            properties = item["attributes"]
        elif isinstance(item.get("properties"), Mapping):
            properties = item["properties"]
        else:
            properties = item
        value = properties.get("Year") or properties.get("year") or properties.get("naip_year")
        if value in (None, ""):
            continue
        match = re.search(r"\b(?:19|20)\d{2}\b", str(value))
        if match:
            years.add(int(match.group()))
    return sorted(years)


def build_region_report(
    region: Mapping[str, Any],
    *,
    registry_path: str | Path = REGISTRY_PATH,
    slido_path: Path | None = None,
    tnm_path: Path | None = None,
    naip_path: Path | None = None,
    allow_missing: bool = False,
) -> dict[str, Any]:
    paths = {
        "slido": slido_path or resolve_path(region, "slido_output", registry_path),
        "tnm": tnm_path or resolve_path(region, "tnm_records", registry_path),
        "naip": naip_path or resolve_path(region, "naip_records", registry_path),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing and region.get("status", "active") == "active" and not allow_missing:
        details = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(
            f"Missing required metadata for active region {region['id']}: {details}. "
            "Run with --refresh where applicable or pass --allow-missing for an offline skeleton."
        )
    report: dict[str, Any] = {
        "region_id": region["id"],
        "slug": region["slug"],
        "name": region["name"],
        "status": region.get("status", "active"),
        "role": region["role"],
        "bbox_wgs84": region["bbox"],
        "tile_budget": region["tile_budget"],
        "storage_budget_gb": region["storage_budget_gb"],
        "candidate_projects": region["candidate_projects"],
        "inputs": {key: str(value) for key, value in paths.items()},
    }
    if region.get("rejection_reason"):
        report["rejection_reason"] = region["rejection_reason"]

    if paths["slido"].is_file():
        features = _records(_load_json(paths["slido"]))
        confidence, sources = property_counts(features)
        report["slido"] = {
            "available": True,
            "feature_count": len(features),
            "by_confidence": confidence,
            "by_source": sources,
        }
    else:
        report["slido"] = {"available": False, "feature_count": None, "by_confidence": {}, "by_source": {}}

    if paths["tnm"].is_file():
        items = _records(_load_json(paths["tnm"]))
        projects = group_tnm_projects(items, region["bbox"], int(region["tile_budget"]))
        for canonical, values in projects.items():
            values["registry_candidates"] = [
                candidate
                for candidate in region["candidate_projects"]
                if project_matches(canonical, candidate)
            ]
        report["tnm_lpc"] = {
            "available": True,
            "record_count": len(items),
            "projects": projects,
        }
    else:
        report["tnm_lpc"] = {"available": False, "record_count": None, "projects": {}}

    if paths["naip"].is_file():
        items = _records(_load_json(paths["naip"]))
        report["naip"] = {"available": True, "years": naip_years(items)}
    else:
        report["naip"] = {"available": False, "years": []}
    return report


def render_markdown(reports: Iterable[Mapping[str, Any]]) -> str:
    reports = list(reports)
    lines = [
        "# Region Candidate Comparison",
        "",
        "Generated from persisted local inputs; use `--refresh` to update TNM and NAIP metadata first.",
        "",
        "| Region | Status | SLIDO | High | Moderate | TNM records | NAIP years | Tile budget | Storage budget (GB) |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for report in reports:
        slido = report["slido"]
        confidence = slido["by_confidence"]
        lines.append(
            f"| {report['name']} | {report['status']} | {slido['feature_count'] if slido['available'] else 'unavailable'} "
            f"| {confidence.get('high', '—')} | {confidence.get('moderate', '—')} "
            f"| {report['tnm_lpc']['record_count'] if report['tnm_lpc']['available'] else 'unavailable'} "
            f"| {', '.join(map(str, report['naip']['years'])) or 'unavailable'} "
            f"| {report['tile_budget']} | {report['storage_budget_gb']} |"
        )
    for report in reports:
        projects = report["tnm_lpc"]["projects"]
        if not projects:
            continue
        lines.extend(["", f"## {report['name']} TNM LPC projects", "", "| Project | Tiles | GB | AOI coverage | Projected GB at budget |", "|---|---:|---:|---:|---:|"])
        for project, values in projects.items():
            lines.append(
                f"| {project} | {values['tile_count']} | {values['summed_gb']:.4f} "
                f"| {values['aoi_coverage_share']:.1%} | {values['projected_gb_at_tile_budget']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build comparison reports from persisted region metadata.")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--region", action="append", help="Region id/slug/name; repeat as needed. Default: five candidates.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("regions"))
    parser.add_argument("--markdown", type=Path, help="Comparison output (default: <out-dir>/comparison.md).")
    parser.add_argument("--slido", type=Path, help="Override SLIDO input; requires exactly one --region.")
    parser.add_argument("--tnm", type=Path, help="Override TNM input; requires exactly one --region.")
    parser.add_argument("--naip", type=Path, help="Override NAIP input; requires exactly one --region.")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Query live TNM and NAIP services and persist metadata before reporting.",
    )
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Allow unavailable inputs for offline skeleton reports.",
    )
    args = parser.parse_args()

    registry = load_registry(args.registry)
    if args.region:
        try:
            regions = [resolve_region(value, registry, include_comparison=True) for value in args.region]
        except KeyError as exc:
            parser.error(str(exc))
    else:
        regions = [
            entry
            for entry in [*registry["regions"], *registry.get("comparison_candidates", [])]
            if entry.get("include_in_candidate_report")
        ]
    if any((args.slido, args.tnm, args.naip)) and len(regions) != 1:
        parser.error("input path overrides require exactly one --region")

    if args.refresh:
        for region in regions:
            paths = refresh_region_metadata(
                region,
                registry_path=args.registry,
                tnm_path=args.tnm,
                naip_path=args.naip,
            )
            print(f"Refreshed {region['id']}: {paths['tnm']}, {paths['naip']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for region in regions:
        report = build_region_report(
            region,
            registry_path=args.registry,
            slido_path=args.slido,
            tnm_path=args.tnm,
            naip_path=args.naip,
            allow_missing=args.allow_missing,
        )
        output = args.out_dir / f"{region['slug']}_report.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        reports.append(report)
        print(f"Wrote {output}")
    markdown_path = args.markdown or args.out_dir / "comparison.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(reports), encoding="utf-8")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
