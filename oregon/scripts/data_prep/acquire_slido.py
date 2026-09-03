#!/usr/bin/env python3
"""Download Oregon SLIDO deposit polygons from DOGAMI as GeoJSON.

Without ``--region`` this retains the original Oregon City bbox, 1,000-feature
cap, and output filename.  Region mode resolves the bbox, output, and a larger
per-region safety cap from ``regions.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from .region_registry import REGISTRY_PATH, load_registry, resolve_path, resolve_region
    from .slido_utils import add_confidence_class, property_counts
except ImportError:
    from region_registry import REGISTRY_PATH, load_registry, resolve_path, resolve_region
    from slido_utils import add_confidence_class, property_counts

SERVICE_URL = (
    "https://gis.dogami.oregon.gov/arcgis/rest/services/"
    "Public/SLIDO42/MapServer/3/query"
)
DEFAULT_BBOX = (-122.90, 45.35, -122.55, 45.65)
DEFAULT_OUTPUT = Path("slido_deposits_oregon_city.geojson")


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "oregon-slido-lidar-pipeline/1.0"})
    return session


def validate_bbox(values: Sequence[float]) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = values
    if not (-180 <= xmin < xmax <= 180):
        raise argparse.ArgumentTypeError("invalid longitude range in --bbox")
    if not (-90 <= ymin < ymax <= 90):
        raise argparse.ArgumentTypeError("invalid latitude range in --bbox")
    return xmin, ymin, xmax, ymax


def query_page(
    session: requests.Session,
    *,
    bbox: tuple[float, float, float, float],
    where: str,
    offset: int,
    page_size: int,
    timeout: int,
) -> dict[str, Any]:
    params = {
        "where": where,
        "geometry": ",".join(str(v) for v in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "orderByFields": "OBJECTID ASC",
        "resultOffset": str(offset),
        "resultRecordCount": str(page_size),
        "f": "geojson",
    }
    response = session.get(SERVICE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError("DOGAMI ArcGIS error:\n" + json.dumps(data["error"], indent=2))
    if data.get("type") != "FeatureCollection":
        raise RuntimeError("Unexpected DOGAMI response:\n" + json.dumps(data, indent=2)[:2000])
    return data


def enforce_feature_cap(feature_count: int, max_features: int) -> None:
    """Fail when the result reaches the cap because completeness is unknown."""
    if max_features and feature_count >= max_features:
        raise RuntimeError(
            f"Query reached --max-features={max_features}; output may be truncated. "
            "Increase the cap and run again. No output was written."
        )


def annotate_and_count(features: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    for feature in features:
        add_confidence_class(feature)
    return property_counts(features)


def acquire_features(
    session: requests.Session,
    *,
    bbox: tuple[float, float, float, float],
    where: str,
    page_size: int,
    max_features: int,
    timeout: int,
    sleep: float,
) -> list[dict[str, Any]]:
    """Fetch all pages and reject results that reach the configured cap."""
    features: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    offset = 0
    while True:
        remaining = max_features - len(features) if max_features else page_size
        requested = min(page_size, remaining) if max_features else page_size
        if requested <= 0:
            break
        page = query_page(
            session,
            bbox=bbox,
            where=where,
            offset=offset,
            page_size=requested,
            timeout=timeout,
        )
        page_features = page.get("features", [])
        if not page_features:
            break
        added = 0
        for feature in page_features:
            properties = feature.get("properties") or {}
            object_id = properties.get("OBJECTID", properties.get("objectid"))
            identity = str(object_id) if object_id is not None else f"offset-{offset + added}"
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            features.append(feature)
            added += 1
            if max_features and len(features) >= max_features:
                break
        print(f"Fetched {len(page_features)}; added {added}; total {len(features)}")
        offset += len(page_features)
        if len(page_features) < requested or (max_features and len(features) >= max_features):
            break
        time.sleep(max(0.0, sleep))
    enforce_feature_cap(len(features), max_features)
    return features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch a SLIDO landslide GeoJSON subset.")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--region", help="Registry region id, slug, or name.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="WGS84 west south east north bounding box (legacy default: Oregon City).",
    )
    parser.add_argument(
        "--where",
        default="DESCRIPTION = 'Landslide'",
        help="ArcGIS SQL filter. Default keeps only Landslide deposits.",
    )
    parser.add_argument("--page-size", type=int, default=500, help="Features requested per page (1-1000).")
    parser.add_argument(
        "--max-features",
        type=int,
        help="Safety cap. Region mode uses its registry cap; legacy default is 1000; 0 disables.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2, help="Pause between pages.")
    parser.add_argument("--out", type=Path, help="Output GeoJSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    region = None
    if args.region:
        if args.bbox:
            parser.error("--bbox cannot be combined with --region")
        try:
            region = resolve_region(args.region, load_registry(args.registry))
        except (KeyError, ValueError, OSError) as exc:
            parser.error(str(exc))
        bbox = validate_bbox(region["bbox"])
        output_path = args.out or resolve_path(region, "slido_output", args.registry)
        max_features = args.max_features if args.max_features is not None else int(region["slido_max_features"])
    else:
        bbox = validate_bbox(args.bbox or DEFAULT_BBOX)
        output_path = args.out or DEFAULT_OUTPUT
        max_features = args.max_features if args.max_features is not None else 1000

    if not 1 <= args.page_size <= 1000:
        parser.error("--page-size must be between 1 and 1000")
    if max_features < 0:
        parser.error("--max-features must be zero or positive")

    print(f"DOGAMI layer: {SERVICE_URL}")
    print(f"Region: {region['id'] if region else 'legacy Oregon City bbox'}")
    print(f"Bbox: {bbox}")
    print(f"Where: {args.where}")
    print(f"Feature cap: {max_features or 'disabled'}")
    features = acquire_features(
        build_session(),
        bbox=bbox,
        where=args.where,
        page_size=args.page_size,
        max_features=max_features,
        timeout=args.timeout,
        sleep=args.sleep,
    )
    if not features:
        raise RuntimeError("The query returned zero features. Expand or move the bounding box.")

    confidence_counts, source_counts = annotate_and_count(features)
    output = {"type": "FeatureCollection", "features": features}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata = {
        "retrieved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "service_url": SERVICE_URL,
        "region_id": region["id"] if region else None,
        "region_slug": region["slug"] if region else None,
        "bbox_wgs84": bbox,
        "where": args.where,
        "feature_count": len(features),
        "max_features": max_features,
        "counts_by_confidence": confidence_counts,
        "counts_by_source": source_counts,
        "output": str(output_path.resolve()),
        "note": "DESCRIPTION='Landslide' is the default to exclude Fan and Talus-Colluvium labels.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Confidence counts: {confidence_counts}")
    print(f"Source counts: {source_counts}")
    print(f"Saved {len(features)} features to {output_path.resolve()}")
    print(f"Saved provenance metadata to {metadata_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
