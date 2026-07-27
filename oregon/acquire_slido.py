#!/usr/bin/env python3
"""Download Oregon SLIDO deposit polygons from DOGAMI as GeoJSON.

By default this fetches only ``DESCRIPTION = 'Landslide'`` so fans and
Talus-Colluvium are not accidentally used as positive training labels.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SERVICE_URL = (
    "https://gis.dogami.oregon.gov/arcgis/rest/services/"
    "Public/SLIDO42/MapServer/3/query"
)
DEFAULT_BBOX = (-122.90, 45.35, -122.55, 45.65)


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


def validate_bbox(values: list[float]) -> tuple[float, float, float, float]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a compact SLIDO landslide GeoJSON pilot subset.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        default=list(DEFAULT_BBOX),
        help="WGS84 west south east north bounding box.",
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
        default=1000,
        help="Safety cap. Use 0 for no cap; keep this small for the pilot.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2, help="Pause between pages.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("slido_deposits_oregon_city.geojson"),
    )
    args = parser.parse_args()

    bbox = validate_bbox(args.bbox)
    if not 1 <= args.page_size <= 1000:
        parser.error("--page-size must be between 1 and 1000")
    if args.max_features < 0:
        parser.error("--max-features must be zero or positive")

    session = build_session()
    features: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    offset = 0

    print(f"DOGAMI layer: {SERVICE_URL}")
    print(f"Bbox: {bbox}")
    print(f"Where: {args.where}")

    while True:
        remaining = args.max_features - len(features) if args.max_features else args.page_size
        requested = min(args.page_size, remaining) if args.max_features else args.page_size
        if requested <= 0:
            break

        page = query_page(
            session,
            bbox=bbox,
            where=args.where,
            offset=offset,
            page_size=requested,
            timeout=args.timeout,
        )
        page_features = page.get("features", [])
        if not page_features:
            break

        added = 0
        for feature in page_features:
            props = feature.get("properties") or {}
            object_id = str(props.get("OBJECTID", f"offset-{offset + added}"))
            if object_id in seen_ids:
                continue
            seen_ids.add(object_id)
            features.append(feature)
            added += 1
            if args.max_features and len(features) >= args.max_features:
                break

        print(f"Fetched {len(page_features)}; added {added}; total {len(features)}")
        offset += len(page_features)

        if len(page_features) < requested:
            break
        if args.max_features and len(features) >= args.max_features:
            break
        time.sleep(max(0.0, args.sleep))

    if not features:
        raise RuntimeError("The query returned zero features. Expand or move the bounding box.")

    output = {"type": "FeatureCollection", "features": features}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    metadata_path = args.out.with_suffix(args.out.suffix + ".metadata.json")
    metadata = {
        "retrieved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "service_url": SERVICE_URL,
        "bbox_wgs84": bbox,
        "where": args.where,
        "feature_count": len(features),
        "output": str(args.out.resolve()),
        "note": "DESCRIPTION='Landslide' is the default to exclude Fan and Talus-Colluvium labels.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved {len(features)} features to {args.out.resolve()}")
    print(f"Saved provenance metadata to {metadata_path.resolve()}")
    if args.max_features and len(features) == args.max_features:
        print("Reached --max-features safety cap. This is expected for a pilot subset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
