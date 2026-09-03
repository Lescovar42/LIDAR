#!/usr/bin/env python3
"""Cache Sentinel-2 L2A context for an existing Oregon patch dataset.

The script reads patch bounds from patches.csv / patches_qc.csv, searches the
public Element 84 Earth Search STAC API, selects one low-cloud Sentinel-2 scene
per LiDAR tile, and writes a compact multispectral NPZ for every patch.

No Copernicus account or API key is required for this first implementation.

Saved patch arrays
------------------
bands:
    float32 array with B02, B03, B04, B08 surface reflectance at ~10 m.
scl:
    Sentinel-2 scene classification, nearest-neighbour resampled to the same grid.
valid_mask:
    Pixels that are not NoData, saturated, cloud shadow, cloud, cirrus, or snow.
band_names:
    ["blue", "green", "red", "nir"]

Outputs
-------
DATASET_DIR/sentinel2/patches/<split>/<patch_id>.npz
DATASET_DIR/sentinel2/sentinel2_manifest.csv
DATASET_DIR/sentinel2/tile_scenes.json

Example
-------
python fetch_sentinel2_qc.py --dataset-dir dataset_pilot --max-tiles 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np
import requests
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

from lidar_vintage import acquisition_year_from_row
from rasterio.warp import reproject


STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
BAND_KEYS = ("blue", "green", "red", "nir")
INVALID_SCL_CLASSES = {0, 1, 3, 8, 9, 10, 11}
USER_AGENT = "LIDAR-Oregon-Sentinel2-QC/1.0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def infer_year(text: str) -> int | None:
    years = [
        int(value)
        for value in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text or "")
    ]
    plausible = [year for year in years if 2015 <= year <= 2035]
    return plausible[0] if plausible else None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def s3_to_https(href: str) -> str:
    """Convert public s3:// COG paths to anonymous HTTPS paths for Windows GDAL."""
    if not href.startswith("s3://"):
        return href
    remainder = href[5:]
    bucket, _, key = remainder.partition("/")
    if not bucket or not key:
        return href
    encoded_key = quote(key, safe="/:@+,-_.~")
    return f"https://{bucket}.s3.us-west-2.amazonaws.com/{encoded_key}"


def bounds_to_wgs84(
    bounds: tuple[float, float, float, float], source_crs: str
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    transformer = Transformer.from_crs(
        CRS.from_user_input(source_crs), "EPSG:4326", always_xy=True
    )
    corners = [
        transformer.transform(xmin, ymin),
        transformer.transform(xmin, ymax),
        transformer.transform(xmax, ymin),
        transformer.transform(xmax, ymax),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), min(ys), max(xs), max(ys)


def query_items(
    session: requests.Session,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_cloud: float,
    limit: int,
    timeout: float,
    retries: int,
) -> list[dict[str, Any]]:
    payload = {
        "collections": [COLLECTION],
        "bbox": list(bbox_wgs84),
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit,
        "query": {"eo:cloud_cover": {"lte": max_cloud}},
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(STAC_SEARCH_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if "features" not in data:
                raise RuntimeError(f"Unexpected STAC response: {str(data)[:500]}")
            return list(data["features"])
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(10.0, 2.0 ** (attempt - 1)))
    assert last_error is not None
    raise RuntimeError(f"STAC search failed after {retries} attempts: {last_error}") from last_error


def item_has_required_assets(item: dict[str, Any]) -> bool:
    assets = item.get("assets", {})
    return all(key in assets for key in (*BAND_KEYS, "scl"))


def item_score(
    item: dict[str, Any],
    *,
    bbox_wgs84: tuple[float, float, float, float],
    target_year: int,
    preferred_months: set[int],
) -> tuple[float, ...]:
    properties = item.get("properties", {})
    cloud = float(properties.get("eo:cloud_cover", 100.0) or 100.0)
    acquired = parse_datetime(properties.get("datetime"))
    if acquired is None:
        acquired = datetime(target_year, 7, 15, tzinfo=timezone.utc)
        date_penalty = 99999.0
        season_penalty = 1.0
    else:
        target = datetime(target_year, 7, 15, tzinfo=timezone.utc)
        date_penalty = abs((acquired - target).days)
        season_penalty = 0.0 if acquired.month in preferred_months else 1.0

    xmin, ymin, xmax, ymax = bbox_wgs84
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    item_bbox = item.get("bbox") or [-180, -90, 180, 90]
    center_penalty = 0.0
    if len(item_bbox) >= 4:
        if not (
            float(item_bbox[0]) <= center_x <= float(item_bbox[2])
            and float(item_bbox[1]) <= center_y <= float(item_bbox[3])
        ):
            center_penalty = 1.0

    return center_penalty, season_penalty, cloud, date_penalty


def choose_item(
    items: Iterable[dict[str, Any]],
    *,
    bbox_wgs84: tuple[float, float, float, float],
    target_year: int,
    preferred_months: set[int],
) -> dict[str, Any]:
    candidates = [item for item in items if item_has_required_assets(item)]
    if not candidates:
        raise RuntimeError("No returned Sentinel-2 item has blue/green/red/nir/scl assets")
    return min(
        candidates,
        key=lambda item: item_score(
            item,
            bbox_wgs84=bbox_wgs84,
            target_year=target_year,
            preferred_months=preferred_months,
        ),
    )


def asset_scale(asset: dict[str, Any], default: float) -> tuple[float, float]:
    bands = asset.get("raster:bands") or []
    if bands and isinstance(bands[0], dict):
        return float(bands[0].get("scale", default)), float(
            bands[0].get("offset", 0.0)
        )
    return default, 0.0


def target_grid(
    bounds: tuple[float, float, float, float], resolution: float
) -> tuple[int, int, Any]:
    xmin, ymin, xmax, ymax = bounds
    width = max(1, int(math.ceil((xmax - xmin) / resolution)))
    height = max(1, int(math.ceil((ymax - ymin) / resolution)))
    return height, width, from_bounds(xmin, ymin, xmax, ymax, width, height)


def reproject_band(
    source: rasterio.io.DatasetReader,
    *,
    bounds: tuple[float, float, float, float],
    destination_crs: str,
    resolution: float,
    resampling: Resampling,
    destination_dtype: np.dtype,
    destination_nodata: float | int,
) -> np.ndarray:
    height, width, transform = target_grid(bounds, resolution)
    destination = np.full(
        (height, width), destination_nodata, dtype=destination_dtype
    )
    reproject(
        source=rasterio.band(source, 1),
        destination=destination,
        src_nodata=source.nodata,
        dst_transform=transform,
        dst_crs=CRS.from_user_input(destination_crs),
        dst_nodata=destination_nodata,
        resampling=resampling,
        num_threads=2,
    )
    return destination


def save_patch(
    *,
    row: dict[str, str],
    output_path: Path,
    sources: dict[str, rasterio.io.DatasetReader],
    assets: dict[str, Any],
    item: dict[str, Any],
    resolution: float,
) -> dict[str, Any]:
    bounds = tuple(float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
    destination_crs = row["crs"]

    reflectance: list[np.ndarray] = []
    for band_name in BAND_KEYS:
        raw = reproject_band(
            sources[band_name],
            bounds=bounds,
            destination_crs=destination_crs,
            resolution=resolution,
            resampling=Resampling.bilinear,
            destination_dtype=np.float32,
            destination_nodata=np.nan,
        )
        scale, offset = asset_scale(assets[band_name], 0.0001)
        reflectance.append(raw * scale + offset)

    scl = reproject_band(
        sources["scl"],
        bounds=bounds,
        destination_crs=destination_crs,
        resolution=resolution,
        resampling=Resampling.nearest,
        destination_dtype=np.uint8,
        destination_nodata=0,
    )
    bands = np.stack(reflectance, axis=0).astype(np.float32)
    finite = np.all(np.isfinite(bands), axis=0)
    nonzero = np.any(bands > 0, axis=0)
    invalid_scl = np.isin(scl, list(INVALID_SCL_CLASSES))
    valid = finite & nonzero & ~invalid_scl

    output_path.parent.mkdir(parents=True, exist_ok=True)
    properties = item.get("properties", {})
    metadata = {
        "source": "Element 84 Earth Search / AWS Open Data",
        "collection": COLLECTION,
        "item_id": item.get("id", ""),
        "datetime": properties.get("datetime", ""),
        "eo_cloud_cover": properties.get("eo:cloud_cover", ""),
        "resolution_m": resolution,
        "patch_bounds": bounds,
        "patch_crs": destination_crs,
        "band_names": list(BAND_KEYS),
        "invalid_scl_classes": sorted(INVALID_SCL_CLASSES),
    }
    np.savez_compressed(
        output_path,
        bands=bands,
        scl=scl,
        valid_mask=valid.astype(np.uint8),
        band_names=np.asarray(BAND_KEYS),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    return {
        "patch_id": row["patch_id"],
        "tile_name": row.get("tile_name", ""),
        "split": row.get("split", ""),
        "sentinel2_path": str(output_path),
        "sentinel2_item_id": item.get("id", ""),
        "sentinel2_datetime": properties.get("datetime", ""),
        "sentinel2_cloud_cover": properties.get("eo:cloud_cover", ""),
        "sentinel2_valid_fraction": float(valid.mean()),
        "sentinel2_height": int(bands.shape[1]),
        "sentinel2_width": int(bands.shape[2]),
        "status": "ok",
        "error": "",
    }


def resolve_date_range(
    *,
    tile_name: str,
    explicit_start: str | None,
    explicit_end: str | None,
    year_radius: int,
    acquisition_year: int | None = None,
) -> tuple[str, str, int]:
    """Choose the Sentinel-2 search window.

    The authoritative LiDAR acquisition year is preferred. The filename year is
    only a fallback for manifests that predate acquisition provenance, and it is
    never treated as acquisition evidence.
    """
    if bool(explicit_start) != bool(explicit_end):
        raise ValueError("--start-date and --end-date must be supplied together")
    inferred = infer_year(tile_name)
    target = acquisition_year if acquisition_year is not None else inferred
    if explicit_start and explicit_end:
        return explicit_start, explicit_end, target or int(explicit_start[:4])
    if target is None:
        raise ValueError(
            f"No authoritative LiDAR acquisition year for tile {tile_name!r} and no year "
            "in its filename. Rebuild the dataset with acquisition metadata, or supply "
            "--start-date and --end-date."
        )
    return (
        f"{target - year_radius:04d}-06-01",
        f"{target + year_radius:04d}-09-30",
        target,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cache Sentinel-2 L2A RGB/NIR/SCL context for Oregon QC patches."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_pilot"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Patch CSV. Default: patches_qc.csv if present, otherwise patches.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Default: DATASET_DIR/sentinel2.",
    )
    parser.add_argument("--start-date", help="YYYY-MM-DD; use with --end-date.")
    parser.add_argument("--end-date", help="YYYY-MM-DD; use with --start-date.")
    parser.add_argument(
        "--year-radius",
        type=int,
        default=1,
        help="When dates are omitted, search inferred tile year +/- this many years.",
    )
    parser.add_argument(
        "--months",
        default="6,7,8,9",
        help="Preferred acquisition months, comma separated. Default: 6,7,8,9.",
    )
    parser.add_argument("--max-cloud", type=float, default=30.0)
    parser.add_argument("--search-limit", type=int, default=100)
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--max-tiles", type=int, default=0, help="0 means all tiles.")
    parser.add_argument("--max-patches", type=int, default=0, help="0 means all patches.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    if args.year_radius < 0:
        parser.error("--year-radius cannot be negative")
    if not 0 <= args.max_cloud <= 100:
        parser.error("--max-cloud must be between 0 and 100")

    dataset_dir = args.dataset_dir.resolve()
    qc_manifest = dataset_dir / "patches_qc.csv"
    default_manifest = qc_manifest if qc_manifest.exists() else dataset_dir / "patches.csv"
    manifest_path = (args.manifest or default_manifest).resolve()
    outdir = (args.outdir or dataset_dir / "sentinel2").resolve()

    if not dataset_dir.exists():
        parser.error(f"Dataset directory does not exist: {dataset_dir}")
    if not manifest_path.exists():
        parser.error(f"Patch manifest does not exist: {manifest_path}")

    rows = read_csv(manifest_path)
    required = {
        "patch_id", "split", "tile_name", "x_min", "y_min", "x_max", "y_max", "crs"
    }
    missing = sorted(required - set(rows[0] if rows else []))
    if missing:
        parser.error(f"Patch manifest is missing columns: {missing}")
    if args.max_patches:
        rows = rows[: args.max_patches]

    by_tile: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_tile[row["tile_name"]].append(row)
    tile_names = sorted(by_tile, key=str.casefold)
    if args.max_tiles:
        tile_names = tile_names[: args.max_tiles]

    preferred_months = {
        int(value.strip()) for value in args.months.split(",") if value.strip()
    }
    if not preferred_months or any(month < 1 or month > 12 for month in preferred_months):
        parser.error("--months must contain month numbers from 1 to 12")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/geo+json"})

    result_rows: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    print(f"Patch manifest: {manifest_path}")
    print(f"Tiles selected: {len(tile_names)}")
    print(f"Sentinel cache: {outdir}")
    print("Source: public Element 84 Earth Search Sentinel-2 L2A COGs")

    raster_env = {
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "50000000",
    }

    for tile_number, tile_name in enumerate(tile_names, start=1):
        tile_rows = by_tile[tile_name]
        print("\n" + "=" * 78)
        print(f"[{tile_number}/{len(tile_names)}] {tile_name} ({len(tile_rows)} patches)")
        print("=" * 78)

        try:
            acquisition_year = next(
                (
                    year
                    for row in tile_rows
                    if (year := acquisition_year_from_row(row)) is not None
                ),
                None,
            )
            start_date, end_date, target_year = resolve_date_range(
                tile_name=tile_name,
                explicit_start=args.start_date,
                explicit_end=args.end_date,
                year_radius=args.year_radius,
                acquisition_year=acquisition_year,
            )
            if acquisition_year is None:
                print(
                    "WARNING: no authoritative LiDAR acquisition year in the patch "
                    f"manifest for {tile_name}; falling back to the non-authoritative "
                    "filename year for the search window only."
                )
            source_crs = tile_rows[0]["crs"]
            xmin = min(float(row["x_min"]) for row in tile_rows)
            ymin = min(float(row["y_min"]) for row in tile_rows)
            xmax = max(float(row["x_max"]) for row in tile_rows)
            ymax = max(float(row["y_max"]) for row in tile_rows)
            bbox_wgs84 = bounds_to_wgs84((xmin, ymin, xmax, ymax), source_crs)

            print(f"Search dates: {start_date} to {end_date}")
            print(f"WGS84 bbox: {bbox_wgs84}")
            items = query_items(
                session,
                bbox_wgs84=bbox_wgs84,
                start_date=start_date,
                end_date=end_date,
                max_cloud=args.max_cloud,
                limit=args.search_limit,
                timeout=args.timeout,
                retries=args.retries,
            )
            item = choose_item(
                items,
                bbox_wgs84=bbox_wgs84,
                target_year=target_year,
                preferred_months=preferred_months,
            )
            properties = item.get("properties", {})
            assets = item["assets"]
            print(
                f"Selected: {item.get('id')} | {properties.get('datetime')} | "
                f"catalog cloud={properties.get('eo:cloud_cover')}%"
            )

            scene_records.append(
                {
                    "tile_name": tile_name,
                    "item_id": item.get("id", ""),
                    "datetime": properties.get("datetime", ""),
                    "eo_cloud_cover": properties.get("eo:cloud_cover", ""),
                    "search_start": start_date,
                    "search_end": end_date,
                    "bbox_wgs84": list(bbox_wgs84),
                    "assets": {
                        key: s3_to_https(assets[key]["href"])
                        for key in (*BAND_KEYS, "scl")
                    },
                }
            )

            with rasterio.Env(**raster_env), ExitStack() as stack:
                sources = {
                    key: stack.enter_context(
                        rasterio.open(s3_to_https(assets[key]["href"]))
                    )
                    for key in (*BAND_KEYS, "scl")
                }
                for patch_number, row in enumerate(tile_rows, start=1):
                    relative = (
                        Path("patches")
                        / row.get("split", "unknown")
                        / f"{row['patch_id']}.npz"
                    )
                    destination = outdir / relative
                    if destination.exists() and not args.overwrite:
                        try:
                            with np.load(destination) as cached:
                                valid_fraction = float(
                                    cached["valid_mask"].astype(bool).mean()
                                )
                                shape = cached["valid_mask"].shape
                            result = {
                                "patch_id": row["patch_id"],
                                "tile_name": tile_name,
                                "split": row.get("split", ""),
                                "sentinel2_path": str(relative),
                                "sentinel2_item_id": item.get("id", ""),
                                "sentinel2_datetime": properties.get("datetime", ""),
                                "sentinel2_cloud_cover": properties.get("eo:cloud_cover", ""),
                                "sentinel2_valid_fraction": valid_fraction,
                                "sentinel2_height": int(shape[0]),
                                "sentinel2_width": int(shape[1]),
                                "status": "cached",
                                "error": "",
                            }
                        except Exception:
                            destination.unlink(missing_ok=True)
                            result = save_patch(
                                row=row,
                                output_path=destination,
                                sources=sources,
                                assets=assets,
                                item=item,
                                resolution=args.resolution,
                            )
                            result["sentinel2_path"] = str(relative)
                    else:
                        result = save_patch(
                            row=row,
                            output_path=destination,
                            sources=sources,
                            assets=assets,
                            item=item,
                            resolution=args.resolution,
                        )
                        result["sentinel2_path"] = str(relative)

                    result_rows.append(result)
                    status_counts[result["status"]] += 1
                    print(
                        f"  [{patch_number}/{len(tile_rows)}] {row['patch_id']} "
                        f"valid={100 * float(result['sentinel2_valid_fraction']):.1f}%"
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"FAILED TILE: {error}")
            scene_records.append({"tile_name": tile_name, "error": error})
            for row in tile_rows:
                result_rows.append(
                    {
                        "patch_id": row["patch_id"],
                        "tile_name": tile_name,
                        "split": row.get("split", ""),
                        "sentinel2_path": "",
                        "sentinel2_item_id": "",
                        "sentinel2_datetime": "",
                        "sentinel2_cloud_cover": "",
                        "sentinel2_valid_fraction": "",
                        "sentinel2_height": "",
                        "sentinel2_width": "",
                        "status": "error",
                        "error": error,
                    }
                )
                status_counts["error"] += 1

        manifest_fields = [
            "patch_id", "tile_name", "split", "sentinel2_path",
            "sentinel2_item_id", "sentinel2_datetime", "sentinel2_cloud_cover",
            "sentinel2_valid_fraction", "sentinel2_height", "sentinel2_width",
            "status", "error",
        ]
        atomic_write_csv(outdir / "sentinel2_manifest.csv", result_rows, manifest_fields)
        (outdir / "tile_scenes.json").write_text(
            json.dumps(scene_records, indent=2), encoding="utf-8"
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_dir": str(dataset_dir),
        "patch_manifest": str(manifest_path),
        "outdir": str(outdir),
        "collection": COLLECTION,
        "stac_search_url": STAC_SEARCH_URL,
        "band_names": list(BAND_KEYS),
        "resolution_m": args.resolution,
        "invalid_scl_classes": sorted(INVALID_SCL_CLASSES),
        "status_counts": dict(status_counts),
        "tile_count": len(tile_names),
        "patch_count": len(result_rows),
        "parameters": vars(args)
        | {
            "dataset_dir": str(args.dataset_dir),
            "manifest": str(args.manifest) if args.manifest else None,
            "outdir": str(args.outdir) if args.outdir else None,
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print(f"Sentinel patch records: {len(result_rows)}")
    print(f"Status counts: {dict(status_counts)}")
    print(f"Manifest: {outdir / 'sentinel2_manifest.csv'}")
    print("Next: run diagnostics/qc_patch_viewer.py --dataset-dir dataset_pilot")
    return 0 if result_rows and status_counts["error"] < len(result_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
