#!/usr/bin/env python3
"""
Cache USGS NAIP orthoimagery for Oregon LiDAR/SLIDO patch QC.

This first NAIP implementation uses the official USGS National Map
USGSNAIPImagery ImageServer. It downloads only a clipped context around each
existing patch instead of full NAIP quarter-quadrangle files.

Why use the ImageServer first?
------------------------------
* It is ideal for visual QC because only the needed area is transferred.
* No credentials are required.
* The service exposes acquisition year/date and four-band imagery.
* Full original quarter-quads can be downloaded later through USGS M2M using
  the user's approved application token.

Inputs
------
DATASET_DIR/patches.csv or DATASET_DIR/patches_qc.csv

Outputs
-------
DATASET_DIR/naip/patches/<split>/<patch_id>.npz
DATASET_DIR/naip/naip_manifest.csv
DATASET_DIR/naip/tile_selections.json
DATASET_DIR/naip/summary.json

Each patch NPZ contains:
    bands         uint8 [4, H, W] in R, G, B, NIR order
    valid_mask    uint8 [H, W]
    metadata_json JSON string with image/patch bounds and source information

Example
-------
python fetch_naip_qc.py --dataset-dir dataset_pilot --max-tiles 10 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import rasterio
import requests
from pyproj import CRS, Transformer

SERVICE_ROOT = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/"
    "USGSNAIPImagery/ImageServer"
)
QUERY_URL = f"{SERVICE_ROOT}/query"
EXPORT_URL = f"{SERVICE_ROOT}/exportImage"
USER_AGENT = "LIDAR-Oregon-NAIP-QC/1.0"


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
    plausible = [year for year in years if 2003 <= year <= 2035]
    return plausible[0] if plausible else None


def transform_bounds(
    bounds: tuple[float, float, float, float],
    source_crs: str,
    destination_crs: str = "EPSG:3857",
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    transformer = Transformer.from_crs(
        CRS.from_user_input(source_crs),
        CRS.from_user_input(destination_crs),
        always_xy=True,
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


def expand_bounds(
    patch_bounds: tuple[float, float, float, float],
    context_size_m: float,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = patch_bounds
    width = max(context_size_m, xmax - xmin)
    height = max(context_size_m, ymax - ymin)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise RuntimeError(json.dumps(data["error"], indent=2))
            return data
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(10.0, 2.0 ** (attempt - 1)))
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}") from last_error


def query_records(
    session: requests.Session,
    *,
    bounds_3857: tuple[float, float, float, float],
    timeout: float,
    retries: int,
) -> list[dict[str, Any]]:
    params = {
        "where": "1=1",
        "geometry": ",".join(f"{value:.3f}" for value in bounds_3857),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "3857",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": (
            "OBJECTID,Name,State,Year,raster_name,download_url,"
            "acquisition_date,agency,vendor,resolution_value,"
            "resolution_units,band_count"
        ),
        "returnGeometry": "false",
        "orderByFields": "Year DESC, acquisition_date DESC",
        "resultRecordCount": "50",
        "f": "json",
    }
    data = request_json(
        session, QUERY_URL, params=params, timeout=timeout, retries=retries
    )
    return [feature.get("attributes", {}) for feature in data.get("features", [])]


def choose_year(
    records: list[dict[str, Any]],
    *,
    target_year: int | None,
    requested_year: int | None,
) -> int:
    years = sorted(
        {
            int(record["Year"])
            for record in records
            if record.get("Year") not in (None, "")
        }
    )
    if not years:
        raise RuntimeError("NAIP query returned no usable acquisition years")

    if requested_year is not None:
        if requested_year not in years:
            raise RuntimeError(
                f"Requested NAIP year {requested_year} is unavailable. "
                f"Available years: {years}"
            )
        return requested_year

    if target_year is None:
        return max(years)

    # Prefer the closest year. When equally close, prefer imagery acquired after
    # the nominal LiDAR year, then the more recent year.
    return min(
        years,
        key=lambda year: (
            abs(year - target_year),
            0 if year >= target_year else 1,
            -year,
        ),
    )


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    timeout: float,
    retries: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(temporary, destination)
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(10.0, 2.0 ** (attempt - 1)))

    raise RuntimeError(f"Image download failed: {last_error}") from last_error


def export_naip(
    session: requests.Session,
    *,
    bounds_3857: tuple[float, float, float, float],
    year: int,
    resolution_m: float,
    output_tif: Path,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    xmin, ymin, xmax, ymax = bounds_3857
    width = max(1, int(math.ceil((xmax - xmin) / resolution_m)))
    height = max(1, int(math.ceil((ymax - ymin) / resolution_m)))

    if width > 4000 or height > 4000:
        raise ValueError(
            f"Requested export is {width}×{height}, exceeding the service "
            "limit of 4000×4000. Increase --resolution or reduce --context-size-m."
        )

    mosaic_rule = {
        "mosaicMethod": "esriMosaicAttribute",
        "where": f"Year = {year}",
        "sortField": "acquisition_date",
        "ascending": False,
        "mosaicOperation": "MT_FIRST",
    }

    params = {
        "bbox": ",".join(f"{value:.3f}" for value in bounds_3857),
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "U8",
        "bandIds": "0,1,2,3",
        "interpolation": "RSP_BilinearInterpolation",
        "mosaicRule": json.dumps(mosaic_rule, separators=(",", ":")),
        "returnSquarePixels": "true",
        "f": "json",
    }

    data = request_json(
        session, EXPORT_URL, params=params, timeout=timeout, retries=retries
    )
    href = data.get("href")
    if not href:
        raise RuntimeError(f"Export response contains no href: {data}")
    download_file(
        session,
        urljoin(SERVICE_ROOT + "/", href),
        output_tif,
        timeout=max(timeout, 180.0),
        retries=retries,
    )
    return {
        "service_response": data,
        "width": width,
        "height": height,
        "mosaic_rule": mosaic_rule,
    }


def tif_to_npz(
    source_tif: Path,
    destination_npz: Path,
    *,
    patch_bounds_3857: tuple[float, float, float, float],
    context_bounds_3857: tuple[float, float, float, float],
    selected_year: int,
    source_records: list[dict[str, Any]],
    export_info: dict[str, Any],
    requested_resolution_m: float,
) -> dict[str, Any]:
    with rasterio.open(source_tif) as dataset:
        bands = dataset.read()
        if bands.shape[0] < 3:
            raise RuntimeError(f"Expected at least 3 bands, received {bands.shape}")
        if bands.shape[0] == 3:
            # Some service configurations may return RGB only. Preserve the
            # interface while making the missing NIR explicit.
            nir = np.zeros_like(bands[0:1])
            bands = np.concatenate([bands, nir], axis=0)
            has_nir = False
        else:
            bands = bands[:4]
            has_nir = True

        valid = dataset.dataset_mask() > 0
        valid &= np.any(bands[:3] > 0, axis=0)

        metadata = {
            "version": 1,
            "source": "USGS National Map USGSNAIPImagery ImageServer",
            "service_url": SERVICE_ROOT,
            "selected_year": selected_year,
            "patch_bounds_3857": patch_bounds_3857,
            "context_bounds_3857": context_bounds_3857,
            "image_crs": dataset.crs.to_string() if dataset.crs else "EPSG:3857",
            "image_transform": list(dataset.transform)[:6],
            "requested_resolution_m": requested_resolution_m,
            "actual_pixel_size_x": abs(float(dataset.transform.a)),
            "actual_pixel_size_y": abs(float(dataset.transform.e)),
            "band_order": ["red", "green", "blue", "nir"],
            "has_nir": has_nir,
            "source_records": source_records,
            "export_info": export_info,
        }

    destination_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination_npz,
        bands=bands.astype(np.uint8),
        valid_mask=valid.astype(np.uint8),
        band_names=np.asarray(["red", "green", "blue", "nir"]),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return {
        "height": int(bands.shape[1]),
        "width": int(bands.shape[2]),
        "valid_fraction": float(valid.mean()),
        "has_nir": has_nir,
        "actual_resolution_m": float(metadata["actual_pixel_size_x"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cache clipped USGS NAIP imagery for Oregon patch QC."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_pilot"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Default: patches_qc.csv if present, otherwise patches.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Default: DATASET_DIR/naip.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Force one available NAIP acquisition year.",
    )
    parser.add_argument(
        "--context-size-m",
        type=float,
        default=512.0,
        help="Context width/height around each 256 m patch. Default: 512 m.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.6,
        help="Requested output resolution in metres. Default: 0.6.",
    )
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--max-patches", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.context_size_m <= 0 or args.resolution <= 0:
        parser.error("--context-size-m and --resolution must be positive")

    dataset_dir = args.dataset_dir.resolve()
    qc_manifest = dataset_dir / "patches_qc.csv"
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else (qc_manifest if qc_manifest.exists() else dataset_dir / "patches.csv")
    )
    outdir = (args.outdir or dataset_dir / "naip").resolve()

    if not manifest_path.exists():
        parser.error(f"Patch manifest does not exist: {manifest_path}")

    rows = read_csv(manifest_path)
    required = {
        "patch_id", "split", "tile_name", "x_min", "y_min", "x_max", "y_max", "crs"
    }
    if not rows:
        parser.error("Patch manifest is empty")
    missing = sorted(required - set(rows[0]))
    if missing:
        parser.error(f"Patch manifest is missing columns: {missing}")

    if args.max_patches:
        rows = rows[: args.max_patches]

    rows_by_tile: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_tile[row["tile_name"]].append(row)
    tile_names = sorted(rows_by_tile, key=str.casefold)
    if args.max_tiles:
        tile_names = tile_names[: args.max_tiles]

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json,image/tiff,*/*",
        }
    )

    result_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    temp_dir = outdir / "_temporary"

    print(f"Patch manifest: {manifest_path}")
    print(f"Tiles selected: {len(tile_names)}")
    print(f"NAIP cache: {outdir}")
    print(
        f"Requested context/resolution: {args.context_size_m:g} m / "
        f"{args.resolution:g} m"
    )

    for tile_number, tile_name in enumerate(tile_names, start=1):
        tile_rows = rows_by_tile[tile_name]
        print(f"\n[{tile_number}/{len(tile_names)}] {tile_name}")

        try:
            context_bounds_list: list[tuple[float, float, float, float]] = []
            patch_bounds_list: list[tuple[float, float, float, float]] = []
            for row in tile_rows:
                source_bounds = tuple(
                    float(row[key])
                    for key in ("x_min", "y_min", "x_max", "y_max")
                )
                patch_3857 = transform_bounds(source_bounds, row["crs"], "EPSG:3857")
                context_3857 = expand_bounds(patch_3857, args.context_size_m)
                patch_bounds_list.append(patch_3857)
                context_bounds_list.append(context_3857)

            tile_search_bounds = (
                min(bounds[0] for bounds in context_bounds_list),
                min(bounds[1] for bounds in context_bounds_list),
                max(bounds[2] for bounds in context_bounds_list),
                max(bounds[3] for bounds in context_bounds_list),
            )
            records = query_records(
                session,
                bounds_3857=tile_search_bounds,
                timeout=args.timeout,
                retries=args.retries,
            )
            target_year = infer_year(tile_name)
            selected_year = choose_year(
                records,
                target_year=target_year,
                requested_year=args.year,
            )
            selected_records = [
                record for record in records if int(record.get("Year", -1)) == selected_year
            ]
            print(
                f"  selected NAIP year={selected_year} "
                f"(LiDAR-name year={target_year}, records={len(selected_records)})"
            )

            selections.append(
                {
                    "tile_name": tile_name,
                    "target_lidar_year": target_year,
                    "selected_naip_year": selected_year,
                    "available_years": sorted(
                        {
                            int(record["Year"])
                            for record in records
                            if record.get("Year") not in (None, "")
                        }
                    ),
                    "source_records": selected_records,
                }
            )

            for patch_number, (row, patch_3857, context_3857) in enumerate(
                zip(tile_rows, patch_bounds_list, context_bounds_list),
                start=1,
            ):
                relative = (
                    Path("patches") / row["split"] / f"{row['patch_id']}.npz"
                )
                destination = outdir / relative

                if destination.exists() and not args.overwrite:
                    with np.load(destination) as cached:
                        metadata = json.loads(str(cached["metadata_json"].item()))
                        valid = cached["valid_mask"].astype(bool)
                        shape = cached["valid_mask"].shape
                    result = {
                        "height": int(shape[0]),
                        "width": int(shape[1]),
                        "valid_fraction": float(valid.mean()),
                        "has_nir": bool(metadata.get("has_nir", True)),
                        "actual_resolution_m": float(
                            metadata.get("actual_pixel_size_x", args.resolution)
                        ),
                    }
                    status = "cached"
                else:
                    temporary_tif = temp_dir / f"{row['patch_id']}.tif"
                    export_info = export_naip(
                        session,
                        bounds_3857=context_3857,
                        year=selected_year,
                        resolution_m=args.resolution,
                        output_tif=temporary_tif,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                    result = tif_to_npz(
                        temporary_tif,
                        destination,
                        patch_bounds_3857=patch_3857,
                        context_bounds_3857=context_3857,
                        selected_year=selected_year,
                        source_records=selected_records,
                        export_info=export_info,
                        requested_resolution_m=args.resolution,
                    )
                    temporary_tif.unlink(missing_ok=True)
                    status = "ok"

                result_rows.append(
                    {
                        "patch_id": row["patch_id"],
                        "tile_name": tile_name,
                        "split": row["split"],
                        "naip_path": str(relative),
                        "naip_year": selected_year,
                        "naip_width": result["width"],
                        "naip_height": result["height"],
                        "naip_resolution_m": result["actual_resolution_m"],
                        "naip_valid_fraction": result["valid_fraction"],
                        "naip_has_nir": result["has_nir"],
                        "status": status,
                        "error": "",
                    }
                )
                status_counts[status] += 1
                print(
                    f"  [{patch_number}/{len(tile_rows)}] {row['patch_id']} "
                    f"{result['width']}×{result['height']} "
                    f"valid={100 * result['valid_fraction']:.1f}%"
                )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED TILE: {error}")
            selections.append({"tile_name": tile_name, "error": error})
            for row in tile_rows:
                result_rows.append(
                    {
                        "patch_id": row["patch_id"],
                        "tile_name": tile_name,
                        "split": row["split"],
                        "naip_path": "",
                        "naip_year": "",
                        "naip_width": "",
                        "naip_height": "",
                        "naip_resolution_m": "",
                        "naip_valid_fraction": "",
                        "naip_has_nir": "",
                        "status": "error",
                        "error": error,
                    }
                )
                status_counts["error"] += 1

        fields = [
            "patch_id", "tile_name", "split", "naip_path", "naip_year",
            "naip_width", "naip_height", "naip_resolution_m",
            "naip_valid_fraction", "naip_has_nir", "status", "error",
        ]
        atomic_write_csv(outdir / "naip_manifest.csv", result_rows, fields)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "tile_selections.json").write_text(
            json.dumps(selections, indent=2), encoding="utf-8"
        )

    if temp_dir.exists():
        shutil_errors = []
        for path in temp_dir.glob("*"):
            try:
                path.unlink()
            except OSError as exc:
                shutil_errors.append(str(exc))
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    summary = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_service": SERVICE_ROOT,
        "dataset_dir": str(dataset_dir),
        "patch_manifest": str(manifest_path),
        "outdir": str(outdir),
        "requested_context_size_m": args.context_size_m,
        "requested_resolution_m": args.resolution,
        "status_counts": dict(status_counts),
        "tile_count": len(tile_names),
        "patch_count": len(result_rows),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nStatus counts: {dict(status_counts)}")
    print(f"Manifest: {outdir / 'naip_manifest.csv'}")
    print("Next: run diagnostics/qc_patch_viewer.py --dataset-dir dataset_pilot")
    return 0 if result_rows and status_counts["error"] < len(result_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
