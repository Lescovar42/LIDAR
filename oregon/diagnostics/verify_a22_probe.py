#!/usr/bin/env python3
"""Verify A22 probe downloads against the delivered HTTP object and LAZ content."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import numpy as np
import requests


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


def load_selection(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("A22 probe selection must be a JSON list")
    return [dict(record) for record in raw if isinstance(record, Mapping)]


def delivered_size(url: str, *, timeout: float) -> tuple[int, str, int]:
    response = requests.head(url, allow_redirects=True, timeout=timeout)
    try:
        if response.ok and response.headers.get("Content-Length"):
            return (
                int(response.headers["Content-Length"]),
                response.headers.get("ETag", ""),
                response.status_code,
            )
    finally:
        response.close()

    response = requests.get(
        url,
        headers={"Range": "bytes=0-0"},
        allow_redirects=True,
        stream=True,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            size = int(content_range.rsplit("/", 1)[1])
        elif response.headers.get("Content-Length"):
            size = int(response.headers["Content-Length"])
        else:
            raise RuntimeError("server returned neither Content-Range nor Content-Length")
        return size, response.headers.get("ETag", ""), response.status_code
    finally:
        response.close()


def inspect_laz(path: Path, *, chunk_points: int) -> dict[str, Any]:
    try:
        import laspy
    except ImportError as exc:
        raise RuntimeError("Install LAZ support with: pip install 'laspy[lazrs]'") from exc

    with laspy.open(path) as reader:
        header = reader.header
        crs = header.parse_crs()
        point_count = int(header.point_count)
        ground_count = 0
        for points in reader.chunk_iterator(chunk_points):
            ground_count += int(
                np.count_nonzero(np.asarray(points.classification) == 2)
            )
    return {
        "point_count": point_count,
        "ground_point_count": ground_count,
        "crs": crs.to_string() if crs is not None else "",
    }


def append_exclusions(path: Path, filenames: list[str]) -> None:
    if not filenames:
        return
    existing: list[str] = []
    if path.exists():
        existing = path.read_text(encoding="utf-8").splitlines()
    known = {
        line.strip().casefold()
        for line in existing
        if line.strip() and not line.lstrip().startswith("#")
    }
    additions = [name for name in filenames if name.casefold() not in known]
    if not additions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if path.stat().st_size == 0:
            handle.write("# A22 tiles excluded from representative probes.\n")
        for name in additions:
            handle.write(name + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify selected A22 files using HTTP size, CRS, and class-2 ground points."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--laz-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions-file", type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--chunk-points", type=int, default=5_000_000)
    parser.add_argument("--skip-http", action="store_true")
    args = parser.parse_args()

    if args.chunk_points <= 0:
        parser.error("--chunk-points must be positive")
    if not args.selection.is_file():
        parser.error(f"Missing selection: {args.selection}")
    if not args.laz_dir.is_dir():
        parser.error(f"Missing LAZ directory: {args.laz_dir}")

    try:
        selection = load_selection(args.selection)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    rows: list[dict[str, Any]] = []
    structural_exclusions: list[str] = []

    for index, record in enumerate(selection, 1):
        filename = record_filename(record)
        path = args.laz_dir / filename
        url = str(
            record.get("downloadURL")
            or record.get("downloadUrl")
            or record.get("download_url")
            or ""
        )
        catalog_size = int(
            record.get("sizeInBytes")
            or record.get("size_bytes")
            or record.get("bytes")
            or 0
        )
        row: dict[str, Any] = {
            "filename": filename,
            "url": url,
            "catalog_size": catalog_size,
            "local_size": path.stat().st_size if path.exists() else -1,
            "remote_content_length": -1,
            "etag": "",
            "http_status": -1,
            "point_count": 0,
            "ground_point_count": 0,
            "crs": "",
            "catalog_size_matches_remote": None,
            "local_size_matches_remote": None,
            "valid": False,
            "structural_invalid": False,
            "errors": [],
        }

        if not path.is_file():
            row["errors"].append("missing_local_file")
        if not url:
            row["errors"].append("missing_download_url")

        if not args.skip_http and url:
            try:
                remote_size, etag, status = delivered_size(url, timeout=args.timeout)
                row.update(
                    remote_content_length=remote_size,
                    etag=etag,
                    http_status=status,
                    catalog_size_matches_remote=(catalog_size == remote_size),
                    local_size_matches_remote=(path.is_file() and path.stat().st_size == remote_size),
                )
                if path.is_file() and path.stat().st_size != remote_size:
                    row["errors"].append("local_size_mismatch")
            except Exception as exc:
                row["errors"].append(f"http_error: {type(exc).__name__}: {exc}")
        elif path.is_file():
            row["local_size_matches_remote"] = None

        if path.is_file():
            try:
                info = inspect_laz(path, chunk_points=args.chunk_points)
                row.update(info)
                if not row["crs"]:
                    row["errors"].append("missing_crs")
                    row["structural_invalid"] = True
                if row["ground_point_count"] <= 0:
                    row["errors"].append("no_asprs_class_2_ground")
                    row["structural_invalid"] = True
            except Exception as exc:
                row["errors"].append(f"laz_error: {type(exc).__name__}: {exc}")
                row["structural_invalid"] = True

        size_ok = (
            path.is_file()
            and (
                args.skip_http
                or row["remote_content_length"] <= 0
                or row["local_size_matches_remote"] is True
            )
        )
        network_ok = args.skip_http or not any(
            str(error).startswith("http_error:") for error in row["errors"]
        )
        row["valid"] = bool(
            size_ok
            and network_ok
            and row["point_count"] > 0
            and row["ground_point_count"] > 0
            and row["crs"]
            and not row["structural_invalid"]
        )
        if row["structural_invalid"]:
            structural_exclusions.append(filename)

        print(
            f"[{index}/{len(selection)}] "
            f"{'OK' if row['valid'] else 'FAIL'} {filename} "
            f"local={row['local_size']} remote={row['remote_content_length']} "
            f"ground={row['ground_point_count']}"
        )
        rows.append(row)

    if args.exclusions_file is not None:
        append_exclusions(args.exclusions_file, structural_exclusions)

    summary = {
        "selection": str(args.selection.resolve()),
        "laz_dir": str(args.laz_dir.resolve()),
        "passed": sum(bool(row["valid"]) for row in rows),
        "failed": sum(not bool(row["valid"]) for row in rows),
        "structural_exclusions_added": sorted(set(structural_exclusions)),
        "files": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Wrote: {args.output}")
    if structural_exclusions:
        print(
            "Structural exclusions: "
            + ", ".join(sorted(set(structural_exclusions)))
        )
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
