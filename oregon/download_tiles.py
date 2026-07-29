"""Resume-safe, budgeted downloader for selected TNM LiDAR tiles."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlparse

import requests

LOG_FIELDS = ("tile", "url", "region", "path", "bytes", "elapsed_seconds", "status", "error")


def _filename(url: str, tile: Mapping[str, Any]) -> str:
    name = Path(unquote(urlparse(url).path)).name
    if name:
        return name
    fallback = str(tile.get("title") or tile.get("tile_id") or "tile.laz")
    return fallback if Path(fallback).suffix else f"{fallback}.laz"


def _expected_size(tile: Mapping[str, Any]) -> int:
    try:
        return max(0, int(tile.get("sizeInBytes") or tile.get("size_bytes") or 0))
    except (TypeError, ValueError):
        return 0


def _write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _region_directory(root: Path, region: Any) -> Path:
    if region in (None, ""):
        return root
    value = str(region)
    component = Path(value)
    if component.is_absolute() or component.name != value or value in {".", ".."}:
        raise ValueError(f"region must be a single safe directory name: {value!r}")
    destination = (root / component).resolve()
    resolved_root = root.resolve()
    if resolved_root not in destination.parents:
        raise ValueError(f"region escapes output directory: {value!r}")
    return destination


def download_tiles(
    subset_path: str | Path,
    outdir: str | Path,
    *,
    max_total_gb: float | None = None,
    region: str | None = None,
    session: Any = None,
    log_path: str | Path | None = None,
    timeout: float = 120,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Download a JSON tile list, with injectable transport for offline tests.

    The original ``download_tiles(subset_path, outdir)`` call remains valid.
    A supplied region writes beneath ``outdir/region``; otherwise a tile's
    ``region_id``/``_region_id`` is honored when present.
    """
    if max_total_gb is not None and max_total_gb < 0:
        raise ValueError("max_total_gb must be non-negative")
    with Path(subset_path).open(encoding="utf-8") as handle:
        tiles = json.load(handle)
    if not isinstance(tiles, list):
        raise ValueError("subset JSON must contain a list of tile records")

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    transport = session or requests
    budget_bytes = int(max_total_gb * 1_000_000_000) if max_total_gb is not None else None
    committed_bytes = 0
    rows_by_directory: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    downloaded = skipped = budget_skipped = 0
    failures: list[tuple[str, str]] = []

    print(f"Downloading {len(tiles)} tile(s) to {root}")
    if budget_bytes is not None:
        print(f"Run budget: {budget_bytes / 1e9:.3f} GB")

    for index, tile in enumerate(tiles, 1):
        url = str(tile.get("downloadURL") or tile.get("downloadUrl") or tile.get("download_url") or "")
        tile_region = region or tile.get("region_id") or tile.get("_region_id")
        destination_dir = _region_directory(root, tile_region)
        destination_dir.mkdir(parents=True, exist_ok=True)
        started = clock()
        filename = _filename(url, tile)
        out_path = destination_dir / filename
        part_path = out_path.with_name(out_path.name + ".part")
        expected_size = _expected_size(tile)
        record = {
            "tile": str(tile.get("title") or tile.get("tile_id") or filename),
            "url": url,
            "region": str(tile_region or ""),
            "path": str(out_path),
            "bytes": 0,
            "elapsed_seconds": 0.0,
            "status": "",
            "error": "",
        }

        if not url:
            record["status"] = "skipped_no_url"
            record["elapsed_seconds"] = round(clock() - started, 6)
            rows_by_directory[destination_dir].append(record)
            print(f"  [{index}/{len(tiles)}] SKIP: no downloadURL for {record['tile']}")
            continue

        if out_path.exists():
            actual_size = out_path.stat().st_size
            size_matches = abs(actual_size - expected_size) < 1024 if expected_size else actual_size > 0
            if size_matches:
                committed_bytes += actual_size
                skipped += 1
                record.update(bytes=actual_size, elapsed_seconds=round(clock() - started, 6), status="skipped_existing")
                rows_by_directory[destination_dir].append(record)
                print(f"  [{index}/{len(tiles)}] SKIP (already downloaded): {filename}")
                continue
            print(f"  [{index}/{len(tiles)}] Existing size mismatch; re-downloading: {filename}")
            out_path.unlink()

        if budget_bytes is not None and expected_size and committed_bytes + expected_size > budget_bytes:
            budget_skipped += 1
            record.update(elapsed_seconds=round(clock() - started, 6), status="skipped_budget")
            rows_by_directory[destination_dir].append(record)
            print(f"  [{index}/{len(tiles)}] BUDGET STOP: {filename}")
            continue

        bytes_written = 0
        try:
            if part_path.exists():
                part_path.unlink()
            print(f"  [{index}/{len(tiles)}] Downloading: {filename} ({expected_size / 1e6:.1f} MB)")
            response = transport.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            total_length = expected_size or int(response.headers.get("content-length", 0))
            last_print = started
            with part_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    if budget_bytes is not None and committed_bytes + bytes_written + len(chunk) > budget_bytes:
                        raise RuntimeError("download would exceed --max-total-gb")
                    output.write(chunk)
                    bytes_written += len(chunk)
                    now = clock()
                    if now - last_print > 0.5:
                        elapsed = max(now - started, 1e-12)
                        speed = (bytes_written / 1e6) / elapsed
                        if total_length:
                            sys.stdout.write(f"\r    {bytes_written / total_length * 100:.1f}% | {speed:.2f} MB/s")
                        else:
                            sys.stdout.write(f"\r    {bytes_written / 1e6:.1f} MB | {speed:.2f} MB/s")
                        sys.stdout.flush()
                        last_print = now
            os.replace(part_path, out_path)
            committed_bytes += bytes_written
            downloaded += 1
            record.update(bytes=bytes_written, elapsed_seconds=round(clock() - started, 6), status="downloaded")
            print()
        except Exception as exc:
            if part_path.exists():
                part_path.unlink()
            # Remove only the temporary stream; mismatched finals were removed before retry.
            failures.append((filename, str(exc)))
            record.update(bytes=bytes_written, elapsed_seconds=round(clock() - started, 6), status="failed", error=str(exc))
            print(f"\n    FAILED: {exc}")
        rows_by_directory[destination_dir].append(record)

    if log_path is not None:
        all_rows = [row for rows in rows_by_directory.values() for row in rows]
        _write_log(Path(log_path), all_rows)
    else:
        if not rows_by_directory:
            rows_by_directory[root] = []
        for directory, rows in rows_by_directory.items():
            _write_log(directory / "download_log.csv", rows)

    summary = {
        "downloaded": downloaded,
        "skipped": skipped,
        "budget_skipped": budget_skipped,
        "failed": len(failures),
        "bytes": committed_bytes,
        "failures": failures,
    }
    print("\n--- Summary ---")
    print(f"Downloaded: {downloaded}; skipped: {skipped}; budget-skipped: {budget_skipped}; failed: {len(failures)}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a selected LiDAR subset with resume and byte-budget guards.")
    parser.add_argument("--subset", required=True, type=Path, help="Path to selected tile JSON")
    parser.add_argument("--outdir", default=Path("./lidar_tiles"), type=Path, help="Base output directory")
    parser.add_argument("--region", help="Optional region subdirectory name")
    parser.add_argument("--max-total-gb", type=float, help="Maximum bytes committed by this invocation")
    parser.add_argument("--download-log", type=Path, help="Override download_log.csv path")
    args = parser.parse_args()
    if args.max_total_gb is not None and args.max_total_gb < 0:
        parser.error("--max-total-gb must be non-negative")
    summary = download_tiles(
        args.subset, args.outdir, max_total_gb=args.max_total_gb,
        region=args.region, log_path=args.download_log,
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
