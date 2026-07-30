"""Resume-safe, budgeted downloader for selected TNM LiDAR tiles.

The downloader accepts both normal selections (a JSON list of tile records) and
probe selections emitted by ``select_tiles.py --probe`` (a JSON mapping from
project name to a list of co-located tile records). Probe groups are written to
separate project directories so they can be passed directly to
``diagnostics/probe_tiles.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urlparse

import requests

LOG_FIELDS = (
    "tile",
    "url",
    "region",
    "path",
    "bytes",
    "elapsed_seconds",
    "status",
    "error",
)
_SELECTION_LIST_KEYS = ("items", "features", "selected_tiles", "tiles", "records")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


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


def _size_matches(actual: int, expected: int) -> bool:
    """Apply exact checks to small fixtures and a 1 KiB tolerance to real tiles."""
    if expected <= 0:
        return actual > 0
    tolerance = 0 if expected < 1_000_000 else 1024
    return abs(actual - expected) <= tolerance


def _write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _safe_component(value: Any, *, label: str, sanitize: bool = False) -> str:
    text = str(value or "").strip()
    if sanitize:
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
        text = re.sub(r"\s+", "_", text).strip(" ._")
        if text.upper() in _WINDOWS_RESERVED_NAMES:
            text += "_"
    component = Path(text)
    if (
        not text
        or component.is_absolute()
        or component.name != text
        or text in {".", ".."}
    ):
        raise ValueError(f"{label} must resolve to one safe directory name: {value!r}")
    return text


def _region_directory(root: Path, region: Any) -> Path:
    if region in (None, ""):
        return root
    value = _safe_component(region, label="region")
    destination = (root / value).resolve()
    resolved_root = root.resolve()
    if resolved_root not in destination.parents:
        raise ValueError(f"region escapes output directory: {region!r}")
    return destination


def _destination_directory(root: Path, region: Any, group: str | None) -> Path:
    destination = _region_directory(root, region)
    if group not in (None, ""):
        group_component = _safe_component(group, label="project group", sanitize=True)
        destination = (destination / group_component).resolve()
        resolved_root = root.resolve()
        if resolved_root not in destination.parents:
            raise ValueError(f"project group escapes output directory: {group!r}")
    return destination


def _validate_tile_list(value: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must contain a list of tile records")
    output: list[dict[str, Any]] = []
    for index, tile in enumerate(value):
        if not isinstance(tile, Mapping):
            raise ValueError(f"{context}[{index}] is not a tile record")
        output.append(dict(tile))
    return output


def _load_selection(path: Path) -> tuple[list[tuple[str | None, dict[str, Any]]], str]:
    """Load normal, wrapped, or project-grouped selection JSON.

    Returns ``(entries, format_name)`` where every entry is ``(group, tile)``.
    ``group`` is populated only for project-keyed probe output.
    """
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, list):
        return [(None, tile) for tile in _validate_tile_list(raw, context="subset JSON")], "list"

    if not isinstance(raw, Mapping):
        raise ValueError("subset JSON must contain a tile list or a project-to-list mapping")

    for key in _SELECTION_LIST_KEYS:
        if key in raw:
            tiles = _validate_tile_list(raw[key], context=f"subset JSON field {key!r}")
            return [(None, tile) for tile in tiles], f"wrapped:{key}"

    if not raw:
        return [], "grouped"

    if not all(isinstance(value, list) for value in raw.values()):
        raise ValueError(
            "subset JSON object must contain one supported tile-list field or map every project to a tile list"
        )

    entries: list[tuple[str | None, dict[str, Any]]] = []
    safe_groups: dict[str, str] = {}
    for project, value in raw.items():
        project_name = str(project).strip()
        safe_name = _safe_component(project_name, label="project group", sanitize=True)
        collision_key = safe_name.casefold()
        previous = safe_groups.get(collision_key)
        if previous is not None and previous != project_name:
            raise ValueError(
                f"project names {previous!r} and {project_name!r} map to the same output directory {safe_name!r}"
            )
        safe_groups[collision_key] = project_name
        tiles = _validate_tile_list(value, context=f"project group {project_name!r}")
        entries.extend((project_name, tile) for tile in tiles)
    return entries, "grouped"


def _relative_directory(root: Path, directory: Path) -> str:
    try:
        return directory.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(directory)


def _preflight_destinations(
    entries: Sequence[tuple[str | None, Mapping[str, Any]]],
    *,
    root: Path,
    region: str | None,
) -> None:
    """Reject output collisions before any transport call or file deletion."""
    destinations: dict[str, tuple[str, str]] = {}
    for group, tile in entries:
        url = str(tile.get("downloadURL") or tile.get("downloadUrl") or tile.get("download_url") or "")
        tile_region = region or tile.get("region_id") or tile.get("_region_id")
        directory = _destination_directory(root, tile_region, group)
        filename = _filename(url, tile)
        key = str((directory / filename).resolve()).casefold()
        identity = (
            url,
            str(tile.get("title") or tile.get("tile_id") or filename),
        )
        previous = destinations.get(key)
        if previous is not None and previous != identity:
            raise ValueError(
                f"multiple tile records resolve to the same output path: {directory / filename}"
            )
        destinations[key] = identity


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
    """Download selected tiles with resume, integrity, and byte-budget guards.

    Supported subset formats:

    * a normal JSON list of tile records;
    * a wrapper such as ``{"items": [...]}``; or
    * the project-keyed mapping written by ``select_tiles.py --probe``.

    For grouped probe selections, each project is written below its own safe
    subdirectory. A supplied region is retained as the parent directory, so the
    layout becomes ``outdir/region/project``.
    """
    if max_total_gb is not None and max_total_gb < 0:
        raise ValueError("max_total_gb must be non-negative")

    subset = Path(subset_path)
    entries, selection_format = _load_selection(subset)
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    _preflight_destinations(entries, root=root, region=region)

    transport = session or requests
    budget_bytes = int(max_total_gb * 1_000_000_000) if max_total_gb is not None else None
    committed_bytes = 0
    rows_by_directory: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    downloaded = skipped = budget_skipped = no_url_skipped = 0
    failures: list[tuple[str, str]] = []

    groups = sorted({group for group, _ in entries if group is not None}, key=str.casefold)
    print(f"Downloading {len(entries)} tile(s) to {root}")
    print(f"Selection format: {selection_format}")
    if groups:
        print("Project groups: " + ", ".join(groups))
    if budget_bytes is not None:
        print(f"Total on-disk budget for this selection: {budget_bytes / 1e9:.3f} GB")

    for index, (group, tile) in enumerate(entries, 1):
        url = str(tile.get("downloadURL") or tile.get("downloadUrl") or tile.get("download_url") or "")
        tile_region = region or tile.get("region_id") or tile.get("_region_id")
        destination_dir = _destination_directory(root, tile_region, group)
        destination_dir.mkdir(parents=True, exist_ok=True)
        started = clock()
        filename = _filename(url, tile)
        out_path = destination_dir / filename
        part_path = out_path.with_name(out_path.name + ".part")
        expected_size = _expected_size(tile)
        record = {
            "tile": str(tile.get("title") or tile.get("tile_id") or filename),
            "url": url,
            "region": _relative_directory(root, destination_dir),
            "path": str(out_path),
            "bytes": 0,
            "elapsed_seconds": 0.0,
            "status": "",
            "error": "",
        }

        if not url:
            no_url_skipped += 1
            record["status"] = "skipped_no_url"
            record["elapsed_seconds"] = round(clock() - started, 6)
            rows_by_directory[destination_dir].append(record)
            print(f"  [{index}/{len(entries)}] SKIP: no downloadURL for {record['tile']}")
            continue

        if out_path.exists():
            actual_size = out_path.stat().st_size
            if _size_matches(actual_size, expected_size):
                if budget_bytes is not None and committed_bytes + actual_size > budget_bytes:
                    raise RuntimeError(
                        "existing selected files already exceed --max-total-gb; "
                        "increase the budget or use a different output directory"
                    )
                committed_bytes += actual_size
                skipped += 1
                record.update(
                    bytes=actual_size,
                    elapsed_seconds=round(clock() - started, 6),
                    status="skipped_existing",
                )
                rows_by_directory[destination_dir].append(record)
                print(f"  [{index}/{len(entries)}] SKIP (already downloaded): {filename}")
                continue
            print(f"  [{index}/{len(entries)}] Existing size mismatch; re-downloading: {filename}")
            out_path.unlink()

        if budget_bytes is not None and expected_size and committed_bytes + expected_size > budget_bytes:
            budget_skipped += 1
            record.update(elapsed_seconds=round(clock() - started, 6), status="skipped_budget")
            rows_by_directory[destination_dir].append(record)
            print(f"  [{index}/{len(entries)}] BUDGET STOP: {filename}")
            continue

        bytes_written = 0
        try:
            if part_path.exists():
                part_path.unlink()
            print(f"  [{index}/{len(entries)}] Downloading: {filename} ({expected_size / 1e6:.1f} MB)")
            response = transport.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            try:
                response_size = max(0, int((response.headers or {}).get("content-length", 0) or 0))
            except (TypeError, ValueError):
                response_size = 0
            authoritative_size = expected_size or response_size
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
                        if authoritative_size:
                            sys.stdout.write(
                                f"\r    {bytes_written / authoritative_size * 100:.1f}% | {speed:.2f} MB/s"
                            )
                        else:
                            sys.stdout.write(f"\r    {bytes_written / 1e6:.1f} MB | {speed:.2f} MB/s")
                        sys.stdout.flush()
                        last_print = now
                output.flush()
                os.fsync(output.fileno())

            if authoritative_size and not _size_matches(bytes_written, authoritative_size):
                raise RuntimeError(
                    f"downloaded size mismatch for {filename}: expected {authoritative_size}, got {bytes_written}"
                )

            os.replace(part_path, out_path)
            committed_bytes += bytes_written
            downloaded += 1
            record.update(
                bytes=bytes_written,
                elapsed_seconds=round(clock() - started, 6),
                status="downloaded",
            )
            print()
        except Exception as exc:
            if part_path.exists():
                part_path.unlink()
            failures.append((filename, str(exc)))
            record.update(
                bytes=bytes_written,
                elapsed_seconds=round(clock() - started, 6),
                status="failed",
                error=str(exc),
            )
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
        "selection_format": selection_format,
        "project_groups": groups,
        "selected": len(entries),
        "downloaded": downloaded,
        "skipped": skipped,
        "no_url_skipped": no_url_skipped,
        "budget_skipped": budget_skipped,
        "failed": len(failures),
        "bytes": committed_bytes,
        "failures": failures,
    }
    print("\n--- Summary ---")
    print(
        f"Downloaded: {downloaded}; skipped: {skipped}; no-url: {no_url_skipped}; "
        f"budget-skipped: {budget_skipped}; failed: {len(failures)}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download normal or grouped LiDAR selections with resume, integrity, and byte-budget guards."
    )
    parser.add_argument("--subset", required=True, type=Path, help="Selected tile JSON or project-grouped probe JSON")
    parser.add_argument("--outdir", default=Path("./lidar_tiles"), type=Path, help="Base output directory")
    parser.add_argument("--region", help="Optional parent region subdirectory name")
    parser.add_argument(
        "--max-total-gb",
        type=float,
        help="Maximum total size of existing plus newly downloaded files from this selection",
    )
    parser.add_argument("--download-log", type=Path, help="Override download_log.csv path")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    args = parser.parse_args()
    if args.max_total_gb is not None and args.max_total_gb < 0:
        parser.error("--max-total-gb must be non-negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    summary = download_tiles(
        args.subset,
        args.outdir,
        max_total_gb=args.max_total_gb,
        region=args.region,
        log_path=args.download_log,
        timeout=args.timeout,
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
