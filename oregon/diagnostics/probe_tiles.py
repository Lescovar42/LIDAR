#!/usr/bin/env python3
"""Coverage diagnostics for candidate LiDAR projects; never writes patches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.ndimage import label

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from region_registry import REGISTRY_PATH, pin_region_decision
from terrain_utils import TerrainTile, iter_patch_windows, read_laz_ground_dem

DEFAULT_CELL_SIZES = (1.0, 1.5, 2.0)


def patch_ground_fractions(mask: np.ndarray, *, patch_size: int = 256, stride: int = 128) -> list[float]:
    return [
        float(mask[row : row + patch_size, col : col + patch_size].mean())
        for row, col in iter_patch_windows(*mask.shape, patch_size=patch_size, stride=stride)
    ]


def _runs(values: np.ndarray) -> list[int]:
    padded = np.pad(np.asarray(values, dtype=bool), (1, 1), constant_values=True)
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [int(edges[index + 1] - edges[index]) for index in range(0, len(edges), 2)]


def void_run_statistics(mask: np.ndarray) -> dict[str, float | int]:
    runs: list[int] = []
    for row in mask:
        runs.extend(_runs(row))
    for column in mask.T:
        runs.extend(_runs(column))
    if not runs:
        return {"run_count": 0, "max_cells": 0, "mean_cells": 0.0, "p95_cells": 0.0}
    return {
        "run_count": len(runs),
        "max_cells": max(runs),
        "mean_cells": float(np.mean(runs)),
        "p95_cells": float(np.percentile(runs, 95)),
    }


def dem_fragmentation(mask: np.ndarray) -> dict[str, float | int]:
    components, count = label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(components.ravel())[1:]
    valid_cells = int(mask.sum())
    return {
        "component_count": int(count),
        "largest_component_cells": int(sizes.max()) if sizes.size else 0,
        "largest_component_fraction": float(sizes.max() / valid_cells) if sizes.size and valid_cells else 0.0,
        "singleton_components": int((sizes == 1).sum()),
    }


def summarize_tile(
    tile: TerrainTile,
    *,
    project: str,
    cell_size: float,
    patch_size: int = 256,
    stride: int = 128,
    min_patch_ground_fraction: float = 0.5,
) -> dict[str, Any]:
    fractions = patch_ground_fractions(tile.valid_ground_mask, patch_size=patch_size, stride=stride)
    histogram_counts, _ = np.histogram(fractions, bins=(0.0, 0.25, 0.5, 0.75, 1.0000001))
    distribution = {
        "count": len(fractions),
        "min": min(fractions) if fractions else None,
        "mean": float(np.mean(fractions)) if fractions else None,
        "median": float(np.median(fractions)) if fractions else None,
        "p05": float(np.percentile(fractions, 5)) if fractions else None,
        "p95": float(np.percentile(fractions, 95)) if fractions else None,
        "histogram": {
            "0.00-0.25": int(histogram_counts[0]),
            "0.25-0.50": int(histogram_counts[1]),
            "0.50-0.75": int(histogram_counts[2]),
            "0.75-1.00": int(histogram_counts[3]),
        },
        "surviving_count": sum(value >= min_patch_ground_fraction for value in fractions),
        "surviving_fraction": float(np.mean(np.asarray(fractions) >= min_patch_ground_fraction)) if fractions else 0.0,
    }
    void_runs = void_run_statistics(tile.valid_ground_mask)
    void_runs.update({
        "max_m": float(void_runs["max_cells"] * cell_size),
        "mean_m": float(void_runs["mean_cells"] * cell_size),
        "p95_m": float(void_runs["p95_cells"] * cell_size),
    })
    return {
        "project": project,
        "tile": tile.source_path.name,
        "cell_size": cell_size,
        "shape": list(tile.shape),
        "ground_cell_fraction": tile.ground_cell_fraction,
        "patch_ground_fraction": distribution,
        "void_runs": void_runs,
        "dem_fragmentation": dem_fragmentation(tile.valid_ground_mask),
    }


def probe_projects(
    projects: dict[str, Sequence[Path]],
    *,
    cell_sizes: Iterable[float] = DEFAULT_CELL_SIZES,
    patch_size: int = 256,
    stride: int = 128,
    min_patch_ground_fraction: float = 0.5,
    max_cells: int = 80_000_000,
    reader: Callable[..., TerrainTile] = read_laz_ground_dem,
) -> dict[str, Any]:
    cell_sizes = tuple(cell_sizes)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for project, paths in projects.items():
        for path in paths:
            for cell_size in cell_sizes:
                try:
                    tile = reader(path, cell_size=cell_size, max_cells=max_cells)
                    rows.append(summarize_tile(
                        tile, project=project, cell_size=cell_size, patch_size=patch_size,
                        stride=stride, min_patch_ground_fraction=min_patch_ground_fraction,
                    ))
                except Exception as exc:
                    failures.append({"project": project, "tile": path.name, "cell_size": str(cell_size), "error": f"{type(exc).__name__}: {exc}"})

    summaries: list[dict[str, Any]] = []
    keys = sorted({(row["project"], row["cell_size"]) for row in rows})
    for project, cell_size in keys:
        group = [row for row in rows if row["project"] == project and row["cell_size"] == cell_size]
        patch_counts = [row["patch_ground_fraction"]["count"] for row in group]
        survivors = [row["patch_ground_fraction"]["surviving_count"] for row in group]
        patch_count = sum(patch_counts)
        void_count = sum(row["void_runs"]["run_count"] for row in group)
        histogram_keys = tuple(group[0]["patch_ground_fraction"]["histogram"])
        summaries.append({
            "project": project,
            "cell_size": cell_size,
            "tile_count": len(group),
            "ground_cell_fraction": {
                "mean": float(np.mean([row["ground_cell_fraction"] for row in group])),
                "min": float(min(row["ground_cell_fraction"] for row in group)),
                "max": float(max(row["ground_cell_fraction"] for row in group)),
            },
            "patch_ground_fraction": {
                "count": patch_count,
                "weighted_mean": (
                    sum(row["patch_ground_fraction"]["mean"] * row["patch_ground_fraction"]["count"] for row in group if row["patch_ground_fraction"]["mean"] is not None) / patch_count
                    if patch_count else None
                ),
                "histogram": {
                    key: sum(row["patch_ground_fraction"]["histogram"][key] for row in group)
                    for key in histogram_keys
                },
                "surviving_count": sum(survivors),
                "surviving_fraction": sum(survivors) / patch_count if patch_count else 0.0,
            },
            "void_runs": {
                "run_count": void_count,
                "max_cells": max(row["void_runs"]["max_cells"] for row in group),
                "max_m": max(row["void_runs"]["max_m"] for row in group),
                "weighted_mean_cells": (
                    sum(row["void_runs"]["mean_cells"] * row["void_runs"]["run_count"] for row in group) / void_count
                    if void_count else 0.0
                ),
            },
            "dem_fragmentation": {
                "component_count": sum(row["dem_fragmentation"]["component_count"] for row in group),
                "singleton_components": sum(row["dem_fragmentation"]["singleton_components"] for row in group),
                "mean_largest_component_fraction": float(np.mean([
                    row["dem_fragmentation"]["largest_component_fraction"] for row in group
                ])),
            },
        })
    return {
        "parameters": {
            "cell_sizes": list(cell_sizes), "patch_size": patch_size, "stride": stride,
            "min_patch_ground_fraction": min_patch_ground_fraction,
        },
        "tiles": rows,
        "project_cell_size_summary": summaries,
        "failures": failures,
    }


def parse_project(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--project must be NAME=LAZ_DIR")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--project must be NAME=LAZ_DIR")
    return name.strip(), Path(raw_path)


def paths_refer_to_same_file(left: Path, right: Path) -> bool:
    """Detect direct, normalized, symlink, and hard-link path aliases."""
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe ground coverage without writing dataset patches.")
    parser.add_argument("--project", action="append", type=parse_project, required=True, metavar="NAME=LAZ_DIR")
    parser.add_argument("--cell-size", action="append", type=float, dest="cell_sizes")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-patch-ground-fraction", type=float, default=0.5)
    parser.add_argument("--max-cells", type=int, default=80_000_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH, help="Registry to update only when all --pin-* options are supplied.")
    parser.add_argument("--pin-region", help="Explicitly pin this registry region after a successful matching probe.")
    parser.add_argument("--pin-lidar-project", help="Candidate project selected from the probe results.")
    parser.add_argument("--pin-cell-size", type=float, help="Positive cell size selected from the probe results.")
    parser.add_argument("--pin-reason", help="Non-empty reasoning for the explicit selection decision.")
    args = parser.parse_args()
    if paths_refer_to_same_file(args.output, args.registry):
        parser.error("--output must not refer to the registry file")
    pin_values = (args.pin_region, args.pin_lidar_project, args.pin_cell_size, args.pin_reason)
    if any(value is not None for value in pin_values) and not all(value is not None for value in pin_values):
        parser.error("--pin-region, --pin-lidar-project, --pin-cell-size, and --pin-reason must be supplied together")
    if args.pin_reason is not None and not args.pin_reason.strip():
        parser.error("--pin-reason must be non-empty")
    if args.patch_size <= 0 or args.stride <= 0:
        parser.error("--patch-size and --stride must be positive")
    if not 0 <= args.min_patch_ground_fraction <= 1:
        parser.error("--min-patch-ground-fraction must be between 0 and 1")
    cell_sizes = tuple(args.cell_sizes or DEFAULT_CELL_SIZES)
    if any(value <= 0 for value in cell_sizes):
        parser.error("--cell-size must be positive")

    projects: dict[str, list[Path]] = {}
    for name, directory in args.project:
        if not directory.is_dir():
            parser.error(f"Project directory does not exist: {directory}")
        projects.setdefault(name, []).extend(sorted([*directory.glob("*.laz"), *directory.glob("*.las")]))
    result = probe_projects(
        projects, cell_sizes=cell_sizes, patch_size=args.patch_size, stride=args.stride,
        min_patch_ground_fraction=args.min_patch_ground_fraction, max_cells=args.max_cells,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.pin_region is not None:
        measured = any(
            row["project"] == args.pin_lidar_project and row["cell_size"] == args.pin_cell_size
            for row in result["project_cell_size_summary"]
        )
        selected_failures = [
            failure
            for failure in result["failures"]
            if failure["project"] == args.pin_lidar_project
            and float(failure["cell_size"]) == args.pin_cell_size
        ]
        if not measured or selected_failures:
            parser.error(
                "the pinned project/cell-size pair must have complete successful results in this probe"
            )
        try:
            pin_region_decision(
                args.registry,
                args.pin_region,
                lidar_project=args.pin_lidar_project,
                cell_size=args.pin_cell_size,
                reason=args.pin_reason,
                decision_metadata={"probe_output": str(args.output.resolve())},
            )
        except (KeyError, ValueError, OSError) as exc:
            parser.error(str(exc))
        print(f"Pinned {args.pin_region} to {args.pin_lidar_project} at {args.pin_cell_size:g}m in {args.registry}.")
    for row in result["project_cell_size_summary"]:
        print(
            f"{row['project']} {row['cell_size']:.1f}m: ground={row['ground_cell_fraction']['mean']:.3f}, "
            f"patches={row['patch_ground_fraction']['surviving_count']}/{row['patch_ground_fraction']['count']}"
        )
    print(f"Wrote {args.output}; no patch files were created.")
    return 0 if result["tiles"] and not result["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
