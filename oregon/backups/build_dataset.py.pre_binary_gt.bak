#!/usr/bin/env python3
"""Build a multi-region, leakage-resistant LiDAR/SLIDO patch dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from lidar_vintage import (
    CLI_ORIGIN,
    REGISTRY_ORIGIN,
    AcquisitionMetadataError,
    LidarAcquisition,
    LidarVintage,
    acquisition_from_cli,
    file_metadata_from_header,
    infer_year_hint,
    parse_year_hint,
    resolve_acquisition,
    summarize_vintages,
    VINTAGE_ROW_FIELDS,
)
from region_registry import (
    REGISTRY_PATH,
    load_registry,
    region_acquisition,
    resolve_path,
    resolve_region,
)
from terrain_utils import (
    build_feature_stack,
    classify_patch,
    intersecting_ids,
    iter_patch_windows,
    label_quality,
    rasterize_slido_mask,
    read_laz_ground_dem,
)

METRIC_CRS = CRS.from_epsg(5070)
VALID_SPLITS = {"train", "validation", "test_rural", "test_urban_ood"}

#: ``patches.csv`` column order. Temporal provenance columns come from
#: ``lidar_vintage.VINTAGE_ROW_FIELDS`` so every writer/reader agrees.
PATCH_FIELDS: tuple[str, ...] = (
    "patch_id",
    "split",
    "category",
    "region_id",
    "region_role",
    "lidar_project",
    *VINTAGE_ROW_FIELDS,
    "label_quality",
    "tile_name",
    "tile_path",
    "patch_path",
    "row_offset",
    "col_offset",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "crs",
    "positive_fraction",
    "ignore_fraction",
    "ground_fraction",
    "mean_slope_degrees",
    "distance_to_positive_m",
    "is_hard_negative",
    "landslide_ids_in_tile",
    "slido_ref_ids_in_tile",
    "qc_status",
    "qc_notes",
)


@dataclass(frozen=True)
class RegionInput:
    region_id: str
    region_role: str
    laz_dir: Path
    slido_path: Path
    lidar_project_hint: str = ""
    cell_size: float | None = None
    lidar_project_pinned: bool = False
    lidar_acquisition: LidarAcquisition | None = None


@dataclass(frozen=True)
class SpatialTile:
    """Minimal projected tile record used by the deterministic splitter."""
    tile_id: str
    region_id: str
    region_role: str
    footprint: BaseGeometry


@dataclass(frozen=True)
class TileMetadata:
    path: Path
    tile_id: str
    region_id: str
    region_role: str
    lidar_project: str
    vintage: LidarVintage
    source_crs: CRS
    metric_footprint: BaseGeometry

    @property
    def lidar_year(self) -> int | None:
        """Compatibility alias for the authoritative nominal acquisition year."""
        return self.vintage.legacy_lidar_year

    @property
    def lidar_year_source(self) -> str:
        return self.vintage.legacy_lidar_year_source

    def slido_lidar_year(self) -> int | None:
        """The only year permitted to drive SLIDO temporal filtering."""
        return self.vintage.temporal_filter_year(
            f"SLIDO temporal filtering for {self.tile_id}"
        )


@dataclass(frozen=True)
class SplitResult:
    assignments: dict[str, str]
    dropped: dict[str, str]
    blocks: dict[str, tuple[int, int]]


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "tile"


def parse_lidar_year(text: str) -> int | None:
    """Parse a NON-AUTHORITATIVE four-digit year or TNM-style ``A22`` token.

    Retained as a diagnostic hint parser only. A project or filename token is
    never acquisition evidence; see ``lidar_vintage`` for the authoritative path.
    """
    return parse_year_hint(text)


def _project_from_name(path: Path) -> str:
    stem = path.stem
    match = re.match(r"(.+?)(?:_w\d+n\d+|_\d{4,}_[0-9]+)$", stem, re.I)
    return (match.group(1) if match else stem).strip("_-")


def _normalized_project_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", value.casefold())
    return token.removeprefix("usgslpc")


def tile_matches_project(path: Path, lidar_project: str) -> bool:
    """Match TNM-style tile names to a pinned candidate without relabeling stale files."""
    project_token = _normalized_project_token(lidar_project)
    tile_token = _normalized_project_token(path.stem)
    return bool(project_token) and project_token in tile_token


def read_tile_metadata(path: Path, region: RegionInput) -> TileMetadata:
    """Read only a LAS/LAZ header; no point records are loaded."""
    try:
        import laspy
    except ImportError as exc:
        raise RuntimeError("Install LAZ support with: pip install 'laspy[lazrs]'") from exc

    with laspy.open(path) as reader:
        header = reader.header
        source_crs_value = header.parse_crs()
        if source_crs_value is None:
            raise RuntimeError(f"No CRS found in LAZ header: {path.name}")
        source_crs = CRS.from_user_input(source_crs_value)
        xmin, ymin = float(header.mins[0]), float(header.mins[1])
        xmax, ymax = float(header.maxs[0]), float(header.maxs[1])
        header_creation_date = getattr(header, "creation_date", None)

    lidar_project = region.lidar_project_hint or _project_from_name(path)
    acquisition = region.lidar_acquisition
    if (
        acquisition is not None
        and acquisition.lidar_project
        and not tile_matches_project(path, acquisition.lidar_project)
    ):
        raise RuntimeError(
            f"Acquisition metadata is declared for LiDAR project "
            f"{acquisition.lidar_project!r} but tile {path.name} does not belong to it; "
            "acquisition vintages must not be reused across projects"
        )
    # The header date and any name token are recorded as provenance only. The
    # authoritative acquisition vintage comes from the region/CLI resolution.
    vintage = LidarVintage(
        acquisition=acquisition,
        file_metadata=file_metadata_from_header(header_creation_date),
        hint=infer_year_hint(lidar_project, path.name),
    )

    transformer = Transformer.from_crs(source_crs, METRIC_CRS, always_xy=True)
    metric_footprint = shapely_transform(transformer.transform, box(xmin, ymin, xmax, ymax))
    tile_id = f"{region.region_id}:{path.name}"
    return TileMetadata(
        path=path,
        tile_id=tile_id,
        region_id=region.region_id,
        region_role=region.region_role,
        lidar_project=lidar_project,
        vintage=vintage,
        source_crs=source_crs,
        metric_footprint=metric_footprint,
    )


def rasterize_tile_mask(
    item: TileMetadata,
    region: RegionInput,
    tile: Any,
    *,
    description: str = "Landslide",
    positive_buffer_m: float = 0.0,
    all_touched: bool = False,
    rasterize: Any = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Rasterize SLIDO labels using ONLY the authoritative acquisition year.

    Isolated so tests can assert the exact ``lidar_year`` argument that reaches
    temporal filtering. ``rasterize`` is injectable for that purpose.
    """
    rasterize_fn = rasterize_slido_mask if rasterize is None else rasterize
    return rasterize_fn(
        region.slido_path,
        tile,
        description=description,
        lidar_year=item.slido_lidar_year(),
        positive_buffer_m=positive_buffer_m,
        all_touched=all_touched,
    )


def _block_for(footprint: BaseGeometry, block_size_m: float) -> tuple[int, int]:
    centroid = footprint.centroid
    return math.floor(centroid.x / block_size_m), math.floor(centroid.y / block_size_m)


def _block_polygon(block: tuple[int, int], block_size_m: float) -> BaseGeometry:
    x, y = block
    return box(x * block_size_m, y * block_size_m, (x + 1) * block_size_m, (y + 1) * block_size_m)


def _stable_region_seed(seed: int, region_id: str) -> int:
    digest = hashlib.sha1(f"{seed}:{region_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def assign_spatial_splits(
    tiles: Sequence[SpatialTile],
    *,
    seed: int = 42,
    block_size_m: float = 5_000.0,
    split_buffer_m: float = 500.0,
    validation_fraction: float = 0.20,
) -> SplitResult:
    """Assign occupied metric blocks and drop boundary-near train/validation tiles."""
    if block_size_m <= 0:
        raise ValueError("block_size_m must be positive")
    if split_buffer_m < 0:
        raise ValueError("split_buffer_m must be non-negative")
    if not 0 <= validation_fraction <= 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    assignments: dict[str, str] = {}
    dropped: dict[str, str] = {}
    blocks = {tile.tile_id: _block_for(tile.footprint, block_size_m) for tile in tiles}
    grouped: dict[str, list[SpatialTile]] = defaultdict(list)
    for tile in tiles:
        grouped[tile.region_id].append(tile)

    for region_id, region_tiles in sorted(grouped.items()):
        roles = {tile.region_role for tile in region_tiles}
        if len(roles) != 1:
            raise ValueError(f"Region {region_id} has inconsistent roles: {sorted(roles)}")
        role = next(iter(roles))
        if role in {"test_rural", "test_urban_ood"}:
            for tile in region_tiles:
                assignments[tile.tile_id] = role
            continue
        if role != "train_val":
            raise ValueError(f"Unknown region role for {region_id}: {role}")

        occupied = sorted({blocks[tile.tile_id] for tile in region_tiles})
        shuffled = occupied.copy()
        random.Random(_stable_region_seed(seed, region_id)).shuffle(shuffled)
        if len(shuffled) <= 1 or validation_fraction == 0:
            validation_blocks: set[tuple[int, int]] = set()
        else:
            count = max(1, round(len(shuffled) * validation_fraction))
            if validation_fraction < 1:
                count = min(count, len(shuffled) - 1)
            validation_blocks = set(shuffled[:count])
        block_splits = {
            block: "validation" if block in validation_blocks else "train" for block in occupied
        }
        for tile in region_tiles:
            assignments[tile.tile_id] = block_splits[blocks[tile.tile_id]]

        if split_buffer_m > 0 and len(set(block_splits.values())) > 1:
            polygons = {block: _block_polygon(block, block_size_m) for block in occupied}
            for tile in region_tiles:
                own_split = assignments[tile.tile_id]
                nearest = min(
                    (
                        tile.footprint.distance(polygons[block])
                        for block, split in block_splits.items()
                        if split != own_split
                    ),
                    default=math.inf,
                )
                if nearest <= split_buffer_m:
                    dropped[tile.tile_id] = (
                        f"footprint is {nearest:.3f} m from a differently assigned block "
                        f"(required > {split_buffer_m:.3f} m)"
                    )
                    assignments.pop(tile.tile_id, None)

    if set(assignments.values()) - VALID_SPLITS:
        raise AssertionError("splitter produced an unsupported split")
    return SplitResult(assignments=assignments, dropped=dropped, blocks=blocks)


def assign_tile_splits(
    tiles: Sequence[SpatialTile], seed: int = 42, *, block_size_m: float = 5_000.0, split_buffer_m: float = 500.0
) -> dict[str, str]:
    """Compatibility wrapper returning only retained tile assignments."""
    return assign_spatial_splits(
        tiles, seed=seed, block_size_m=block_size_m, split_buffer_m=split_buffer_m
    ).assignments


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_negative_candidates(
    candidates: list[dict[str, Any]], *, positive_count: int, ratio: float,
    max_per_tile: int, seed_text: str,
) -> list[dict[str, Any]]:
    if not candidates or max_per_tile <= 0:
        return []
    desired = max(10, int(math.ceil(max(1, positive_count) * ratio)))
    desired = min(desired, max_per_tile, len(candidates))
    seed = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    hard = [row for row in candidates if row["is_hard_negative"]]
    ordinary = [row for row in candidates if not row["is_hard_negative"]]
    rng.shuffle(hard)
    rng.shuffle(ordinary)
    hard_target = min(len(hard), int(math.ceil(desired * 0.70)))
    selected = hard[:hard_target]
    remainder = desired - len(selected)
    selected.extend(ordinary[:remainder])
    remainder = desired - len(selected)
    if remainder:
        selected.extend(hard[hard_target : hard_target + remainder])
    return selected


def _flatten_region_args(region: list[str] | None, regions: list[list[str]] | None) -> list[str]:
    values = list(region or [])
    for group in regions or []:
        for item in group:
            values.extend(part for part in item.split(",") if part)
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def cli_acquisition(args: argparse.Namespace) -> LidarAcquisition | None:
    """Build the explicit CLI acquisition override, if any was supplied."""
    return acquisition_from_cli(
        year=getattr(args, "lidar_acquisition_year", None),
        start_year=getattr(args, "lidar_acquisition_start_year", None),
        end_year=getattr(args, "lidar_acquisition_end_year", None),
        source=getattr(args, "lidar_acquisition_source", None),
        evidence=getattr(args, "lidar_acquisition_evidence", None),
        verified=bool(getattr(args, "lidar_acquisition_verified", False)),
        origin=CLI_ORIGIN,
    )


def missing_acquisition_regions(region_inputs: Sequence[RegionInput]) -> list[str]:
    """Regions that reached the builder without authoritative acquisition metadata."""
    return [
        region.region_id for region in region_inputs if region.lidar_acquisition is None
    ]


def acquisition_scalar_errors(region_inputs: Sequence[RegionInput]) -> list[str]:
    """Regions whose multi-year acquisition range lacks a required nominal year."""
    errors: list[str] = []
    for region in region_inputs:
        acquisition = region.lidar_acquisition
        if acquisition is None:
            continue
        try:
            acquisition.require_nominal_year(
                f"Region {region.region_id} SLIDO temporal filtering"
            )
        except AcquisitionMetadataError as exc:
            errors.append(str(exc))
    return errors


def resolve_region_inputs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[RegionInput]:
    names = _flatten_region_args(args.region, args.regions)
    try:
        override = cli_acquisition(args)
    except AcquisitionMetadataError as exc:
        parser.error(str(exc))
    if not names:
        return [
            RegionInput(
                region_id="legacy",
                region_role="train_val",
                laz_dir=args.laz_dir.resolve(),
                slido_path=args.slido_geojson.resolve(),
                lidar_acquisition=override,
            )
        ]

    try:
        registry = load_registry(args.registry)
        entries = [resolve_region(name, registry) for name in names]
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))
    registry_path = Path(registry["_registry_path"])
    root = args.laz_dir.resolve()
    output: list[RegionInput] = []
    for entry in entries:
        if entry.get("lidar_dir"):
            laz_dir = resolve_path(entry, "lidar_dir", registry_path)
        else:
            candidates = [root / str(entry["slug"]), root / str(entry["id"])]
            laz_dir = next((path for path in candidates if path.exists()), candidates[0])
        projects = entry.get("candidate_projects") or []
        pinned_project = str(entry.get("lidar_project") or "")
        pinned_cell_size = entry.get("cell_size")
        is_rural = entry["role"] in {"train_val", "test_rural"}
        if is_rural and (not pinned_project or pinned_cell_size is None) and not getattr(
            args, "allow_unpinned_rural_diagnostic", False
        ):
            parser.error(
                f"Region {entry['id']} has no pinned lidar_project/cell_size; run the probe and pin a decision, "
                "or use --allow-unpinned-rural-diagnostic only for diagnostics"
            )
        project_hint = pinned_project or (str(projects[0]) if len(projects) == 1 else "")
        try:
            registry_acquisition = region_acquisition(
                entry, expected_project=project_hint or None
            )
            acquisition = resolve_acquisition(
                [
                    (f"{CLI_ORIGIN} (CLI)", override),
                    (f"{REGISTRY_ORIGIN} ({entry['id']} in {registry_path.name})",
                     registry_acquisition),
                ],
                context=f"region {entry['id']}",
            )
        except AcquisitionMetadataError as exc:
            parser.error(str(exc))
        output.append(
            RegionInput(
                region_id=str(entry["id"]),
                region_role=str(entry["role"]),
                laz_dir=laz_dir.resolve(),
                slido_path=resolve_path(entry, "slido_output", registry_path).resolve(),
                lidar_project_hint=project_hint,
                cell_size=float(pinned_cell_size) if pinned_cell_size is not None else None,
                lidar_project_pinned=bool(pinned_project),
                lidar_acquisition=acquisition,
            )
        )
    return output


def resolve_build_cell_size(
    requested_cell_size: float | None,
    region_inputs: Sequence[RegionInput],
    parser: argparse.ArgumentParser,
) -> float:
    """Resolve the legacy default or a compatible registry-pinned resolution."""
    if len(region_inputs) == 1 and region_inputs[0].region_id == "legacy":
        return 1.0 if requested_cell_size is None else requested_cell_size

    pinned = {region.cell_size for region in region_inputs if region.cell_size is not None}
    if len(pinned) > 1:
        values = ", ".join(f"{value:g}" for value in sorted(pinned))
        parser.error(f"Requested registry regions have incompatible pinned cell sizes: {values}")
    pinned_cell_size = next(iter(pinned), None)
    unpinned = [region.region_id for region in region_inputs if region.cell_size is None]
    if unpinned and requested_cell_size is None:
        parser.error(
            "Registry regions without a pinned cell_size require an explicit --cell-size for diagnostic use: "
            + ", ".join(unpinned)
        )
    if (
        pinned_cell_size is not None
        and requested_cell_size is not None
        and not math.isclose(pinned_cell_size, requested_cell_size, rel_tol=0.0, abs_tol=1e-12)
    ):
        parser.error(
            f"--cell-size {requested_cell_size:g} conflicts with registry-pinned cell_size {pinned_cell_size:g}"
        )
    return pinned_cell_size if pinned_cell_size is not None else float(requested_cell_size)


def _region_summaries(
    region_inputs: Sequence[RegionInput],
    tile_summaries: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    failed_tiles: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Summarize every requested region, including empty or failed regions."""
    result: dict[str, Any] = {}
    for region in region_inputs:
        region_id = region.region_id
        region_tiles = [item for item in tile_summaries if item["region_id"] == region_id]
        region_rows = [row for row in rows if row["region_id"] == region_id]
        region_failures = [item for item in failed_tiles if item["region_id"] == region_id]
        pixels = sum(int(item["mask_pixels"]) for item in region_tiles)
        ignored = sum(int(item["ignore_pixels"]) for item in region_tiles)
        temporal_keys = {
            str(key)
            for item in region_tiles
            for key in item.get("temporally_excluded_polygon_keys", [])
        }
        result[region_id] = {
            "region_role": region.region_role,
            "lidar_acquisition": (
                None
                if region.lidar_acquisition is None
                else region.lidar_acquisition.as_summary_mapping()
            ),
            "status": "complete" if region_tiles else "incomplete",
            "processed_tiles": len(region_tiles),
            "failed_or_dropped_tiles": len(region_failures),
            "saved_patches": len(region_rows),
            "split_counts": dict(Counter(str(row["split"]) for row in region_rows)),
            "category_counts": dict(Counter(str(row["category"]) for row in region_rows)),
            "ignore_pixel_fraction": float(ignored / pixels) if pixels else 0.0,
            "temporally_excluded_polygon_count": len(temporal_keys),
        }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an auditable multi-region rural landslide patch dataset.")
    parser.add_argument("--region", action="append", help="Registry region id/slug/name; repeat as needed.")
    parser.add_argument("--regions", action="append", nargs="+", help="One or more registry regions; may be repeated.")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--laz-dir", type=Path, default=Path("lidar_tiles"), help="Legacy tile directory or regional tile root.")
    parser.add_argument("--slido-geojson", type=Path, default=Path("slido_deposits_oregon_city.geojson"), help="Legacy single-region SLIDO path.")
    parser.add_argument("--outdir", type=Path, default=Path("dataset_pilot"))
    parser.add_argument(
        "--cell-size",
        type=float,
        default=None,
        help="Legacy resolution (default 1.0m), or explicit diagnostic resolution for an unpinned registry region.",
    )
    parser.add_argument(
        "--allow-unpinned-rural-diagnostic",
        action="store_true",
        help="DIAGNOSTIC ONLY: allow an unpinned train/test rural registry region; requires explicit --cell-size.",
    )
    acquisition = parser.add_argument_group(
        "authoritative LiDAR acquisition metadata",
        "Explicit airborne acquisition vintage for legacy/single-region builds. The LAS/LAZ "
        "header creation_date, project tokens such as A22, and filename years are NEVER used "
        "as acquisition evidence.",
    )
    acquisition.add_argument(
        "--lidar-acquisition-year",
        "--lidar-year",
        dest="lidar_acquisition_year",
        type=int,
        default=None,
        help="Nominal acquisition year used for SLIDO temporal filtering and NAIP selection. "
        "--lidar-year is a deprecated alias routed to this field.",
    )
    acquisition.add_argument(
        "--lidar-acquisition-start-year",
        dest="lidar_acquisition_start_year",
        type=int,
        default=None,
        help="First year of a multi-year survey; requires --lidar-acquisition-end-year.",
    )
    acquisition.add_argument(
        "--lidar-acquisition-end-year",
        dest="lidar_acquisition_end_year",
        type=int,
        default=None,
        help="Last year of a multi-year survey; requires --lidar-acquisition-start-year.",
    )
    acquisition.add_argument(
        "--lidar-acquisition-source",
        dest="lidar_acquisition_source",
        default=None,
        help="Required whenever acquisition metadata is supplied: who/what states the vintage.",
    )
    acquisition.add_argument(
        "--lidar-acquisition-evidence",
        dest="lidar_acquisition_evidence",
        default=None,
        help="Traceable record or repository path backing the acquisition claim.",
    )
    acquisition.add_argument(
        "--lidar-acquisition-verified",
        dest="lidar_acquisition_verified",
        action="store_true",
        help="Mark the acquisition metadata as verified against the cited evidence.",
    )
    acquisition.add_argument(
        "--allow-unknown-lidar-acquisition",
        action="store_true",
        help="DIAGNOSTIC ONLY: build without authoritative acquisition metadata. SLIDO temporal "
        "filtering is then disabled for the affected regions and no vintage gap is claimed.",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-tiles", type=int, default=10, help="Maximum tiles per region; 0 means all.")
    parser.add_argument("--max-cells", type=int, default=80_000_000)
    parser.add_argument("--min-ground-cell-fraction", type=float, default=0.25)
    parser.add_argument("--min-patch-ground-fraction", type=float, default=0.50)
    parser.add_argument("--interior-threshold", type=float, default=0.10)
    parser.add_argument("--boundary-threshold", type=float, default=0.01)
    parser.add_argument("--include-trace-positives", action="store_true")
    parser.add_argument("--negative-buffer-m", type=float, default=50.0)
    parser.add_argument(
        "--positive-buffer-m", "--positive-ignore-buffer-m",
        dest="positive_ignore_buffer_m", type=float, default=50.0,
        help="Width of the uint8=255 ring outside accepted positives.",
    )
    parser.add_argument("--hard-negative-slope", type=float, default=8.0)
    parser.add_argument("--negative-ratio", type=float, default=1.5)
    parser.add_argument("--max-negatives-per-tile", type=int, default=100)
    parser.add_argument("--block-size-m", type=float, default=5_000.0)
    parser.add_argument("--split-buffer-m", type=float, default=500.0)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--all-touched", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.patch_size <= 0 or args.stride <= 0:
        parser.error("--patch-size and --stride must be positive")
    if args.cell_size is not None and args.cell_size <= 0:
        parser.error("--cell-size must be positive")
    if args.block_size_m <= 0:
        parser.error("--block-size-m must be positive")
    if args.split_buffer_m < 0 or args.positive_ignore_buffer_m < 0:
        parser.error("buffer distances must be non-negative")
    if not 0 <= args.min_ground_cell_fraction <= 1 or not 0 <= args.min_patch_ground_fraction <= 1:
        parser.error("ground fractions must be between 0 and 1")
    if not 0 <= args.boundary_threshold <= args.interior_threshold <= 1:
        parser.error("thresholds must satisfy 0 <= boundary <= interior <= 1")
    if not 0 <= args.validation_fraction <= 1:
        parser.error("--validation-fraction must be between 0 and 1")

    region_inputs = resolve_region_inputs(args, parser)
    args.cell_size = resolve_build_cell_size(args.cell_size, region_inputs, parser)

    scalar_errors = acquisition_scalar_errors(region_inputs)
    if scalar_errors:
        parser.error(" | ".join(scalar_errors))
    unknown_acquisition = missing_acquisition_regions(region_inputs)
    if unknown_acquisition and not args.allow_unknown_lidar_acquisition:
        parser.error(
            "No authoritative LiDAR acquisition metadata for: "
            + ", ".join(unknown_acquisition)
            + ". Supply --lidar-acquisition-year with --lidar-acquisition-source, add a "
            "validated lidar_acquisition block to the registry region, or pass "
            "--allow-unknown-lidar-acquisition for a diagnostic build. The LAS header "
            "creation_date and filename/project year tokens are never used instead."
        )
    if unknown_acquisition:
        print(
            "WARNING: building without authoritative LiDAR acquisition metadata for "
            f"{', '.join(unknown_acquisition)}; SLIDO temporal filtering is disabled for "
            "those regions and no LiDAR/NAIP vintage gap can be claimed."
        )

    for region in region_inputs:
        if not region.laz_dir.exists():
            parser.error(f"LAZ directory does not exist for {region.region_id}: {region.laz_dir}")
        if not region.slido_path.exists():
            parser.error(f"SLIDO GeoJSON does not exist for {region.region_id}: {region.slido_path}")

    outdir = args.outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        if not args.overwrite:
            parser.error(f"Output directory is not empty: {outdir}. Use --overwrite deliberately.")
        shutil.rmtree(outdir)
    (outdir / "patches").mkdir(parents=True, exist_ok=True)

    metadata: list[TileMetadata] = []
    failed_tiles: list[dict[str, str]] = []
    region_by_id = {region.region_id: region for region in region_inputs}
    for region in region_inputs:
        discovered_paths = sorted([*region.laz_dir.glob("*.laz"), *region.laz_dir.glob("*.las")])
        paths = discovered_paths
        if region.lidar_project_pinned:
            paths = [
                path for path in discovered_paths
                if tile_matches_project(path, region.lidar_project_hint)
            ]
        if args.max_tiles:
            paths = paths[: args.max_tiles]
        if not paths:
            if discovered_paths and region.lidar_project_pinned:
                error = f"No tiles match pinned lidar_project {region.lidar_project_hint!r}"
            else:
                error = "No .laz or .las files found"
            failed_tiles.append({"region_id": region.region_id, "tile_name": "", "error": error})
        for path in paths:
            try:
                metadata.append(read_tile_metadata(path, region))
            except Exception as exc:
                failed_tiles.append({"region_id": region.region_id, "tile_name": path.name, "error": f"{type(exc).__name__}: {exc}"})

    split_result = assign_spatial_splits(
        [SpatialTile(item.tile_id, item.region_id, item.region_role, item.metric_footprint) for item in metadata],
        seed=args.seed,
        block_size_m=args.block_size_m,
        split_buffer_m=args.split_buffer_m,
        validation_fraction=args.validation_fraction,
    )
    for tile_id, reason in split_result.dropped.items():
        item = next(value for value in metadata if value.tile_id == tile_id)
        failed_tiles.append({"region_id": item.region_id, "tile_name": item.path.name, "error": f"dropped_split_buffer: {reason}"})

    rows: list[dict[str, Any]] = []
    tile_summaries: list[dict[str, Any]] = []
    processed_vintages: list[LidarVintage] = []
    feature_names: tuple[str, ...] | None = None
    retained = [item for item in metadata if item.tile_id in split_result.assignments]
    print(f"Processing {len(retained)} retained tile(s); split-buffer dropped {len(split_result.dropped)}")
    print(f"Tile split counts: {dict(Counter(split_result.assignments.values()))}")

    for tile_index, item in enumerate(retained, start=1):
        laz_path = item.path
        region = region_by_id[item.region_id]
        split = split_result.assignments[item.tile_id]
        print(f"[{tile_index}/{len(retained)}] {item.region_id} {laz_path.name} -> {split}")
        try:
            tile = read_laz_ground_dem(laz_path, cell_size=args.cell_size, max_cells=args.max_cells)
            if tile.ground_cell_fraction < args.min_ground_cell_fraction:
                raise RuntimeError(
                    f"ground-cell coverage {tile.ground_cell_fraction:.3f} below minimum {args.min_ground_cell_fraction:.3f}"
                )
            features, current_feature_names = build_feature_stack(tile.dem, cell_size=args.cell_size)
            if feature_names is None:
                feature_names = current_feature_names
            elif feature_names != current_feature_names:
                raise RuntimeError("Feature channel definitions changed between tiles")

            mask, intersecting_records = rasterize_tile_mask(
                item,
                region,
                tile,
                description="Landslide",
                positive_buffer_m=args.positive_ignore_buffer_m,
                all_touched=args.all_touched,
            )
            landslide_ids, ref_ids = intersecting_ids(intersecting_records)
            positive_pixels = mask == 1
            distance_to_positive = (
                distance_transform_edt(~positive_pixels) * args.cell_size
                if positive_pixels.any() else np.full(mask.shape, np.inf, dtype=np.float32)
            )
            positive_candidates: list[dict[str, Any]] = []
            negative_candidates: list[dict[str, Any]] = []
            skipped_low_coverage = skipped_trace = skipped_ignore_only = 0

            for row_offset, col_offset in iter_patch_windows(*tile.shape, patch_size=args.patch_size, stride=args.stride):
                row_slice = slice(row_offset, row_offset + args.patch_size)
                col_slice = slice(col_offset, col_offset + args.patch_size)
                patch_mask = mask[row_slice, col_slice]
                patch_valid = tile.valid_ground_mask[row_slice, col_slice]
                valid_fraction = float(patch_valid.mean())
                if valid_fraction < args.min_patch_ground_fraction:
                    skipped_low_coverage += 1
                    continue
                positive_fraction = float(np.mean(patch_mask == 1))
                ignore_fraction = float(np.mean(patch_mask == 255))
                category = classify_patch(
                    positive_fraction,
                    interior_threshold=args.interior_threshold,
                    boundary_threshold=args.boundary_threshold,
                )
                patch_slope_mean = float(features[1, row_slice, col_slice].mean())
                min_distance = float(distance_to_positive[row_slice, col_slice].min())
                candidate = {
                    "row_offset": row_offset,
                    "col_offset": col_offset,
                    "positive_fraction": positive_fraction,
                    "ignore_fraction": ignore_fraction,
                    "label_quality": label_quality(patch_mask),
                    "category": category,
                    "ground_fraction": valid_fraction,
                    "mean_slope_degrees": patch_slope_mean,
                    "distance_to_positive_m": min_distance,
                    "is_hard_negative": category == "negative" and patch_slope_mean >= args.hard_negative_slope,
                }
                if category == "negative":
                    if ignore_fraction > 0:
                        skipped_ignore_only += 1
                    elif min_distance >= args.negative_buffer_m:
                        negative_candidates.append(candidate)
                elif category == "positive_trace" and not args.include_trace_positives:
                    skipped_trace += 1
                else:
                    positive_candidates.append(candidate)

            selected_negatives = choose_negative_candidates(
                negative_candidates,
                positive_count=len(positive_candidates),
                ratio=args.negative_ratio,
                max_per_tile=args.max_negatives_per_tile,
                seed_text=f"{args.seed}:{item.tile_id}",
            )
            selected_candidates = sorted(
                positive_candidates + selected_negatives,
                key=lambda row: (row["row_offset"], row["col_offset"]),
            )
            tile_stem = safe_name(f"{item.region_id}_{laz_path.stem}")
            split_dir = outdir / "patches" / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for candidate in selected_candidates:
                row_offset, col_offset = int(candidate["row_offset"]), int(candidate["col_offset"])
                row_slice = slice(row_offset, row_offset + args.patch_size)
                col_slice = slice(col_offset, col_offset + args.patch_size)
                patch_id = f"{tile_stem}_r{row_offset:06d}_c{col_offset:06d}"
                patch_path = split_dir / f"{patch_id}.npz"
                np.savez_compressed(
                    patch_path,
                    features=features[:, row_slice, col_slice].astype(args.feature_dtype),
                    mask=mask[row_slice, col_slice].astype(np.uint8),
                )
                x_min = tile.transform.c + col_offset * tile.transform.a
                y_max = tile.transform.f + row_offset * tile.transform.e
                x_max = x_min + args.patch_size * tile.transform.a
                y_min = y_max + args.patch_size * tile.transform.e
                rows.append(
                    {
                        "patch_id": patch_id, "split": split, "category": candidate["category"],
                        "region_id": item.region_id, "region_role": item.region_role,
                        "lidar_project": item.lidar_project,
                        **item.vintage.as_row_fields(),
                        "label_quality": candidate["label_quality"],
                        "tile_name": laz_path.name, "tile_path": str(laz_path),
                        "patch_path": str(patch_path.relative_to(outdir)),
                        "row_offset": row_offset, "col_offset": col_offset,
                        "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
                        "crs": tile.crs.to_string(), "positive_fraction": candidate["positive_fraction"],
                        "ignore_fraction": candidate["ignore_fraction"], "ground_fraction": candidate["ground_fraction"],
                        "mean_slope_degrees": candidate["mean_slope_degrees"],
                        "distance_to_positive_m": candidate["distance_to_positive_m"],
                        "is_hard_negative": candidate["is_hard_negative"],
                        "landslide_ids_in_tile": landslide_ids, "slido_ref_ids_in_tile": ref_ids,
                        "qc_status": "", "qc_notes": "",
                    }
                )

            category_counts = Counter(row["category"] for row in selected_candidates)
            tile_summary = {
                "tile_name": laz_path.name, "region_id": item.region_id, "region_role": item.region_role,
                "split": split, "lidar_project": item.lidar_project,
                "lidar_year": item.lidar_year, "lidar_year_source": item.lidar_year_source,
                "lidar_vintage": item.vintage.as_summary_mapping(),
                "slido_temporal_filter_year": item.slido_lidar_year(),
                "crs": tile.crs.to_string(),
                "dem_shape": list(tile.shape), "ground_point_count": tile.ground_point_count,
                "ground_cell_fraction": tile.ground_cell_fraction,
                "intersecting_slido_polygons": len(intersecting_records),
                "temporally_excluded_polygons": sum(bool(record["temporal_excluded"]) for record in intersecting_records),
                "temporally_excluded_polygon_keys": sorted(
                    str(record["polygon_key"])
                    for record in intersecting_records
                    if record["temporal_excluded"]
                ),
                "mask_positive_fraction": float(np.mean(mask == 1)),
                "mask_ignore_fraction": float(np.mean(mask == 255)),
                "mask_pixels": int(mask.size), "ignore_pixels": int(np.sum(mask == 255)),
                "saved_patches": len(selected_candidates), "category_counts": dict(category_counts),
                "skipped_low_ground_coverage": skipped_low_coverage,
                "skipped_trace_positives": skipped_trace, "skipped_ignore_only": skipped_ignore_only,
                "eligible_negative_candidates": len(negative_candidates),
            }
            tile_summaries.append(tile_summary)
            processed_vintages.append(item.vintage)
            print(json.dumps(tile_summary, indent=2))
        except Exception as exc:
            failed_tiles.append({"region_id": item.region_id, "tile_name": laz_path.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED: {type(exc).__name__}: {exc}")

    write_csv(outdir / "patches.csv", rows, list(PATCH_FIELDS))
    write_csv(outdir / "failed_tiles.csv", failed_tiles, ["region_id", "tile_name", "error"])
    split_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        split_category_counts[str(row["split"])][str(row["category"])] += 1
    acquisition_conflict_failures = [
        item for item in failed_tiles if "acquisition" in item["error"].casefold()
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "regions": [region.region_id for region in region_inputs], "outdir": str(outdir),
        "processed_tiles": len(tile_summaries), "failed_or_dropped_tiles": len(failed_tiles),
        "split_buffer_dropped_tiles": len(split_result.dropped), "saved_patches": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "split_category_counts": {key: dict(value) for key, value in split_category_counts.items()},
        "region_summaries": _region_summaries(region_inputs, tile_summaries, rows, failed_tiles),
        "lidar_vintage_summary": {
            **summarize_vintages(processed_vintages),
            "acquisition_conflict_failures": acquisition_conflict_failures,
            "requested_region_acquisition": {
                region.region_id: (
                    None
                    if region.lidar_acquisition is None
                    else region.lidar_acquisition.as_summary_mapping()
                )
                for region in region_inputs
            },
            "unknown_acquisition_allowed": bool(args.allow_unknown_lidar_acquisition),
        },
        "feature_names": list(feature_names or []), "feature_dtype": args.feature_dtype,
        "cell_size": args.cell_size, "patch_size": args.patch_size, "stride": args.stride,
        "description_filter": "Landslide", "mask_codes": {"negative": 0, "positive": 1, "ignore": 255},
        "tile_summaries": tile_summaries,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (outdir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (outdir / "channels.json").write_text(json.dumps({"feature_names": list(feature_names or [])}, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} patches to {outdir}")
    print(f"Split counts: {summary['split_counts']}; failed/dropped tiles: {len(failed_tiles)}")
    incomplete_regions = [
        region_id
        for region_id, value in summary["region_summaries"].items()
        if value["status"] != "complete"
    ]
    if incomplete_regions:
        print(f"Incomplete regions: {', '.join(incomplete_regions)}")
    return 0 if rows and not incomplete_regions else 2


if __name__ == "__main__":
    raise SystemExit(main())
