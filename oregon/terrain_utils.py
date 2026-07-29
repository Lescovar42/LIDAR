#!/usr/bin/env python3
"""Shared LiDAR terrain and SLIDO rasterization utilities.

This module is intentionally independent from the model-training code so LAZ
processing, label generation, QC, and training all use the same definitions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform

from build_manifest import extract_landslide_date
from slido_utils import load_deposits, normalize_confidence

FEATURE_NAMES = (
    "local_relief",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "multidirectional_hillshade",
    "tri",
)


@dataclass
class TerrainTile:
    dem: np.ndarray
    transform: Affine
    crs: CRS
    valid_ground_mask: np.ndarray
    ground_point_count: int
    ground_cell_fraction: float
    source_path: Path

    @property
    def shape(self) -> tuple[int, int]:
        return self.dem.shape

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        height, width = self.dem.shape
        xmin = self.transform.c
        ymax = self.transform.f
        xmax = xmin + width * self.transform.a
        ymin = ymax + height * self.transform.e
        return xmin, ymin, xmax, ymax


def read_laz_ground_dem(
    laz_path: str | Path,
    *,
    cell_size: float = 1.0,
    smoothing_size: int = 3,
    max_cells: int = 80_000_000,
) -> TerrainTile:
    """Read classified ground points from LAZ and grid a bare-earth DEM."""
    try:
        import laspy
    except ImportError as exc:
        raise RuntimeError("Install LAZ support with: pip install 'laspy[lazrs]'") from exc

    path = Path(laz_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    las = laspy.read(path)
    crs = las.header.parse_crs()
    if crs is None:
        raise RuntimeError(f"No CRS found in LAZ header: {path.name}")
    crs = CRS.from_user_input(crs)

    classification = np.asarray(las.classification)
    ground = classification == 2
    ground_count = int(ground.sum())
    if ground_count == 0:
        raise RuntimeError(f"No ASPRS class-2 ground points in {path.name}")

    x = np.asarray(las.x[ground], dtype=np.float64)
    y = np.asarray(las.y[ground], dtype=np.float64)
    z = np.asarray(las.z[ground], dtype=np.float64)

    # Use the complete point-cloud extent, not only class-2 points. Otherwise
    # ground-free margins disappear from coverage, void, and fragmentation QC.
    header_min = np.asarray(las.header.mins, dtype=np.float64)
    header_max = np.asarray(las.header.maxs, dtype=np.float64)
    xmin = float(np.floor(header_min[0] / cell_size) * cell_size)
    xmax = float(np.ceil(header_max[0] / cell_size) * cell_size)
    ymin = float(np.floor(header_min[1] / cell_size) * cell_size)
    ymax = float(np.ceil(header_max[1] / cell_size) * cell_size)

    width = max(1, int(np.ceil((xmax - xmin) / cell_size)))
    height = max(1, int(np.ceil((ymax - ymin) / cell_size)))
    cells = width * height
    if cells > max_cells:
        raise RuntimeError(
            f"DEM would contain {cells:,} cells ({height}x{width}). "
            "Increase --cell-size or --max-cells deliberately."
        )

    col = np.clip(((x - xmin) / cell_size).astype(np.int64), 0, width - 1)
    row = np.clip(((ymax - y) / cell_size).astype(np.int64), 0, height - 1)

    z_sum = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.uint32)
    np.add.at(z_sum, (row, col), z)
    np.add.at(count, (row, col), 1)

    valid = count > 0
    dem = np.full((height, width), np.nan, dtype=np.float64)
    dem[valid] = z_sum[valid] / count[valid]
    ground_cell_fraction = float(valid.mean())

    if not valid.all():
        # Nearest-neighbour fill is used only to make derivative calculation
        # possible. The original coverage mask is preserved for QC/filtering.
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        dem = dem[tuple(nearest)]

    if smoothing_size > 1:
        dem = uniform_filter(dem, size=smoothing_size, mode="nearest")

    transform = Affine(cell_size, 0.0, xmin, 0.0, -cell_size, ymax)
    return TerrainTile(
        dem=dem.astype(np.float32),
        transform=transform,
        crs=crs,
        valid_ground_mask=valid,
        ground_point_count=ground_count,
        ground_cell_fraction=ground_cell_fraction,
        source_path=path,
    )


def compute_slope(dem: np.ndarray, cell_size: float) -> np.ndarray:
    dy, dx = np.gradient(dem.astype(np.float64), cell_size)
    return np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)


def compute_aspect_components(dem: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    dy, dx = np.gradient(dem.astype(np.float64), cell_size)
    aspect = np.arctan2(-dx, dy)
    return np.sin(aspect).astype(np.float32), np.cos(aspect).astype(np.float32)


def compute_curvature(dem: np.ndarray, cell_size: float) -> np.ndarray:
    dy, dx = np.gradient(dem.astype(np.float64), cell_size)
    dyy, _ = np.gradient(dy, cell_size)
    _, dxx = np.gradient(dx, cell_size)
    return (dxx + dyy).astype(np.float32)


def compute_hillshade(
    dem: np.ndarray,
    cell_size: float,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    dy, dx = np.gradient(dem.astype(np.float64), cell_size)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    azimuth_radians = np.radians(azimuth)
    altitude_radians = np.radians(altitude)
    shaded = (
        np.sin(altitude_radians) * np.cos(slope)
        + np.cos(altitude_radians) * np.sin(slope) * np.cos(azimuth_radians - aspect)
    )
    return np.clip((shaded + 1.0) * 127.5, 0.0, 255.0).astype(np.float32)


def compute_multidirectional_hillshade(dem: np.ndarray, cell_size: float) -> np.ndarray:
    hillshades = [compute_hillshade(dem, cell_size, azimuth=az) for az in (45, 135, 225, 315)]
    return np.mean(hillshades, axis=0, dtype=np.float32)


def compute_tri(dem: np.ndarray) -> np.ndarray:
    """Vectorized 3x3 terrain ruggedness index with nearest-edge padding."""
    values = np.asarray(dem, dtype=np.float64)
    local_mean = uniform_filter(values, size=3, mode="nearest")
    local_square_mean = uniform_filter(values * values, size=3, mode="nearest")
    mean_squared_difference = local_square_mean - 2.0 * values * local_mean + values * values
    # Roundoff can make mathematically non-negative values slightly negative.
    return np.sqrt(np.maximum(mean_squared_difference, 0.0)).astype(np.float32)


def build_feature_stack(dem: np.ndarray, *, cell_size: float) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build model inputs without using absolute elevation as a shortcut."""
    broad_sigma = max(1.0, 30.0 / cell_size)
    broad_surface = gaussian_filter(dem, sigma=broad_sigma, mode="nearest")
    local_relief = (dem - broad_surface).astype(np.float32)
    slope = compute_slope(dem, cell_size)
    aspect_sin, aspect_cos = compute_aspect_components(dem, cell_size)
    curvature = compute_curvature(dem, cell_size)
    hillshade = compute_multidirectional_hillshade(dem, cell_size)
    tri = compute_tri(dem)

    stack = np.stack(
        [local_relief, slope, aspect_sin, aspect_cos, curvature, hillshade, tri],
        axis=0,
    ).astype(np.float32)
    return stack, FEATURE_NAMES


def _property_value(properties: Mapping[str, Any], name: str) -> Any:
    wanted = name.casefold()
    for key, value in properties.items():
        if str(key).casefold() == wanted:
            return value
    return None


def label_quality(mask: np.ndarray) -> str:
    """Describe the scoreability of a three-state label array."""
    has_positive = bool(np.any(mask == 1))
    has_ignore = bool(np.any(mask == 255))
    if has_positive and has_ignore:
        return "accepted_with_ignore"
    if has_positive:
        return "accepted"
    if has_ignore:
        return "ignore_only"
    return "negative"


def rasterize_slido_mask(
    slido_geojson: str | Path,
    tile: TerrainTile,
    *,
    description: str = "Landslide",
    lidar_year: int | None = None,
    positive_buffer_m: float = 0.0,
    all_touched: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Create a three-state (negative/positive/ignore) SLIDO mask.

    High and moderate confidence polygons are positive unless their known event
    year postdates the LiDAR year. Low/unknown confidence and post-LiDAR
    polygons are ignore. Exclusion polygons override positives when geometries
    overlap; the configurable ring only replaces background, so accepted
    positive interiors remain positive.
    """
    if positive_buffer_m < 0:
        raise ValueError("positive_buffer_m must be non-negative")
    deposits = load_deposits(slido_geojson, description=description)
    transformer = Transformer.from_crs("EPSG:4326", tile.crs, always_xy=True)
    tile_polygon = box(*tile.bounds)

    positive_shapes: list[tuple[dict[str, object], int]] = []
    excluded_shapes: list[tuple[dict[str, object], int]] = []
    records: list[dict[str, object]] = []
    for geometry, properties in deposits:
        projected = shapely_transform(transformer.transform, geometry)
        if projected.is_empty or not projected.intersects(tile_polygon):
            continue
        clipped = projected.intersection(tile_polygon)
        if clipped.is_empty:
            continue

        confidence = normalize_confidence(
            _property_value(properties, "confidence_class")
            or _property_value(properties, "CONFIDENCE")
        )
        event_date, event_date_source = extract_landslide_date(properties)
        event_year = event_date.year if event_date is not None else None
        temporal_excluded = bool(
            lidar_year is not None and event_year is not None and event_year > lidar_year
        )
        confidence_excluded = confidence not in {"high", "moderate"}
        disposition = "ignore" if confidence_excluded or temporal_excluded else "positive"
        target = excluded_shapes if disposition == "ignore" else positive_shapes
        target.append((mapping(clipped), 1))
        landslide_id = _property_value(properties, "UNIQUE_ID") or _property_value(properties, "OBJECTID") or ""
        polygon_key = str(landslide_id) if landslide_id else hashlib.sha1(geometry.wkb).hexdigest()
        records.append(
            {
                "landslide_id": landslide_id,
                "polygon_key": polygon_key,
                "ref_id_cod": _property_value(properties, "REF_ID_COD") or "",
                "confidence": _property_value(properties, "CONFIDENCE") or "",
                "confidence_class": confidence,
                "event_year": event_year,
                "event_year_source": event_date_source or "",
                "temporal_excluded": temporal_excluded,
                "disposition": disposition,
                "move_code": _property_value(properties, "MOVE_CODE") or "",
                "description": _property_value(properties, "DESCRIPTION") or "",
            }
        )

    mask = np.zeros(tile.dem.shape, dtype=np.uint8)
    if positive_shapes:
        positives = rasterize(
            shapes=positive_shapes,
            out_shape=tile.dem.shape,
            transform=tile.transform,
            fill=0,
            default_value=1,
            all_touched=all_touched,
            dtype="uint8",
        ).astype(bool)
        mask[positives] = 1
        if positive_buffer_m > 0:
            x_size = abs(float(tile.transform.a))
            y_size = abs(float(tile.transform.e))
            distances = distance_transform_edt(~positives, sampling=(y_size, x_size))
            ring = (~positives) & (distances <= positive_buffer_m)
            mask[ring] = 255

    if excluded_shapes:
        excluded = rasterize(
            shapes=excluded_shapes,
            out_shape=tile.dem.shape,
            transform=tile.transform,
            fill=0,
            default_value=1,
            all_touched=all_touched,
            dtype="uint8",
        ).astype(bool)
        mask[excluded] = 255
    return mask, records


def iter_patch_windows(
    height: int,
    width: int,
    *,
    patch_size: int,
    stride: int,
) -> Iterator[tuple[int, int]]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if height < patch_size or width < patch_size:
        return
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            yield row, col


def classify_patch(
    positive_fraction: float,
    *,
    interior_threshold: float = 0.10,
    boundary_threshold: float = 0.01,
) -> str:
    if positive_fraction >= interior_threshold:
        return "positive_interior"
    if positive_fraction >= boundary_threshold:
        return "positive_boundary"
    if positive_fraction > 0:
        return "positive_trace"
    return "negative"


def intersecting_ids(records: Sequence[dict[str, object]]) -> tuple[str, str]:
    landslide_ids = sorted({str(record.get("landslide_id", "")) for record in records if record.get("landslide_id")})
    ref_ids = sorted({str(record.get("ref_id_cod", "")) for record in records if record.get("ref_id_cod")})
    return ";".join(landslide_ids), ";".join(ref_ids)
