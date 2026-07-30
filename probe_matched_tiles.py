#!/usr/bin/env python3
"""Compare co-located LiDAR projects over identical intersection windows."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse
import numpy as np
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shapely_transform

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))
from terrain_utils import TerrainTile, read_laz_ground_dem
from probe_tiles import DEFAULT_CELL_SIZES, parse_project, probe_projects

def selection_filename(record: Mapping[str, Any]) -> str:
    url = str(record.get("downloadURL") or record.get("downloadUrl") or record.get("download_url") or "")
    name = Path(unquote(urlparse(url).path)).name
    if name:
        return name
    fallback = str(record.get("title") or record.get("tile_id") or "tile.laz")
    return fallback if Path(fallback).suffix else f"{fallback}.laz"

def crop_tile_to_wgs84_geometry(tile: TerrainTile, geometry: Mapping[str, Any], *, source_crs: str = "EPSG:4326") -> TerrainTile:
    source_geometry = shape(geometry)
    if source_geometry.is_empty:
        raise ValueError("probe intersection geometry is empty")
    transformer = Transformer.from_crs(CRS.from_user_input(source_crs), tile.crs, always_xy=True)
    projected = shapely_transform(transformer.transform, source_geometry)
    clipped = projected.intersection(box(*tile.bounds))
    if clipped.is_empty or clipped.area <= 0:
        raise ValueError(f"probe intersection does not overlap {tile.source_path.name}")
    window = from_bounds(*clipped.bounds, transform=tile.transform)
    row0 = max(0, int(math.floor(window.row_off)))
    col0 = max(0, int(math.floor(window.col_off)))
    row1 = min(tile.shape[0], int(math.ceil(window.row_off + window.height)))
    col1 = min(tile.shape[1], int(math.ceil(window.col_off + window.width)))
    if row1 <= row0 or col1 <= col0:
        raise ValueError(f"empty matched raster window for {tile.source_path.name}")
    transform = tile.transform * Affine.translation(col0, row0)
    dem = tile.dem[row0:row1, col0:col1].copy()
    source_valid = tile.valid_ground_mask[row0:row1, col0:col1]
    analysis_mask = geometry_mask([mapping(clipped)], out_shape=dem.shape, transform=transform, invert=True)
    analysis_cells = int(analysis_mask.sum())
    if analysis_cells == 0:
        raise ValueError(f"matched window has no raster cells for {tile.source_path.name}")
    valid = source_valid & analysis_mask
    return TerrainTile(dem=dem, transform=transform, crs=tile.crs, valid_ground_mask=valid,
                       ground_point_count=tile.ground_point_count,
                       ground_cell_fraction=float(valid.sum()/analysis_cells),
                       source_path=tile.source_path)

def load_matched_projects(selection_path: Path, project_directories: Mapping[str, Path]):
    raw = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("matched selection must be a project-to-list mapping")
    if set(raw) != set(project_directories):
        raise ValueError("selection project names must exactly match --project names")
    projects = {}
    windows = {}
    pair_members = {}
    for project, directory in project_directories.items():
        records = raw[project]
        if not isinstance(records, list):
            raise ValueError(f"selection group {project!r} is not a list")
        paths=[]
        for record in records:
            geometry = record.get("_probe_intersection_geometry") if isinstance(record, Mapping) else None
            pair_id = str(record.get("_probe_pair_id") or "") if isinstance(record, Mapping) else ""
            if not isinstance(geometry, Mapping) or not pair_id:
                raise ValueError("selection lacks matched-window metadata; rerun selection with --probe-overlap-metric smaller")
            path=(directory/selection_filename(record)).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            windows[path]={"geometry":dict(geometry), "crs":str(record.get("_probe_intersection_crs") or "EPSG:4326"), "pair_id":pair_id,
                           "iou":record.get("_probe_footprint_iou"), "smaller":record.get("_probe_smaller_overlap")}
            pair_members.setdefault(pair_id,set()).add(project)
            paths.append(path)
        projects[project]=paths
    required=set(project_directories)
    incomplete={k:sorted(required-v) for k,v in pair_members.items() if v != required}
    if incomplete:
        raise ValueError(f"incomplete matched pairs: {incomplete}")
    return projects, windows

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--project", action="append", type=parse_project, required=True, metavar="NAME=LAZ_DIR")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cell-size", action="append", type=float, dest="cell_sizes")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-patch-ground-fraction", type=float, default=0.5)
    parser.add_argument("--max-cells", type=int, default=80_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    if not args.selection.is_file(): parser.error(f"missing selection: {args.selection}")
    directories={}
    for name,directory in args.project:
        if name in directories: parser.error(f"duplicate project: {name}")
        if not directory.is_dir(): parser.error(f"missing project directory: {directory}")
        directories[name]=directory
    try:
        projects, windows=load_matched_projects(args.selection,directories)
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    cell_sizes=tuple(args.cell_sizes or DEFAULT_CELL_SIZES)
    def reader(path: Path, **kwargs: Any):
        metadata=windows.get(Path(path).resolve())
        if metadata is None: raise ValueError(f"no matched window for {path}")
        tile=read_laz_ground_dem(path, **kwargs)
        return crop_tile_to_wgs84_geometry(tile, metadata["geometry"], source_crs=metadata["crs"])
    result=probe_projects(projects, cell_sizes=cell_sizes, patch_size=args.patch_size, stride=args.stride,
                          min_patch_ground_fraction=args.min_patch_ground_fraction, max_cells=args.max_cells, reader=reader)
    result["parameters"]["analysis_extent"]="matched_project_intersection"
    result["parameters"]["selection"]=str(args.selection.resolve())
    result["matched_windows"]=[{"path":str(p),"pair_id":m["pair_id"],"footprint_iou":m["iou"],"smaller_footprint_overlap":m["smaller"]}
                               for p,m in sorted(windows.items(), key=lambda item:str(item[0]))]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    for row in result["project_cell_size_summary"]:
        print(f"{row['project']} {row['cell_size']:.1f}m: ground={row['ground_cell_fraction']['mean']:.3f}, patch survival={row['patch_ground_fraction']['surviving_fraction']:.3f}")
    print(f"Failures: {len(result['failures'])}")
    print(f"Wrote {args.output}")
    return 1 if result["failures"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
