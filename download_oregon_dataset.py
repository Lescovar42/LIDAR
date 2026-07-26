"""
Orchestrator: download_oregon_dataset.py
Downloads LiDAR tiles for a given SLIDO landslide GeoJSON.
Filters out LiDAR tiles that were acquired *before* the landslide occurred.
"""
import argparse
import json
import datetime
from pathlib import Path

# Import existing pipeline stages
from importlib.machinery import SourceFileLoader

discover_module = SourceFileLoader("discover", "03_discover_3dep_tiles.py").load_module()
select_module = SourceFileLoader("select", "04_select_tile_subset.py").load_module()
download_module = SourceFileLoader("download", "05_download_tile_subset.py").load_module()

def get_year_from_string(date_str):
    if not date_str:
        return None
    try:
        # e.g., '2019-10-14'
        return int(str(date_str)[:4])
    except ValueError:
        return None

def filter_tiles_by_date(deposits, tiles):
    """
    Select tiles that intersect deposits AND ensure the LiDAR was acquired AFTER the landslide.
    """
    try:
        from shapely.geometry import box
    except ImportError:
        print("This script needs shapely: pip install shapely")
        raise

    selected = []
    for tile in tiles:
        bb = tile["boundingBox"]
        tile_box = box(bb["minX"], bb["minY"], bb["maxX"], bb["maxY"])
        
        # Get tile year
        tile_date = tile.get("publicationDate") or tile.get("dateCreated")
        tile_year = get_year_from_string(tile_date)

        valid_deposits = []
        for geom, props in deposits:
            if tile_box.intersects(geom):
                # Date check
                landslide_year = props.get("YEAR")
                # Clean up year (e.g. some might be '2010', some might be None)
                if landslide_year is not None:
                    try:
                        ls_year = int(landslide_year)
                        if tile_year and tile_year < ls_year:
                            print(f"Skipping tile {tile.get('title')} (year {tile_year}) for landslide {props.get('UNIQUE_ID')} (year {ls_year}) - LiDAR is older than landslide.")
                            continue
                    except ValueError:
                        pass # Ignore if YEAR isn't parseable as int (e.g., 'Pre-Historic')
                
                valid_deposits.append(props.get("MOVE_CODE", "unknown"))

        if valid_deposits:
            selected.append({
                **tile,
                "_landslide_count": len(valid_deposits),
                "_move_codes": valid_deposits,
            })

    return selected

def main():
    parser = argparse.ArgumentParser(description="Oregon LiDAR Downloader")
    parser.add_argument("--slido_geojson", required=True, help="Path to SLIDO deposits GeoJSON")
    parser.add_argument("--outdir", default="./oregon_lidar", help="Output directory for LAZ files")
    args = parser.parse_args()

    # 1. Discover tiles based on bbox of the geojson
    print(f"\n--- Stage 1: Discovering LiDAR for {args.slido_geojson} ---")
    bbox = discover_module.bbox_from_geojson(args.slido_geojson)
    tiles = discover_module.discover_lidar(bbox)

    # 2. Select tiles intersecting deposits and apply date filter
    print(f"\n--- Stage 2: Filtering tiles based on location and time ---")
    deposits = select_module.load_deposits(args.slido_geojson)
    selected_tiles = filter_tiles_by_date(deposits, tiles)
    
    subset_json = "oregon_selected_tiles.json"
    with open(subset_json, "w") as fh:
        json.dump(selected_tiles, fh, indent=2)
    print(f"Saved {len(selected_tiles)} valid tiles to {subset_json}")

    # 3. Download the tiles
    if selected_tiles:
        print(f"\n--- Stage 3: Downloading {len(selected_tiles)} tiles ---")
        download_module.download_tiles(subset_json, args.outdir)
    else:
        print("\nNo valid LiDAR tiles found for these deposits after filtering.")

if __name__ == "__main__":
    main()
