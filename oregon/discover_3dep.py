"""
Stage 1c: Discover available 3DEP lidar point cloud tiles for the study area.

RUN THIS LOCALLY (same network restriction as the other scripts in this
pipeline -- tnmaccess.nationalmap.gov is not on Claude's sandbox allowlist).

This uses the official TNM Access API (no auth required), confirmed via a
public reference implementation (github.com/wangsen992/usgs-lidar-downloader).

IMPORTANT DECISION POINT, not yet resolved -- read before running downstream steps:
DOGAMI's SLIDO landslide inventory was mapped using DOGAMI / Oregon Lidar
Consortium bare-earth DEMs, which may differ in acquisition date and/or
processing from whatever 3DEP project tile covers the same footentity.
This script will print the 3DEP project name + acquisition date for your
area. Compare that against the source citation in the SLIDO Deposits
attribute table (there should be a source/citation field -- check
slido_deposits_schema.json from script 01) to see if they're the same
underlying survey or a different one. If different, that's a limitation
worth stating explicitly in your methods section (label geometry may not
perfectly align with terrain derivatives from a different-vintage DEM).

This script only DISCOVERS tiles -- it deliberately does not auto-download,
since:
  1. Tiles can be large (50-500MB each)
  2. You should look at acquisition dates/quality level before committing
  3. The actual point-cloud -> bare-earth DEM step is non-trivial (ground
     classification + gridding) and is better done via PDAL against the
     cloud-native Entwine Point Tile (EPT) data on s3://usgs-lidar, using
     OpenTopography's existing Jupyter notebook workflows as a base, rather
     than reimplemented from raw LAZ tiles here. See:
     https://opentopography.org/blog/new-collection-jupyter-notebooks-...

Usage:
    python 03_discover_3dep_tiles.py --geojson oregon_city_study_boundary.geojson
    python 03_discover_3dep_tiles.py --bbox -122.68 45.32 -122.54 45.43
"""
import requests
import json
import argparse

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"


def bbox_from_geojson(path):
    with open(path) as fh:
        data = json.load(fh)
    xs, ys = [], []
    for feat in data["features"]:
        geom = feat["geometry"]
        coords = geom["coordinates"]
        # handle Polygon and MultiPolygon
        rings = coords if geom["type"] == "MultiPolygon" else [coords]
        for poly in rings:
            for ring in poly:
                for pt in ring:
                    xs.append(pt[0])
                    ys.append(pt[1])
    return min(xs), min(ys), max(xs), max(ys)


def discover_lidar(bbox, dataset="Lidar Point Cloud (LPC)"):
    """
    Paginates through ALL matching tiles, not just the first page.

    IMPORTANT, confirmed via USGS TNM API docs: max + offset combined is
    capped at 50 by the API itself (not just a default -- a hard ceiling).
    This was NOT correctly handled in the original version of this script,
    which requested max=100 with no offset loop and would have silently
    truncated results on any area with >100 (or even >50, per the docs)
    matching tiles. Fixed here to page in chunks of 50 and check the
    response's 'total' field rather than assuming one page is everything.
    """
    xmin, ymin, xmax, ymax = bbox
    page_size = 50  # documented hard cap for max+offset combined
    all_items = []
    offset = 0
    reported_total = None

    while True:
        params = {
            "datasets": dataset,
            "bbox": f"{xmin},{ymin},{xmax},{ymax}",
            "outputFormat": "JSON",
            "max": page_size,
            "offset": offset,
        }
        resp = requests.get(TNM_API, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        reported_total = data.get("total", reported_total)
        all_items.extend(items)
        print(f"  fetched {len(items)} (offset={offset}, running total={len(all_items)}"
              f"{f', API reports total={reported_total}' if reported_total is not None else ''})")

        if len(items) < page_size:
            break
        offset += page_size

        # safety valve -- bail rather than loop forever if something's wrong
        if offset > 5000:
            print("  WARNING: stopped after 5000 offset without exhausting results. "
                  "Something may be wrong -- check the API response manually.")
            break

    if reported_total is not None and len(all_items) != reported_total:
        print(f"\nWARNING: fetched {len(all_items)} items but API reported total={reported_total}. "
              f"These should match -- if they don't, treat this result as incomplete "
              f"rather than authoritative.")

    print(f"\nFound {len(all_items)} lidar product(s) total, intersecting bbox {bbox}\n")
    items = all_items

    projects = {}
    for item in items:
        title = item.get("title", "unknown")
        source_id = item.get("sourceId")
        date = item.get("publicationDate") or item.get("dateCreated") or "unknown date"
        size_mb = item.get("sizeInBytes", 0) / 1e6 if item.get("sizeInBytes") else None
        download_url = item.get("downloadURL")

        # Group by sourceId (the actual project identifier) when present.
        # Only fall back to a title-derived key when sourceId is missing,
        # stripping a trailing numeric tile ID rather than using the whole
        # (per-tile-unique) title -- otherwise every tile becomes its own
        # "project" of size 1, which defeats the point of grouping.
        # (Bug found via synthetic-data testing while building the Colab
        # notebook version of this script -- the original hyphen-split
        # fallback put every tile in its own group when titles had no
        # hyphen, which silently produced misleading output.)
        if source_id:
            key = source_id
        else:
            parts = title.rsplit(" ", 1)
            key = parts[0] if len(parts) == 2 and parts[1].isdigit() else title

        projects.setdefault(key, []).append({
            "title": title,
            "date": date,
            "size_mb": size_mb,
            "download_url": download_url,
        })

    print(f"Grouped into {len(projects)} project(s):\n")
    for proj_name, tiles in projects.items():
        print(f"  Project: {proj_name}  ({len(tiles)} tiles)")
        dates = set(t["date"] for t in tiles)
        print(f"    Dates present: {dates}")
        if tiles:
            print(f"    Example tile: {tiles[0]['title']}")
            print(f"    Example URL:  {tiles[0]['download_url']}")
        print()

    with open("3dep_tile_discovery.json", "w") as fh:
        json.dump(items, fh, indent=2)
    print("Full results saved to 3dep_tile_discovery.json")
    print("\nNEXT STEP: cross-check project date(s) above against the SLIDO")
    print("Deposits source/citation field before treating this as your DEM source.")

    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson", type=str, help="Path to study boundary geojson (from script 02)")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("xmin", "ymin", "xmax", "ymax"))
    args = parser.parse_args()

    if args.geojson:
        bbox = bbox_from_geojson(args.geojson)
    elif args.bbox:
        bbox = tuple(args.bbox)
    else:
        print("Provide either --geojson or --bbox")
        exit(1)

    discover_lidar(bbox)
