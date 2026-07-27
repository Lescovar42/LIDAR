"""
download_hro.py
================
Discovers and downloads USGS High Resolution Orthoimagery (HRO) data 
that intersects with a given GeoJSON file's bounding box.

Usage:
    python download_hro.py --geojson ridgecrest_observations.geojson --download_dir ./hro_tiles
"""
import requests
import json
import argparse
import os

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"

def bbox_from_geojson(path):
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    xs, ys = [], []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        coords = geom["coordinates"]
        
        if geom["type"] == "Point":
            xs.append(coords[0])
            ys.append(coords[1])
        elif geom["type"] in ["Polygon", "MultiPolygon"]:
            rings = coords if geom["type"] == "MultiPolygon" else [coords]
            for poly in rings:
                for ring in poly:
                    for pt in ring:
                        xs.append(pt[0])
                        ys.append(pt[1])
    if not xs:
        raise ValueError("No valid geometries found in GeoJSON.")
    return min(xs), min(ys), max(xs), max(ys)

def discover_hro(bbox, dataset="High Resolution Orthoimagery (HRO)"):
    xmin, ymin, xmax, ymax = bbox
    page_size = 50
    all_items = []
    offset = 0

    print(f"Querying USGS TNM API for '{dataset}' data in bbox: {bbox}...")
    
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
        all_items.extend(items)
        
        if len(items) < page_size:
            break
        offset += page_size

    print(f"Found {len(all_items)} product(s) intersecting the bounding box.")
    return all_items

def download_file(url, out_path):
    print(f"Downloading {url} to {out_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    downloaded = 0
    chunk_size = 8192
    last_print_mb = 0
    
    with open(out_path, 'wb') as file:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            downloaded += size
            
            # Print progress every 10 MB
            downloaded_mb = downloaded / (1024 * 1024)
            if downloaded_mb - last_print_mb > 10:
                total_mb_str = f" / {total_size / (1024 * 1024):.1f} MB" if total_size else ""
                print(f"  ... downloaded {downloaded_mb:.1f} MB{total_mb_str}")
                last_print_mb = downloaded_mb
                
    print("  Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson", type=str, required=True, help="Path to geojson file with coordinates")
    parser.add_argument("--download_dir", type=str, default="./hro_tiles", help="Directory to save downloaded tiles")
    parser.add_argument("--dataset", type=str, default="High Resolution Orthoimagery (HRO)", help="USGS dataset name to query")
    parser.add_argument("--dry-run", action="store_true", help="Only discover tiles without downloading")
    args = parser.parse_args()

    # 1. Get bounding box from the geojson
    print(f"Reading bounds from {args.geojson}...")
    bbox = bbox_from_geojson(args.geojson)
    
    # 2. Discover tiles intersecting the bbox
    items = discover_hro(bbox, args.dataset)
    
    if args.dry_run:
        print("\nDry run completed. Tiles found:")
        for item in items:
            print(f" - {item['title']}: {item['downloadURL']}")
        exit(0)

    # 3. Download the tiles
    if items:
        os.makedirs(args.download_dir, exist_ok=True)
        for i, item in enumerate(items, 1):
            url = item.get("downloadURL")
            if url:
                filename = url.split("/")[-1]
                out_path = os.path.join(args.download_dir, filename)
                print(f"\nProcessing file {i}/{len(items)}: {filename}")
                if not os.path.exists(out_path):
                    download_file(url, out_path)
                else:
                    print(f"  File already exists, skipping...")
        print(f"\nSuccessfully downloaded {len(items)} tiles to {args.download_dir}")
    else:
        print(f"\nNo {args.dataset} tiles found for this area.")
