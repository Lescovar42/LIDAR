"""
download_hro_m2m.py
===================
Discovers and downloads USGS imagery from EarthExplorer 
using the Machine-to-Machine (M2M) API.

Requires EROS credentials (username and password or an application token).

Usage:
    python download_hro_m2m.py --geojson ridgecrest_observations.geojson
"""
import requests
import json
import argparse
import os
import sys
import getpass

M2M_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"

def send_request(endpoint, payload, api_key=None):
    url = f"{M2M_URL}/{endpoint}"
    headers = {"X-Auth-Token": api_key} if api_key else {}
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errorCode"):
        raise Exception(f"M2M API Error ({data['errorCode']}): {data['errorMessage']}")
    return data.get("data")

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

def download_file(url, out_path):
    print(f"Downloading {out_path}...")
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
            downloaded_mb = downloaded / (1024 * 1024)
            # Print every 25MB
            if downloaded_mb - last_print_mb > 25:
                total_mb_str = f" / {total_size / (1024 * 1024):.1f} MB" if total_size else ""
                print(f"  ... {downloaded_mb:.1f} MB{total_mb_str}")
                last_print_mb = downloaded_mb
    print("  Done!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson", type=str, required=True, help="GeoJSON bounding box")
    parser.add_argument("--dataset", type=str, default="hro", help="M2M datasetName (e.g. 'hro', 'naip')")
    parser.add_argument("--download_dir", type=str, default="./hro_tiles", help="Output directory")
    args = parser.parse_args()

    print("USGS EarthExplorer (M2M) Authentication")
    token = os.environ.get("USGS_TOKEN")
    
    if not token:
        print("USGS M2M API now requires an Application Token (passwords are no longer supported).")
        print("To generate one:")
        print(" 1. Go to https://ers.cr.usgs.gov/profile/access")
        print(" 2. Generate a new 'Application Token'")
        token = input("Enter EROS Application Token (it will be visible as you type): ")
    
    # login-token endpoint requires username as well
    username = os.environ.get("USGS_USERNAME")
    if not username:
        username = input("Enter EROS Username: ")

    token = token.strip()
    username = username.strip()
    
    login_payload = {"username": username, "token": token}

    print("Logging in to USGS M2M API...")
    try:
        api_key = send_request("login-token", login_payload)
    except Exception as e:
        print(f"Login failed: {e}")
        return
        
    print("Login successful.")

    try:
        # 1. Extract bbox
        xmin, ymin, xmax, ymax = bbox_from_geojson(args.geojson)
        print(f"BBox from GeoJSON: ({xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f})")

        # 2. Search Scenes
        print(f"\nSearching for scenes in dataset '{args.dataset}'...")
        search_payload = {
            "datasetName": args.dataset,
            "maxResults": 100,
            "sceneFilter": {
                "spatialFilter": {
                    "filterType": "mbr",
                    "lowerLeft": {"latitude": ymin, "longitude": xmin},
                    "upperRight": {"latitude": ymax, "longitude": xmax}
                }
            }
        }
        
        try:
            search_results = send_request("scene-search", search_payload, api_key)
        except Exception as e:
            if "DATASET_INVALID" in str(e):
                print(f"\n[ERROR] '{args.dataset}' is not a valid dataset name in EarthExplorer.")
                print("Fetching valid dataset names for you...")
                
                # Fetch available datasets to help the user
                datasets_response = send_request("dataset-search", {}, api_key)
                valid_datasets = []
                for ds in datasets_response:
                    alias = ds.get("datasetAlias", "").lower()
                    name = ds.get("datasetName", "").lower()
                    # Look for ortho, hro, naip, or imagery
                    if any(kw in alias or kw in name for kw in ["ortho", "hro", "naip", "aerial"]):
                        valid_datasets.append(ds.get("datasetAlias"))
                
                print("\nHere are some valid dataset names you can use with the --dataset flag:")
                for vd in valid_datasets:
                    print(f"  --dataset \"{vd}\"")
                print("\n(Run the script again using one of the names above!)")
                return
            else:
                raise e

        scenes = search_results.get("results", [])
        print(f"Found {len(scenes)} matching scenes.")

        if not scenes:
            print(f"No scenes found. Double check if '{args.dataset}' is the correct datasetName on EarthExplorer.")
            return

        # 3. Get Download Options
        entity_ids = [s["entityId"] for s in scenes]
        options_payload = {
            "datasetName": args.dataset,
            "entityIds": entity_ids
        }
        
        print("\nChecking download options (this validates which files you have permission to download)...")
        
        try:
            download_options = send_request("download-options", options_payload, api_key)
            
            # Filter for products that are available for immediate download or require a processing request
            downloads_to_request = []
            for product in download_options:
                if product.get("available"):
                    downloads_to_request.append({
                        "entityId": product["entityId"],
                        "productId": product["id"]
                    })

            if not downloads_to_request:
                print("No downloadable products available for these scenes.")
                return
                
            print(f"Requesting download URLs for {len(downloads_to_request)} products...")
            
            # 4. Request Downloads
            request_payload = {
                "downloads": downloads_to_request,
                "label": "m2m_hro_download"
            }
            
            request_results = send_request("download-request", request_payload, api_key)
            
            available_downloads = request_results.get("availableDownloads", [])
            preparing_downloads = request_results.get("preparingDownloads", [])
            
            if preparing_downloads:
                print(f"\nWARNING: {len(preparing_downloads)} files are still processing on USGS servers (e.g. tape retrieval).")
                print("You will receive an email from USGS when they are ready to download from EarthExplorer.")
            
            if not available_downloads:
                print("\nNo immediate downloads available right now.")
                return
                
            # 5. Download the actual files
            os.makedirs(args.download_dir, exist_ok=True)
            print(f"\nStarting {len(available_downloads)} downloads...")
            for i, dl in enumerate(available_downloads, 1):
                url = dl.get("url")
                filename = url.split("/")[-1] if "/" in url else f"scene_{dl['entityId']}.zip"
                filename = filename.split("?")[0]  # Remove query parameters from filename
                out_path = os.path.join(args.download_dir, filename)
                
                print(f"\n[{i}/{len(available_downloads)}] {filename}")
                if not os.path.exists(out_path):
                    download_file(url, out_path)
                else:
                    print(f"  File already exists locally, skipping...")

        except Exception as e:
            if "403" in str(e):
                print("\n[ERROR] 403 Forbidden: Your account does not have API download permissions!")
                print("Even with an Application Token, USGS requires you to manually request the 'Machine' role to download files programmatically.")
                print("To fix this for the future:")
                print("  1. Go to https://ers.cr.usgs.gov/profile/access")
                print("  2. Request 'MACHINE' access (it may take a few days for them to approve it).")
                
                print("\n--- FALLBACK SOLUTION ---")
                scene_file = "found_scenes.txt"
                with open(scene_file, "w") as f:
                    for eid in entity_ids:
                        f.write(f"{eid}\n")
                print(f"I have saved the {len(entity_ids)} scene IDs to a file called '{scene_file}' in this directory.")
                print("You can upload this text file directly into EarthExplorer or the USGS Bulk Download Application (BDA) to download the images immediately without needing API access!")
            else:
                raise e

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
    finally:
        print("\nLogging out of M2M API...")
        send_request("logout", {}, api_key)
        print("Logged out successfully.")

if __name__ == "__main__":
    main()
