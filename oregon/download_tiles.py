"""
Stage 1e: Download the actual LAZ point cloud files for a selected tile
subset (output of 04_select_tile_subset.py).

RUN THIS LOCALLY -- needs open internet access to rockyweb.usgs.gov, which
is not on Claude's sandbox allowlist.

Supports resuming: if a file already exists and matches the expected size,
it's skipped rather than re-downloaded. This matters because even a
landslide-only subset could be several GB, and you don't want a network
hiccup partway through to mean starting over.

Usage:
    python 05_download_tile_subset.py --subset tile_subset_feasibility.json --outdir ./lidar_tiles
"""
import json
import argparse
import os
import requests
import time
import sys
from pathlib import Path


def download_tiles(subset_path, outdir):
    with open(subset_path) as fh:
        tiles = json.load(fh)

    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(tiles)} tile(s) to {outdir}\n")
    total_bytes = sum(t.get("sizeInBytes", 0) for t in tiles)
    print(f"Total expected size: {total_bytes / 1e9:.2f} GB\n")

    downloaded = 0
    skipped = 0
    failed = []

    for i, tile in enumerate(tiles, 1):
        url = tile.get("downloadURL")
        if not url:
            print(f"  [{i}/{len(tiles)}] SKIP: no downloadURL for {tile.get('title')}")
            continue

        filename = url.split("/")[-1]
        out_path = os.path.join(outdir, filename)
        expected_size = tile.get("sizeInBytes")

        # Resume support: skip if file exists and size matches
        if os.path.exists(out_path):
            actual_size = os.path.getsize(out_path)
            if expected_size:
                if abs(actual_size - expected_size) < 1024:  # within 1KB tolerance
                    print(f"  [{i}/{len(tiles)}] SKIP (already downloaded): {filename}")
                    skipped += 1
                    continue
                else:
                    print(f"  [{i}/{len(tiles)}] Existing file size mismatch ({actual_size} vs {expected_size}), re-downloading: {filename}")
            elif actual_size > 0:
                print(f"  [{i}/{len(tiles)}] SKIP (already exists, unknown expected size): {filename}")
                skipped += 1
                continue

        try:
            print(f"  [{i}/{len(tiles)}] Downloading: {filename} ({(expected_size or 0)/1e6:.1f} MB)")
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()

            total_length = expected_size or int(resp.headers.get('content-length', 0))
            downloaded_bytes = 0
            start_time = time.time()
            last_print_time = start_time

            with open(out_path, "wb") as out_fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        out_fh.write(chunk)
                        downloaded_bytes += len(chunk)

                        current_time = time.time()
                        if current_time - last_print_time > 0.5:
                            elapsed = current_time - start_time
                            mbps = (downloaded_bytes / 1e6) / elapsed if elapsed > 0 else 0
                            
                            if total_length > 0:
                                percent = (downloaded_bytes / total_length) * 100
                                sys.stdout.write(f"\r    Progress: {percent:.1f}% | Speed: {mbps:.2f} MB/s | Downloaded: {downloaded_bytes/1e6:.1f}/{total_length/1e6:.1f} MB")
                            else:
                                sys.stdout.write(f"\r    Speed: {mbps:.2f} MB/s | Downloaded: {downloaded_bytes/1e6:.1f} MB")
                            sys.stdout.flush()
                            last_print_time = current_time
            print() # New line after download finishes

            downloaded += 1

        except Exception as e:
            print(f"\n    FAILED: {e}")
            failed.append((filename, str(e)))
            # remove partial file so it doesn't get mistaken for complete
            if os.path.exists(out_path):
                os.remove(out_path)

    print(f"\n--- Summary ---")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (already present): {skipped}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("\nFailed tiles:")
        for filename, error in failed:
            print(f"  {filename}: {error}")
        print("\nRe-run this script to retry failed downloads (successful ones will be skipped).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, help="Path to tile subset json (from 04_select_tile_subset.py)")
    parser.add_argument("--outdir", default="./lidar_tiles", help="Output directory for LAZ files")
    args = parser.parse_args()

    download_tiles(args.subset, args.outdir)
