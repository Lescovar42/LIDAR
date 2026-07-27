"""
Stage 1d: Select a MINIMAL subset of OR_OLCMetro_2019_A19 tiles for a
feasibility check, rather than downloading all 560 (58 GB).

Strategy: intersect actual SLIDO Deposits landslide polygon geometries
against each tile's bounding box. Only tiles that actually contain (or
touch) a landslide polygon are kept -- this gives you real positive labels
to test the lidar-derivative approach against, without downloading tiles
that cover zero landslides.

IMPORTANT CAVEAT, worth thinking about before you commit to this subset:
tiles with ZERO landslide polygons are exactly where your model's negative
(non-landslide) samples would normally come from. A subset selected only
for landslide-containing tiles is fine for an initial "does the terrain
signal even separate these" check, but is not representative for actual
model training/evaluation later -- you'll need some landslide-free tiles
too, ideally spatially adjacent to the ones below, once you move past pure
feasibility checking. This script flags that rather than hiding it.

RUN THIS LOCALLY -- needs `shapely`, not on Claude's sandbox allowlist for
network access anyway (this script itself makes no network calls, it's
pure geometry, but keeping it consistent with the rest of the pipeline).

Usage:
    python 04_select_tile_subset.py \\
        --deposits slido_deposits_oregon_city.geojson \\
        --tiles or_olcmetro_2019_tiles.json
"""
import json
import argparse

try:
    from shapely.geometry import shape, box
except ImportError:
    print("This script needs shapely: pip install shapely")
    raise


def load_deposits(path):
    with open(path) as fh:
        data = json.load(fh)
    polygons = []
    for feat in data["features"]:
        geom = shape(feat["geometry"])
        props = feat.get("properties", {})
        polygons.append((geom, props))
    print(f"Loaded {len(polygons)} landslide deposit polygon(s) from {path}")
    return polygons


def load_tiles(path):
    with open(path) as fh:
        tiles = json.load(fh)
    print(f"Loaded {len(tiles)} tile record(s) from {path}")
    return tiles


def select_intersecting_tiles(deposits, tiles):
    selected = []
    for tile in tiles:
        bb = tile["boundingBox"]
        tile_box = box(bb["minX"], bb["minY"], bb["maxX"], bb["maxY"])

        matching_deposits = [
            props.get("MOVE_CODE", "unknown")
            for geom, props in deposits
            if tile_box.intersects(geom)
        ]

        if matching_deposits:
            selected.append({
                **tile,
                "_landslide_count": len(matching_deposits),
                "_move_codes": matching_deposits,
            })

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deposits", required=True, help="Path to SLIDO deposits geojson (from Step 1b)")
    parser.add_argument("--tiles", required=True, help="Path to project tile list json (e.g. or_olcmetro_2019_tiles.json)")
    parser.add_argument("--out", default="tile_subset_feasibility.json", help="Output path for selected tile subset")
    args = parser.parse_args()

    deposits = load_deposits(args.deposits)
    tiles = load_tiles(args.tiles)

    selected = select_intersecting_tiles(deposits, tiles)

    print(f"\n{len(selected)} of {len(tiles)} tiles intersect at least one landslide polygon")
    total_mb = sum(t.get("sizeInBytes", 0) for t in selected) / 1e6
    print(f"Subset download size: {total_mb:.0f} MB ({total_mb/1024:.2f} GB)")
    print(f"vs full project: {sum(t.get('sizeInBytes', 0) for t in tiles) / 1e9:.2f} GB")

    landslide_counts = [t["_landslide_count"] for t in selected]
    if landslide_counts:
        print(f"\nLandslides per tile: min={min(landslide_counts)}, max={max(landslide_counts)}, "
              f"avg={sum(landslide_counts)/len(landslide_counts):.1f}")

    with open(args.out, "w") as fh:
        json.dump(selected, fh, indent=2)
    print(f"\nSaved subset to {args.out}")

    print("\nREMINDER: this subset contains ONLY landslide-positive tiles.")
    print("Fine for an initial feasibility check on terrain-derivative separability,")
    print("but you'll need landslide-free (negative) tiles too before real model")
    print("training/eval -- come back to this once feasibility looks promising.")


if __name__ == "__main__":
    main()
