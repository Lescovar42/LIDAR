import json
import argparse
import os

try:
    import folium
except ImportError:
    print("This script requires folium.")
    print("pip install folium")
    exit(1)

def bbox_to_geojson(tile):
    bb = tile["boundingBox"]
    return {
        "type": "Feature",
        "properties": {"title": tile.get("title", "")},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bb["minX"], bb["minY"]],
                [bb["maxX"], bb["minY"]],
                [bb["maxX"], bb["maxY"]],
                [bb["minX"], bb["maxY"]],
                [bb["minX"], bb["minY"]]
            ]]
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deposits", default="slido_deposits_oregon_city.geojson")
    parser.add_argument("--boundary", default="oregon_city_study_boundary.geojson")
    parser.add_argument("--all-tiles", default="3dep_tile_discovery.json")
    parser.add_argument("--subset", default="tile_subset_feasibility.json")
    parser.add_argument("--out", default="spatial_coverage_map.html")
    args = parser.parse_args()

    print("Loading data...")
    try:
        with open(args.deposits) as f:
            deposits_geojson = json.load(f)
            
        with open(args.boundary) as f:
            boundary_geojson = json.load(f)
            
        with open(args.all_tiles) as f:
            all_tiles = json.load(f)
            
        with open(args.subset) as f:
            subset_tiles = json.load(f)
            
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Convert tile bounding boxes to GeoJSON features
    all_tiles_features = [bbox_to_geojson(t) for t in all_tiles]
    subset_tiles_features = [bbox_to_geojson(t) for t in subset_tiles]
    
    all_tiles_geojson = {"type": "FeatureCollection", "features": all_tiles_features}
    subset_tiles_geojson = {"type": "FeatureCollection", "features": subset_tiles_features}

    print("Generating interactive HTML map...")
    
    # Calculate rough center from boundary bounding box or first point
    try:
        coords = boundary_geojson["features"][0]["geometry"]["coordinates"][0][0]
        center_y, center_x = coords[1], coords[0]
    except:
        center_y, center_x = 45.35, -122.6  # Rough center of Oregon City
        
    m = folium.Map(location=[center_y, center_x], zoom_start=11, tiles="cartodbpositron")
    
    # Add all tiles
    folium.GeoJson(
        all_tiles_geojson,
        name="All 3DEP Tiles (Background)",
        style_function=lambda x: {'fillColor': 'none', 'color': 'gray', 'weight': 1, 'opacity': 0.5}
    ).add_to(m)
    
    # Add selected tiles
    folium.GeoJson(
        subset_tiles_geojson,
        name="Selected Tiles",
        style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'weight': 2, 'fillOpacity': 0.1}
    ).add_to(m)
    
    # Add SLIDO deposits
    folium.GeoJson(
        deposits_geojson,
        name="SLIDO Landslides",
        style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 1, 'fillOpacity': 0.5}
    ).add_to(m)
    
    # Add boundary
    folium.GeoJson(
        boundary_geojson,
        name="Study Boundary",
        style_function=lambda x: {'fillColor': 'none', 'color': 'black', 'weight': 3}
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    m.save(args.out)
    print(f"Saved interactive map to {args.out}")

if __name__ == "__main__":
    main()
