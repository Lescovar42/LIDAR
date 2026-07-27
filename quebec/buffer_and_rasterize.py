"""
02_buffer_and_rasterize.py
============================
Buffers Quebec landslide event points into circular polygons, then
rasterizes them into binary masks aligned to an HRDEM tile -- adapting
06_mvp_training_pipeline.py's rasterize_landslides() for point+buffer
input instead of SLIDO's real polygons.

BUFFER RADIUS -- grounded in literature, not arbitrary:
Real Quebec sensitive-clay landslide dimensions from Demers/Locat et al.:
  - Saint-Jude area spread: ~275 m width (Earth.com/Locat team reporting)
  - Bristol landslide (found via LiDAR hillshade, Demers et al. 2017):
    780 m length x 520 m width
These are documented LARGE events -- plausible given a civil-security
database likely captures landslides significant enough to be reported,
not every small slope failure. Given this, a 150 m buffer RADIUS (300 m
diameter) is used as a defensible middle estimate: smaller than the
Bristol event, in the neighborhood of the Saint-Jude width. This is a
real modeling assumption, not a precise measurement -- state it plainly
in your methods as "landslide extent approximated via literature-informed
buffer, actual scarp geometry unknown," since the true extent of any
given point in this dataset could be considerably smaller or larger.

RUN THIS LOCALLY -- needs the same lidar_tiles/ + rasterio/pyproj/shapely
stack as the Oregon pipeline, plus output from
01_fetch_quebec_landslide_points.py.

Usage:
    python 02_buffer_and_rasterize.py <path_to_quebec_hrdem.tif>
"""
import json
import sys
import numpy as np

BUFFER_RADIUS_M = 150.0  # see literature justification above
LANDSLIDE_POINTS_PATH = "quebec_landslide_points.geojson"


def load_canadian_dem(tif_path):
    """Load a Canadian HRDEM GeoTIFF directly -- no LAZ->DEM step needed,
    since NRCan distributes HRDEM as pre-processed bare-earth rasters."""
    import rasterio
    with rasterio.open(tif_path) as src:
        dem = src.read(1).astype(np.float32)
        geo_transform = src.transform
        crs = src.crs
        nodata = src.nodata
    print(f"  Loaded DEM: {dem.shape}, CRS: {crs}")
    if nodata is not None:
        dem[dem == nodata] = np.nan
    return dem, geo_transform, crs


def buffer_and_rasterize(points_path, geo_transform, crs, dem_shape,
                          buffer_radius_m=BUFFER_RADIUS_M):
    """
    Project WGS84 points into the DEM's CRS, buffer into circles, rasterize.
    """
    from pyproj import Transformer
    from PIL import Image, ImageDraw

    with open(points_path) as f:
        points_data = json.load(f)

    features = points_data.get("features", [])
    print(f"  Loaded {len(features)} landslide point(s)")

    # WGS84 -> DEM's native CRS (whatever the HRDEM tile actually uses --
    # confirmed dynamically from the raster itself, not hardcoded, since
    # Quebec HRDEM tiles may be distributed in different UTM zones/MTM
    # zones depending on region)
    dem_epsg = crs.to_epsg()
    if dem_epsg is None:
        raise ValueError(
            "Could not determine EPSG code from DEM CRS. Check the DEM's "
            "projection manually (e.g. with `gdalinfo`) and hardcode it."
        )
    print(f"  Reprojecting points from EPSG:4326 to EPSG:{dem_epsg}")
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{dem_epsg}", always_xy=True)

    nrows, ncols = dem_shape
    mask_img = Image.new("L", (ncols, nrows), 0)
    draw = ImageDraw.Draw(mask_img)

    n_drawn = 0
    n_outside = 0
    for feat in features:
        lon, lat = feat["geometry"]["coordinates"]
        px, py = transformer.transform(lon, lat)

        # Projected coords -> pixel coords (using rasterio's affine transform)
        col, row = ~geo_transform * (px, py)
        col, row = int(col), int(row)

        # Buffer radius in pixels (assumes square, meter-based pixel size --
        # true for HRDEM, which is typically 1m or 2m resolution)
        pixel_size = abs(geo_transform.a)
        radius_px = buffer_radius_m / pixel_size

        if -radius_px <= col <= ncols + radius_px and -radius_px <= row <= nrows + radius_px:
            draw.ellipse(
                [col - radius_px, row - radius_px, col + radius_px, row + radius_px],
                fill=1,
            )
            n_drawn += 1
        else:
            n_outside += 1

    print(f"  Drew {n_drawn} buffered landslide circle(s) onto mask")
    print(f"  {n_outside} point(s) fell outside this tile's extent")

    mask = np.array(mask_img, dtype=np.float32)
    landslide_pixels = int(mask.sum())
    total_pixels = nrows * ncols
    print(f"  Landslide pixels: {landslide_pixels:,} / {total_pixels:,} "
          f"({100*landslide_pixels/total_pixels:.2f}%)")

    return mask


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 02_buffer_and_rasterize.py <path_to_hrdem.tif>")
        sys.exit(1)

    tif_path = sys.argv[1]
    print(f"Loading {tif_path}...")
    dem, geo_transform, crs = load_canadian_dem(tif_path)

    print("\nBuffering and rasterizing landslide points...")
    mask = buffer_and_rasterize(LANDSLIDE_POINTS_PATH, geo_transform, crs, dem.shape)

    np.save("quebec_landslide_mask.npy", mask)
    print("\nSaved mask to quebec_landslide_mask.npy")

    if mask.sum() == 0:
        print("\nWARNING: zero landslide pixels in this tile. Either no")
        print("landslide points fall within this HRDEM tile's extent, or")
        print("something is wrong with the CRS transform above -- double")
        print("check by plotting points and DEM extent together before")
        print("assuming the pipeline is broken.")
