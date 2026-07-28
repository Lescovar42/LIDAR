# Sentinel-2 QC: First Implementation

This adds Sentinel-2 Level-2A visual context to the existing Oregon patch QC
workflow without changing the LiDAR training arrays.

## Files to copy

From this bundle, copy:

```text
oregon/fetch_sentinel2_qc.py
oregon/diagnostics/qc_patch_viewer.py
```

Allow `qc_patch_viewer.py` to replace the previous version.

The implementation uses packages already listed in the Oregon requirements:

```text
requests
numpy
pyproj
rasterio
matplotlib
```

No Copernicus credentials are required. It searches the public Element 84 Earth
Search STAC catalog and reads public Sentinel-2 L2A Cloud-Optimized GeoTIFFs.

## Step 1: build the Sentinel cache

From `D:\LIDAR\oregon`:

```powershell
python .\fetch_sentinel2_qc.py `
  --dataset-dir .\dataset_pilot `
  --max-tiles 10
```

The LiDAR tile names contain `2019`, so the default behavior searches summer
imagery from the inferred year plus or minus one year and selects a low-cloud
scene close to summer 2019.

To force a date range:

```powershell
python .\fetch_sentinel2_qc.py `
  --dataset-dir .\dataset_pilot `
  --start-date 2019-06-01 `
  --end-date 2020-09-30 `
  --max-cloud 30 `
  --max-tiles 10
```

To retry and replace existing Sentinel files:

```powershell
python .\fetch_sentinel2_qc.py `
  --dataset-dir .\dataset_pilot `
  --max-tiles 10 `
  --overwrite
```

Outputs:

```text
dataset_pilot/
└── sentinel2/
    ├── sentinel2_manifest.csv
    ├── tile_scenes.json
    ├── summary.json
    └── patches/
        ├── train/
        ├── validation/
        └── test/
```

Each Sentinel NPZ contains native-resolution context at approximately 10 m:

```text
bands       B02, B03, B04, B08 reflectance
scl         Sentinel scene classification
valid_mask  cloud/shadow/snow/NoData validity mask
```

The display is enlarged to match the LiDAR panel size. Enlargement does not add
spatial detail.

## Step 2: open the updated viewer

```powershell
python .\diagnostics\qc_patch_viewer.py `
  --dataset-dir .\dataset_pilot `
  --only-unreviewed
```

The six panels are:

```text
Sentinel true color
Sentinel true color + SLIDO
Sentinel false color (NIR / Red / Green)
LiDAR hillshade
LiDAR slope
LiDAR hillshade + SLIDO
```

Sentinel metadata appears above the plots:

```text
item ID
acquisition date
catalog cloud cover
local valid-pixel fraction
```

## How to interpret Sentinel panels

True color helps identify:

```text
roads and urban areas
quarries and excavations
rivers and exposed bars
forest and clear cuts
bare soil and recent disturbance
```

False color shows healthy vegetation as bright red. Bare soil, roads, water, and
engineered surfaces generally appear different from forest, making confusing
terrain easier to identify.

Use Sentinel as context only. The landslide boundary and fine morphology must
still be judged mainly from the 1 m LiDAR products because each 256 m LiDAR patch
contains only about 26 × 26 native Sentinel pixels.

## Network troubleshooting

The first run requires internet access. If GDAL reports an SSL, HTTP range, or
S3 error:

1. Upgrade rasterio and requests:

```powershell
python -m pip install --upgrade rasterio requests certifi
```

2. Confirm this URL opens in a browser:

```text
https://earth-search.aws.element84.com/v1
```

3. Retry one tile:

```powershell
python .\fetch_sentinel2_qc.py `
  --dataset-dir .\dataset_pilot `
  --max-tiles 1 `
  --overwrite
```

Errors are written to `sentinel2_manifest.csv`; successful tiles remain cached.

## Scope

This first implementation is for visual QC and future reuse. It does not yet
concatenate Sentinel channels into the LiDAR model. Keep the current LiDAR-only
baseline unchanged until the Sentinel cache and QC workflow are verified.
