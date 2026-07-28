# USGS NAIP QC — First Implementation

NAIP is the better visual companion for this project.

The official USGS NAIP service is primarily 0.6 m, four-band imagery. At that
resolution, a 256 m LiDAR patch is approximately 427 pixels across, compared
with only about 26 pixels in Sentinel-2.

## Files

Copy these into the existing repository:

```text
oregon/fetch_naip_qc.py
oregon/diagnostics/qc_patch_viewer.py
```

The QC viewer replaces the Sentinel panels with NAIP panels.

## Build the NAIP cache

From `D:\LIDAR\oregon`:

```powershell
python .\fetch_naip_qc.py `
  --dataset-dir .\dataset_pilot `
  --max-tiles 10 `
  --overwrite
```

Defaults:

```text
context:    512 m × 512 m
resolution: 0.6 m
bands:      Red, Green, Blue, NIR
```

This creates approximately 854×854-pixel NAIP contexts.

To reduce storage and align more closely with the 1 m LiDAR grid:

```powershell
python .\fetch_naip_qc.py `
  --dataset-dir .\dataset_pilot `
  --max-tiles 10 `
  --resolution 1.0 `
  --overwrite
```

To force one available NAIP year:

```powershell
python .\fetch_naip_qc.py `
  --dataset-dir .\dataset_pilot `
  --year 2022 `
  --max-tiles 10 `
  --overwrite
```

Without `--year`, the script queries available imagery and chooses the year
closest to the year embedded in each LiDAR tile name. When two years are equally
close, it prefers the later year.

## Open the QC viewer

```powershell
python .\diagnostics\qc_patch_viewer.py `
  --dataset-dir .\dataset_pilot
```

The panels are:

```text
NAIP natural color
NAIP natural color + SLIDO
NAIP color infrared
LiDAR hillshade
LiDAR slope
LiDAR + SLIDO
```

A cyan rectangle marks the central 256 m LiDAR patch inside the 512 m NAIP
context.

## What NAIP helps identify

```text
roads and engineered cuts
quarries and construction
forest, clear-cuts, and fields
rivers and exposed bars
buildings and developed areas
bare-earth disturbances
```

LiDAR still provides the decisive evidence for scarps, toes, hummocky deposits,
and terrain deformation.

## Why this version does not use M2M yet

The ImageServer clips only the needed context, which is ideal for QC and avoids
downloading large quarter-quadrangle files.

Your approved USGS Machine-to-Machine access will be used for the training-data
stage, where the pipeline should download and preserve original NAIP source
tiles, acquisition dates, product IDs, and checksums. Authentication should use
a USGS application token with the `login-token` endpoint—not an ERS password.

Do not commit the application token to GitHub.
