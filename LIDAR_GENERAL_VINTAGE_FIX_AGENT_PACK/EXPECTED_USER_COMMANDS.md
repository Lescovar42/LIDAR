# Expected User Commands After the Fix

The agent must adapt these commands to the exact final CLI.

Do not run destructive rebuilds automatically.

## 1. Confirm the new CLI

```powershell
cd F:\LIDAR\oregon
python .\build_dataset.py --help
```

## 2. Rebuild the Tillamook diagnostic dataset with explicit acquisition provenance

Illustrative command; use the exact names implemented by the agent:

```powershell
python .\build_dataset.py `
  --laz-dir ".\probe_lidar\tillamook_probe\USGS_LPC_OR_WesternWildfires_A22" `
  --slido-geojson ".\slido_tillamook.geojson" `
  --outdir ".\dataset_tillamook_probe_15m" `
  --lidar-acquisition-year 2020 `
  --lidar-acquisition-source "User-verified USGS Tillamook A22 project metadata" `
  --cell-size 1.5 `
  --patch-size 256 `
  --stride 128 `
  --max-tiles 24 `
  --min-ground-cell-fraction 0.0 `
  --min-patch-ground-fraction 0.20 `
  --overwrite
```

If the implementation requires an evidence field, provide a traceable value
rather than inventing one. The agent should tell the user what repository file
or project record should be cited.

## 3. Verify provenance in `patches.csv`

Illustrative PowerShell:

```powershell
Import-Csv .\dataset_tillamook_probe_15m\patches.csv |
  Group-Object `
    lidar_acquisition_year, `
    lidar_acquisition_source, `
    lidar_file_creation_year |
  Select-Object Name, Count
```

Expected semantic result:

```text
acquisition = 2020
file creation = 2024, if that is what the LAZ header contains
```

## 4. Refresh NAIP manifests while reusing imagery cache

The agent must provide the exact safe command.

Expected behavior:

- no unnecessary imagery download;
- `naip_manifest.csv` refreshed from corrected patch provenance;
- target LiDAR acquisition year is 2020;
- NAIP year remains based on the existing selection policy;
- gap is calculated against acquisition, not file creation.

## 5. Verify viewer

Expected text semantics:

```text
LiDAR acquisition 2020
LAS file created 2024
NAIP 2022
gap +2 years
```

The agent should provide the exact QC viewer command matching current HEAD.
