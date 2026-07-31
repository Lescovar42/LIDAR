# LiDAR Boundary-Aware Training Fix

This is a drop-in, non-destructive fix for the current Tillamook diagnostic dataset.
It does **not** change the 1.5 m cell size, 256×256 patch size, masks, spatial split,
NPZ files, validation distribution, or SLIDO labels.

It creates a new training manifest that:

1. recomputes positive pixels as `mask == 1` and ignore pixels as `mask == 255`;
2. detects a real label boundary inside each patch without treating the crop edge as a boundary;
3. intersects each patch footprint with eligible High/Moderate SLIDO polygons;
4. records patch-specific landslide and polygon IDs;
5. removes positive training rows from a landslide polygon that also appears in validation/test;
6. keeps all ordinary/mixed/boundary-context training patches;
7. caps redundant 90–99% positives to 4 per landslide and 99–100% positives to 2 per landslide;
8. leaves every validation/test row untouched.

The original `patches.csv`, `patches_qc.csv`, and NPZ files are not modified.

## Install

Extract this folder into either:

- `F:\LIDAR` (repository root), or
- `F:\LIDAR\oregon`.

Then run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\INSTALL.ps1
```

The installer:

- copies `prepare_boundary_aware_manifest.py` to `oregon\diagnostics`;
- adds optional `--manifest` support to `train_baseline.py`;
- creates a backup named `train_baseline.py.before_boundary_aware_fix`;
- is safe to run more than once.

## Build the manifest

From `F:\LIDAR\oregon`:

```powershell
.\BUILD_TILLAMOOK_BOUNDARY_MANIFEST.ps1
```

Equivalent explicit command:

```powershell
python .\diagnostics\prepare_boundary_aware_manifest.py `
  --dataset-dir .\dataset_tillamook_probe_15m `
  --manifest .\patches.csv `
  --slido .\slido_tillamook.geojson `
  --max-near-full-per-landslide 4 `
  --max-full-per-landslide 2 `
  --seed 42
```

Outputs inside `dataset_tillamook_probe_15m`:

- `patches_boundary_aware.csv` — manifest used for training;
- `boundary_sampling_audit.csv` — every original row plus keep/drop reason;
- `boundary_sampling_report.json` — before/after counts and polygon-overlap report.

## Verify before training

```powershell
$rows = Import-Csv .\dataset_tillamook_probe_15m\patches_boundary_aware.csv

$rows |
  Group-Object split, coverage_class |
  Select-Object Name, Count

Get-Content .\dataset_tillamook_probe_15m\boundary_sampling_report.json
```

The report must retain all validation rows. Only training rows may be removed.

## Train

The patched trainer automatically prefers `patches_boundary_aware.csv` when it
exists. The explicit form is safer:

```powershell
python .\train_baseline.py `
  --dataset-dir .\dataset_tillamook_probe_15m `
  --manifest .\patches_boundary_aware.csv `
  --outdir .\training_output_tillamook_15m_boundary `
  --epochs 30 `
  --batch-size 4 `
  --device auto
```

Do not add `--require-qc` unless the selected manifest has actually been reviewed
and contains accepted `qc_status` values.

## Default cap rationale

The current diagnostic set had 219/783 patches at 99–100% positive. The defaults
retain some fully positive interior morphology but stop a single large polygon
from contributing many overlapping, nearly identical interiors. Boundary and
mixed-context patches are not capped. Validation/test are not resampled.
