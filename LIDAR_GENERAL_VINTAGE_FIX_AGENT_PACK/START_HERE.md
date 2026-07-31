# Start Here — LiDAR Acquisition-Year General Fix

Repository:

`https://github.com/Lescovar42/LIDAR`

Primary working directory on the user's machine:

`F:\LIDAR\oregon`

## Goal

Implement a general, auditable fix for LiDAR temporal metadata.

The current pipeline can treat the LAS/LAZ header `creation_date` as the LiDAR
acquisition year. In the current Tillamook diagnostic dataset this produced:

- displayed LiDAR year: `2024`
- actual Tillamook A22 acquisition year used by the research design: `2020`
- NAIP year: `2022`

The LAS/LAZ creation date may describe file creation, processing, repackaging,
or publication. It must not silently become the airborne acquisition year.

This task must fix that behavior across:

- dataset construction;
- region/project configuration;
- patch and tile manifests;
- SLIDO temporal filtering;
- NAIP year selection;
- NAIP manifests;
- QC viewer vintage display;
- tests and documentation.

## First actions for the agent

From the repository root:

```powershell
git status --short
git rev-parse HEAD
git log -5 --oneline
```

Preserve all current user changes. Do not reset, revert, or overwrite unrelated
work.

Then read, at minimum:

```text
oregon/build_dataset.py
oregon/region_registry.py
oregon/regions.json
oregon/fetch_naip_qc.py
oregon/diagnostics/qc_patch_viewer.py
oregon/terrain_utils.py
oregon/tests/
```

Read `AGENT_INSTRUCTIONS.md` before changing code.
