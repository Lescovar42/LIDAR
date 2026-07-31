# Current Context for the Agent

## Research workflow

The project builds a LiDAR/SLIDO segmentation dataset for landslide mapping.

Current Tillamook design:

- primary source: `USGS_LPC_OR_WesternWildfires_A22`;
- labels: SLIDO landslide polygons;
- imagery context: NAIP;
- model family: U-Net segmentation;
- final evaluation must remain spatially separated;
- do not replace design choices casually.

The user-provided project decision is:

- Tillamook A22 local acquisition year: `2020`;
- the LAS/LAZ files may report a file creation date of `2024`;
- the current NAIP sample is `2022`.

The temporal display currently shows `LiDAR 2024 | NAIP 2022`, which is wrong
for acquisition-based interpretation.

## Current code behavior observed before the agent runs

At the time this handoff was prepared, `oregon/build_dataset.py` contained a
`determine_lidar_year(...)` flow that preferred a valid LAS header creation
year before project-name and filename parsing.

`read_tile_metadata(...)` passed that value into `TileMetadata.lidar_year`.

The dataset builder later passed `item.lidar_year` to:

```python
rasterize_slido_mask(..., lidar_year=item.lidar_year, ...)
```

It also wrote `lidar_year` and `lidar_year_source` into patch and tile outputs.

`oregon/fetch_naip_qc.py` read `lidar_year` from patch rows, used it as the
target for NAIP year selection, and calculated `NAIP - LiDAR` gap.

`oregon/diagnostics/qc_patch_viewer.py` displayed that value and showed a
within/exceeds-two-years message.

The agent must inspect current HEAD because the user has committed additional
changes since this pack was prepared.

## Current dataset status

The user has already created:

```text
F:\LIDAR\oregon\dataset_tillamook_probe_15m
```

Diagnostic build properties:

```text
cell size = 1.5 m
patch size = 256
stride = 128
minimum patch ground occupancy = 0.20
```

The spatial split leakage check passed.

The user also fixed the mask visualization bug so that:

```text
positive = mask == 1
ignore = mask == 255
```

Do not undo that change.

Do not automatically delete or rebuild the current dataset. Provide commands
for the user after the code fix.

## Source locations

Repository:

```text
https://github.com/Lescovar42/LIDAR
```

Relevant source URLs:

```text
https://raw.githubusercontent.com/Lescovar42/LIDAR/main/oregon/build_dataset.py
https://raw.githubusercontent.com/Lescovar42/LIDAR/main/oregon/region_registry.py
https://raw.githubusercontent.com/Lescovar42/LIDAR/main/oregon/regions.json
https://raw.githubusercontent.com/Lescovar42/LIDAR/main/oregon/fetch_naip_qc.py
https://raw.githubusercontent.com/Lescovar42/LIDAR/main/oregon/diagnostics/qc_patch_viewer.py
```
