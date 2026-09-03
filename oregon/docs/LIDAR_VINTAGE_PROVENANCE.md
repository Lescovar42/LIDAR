# LiDAR acquisition-year provenance

The pipeline used to resolve a single `lidar_year` by preferring the LAS/LAZ
header `creation_date`, then a year parsed from the project name, then a year
parsed from the tile name. That scalar then drove SLIDO temporal filtering,
NAIP nearest-year selection, the displayed LiDAR/NAIP gap, and patch/tile
provenance.

A LAS `creation_date` can describe file writing, processing, or repackaging. A
token such as `A22` is a catalog/project name, not a flight year. Using either
as acquisition evidence changes both data eligibility and interpretation.

Acquisition metadata is now explicit, validated, and separated from file
metadata and from name-derived hints. `oregon/lidar_vintage.py` owns the model
and is the only place that decides what counts as authoritative.

## Three separate concepts

| concept | type | may drive temporal reasoning? |
| --- | --- | --- |
| acquisition vintage | `LidarAcquisition` | yes, and only this |
| LAS/LAZ header date | `LidarFileMetadata` | never |
| project/filename year token | `InferredYearHint` | never |

## Schema

### Registry (`regions.json`, `schema_version: 2`)

`lidar_acquisition` is optional per region and is tied to one LiDAR project:

```json
"lidar_acquisition": {
  "start_year": 2020,
  "end_year": 2020,
  "nominal_year": 2020,
  "source": "User-supplied project decision for the Tillamook subset of USGS 3DEP OR_WesternWildfires_A22",
  "evidence": "regions/tillamook_lidar_acquisition.md",
  "verified": true,
  "lidar_project": "OR_WesternWildfires_A22"
}
```

Validation (`region_registry.validate_registry`) rejects: years outside
`1990..currentYear+1`, `start_year > end_year`, a `nominal_year` outside the
range, a missing/blank `source`, a missing/blank `evidence`, a `lidar_project`
outside `candidate_projects`, a `lidar_project` that disagrees with a pinned
`lidar_project`, `start_year` without `end_year`, an empty object, unsupported
fields, and a non-boolean `verified`.

Acquisition metadata is independent of pinning: a region may carry a verified
acquisition vintage while `lidar_project`/`cell_size` remain intentionally
unpinned. `pin_region_decision` refuses to pin a project that contradicts stored
acquisition metadata, and `set_region_acquisition` is the atomic writer for the
acquisition block.

### `patches.csv`

```text
lidar_year                        compatibility alias = lidar_acquisition_year
lidar_year_source                 resolution origin, or "unknown"
lidar_acquisition_year            authoritative nominal year
lidar_acquisition_start_year
lidar_acquisition_end_year
lidar_acquisition_source
lidar_acquisition_evidence
lidar_acquisition_verified        "true" / "false" / "" when unknown
lidar_file_creation_year          LAS header, audit only
lidar_file_creation_date          LAS header, audit only
lidar_inferred_year_hint          non-authoritative
lidar_inferred_year_hint_source   "project_name" | "tile_name" | "none"
```

Tile summaries add a typed `lidar_vintage` block plus
`slido_temporal_filter_year`. `dataset_summary.json` adds
`lidar_vintage_summary` with distinct acquisition vintages and their tile
counts, distinct file-creation years, inferred hints, unknown-acquisition tile
count, conflict failures, per-region requested acquisition, and whether the
unknown-acquisition diagnostic flag was used.

`naip_manifest.csv` carries the same acquisition and file-creation columns.

## Resolution precedence

1. explicit CLI acquisition override (legacy/single-region diagnostic builds);
2. verified registry region/project acquisition metadata;
3. unknown.

When more than one authoritative source is present they must agree on
`(start_year, end_year, nominal_year)`. Disagreement raises
`AcquisitionConflictError` naming every source and its years; no source is
chosen automatically.

These are never promoted to authoritative acquisition metadata: LAS
`creation_date`, `A22`, a four-digit year in a filename or project name, the
current year, the NAIP year.

## CLI

```text
--lidar-acquisition-year        (alias: --lidar-year, deprecated)
--lidar-acquisition-start-year
--lidar-acquisition-end-year
--lidar-acquisition-source      required whenever any acquisition value is given
--lidar-acquisition-evidence
--lidar-acquisition-verified
--allow-unknown-lidar-acquisition   diagnostic only
```

A production build without authoritative acquisition metadata fails. With
`--allow-unknown-lidar-acquisition` the build proceeds, prints a warning, leaves
every acquisition column blank, passes `lidar_year=None` into SLIDO temporal
filtering (so no polygon is temporally excluded), and no vintage gap is claimed
anywhere downstream.

A multi-year survey keeps `start_year`/`end_year`. If a scalar is required and
no `nominal_year` was given, the build fails rather than picking a midpoint,
start, end, or latest year.

## SLIDO temporal filtering

`build_dataset.rasterize_tile_mask` is the single call site. It passes
`TileMetadata.slido_lidar_year()`, which returns the authoritative nominal
acquisition year or `None`. The LAS creation year and name hints are structurally
unable to reach `rasterize_slido_mask`.

Mask semantics are unchanged: `0` background, `1` landslide, `255` ignore.

## NAIP

`fetch_naip_qc.py` selects the nearest NAIP year against the authoritative
acquisition year only. Unknown acquisition means no target year, so the existing
"most recent available" policy applies and `year_gap`/`gap_flag` stay blank.

Cached imagery is reused, but every LiDAR provenance column in
`naip_manifest.csv` is rebuilt from the current patch manifest, never from the
cached NPZ. `--refresh-manifest-only` rewrites the manifest from cache with no
network access at all.

Rows from one tile that disagree on the acquisition year raise an error naming
the tile and both years.

A legacy `patches.csv` that has no `lidar_acquisition_year` column is treated as
*unknown* acquisition, because its `lidar_year` may hold a file-creation year.
`--trust-legacy-lidar-year` opts back in explicitly.

## QC viewer

```text
LiDAR acquisition 2020 | LAS file created 2024 | NAIP 2022 | gap +2 years
```

When acquisition or the NAIP year is unknown:

```text
LiDAR acquisition unknown | LAS file created 2024 | NAIP 2022 | gap unavailable
Vintage gap unavailable (LiDAR acquisition year unknown); do not assume the vintages match.
```

The viewer recomputes the gap from the acquisition year, so stale
`year_gap`/`gap_flag` columns in an old NAIP manifest are ignored. It never
claims "within 2 years" unless both the acquisition year and the NAIP year are
known. The mask overlay still separates `mask == 1` from `mask == 255`.

## Rebuilding the Tillamook diagnostic dataset

Registry mode (R1 already carries verified acquisition metadata):

```powershell
cd F:\LIDAR\oregon
python .\build_dataset.py `
  --region R1 `
  --allow-unpinned-rural-diagnostic `
  --cell-size 1.5 `
  --laz-dir ".\probe_lidar" `
  --outdir ".\dataset_tillamook_probe_15m" `
  --patch-size 256 --stride 128 --max-tiles 24 `
  --min-ground-cell-fraction 0.0 --min-patch-ground-fraction 0.20 `
  --overwrite
```

Legacy/direct mode with an explicit override:

```powershell
python .\build_dataset.py `
  --laz-dir ".\probe_lidar\tillamook_probe\USGS_LPC_OR_WesternWildfires_A22" `
  --slido-geojson ".\slido_tillamook.geojson" `
  --outdir ".\dataset_tillamook_probe_15m" `
  --lidar-acquisition-year 2020 `
  --lidar-acquisition-source "User-verified USGS Tillamook A22 project metadata" `
  --lidar-acquisition-evidence "regions/tillamook_lidar_acquisition.md" `
  --lidar-acquisition-verified `
  --cell-size 1.5 --patch-size 256 --stride 128 --max-tiles 24 `
  --min-ground-cell-fraction 0.0 --min-patch-ground-fraction 0.20 `
  --overwrite
```

Verify provenance:

```powershell
Import-Csv .\dataset_tillamook_probe_15m\patches.csv |
  Group-Object lidar_acquisition_year, lidar_acquisition_source, lidar_file_creation_year |
  Select-Object Name, Count
```

Refresh NAIP manifests without redownloading imagery:

```powershell
python .\fetch_naip_qc.py --dataset-dir .\dataset_tillamook_probe_15m --refresh-manifest-only
```

Review:

```powershell
python .\diagnostics\qc_patch_viewer.py --dataset-dir .\dataset_tillamook_probe_15m
```

## Outstanding evidence

`regions/tillamook_lidar_acquisition.md` records the 2020 acquisition year as a
project-owner attestation and lists the USGS records that would make it
externally traceable. Update `source`/`evidence` once such a record is attached.
