# Agent Instructions — General LiDAR Vintage Provenance Fix

## 1. Problem statement

The current implementation in `oregon/build_dataset.py` resolves `lidar_year`
with a precedence equivalent to:

1. valid LAS/LAZ header `creation_date`;
2. year parsed from project name;
3. year parsed from tile name;
4. unknown.

This is methodologically unsafe.

A LAS/LAZ `creation_date` can be a file-writing or processing date. A project
name token such as `A22` can be a catalog/project naming token rather than the
local acquisition year. Filename years can also describe a project range,
publication, or repackaging.

The downstream pipeline currently uses the resulting scalar year for important
scientific and QC behavior:

- SLIDO temporal inclusion/exclusion;
- nearest-year NAIP selection;
- LiDAR–NAIP gap calculation;
- viewer statements such as “gap is within 2 years”;
- patch and tile provenance.

A wrong year therefore changes both data eligibility and interpretation.

## 2. Required semantics

Separate at least these concepts:

### A. Authoritative acquisition metadata

The airborne data-acquisition vintage used for temporal reasoning.

Recommended canonical fields:

```text
lidar_acquisition_start_year
lidar_acquisition_end_year
lidar_acquisition_year
lidar_acquisition_source
lidar_acquisition_evidence
lidar_acquisition_verified
```

`lidar_acquisition_year` is the explicit scalar/nominal year used by existing
downstream components that require one year.

For a one-year survey:

```text
start_year = end_year = nominal_year
```

For a multi-year survey:

- preserve start and end;
- require an explicit nominal year if downstream scalar behavior is needed;
- do not silently choose midpoint, start, end, or latest year.

A smaller equivalent schema is acceptable only when it preserves the same
semantics and auditability.

### B. File metadata

Keep the LAS/LAZ header date separately, for example:

```text
lidar_file_creation_year
lidar_file_creation_date
```

This metadata must never silently drive temporal filtering or NAIP selection.

### C. Non-authoritative hints

A year parsed from a project or filename may be retained as a diagnostic hint:

```text
lidar_inferred_year_hint
lidar_inferred_year_hint_source
```

It must not be labeled “authoritative” and must not drive scientific filtering.

## 3. Resolution precedence

Use explicit provenance, not heuristic precedence.

Recommended acquisition resolution order:

1. explicit CLI acquisition override for legacy/single-region diagnostic mode;
2. verified region/project acquisition metadata from the registry;
3. verified tile/project provenance metadata from a machine-readable selection
   or download manifest, if implemented and internally consistent;
4. unknown.

Never promote these to authoritative acquisition metadata automatically:

- LAS/LAZ `creation_date`;
- `A22`;
- a four-digit year in a filename;
- a four-digit year in a project name;
- current year;
- NAIP year.

If more than one authoritative source exists and they conflict, fail with a
clear error. Do not silently choose one.

## 4. Registry design

Extend the region/project configuration so acquisition metadata is tied to the
selected LiDAR project, not merely to a geographic region.

Recommended nested structure:

```json
{
  "lidar_project": "OR_WesternWildfires_A22",
  "cell_size": 1.5,
  "lidar_acquisition": {
    "start_year": 2020,
    "end_year": 2020,
    "nominal_year": 2020,
    "source": "USGS project/acquisition metadata",
    "evidence": "path or stable project-record identifier",
    "verified": true
  },
  "selection_decision": {
    "reason": "..."
  }
}
```

Important:

- do not automatically pin `lidar_project` or `cell_size` if the current registry
  intentionally remains diagnostic/unpinned;
- allow acquisition metadata to be supplied through the legacy CLI path;
- validate year bounds, range order, required source/evidence text, and project
  association;
- increase `schema_version` if the registry schema changes;
- preserve backward compatibility where practical, but do not preserve unsafe
  behavior.

For the current Tillamook diagnostic workflow, the user-provided acquisition
year is 2020. Implement a general configuration or CLI mechanism that can carry
that value. Do not put a special Tillamook conditional in code.

## 5. CLI design

For direct/legacy single-region builds, add explicit acquisition arguments.
Prefer clear names such as:

```text
--lidar-acquisition-year
--lidar-acquisition-start-year
--lidar-acquisition-end-year
--lidar-acquisition-source
--lidar-acquisition-evidence
```

A minimal scalar-only implementation may start with:

```text
--lidar-acquisition-year
--lidar-acquisition-source
```

only if the code is structured to support a future range and does not erase
file-creation metadata.

Do not call the authoritative argument merely `--lidar-year` unless backward
compatibility requires an alias. If an alias is retained, document it and route
it to the acquisition field.

Validation requirements:

- four-digit year within a reasonable range;
- start <= end;
- nominal year within the range;
- source text required when an acquisition year is provided;
- reject partial or contradictory metadata;
- no silent fallback to header/file/project year.

## 6. Dataset-build behavior

Refactor `RegionInput` and `TileMetadata` or introduce a dedicated vintage
dataclass.

Required output behavior:

### `patches.csv`

Include authoritative acquisition fields and file-creation fields. Keep legacy
`lidar_year` only as a compatibility alias to the authoritative nominal
acquisition year.

Example:

```text
lidar_year=2020
lidar_year_source=registry_verified_project_metadata
lidar_acquisition_start_year=2020
lidar_acquisition_end_year=2020
lidar_acquisition_year=2020
lidar_acquisition_source=USGS project/acquisition metadata
lidar_acquisition_evidence=...
lidar_acquisition_verified=true
lidar_file_creation_year=2024
```

### Tile summaries and `dataset_summary.json`

Preserve the same provenance and summarize:

- distinct authoritative acquisition vintages;
- distinct file-creation years;
- unknown acquisition count;
- conflicting metadata failures;
- exact source/evidence values where reasonable.

### SLIDO temporal filtering

The year sent to `rasterize_slido_mask(..., lidar_year=...)` must be the
authoritative nominal acquisition year only.

If no authoritative acquisition year exists:

- do not substitute file creation or inferred hints;
- follow the existing unknown-year behavior if safe;
- for production builds, strongly prefer a clear failure unless an explicit
  diagnostic flag permits unknown acquisition metadata;
- document the behavior.

Do not change SLIDO label semantics or the mask encoding in this task.

## 7. NAIP downloader behavior

`fetch_naip_qc.py` must consume the authoritative acquisition field.

Required behavior:

- use authoritative acquisition year for nearest-year NAIP selection;
- preserve acquisition source/evidence in the NAIP manifest;
- preserve file creation year separately if available;
- if acquisition year is unknown, do not manufacture a gap;
- if cached NAIP arrays are reused, refresh manifest provenance from the current
  patch manifest rather than preserving stale LiDAR vintage fields;
- detect conflicting authoritative years for rows from the same tile.

Backward compatibility:

- legacy manifests containing only `lidar_year` may still be read, but treat the
  field as authoritative only if the legacy contract explicitly says so;
- add tests so a stale cached NAIP file cannot force the old wrong year back
  into a regenerated manifest.

## 8. QC viewer behavior

The viewer must distinguish acquisition from file creation.

Recommended display:

```text
LiDAR acquisition 2020
Source: USGS project/acquisition metadata
LAS file created 2024
NAIP 2022
NAIP − LiDAR acquisition gap: +2 years
```

When acquisition is unknown:

```text
LiDAR acquisition unknown
LAS file created 2024
NAIP 2022
Vintage gap unavailable
```

Do not display “gap is within 2 years” unless both authoritative LiDAR
acquisition year and NAIP year are available.

Do not use the file creation year in the visible LiDAR/NAIP gap.

Keep the existing positive/ignore mask fix intact.

## 9. General conflict and quality rules

Fail loudly when:

- authoritative sources disagree;
- a multi-year range lacks a required nominal year;
- nominal year falls outside the range;
- a registry project’s acquisition metadata is attached to a different project;
- one tile has conflicting authoritative acquisition values;
- a provided source or evidence field is blank.

Do not fail solely because:

- LAS creation date differs from acquisition;
- project-name hint differs from acquisition;
- filename hint differs from acquisition.

Instead, record those differences as provenance diagnostics.

## 10. Files likely to change

Expected:

```text
oregon/build_dataset.py
oregon/region_registry.py
oregon/regions.json or an example/fixture registry
oregon/fetch_naip_qc.py
oregon/diagnostics/qc_patch_viewer.py
oregon/tests/test_region_registry.py
oregon/tests/test_fetch_naip_qc.py
oregon/tests/test_qc_patch_viewer.py
new focused vintage/provenance tests
documentation
```

Potentially:

```text
oregon/terrain_utils.py
```

Only change it if needed to guarantee that temporal filtering receives the
authoritative year.

## 11. Explicit non-goals

Do not:

- pin 1.5 m or 2.0 m automatically;
- alter train/validation/test spatial splitting;
- change the 500 m leakage buffer;
- change SLIDO mask values `0`, `1`, `255`;
- change positive/negative sampling;
- reinterpret ambiguous landslide morphology;
- add random patch splitting;
- download more tiles;
- rebuild or delete the user’s datasets automatically;
- change NAIP imagery content or resolution;
- hardcode Tillamook, A22, 2020, or 2024 into generic logic;
- silently assign a CRS;
- push to GitHub without user authorization.

## 12. Required completion report

At the end, report:

1. current base commit inspected;
2. files changed;
3. final temporal metadata schema;
4. authoritative-source precedence;
5. how file creation and inferred hints are preserved;
6. how SLIDO temporal filtering is protected;
7. how NAIP caching and manifests were handled;
8. tests and exact results;
9. exact rebuild commands for the user’s diagnostic Tillamook dataset;
10. any missing evidence that still prevents registry-level provenance.
