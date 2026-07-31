# Acceptance Tests

The agent must add automated tests covering the following behavior. Tests should
not require network access or full-size LAZ downloads.

## Core provenance tests

### 1. Header creation year cannot override acquisition

Input:

```text
authoritative acquisition year = 2020
LAS creation date = 2024
project name contains A22
```

Expected:

```text
canonical acquisition year = 2020
file creation year = 2024
inferred project hint may be 2022, but is non-authoritative
```

SLIDO temporal filtering and NAIP selection must receive `2020`.

### 2. No explicit acquisition metadata

Input:

```text
LAS creation date = 2024
project name contains A22
filename contains no trusted acquisition metadata
```

Expected:

```text
canonical acquisition year = unknown
file creation year = 2024
project-name hint = 2022, non-authoritative
```

No vintage gap may be claimed.

### 3. Explicit CLI override

Input:

```text
--lidar-acquisition-year 2020
--lidar-acquisition-source "verified project metadata"
```

Expected:

- acquisition resolves to 2020;
- source is preserved;
- file creation is still recorded separately;
- output manifests contain both.

### 4. Registry acquisition metadata

A pinned project with valid acquisition metadata must resolve correctly.

Invalid cases must fail:

- year outside allowed range;
- start > end;
- nominal outside range;
- missing source;
- missing evidence when required;
- project mismatch;
- partially specified acquisition object.

### 5. Conflicting authoritative sources

Registry says 2020 and an explicit tile provenance source says 2021.

Expected: clear failure with both sources named.

Do not silently choose one.

### 6. Multi-year survey

Input:

```text
start=2008
end=2009
nominal missing
```

Expected: fail when a scalar year is required.

Input:

```text
start=2008
end=2009
nominal=2009
```

Expected: preserve all three fields and use 2009 only where a scalar is
explicitly required.

## Output tests

### 7. Patch manifest propagation

`patches.csv` includes:

- canonical acquisition year;
- acquisition source/evidence/verified status;
- start/end where supported;
- file creation year;
- compatibility `lidar_year` alias, if retained.

### 8. Tile and dataset summary propagation

Tile summaries and dataset summary preserve the same temporal provenance and do
not collapse file creation into acquisition.

### 9. SLIDO filtering protection

Mock or patch `rasterize_slido_mask`.

Assert that the function receives:

```text
lidar_year = authoritative acquisition nominal year
```

and never the LAS creation year or project-name hint.

### 10. Unknown-year production safety

Verify documented behavior for a production build without authoritative
acquisition metadata.

Preferred expected behavior: fail unless a clearly named diagnostic override is
used.

## NAIP tests

### 11. Nearest-year selection

With authoritative LiDAR acquisition 2020 and available NAIP years 2018, 2020,
2022, select according to the existing nearest-year policy using 2020 as the
target.

### 12. Cached imagery manifest refresh

An existing cached NAIP NPZ contains stale LiDAR-related metadata or was
generated before the fix.

Regenerating the NAIP manifest from current `patches.csv` must write the current
authoritative acquisition provenance, without redownloading imagery.

### 13. Unknown acquisition

When LiDAR acquisition is unknown:

- `year_gap` is blank/null;
- `gap_flag` is blank/null or explicitly unavailable;
- no “within 2 years” statement is produced.

### 14. Tile conflict

Rows for the same tile contain conflicting authoritative acquisition years.

Expected: error naming the tile and conflicting years.

## Viewer tests

### 15. Correct vintage context

Input:

```text
acquisition = 2020
file creation = 2024
NAIP = 2022
```

Expected visible semantics:

```text
LiDAR acquisition 2020
LAS file created 2024
NAIP 2022
gap +2
```

### 16. Unknown acquisition display

Expected:

```text
LiDAR acquisition unknown
Vintage gap unavailable
```

The viewer must not claim the gap is within two years.

### 17. Existing mask semantics remain intact

Retain:

```text
0 = background
1 = landslide
255 = ignore
```

Positive and ignore percentages must remain separately calculated.

## Regression and quality commands

From repository root, run at minimum:

```powershell
python -m unittest discover -s .\oregon\tests -v
python -m unittest discover -s .\tests -v
python -m compileall -q .\oregon .\tests
```

Run any narrower tests during development, but complete the relevant suites
before reporting success.

If the repository uses another test runner at current HEAD, run it too.
