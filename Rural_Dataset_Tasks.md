# Implementation Plan — Rural AOI Diversification (Oregon SLIDO/3DEP)

## Problem Statement

The pilot dataset is 10 tiles of `USGS_LPC_OR_OLCMetro_2019` around Oregon City:
199 patches, 131 negatives dominated by metro roads, cuts, and development. A
model trained here can score well by learning urban artifacts rather than
landslide morphology.

Goal: build a rural training/validation set of 50-100 LiDAR tiles (stretch 200),
hold out one rural region geographically, and retain Oregon City as an urban
out-of-domain (OOD) test set.

## Requirements

- Train and validate on rural regions using spatially separated blocks.
- One rural region held out entirely as a rural geographic test.
- Oregon City becomes a separate urban OOD test, not training data.
- Prioritize SLIDO inventories derived from DOGAMI lidar-based mapping, even at
  the cost of tile count.
- Unknown event dates are acceptable for morphology detection. Known
  LiDAR-before-event pairs must be excluded.
- NAIP replaces Sentinel-2 as visual QC context: cached near-native (0.6-1 m),
  RGB+NIR, displayed beside hillshade, slope, and the SLIDO mask.
- NAIP stays QC-only for the first rural experiment. The 7 LiDAR-derived
  channels remain the model input. Later experiments: LiDAR only, NAIP only,
  LiDAR + NAIP.
- NAIP and LiDAR are not simultaneous. The gap is recorded and surfaced, and
  treated as material for clear-cuts, roads, and construction.
- Storage ceiling: 100 GB for LAZ, excluding processed patches and NAIP cache.

## Locked Decisions

| Decision | Choice | Notes |
|---|---|---|
| Cell size | Start 1.0 m, decided in Task 6 | If 2.0 m materially improves usable ground coverage or deep-seated morphology capture, adopt it and rebuild Oregon City at 2.0 m so the OOD comparison stays apples-to-apples |
| Positive labels | `confidence_class` in {high, moderate} | `low` and `unknown` become ignore (255), never negatives. ~1670 of 2123 polygons in the R1 sample box |
| LiDAR project | Unpinned until Task 6 | Compared on footprint overlap with the dense inventory, usable ground coverage, tile count, and acquisition metadata. Neither vintage-match nor NAIP-match alone decides it |

## Background: Measured Evidence

### SLIDO 4.2 label quality

Live queries against the deposits layer (layer 3), `DESCRIPTION='Landslide'`,
grouped by `CONFIDENCE` and `REF_ID_COD`:

| Candidate (sample bbox) | Landslide polys | High conf | Null conf | Dominant source |
|---|---|---|---|---|
| Tillamook Coast Range (-123.95,45.35,-123.55,45.70) | 2123 | 495 | 270 | `CalhNC2020a` 1849 (87%) |
| Buxton / Vernonia (-123.30,45.65,-123.00,45.85) | 1124 | 627 | 248 | `BurnWJ2012k/2013a/2018`, `WellRE2020`, `HairRW2021` — 5 sources total |
| Marion foothills (-122.90,44.75,-122.40,45.05) | 1119 | 341 | 40 | `CALHNC2020` 862, `SobiS2010` 155, `BurnWJ2012b` 62 |
| Columbia River Gorge (-122.20,45.55,-121.60,45.75) | 221 | 8 | 194 | old compilations, unattributed |
| Rogue River-Siskiyou (-123.90,42.25,-123.30,42.65) | 204 | 0 | 204 | old compilations only |

### 3DEP LiDAR coverage (TNM Access LPC)

- Tillamook interior point: 8 overlapping tiles from `OR_TILLAMOOK_ODF_2007`,
  `OR_NORTHCOAST_2008_2009`, and `OR_WesternWildfires_A22` (2022 acquisition,
  published 2025). Whole-bbox total 2338 LPC records, including
  `CA_West_Coast_LiDAR_2016` along the coast.
- Buxton point: only `OR_WILLAMETTE_VALLEY_OLC_2008`.
- Vernonia point: `OR_NORTHCOAST_2008_2009`.
- Marion point (Silver Falls): only `OR_SILVER_FALLS_OPRD_2007`.
- Rogue interior point: `"total": 0` — no 3DEP point cloud at all.

Tile sizes run 110-230 MB with footprints roughly 1.0-1.5 km x 1.3-1.5 km
(~1.6-2.2 km²), versus the pilot's 1 km² OLC Metro tiles. Patch yield per rural
tile will be roughly 4-7x the pilot's 10-35.

### NAIP availability

The USGS NAIP ImageServer reports only `Year = 2022` (plus records with a null
`Year`) over the Tillamook bbox. NAIP context will therefore be 2022 against
2007-2009 LiDAR across most of the Coast Range — a 13-15 year gap in actively
logged forest. Where `OR_WesternWildfires_A22` covers, LiDAR is 2022 and the gap
nearly vanishes.

### Three findings that shape the plan

1. **Rogue River-Siskiyou and the Gorge are out.** Rogue has no LPC coverage and
   zero attributed polygons. The Gorge has 221 polygons, 88% with no confidence
   attribute, plus I-84, the rail line, and Bonneville pool as hard negatives.
   Neither can carry a training region.
2. **Overlapping projects are a leakage hazard.** The same ground appears in
   `OR_TILLAMOOK_ODF_2007` and `OR_WesternWildfires_A22`. Selecting tiles
   without footprint deduplication can place the same hillslope in train and
   test. `build_manifest.aggregate_tiles_for_download` currently dedupes only by
   `tile_id`.
3. **`CONFIDENCE` values are dirty.** Live values include `High (=>30)`,
   `High (>30)`, `High (=<30)`, and `Moderate (11-29) ` with a trailing space.
   Any confidence gate needs normalization first.

### Two code issues the rural move will expose

- `aggregate_tiles_for_download` only selects tiles intersecting a deposit, so
  pure-negative tiles are never downloadable. Rural hard negatives (clear-cuts,
  quarries, river bars) need a deliberate quota.
- `compute_tri` uses `scipy.ndimage.generic_filter` with a Python callable,
  which will dominate runtime on tiles 60-120% larger than the pilot's.

## Recommended AOIs

| Role | Region | Budget | Rationale |
|---|---|---|---|
| Train + validation | **R1 Tillamook Coast Range** | 55-60 tiles | Highest polygon density from a single lidar-era inventory (`CalhNC2020a`, 87%); marine sedimentary Coast Range; clear-cuts, logging roads, quarries, stream banks as honest hard negatives |
| Rural geographic test | **R2 Buxton / Vernonia** | 25-30 tiles | Best high-confidence density measured (627 in ~515 km²); only 5 source publications, all lidar-era; adjacent lithology, so it tests spatial rather than geological transfer |
| Urban OOD test | **R3 Oregon City** (existing) | 10 tiles, unchanged | Already built and QC'd; becomes evaluation only |
| Stretch, cross-lithology | **R4 Marion foothills** | +20-25 tiles | Western Cascades volcanics; `CALHNC2020` dominant; lowest null-confidence rate (40/1119). Add only after R1/R2 works |

Phase one: 85-90 rural tiles, ~140-160 km², roughly 15-20 GB.

## Storage Budget (100 GB LAZ)

The 100 GB refers to the execution machine, which has that capacity confirmed and
available. It is not a limit inherited from the current development machine, so
none of the sub-budgets below need to be trimmed to fit local free space.

| Allocation | Sub-budget | Notes |
|---|---|---|
| R1 Tillamook | ~20 GB | 55-60 tiles |
| R2 Buxton / Vernonia | ~10 GB | 25-30 tiles |
| Task 6 probe | ~6 GB | 8-10 co-located tiles per candidate project |
| R4 Marion (later) | ~10 GB | Stretch region |
| Dual-vintage diagnostic | ~6 GB | Optional: both 2007 and 2022 over the same ~15 tile footprint |
| Reserve | remainder | Headroom, not a spending target |

The 200-tile stretch is affordable: 200 rural tiles at ~190 MB average is
roughly 38 GB.

Two guardrails:

- Do not pass `--max-total-gb 100` on every call. That defeats the guard.
  Use the per-region sub-budgets above.
- Patches and NAIP cache sit outside this ceiling and are not free. ~90 tiles
  yields roughly 3000-6000 patches at 2-4 GB of npz. NAIP over every patch at
  0.6 m would approach 10 GB; the Task 10 stratified sample keeps it under 1 GB.

The dual-vintage diagnostic set must stay outside the train/test splits to avoid
the footprint leakage Task 4 guards against. It is a diagnostic set, not
training data.

## Architecture
flowchart TD
    A[regions.json registry\nbbox, role, candidate projects, budget] --> B[acquire_slido.py --region\nCONFIDENCE normalized + provenance stats]
    A --> C[select_tiles.py\nTNM enumerate, group by project,\ndedupe footprints, GB budget,\nnegative-only tile quota, probe mode]
    C --> D[download_tiles.py\nresume + --max-total-gb]
    D --> P[probe_tiles.py\nground coverage at 1.0/1.5/2.0 m\nper candidate project]
    P --> A
    B --> E[build_dataset.py --regions\nregion_id, lidar_project, lidar_year,\n3-state mask, spatial-block splits]
    D --> E
    E --> F[patches.csv + patches_qc.csv\nregion + label_quality columns]
    F --> G[fetch_naip_qc.py --sample\nnaip_year, lidar_year, year_gap]
    G --> H[qc_patch_viewer.py\nNAIP RGB/CIR + hillshade/slope/SLIDO\nyear-gap banner]
    F --> I[train_baseline.py\nR1 train/val, ignore-index loss]
    I --> J[evaluate.py\nR2 rural test, R3 urban OOD]

Region roles drive splits: R1 tiles split into `train`/`validation` by spatial
blocks with a buffer wider than one patch; R2 tiles all become `test_rural`; R3
all become `test_urban_ood`. No region ever spans two splits.

The mask is three-state: `0` negative, `1` positive, `255` ignore. Ignore
absorbs what should not be scored either way — polygons below the confidence
gate, polygons whose known event date postdates the LiDAR acquisition year, and
the buffer ring around positives. This enforces "exclude LiDAR-before-event
pairs" at pixel level instead of by dropping whole tiles.

## Task Breakdown

### Build-only implementation status (2026-07-29)

The local build phase is complete for Tasks 1-6 and 8-12. Task 7 and every
live-data or compute-heavy demo remain intentionally deferred to the execution
machine. "Build complete" means the implementation and synthetic/mocked tests
exist; it does not claim that network services, downloaded data, generated
patches, trained weights, or real-dataset acceptance criteria have been
validated.

| Task | Build status | Deferred execution / acceptance work |
|---|---|---|
| 1. Region candidate report | **Build complete** | Refresh live SLIDO/TNM/NAIP metadata and generate the five-region comparison plus R1 coverage report. |
| 2. Region registry | **Build complete** | No download is needed; pin rural project/cell-size decisions only after Task 6 probes. |
| 3. Region-aware SLIDO acquisition | **Build complete** | Fetch R1/R2 SLIDO GeoJSON and compare live source/confidence counts. |
| 4. Tile selection | **Build complete** | Run live TNM enumeration, co-located probe selection, and final budgeted selections. |
| 5. Budgeted download | **Build complete** | Download probe/full selections on the execution machine and demonstrate resume/budget behavior. |
| 6. Probe and pin decision | **Build complete** for diagnostics, vectorized TRI, and atomic registry pinning | Process real LAZ probes at 1.0/1.5/2.0 m, inspect overlays, choose project/cell size, and pin the evidence-based decision. No value was invented locally. |
| 7. Full R1/R2 selection and download | **Execution-only; deferred** | Select and download 80-90 rural tiles after Task 6 pinning, then verify footprint separation and storage totals. |
| 8. Multi-region dataset build | **Build complete** | Run real LAZ processing and the five-tile/full-build acceptance demonstrations. |
| 9. Spatial splits and verification | **Build complete** | Run the verifier against generated R1/R2/R3 manifests and confirm zero real-data violations. |
| 10. NAIP QC | **Build complete** | Fetch the stratified NAIP sample and perform visual QC/cache-size acceptance checks. |
| 11. Rural baseline training | **Build complete** | Train on the execution machine after QC and persist real weights, normalization, and validation metrics. |
| 12. Held-out/OOD evaluation | **Build complete** | Evaluate trained weights on R1 validation, R2 rural test, and R3 urban OOD; generate real overlays and comparison metrics. |

Post-review static validation passed locally: **76 unittest tests** (53 core and
23 Oregon), Python bytecode compilation for `oregon` and `tests`, all 11 affected
CLI `--help` smoke checks, and `git diff --check`. No live network requests,
downloads, real probes, LAZ dataset generation, NAIP fetches, model training, or
model evaluation were run during this build-only phase.

### Task 1: Region candidate report

Add `oregon/report_regions.py`. Per region: SLIDO counts grouped by normalized
`CONFIDENCE` and by `REF_ID_COD`; TNM LPC records grouped by project with tile
counts, summed bytes, and share of the AOI covered; NAIP available years;
projected GB per budget. Writes `oregon/regions/<name>_report.json` plus a
Markdown summary.

- **Tests:** confidence normalizer against the live dirty values
  (`High (=>30)`, `High (>30)`, `High (=<30)`, trailing-space variants);
  project grouping on synthetic TNM records.
- **Demo:** comparison table for all five candidates, plus per-project AOI
  coverage share for R1 — the input to the Task 6 pinning decision.

### Task 2: Region registry

Create `oregon/regions.json` with, per region: id, WGS84 bbox or polygon, role
(`train_val`, `test_rural`, `test_urban_ood`), tile budget, SLIDO output path,
and a `candidate_projects` list that stays a list until Task 6 pins it. Backfill
Oregon City against existing data. Add a loader with schema validation and a
`--region` resolver used by later stages.

- **Tests:** registry validation rejects unknown roles, overlapping bboxes,
  missing paths, and an empty candidate list.
- **Demo:** registry loads four regions with budgets summing to the intended
  tile count; Oregon City resolves to existing pilot data without re-downloading.

### Task 3: Region-aware SLIDO acquisition

Extend `acquire_slido.py` with `--region`, writing `slido_<region>.geojson` plus
a metadata sidecar, adding a normalized `confidence_class` property
(`high`/`moderate`/`low`/`unknown`), and recording per-source and
per-confidence counts. Raise `--max-features` per region — R1 needs well over
the current default of 1000, which would truncate silently today.

- **Tests:** normalization mapping; hard failure when feature count equals the cap.
- **Demo:** R1 and R2 fetched with source and confidence breakdowns matching
  Task 1, cap never hit.

### Task 4: Tile selection with footprint dedup and probe mode

Add `oregon/select_tiles.py`, replacing the download-selection half of
`build_manifest.py`. Filters to one project, deduplicates by footprint overlap
so no two selected tiles cover the same ground, ranks by intersecting
high+moderate polygon area, reserves a configurable share (default 25%) for
landslide-free hard-negative tiles, and enforces `--max-total-gb`. Adds
`--probe N --project <name>` to select co-located tiles per candidate project
for Task 6.

- **Tests:** dedup on synthetic overlapping footprints from two projects;
  budget enforcement; negative quota; probe returns co-located tiles.
- **Demo:** probe sets of 8-10 tiles per candidate project over the same
  footprints, plus a full 55-tile R1 selection projecting under 20 GB.

### Task 5: Budgeted download

Add `--max-total-gb` and per-region output directories to `download_tiles.py`,
keeping the existing size-match resume behavior, and write `download_log.csv`
with bytes, elapsed time, and status per tile.

- **Tests:** budget stop with a mocked session; resume skip on a size-matching
  file; partial-file cleanup on failure.
- **Demo:** download the probe sets, interrupt, rerun, show skips and a clean
  budget stop.

### Task 6: Probe, then pin cell size and project (decision gate)

Add `oregon/diagnostics/probe_tiles.py` reporting, per tile and per candidate
project, `ground_cell_fraction`, per-patch ground-fraction distribution,
void-run statistics, and DEM fragmentation at 1.0, 1.5, and 2.0 m — without
writing patches. Also vectorize `compute_tri` using `uniform_filter` on the DEM
and its square.

- **Tests:** TRI equivalence to the current `generic_filter` output within float
  tolerance, with a reported speedup.
- **Demo:** a table of how many patches survive `min_patch_ground_fraction` at
  each cell size under Coast Range canopy, plus overlays of one known
  deep-seated slide at 1.0 versus 2.0 m. Outcome is written back into
  `regions.json` as a pinned `lidar_project` and a recorded `cell_size`, with
  reasoning in the region report. If 2.0 m wins, this task also rebuilds Oregon
  City at 2.0 m.

### Task 7: Full selection and download for R1 + R2

Run Tasks 4 and 5 with the pinned project and final budgets. Verify zero
footprint overlap between regions and total GB within the sub-budgets.

- **Demo:** 80-90 rural tiles on disk with a download log and size summary.

### Task 8: Multi-region dataset build, 3-state masks, temporal exclusion

Extend `build_dataset.py` to accept multiple regions and add `region_id`,
`lidar_project`, `lidar_year`, `lidar_year_source` (LAZ header creation date
preferred, project-name parse as fallback), and `label_quality` columns.
Positives from `confidence_class` in {high, moderate}. Write `255` for
low/unknown confidence, for polygons whose known event year postdates the LiDAR
year, and for the existing negative buffer ring. Keep the
`DESCRIPTION='Landslide'` filter.

- **Tests:** synthetic polygons across all four confidence classes and
  pre/post-LiDAR dates asserting exact mask codes; a post-LiDAR polygon becomes
  neither positive nor negative.
- **Demo:** 5-tile build whose `dataset_summary.json` reports per-region counts,
  ignore-pixel fraction, and a nonzero temporally excluded polygon count.

### Task 9: Spatial-block splitting and leakage verification

Replace `assign_tile_splits`: project tile centroids to a metric CRS, bin into
configurable blocks (default 5 km), assign whole blocks to `train`/`validation`
for `train_val` regions, force test-role regions entirely to their split, and
drop tiles whose footprint falls within a buffer (default 500 m, wider than one
256 m patch) of a differently-assigned block. Add `verify_splits.py` asserting
no cross-split patch bounds fall within the buffer distance.

- **Tests:** synthetic tile grids verifying block integrity, buffer drops, and
  role forcing.
- **Demo:** full R1+R2+R3 build, then zero verifier violations alongside a
  split-count table.

### Task 10: NAIP QC on a stratified sample

Extend `fetch_naip_qc.py` with `--sample-per-region`, stratified across
`category` and `region_id`, adding `lidar_year`, `naip_year`, `year_gap`, and
`gap_flag` to `naip_manifest.csv`. Update `diagnostics/qc_patch_viewer.py` to
print LiDAR year, NAIP year, and gap on every page, warning when the gap exceeds
2 years so clear-cut and road differences are not read as terrain change. Add
`unmapped_landslide_suspected` and `reject_vintage_mismatch` QC statuses.

- **Tests:** gap computation and flagging; deterministic stratified sampling.
- **Demo:** ~150 stratified R1 patches cached at 1.0 m, cache under 1 GB, year
  gap and land-cover context visible on every panel.

### Task 11: LiDAR-only rural baseline with ignore-aware loss

Update `train_baseline.py` to ignore `255` in loss and all metrics, read
`region_id`, and refuse to train when the training split spans more than one
region role. Train on R1 `train`/`validation` with the 7 existing channels,
`--require-qc` once review is done, and persist `normalization.json` computed
from R1 training patches only.

- **Tests:** loss and Dice ignore masked pixels; run guard fails when a
  test-role region appears in training.
- **Demo:** `metrics.json` with Dice, IoU, precision, recall, and specificity on
  R1 validation, trained on rural terrain only, with ignore fraction reported.

### Task 12: Held-out and out-of-domain evaluation

Add `oregon/evaluate.py` that loads `best_model.pt` and `normalization.json` and
scores any dataset directory, split, and region without retraining, writing
per-region metrics and prediction overlays for the best and worst cases. Run on
R2 (rural geographic test) and R3 (Oregon City urban OOD).

- **Tests:** normalization is loaded rather than recomputed; per-region metric
  aggregation matches a hand-computed fixture.
- **Demo:** one table comparing R1 validation, R2 rural test, and R3 urban OOD,
  with R3 overlays showing whether errors track roads and development — the
  direct answer to whether the model learned morphology or metro artifacts.

## Deferred

- R4 Marion County as a fourth region (cross-lithology test).
- NAIP-only and LiDAR + NAIP channel experiments against the frozen R1/R2/R3
  splits.
- Optional dual-vintage diagnostic set (2007 vs 2022 over identical footprints)
  to measure how DEM vintage moves polygon alignment and predictions.

## Data Sources

- SLIDO 4.2, Oregon DOGAMI:
  `https://gis.dogami.oregon.gov/arcgis/rest/services/Public/SLIDO42/MapServer`
  (deposits layer 3). Partially funded by the FEMA Hazard Mitigation Grant Program.
- USGS 3DEP Lidar Point Cloud via TNM Access API:
  `https://tnmaccess.nationalmap.gov/api/v1/products`
- USGS NAIP Imagery ImageServer:
  `https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer`

Coverage and count figures in this plan were measured from live queries to these
services against sample bounding boxes. Task 1 replaces the sampled figures with
full per-AOI enumeration.
```

Two notes on it. The mermaid diagram now has a feedback edge (`probe_tiles.py` back into `regions.json`), which reflects that Task 6 writes the pinned project and cell size into the registry rather than just reporting. And the storage table assigns sub-budgets rather than the flat 100 GB, since a single global ceiling passed to every download call stops being a guard.