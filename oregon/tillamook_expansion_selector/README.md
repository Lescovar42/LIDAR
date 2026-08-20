# Tillamook 50–100 Tile Expansion Selector

This package implements the next phase after the class-weight ablation and manual validation-error review.

It **does not download or train anything**. It produces a reviewable source-region selection manifest first.

## Method encoded

The selector deliberately separates what can be measured **before download** from hypotheses that need terrain imagery.

Measured pre-download criteria:

- frozen train/validation tile identity;
- 500 m guard around frozen validation tile footprints;
- A22-only source-region candidate filtering;
- high/moderate SLIDO polygon intersection;
- number of high/moderate polygons not already represented by the frozen training tiles;
- spatial novelty relative to existing training and already-selected expansion tiles;
- clean hard-negative status only when a tile has no SLIDO deposit intersection at all;
- known 2 m probe results when available;
- reuse of complete already-downloaded LAZ files when they remain methodologically eligible;
- explicit known exclusions such as split-buffer failures and no-ground tiles.

Not inferred before download:

- rough/dissected natural terrain;
- steep non-landslide slope;
- ridge/convex terrain;
- drainage/valley side;
- road/engineered cut;
- forest-management disturbance.

Those semantic classes were identified during error review as collection priorities, but unseen TNM metadata cannot prove that a candidate belongs to one of them. After preprocessing, semantic hard-negative screening should target those classes.

## Default composition

For a 100-tile expansion:

- 60 positive-diversity tiles;
- 40 clean hard-negative tiles.

The 40% hard-negative quota reflects the current false-positive problem and error-review findings. It is a sampling target, not a claim that 40% is universally optimal.

## Install

Extract this package inside `F:\LIDAR\oregon`, then:

```powershell
Set-Location F:\LIDAR\oregon

.\tillamook_expansion_selector\install_tillamook_expansion_selector.ps1 `
  -RepoRoot .
```

Expected test result: **6 tests, OK**.

## Run the 100-tile proposal

```powershell
.\tillamook_expansion_selector\run_tillamook_expansion_selection.ps1 `
  -RepoRoot . `
  -TargetTiles 100 `
  -HardNegativeFraction 0.40 `
  -ValidationBufferM 500
```

This expects the existing project files, primarily:

- `regions\tillamook_tnm.json`
- `slido_tillamook.geojson`
- `dataset_tillamook_probe_15m\patches_boundary_aware.csv`
- `actual_100tile_attempt.csv` if present
- `failed_tiles.csv` if present
- `tillamook_a22_probe_ground_020.json` if present

The runner searches common repository locations for the optional inputs.

## Outputs

Created under `selection_tillamook_expansion_100\`:

- `selection_summary.md`
- `proposed_tillamook_expansion.csv`
- `proposed_tillamook_expansion.json`
- `candidate_scores.csv`
- `excluded_tiles.csv`
- `download_urls.txt`
- `selection_provenance.json`

**Do not start the downloader until `proposed_tillamook_expansion.csv` has been reviewed.**

`download_urls.txt` contains only selected files not already represented by `actual_100tile_attempt.csv`.

## Optional storage cap

Example with a 30 GB cap:

```powershell
.\tillamook_expansion_selector\run_tillamook_expansion_selection.ps1 `
  -RepoRoot . `
  -TargetTiles 100 `
  -HardNegativeFraction 0.40 `
  -ValidationBufferM 500 `
  -MaxTotalGB 30
```

## Research guardrails preserved

- Validation rows are not changed.
- Candidate tiles near frozen validation are excluded.
- Buxton/Vernonia and Oregon City are never loaded.
- No target-region normalization, threshold tuning, or feature selection occurs.
- Existing source training tiles are not counted as expansion.
- Low/unknown-only inventory overlap is not mislabeled as a clean negative.
- No architecture or loss changes are made.
