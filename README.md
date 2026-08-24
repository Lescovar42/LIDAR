# Oregon LiDAR Landslide Segmentation

An auditable research pipeline for building LiDAR terrain datasets from the
Oregon Statewide Landslide Information Database (SLIDO), reviewing them with
NAIP imagery, and training a baseline semantic-segmentation model.

The primary study design is multi-region:

- **R1 Tillamook Coast Range:** rural train/validation data.
- **R2 Buxton / Vernonia:** geographically held-out rural test data.
- **R3 Oregon City:** existing urban out-of-domain test data.
- **R4 Marion foothills:** optional cross-lithology extension.

The model uses seven LiDAR-derived terrain channels. NAIP is currently a QC
context layer, not a model input. Region boundaries and acquisition provenance
are kept in [`oregon/regions.json`](oregon/regions.json).

## Status

The implementation and synthetic/static checks for the multi-region workflow
are in place. Live service queries, tile downloads, real LAZ processing, NAIP
fetches, training, and final evaluation must be run on a machine with the
required network access, storage, and compute. Do not interpret checked-in
generated outputs as a completed scientific validation.

The immediate goal is a small, visually audited pilot before expanding the
download budget. See [`OREGON_NEXT_STEPS.md`](OREGON_NEXT_STEPS.md) for the
long-form pilot procedure and [`Rural_Dataset_Tasks.md`](Rural_Dataset_Tasks.md)
for the rural diversification design.

## Requirements

- Windows PowerShell is the documented environment; Python 3.12 is the pinned
	development version.
- Network access is needed for SLIDO, USGS TNM 3DEP, and NAIP services.
- Disk space is needed for LAZ tiles, processed patches, and optional imagery
	cache. The rural plan budgets up to 100 GB for LAZ on the execution machine.
- An NVIDIA GPU is optional. The default dependency file contains the CUDA
	12.6 PyTorch wheel; CPU-only installations are supported by replacing that
	wheel as described in [`requirements.txt`](requirements.txt).
- The optional QC patch viewer requires a Python installation with Tkinter.

## Setup

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

Verify the selected PyTorch device:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

All Oregon commands below are run from `oregon`:

```powershell
cd .\oregon
```

## Quick Start: Ten-Tile Pilot

This is the smallest end-to-end path. It is intended to expose data, label,
and vintage problems before a larger rural download.

### 1. Acquire SLIDO labels

Skip this step when the existing GeoJSON is known to be correct and contains
only `DESCRIPTION = Landslide` features.

```powershell
python .\acquire_slido.py `
	--bbox -122.90 45.35 -122.55 45.65 `
	--max-features 1000 `
	--out .\slido_deposits_oregon_city.geojson
```

The command writes a metadata sidecar next to the GeoJSON.

### 2. Build a manifest without downloading

```powershell
python .\build_manifest.py `
	--slido-geojson .\slido_deposits_oregon_city.geojson `
	--pipeline-dir . `
	--manifest-dir .\outputs\manifest_pilot `
	--max-tiles 10
```

### 3. Download the pilot tiles

```powershell
python .\build_manifest.py `
	--slido-geojson .\slido_deposits_oregon_city.geojson `
	--pipeline-dir . `
	--manifest-dir .\outputs\manifest_pilot `
	--outdir .\lidar_tiles `
	--download-statuses uncertain `
	--max-tiles 10 `
	--download
```

Keep the tile limit until visual QC passes.

### 4. Build patches and review them

```powershell
python .\build_dataset.py `
	--laz-dir .\lidar_tiles `
	--slido-geojson .\slido_deposits_oregon_city.geojson `
	--outdir .\dataset_pilot `
	--max-tiles 10 `
	--overwrite
```

The dataset includes terrain arrays, `patches.csv`, `channels.json`, and a
dataset summary. Generate QC pages with the diagnostic tools under
`oregon/diagnostics/`, then review the generated `qc_review.csv` before using
`patches_qc.csv` for training.

### 5. Train a baseline

```powershell
python .\train_baseline.py `
	--dataset-dir .\dataset_pilot `
	--outdir .\training_output_pilot `
	--epochs 10
```

Training writes a checkpoint, normalization parameters, training history, and
segmentation metrics. Training automatically selects CUDA when available;
pass `--device cpu` or `--device cuda` to override it.

## Rural Multi-Region Workflow

Use this path after the pilot and after the candidate LiDAR project has been
selected using a co-located probe. The stages are:

1. **Inspect the registry:** validate a region with
	 `python .\region_registry.py --region R1`.
2. **Acquire region labels:** use `acquire_slido.py` with each region's bbox.
3. **Discover TNM coverage:** run `discover_3dep.py` and retain the persisted
	 records under `oregon/regions/`.
4. **Compare candidates:** use `report_regions.py` before committing to a
	 LiDAR project.
5. **Select footprint-safe tiles:** use `select_tiles.py` with one project,
	 `--max-tiles`, `--max-total-gb`, and a deliberate negative quota. Use
	 `--probe N` to compare co-located tiles across projects.
6. **Download with guards:** use `download_tiles.py`; keep the region name and
	 storage budget in the command and preserve its download log.
7. **Build the dataset:** use `build_dataset.py` with region-aware inputs.
	 Splits must be spatially separated and must not share a region across roles.
8. **Run QC:** fetch a stratified NAIP sample with `fetch_naip_qc.py` and use
	 the diagnostic viewer. Record the LiDAR/NAIP year gap in review decisions.
9. **Train and evaluate:** train on R1, evaluate on R1 validation, R2
	 `test_rural`, and R3 `test_urban_ood` with `evaluate.py`.
10. **Verify leakage:** run `verify_splits.py` before trusting metrics.

Do not use R2 or R3 data for training. Keep the optional R4 region out of the
first rural experiment until the R1/R2 workflow is stable.

## Important Data Rules

- SLIDO polygons are labels, not proof that a landslide is visible in every
	LiDAR acquisition.
- Known LiDAR-before-event pairs are excluded at mask level; uncertain dates
	are retained only when the project design allows them.
- Low-confidence and unknown-confidence polygons become ignore pixels, not
	negative examples.
- Overlapping 3DEP projects must be footprint-deduplicated so the same ground
	cannot appear in multiple splits.
- Pure-negative terrain is required for useful training. Roads, quarries,
	clear-cuts, river bars, and development should be sampled intentionally and
	reviewed as hard negatives.
- A high validation score from Oregon City alone is not evidence of rural
	generalization.

## Repository Map

| Path | Purpose |
|---|---|
| `oregon/regions.json` | Region roles, bboxes, budgets, projects, and provenance |
| `oregon/acquire_slido.py` | Fetch and record SLIDO label subsets |
| `oregon/discover_3dep.py` | Query USGS TNM LiDAR coverage |
| `oregon/select_tiles.py` | Select budgeted, footprint-safe tiles |
| `oregon/download_tiles.py` | Resumable downloads with integrity and size guards |
| `oregon/build_manifest.py` | Pilot manifest construction and download orchestration |
| `oregon/build_dataset.py` | Rasterize labels and create terrain patches |
| `oregon/fetch_naip_qc.py` | Retrieve NAIP context for QC samples |
| `oregon/train_baseline.py` | Train the ignore-aware MiniUNet baseline |
| `oregon/evaluate.py` | Evaluate a saved checkpoint by split and region |
| `oregon/verify_splits.py` | Check spatial split leakage |
| `oregon/diagnostics/` | Visualization and QC tooling |
| `tests/` and `oregon/tests/` | Unit and mocked integration tests |
| `archive/` | Historical experiments and superseded scripts |

## Validation

Run the static test suite from the repository root:

```powershell
python -m unittest discover -s .\tests
python -m unittest discover -s .\oregon\tests
```

Before expanding the dataset, confirm that download failures are understood,
positive masks align with terrain, QC rejects engineered landforms, and test
metrics are computed on geographically separate tiles. The research milestone
is a trusted dataset, not simply a larger model or more downloaded tiles.

## License and Data Sources

This repository contains research code and project documentation. SLIDO,
USGS 3DEP, and NAIP data remain subject to their respective source terms and
should be retrieved from the official services at runtime.

