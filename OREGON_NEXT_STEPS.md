# Oregon SLIDO + 3DEP: next steps

These files convert the existing proof-of-concept into a small, auditable pilot.
The old `train_mvp.py` can remain for reference, but use `build_dataset.py` and
`train_baseline.py` for the next run.

## 1. Place the files

Copy the bundle over the repository root. The important layout is:

```text
LIDAR/
├── requirements.txt
├── .gitignore
├── OREGON_NEXT_STEPS.md
└── oregon/
    ├── acquire_slido.py
    ├── slido_utils.py
    ├── discover_3dep.py              # keep your existing file
    ├── download_tiles.py             # keep your existing file
    ├── build_manifest.py
    ├── terrain_utils.py
    ├── build_dataset.py
    ├── train_baseline.py
    └── diagnostics/
        ├── visualize_dataset.py
        └── apply_qc.py
```

`build_manifest.py` no longer imports `archive/04_select_tile_subset.py`.

## 2. Set up the environment

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\requirements.txt
```

## 3. Acquire a compact, landslide-only SLIDO subset

Skip this when your existing GeoJSON is already correct and contains only
`DESCRIPTION = Landslide`.

```powershell
cd .\oregon
python .\acquire_slido.py `
  --bbox -122.90 45.35 -122.55 45.65 `
  --max-features 1000 `
  --out .\slido_deposits_oregon_city.geojson
```

The script also writes a `.metadata.json` sidecar recording the query and
retrieval time.

## 4. Rebuild the candidate manifest without downloading

```powershell
python .\build_manifest.py `
  --slido-geojson .\slido_deposits_oregon_city.geojson `
  --pipeline-dir . `
  --manifest-dir .\outputs\manifest_pilot `
  --max-tiles 10
```

Expected: most pairs may remain `uncertain` because TNM tile records often do
not expose acquisition dates. That is acceptable for a visual post-event pilot.

## 5. Download only ten uncertain candidate tiles

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

Do not remove `--max-tiles 10` until the complete pilot has passed visual QC.

## 6. Build reusable patches

```powershell
python .\build_dataset.py `
  --laz-dir .\lidar_tiles `
  --slido-geojson .\slido_deposits_oregon_city.geojson `
  --outdir .\dataset_pilot `
  --max-tiles 10 `
  --overwrite
```

The builder:

- reads the actual CRS from each LAZ header;
- keeps only SLIDO `DESCRIPTION = Landslide`;
- clips and rasterizes polygons with holes correctly;
- creates local relief, slope, aspect components, curvature, multidirectional
  hillshade, and TRI;
- drops tiny trace-positive patches by default;
- samples steep hard negatives;
- splits complete tiles into train/validation/test before saving patches;
- writes `patches.csv`, `channels.json`, `dataset_summary.json`, and compressed
  `.npz` patches.

## 7. Generate diverse QC pages

```powershell
python .\diagnostics\visualize_dataset.py `
  --dataset-dir .\dataset_pilot `
  --samples 30
```

Open `dataset_pilot/qc/qc_page_*.png` and fill the `qc_status` column in:

```text
dataset_pilot/qc/qc_review.csv
```

Recommended statuses:

```text
accept
accept_approximate_boundary
reject_misaligned
reject_not_visible
reject_engineered_landform
reject_bad_dem
```

Merge the decisions:

```powershell
python .\diagnostics\apply_qc.py --dataset-dir .\dataset_pilot
```

This creates `dataset_pilot/patches_qc.csv` without overwriting the original
manifest.

## 8. Train the first measurable baseline

For an initial smoke test using every generated patch:

```powershell
python .\train_baseline.py `
  --dataset-dir .\dataset_pilot `
  --outdir .\training_output_pilot `
  --epochs 10
```

After enough patches have been reviewed, train only accepted QC rows:

```powershell
python .\train_baseline.py `
  --dataset-dir .\dataset_pilot `
  --outdir .\training_output_qc `
  --epochs 20 `
  --require-qc
```

Outputs include `best_model.pt`, `normalization.json`, `history.csv`, and
`metrics.json` with Dice, IoU, precision, recall, and specificity.

## Stop/go criteria before downloading more Oregon tiles

Proceed beyond ten tiles only when:

1. `failed_tiles.csv` is empty or every failure is understood.
2. Most inspected positive patches show recognizable landslide morphology.
3. Labels are not systematically displaced from the terrain.
4. Fans and talus-colluvium are absent from positive masks.
5. Validation/test metrics run on different tiles, not random neighbouring
   patches.
6. Predictions are not simply following roads, quarries, or steep slopes.

The next milestone is a trusted dataset, not a larger neural network.
