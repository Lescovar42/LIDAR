# Tillamook class-weight ablation and training-diversity phase

This pack implements the controlled source-region experiment requested for `F:\LIDAR\oregon`.

It does **not**:

- change Mini U-Net;
- alter the ignore-mask policy;
- change train/validation rows between runs;
- use Buxton/Vernonia or Oregon City data;
- select thresholds on external labels;
- overwrite `training_output_tillamook_15m_boundary`.

## Repository finding

The inspected `train_baseline.py` has no `--pos-weight` CLI option. It always computes the training-pixel negative/positive ratio and uses that value in BCE. `apply_pos_weight_override.py` adds a narrow `auto|number` override plus run provenance without changing the architecture, optimizer, normalization, loss composition, or model-selection logic.

Because the old checkpoint does not record all run arguments, the publication-grade workflow trains a fresh auto-weight control and a `pos_weight=1.0` ablation into separate directories. The old baseline remains available for descriptive re-sweeping.

## Install

Place or extract this folder under `F:\LIDAR\oregon`, then run:

```powershell
Set-Location F:\LIDAR\oregon
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\lidar_class_weight_diversity_phase\install_phase_tools.ps1 -RepoRoot .
```

The installer:

1. copies the diagnostic tools into `oregon\diagnostics`;
2. creates `train_baseline.py.pre_pos_weight.bak`;
3. adds `--pos-weight` and provenance logging;
4. runs unit tests and Python syntax compilation.

Review the exact tracked change before training:

```powershell
git diff -- .\train_baseline.py .\diagnostics
```

## Run the complete controlled phase

```powershell
Set-Location F:\LIDAR\oregon
.\lidar_class_weight_diversity_phase\run_controlled_phase.ps1 -Device auto
```

Use `-Device cuda` to fail immediately rather than silently use CPU when CUDA is unavailable:

```powershell
.\lidar_class_weight_diversity_phase\run_controlled_phase.ps1 -Device cuda
```

The current checked-in manifest uses `region_id=legacy` for all Tillamook source rows, so the script's default `-TrainingRegion legacy` is intentional. This is an inherited identifier, not permission to include a legacy external region. Confirm it locally with:

```powershell
Import-Csv .\dataset_tillamook_probe_15m\patches_boundary_aware.csv |
    Select-Object -ExpandProperty region_id -Unique
```

## Exact experiment settings

Both fresh runs use:

```text
manifest: patches_boundary_aware.csv
training region: legacy
train/validation selection: identical manifest rows
epochs: 30
batch size: 8
learning rate: 0.001
optimizer: existing AdamW implementation
seed: 42
normalization: existing train-only computation
architecture: existing Mini U-Net
loss: existing BCE + soft Dice
```

Only `--pos-weight` differs:

```powershell
# Fresh auto-weight control
python .\train_baseline.py `
  --dataset-dir .\dataset_tillamook_probe_15m `
  --manifest patches_boundary_aware.csv `
  --outdir .\training_output_tillamook_15m_boundary_auto_control `
  --epochs 30 `
  --batch-size 8 `
  --learning-rate 0.001 `
  --num-workers 0 `
  --seed 42 `
  --training-region legacy `
  --device auto `
  --pos-weight auto

# Fixed-weight ablation
python .\train_baseline.py `
  --dataset-dir .\dataset_tillamook_probe_15m `
  --manifest patches_boundary_aware.csv `
  --outdir .\training_output_tillamook_15m_boundary_posw1 `
  --epochs 30 `
  --batch-size 8 `
  --learning-rate 0.001 `
  --num-workers 0 `
  --seed 42 `
  --training-region legacy `
  --device auto `
  --pos-weight 1.0
```

No `--require-qc` flag is introduced because the stated baseline used the boundary-aware manifest as supplied. Adding it now would change the selected rows and invalidate the one-variable ablation.

## Independent threshold sweeps

Each checkpoint is independently swept over `0.30, 0.35, ..., 0.90`:

```powershell
python .\diagnostics\evaluate_threshold_sweep.py `
  --dataset-dir .\dataset_tillamook_probe_15m `
  --manifest patches_boundary_aware.csv `
  --checkpoint .\training_output_tillamook_15m_boundary_auto_control\best_model.pt `
  --outdir .\evaluation_tillamook_15m_boundary_auto_control_validation `
  --split validation `
  --region legacy `
  --threshold-start 0.30 `
  --threshold-stop 0.90 `
  --threshold-step 0.05 `
  --review-count-each 6 `
  --device auto

python .\diagnostics\evaluate_threshold_sweep.py `
  --dataset-dir .\dataset_tillamook_probe_15m `
  --manifest patches_boundary_aware.csv `
  --checkpoint .\training_output_tillamook_15m_boundary_posw1\best_model.pt `
  --outdir .\evaluation_tillamook_15m_boundary_posw1_validation `
  --split validation `
  --region legacy `
  --threshold-start 0.30 `
  --threshold-stop 0.90 `
  --threshold-step 0.05 `
  --review-count-each 6 `
  --device auto
```

Each evaluation writes:

- `threshold_sweep.csv`;
- `best_threshold.json`;
- `per_patch_metrics.csv`;
- `dataset_fingerprint.json`;
- `comparison_images\*.png` with terrain context, GT, prediction, and FP/FN errors;
- `error_review_candidates.csv`.

The comparison script refuses to compare the two runs when validation manifest hashes or row/path hashes differ.

## Training-diversity audit

```powershell
python .\diagnostics\audit_training_diversity.py `
  --dataset-dir .\dataset_tillamook_probe_15m `
  --manifest patches_boundary_aware.csv `
  --outdir .\audit_tillamook_15m_training_diversity
```

Outputs include:

- per-tile composition;
- positive/negative/category and coverage summaries;
- positive and ignored-fraction distributions;
- boundary presence;
- polygon-key repetition;
- spatial grid coverage;
- overlapping-window redundancy screening;
- terrain-channel per-patch distributions from NPZ files;
- source train/validation support screening;
- provenance summary;
- stratified hard-negative review candidates;
- `training_data_diversity_report.md` with measured findings separated from hypotheses.

## Semantic error categories

The evaluator intentionally does not guess roads, cuts, drainage, ridges, forest disturbance, or inventory incompleteness from pixel metrics. Open the generated comparison images, then edit `manual_error_category` in each `error_review_candidates.csv` without changing the labels or masks.

Allowed categories:

```text
steep_non_landslide_slope
drainage_or_valley_side
ridge_or_convex_terrain
rough_natural_terrain
road_or_engineered_cut
forest_management_disturbance
boundary_mismatch
incomplete_or_uncertain_inventory_label
other
```

Then summarize:

```powershell
python .\diagnostics\summarize_error_categories.py `
  --review-csv .\evaluation_tillamook_15m_boundary_posw1_validation\error_review_candidates.csv `
  --outdir .\evaluation_tillamook_15m_boundary_posw1_validation\error_category_summary
```

This review is for data-expansion diagnosis, not retrospective relabeling to improve metrics.

## Baseline-versus-ablation table and decision

```powershell
python .\diagnostics\summarize_ablation.py `
  --auto-eval .\evaluation_tillamook_15m_boundary_auto_control_validation `
  --posw1-eval .\evaluation_tillamook_15m_boundary_posw1_validation `
  --outdir .\comparison_tillamook_15m_pos_weight_ablation `
  --max-recall-loss 0.05 `
  --dice-tolerance 0.01
```

The output contains measured deltas and an explicit rule-based recommendation among:

- use `pos_weight=1.0`;
- test an intermediate fixed weight;
- retain auto while improving diversity;
- retain the weight but improve diversity first.

The tolerances are declared parameters, not hidden post-hoc choices. Change them only with a written methodological justification before inspecting the ablation result.

## Optional descriptive re-sweep of the old baseline

This preserves the existing training output:

```powershell
.\lidar_class_weight_diversity_phase\resweep_existing_baseline.ps1 -Device auto
```

Use this to verify the reported old baseline threshold/metrics. Do not treat it as the strongest paired comparison because its checkpoint does not record every training argument.

## Current manifest-only measured audit

The included `measured_audit` folder was generated from the repository's checked-in `patches_boundary_aware.csv` without NPZ files. It measured:

- train: 513 patches, 13 tiles, 272 positive patches, 23 unique positive polygon keys;
- validation: 54 patches, 3 tiles, 14 positive patches, 3 unique positive polygon keys;
- zero train/validation positive polygon-key overlap;
- 67.28% of positive training patches concentrated in the top four tiles;
- 57.77% of positive polygon-patch memberships concentrated in the top four polygon keys;
- 322 same-polygon positive patch pairs with at least 50% overlap of the smaller window;
- validation has only mixed and low-coverage positives, with no near-full or full-positive patches;
- acquisition provenance currently uses 2020 based on project-owner attestation; the repository evidence file explicitly says a machine-readable USGS citation is still outstanding, while `A22` supplies only a non-authoritative 2022 name hint.

These are measured composition/provenance facts. They do not by themselves prove that the model's false positives are caused by redundancy or missing semantic hard negatives.

## Git tracking

```powershell
Set-Location F:\LIDAR\oregon
git switch -c experiment/class-weight-diversity
git status
git add .\train_baseline.py .\diagnostics .\lidar_class_weight_diversity_phase
git commit -m "Add controlled class-weight ablation and diversity audit"
```

Do not commit large checkpoints, NPZ patch files, or generated image directories unless the repository's data policy explicitly requires them. Commit code, small CSV/JSON summaries, run configurations, and the final Markdown reports.
