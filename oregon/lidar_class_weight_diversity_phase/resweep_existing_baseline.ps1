param(
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [string]$TrainingRegion = "legacy"
)
$ErrorActionPreference = "Stop"
Set-Location "F:\LIDAR\oregon"
$Outdir = ".\evaluation_tillamook_15m_boundary_existing_baseline_resweep"
if (Test-Path $Outdir) { throw "Refusing to overwrite $Outdir" }
python .\diagnostics\evaluate_threshold_sweep.py `
    --dataset-dir .\dataset_tillamook_probe_15m `
    --manifest patches_boundary_aware.csv `
    --checkpoint .\training_output_tillamook_15m_boundary\best_model.pt `
    --outdir $Outdir `
    --split validation `
    --region $TrainingRegion `
    --threshold-start 0.30 `
    --threshold-stop 0.90 `
    --threshold-step 0.05 `
    --review-count-each 6 `
    --device $Device
if ($LASTEXITCODE -ne 0) { throw "Existing baseline resweep failed" }
