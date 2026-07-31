param(
    [string]$DatasetDir = ".\dataset_tillamook_probe_15m",
    [string]$Manifest = "patches_boundary_aware.csv",
    [string]$Checkpoint = ".\training_output_tillamook_15m_boundary\best_model.pt",
    [string]$Normalization = ".\training_output_tillamook_15m_boundary\normalization.json",
    [string]$OutDir = ".\evaluation_tillamook_15m_prediction_maps",
    [double]$Threshold = 0.65,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$oregonDir = Join-Path $repoRoot "oregon"
$script = Join-Path $oregonDir "diagnostics\export_prediction_maps.py"

if (-not (Test-Path $script)) {
    throw "Missing prediction-map exporter: $script"
}

Push-Location $oregonDir
try {
    python $script `
      --dataset-dir $DatasetDir `
      --manifest $Manifest `
      --checkpoint $Checkpoint `
      --normalization $Normalization `
      --outdir $OutDir `
      --split validation `
      --threshold $Threshold `
      --max-ignore-fraction 0.30 `
      --professor-case-count 5 `
      --device $Device

    if ($LASTEXITCODE -ne 0) {
        throw "Prediction-map export failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "Open this first:"
    Write-Host "  $(Join-Path $oregonDir "$OutDir\professor_predicted_vs_gt.png")"
    Write-Host ""
    Write-Host "Stitched validation tile maps:"
    Write-Host "  $(Join-Path $oregonDir "$OutDir\stitched_tile_maps")"
}
finally {
    Pop-Location
}
