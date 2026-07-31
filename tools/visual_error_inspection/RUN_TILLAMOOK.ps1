param(
    [string]$DatasetDir = ".\dataset_tillamook_probe_15m",
    [string]$Manifest = "patches_boundary_aware.csv",
    [string]$Checkpoint = ".\training_output_tillamook_15m_boundary\best_model.pt",
    [string]$Normalization = ".\training_output_tillamook_15m_boundary\normalization.json",
    [string]$OutDir = ".\evaluation_tillamook_15m_boundary_visual_errors",
    [double]$SelectedThreshold = 0.65,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$oregonDir = Join-Path $repoRoot "oregon"
$script = Join-Path $oregonDir "diagnostics\inspect_visual_errors.py"

if (-not (Test-Path $script)) {
    throw "Missing diagnostic script: $script"
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
      --selected-threshold $SelectedThreshold `
      --comparison-threshold 0.50 `
      --comparison-threshold 0.60 `
      --comparison-threshold $SelectedThreshold `
      --max-ignore-fraction 0.30 `
      --device $Device

    if ($LASTEXITCODE -ne 0) {
        throw "Visual error inspection failed with exit code $LASTEXITCODE"
    }

    $predictionScript = Join-Path $oregonDir "diagnostics\export_prediction_maps.py"
    $predictionOutDir = ".\evaluation_tillamook_15m_prediction_maps"

    python $predictionScript `
      --dataset-dir $DatasetDir `
      --manifest $Manifest `
      --checkpoint $Checkpoint `
      --normalization $Normalization `
      --outdir $predictionOutDir `
      --split validation `
      --threshold $SelectedThreshold `
      --max-ignore-fraction 0.30 `
      --professor-case-count 5 `
      --device $Device

    if ($LASTEXITCODE -ne 0) {
        throw "Prediction-map export failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "Open these first:"
    Write-Host "  $(Join-Path $oregonDir "$predictionOutDir\professor_predicted_vs_gt.png")"
    Write-Host "  $(Join-Path $oregonDir "$OutDir\professor_comparison.png")"
    Write-Host "  $(Join-Path $oregonDir "$OutDir\error_summary.md")"
    Write-Host ""
    Write-Host "Detailed cases:"
    Write-Host "  $(Join-Path $oregonDir "$OutDir\representative_cases")"
}
finally {
    Pop-Location
}
