param(
    [string]$Device = "cuda",
    [int]$Epochs = 50,
    [int]$BatchSize = 2,
    [int]$NumWorkers = 0,
    [switch]$Amp
)

$ErrorActionPreference = "Stop"

$Dataset = ".\dataset_tillamook_100_binary_15m_split641620"
$Root = ".\phase4_tillamook_binary_2x2"
$Trainer = ".\train_tillamook_binary_feature_depth.py"

if (-not (Test-Path $Trainer)) {
    throw "Missing trainer: $Trainer"
}
if (-not (Test-Path "$Dataset\phase3_qc.json")) {
    throw "Missing Phase 3 QC report."
}

$qc = Get-Content "$Dataset\phase3_qc.json" | ConvertFrom-Json
if ($qc.status -ne "PASS") {
    throw "Phase 3 QC is not PASS."
}

New-Item -ItemType Directory -Force -Path $Root | Out-Null

$common = @(
    "--dataset-dir", $Dataset,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--learning-rate", "0.001",
    "--seed", "42",
    "--pos-weight", "auto",
    "--device", $Device,
    "--threshold-min", "0.30",
    "--threshold-max", "0.90",
    "--threshold-step", "0.05"
)

if ($Amp) {
    $common += "--amp"
}

function Run-Experiment {
    param(
        [string]$Name,
        [string]$Architecture,
        [string]$Features
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "$Name : architecture=$Architecture features=$Features"
    Write-Host "============================================================"

    python $Trainer `
        @common `
        --architecture $Architecture `
        --features $Features `
        --outdir "$Root\$Name"

    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Run-Experiment "shallow_7ch" "shallow" "7ch"
Run-Experiment "shallow_3ch" "shallow" "3ch"
Run-Experiment "deep_7ch"    "deep"    "7ch"
Run-Experiment "deep_3ch"    "deep"    "3ch"

python .\compare_tillamook_binary_2x2.py --root $Root

if ($LASTEXITCODE -ne 0) {
    throw "Comparison failed."
}

Write-Host ""
Write-Host "PHASE 4 COMPLETE."
Write-Host "Internal test split has NOT been evaluated."
Write-Host "Review: $Root\comparison.md"
