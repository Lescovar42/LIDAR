param([string]$OregonDir = "")

$ErrorActionPreference = "Stop"
if (-not $OregonDir) {
    $packageParent = Split-Path $PSScriptRoot -Parent
    $candidatePaths = @(
        (Get-Location).Path,
        $PSScriptRoot,
        $packageParent,
        (Join-Path (Get-Location).Path "oregon"),
        (Join-Path $packageParent "oregon"),
        (Join-Path (Split-Path $packageParent -Parent) "oregon")
    ) | Select-Object -Unique

    foreach ($candidate in $candidatePaths) {
        if (Test-Path (Join-Path $candidate "train_baseline.py")) {
            $OregonDir = $candidate
            break
        }
    }
}

if (-not $OregonDir) {
    throw "Could not locate train_baseline.py. Pass -OregonDir F:\LIDAR\oregon."
}

$train = Join-Path $OregonDir "train_baseline.py"
$backup = "$train.before_boundary_aware_fix"
if (Test-Path $backup) {
    Copy-Item $backup $train -Force
    Write-Host "Restored $train"
}
else {
    Write-Host "No trainer backup found; nothing restored."
}

$tool = Join-Path $OregonDir "diagnostics\prepare_boundary_aware_manifest.py"
if (Test-Path $tool) {
    Remove-Item $tool
    Write-Host "Removed $tool"
}
