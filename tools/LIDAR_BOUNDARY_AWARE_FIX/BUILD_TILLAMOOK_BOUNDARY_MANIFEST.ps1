param(
    [string]$OregonDir = "",
    [int]$MaxNearFullPerLandslide = 4,
    [int]$MaxFullPerLandslide = 2,
    [int]$Seed = 42
)

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
        if (Test-Path (Join-Path $candidate "diagnostics\prepare_boundary_aware_manifest.py")) {
            $OregonDir = $candidate
            break
        }
    }
}

if (-not $OregonDir) {
    throw "Could not locate the Oregon project. Run INSTALL.ps1 first, or pass -OregonDir F:\LIDAR\oregon."
}

Push-Location $OregonDir
try {
    python .\diagnostics\prepare_boundary_aware_manifest.py `
      --dataset-dir .\dataset_tillamook_probe_15m `
      --manifest .\patches.csv `
      --slido .\slido_tillamook.geojson `
      --max-near-full-per-landslide $MaxNearFullPerLandslide `
      --max-full-per-landslide $MaxFullPerLandslide `
      --seed $Seed
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host ""
    Write-Host "Selected manifest counts:" -ForegroundColor Cyan
    Import-Csv .\dataset_tillamook_probe_15m\patches_boundary_aware.csv |
      Group-Object split, coverage_class |
      Select-Object Name, Count |
      Format-Table -AutoSize
}
finally {
    Pop-Location
}
