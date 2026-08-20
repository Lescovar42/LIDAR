param(
    [string]$RepoRoot = ".",
    [int]$TargetTiles = 100,
    [double]$HardNegativeFraction = 0.40,
    [double]$ValidationBufferM = 500.0,
    [Nullable[double]]$MaxTotalGB = $null
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $RepoRoot).Path

function First-ExistingPath([string[]]$Candidates, [switch]$Required) {
    foreach ($candidate in $Candidates) {
        $p = Join-Path $RepoRoot $candidate
        if (Test-Path $p) { return (Resolve-Path $p).Path }
    }
    if ($Required) {
        throw "Required input not found. Tried: $($Candidates -join ', ')"
    }
    return $null
}

$Selector = First-ExistingPath @(
    "diagnostics\select_tillamook_expansion.py"
) -Required

$Tiles = First-ExistingPath @(
    "regions\tillamook_tnm.json",
    "tillamook_tnm.json"
) -Required

$Polygons = First-ExistingPath @(
    "slido_tillamook.geojson",
    "regions\slido_tillamook.geojson"
) -Required

$Manifest = First-ExistingPath @(
    "dataset_tillamook_probe_15m\patches_boundary_aware.csv"
) -Required

$Downloaded = First-ExistingPath @(
    "actual_100tile_attempt.csv"
)

$Exclusions = First-ExistingPath @(
    "dataset_tillamook_probe_15m\failed_tiles.csv",
    "regions\failed_tiles.csv",
    "failed_tiles.csv"
)

$ProbeMetrics = First-ExistingPath @(
    "regions\tillamook_a22_probe_ground_020.json",
    "tillamook_a22_probe_ground_020.json",
    "regions\tillamook_a22_probe_metrics.json",
    "tillamook_a22_probe_metrics.json"
)

$OutDir = Join-Path $RepoRoot "selection_tillamook_expansion_$TargetTiles"

$argsList = @(
    $Selector,
    "--tiles", $Tiles,
    "--polygons", $Polygons,
    "--frozen-manifest", $Manifest,
    "--target-tiles", "$TargetTiles",
    "--hard-negative-fraction", "$HardNegativeFraction",
    "--validation-buffer-m", "$ValidationBufferM",
    "--outdir", $OutDir
)

if ($Downloaded) {
    $argsList += @("--downloaded-csv", $Downloaded)
    Write-Host "Downloaded inventory: $Downloaded"
} else {
    Write-Warning "actual_100tile_attempt.csv not found; local-file reuse bonus disabled."
}

if ($Exclusions) {
    $argsList += @("--exclusions-csv", $Exclusions)
    Write-Host "Explicit exclusions: $Exclusions"
} else {
    Write-Warning "failed_tiles.csv not found; explicit split-buffer/ground exclusions are not supplied."
}

if ($ProbeMetrics) {
    $argsList += @("--probe-metrics", $ProbeMetrics)
    Write-Host "Probe metrics: $ProbeMetrics"
} else {
    Write-Warning "Probe metrics not found; known probe-quality bonus disabled."
}

if ($null -ne $MaxTotalGB) {
    $argsList += @("--max-total-gb", "$MaxTotalGB")
}

Write-Host "Selecting Tillamook source-region expansion only; no downloads will be started."
python @argsList
if ($LASTEXITCODE -ne 0) { throw "Tillamook expansion selection failed" }

Write-Host ""
Write-Host "Selection complete. Review before downloading:"
Write-Host "  $OutDir\selection_summary.md"
Write-Host "  $OutDir\proposed_tillamook_expansion.csv"
Write-Host "  $OutDir\excluded_tiles.csv"
Write-Host ""
Get-Content (Join-Path $OutDir "selection_summary.md")
