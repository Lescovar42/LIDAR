[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Region,
    [string]$Python = "python",
    [double]$NegativeQuota = 0.25,
    [string]$Registry = ".\regions.json",
    [string]$DownloadRoot = ".\lidar_tiles"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$OregonDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $OregonDir
try {
    if (-not (Test-Path $Registry -PathType Leaf)) {
        throw "Missing registry: $Registry"
    }
    if ($NegativeQuota -lt 0 -or $NegativeQuota -gt 1) {
        throw "NegativeQuota must be between 0 and 1."
    }

    $ConfigJson = & $Python -c @"
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from region_registry import load_registry, resolve_region, resolve_path
registry_path = Path(r'$Registry')
registry = load_registry(registry_path)
region = resolve_region(r'$Region', registry)
required = ('lidar_project', 'cell_size')
missing = [key for key in required if region.get(key) in (None, '')]
if missing:
    raise SystemExit(f"Region {region['id']} is not pinned: missing {', '.join(missing)}")
if region.get('existing_data'):
    raise SystemExit(f"Region {region['id']} uses existing data and should not be downloaded")
if int(region.get('tile_budget', 0)) <= 0 or float(region.get('storage_budget_gb', 0)) <= 0:
    raise SystemExit(f"Region {region['id']} has no positive tile/storage budget")
print(json.dumps({
    'id': region['id'],
    'slug': region['slug'],
    'project': region['lidar_project'],
    'cell_size': region['cell_size'],
    'tile_budget': int(region['tile_budget']),
    'storage_budget_gb': float(region['storage_budget_gb']),
    'tnm_records': str(resolve_path(region, 'tnm_records', registry_path)),
    'slido_output': str(resolve_path(region, 'slido_output', registry_path)),
}))
"@
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve a pinned download configuration for region $Region."
    }
    $Config = $ConfigJson | ConvertFrom-Json

    if (-not (Test-Path $Config.tnm_records -PathType Leaf)) {
        throw "Missing TNM records: $($Config.tnm_records)"
    }
    if (-not (Test-Path $Config.slido_output -PathType Leaf)) {
        throw "Missing SLIDO polygons: $($Config.slido_output)"
    }

    $Selection = Join-Path $OregonDir ("regions\{0}_selected_tiles.json" -f $Config.slug)
    $Log = Join-Path $OregonDir ("regions\{0}_download_log.csv" -f $Config.slug)

    Write-Host "Region: $($Config.id) / $($Config.slug)"
    Write-Host "Pinned project: $($Config.project)"
    Write-Host "Pinned cell size: $($Config.cell_size) m (not changed by this downloader)"
    Write-Host "Tile budget: $($Config.tile_budget)"
    Write-Host "Storage budget: $($Config.storage_budget_gb) GB"

    & $Python ".\select_tiles.py" `
        --tiles $Config.tnm_records `
        --polygons $Config.slido_output `
        --project $Config.project `
        --max-tiles $Config.tile_budget `
        --negative-quota $NegativeQuota `
        --max-total-gb $Config.storage_budget_gb `
        --output $Selection
    if ($LASTEXITCODE -ne 0) {
        throw "Tile selection failed. The runner will not substitute another project or change the budget."
    }

    & $Python ".\download_tiles.py" `
        --subset $Selection `
        --outdir $DownloadRoot `
        --region $Config.slug `
        --max-total-gb $Config.storage_budget_gb `
        --download-log $Log
    if ($LASTEXITCODE -ne 0) {
        throw "Download finished with failures. Review $Log and rerun to resume completed files."
    }

    Write-Host "`nRegion download complete."
    Write-Host "Selection: $Selection"
    Write-Host "Download directory: $(Join-Path $DownloadRoot $Config.slug)"
    Write-Host "Download log: $Log"
}
finally {
    Pop-Location
}
