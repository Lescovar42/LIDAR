[CmdletBinding()]
param(
    [string]$Python = "python",
    [int]$ProbeCount = 8,
    [double]$MaxTotalGB = 5.0,
    [double]$OverlapThreshold = 0.8,
    [string]$A22Project = "USGS_LPC_OR_WesternWildfires_A22",
    [string]$LegacyProject = "USGS_LPC_legacy",
    [switch]$SkipSelection,
    [switch]$SkipDownload,
    [switch]$SkipProbe
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$OregonDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $OregonDir
try {
    $TnmRecords = Join-Path $OregonDir "regions\tillamook_tnm.json"
    $Selection = Join-Path $OregonDir "regions\tillamook_probe_selection.json"
    $DownloadRoot = Join-Path $OregonDir "probe_lidar"
    $RegionFolder = "tillamook_probe"
    $CombinedLog = Join-Path $OregonDir "regions\tillamook_probe_download_log.csv"
    $Metrics = Join-Path $OregonDir "regions\tillamook_probe_metrics.json"

    if (-not (Test-Path $TnmRecords -PathType Leaf)) {
        throw "Missing $TnmRecords. Run report_regions.py --region R1 --refresh first."
    }
    if ($ProbeCount -lt 1) {
        throw "ProbeCount must be at least 1."
    }
    if ($MaxTotalGB -le 0) {
        throw "MaxTotalGB must be positive."
    }
    if ($OverlapThreshold -le 0 -or $OverlapThreshold -gt 1) {
        throw "OverlapThreshold must be in (0, 1]."
    }

    if (-not $SkipSelection) {
        Write-Host "`n[1/3] Selecting $ProbeCount co-located Tillamook footprints per project..."
        & $Python ".\select_tiles.py" `
            --tiles $TnmRecords `
            --project $A22Project `
            --project $LegacyProject `
            --probe $ProbeCount `
            --overlap-threshold $OverlapThreshold `
            --max-total-gb $MaxTotalGB `
            --output $Selection
        if ($LASTEXITCODE -ne 0) {
            throw "Probe selection failed with exit code $LASTEXITCODE. Do not lower the overlap threshold automatically; inspect project overlap first."
        }
    }

    if (-not (Test-Path $Selection -PathType Leaf)) {
        throw "Missing probe selection: $Selection"
    }

    if (-not $SkipDownload) {
        Write-Host "`n[2/3] Downloading grouped probe selection..."
        & $Python ".\download_tiles.py" `
            --subset $Selection `
            --outdir $DownloadRoot `
            --region $RegionFolder `
            --max-total-gb $MaxTotalGB `
            --download-log $CombinedLog
        if ($LASTEXITCODE -ne 0) {
            throw "Probe download finished with failures. Review $CombinedLog and rerun; completed files will be skipped after size verification."
        }
    }

    $A22Directory = Join-Path (Join-Path $DownloadRoot $RegionFolder) $A22Project
    $LegacyDirectory = Join-Path (Join-Path $DownloadRoot $RegionFolder) $LegacyProject
    if (-not (Test-Path $A22Directory -PathType Container)) {
        throw "Missing downloaded A22 directory: $A22Directory"
    }
    if (-not (Test-Path $LegacyDirectory -PathType Container)) {
        throw "Missing downloaded legacy directory: $LegacyDirectory"
    }

    if (-not $SkipProbe) {
        Write-Host "`n[3/3] Measuring 1.0, 1.5, and 2.0 m ground coverage without writing patches..."
        & $Python ".\diagnostics\probe_tiles.py" `
            --project "$A22Project=$A22Directory" `
            --project "$LegacyProject=$LegacyDirectory" `
            --cell-size 1.0 `
            --cell-size 1.5 `
            --cell-size 2.0 `
            --output $Metrics
        if ($LASTEXITCODE -ne 0) {
            throw "Ground-coverage probe reported failures. Review $Metrics before pinning a project or cell size."
        }
    }

    Write-Host "`nTillamook probe stage complete."
    Write-Host "Selection: $Selection"
    Write-Host "Download log: $CombinedLog"
    Write-Host "Probe metrics: $Metrics"
    Write-Host "No project or cell size was pinned automatically."
}
finally {
    Pop-Location
}
