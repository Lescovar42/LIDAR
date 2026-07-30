[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Aria2 = "aria2c",
    [int]$ProbeCount = 24,
    [double]$MaxTotalGB = 8.0,
    [double]$NegativeQuota = 0.25,
    [int]$MaxAttempts = 3,
    [string]$A22Project = "USGS_LPC_OR_WesternWildfires_A22",
    [string]$A22Directory = "",
    [switch]$SkipSelection,
    [switch]$SkipDownload,
    [switch]$SkipVerification,
    [switch]$SkipProbe
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$OregonDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $OregonDir

try {
    $TnmRecords = Join-Path $OregonDir "regions\tillamook_tnm.json"
    $Slido = Join-Path $OregonDir "slido_tillamook.geojson"
    $Selection = Join-Path $OregonDir "regions\tillamook_a22_probe_selection.json"
    $Aria2Input = Join-Path $OregonDir "regions\tillamook_a22_probe_aria2.txt"
    $Exclusions = Join-Path $OregonDir "regions\tillamook_a22_probe_exclusions.txt"
    $Verification = Join-Path $OregonDir "regions\tillamook_a22_probe_verification.json"
    $Metrics = Join-Path $OregonDir "regions\tillamook_a22_probe_metrics.json"
    $DownloadLog = Join-Path $OregonDir "regions\tillamook_a22_probe_download_log.csv"

    if ([string]::IsNullOrWhiteSpace($A22Directory)) {
        # Reuse the seven valid A22 files already downloaded for the matched probe.
        $A22Directory = Join-Path $OregonDir `
            "probe_lidar\tillamook_probe\USGS_LPC_OR_WesternWildfires_A22"
    } elseif (-not [System.IO.Path]::IsPathRooted($A22Directory)) {
        $A22Directory = Join-Path $OregonDir $A22Directory
    }

    if (-not (Test-Path $TnmRecords -PathType Leaf)) {
        throw "Missing $TnmRecords. Run report_regions.py --region R1 --refresh first."
    }
    if (-not (Test-Path $Slido -PathType Leaf)) {
        throw "Missing $Slido."
    }
    if ($ProbeCount -lt 1) {
        throw "ProbeCount must be at least 1."
    }
    if ($MaxTotalGB -le 0) {
        throw "MaxTotalGB must be positive."
    }
    if ($NegativeQuota -lt 0 -or $NegativeQuota -gt 1) {
        throw "NegativeQuota must be between 0 and 1."
    }
    if ($MaxAttempts -lt 1) {
        throw "MaxAttempts must be at least 1."
    }

    New-Item -ItemType Directory -Force -Path $A22Directory | Out-Null

    $attempt = 1
    while ($true) {
        Write-Host "`nA22 representative probe attempt $attempt/$MaxAttempts"

        if (-not $SkipSelection) {
            Write-Host "`n[1/4] Selecting $ProbeCount spatially diverse A22 tiles..."
            & $Python ".\diagnostics\select_a22_probe.py" `
                --tiles $TnmRecords `
                --slido $Slido `
                --project $A22Project `
                --count $ProbeCount `
                --negative-quota $NegativeQuota `
                --max-total-gb $MaxTotalGB `
                --exclude-file $Exclusions `
                --output $Selection `
                --aria2-input $Aria2Input `
                --download-dir $A22Directory
            if ($LASTEXITCODE -ne 0) {
                throw "A22 selection failed with exit code $LASTEXITCODE."
            }
        }

        if (-not (Test-Path $Selection -PathType Leaf)) {
            throw "Missing selection: $Selection"
        }

        if (-not $SkipDownload) {
            Write-Host "`n[2/4] Downloading selected A22 tiles..."
            $aria2Command = Get-Command $Aria2 -ErrorAction SilentlyContinue
            if ($null -ne $aria2Command) {
                & $aria2Command.Source `
                    "--input-file=$Aria2Input" `
                    "--continue=true" `
                    "--max-concurrent-downloads=2" `
                    "--split=8" `
                    "--max-connection-per-server=8" `
                    "--min-split-size=8M" `
                    "--file-allocation=none" `
                    "--auto-file-renaming=false" `
                    "--allow-overwrite=false" `
                    "--check-integrity=true" `
                    "--summary-interval=5" `
                    "--console-log-level=notice"
                if ($LASTEXITCODE -ne 0) {
                    throw "aria2 download failed with exit code $LASTEXITCODE."
                }
            } else {
                Write-Warning "aria2c not found; using the slower Python downloader."
                & $Python ".\download_tiles.py" `
                    --subset $Selection `
                    --outdir $A22Directory `
                    --max-total-gb $MaxTotalGB `
                    --download-log $DownloadLog
                if ($LASTEXITCODE -ne 0) {
                    throw "Python download failed with exit code $LASTEXITCODE."
                }
            }
        }

        if ($SkipVerification) {
            break
        }

        Write-Host "`n[3/4] Verifying HTTP size, CRS, and class-2 ground..."
        & $Python ".\diagnostics\verify_a22_probe.py" `
            --selection $Selection `
            --laz-dir $A22Directory `
            --output $Verification `
            --exclusions-file $Exclusions
        $verificationExit = $LASTEXITCODE

        if ($verificationExit -eq 0) {
            break
        }

        if (-not (Test-Path $Verification -PathType Leaf)) {
            throw "Verification failed without writing $Verification."
        }

        $verificationData = Get-Content $Verification -Raw | ConvertFrom-Json
        $structuralCount = @($verificationData.structural_exclusions_added).Count
        if (
            $structuralCount -gt 0 `
            -and -not $SkipSelection `
            -and -not $SkipDownload `
            -and $attempt -lt $MaxAttempts
        ) {
            Write-Warning (
                "$structuralCount structurally unusable tile(s) were added to " +
                "$Exclusions. Reselecting replacements."
            )
            $attempt += 1
            continue
        }

        throw (
            "A22 verification failed. Review $Verification. " +
            "Network/size failures are not automatically converted into scientific exclusions."
        )
    }

    if (-not $SkipProbe) {
        Write-Host "`n[4/4] Measuring A22 ground, patch, and SLIDO retention metrics..."
        & $Python ".\diagnostics\probe_a22_labels.py" `
            --selection $Selection `
            --laz-dir $A22Directory `
            --slido $Slido `
            --project $A22Project `
            --lidar-year 2020 `
            --cell-size 1.0 `
            --cell-size 1.5 `
            --cell-size 2.0 `
            --patch-size 256 `
            --stride 128 `
            --min-patch-ground-fraction 0.5 `
            --output $Metrics
        if ($LASTEXITCODE -ne 0) {
            throw "A22 probe reported failures. Review $Metrics before choosing a resolution."
        }
    }

    Write-Host "`nA22-only Tillamook probe complete."
    Write-Host "Selection: $Selection"
    Write-Host "Verification: $Verification"
    Write-Host "Metrics: $Metrics"
    Write-Host "No cell size was pinned automatically."
}
finally {
    Pop-Location
}
