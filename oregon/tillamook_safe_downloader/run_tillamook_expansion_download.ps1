param(
    [string]$RepoRoot = ".",
    [string]$SelectionCsv = ".\selection_tillamook_expansion_100\proposed_tillamook_expansion.csv",
    [string]$OutDir = ".\lidar_tiles\tillamook_expansion_100",
    [int]$ConcurrentFiles = 4,
    [int]$ConnectionsPerFile = 4,
    [int]$MaxPasses = 3,
    [string]$Aria2Path = "aria2c"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $RepoRoot).Path
Set-Location $RepoRoot

if (-not (Test-Path $SelectionCsv)) { throw "Selection CSV not found: $SelectionCsv" }
if ($ConcurrentFiles -lt 1 -or $ConcurrentFiles -gt 12) { throw "ConcurrentFiles must be 1..12" }
if ($ConnectionsPerFile -lt 1 -or $ConnectionsPerFile -gt 8) { throw "ConnectionsPerFile must be 1..8" }
if ($MaxPasses -lt 1 -or $MaxPasses -gt 10) { throw "MaxPasses must be 1..10" }

$aria = Get-Command $Aria2Path -ErrorAction SilentlyContinue
if (-not $aria) {
    throw "aria2c not found. Install aria2 or pass -Aria2Path <full path to aria2c.exe>."
}

$ToolRoot = Join-Path $RepoRoot "tillamook_safe_downloader"
$Prepare = Join-Path $ToolRoot "diagnostics\prepare_tillamook_download.py"
$Verify = Join-Path $ToolRoot "diagnostics\verify_tillamook_download.py"
$WorkDir = Join-Path $RepoRoot "download_tillamook_expansion_100_status"
$ResolvedOut = [IO.Path]::GetFullPath((Join-Path $RepoRoot $OutDir))
$ResolvedSelection = [IO.Path]::GetFullPath((Join-Path $RepoRoot $SelectionCsv))
New-Item -ItemType Directory -Force -Path $ResolvedOut, $WorkDir | Out-Null

Write-Host "[preflight] Build exact-size manifest and detect existing files"
python $Prepare --selection-csv $ResolvedSelection --repo-root $RepoRoot --outdir $ResolvedOut --workdir $WorkDir
if ($LASTEXITCODE -ne 0) { throw "Preflight failed" }

$pre = Get-Content (Join-Path $WorkDir "preflight_summary.json") | ConvertFrom-Json
$required = [int64]$pre.missing_bytes
$driveRoot = [IO.Path]::GetPathRoot($ResolvedOut)
$drive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$($driveRoot.TrimEnd('\'))'" -ErrorAction SilentlyContinue
if ($drive -and $drive.FreeSpace -lt [int64]($required * 1.05)) {
    throw "Insufficient free disk space. Need about $([math]::Round($required/1GB,2)) GiB plus headroom; free=$([math]::Round($drive.FreeSpace/1GB,2)) GiB."
}

if ([int]$pre.to_download -eq 0) {
    Write-Host "Nothing to download; all selected tiles are already complete."
} else {
    Write-Host "Missing selected tiles: $($pre.to_download); bytes: $([math]::Round($required/1GB,2)) GiB"
    Write-Host "aria2: $ConcurrentFiles concurrent files x $ConnectionsPerFile connections/file"
}

for ($pass = 1; $pass -le $MaxPasses; $pass++) {
    python $Prepare --selection-csv $ResolvedSelection --repo-root $RepoRoot --outdir $ResolvedOut --workdir $WorkDir | Out-Null
    $pre = Get-Content (Join-Path $WorkDir "preflight_summary.json") | ConvertFrom-Json
    if ([int]$pre.to_download -eq 0) { break }

    $inputFile = Join-Path $WorkDir "aria2_input.txt"
    $logFile = Join-Path $WorkDir "aria2_pass_$pass.log"
    Write-Host "`n[download pass $pass/$MaxPasses] $($pre.to_download) files remaining"

    & $aria.Source `
      --input-file=$inputFile `
      --dir=$ResolvedOut `
      --continue=true `
      --always-resume=true `
      --auto-file-renaming=false `
      --allow-overwrite=false `
      --max-concurrent-downloads=$ConcurrentFiles `
      --max-connection-per-server=$ConnectionsPerFile `
      --split=$ConnectionsPerFile `
      --min-split-size=8M `
      --max-tries=20 `
      --retry-wait=5 `
      --connect-timeout=30 `
      --timeout=90 `
      --max-file-not-found=5 `
      --file-allocation=none `
      --summary-interval=10 `
      --console-log-level=notice `
      --log=$logFile `
      --log-level=notice

    $ariaExit = $LASTEXITCODE
    Write-Host "aria2 exit code: $ariaExit"

    python $Verify --manifest-csv (Join-Path $WorkDir "expected_manifest.csv") --repo-root $RepoRoot --outdir $ResolvedOut --workdir $WorkDir
    $verifyExit = $LASTEXITCODE
    $summary = Get-Content (Join-Path $WorkDir "download_summary.json") | ConvertFrom-Json
    Write-Host "Verified: $($summary.ok)/$($summary.selected); partial=$($summary.partial); missing=$($summary.failed_or_missing)"
    if ($summary.complete) { break }
    if ($pass -lt $MaxPasses) {
        Write-Host "Retrying only incomplete tiles..."
        Start-Sleep -Seconds 10
    }
}

python $Verify --manifest-csv (Join-Path $WorkDir "expected_manifest.csv") --repo-root $RepoRoot --outdir $ResolvedOut --workdir $WorkDir
$finalExit = $LASTEXITCODE
$final = Get-Content (Join-Path $WorkDir "download_summary.json") | ConvertFrom-Json

Write-Host "`n--- Final verified status ---"
Write-Host "Exact-size OK: $($final.ok)/$($final.selected)"
Write-Host "Partial:       $($final.partial)"
Write-Host "Missing:       $($final.failed_or_missing)"
Write-Host "Logs/status:   $WorkDir"
Write-Host "Tile output:   $ResolvedOut"

if (-not $final.complete) {
    Write-Warning "Some tiles remain incomplete. Rerun the same command; aria2 will resume partial files."
    exit 2
}
exit 0
