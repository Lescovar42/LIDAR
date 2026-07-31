param([string]$RepoRoot = "")

$ErrorActionPreference = "Stop"
$PackageDir = $PSScriptRoot

if (-not $RepoRoot) {
    $candidates = @(
        (Get-Location).Path,
        (Split-Path $PackageDir -Parent),
        $PackageDir
    ) | Select-Object -Unique

    foreach ($candidate in $candidates) {
        if ((Test-Path (Join-Path $candidate "oregon\train_baseline.py")) -or
            (Test-Path (Join-Path $candidate "train_baseline.py"))) {
            $RepoRoot = $candidate
            break
        }
    }
}

if (-not $RepoRoot) {
    throw "Could not locate the repository. Run: .\INSTALL.ps1 -RepoRoot F:\LIDAR"
}

python (Join-Path $PackageDir "apply_boundary_aware_fix.py") --repo-root $RepoRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
