param(
    [string]$RepoRoot = "."
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $RepoRoot).Path
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$DiagnosticsDir = Join-Path $RepoRoot "diagnostics"
$TestsDir = Join-Path $RepoRoot "tests"
New-Item -ItemType Directory -Force $DiagnosticsDir | Out-Null
New-Item -ItemType Directory -Force $TestsDir | Out-Null

Copy-Item `
    (Join-Path $PackageRoot "diagnostics\select_tillamook_expansion.py") `
    (Join-Path $DiagnosticsDir "select_tillamook_expansion.py") `
    -Force

Copy-Item `
    (Join-Path $PackageRoot "tests\test_select_tillamook_expansion.py") `
    (Join-Path $TestsDir "test_select_tillamook_expansion.py") `
    -Force

Write-Host "Installed diagnostics\select_tillamook_expansion.py"
Write-Host "Running selector tests..."

$OldPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = "$DiagnosticsDir;$RepoRoot"
    python (Join-Path $TestsDir "test_select_tillamook_expansion.py")
    if ($LASTEXITCODE -ne 0) { throw "Tillamook expansion selector tests failed" }

    python -m py_compile (Join-Path $DiagnosticsDir "select_tillamook_expansion.py")
    if ($LASTEXITCODE -ne 0) { throw "Selector syntax compilation failed" }
}
finally {
    $env:PYTHONPATH = $OldPythonPath
}

Write-Host "Tillamook expansion selector installed successfully."
