param(
    [Parameter(Mandatory = $false)]
    [string]$RepoRoot = "."
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $RepoRoot).Path
$Trainer = Join-Path $RepoRoot "train_baseline.py"
$DiagnosticsDir = Join-Path $RepoRoot "diagnostics"

if (-not (Test-Path $Trainer)) {
    throw "train_baseline.py not found at $Trainer. Run this from F:\LIDAR\oregon or pass -RepoRoot."
}
New-Item -ItemType Directory -Force -Path $DiagnosticsDir | Out-Null
Copy-Item (Join-Path $PSScriptRoot "diagnostics\*.py") $DiagnosticsDir -Force

python (Join-Path $PSScriptRoot "apply_pos_weight_override.py") --file $Trainer
if ($LASTEXITCODE -ne 0) { throw "Failed to patch train_baseline.py" }

Push-Location $RepoRoot
try {
    python (Join-Path $PSScriptRoot "tests\test_phase_common.py")
    if ($LASTEXITCODE -ne 0) { throw "phase_common tests failed" }
    python (Join-Path $PSScriptRoot "tests\test_apply_pos_weight_override.py")
    if ($LASTEXITCODE -ne 0) { throw "patcher tests failed" }
    python (Join-Path $PSScriptRoot "tests\test_train_baseline_pos_weight.py")
    if ($LASTEXITCODE -ne 0) { throw "patched trainer tests failed" }
    python -m compileall -q .\train_baseline.py .\diagnostics\phase_common.py .\diagnostics\evaluate_threshold_sweep.py .\diagnostics\audit_training_diversity.py .\diagnostics\summarize_ablation.py .\diagnostics\summarize_error_categories.py
    if ($LASTEXITCODE -ne 0) { throw "Python syntax compilation failed" }
}
finally {
    Pop-Location
}

Write-Host "Installed phase tools and passed tests."
Write-Host "Backup retained beside trainer: train_baseline.py.pre_pos_weight.bak"
