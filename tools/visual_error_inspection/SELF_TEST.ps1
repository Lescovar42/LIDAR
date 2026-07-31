$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

Push-Location $repoRoot
try {
    python .\oregon\tests\test_inspect_visual_errors.py
    if ($LASTEXITCODE -ne 0) {
        throw "Visual error inspection tests failed with exit code $LASTEXITCODE"
    }

    python .\oregon\tests\test_export_prediction_maps.py

    if ($LASTEXITCODE -ne 0) {
        throw "Visual error inspection tests failed with exit code $LASTEXITCODE"
    }

    python -m compileall -q `
      .\oregon\diagnostics\inspect_visual_errors.py `
      .\oregon\diagnostics\export_prediction_maps.py `
      .\oregon\tests\test_inspect_visual_errors.py `
      .\oregon\tests\test_export_prediction_maps.py

    if ($LASTEXITCODE -ne 0) {
        throw "Compile check failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
