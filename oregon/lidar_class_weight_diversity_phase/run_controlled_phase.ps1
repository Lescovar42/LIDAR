param(
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [string]$TrainingRegion = "legacy"
)

$ErrorActionPreference = "Stop"
Set-Location "F:\LIDAR\oregon"

$DatasetDir = ".\dataset_tillamook_probe_15m"
$Manifest = "patches_boundary_aware.csv"

# The pre-existing training_output_tillamook_15m_boundary is never written by this script.
$AutoTrain = ".\training_output_tillamook_15m_boundary_auto_control"
$PosW1Train = ".\training_output_tillamook_15m_boundary_posw1"
$AutoEval = ".\evaluation_tillamook_15m_boundary_auto_control_validation"
$PosW1Eval = ".\evaluation_tillamook_15m_boundary_posw1_validation"
$AuditOut = ".\audit_tillamook_15m_training_diversity"
$CompareOut = ".\comparison_tillamook_15m_pos_weight_ablation"

$ProtectedOutputs = @($AutoTrain, $PosW1Train, $AutoEval, $PosW1Eval, $AuditOut, $CompareOut)
foreach ($Path in $ProtectedOutputs) {
    if (Test-Path $Path) {
        throw "Refusing to overwrite existing output: $Path. Rename it or move it before rerunning."
    }
}

Write-Host "[1/7] Source-region diversity audit"
python .\diagnostics\audit_training_diversity.py `
    --dataset-dir $DatasetDir `
    --manifest $Manifest `
    --outdir $AuditOut
if ($LASTEXITCODE -ne 0) { throw "Diversity audit failed" }

$CommonTrainingArgs = @(
    "--dataset-dir", $DatasetDir,
    "--manifest", $Manifest,
    "--epochs", "30",
    "--batch-size", "8",
    "--learning-rate", "0.001",
    "--num-workers", "0",
    "--seed", "42",
    "--training-region", $TrainingRegion,
    "--device", $Device
)

Write-Host "[2/7] Fresh auto-weight control (old baseline preserved)"
& python .\train_baseline.py @CommonTrainingArgs `
    --outdir $AutoTrain `
    --pos-weight auto
if ($LASTEXITCODE -ne 0) { throw "Auto-weight control training failed" }

Write-Host "[3/7] Fixed pos_weight=1.0 ablation"
& python .\train_baseline.py @CommonTrainingArgs `
    --outdir $PosW1Train `
    --pos-weight 1.0
if ($LASTEXITCODE -ne 0) { throw "pos_weight=1.0 training failed" }

$CommonEvalArgs = @(
    "--dataset-dir", $DatasetDir,
    "--manifest", $Manifest,
    "--split", "validation",
    "--region", $TrainingRegion,
    "--threshold-start", "0.30",
    "--threshold-stop", "0.90",
    "--threshold-step", "0.05",
    "--review-count-each", "6",
    "--device", $Device
)

Write-Host "[4/7] Independent auto-control threshold sweep and maps"
& python .\diagnostics\evaluate_threshold_sweep.py @CommonEvalArgs `
    --checkpoint (Join-Path $AutoTrain "best_model.pt") `
    --outdir $AutoEval
if ($LASTEXITCODE -ne 0) { throw "Auto-control evaluation failed" }

Write-Host "[5/7] Independent pos_weight=1.0 threshold sweep and maps"
& python .\diagnostics\evaluate_threshold_sweep.py @CommonEvalArgs `
    --checkpoint (Join-Path $PosW1Train "best_model.pt") `
    --outdir $PosW1Eval
if ($LASTEXITCODE -ne 0) { throw "pos_weight=1.0 evaluation failed" }

Write-Host "[6/7] Controlled summary"
python .\diagnostics\summarize_ablation.py `
    --auto-eval $AutoEval `
    --posw1-eval $PosW1Eval `
    --outdir $CompareOut `
    --max-recall-loss 0.05 `
    --dice-tolerance 0.01
if ($LASTEXITCODE -ne 0) { throw "Ablation summary failed" }

Write-Host "[7/7] Initial semantic-error summaries (will be unreviewed until CSV categories are filled)"
python .\diagnostics\summarize_error_categories.py `
    --review-csv (Join-Path $AutoEval "error_review_candidates.csv") `
    --outdir (Join-Path $AutoEval "error_category_summary")
if ($LASTEXITCODE -ne 0) { throw "Auto error summary failed" }
python .\diagnostics\summarize_error_categories.py `
    --review-csv (Join-Path $PosW1Eval "error_review_candidates.csv") `
    --outdir (Join-Path $PosW1Eval "error_category_summary")
if ($LASTEXITCODE -ne 0) { throw "pos_weight=1.0 error summary failed" }

Write-Host "Completed without using external test regions."
Write-Host "Main comparison: $CompareOut\baseline_vs_posw1.md"
Write-Host "Diversity report: $AuditOut\training_data_diversity_report.md"
Write-Host "Review images: $AutoEval\comparison_images and $PosW1Eval\comparison_images"
