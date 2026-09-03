param(
    [string]$Slido = ".\slido_deposits_oregon_city.geojson",
    [string]$LazDir = ".\lidar_tiles",
    [string]$DatasetDir = ".\dataset_pilot",
    [int]$MaxTiles = 10
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Building patch dataset..."
python .\build_dataset.py `
  --laz-dir $LazDir `
  --slido-geojson $Slido `
  --outdir $DatasetDir `
  --max-tiles $MaxTiles `
  --overwrite

Write-Host "[2/3] Creating visual-QC pages..."
python .\diagnostics\visualize_dataset.py `
  --dataset-dir $DatasetDir `
  --samples 30

Write-Host "[3/3] Done. Review: $DatasetDir\qc\qc_page_*.png"
Write-Host "Fill qc_review.csv, then run diagnostics\apply_qc.py."
