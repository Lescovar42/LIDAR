param (
    [Parameter(Mandatory=$true)]
    [string]$InputFile,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputFile
)

if (-not (Test-Path $InputFile)) {
    Write-Host "Error: Input file '$InputFile' does not exist." -ForegroundColor Red
    exit 1
}

if (-not $OutputFile) {
    $OutputFile = [System.IO.Path]::GetFileNameWithoutExtension($InputFile) + "_compressed.mp4"
}

# Target size in MB. We aim for 98MB to leave room for container overhead and avoid going over 100MB.
$TargetSizeMB = 98 
$TargetSizeKilobits = $TargetSizeMB * 8192
$AudioBitrateKbps = 128

# Get duration using ffprobe
Write-Host "Analyzing video duration..."
$DurationStr = (ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $InputFile)

try {
    $Duration = [math]::Round([double]$DurationStr)
} catch {
    Write-Host "Error: Could not determine video duration. Ensure ffprobe is installed and in your PATH." -ForegroundColor Red
    exit 1
}

if ($Duration -le 0) {
    Write-Host "Error: Invalid video duration ($Duration)." -ForegroundColor Red
    exit 1
}

# Calculate target video bitrate
$TargetTotalBitrateKbps = [math]::Floor($TargetSizeKilobits / $Duration)
$VideoBitrateKbps = $TargetTotalBitrateKbps - $AudioBitrateKbps

if ($VideoBitrateKbps -le 0) {
    Write-Host "Error: Video is too long to compress to 100MB with a 128k audio track (Calculated video bitrate is <= 0)." -ForegroundColor Red
    exit 1
}

Write-Host "Input file: $InputFile" -ForegroundColor Cyan
Write-Host "Output file: $OutputFile" -ForegroundColor Cyan
Write-Host "Duration: $Duration seconds" -ForegroundColor Cyan
Write-Host "Target Video Bitrate: ${VideoBitrateKbps}k" -ForegroundColor Cyan
Write-Host "Target Audio Bitrate: ${AudioBitrateKbps}k" -ForegroundColor Cyan
Write-Host ""

# Pass 1
Write-Host "Starting Pass 1 (Video Analysis)..." -ForegroundColor Yellow
$pass1_cmd = "ffmpeg -y -i `"$InputFile`" -c:v libx264 -b:v `"${VideoBitrateKbps}k`" -pass 1 -an -f mp4 NUL"
Invoke-Expression $pass1_cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during Pass 1." -ForegroundColor Red
    exit 1
}

# Pass 2
Write-Host "Starting Pass 2 (Encoding)..." -ForegroundColor Yellow
$pass2_cmd = "ffmpeg -y -i `"$InputFile`" -c:v libx264 -b:v `"${VideoBitrateKbps}k`" -pass 2 -c:a aac -b:a `"${AudioBitrateKbps}k`" `"$OutputFile`""
Invoke-Expression $pass2_cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during Pass 2." -ForegroundColor Red
    exit 1
}

# Cleanup pass log files
if (Test-Path "ffmpeg2pass-0.log") { Remove-Item "ffmpeg2pass-0.log" }
if (Test-Path "ffmpeg2pass-0.log.mbtree") { Remove-Item "ffmpeg2pass-0.log.mbtree" }

Write-Host "Compression complete! Saved to $OutputFile" -ForegroundColor Green
