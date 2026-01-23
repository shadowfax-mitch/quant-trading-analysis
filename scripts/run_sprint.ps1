param(
  [string]$Python = "python",
  [string]$Runner = "src/run_sprint.py",
  [string]$OutRoot = "runs"
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $OutRoot $timestamp
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

Write-Host "Run: $timestamp"
Write-Host "Runner: $Runner"
Write-Host "Output: $runDir"

$env:SPRINT_RUN_DIR = $runDir

if (Test-Path $Runner) {
  & $Python $Runner
} else {
  Write-Host "Runner not found. Update -Runner or add src/run_sprint.py."
}
