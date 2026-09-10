<#
.SYNOPSIS
    Daily TickerLake collection run. Intended for Windows Task Scheduler.

.DESCRIPTION
    Activates the project virtualenv and runs the full pipeline, writing a
    transcript alongside the lake's own logs. Exit code is non-zero if the run
    failed, which is what lets Task Scheduler's "last run result" column and any
    retry policy actually mean something.

.PARAMETER RunDate
    Optional YYYY-MM-DD override. Defaults to today.

.PARAMETER Stages
    Optional comma-separated subset of stages, e.g. "universe,ohlcv".

.EXAMPLE
    .\scripts\run_daily.ps1
    .\scripts\run_daily.ps1 -Stages "universe,ohlcv" -RunDate 2026-09-09
#>
[CmdletBinding()]
param(
    [string]$RunDate,
    [string]$Stages
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$LogDir = Join-Path $ProjectRoot 'data\_logs'

if (-not (Test-Path $Python)) {
    Write-Error "Virtualenv not found at $Python. Run: uv venv; uv pip install -e ."
    exit 2
}
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Transcript = Join-Path $LogDir "scheduled_run_$Stamp.log"

$Arguments = @('-m', 'tickerlake.cli', 'run')
if ($RunDate) { $Arguments += @('--date', $RunDate) }
if ($Stages) { $Arguments += @('--stages', $Stages) }

Start-Transcript -Path $Transcript | Out-Null
try {
    Write-Output "TickerLake scheduled run starting $(Get-Date -Format o)"
    Write-Output "project: $ProjectRoot"
    Write-Output "command: $Python $($Arguments -join ' ')"

    Push-Location $ProjectRoot
    & $Python @Arguments
    $ExitCode = $LASTEXITCODE
    Pop-Location

    Write-Output "TickerLake run finished with exit code $ExitCode at $(Get-Date -Format o)"
}
catch {
    Write-Output "TickerLake run threw: $_"
    $ExitCode = 1
}
finally {
    Stop-Transcript | Out-Null
}

# Prune old transcripts so the scheduler does not slowly fill the disk.
Get-ChildItem -Path $LogDir -Filter 'scheduled_run_*.log' -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-90) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $ExitCode
