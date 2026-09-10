<#
.SYNOPSIS
    Registers the TickerLake scheduled tasks in Windows Task Scheduler.

.DESCRIPTION
    Creates two tasks:

      TickerLake-Daily    weekdays at 17:30 local, running the full pipeline.
                          Options markets close at 16:15 ET and Yahoo needs a
                          little longer to settle end-of-day chains, so this
                          leaves a comfortable margin after the 16:00 close.

      TickerLake-Compact  Sundays at 03:00 local, merging the week's small
                          per-symbol files into consolidated partitions.

    Both run whether or not you are logged in, wake the machine if it is
    asleep, and restart on failure. Run this script from an elevated PowerShell.

.PARAMETER DailyTime
    Local time for the daily run. Default 17:30.

.PARAMETER TimeZoneNote
    The schedule is in *local* time. If you are not on US Eastern, adjust
    -DailyTime so it lands after 16:15 ET.

.EXAMPLE
    # From an elevated PowerShell:
    .\scripts\install_scheduled_tasks.ps1
    .\scripts\install_scheduled_tasks.ps1 -DailyTime "18:00"
#>
[CmdletBinding()]
param(
    [string]$DailyTime = "17:30",
    [string]$CompactTime = "03:00"
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunScript = Join-Path $ProjectRoot 'scripts\run_daily.ps1'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $RunScript)) { Write-Error "Missing $RunScript"; exit 2 }
if (-not (Test-Path $Python)) { Write-Error "Missing venv at $Python"; exit 2 }

$IsAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $IsAdmin) {
    Write-Warning "Not running as Administrator. Task registration will likely fail."
    Write-Warning "Re-run from an elevated PowerShell prompt."
}

# --- Daily collection -------------------------------------------------------

$DailyAction = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$RunScript`"" `
    -WorkingDirectory $ProjectRoot

$DailyTrigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At $DailyTime

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -WakeToRun `
    -StartWhenAvailable `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 30) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName 'TickerLake-Daily' `
    -Action $DailyAction `
    -Trigger $DailyTrigger `
    -Settings $Settings `
    -Description 'TickerLake daily market data collection (OHLCV, options chains, filings, macro, news).' `
    -Force | Out-Null

Write-Output "Registered TickerLake-Daily (weekdays at $DailyTime)."

# --- Weekly compaction ------------------------------------------------------

$CompactAction = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument '-m tickerlake.cli compact' `
    -WorkingDirectory $ProjectRoot

$CompactTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At $CompactTime

Register-ScheduledTask `
    -TaskName 'TickerLake-Compact' `
    -Action $CompactAction `
    -Trigger $CompactTrigger `
    -Settings $Settings `
    -Description 'TickerLake weekly Parquet compaction.' `
    -Force | Out-Null

Write-Output "Registered TickerLake-Compact (Sundays at $CompactTime)."
Write-Output ""
Write-Output "Verify with:   Get-ScheduledTask -TaskName 'TickerLake-*'"
Write-Output "Run once now:  Start-ScheduledTask -TaskName 'TickerLake-Daily'"
Write-Output "Remove:        Unregister-ScheduledTask -TaskName 'TickerLake-Daily' -Confirm:`$false"
