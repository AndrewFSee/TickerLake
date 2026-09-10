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

    Both are registered with an S4U principal so they run whether or not anyone
    is logged on, wake the machine if it is asleep, and restart on failure. Run
    this script from an elevated PowerShell.

    The S4U part matters: the default principal is LogonType Interactive, which
    runs the task ONLY while the user is logged on and silently skips it from
    the lock screen after a sign-out. For a collector capturing perishable data
    that gap would go unnoticed until it was permanent.

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

# Principal. Without this, Register-ScheduledTask defaults to LogonType
# Interactive, which means "run ONLY while this user is logged on" - the task
# silently does not fire from the lock screen after a sign-out. For a collector
# whose whole point is capturing perishable data every weekday, that is a
# failure mode you would not notice until the gap was permanent.
#
# S4U ("service for user") runs the task whether or not anyone is logged on and
# does not require storing a password. It needs the "Log on as a batch job"
# right, which administrators hold by default; if registration fails on a
# locked-down machine, fall back to Interactive and accept the caveat.
$UserId = "$env:USERDOMAIN\$env:USERNAME"
try {
    $Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType S4U -RunLevel Limited
    $LogonNote = 'runs whether or not you are logged on (S4U)'
}
catch {
    Write-Warning "S4U principal unavailable ($($_.Exception.Message)); falling back to Interactive."
    Write-Warning "The task will then run ONLY while $UserId is logged on."
    $Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel Limited
    $LogonNote = 'runs ONLY while you are logged on (Interactive)'
}

Register-ScheduledTask `
    -TaskName 'TickerLake-Daily' `
    -Action $DailyAction `
    -Trigger $DailyTrigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description 'TickerLake daily market data collection (OHLCV, options chains, filings, macro, news).' `
    -Force | Out-Null

Write-Output "Registered TickerLake-Daily (weekdays at $DailyTime) - $LogonNote."

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
    -Principal $Principal `
    -Description 'TickerLake weekly Parquet compaction.' `
    -Force | Out-Null

Write-Output "Registered TickerLake-Compact (Sundays at $CompactTime) - $LogonNote."
Write-Output ""
Write-Output "--- verification ---"
Get-ScheduledTask -TaskName 'TickerLake-*' | ForEach-Object {
    $info = $_ | Get-ScheduledTaskInfo
    $logon = $_.Principal.LogonType
    $flag = if ($logon -eq 'S4U') { 'OK' } else { 'WARNING - only runs while logged on' }
    Write-Output ("  {0,-20} state={1,-8} logon={2,-12} next={3}  [{4}]" -f `
        $_.TaskName, $_.State, $logon, $info.NextRunTime, $flag)
}
Write-Output ""
Write-Output "Verify with:   Get-ScheduledTask -TaskName 'TickerLake-*'"
Write-Output "Run once now:  Start-ScheduledTask -TaskName 'TickerLake-Daily'"
Write-Output "Remove:        Unregister-ScheduledTask -TaskName 'TickerLake-Daily' -Confirm:`$false"
