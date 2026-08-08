#Requires -Version 5.1
[CmdletBinding()]
param([string]$TaskName = "RiskLayer")

$ErrorActionPreference = "Stop"
$task = Get-ScheduledTask -TaskName $TaskName -TaskPath "\" -ErrorAction Stop
if ($task.State -eq "Running") {
    Stop-ScheduledTask -InputObject $task -ErrorAction Stop
}

# Task Scheduler beendet den PowerShell-Wrapper, laesst unter Windows aber den
# von .venv\Scripts\python.exe gestarteten Basis-Interpreter gelegentlich als
# Kindprozess weiterlaufen. Dieser Prozess haelt graph.sqlite offen und wuerde
# auch nach einem scheinbar erfolgreichen stop weiter Requests bedienen.
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = [IO.Path]::GetFullPath((Join-Path $repoRoot ".venv\Scripts\python.exe"))
$processes = @(Get-CimInstance Win32_Process)
$roots = @($processes | Where-Object {
    $_.ExecutablePath -and
    [IO.Path]::GetFullPath($_.ExecutablePath).Equals(
        $venvPython,
        [StringComparison]::OrdinalIgnoreCase
    ) -and
    $_.CommandLine -match "risk_layer\.web"
})

function Get-DescendantIds([int]$ParentId) {
    $children = @($script:processes | Where-Object { $_.ParentProcessId -eq $ParentId })
    foreach ($child in $children) {
        Get-DescendantIds -ParentId ([int]$child.ProcessId)
        [int]$child.ProcessId
    }
}

foreach ($root in $roots) {
    $ids = @(Get-DescendantIds -ParentId ([int]$root.ProcessId))
    foreach ($id in $ids) {
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id ([int]$root.ProcessId) -Force -ErrorAction SilentlyContinue
}

$deadline = [DateTime]::UtcNow.AddSeconds(30)
do {
    Start-Sleep -Milliseconds 200
    $task = Get-ScheduledTask -TaskName $TaskName -TaskPath "\" -ErrorAction Stop
    $remaining = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -and
        [IO.Path]::GetFullPath($_.ExecutablePath).Equals(
            $venvPython,
            [StringComparison]::OrdinalIgnoreCase
        ) -and
        $_.CommandLine -match "risk_layer\.web"
    })
} while (($task.State -eq "Running" -or $remaining.Count -gt 0) -and [DateTime]::UtcNow -lt $deadline)

if ($task.State -eq "Running" -or $remaining.Count -gt 0) {
    throw "Task '$TaskName' oder dessen Risk-Layer-Prozessbaum konnte nicht gestoppt werden."
}

Write-Host "Task '$TaskName' gestoppt."
