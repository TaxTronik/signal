#Requires -Version 5.1
<#
.SYNOPSIS
  Aktivierungs-Hook fuer transaktionale Windows-Releases.
.DESCRIPTION
  Stoppt den begrenzten Benutzer-Task, falls er laeuft, und startet ihn neu.
  Die anschliessende Release-/Health-Pruefung uebernimmt signal. Das
  Skript veraendert weder Taskdefinition noch Token und benoetigt keine
  administrativen Rechte.
#>
[CmdletBinding()]
param([string]$TaskName = "RiskLayer")

$ErrorActionPreference = "Stop"
$task = Get-ScheduledTask -TaskName $TaskName -TaskPath "\" -ErrorAction Stop
if ($task.State -eq "Running") {
    Stop-ScheduledTask -InputObject $task -ErrorAction Stop
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 200
        $task = Get-ScheduledTask -TaskName $TaskName -TaskPath "\" -ErrorAction Stop
    } while ($task.State -eq "Running" -and [DateTime]::UtcNow -lt $deadline)
    if ($task.State -eq "Running") {
        throw "Task '$TaskName' konnte nicht innerhalb von 30 Sekunden gestoppt werden."
    }
}
Start-ScheduledTask -InputObject $task -ErrorAction Stop
Write-Host "Task '$TaskName' neu gestartet."
