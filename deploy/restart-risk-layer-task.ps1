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
$stopScript = Join-Path $PSScriptRoot "stop-risk-layer-task.ps1"
& $stopScript -TaskName $TaskName
$task = Get-ScheduledTask -TaskName $TaskName -TaskPath "\" -ErrorAction Stop
Start-ScheduledTask -InputObject $task -ErrorAction Stop
Write-Host "Task '$TaskName' neu gestartet."
