#Requires -Version 5.1
<#
.SYNOPSIS
  Registriert die Risk-Layer /v1-Engine als begrenzten Windows-Autostart fuer
  den aktuellen Benutzer.

  Der Task startet bei der Benutzeranmeldung mit RunLevel "Limited" und ohne
  gespeichertes Kennwort. Ein Boot-Dienst aus dem benutzerschreibbaren Checkout
  wird aus Sicherheitsgruenden nicht angeboten.

.PARAMETER Remove
  Entfernt den Benutzer-Task und einen eventuell vorhandenen Legacy-Dienst.
.PARAMETER Port
  Bind-Port (Default 8000, Loopback).
.PARAMETER BindHost
  Bind-Adresse (Default 127.0.0.1).
.PARAMETER EmbeddingDevice
  Geraet fuer Embeddings/Reranker (cpu oder cuda[/cuda:N]).

.EXAMPLE
  ./install-service.ps1
.EXAMPLE
  ./install-service.ps1 -EmbeddingDevice cuda
.EXAMPLE
  ./install-service.ps1 -Remove

.NOTES
  Migration von aelteren Versionen: Ein vorhandener Windows-Dienst namens
  "RiskLayer" wird nicht uebernommen. Zuerst aus einer administrativen
  PowerShell bewusst mit -Remove entfernen und anschliessend als normaler
  Benutzer den begrenzten Task neu installieren.
#>
[CmdletBinding()]
param(
    [switch]$Remove,
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [ValidatePattern("^(cpu|cuda(?::[0-9]+)?)$")][string]$EmbeddingDevice = "cpu"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$Name = "RiskLayer"
$wrapper = Join-Path $PSScriptRoot "service-start.ps1"
$legacyWrapper = Join-Path $PSScriptRoot "service-start.cmd"

function Info($t) { Write-Host "  $t" -ForegroundColor DarkGray }
function OkMsg($t) { Write-Host "  [ok] $t" -ForegroundColor Green }

function Get-RiskLayerService {
    # Eine leere erfolgreiche Abfrage bedeutet "nicht vorhanden". Provider-/
    # Berechtigungsfehler muessen dagegen laut bleiben, sonst koennte ein alter
    # privilegierter Dienst unbemerkt parallel weiterlaufen.
    Get-CimInstance -ClassName Win32_Service -Filter "Name='$Name'" -ErrorAction Stop
}

function Assert-NativeSuccess([string]$Operation) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation fehlgeschlagen (exit $LASTEXITCODE)."
    }
}

function Get-RiskLayerTasks {
    try {
        @(Get-ScheduledTask -TaskName $Name -TaskPath "\" -ErrorAction Stop)
    } catch {
        if ($_.FullyQualifiedErrorId -like "CmdletizationQuery_NotFound*,*") {
            return
        }
        throw "Task-Status von '$Name' konnte nicht sicher ermittelt werden: $($_.Exception.Message)"
    }
}

function Remove-RiskLayerTask {
    foreach ($task in @(Get-RiskLayerTasks)) {
        if ($task.State -eq "Running") {
            Stop-ScheduledTask -InputObject $task -ErrorAction Stop
        }
        Unregister-ScheduledTask -InputObject $task -Confirm:$false -ErrorAction Stop
    }
    if (@(Get-RiskLayerTasks).Count -ne 0) {
        throw "Geplanter Task '$Name' besteht nach dem Loeschversuch weiter."
    }
}

function Remove-GeneratedWrappers {
    foreach ($path in @($wrapper, $legacyWrapper)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }
}

if ($Remove) {
    $service = Get-RiskLayerService
    if ($null -ne $service) {
        if ($service.State -ne "Stopped") {
            Stop-Service -Name $Name -Force -ErrorAction Stop
            $service = Get-RiskLayerService
            if ($null -ne $service -and $service.State -ne "Stopped") {
                throw "Windows-Dienst '$Name' konnte nicht sicher gestoppt werden."
            }
        }
        & sc.exe delete $Name | Out-Null
        Assert-NativeSuccess "Legacy-Windows-Dienst '$Name' entfernen"
        if ($null -ne (Get-RiskLayerService)) {
            throw "Windows-Dienst '$Name' besteht nach dem Loeschversuch weiter."
        }
    }

    Remove-RiskLayerTask
    Remove-GeneratedWrappers
    OkMsg "Autostart '$Name' entfernt (Benutzer-Task und gegebenenfalls Legacy-Dienst)."
    return
}

$parsedAddress = $null
if ($BindHost -ne "localhost" -and
    -not [System.Net.IPAddress]::TryParse($BindHost, [ref]$parsedAddress)) {
    throw "BindHost muss eine IP-Adresse oder 'localhost' sein (erhalten: '$BindHost')."
}

$existingService = Get-RiskLayerService
if ($null -ne $existingService) {
    throw @"
Der Legacy-Windows-Dienst '$Name' existiert noch (Konto: $($existingService.StartName)).
Aus einem benutzerschreibbaren Checkout wird kein Windows-Dienst mehr installiert.
Migration: ./install-service.ps1 -Remove
Den Remove-Schritt bei einem Legacy-Dienst administrativ ausfuehren. Danach den
begrenzten CurrentUser-Task als normaler Benutzer mit ./install-service.ps1 neu installieren.
"@
}

# Einen eventuell vorhandenen alten Task (insbesondere einen frueheren
# SYSTEM-Task) vor jeder Aenderung am Wrapper nachweislich entfernen. Ohne
# ausreichende Rechte wird laut verweigert.
Remove-RiskLayerTask
Remove-GeneratedWrappers

# Der Wrapper enthaelt keine absoluten Projektpfade: $PSScriptRoot wird erst zur
# Laufzeit von PowerShell aufgeloest. UTF-16 mit BOM ist unter Windows PowerShell
# 5.1 Unicode-sicher, auch wenn Benutzer- oder Projektpfade Nicht-ASCII enthalten.
$wrapperTemplate = @'
# Auto-generiert von install-service.ps1 - nicht editieren.
$ErrorActionPreference = "Stop"

try {
    Set-Location -LiteralPath $PSScriptRoot
    $tokenPath = Join-Path $PSScriptRoot ".engine_token"
    if (-not (Test-Path -LiteralPath $tokenPath -PathType Leaf)) {
        throw "Auth-Token-Datei fehlt."
    }

    $tokenRaw = [System.IO.File]::ReadAllText($tokenPath)
    $token = $tokenRaw.Trim()
    if ($token.Length -lt 32 -or $token -match "\s") {
        throw "Auth-Token ist ungueltig oder zu kurz."
    }
    $env:RISK_LAYER_TOKEN = $token
    $env:RISK_LAYER_EMB_DEVICE = "@EMBEDDING_DEVICE@"

    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    $python = if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $venvPython
    } else {
        (Get-Command python -CommandType Application -ErrorAction Stop).Source
    }

    & $python -m risk_layer.web --host "@BIND_HOST@" --port @PORT@ --graph corpus/graph.sqlite --llm-autostart
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine("Risk-Layer-Autostart verweigert Start: $($_.Exception.Message)")
    exit 20
}
'@
$wrapperContent = $wrapperTemplate.Replace("@BIND_HOST@", $BindHost)
$wrapperContent = $wrapperContent.Replace("@PORT@", [string]$Port)
$wrapperContent = $wrapperContent.Replace("@EMBEDDING_DEVICE@", $EmbeddingDevice)
[System.IO.File]::WriteAllText($wrapper, $wrapperContent, [System.Text.Encoding]::Unicode)

$powershellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
if (-not (Test-Path -LiteralPath $powershellExe -PathType Leaf)) {
    throw "Vertrauenswuerdiges System-PowerShell nicht gefunden: $powershellExe"
}

Write-Host "=== Begrenzter Autostart fuer aktuellen Benutzer ===" -ForegroundColor Cyan
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
if ($null -eq $identity.User -or [string]::IsNullOrWhiteSpace($identity.Name)) {
    throw "Aktuelles Windows-Benutzerkonto konnte nicht ermittelt werden."
}

$actionArgs = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$wrapper`""
$action = New-ScheduledTaskAction -Execute $powershellExe -Argument $actionArgs -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity.Name
$principal = New-ScheduledTaskPrincipal -UserId $identity.Name -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $Name -TaskPath "\" -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $Name -TaskPath "\"
OkMsg "Task '$Name' fuer '$($identity.Name)' installiert und gestartet (RunLevel Limited)."
Info "Autostart erfolgt bei Benutzeranmeldung; kein Kennwort wird gespeichert."
Info "Ohne lesbares Token mit mindestens 32 Zeichen verweigert der Wrapper den Start."
Info "Status: Get-ScheduledTask -TaskName $Name   |   Entfernen: ./install-service.ps1 -Remove"
