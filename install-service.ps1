#Requires -Version 5.1
<#
.SYNOPSIS
  Registriert die Risk-Layer /v1-Engine als Windows-Autostart, damit sie mit dem
  Rechner hochfährt (kein offenes Konsolenfenster). Startet Engine + Schicht 2
  (--llm-autostart) in einem.

  Bevorzugt einen echten Dienst via NSSM (falls vorhanden/auf dem PATH), sonst
  einen Geplanten Task beim Systemstart (schtasks, ohne Zusatz-Tool).

.PARAMETER Remove   Entfernt Dienst/Task wieder.
.PARAMETER Port     Bind-Port (Default 8000, Loopback).
.PARAMETER Nssm     Pfad zu nssm.exe (sonst wird der PATH durchsucht).
.EXAMPLE  ./install-service.ps1
.EXAMPLE  ./install-service.ps1 -Remove
#>
[CmdletBinding()]
param(
    [switch]$Remove,
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [string]$Nssm = ""
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$Name = "RiskLayer"
$py   = (Get-Command python).Source
# NICHT $args nennen: das ist eine automatische PowerShell-Variable (Funktions-Argumente).
$cmdArgs = "-m risk_layer.web --host $BindHost --port $Port --graph corpus/graph.sqlite --llm-autostart"
$token = if (Test-Path .engine_token) { (Get-Content .engine_token -Raw).Trim() } else { "" }
$nssmExe = if ($Nssm) { $Nssm } elseif (Get-Command nssm -ErrorAction SilentlyContinue) { (Get-Command nssm).Source } else { "" }

function Info($t) { Write-Host "  $t" -ForegroundColor DarkGray }
function OkMsg($t) { Write-Host "  [ok] $t" -ForegroundColor Green }

if ($Remove) {
    if ($nssmExe) { & $nssmExe stop $Name 2>$null; & $nssmExe remove $Name confirm 2>$null }
    schtasks /Delete /TN $Name /F 2>$null | Out-Null
    $wrapper = Join-Path $PSScriptRoot "service-start.cmd"
    if (Test-Path -LiteralPath $wrapper) { Remove-Item -LiteralPath $wrapper -Force }
    OkMsg "Autostart '$Name' entfernt (Dienst und/oder Task)."
    return
}

if ($nssmExe) {
    Write-Host "=== Windows-Dienst via NSSM ===" -ForegroundColor Cyan
    & $nssmExe install $Name $py $cmdArgs
    & $nssmExe set $Name AppDirectory $PSScriptRoot
    & $nssmExe set $Name Start SERVICE_AUTO_START
    if ($token) { & $nssmExe set $Name AppEnvironmentExtra "RISK_LAYER_TOKEN=$token" "RISK_LAYER_EMB_DEVICE=cuda" }
    else        { & $nssmExe set $Name AppEnvironmentExtra "RISK_LAYER_EMB_DEVICE=cuda" }
    & $nssmExe start $Name
    OkMsg "Dienst '$Name' installiert und gestartet (Autostart bei Boot)."
    Info "Status:  nssm status $Name   |   Entfernen: ./install-service.ps1 -Remove"
} else {
    Write-Host "=== Geplanter Task beim Systemstart (kein NSSM gefunden) ===" -ForegroundColor Cyan
    Info "Tipp: 'choco install nssm' fuer einen echten Dienst; sonst Task-Fallback."
    # Wrapper-CMD nach dem start.cmd-Muster: liest .engine_token erst zur LAUFZEIT —
    # der Token landet NICHT im Wrapper und NICHT in der schtasks-Definition (beides
    # ist sonst welt-lesbar). Damit hat der Task-Fallback dieselbe Auth wie NSSM.
    $wrapper = Join-Path $PSScriptRoot "service-start.cmd"
    @(
        '@echo off',
        'REM Auto-generiert von install-service.ps1 (Task-Fallback) - nicht editieren.',
        'REM Liest den Auth-Token zur Laufzeit aus .engine_token (kein Secret in dieser Datei).',
        "cd /d `"$PSScriptRoot`"",
        'if exist ".engine_token" for /f "usebackq delims=" %%t in (".engine_token") do set "RISK_LAYER_TOKEN=%%t"',
        'if not defined RISK_LAYER_EMB_DEVICE set "RISK_LAYER_EMB_DEVICE=cuda"',
        "`"$py`" $cmdArgs"
    ) | Set-Content -LiteralPath $wrapper -Encoding ascii
    schtasks /Create /TN $Name /SC ONSTART /RL HIGHEST /RU SYSTEM /TR "`"$wrapper`"" /F | Out-Null
    OkMsg "Task '$Name' angelegt (laeuft bei jedem Systemstart als SYSTEM, Token via $wrapper)."
    Info "Sofort starten: schtasks /Run /TN $Name   |   Entfernen: ./install-service.ps1 -Remove"
    if ($py -like "*\Users\*") {
        Write-Host "  [warnung] Python liegt user-lokal ($py) - SYSTEM findet es evtl. nicht." -ForegroundColor Yellow
        Write-Host "            Fuer Boot-Autostart: maschinenweites Python ODER nssm verwenden." -ForegroundColor Yellow
    }
}
