#Requires -Version 5.1
<#
.SYNOPSIS
  Installiert einen taeglichen, unprivilegierten Windows-Backup-Task.
.DESCRIPTION
  Der Task laeuft nur im Kontext des aktuell angemeldeten Benutzers und
  speichert kein Kennwort. Alle Pfade werden Base64-kodiert an PowerShell
  uebergeben; UNC-Pfade, Leerzeichen und abschliessende Backslashes bleiben
  dadurch eindeutig. Im Release-Modus waehlt der stabile Launcher die aktuelle
  manifestgebundene Runtime. Nach jedem Backup gilt Retention 30/90.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$BeraterDir,
    [string]$BackupDir,
    [string]$RiskLayerCtl,
    [string]$PythonExe,
    [string]$ReleaseLauncher,
    [string]$ReleaseHome,
    [string]$TaskName = "TaxTronik Risk-Layer Backup",
    [datetime]$DailyAt = [datetime]::Today.AddHours(2).AddMinutes(15),
    [switch]$Remove
)
$ErrorActionPreference = "Stop"

if ($Remove) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess($TaskName, "Scheduled Task entfernen")) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        }
    }
    return
}

if (-not $BeraterDir -or -not $BackupDir) {
    throw "-BeraterDir und -BackupDir sind fuer die Installation Pflicht"
}
$releaseMode = @($PythonExe, $ReleaseLauncher, $ReleaseHome) |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    Measure-Object |
    Select-Object -ExpandProperty Count
if ($releaseMode -ne 0 -and $releaseMode -ne 3) {
    throw "-PythonExe, -ReleaseLauncher und -ReleaseHome nur gemeinsam angeben"
}
$releaseMode = $releaseMode -eq 3
if (-not $releaseMode -and -not $RiskLayerCtl) {
    $commandInfo = Get-Command risk-layerctl -ErrorAction Stop
    $RiskLayerCtl = $commandInfo.Source
}
$pathValues = @($BeraterDir, $BackupDir)
if ($releaseMode) {
    $pathValues += @($PythonExe, $ReleaseLauncher, $ReleaseHome)
} else {
    $pathValues += $RiskLayerCtl
}
foreach ($value in $pathValues) {
    if ($value -match '[\x00-\x1f\x7f]') {
        throw "Steuerzeichen in Pfaden sind unzulaessig"
    }
}
$BeraterDir = [IO.Path]::GetFullPath($BeraterDir)
$BackupDir = [IO.Path]::GetFullPath($BackupDir)
if (-not (Test-Path -LiteralPath $BeraterDir -PathType Container)) {
    throw "Berater-Verzeichnis nicht gefunden: $BeraterDir"
}
if (-not (Test-Path -LiteralPath $BackupDir -PathType Container)) {
    throw "Backup-Ziel nicht gefunden (bewusst nicht automatisch erzeugt): $BackupDir"
}
if ($releaseMode) {
    $PythonExe = [IO.Path]::GetFullPath($PythonExe)
    $ReleaseLauncher = [IO.Path]::GetFullPath($ReleaseLauncher)
    $ReleaseHome = [IO.Path]::GetFullPath($ReleaseHome)
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw "Python-Interpreter nicht gefunden: $PythonExe"
    }
    if (-not (Test-Path -LiteralPath $ReleaseLauncher -PathType Leaf)) {
        throw "Release-Launcher nicht gefunden: $ReleaseLauncher"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ReleaseHome "state.json") -PathType Leaf)) {
        throw "Release-Home hat keinen aktivierten State: $ReleaseHome"
    }
} else {
    $RiskLayerCtl = [IO.Path]::GetFullPath($RiskLayerCtl)
    if (-not (Test-Path -LiteralPath $RiskLayerCtl -PathType Leaf)) {
        throw "risk-layerctl nicht gefunden: $RiskLayerCtl"
    }
}

# Keine manuell gequoteten Windows-Argumentstrings: Pfade werden erst im
# Kindprozess aus konstantem Base64 rekonstruiert und als echte Argumente
# uebergeben, niemals durch Invoke-Expression oder eine Shell ausgewertet.
$beraterB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($BeraterDir))
$backupB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($BackupDir))
$taskCommand = if ($releaseMode) {
    $pythonB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($PythonExe))
    $launcherB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($ReleaseLauncher))
    $homeB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($ReleaseHome))
    @"
`$python = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$pythonB64'))
`$launcher = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$launcherB64'))
`$home = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$homeB64'))
`$berater = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$beraterB64'))
`$backup = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$backupB64'))
& `$python -I `$launcher --home `$home -- backup create --berater `$berater --ziel `$backup --retention-keep-last 30 --retention-keep-days 90
exit `$LASTEXITCODE
"@
} else {
    $ctlB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($RiskLayerCtl))
    @"
`$ctl = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$ctlB64'))
`$berater = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$beraterB64'))
`$backup = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$backupB64'))
& `$ctl backup create --berater `$berater --ziel `$backup --retention-keep-last 30 --retention-keep-days 90
exit `$LASTEXITCODE
"@
}
$encodedCommand = [Convert]::ToBase64String(
    [Text.Encoding]::Unicode.GetBytes($taskCommand)
)
$powershell = Join-Path $PSHOME 'powershell.exe'
$arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $encodedCommand"
$action = New-ScheduledTaskAction -Execute $powershell -Argument $arguments
$trigger = New-ScheduledTaskTrigger -Daily -At $DailyAt
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal `
    -UserId $identity `
    -LogonType Interactive `
    -RunLevel Limited

if ($PSCmdlet.ShouldProcess($TaskName, "taeglichen Scheduled Task installieren")) {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "Konsistentes Risk-Layer-Festwissen-Backup mit lokaler Retention" `
        -Force | Out-Null
    Write-Host "Installiert: $TaskName ($identity, taeglich $($DailyAt.ToString('HH:mm')))" -ForegroundColor Green
    Write-Host "Erstlauf manuell pruefen: Start-ScheduledTask -TaskName '$TaskName'"
}
