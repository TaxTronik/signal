#Requires -Version 5.1
<#
.SYNOPSIS
  Installiert einen taeglichen, unprivilegierten Windows-Backup-Task.
.DESCRIPTION
  Der Task laeuft nur als aktuell angemeldeter Benutzer und speichert kein
  Kennwort. Python, checkout-freies Release-Zipapp und das nicht geheime
  Operator-Profil werden Base64-kodiert uebergeben. Damit nutzt der Scheduler
  exakt dieselbe profilbasierte Fassade wie der Handbetrieb.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$PythonExe,
    [string]$ReleaseTools,
    [string]$OperatorProfile,
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

foreach ($parameter in @{
    PythonExe = $PythonExe
    ReleaseTools = $ReleaseTools
    OperatorProfile = $OperatorProfile
}.GetEnumerator()) {
    if ([string]::IsNullOrWhiteSpace($parameter.Value)) {
        throw "-$($parameter.Key) ist fuer die Installation Pflicht"
    }
    if ($parameter.Value -match '[\x00-\x1f\x7f]') {
        throw "Steuerzeichen in -$($parameter.Key) sind unzulaessig"
    }
}

$PythonExe = [IO.Path]::GetFullPath($PythonExe)
$ReleaseTools = [IO.Path]::GetFullPath($ReleaseTools)
$OperatorProfile = [IO.Path]::GetFullPath($OperatorProfile)
foreach ($path in @($PythonExe, $ReleaseTools, $OperatorProfile)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Pflichtdatei nicht gefunden: $path"
    }
}

# Keine manuell gequoteten Windows-Argumentstrings: Pfade werden erst im
# Kindprozess aus konstantem Base64 rekonstruiert und als echte Argumente
# uebergeben, niemals durch Invoke-Expression oder eine Shell ausgewertet.
$pythonB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($PythonExe))
$toolsB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($ReleaseTools))
$profileB64 = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes($OperatorProfile)
)
$taskCommand = @"
`$ErrorActionPreference = 'Stop'
try {
`$python = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$pythonB64'))
`$tools = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$toolsB64'))
`$profile = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$profileB64'))
if (-not (Test-Path -LiteralPath `$python -PathType Leaf)) {
    throw 'Python-Interpreter fehlt zur Laufzeit'
}
if (-not (Test-Path -LiteralPath `$tools -PathType Leaf)) {
    throw 'Release-Zipapp fehlt zur Laufzeit'
}
if (-not (Test-Path -LiteralPath `$profile -PathType Leaf)) {
    throw 'Operator-Profil fehlt zur Laufzeit'
}
`$LASTEXITCODE = `$null
& `$python -I `$tools ops --config `$profile backup
`$childExit = `$LASTEXITCODE
if (`$null -eq `$childExit) {
    throw 'Python-Prozess lieferte keinen externen Exitcode'
}
exit [int]`$childExit
}
catch {
    [Console]::Error.WriteLine(`$_.Exception.Message)
    exit 1
}
"@
$encodedCommand = [Convert]::ToBase64String(
    [Text.Encoding]::Unicode.GetBytes($taskCommand)
)
$powershell = [IO.Path]::Combine(
    [Environment]::SystemDirectory,
    'WindowsPowerShell',
    'v1.0',
    'powershell.exe'
)
if (-not (Test-Path -LiteralPath $powershell -PathType Leaf)) {
    throw "System-PowerShell nicht gefunden: $powershell"
}
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
        -Description "Profilbasiertes Risk-Layer-Festwissen-Backup mit Retention" `
        -Force | Out-Null
    Write-Host "Installiert: $TaskName ($identity, taeglich $($DailyAt.ToString('HH:mm')))" -ForegroundColor Green
    Write-Host "Erstlauf manuell pruefen: Start-ScheduledTask -TaskName '$TaskName'"
}
