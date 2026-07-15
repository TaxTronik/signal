#Requires -Version 5.1
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$OperatorArgs)

$ErrorActionPreference = "Stop"
$operator = Join-Path $PSScriptRoot "risk_layer\operator_cli.py"
if (
    [string]::IsNullOrWhiteSpace($env:RISK_LAYER_OPS_PROFILE) -and
    (Test-Path -LiteralPath $operator -PathType Leaf)
) {
    $env:RISK_LAYER_OPS_PROFILE = Join-Path $PSScriptRoot "deploy\risk-layer-ops.json"
}
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$python = if (-not [string]::IsNullOrWhiteSpace($env:RISK_LAYER_OPERATOR_PYTHON)) {
    $env:RISK_LAYER_OPERATOR_PYTHON
} elseif (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $venvPython
} else {
    $candidate = Get-Command python3.12, python -CommandType Application `
        -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $candidate) {
        throw "Kein Python gefunden (erwartet .venv, python3.12 oder python)."
    }
    $candidate.Source
}
if (Test-Path -LiteralPath $operator -PathType Leaf) {
    & $python -I $operator @OperatorArgs
    exit $LASTEXITCODE
}
$releaseTools = if ([string]::IsNullOrWhiteSpace($env:RISK_LAYER_RELEASE_TOOLS)) {
    Join-Path $PSScriptRoot "risk-layer-release-tools.pyz"
} else {
    $env:RISK_LAYER_RELEASE_TOOLS
}
if (-not (Test-Path -LiteralPath $releaseTools -PathType Leaf)) {
    throw "Weder Operator-Modul noch Release-Zipapp gefunden."
}
& $python -I $releaseTools ops @OperatorArgs
exit $LASTEXITCODE
