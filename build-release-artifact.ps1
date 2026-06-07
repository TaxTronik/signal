#Requires -Version 5.1
<#
.SYNOPSIS
  Baut das FESTWISSEN-Release-Artefakt EINMAL zentral (Korpus + Normgraph +
  Trigger + Embedding) als versionierte ZIP. Kanzleien ziehen das Artefakt,
  statt es lokal aus gesetze-im-internet zu bauen.

  Mandantenfrei + regenerierbar — nur Festwissen, nie Mandantendaten.
.EXAMPLE  ./build-release-artifact.ps1 2026-06
#>
[CmdletBinding()]
param([string]$Version = (Get-Date -Format "yyyy-MM-dd"))
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Cli([string[]]$c) { & python -m risk_layer.cli @c; if ($LASTEXITCODE -ne 0) { throw "risk-layer $($c -join ' ') fehlgeschlagen" } }

Write-Host "=== Festwissen bauen (Version $Version) ===" -ForegroundColor Cyan
Cli @("korpus","init"); Cli @("korpus","fetch"); Cli @("graph","build"); Cli @("trigger","build")
$items = @("corpus/gesetze.yaml","corpus/graph.sqlite","corpus/trigger.sqlite","catalog/begriffe.yaml")
& python -c "import sentence_transformers" 2>$null
if ($LASTEXITCODE -eq 0) { Cli @("embedding","build"); $items += "corpus/embedding" }
else { Write-Host "  (sentence-transformers fehlt -> Embedding-Index NICHT im Artefakt)" -ForegroundColor DarkGray }

$out = "taxtronik-festwissen-$Version.zip"
if (Test-Path $out) { Remove-Item $out -Force }
Compress-Archive -Path $items -DestinationPath $out
(Get-FileHash $out -Algorithm SHA256).Hash | Set-Content "$out.sha256" -Encoding ascii
Write-Host "=== fertig: $out (+ .sha256) ===" -ForegroundColor Green
Write-Host "Auf dem Zielrechner:  Expand-Archive $out -DestinationPath . -Force"
