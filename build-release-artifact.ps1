#Requires -Version 5.1
<#
.SYNOPSIS
  Baut das FESTWISSEN-Release-Artefakt EINMAL zentral (Korpus + Normgraph +
  Trigger + Embedding) als versionierte .tar.gz — IDENTISCHES Format wie
  build-release-artifact.sh, damit Hash-Check (`sha256sum -c`) und Entpacken
  plattformunabhängig gleich funktionieren (Windows 10+ liefert bsdtar als tar.exe).
  Kanzleien ziehen das Artefakt, statt es lokal aus gesetze-im-internet zu bauen.

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

$out = "taxtronik-festwissen-$Version.tar.gz"
if (Test-Path $out) { Remove-Item $out -Force }
# Windows 10+ bringt bsdtar als tar.exe mit — gleiches Artefakt wie das sh-Pendant.
& tar -czf $out @items
if ($LASTEXITCODE -ne 0) { throw "tar fehlgeschlagen (exit $LASTEXITCODE)" }
# sha256sum-Format ("<hash>  <datei>", lowercase) — verifizierbar via `sha256sum -c`.
$hash = (Get-FileHash $out -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $out" | Set-Content "$out.sha256" -Encoding ascii
Write-Host "=== fertig: $out (+ .sha256) ===" -ForegroundColor Green
Write-Host "Auf dem Zielrechner:  tar -xzf $out   (überschreibt corpus/ + catalog/begriffe.yaml)"
