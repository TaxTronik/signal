#Requires -Version 5.1
<#
.SYNOPSIS
  Baut ein reproduzierbares, manifestiertes FESTWISSEN-Release.
.DESCRIPTION
  Der Default ist explizit OHNE Embedding. Mit -MitEmbedding wird die optionale
  Schicht verbindlich gebaut; ein fehlendes sentence-transformers ist dann ein
  harter Fehler. Mandanten- und Beraterdaten werden nie aufgenommen. Für
  byte-identische Wiederholungen denselben SOURCE_DATE_EPOCH setzen.
.EXAMPLE
  ./build-release-artifact.ps1 -Version 2026.07.13 -MitEmbedding
#>
[CmdletBinding()]
param(
    [string]$Version = (Get-Date -Format "yyyy-MM-dd"),
    [switch]$MitEmbedding
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Cli([string[]]$Command) {
    & python -m risk_layer.cli @Command
    if ($LASTEXITCODE -ne 0) {
        throw "risk-layer $($Command -join ' ') fehlgeschlagen"
    }
}

$gitSha = (& git rev-parse HEAD 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $gitSha) { $gitSha = "unknown" }
$sourceEpochWasSet = Test-Path Env:SOURCE_DATE_EPOCH
$sourceEpochOriginal = $env:SOURCE_DATE_EPOCH
if ($env:SOURCE_DATE_EPOCH) {
    $sourceEpoch = $env:SOURCE_DATE_EPOCH
} else {
    $sourceEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds().ToString()
}
$env:SOURCE_DATE_EPOCH = $sourceEpoch.Trim()
$embeddingEnvNames = @(
    "RISK_LAYER_EMBEDDING_MODEL",
    "RISK_LAYER_EMBEDDING_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE"
)
$embeddingEnvOriginal = @{}
foreach ($name in $embeddingEnvNames) {
    $embeddingEnvOriginal[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$out = "taxtronik-festwissen-$Version.tar.gz"
$stage = Join-Path ([IO.Path]::GetTempPath()) ("risk-layer-release-" + [guid]::NewGuid().ToString("N"))
[void](New-Item -ItemType Directory -Path $stage)
$corpusDir = Join-Path $stage "corpus"
[void](New-Item -ItemType Directory -Path (Join-Path $stage "catalog"))
Copy-Item -LiteralPath "catalog/begriffe.yaml" -Destination (Join-Path $stage "catalog/begriffe.yaml")

try {
    Write-Host "=== Festwissen bauen (Version $Version, Embedding=$([bool]$MitEmbedding)) ===" -ForegroundColor Cyan
    Invoke-Cli @("korpus", "init", "--dir", $corpusDir)
    Invoke-Cli @("korpus", "fetch", "--dir", $corpusDir)
    & python -m tools.verify_corpus_fetch $corpusDir
    if ($LASTEXITCODE -ne 0) { throw "Korpus-Fetch ist nicht vollständig" }
    $graphDb = Join-Path $corpusDir "graph.sqlite"
    $triggerDb = Join-Path $corpusDir "trigger.sqlite"
    Invoke-Cli @("graph", "build", "--dir", $corpusDir, "--db", $graphDb)
    Invoke-Cli @("trigger", "build", "--db", $triggerDb, "--graph-db", $graphDb)

    $packageArgs = @(
        "-m", "tools.package_festwissen_release",
        "--release-id", $Version,
        "--git-sha", $gitSha.Trim(),
        "--source-date-epoch", $sourceEpoch.Trim(),
        "--output", $out,
        "--source-root", $stage
    )
    if ($MitEmbedding) {
        & python -c "import sentence_transformers" 2>$null
        if ($LASTEXITCODE -ne 0) { throw "-MitEmbedding verlangt sentence-transformers" }
        $embeddingDir = Join-Path $corpusDir "embedding"
        $modelDir = Join-Path $stage "models/bge-m3"
        & python -m tools.snapshot_embedding_model --output $modelDir
        if ($LASTEXITCODE -ne 0) { throw "BGE-M3-Snapshot fehlgeschlagen" }
        $env:RISK_LAYER_EMBEDDING_MODEL = $modelDir
        $env:RISK_LAYER_EMBEDDING_OFFLINE = "1"
        $env:HF_HUB_OFFLINE = "1"
        $env:TRANSFORMERS_OFFLINE = "1"
        Invoke-Cli @(
            "embedding", "build", "--encoder", "bge-m3", "--model", $modelDir,
            "--device", "cpu", "--dir", $embeddingDir, "--graph-db", $graphDb
        )
        $packageArgs += "--embedding"
    }

    & python @packageArgs
    if ($LASTEXITCODE -ne 0) { throw "Paketierung fehlgeschlagen" }
} finally {
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    $resolvedStage = [IO.Path]::GetFullPath($stage)
    if ($resolvedStage.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedStage).StartsWith("risk-layer-release-")) {
        Remove-Item -LiteralPath $resolvedStage -Recurse -Force -ErrorAction SilentlyContinue
    } else {
        Write-Warning "Unsicheres Staging-Verzeichnis wird nicht entfernt: $resolvedStage"
    }
    if ($sourceEpochWasSet) {
        $env:SOURCE_DATE_EPOCH = $sourceEpochOriginal
    } else {
        Remove-Item Env:SOURCE_DATE_EPOCH -ErrorAction SilentlyContinue
    }
    foreach ($name in $embeddingEnvNames) {
        [Environment]::SetEnvironmentVariable(
            $name,
            $embeddingEnvOriginal[$name],
            "Process"
        )
    }
}

Write-Host "=== fertig ===" -ForegroundColor Green
Write-Host "  $out"
Write-Host "  $out.sha256"
Write-Host "  $out.manifest.json"
Write-Host "Installation ausschließlich über das profilbasierte risk-layerctl; nicht live entpacken."
