#Requires -Version 5.1
<#
.SYNOPSIS
  Setup/Deploy des Risk-Layer von "frischem Clone" zu lauffaehiger /v1-Engine:
  Abhaengigkeiten, LLM-Modell (GGUF) + llama.cpp-Vulkan-Engine herunterladen,
  Normkorpus/Graph/Trigger(/Embedding) bauen und Auth-Token anlegen.
  Idempotent - vorhandene Artefakte werden uebersprungen (ausser -Force).

.PARAMETER WithEmbedding
  Zusaetzlich den semantischen BGE-M3-Index bauen (installiert sentence-transformers
  + torch, ~2 GB, und laedt das Modell beim ersten Build).
.PARAMETER Voll
  Breiter Steuer-Korpus (~21 Gesetze) statt nur der im Katalog referenzierten.
.PARAMETER Modell
  Modell-Key statt des Defaults (qwen3-14b). Siehe 'risk-layer modell list'.
.PARAMETER Backend
  GPU-Backend der Engine: auto (Default; NVIDIA->cuda, AMD->hip, sonst Vulkan/CPU)
  oder explizit cuda|hip|vulkan|cpu.
.PARAMETER Start
  Nach dem Setup die /v1-Engine starten (blockiert).

.EXAMPLE
  ./deploy.ps1                          # Deps + Modell + Engine + Korpus/Graph/Trigger
.EXAMPLE
  ./deploy.ps1 -WithEmbedding -Voll     # zusaetzlich Embedding-Index + breiter Korpus
.EXAMPLE
  ./deploy.ps1 -SkipModel -SkipEngine   # nur Korpus/Graph bauen (LLM separat)
.EXAMPLE
  ./deploy.ps1 -Start                   # Setup, dann Engine starten
#>
[CmdletBinding()]
param(
    [switch]$SkipDeps,
    [switch]$SkipModel,
    [switch]$SkipEngine,
    [switch]$SkipCorpus,
    [switch]$WithEmbedding,
    [switch]$Voll,
    [switch]$Force,
    [switch]$Start,
    [string]$Modell = "",
    [ValidateSet("auto","cuda","hip","vulkan","cpu")][string]$Backend = "auto",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Schritt($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Info($t)    { Write-Host "  $t" -ForegroundColor DarkGray }
function OkMsg($t)   { Write-Host "  [ok] $t" -ForegroundColor Green }

function Cli([string[]]$cmd) {
    Write-Host "  > python -m risk_layer.cli $($cmd -join ' ')" -ForegroundColor DarkGray
    & python -m risk_layer.cli @cmd
    if ($LASTEXITCODE -ne 0) { throw "Fehlgeschlagen (exit $LASTEXITCODE): risk-layer $($cmd -join ' ')" }
}
function PyOk([string]$code) { & python -c $code 2>$null; return ($LASTEXITCODE -eq 0) }

# --- Python ----------------------------------------------------------------
Schritt "Python pruefen"
$pv = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])")
if ($LASTEXITCODE -ne 0) { throw "python nicht gefunden - Python >= 3.11 installieren." }
$maj, $min = $pv.Split('.')
if ([int]$maj -lt 3 -or ([int]$maj -eq 3 -and [int]$min -lt 11)) {
    throw "Python >= 3.11 noetig (gefunden $pv)."
}
OkMsg "Python $pv"

# --- Abhaengigkeiten -------------------------------------------------------
if (-not $SkipDeps) {
    Schritt "Abhaengigkeiten installieren"
    & python -m pip install --upgrade pip | Out-Null
    & python -m pip install -e . "numpy>=1.26"
    if ($LASTEXITCODE -ne 0) { throw "pip install (Kern) fehlgeschlagen." }
    if ($WithEmbedding) {
        Info "Embedding gewuenscht -> sentence-transformers (+ torch, gross) installieren"
        & python -m pip install "sentence-transformers>=2.7"
        if ($LASTEXITCODE -ne 0) { throw "pip install sentence-transformers fehlgeschlagen." }
        OkMsg "Abhaengigkeiten installiert (PyYAML + numpy + sentence-transformers)"
    } else {
        OkMsg "Abhaengigkeiten installiert (PyYAML + numpy)"
    }
} else { Info "Deps uebersprungen (-SkipDeps)" }

# --- Auth-Token ------------------------------------------------------------
Schritt "Auth-Token (.engine_token)"
if ((-not (Test-Path -LiteralPath ".engine_token")) -or $Force) {
    $tok = & python -c "import secrets; print(secrets.token_hex(32))"
    Set-Content -LiteralPath ".engine_token" -Value $tok -NoNewline -Encoding ascii
    OkMsg "neues Token erzeugt (.engine_token, gitignored)"
} else { OkMsg ".engine_token vorhanden (wiederverwendet)" }
$token = (Get-Content -LiteralPath ".engine_token" -Raw).Trim()
Info "RISK_LAYER_TOKEN = $token"

# --- LLM-Modell (GGUF) -----------------------------------------------------
if (-not $SkipModel) {
    Schritt "LLM-Modell (GGUF) herunterladen"
    if ($Modell) { $mref = "M.MODELLE['$Modell']" } else { $mref = "M.MODELLE[M.DEFAULT_MODELL]" }
    $vorhanden = PyOk "from risk_layer import modell as M; import sys; sys.exit(0 if M.modell_pfad($mref) else 1)"
    if ($vorhanden -and -not $Force) {
        OkMsg "Modell bereits vorhanden (./models) - uebersprungen"
    } else {
        Info "laedt das Default-GGUF (resumebar) nach ./models ..."
        $dl = @("modell", "download"); if ($Modell) { $dl += @("--modell", $Modell) }
        Cli $dl
        OkMsg "Modell geladen"
    }
} else { Info "Modell uebersprungen (-SkipModel)" }

# --- llama.cpp-Engine (llama-server, GPU-Backend) --------------------------
if (-not $SkipEngine) {
    Schritt "llama.cpp-Engine bereitstellen (Backend: $Backend)"
    $hatBin = PyOk "from risk_layer import modell as M; import sys; sys.exit(0 if M.finde_llama_server() else 1)"
    if ($hatBin -and -not $Force) {
        OkMsg "llama-server gefunden (./engine oder PATH) - uebersprungen"
    } else {
        Info "laedt die passende llama.cpp-Release (NVIDIA->cuda, AMD->hip, sonst Vulkan/CPU) nach ./engine ..."
        Cli @("modell", "engine", "--backend", $Backend)
        OkMsg "Engine bereitgestellt"
    }
} else { Info "Engine uebersprungen (-SkipEngine)" }

# --- Normkorpus + Graph + Indizes ------------------------------------------
if (-not $SkipCorpus) {
    Schritt "Normkorpus + Graph + Trigger-Index bauen"

    if ((-not (Test-Path -LiteralPath "corpus/gesetze.yaml")) -or $Voll -or $Force) {
        $init = @("korpus", "init"); if ($Voll) { $init += "--voll" }
        Cli $init
    } else { OkMsg "corpus/gesetze.yaml vorhanden" }

    $hatXml = @(Get-ChildItem -LiteralPath "corpus/gii" -Filter "*.xml" -ErrorAction SilentlyContinue).Count -gt 0
    if ((-not $hatXml) -or $Force) {
        Info "laedt Gesetzes-XML von gesetze-im-internet.de ..."
        Cli @("korpus", "fetch")
    } else { OkMsg "Gesetzes-XML vorhanden (corpus/gii/)" }

    if ((-not (Test-Path -LiteralPath "corpus/graph.sqlite")) -or $Force) {
        Cli @("graph", "build")
    } else { OkMsg "Normgraph vorhanden (corpus/graph.sqlite)" }

    if ((-not (Test-Path -LiteralPath "corpus/trigger.sqlite")) -or $Force) {
        Cli @("trigger", "build")
    } else { OkMsg "Trigger-Index vorhanden (corpus/trigger.sqlite)" }

    if ($WithEmbedding) {
        Schritt "Semantischer Embedding-Index (BGE-M3)"
        if ((-not (Test-Path -LiteralPath "corpus/embedding/vectors.npy")) -or $Force) {
            Info "bettet den Korpus ein (laedt BGE-M3 ~2,3 GB beim ersten Mal) ..."
            Cli @("embedding", "build")
        } else { OkMsg "Embedding-Index vorhanden (corpus/embedding/)" }
    }
} else { Info "Korpus uebersprungen (-SkipCorpus)" }

# --- Status + Abschluss ----------------------------------------------------
Schritt "Status"
Cli @("modell", "status")

Write-Host "`nDeploy abgeschlossen." -ForegroundColor Green
Write-Host "Engine starten (headless /v1):" -ForegroundColor Yellow
Write-Host '  $env:RISK_LAYER_TOKEN = (Get-Content .engine_token -Raw).Trim()'
Write-Host "  python -m risk_layer.web --host $BindHost --port $Port"
Write-Host "LLM ueber die API: POST /v1/llm/start  -  GET /v1/llm/status" -ForegroundColor Yellow

if ($Start) {
    Schritt "Engine starten"
    $env:RISK_LAYER_TOKEN = $token
    & python -m risk_layer.web --host $BindHost --port $Port
}
