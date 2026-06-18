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
.PARAMETER WithQuanten
  Quantenlos-Extras fuer /v1/los/* installieren (qiskit, qiskit-aer,
  qiskit-ibm-runtime).
.PARAMETER OhneNer
  Die NER-Schicht der Anonymisierung (spaCy + de_core_news_lg, ~0,5 GB) NICHT
  installieren. Default ist MIT: NER staerkt das Geheimnisschutz-Gate (Par. 203
  StGB) - es faengt Personen ohne Anrede / Firmen ohne Rechtsform, die die
  Regexe nicht sehen. Opt-out fuer schlanke/Offline-Installationen - die
  Anonymisierung laeuft dann rein regex-basiert weiter (Laufzeit-Schalter:
  RISK_LAYER_NER=aus).
.PARAMETER Voll
  Breiter Steuer-Korpus (~21 Gesetze) statt nur der im Katalog referenzierten.
.PARAMETER Modell
  Modell-Key statt des Defaults (granite-4.1-8b). Siehe 'risk-layer modell list'.
.PARAMETER Backend
  GPU-Backend der Engine: auto (Default; NVIDIA->cuda, AMD->rocm/hip,
  Intel->vulkan, sonst CPU) oder explizit cuda|rocm|hip|vulkan|cpu.
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
    [switch]$WithQuanten,
    [switch]$OhneQuanten,
    [switch]$OhneNer,
    [switch]$Voll,
    [switch]$Force,
    [switch]$Start,
    [switch]$Service,
    [string]$Modell = "",
    [ValidateSet("auto","cuda","rocm","hip","vulkan","cpu")][string]$Backend = "auto",
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

function Test-Interactive {
    return [Environment]::UserInteractive
}

function Detect-Implementation {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { return "nvidia" }
    $names = @()
    try {
        $names = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | ForEach-Object { $_.Name })
    } catch { $names = @() }
    $blob = ($names -join " ").ToLowerInvariant()
    if ($blob -match "nvidia|geforce|rtx|quadro|tesla") { return "nvidia" }
    if ($blob -match "amd|radeon|instinct") { return "amd" }
    return "cpu"
}

function Apply-ImplementationChoice([string]$Choice) {
    $emb = "cuda"
    $label = "Auto"
    switch ($Choice.ToLowerInvariant()) {
        { $_ -in @("1","nvidia","cuda") } {
            $script:Backend = "cuda"; $emb = "cuda"; $label = "NVIDIA/CUDA"
            Info "NVIDIA: llama.cpp nutzt CUDA-Builds, Embeddings/Reranker laufen ueber torch CUDA."
            break
        }
        { $_ -in @("2","amd","rocm","hip") } {
            $script:Backend = "rocm"; $emb = "cuda"; $label = "AMD/ROCm"
            Info "AMD: llama.cpp nutzt ROCm/HIP, wenn verfuegbar. PyTorch/ROCm meldet Geraete als cuda, daher bleibt RISK_LAYER_EMB_DEVICE=cuda korrekt."
            break
        }
        { $_ -in @("3","intel","vulkan") } {
            $script:Backend = "vulkan"; $emb = "cpu"; $label = "Intel/Vulkan"
            Info "Intel/andere iGPU: llama.cpp laeuft ueber Vulkan. Embeddings bleiben standardmaessig CPU."
            break
        }
        { $_ -in @("4","cpu") } {
            $script:Backend = "cpu"; $emb = "cpu"; $label = "CPU"
            Info "CPU: robust und ohne GPU-Treiber, aber langsamer."
            break
        }
        default {
            $script:Backend = "auto"; $emb = "cuda"; $label = "Auto"
            Info "Auto: Vendor-Erkennung entscheidet beim Engine-Download."
        }
    }
    if (-not $env:RISK_LAYER_EMB_DEVICE) { $env:RISK_LAYER_EMB_DEVICE = $emb }
    else { Info "RISK_LAYER_EMB_DEVICE ist bereits gesetzt: $env:RISK_LAYER_EMB_DEVICE" }
    OkMsg "Implementierung: $label (Engine-Backend: $script:Backend, Embedding-Geraet: $env:RISK_LAYER_EMB_DEVICE)"
}

function Choose-Implementation {
    Schritt "Implementierung waehlen"
    if ($PSBoundParameters.ContainsKey("Backend")) {
        Apply-ImplementationChoice $Backend
        return
    }
    $detected = Detect-Implementation
    $default = switch ($detected) {
        "nvidia" { "1" }
        "amd" { "2" }
        default { "4" }
    }
    if (Test-Interactive) {
        Info "Erkannt: $detected. Diese Wahl betrifft llama.cpp-Engine und ML-Geraet."
        Write-Host "  1) NVIDIA/CUDA  - CUDA fuer Engine und Embeddings"
        Write-Host "  2) AMD/ROCm     - ROCm/HIP fuer Engine, torch-ROCm fuer Embeddings"
        Write-Host "  3) Intel/Vulkan - manuell waehlen, wenn wirklich Intel-GPU/Vulkan nutzbar ist"
        Write-Host "  4) CPU          - kein GPU-Treiber noetig, langsamer"
        $choice = Read-Host "Auswahl [$default]"
        if (-not $choice) { $choice = $default }
        Apply-ImplementationChoice $choice
    } else {
        Info "Nicht interaktiv: Implementierung wird automatisch erkannt (erkannt: $detected)."
        Apply-ImplementationChoice $default
    }
}

$script:InstallQuanten = [bool]$WithQuanten
function Choose-QuantenExtras {
    if ($SkipDeps -or $PSBoundParameters.ContainsKey("WithQuanten") -or $PSBoundParameters.ContainsKey("OhneQuanten")) {
        if ($OhneQuanten) { $script:InstallQuanten = $false }
        return
    }
    if (-not (Test-Interactive)) {
        Info "Nicht interaktiv: Quantenlos-Extras werden nicht installiert (bei Bedarf -WithQuanten)."
        return
    }

    Schritt "Quantenlos"
    Info "/v1/los/* braucht fuer qpu/simulator die optionalen qiskit-Extras."
    Info "Ohne Extras liefert die Engine eine 503-Betriebsvoraussetzung; csprng bleibt lokal moeglich."
    $antwort = Read-Host "Quantenlos-Extras jetzt installieren? [J/n]"
    $script:InstallQuanten = ($antwort.Trim().ToLowerInvariant() -notin @("n", "nein", "no"))
}

function Client-BaseUrl {
    $clientHost = $BindHost
    if (@("", "0.0.0.0", "::") -contains $clientHost) { $clientHost = "127.0.0.1" }
    if ($clientHost.Contains(":") -and -not $clientHost.StartsWith("[")) {
        $clientHost = "[$clientHost]"
    }
    return "http://${clientHost}:$Port"
}

$script:DockerNativeUrl = ""
function Maybe-PromptDockerNativeAccess {
    if ($PSBoundParameters.ContainsKey("BindHost")) { return }
    if (-not (Test-Interactive)) { return }

    Schritt "Docker-Zugriff"
    Info "127.0.0.1 ist nur vom Host erreichbar; im Container zeigt es auf den Container selbst."
    Info "Am saubersten: Risk-Layer als Compose-Service starten und RISK_LAYER_URL=http://risk-layer:8000 setzen."
    $antwort = Read-Host "Soll eine Docker-App diese native Host-Engine erreichen koennen? [j/N]"
    if ($antwort.Trim().ToLowerInvariant() -notin @("j", "ja", "y", "yes")) { return }

    $script:DockerNativeUrl = "http://host.docker.internal:$Port"
    Info "Docker Desktop erreicht Host-Dienste ueber host.docker.internal; Bind bleibt $BindHost."
}

# --- Python ----------------------------------------------------------------
Schritt "Python pruefen"
$pv = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])")
if ($LASTEXITCODE -ne 0) { throw "python nicht gefunden - Python >= 3.11 installieren." }
$maj, $min = $pv.Split('.')
if ([int]$maj -lt 3 -or ([int]$maj -eq 3 -and [int]$min -lt 11)) {
    throw "Python >= 3.11 noetig (gefunden $pv)."
}
OkMsg "Python $pv"
Choose-Implementation
Choose-QuantenExtras

# --- Abhaengigkeiten -------------------------------------------------------
if (-not $SkipDeps) {
    Schritt "Abhaengigkeiten installieren"
    & python -m pip install --upgrade pip | Out-Null
    # Kern (PyYAML) + dev exakt aus dem Lock (deterministisch ueber die Flotte,
    # identisch zu CI/Dockerfile); das Paket selbst mit --no-deps, damit die
    # >=-Ranges aus pyproject.toml das Lock nicht ueberstimmen.
    # numpy bewusst ungepinnt (GPU-/Plattform-Matrix, ML-Extra).
    & python -m pip install -r requirements-lock.txt "numpy>=1.26"
    if ($LASTEXITCODE -ne 0) { throw "pip install (Lock + numpy) fehlgeschlagen." }
    & python -m pip install --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw "pip install (Kern) fehlgeschlagen." }
    if ($WithEmbedding) {
        # Bewusste Luecke: kein --cuda-wheel-Pendant zu deploy.sh (torch-CUDA-Vorabinstallation) — Windows-Hosts nutzen das torch, das sentence-transformers waehlt.
        Info "Embedding+Reranker gewuenscht -> sentence-transformers (+ torch, gross) + hf_transfer"
        # bewusst ungepinnt (GPU-/Plattform-Matrix) — ML-Extras sind nicht im Lock.
        & python -m pip install "sentence-transformers>=2.7" hf_transfer
        if ($LASTEXITCODE -ne 0) { throw "pip install sentence-transformers fehlgeschlagen." }
        OkMsg "Abhaengigkeiten installiert (PyYAML + numpy + sentence-transformers + hf_transfer)"
    } else {
        OkMsg "Abhaengigkeiten installiert (PyYAML + numpy)"
    }
    if ($script:InstallQuanten) {
        Info "Quantenlos-Extras gewuenscht -> qiskit + qiskit-aer + qiskit-ibm-runtime"
        & python -m pip install -e ".[quanten]"
        if ($LASTEXITCODE -ne 0) { throw "pip install Quantenlos-Extras fehlgeschlagen." }
        OkMsg "Quantenlos-Extras installiert"
    } else {
        Info "Quantenlos-Extras uebersprungen - /v1/los qpu/simulator meldet sonst fehlende Betriebsvoraussetzung"
    }
} else { Info "Deps uebersprungen (-SkipDeps)" }

# --- NER-Schicht (Anonymisierung, Par. 203 StGB) -----------------------------
# ACHTUNG Encoding: Datei ist UTF-8 OHNE BOM -> Windows PowerShell 5.1 liest sie
# als ANSI. In STRINGS daher nur ASCII (ein Gedankenstrich wuerde als 0x94 = ”
# gelesen und beendet die Zeichenkette!) - in Kommentaren ist er unkritisch.
if (-not $OhneNer) {
    Schritt "NER-Schicht der Anonymisierung (spaCy + de_core_news_lg)"
    # Idempotent: Modell-Paket schon importierbar (lg, Fallback sm) -> ueberspringen.
    # find_spec statt Modell-Laden: reiner Metadaten-Lookup (laedt die ~0,5 GB NICHT).
    $nerDa = PyOk "import importlib.util,sys; sys.exit(0 if any(importlib.util.find_spec(m) for m in ('de_core_news_lg','de_core_news_sm')) else 1)"
    if ($nerDa -and -not $Force) {
        OkMsg "spaCy-Modell bereits installiert - uebersprungen"
    } else {
        # spaCy Range-basiert wie die anderen ML-Extras (bewusst nicht im Lock, siehe
        # requirements.in); -e haelt die editierbare Kern-Installation von oben intakt
        # (ein nicht-editierbares ".[nlp]" wuerde sie durch eine Kopie ersetzen).
        # `spacy download` loest die zur installierten spaCy-Version passende
        # Modellversion auf. Fehler hier sind NICHT fatal (wie der Engine-Download):
        # die Anonymisierung degradiert auf den deterministischen Regex-Pfad.
        Info "installiert spaCy (Extra nlp) + Modell de_core_news_lg ..."
        & python -m pip install -e ".[nlp]"
        if ($LASTEXITCODE -eq 0) { & python -m spacy download de_core_news_lg }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [warnung] NER-Provisionierung fehlgeschlagen - NER-Schicht inaktiv," -ForegroundColor Yellow
            Write-Host "  Engine laeuft regex-only weiter. Nachholen mit:" -ForegroundColor Yellow
            # Nachhol-Kommando mit ';' statt '&&' — Windows PowerShell 5.1 kennt kein '&&'.
            Write-Host '    pip install -e ".[nlp]"; python -m spacy download de_core_news_lg' -ForegroundColor Yellow
        } else {
            OkMsg "NER-Schicht installiert (de_core_news_lg)"
        }
    }
} else { Info "NER uebersprungen (-OhneNer) - Anonymisierung laeuft regex-only" }

# --- Auth-Token ------------------------------------------------------------
Schritt "Auth-Token (.engine_token)"
if ((-not (Test-Path -LiteralPath ".engine_token")) -or $Force) {
    $tok = & python -c "import secrets; print(secrets.token_hex(32))"
    Set-Content -LiteralPath ".engine_token" -Value $tok -NoNewline -Encoding ascii
    OkMsg "neues Token erzeugt (.engine_token, gitignored)"
} else { OkMsg ".engine_token vorhanden (wiederverwendet)" }
$token = (Get-Content -LiteralPath ".engine_token" -Raw).Trim()
# Token NICHT auf die Konsole drucken (Scrollback/Logs) — nur den Speicherort nennen.
Info "RISK_LAYER_TOKEN liegt in .engine_token (nicht angezeigt)"

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
        # Download-Fehler NICHT fatal (wie deploy.sh): ohne llama-server laufen alle
        # deterministischen Layer + /v1; nur 'mitLLM:true' ist inaktiv.
        & python -m risk_layer.cli modell engine --backend $Backend
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [warnung] Engine-Download fehlgeschlagen. llama-server separat auf den PATH legen." -ForegroundColor Yellow
            Info "Ohne ihn laufen alle deterministischen Layer + /v1; nur mitLLM:true ist inaktiv."
        } else {
            OkMsg "Engine bereitgestellt"
        }
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

    # Trigger neu, wenn er fehlt, -Force, ODER der Graph neuer ist (Korpus-Aenderung,
    # z. B. neue Gesetze/DVs -> der Index muss sie aufnehmen).
    $trigDa = Test-Path -LiteralPath "corpus/trigger.sqlite"
    $graphNeuerT = $trigDa -and ((Get-Item "corpus/graph.sqlite").LastWriteTime -gt (Get-Item "corpus/trigger.sqlite").LastWriteTime)
    if ((-not $trigDa) -or $Force -or $graphNeuerT) {
        Cli @("trigger", "build")
    } else { OkMsg "Trigger-Index aktuell (corpus/trigger.sqlite)" }

    if ($WithEmbedding) {
        Schritt "Embedding-Index (BGE-M3) + Reranker (Schicht 1.7 / 1.7b)"
        python -c "import hf_transfer" 2>$null
        if ($LASTEXITCODE -eq 0) { $env:HF_HUB_ENABLE_HF_TRANSFER = "1" }
        # Bauen, wenn Index fehlt, -Force, der Katalog NEUER ist (neue Paraphrasen-
        # Anker in catalog/begriffe.yaml) ODER der Graph neuer ist (Korpus-Aenderung,
        # z. B. neue Gesetze/DVs) — beides MUSS mit-eingebettet werden.
        $idxDa = Test-Path -LiteralPath "corpus/embedding/vectors.npy"
        $idxZeit = if ($idxDa) { (Get-Item "corpus/embedding/vectors.npy").LastWriteTime } else { $null }
        $katNeuer = $idxDa -and ((Get-Item "catalog/begriffe.yaml").LastWriteTime -gt $idxZeit)
        $graphNeuer = $idxDa -and ((Get-Item "corpus/graph.sqlite").LastWriteTime -gt $idxZeit)
        if ((-not $idxDa) -or $Force -or $katNeuer -or $graphNeuer) {
            Info "bettet den Korpus ein (laedt BGE-M3 ~2,3 GB beim ersten Mal) ..."
            Cli @("embedding", "build")
        } else { OkMsg "Embedding-Index aktuell (corpus/embedding/)" }
        # Reranker (Schicht 1.7b) ist in der Engine fest aktiv (GPU). Modell vorab
        # cachen, sonst laedt der ERSTE Request ~600 MB nach. Nur Download, kein GPU.
        Info "cacht den Reranker (bge-reranker-v2-m3, ~600 MB) ..."
        & python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3')"
        if ($LASTEXITCODE -eq 0) { OkMsg "Reranker gecacht" } else { Info "Reranker-Cache fehlgeschlagen (laedt sonst beim 1. Request)" }
    }
} else { Info "Korpus uebersprungen (-SkipCorpus)" }

# --- Status + Abschluss ----------------------------------------------------
Schritt "Status"
Cli @("modell", "status")
# NER-Status nur per find_spec (Metadaten-Lookup) — das Modell wird NICHT geladen
# (Laden waere langsam und unnoetig fuer eine reine Installations-Auskunft).
$nerModell = (& python -c "import importlib.util; print(next((m for m in ('de_core_news_lg','de_core_news_sm') if importlib.util.find_spec(m)), ''))" 2>$null)
if ($LASTEXITCODE -ne 0) { $nerModell = "" }
if ($nerModell) {
    OkMsg "NER aktiv (Modell $nerModell)"
    if (@("aus","off","none","0") -contains ([string]$env:RISK_LAYER_NER).Trim().ToLower()) {
        Info "Hinweis: RISK_LAYER_NER=aus ist gesetzt - Laufzeit nutzt den Regex-Pfad"
    }
} elseif ($OhneNer) {
    Info 'NER inaktiv (-OhneNer) - Anonymisierung regex-only; nachholen: pip install -e ".[nlp]"; python -m spacy download de_core_news_lg'
} else {
    Info 'NER inaktiv (Modell fehlt) - Anonymisierung regex-only; nachholen: pip install -e ".[nlp]"; python -m spacy download de_core_news_lg'
}

Maybe-PromptDockerNativeAccess
$baseUrl = Client-BaseUrl
Write-Host "`nDeploy abgeschlossen." -ForegroundColor Green
Write-Host "Kanzleikonsole .env (Host-Prozess, nicht Docker):" -ForegroundColor Yellow
Write-Output "RISK_LAYER_URL=$baseUrl"
Write-Output "RISK_LAYER_TOKEN=$token"
Write-Output "RISK_LAYER_HEALTH_PATH=/v1/health"
Write-Output "RISK_LAYER_ANALYSE_PATH=/v1/analyse"
Write-Host "Kanzleikonsole .env (Docker, Risk-Layer im selben Compose-Netz):" -ForegroundColor Yellow
Write-Output "RISK_LAYER_URL=http://risk-layer:8000"
Write-Output "RISK_LAYER_TOKEN=$token"
Write-Output "RISK_LAYER_HEALTH_PATH=/v1/health"
Write-Output "RISK_LAYER_ANALYSE_PATH=/v1/analyse"
if ($script:DockerNativeUrl) {
    Write-Host "Kanzleikonsole .env (Docker-App -> nativer Host-Risk-Layer):" -ForegroundColor Yellow
    Write-Output "RISK_LAYER_URL=$script:DockerNativeUrl"
    Write-Output "RISK_LAYER_TOKEN=$token"
    Write-Output "RISK_LAYER_HEALTH_PATH=/v1/health"
    Write-Output "RISK_LAYER_ANALYSE_PATH=/v1/analyse"
}
Write-Host "Ein-Befehl-Start (Engine + LLM):" -ForegroundColor Yellow
Write-Host '  $env:RISK_LAYER_TOKEN = (Get-Content .engine_token -Raw).Trim()'
Write-Host "  python -m risk_layer.web --host $BindHost --port $Port --llm-autostart"
Write-Host "Doppelklick-Alltagsstart: start.cmd   |   Als Dienst: ./deploy.ps1 -Service" -ForegroundColor Yellow

if ($Service) {
    Schritt "Windows-Autostart einrichten (Dienst/Task, --llm-autostart)"
    & "$PSScriptRoot/install-service.ps1" -Port $Port -BindHost $BindHost
    Write-Host "`nDienst eingerichtet - startet mit dem Rechner. (Foreground-Start uebersprungen.)" -ForegroundColor Green
}
elseif ($Start) {
    Schritt "Engine starten (Engine + LLM, Vordergrund)"
    $env:RISK_LAYER_TOKEN = $token
    & python -m risk_layer.web --host $BindHost --port $Port --llm-autostart
}
