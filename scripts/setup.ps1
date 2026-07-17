#Requires -Version 5.1
<#
.SYNOPSIS
  Setup/Deploy des Risk-Layer von "frischem Clone" zu lauffaehiger /v1-Engine:
  Abhaengigkeiten, LLM-Modell (GGUF) + llama.cpp-Vulkan-Engine herunterladen,
  Normkorpus/Graph/Trigger(/Embedding) bauen und Auth-Token anlegen.
  Idempotent - vorhandene Artefakte werden uebersprungen (ausser -Force).

.PARAMETER WithEmbedding
  Zusaetzlich den semantischen BGE-M3-Index bauen (installiert den hash-gepinnten
  Windows/Python-3.12-CPU-Stack und laedt das Modell beim ersten Build).
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
.PARAMETER RotateToken
  Rotiert das Auth-Token bewusst. -Force erneuert Build-Artefakte, rotiert aber
  kein Geheimnis und sperrt dadurch nicht unbeabsichtigt bestehende Clients aus.

.EXAMPLE
  ./scripts/setup.ps1                   # Deps + Modell + Engine + Korpus/Graph/Trigger
.EXAMPLE
  ./scripts/setup.ps1 -WithEmbedding -Voll
.EXAMPLE
  ./scripts/setup.ps1 -SkipModel -SkipEngine
.EXAMPLE
  ./scripts/setup.ps1 -Start
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
    [switch]$RotateToken,
    [switch]$Start,
    [switch]$Service,
    [string]$Modell = "",
    [ValidateSet("auto","cuda","rocm","hip","vulkan","cpu")][string]$Backend = "auto",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location -LiteralPath $RepoRoot

function Schritt($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Info($t)    { Write-Host "  $t" -ForegroundColor DarkGray }
function OkMsg($t)   { Write-Host "  [ok] $t" -ForegroundColor Green }

function Cli([string[]]$cmd) {
    Write-Host "  > python -m risk_layer.cli $($cmd -join ' ')" -ForegroundColor DarkGray
    & python -m risk_layer.cli @cmd
    if ($LASTEXITCODE -ne 0) { throw "Fehlgeschlagen (exit $LASTEXITCODE): risk-layer $($cmd -join ' ')" }
}
function PyOk([string]$code) { & python -c $code 2>$null; return ($LASTEXITCODE -eq 0) }

function Set-EngineTokenAcl([string]$Path) {
    # Sprachunabhaengige SIDs halten die ACL auf lokalisierten Windows-Systemen
    # portabel. Der Autostart laeuft als aktueller Benutzer; SYSTEM bleibt fuer
    # administrative Wiederherstellung leseberechtigt.
    $currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
    if ($null -eq $currentSid) { throw "Aktuelle Windows-Benutzer-SID konnte nicht ermittelt werden." }

    # Bestehenden Descriptor als Basis verwenden, damit Owner/Group unveraendert
    # bleiben und Set-Acl keine sonst unnoetigen Privilegien benoetigt.
    $acl = Get-Acl -LiteralPath $Path
    [void]$acl.SetAccessRuleProtection($true, $false)
    foreach ($existingRule in @($acl.Access)) {
        [void]$acl.RemoveAccessRuleAll($existingRule)
    }
    $allow = [System.Security.AccessControl.AccessControlType]::Allow
    $rules = @(
        [pscustomobject]@{ Sid = $currentSid; Rights = [System.Security.AccessControl.FileSystemRights]::FullControl },
        [pscustomobject]@{ Sid = New-Object System.Security.Principal.SecurityIdentifier("S-1-5-18"); Rights = [System.Security.AccessControl.FileSystemRights]::Read },
        [pscustomobject]@{ Sid = New-Object System.Security.Principal.SecurityIdentifier("S-1-5-32-544"); Rights = [System.Security.AccessControl.FileSystemRights]::FullControl }
    )
    foreach ($entry in $rules) {
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule($entry.Sid, $entry.Rights, $allow)
        [void]$acl.AddAccessRule($rule)
    }
    # Direkt nur die Datei-DACL setzen. Set-Acl versucht bei bereits
    # geschuetzten Dateien teils auch privilegierte Security-Sektionen zu setzen.
    [System.IO.File]::SetAccessControl([System.IO.Path]::GetFullPath($Path), $acl)
}

function Write-EngineTokenAtomically([string]$Path, [string]$Value) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $directory = [System.IO.Path]::GetDirectoryName($fullPath)
    $leaf = [System.IO.Path]::GetFileName($fullPath)
    $tempPath = Join-Path $directory (".$leaf.tmp." + [guid]::NewGuid().ToString("N"))
    $backupPath = Join-Path $directory (".$leaf.bak." + [guid]::NewGuid().ToString("N"))
    $stream = $null
    try {
        # Leere Datei exklusiv anlegen, ACL einschraenken, erst dann das Secret
        # schreiben. Replace/Move im selben Verzeichnis ist atomar.
        $stream = [System.IO.File]::Open(
            $tempPath,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        $stream.Dispose()
        $stream = $null
        Set-EngineTokenAcl -Path $tempPath
        [System.IO.File]::WriteAllText($tempPath, $Value, [System.Text.Encoding]::ASCII)
        if ([System.IO.File]::Exists($fullPath)) {
            [System.IO.File]::Replace($tempPath, $fullPath, $backupPath, $true)
        } else {
            [System.IO.File]::Move($tempPath, $fullPath)
        }
        Set-EngineTokenAcl -Path $fullPath
    } finally {
        if ($null -ne $stream) { $stream.Dispose() }
        if (Test-Path -LiteralPath $tempPath) { Remove-Item -LiteralPath $tempPath -Force }
        if (Test-Path -LiteralPath $backupPath) { Remove-Item -LiteralPath $backupPath -Force }
    }
}

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
    $emb = "cpu"
    $label = "Auto"
    switch ($Choice.ToLowerInvariant()) {
        { $_ -in @("1","nvidia","cuda") } {
            $script:Backend = "cuda"; $emb = "cpu"; $label = "NVIDIA/CUDA"
            Info "NVIDIA: llama.cpp nutzt CUDA; der sichere Embedding-Bootstrap bleibt auf hash-gepinntem CPU-Torch."
            break
        }
        { $_ -in @("2","amd","rocm","hip") } {
            $script:Backend = "rocm"; $emb = "cpu"; $label = "AMD/ROCm"
            Info "AMD: llama.cpp nutzt ROCm/HIP, wenn verfuegbar; Windows-PyTorch hat keinen ROCm-Pfad, Embeddings bleiben auf CPU."
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
            $script:Backend = "auto"; $emb = "cpu"; $label = "Auto"
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
    # Schlanker Produktionskern (PyYAML + numpy + Build-Backend), ohne pytest.
    & python -m pip install --require-hashes --only-binary=:all: -r requirements-runtime-lock.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install (Hash-Lock) fehlgeschlagen." }
    # Kein Editable-Install im Betrieb: Checkout-Aenderungen duerfen nicht
    # unbemerkt den vom Dienst importierten Code austauschen.
    $wheelDir = Join-Path ([System.IO.Path]::GetTempPath()) `
        ("risk-layer-wheel-" + [Guid]::NewGuid().ToString("N"))
    try {
        [void](New-Item -ItemType Directory -Path $wheelDir)
        & python -m pip wheel --no-build-isolation --no-deps --wheel-dir $wheelDir .
        if ($LASTEXITCODE -ne 0) { throw "Wheel-Build (Kern) fehlgeschlagen." }
        $wheels = @(Get-ChildItem -LiteralPath $wheelDir -Filter "risk_layer-*.whl" -File)
        if ($wheels.Count -ne 1) { throw "Erwartete genau ein gebautes Risk-Layer-Wheel." }
        & python -m pip install --no-index --no-deps --force-reinstall $wheels[0].FullName
        if ($LASTEXITCODE -ne 0) { throw "Wheel-Installation (Kern) fehlgeschlagen." }
    } finally {
        if (Test-Path -LiteralPath $wheelDir) {
            Remove-Item -LiteralPath $wheelDir -Recurse -Force
        }
    }
    if ($WithEmbedding) {
        if ($pv -ne "3.12") { throw "-WithEmbedding ist nur mit dem vorhandenen Windows/Python-3.12-Hashlock erlaubt (gefunden: $pv)." }
        if (([string]$env:RISK_LAYER_EMB_DEVICE).Trim().ToLowerInvariant() -ne "cpu") {
            throw "Der automatische Embedding-Bootstrap ist CPU-only. GPU-Torch separat auditiert bereitstellen und -SkipDeps nutzen."
        }
        Info "Embedding+Reranker -> hash-gepinnter Windows/Python-3.12-CPU-Lock"
        & python -m pip install --require-hashes --only-binary=:all: -r requirements-embedding-windows-cpu-py312-lock.txt
        if ($LASTEXITCODE -ne 0) { throw "pip install (Embedding-Hashlock) fehlgeschlagen." }
        OkMsg "Abhaengigkeiten installiert (Runtime + sentence-transformers + CPU-Torch)"
    } else {
        OkMsg "Abhaengigkeiten installiert (PyYAML + numpy)"
    }
    if ($script:InstallQuanten) {
        if ($pv -ne "3.12") { throw "-WithQuanten ist nur mit dem vorhandenen Windows/Python-3.12-Hashlock erlaubt (gefunden: $pv)." }
        Info "Quantenlos-Extras -> hash-gepinnter Windows/Python-3.12-Lock"
        & python -m pip install --require-hashes --only-binary=:all: -r requirements-quanten-windows-py312-lock.txt
        if ($LASTEXITCODE -ne 0) { throw "pip install Quantenlos-Extras fehlgeschlagen." }
        OkMsg "Quantenlos-Extras installiert"
    } else {
        Info "Quantenlos-Extras uebersprungen - /v1/los qpu/simulator meldet sonst fehlende Betriebsvoraussetzung"
    }
    & python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Installierte Python-Abhaengigkeiten sind inkonsistent (pip check)." }
    OkMsg "Python-Abhaengigkeiten konsistent (pip check)"
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
        # spaCy + Modell kommen aus dem separaten Hash-Lock. Fehler hier sind NICHT
        # fatal (wie der Engine-Download):
        # die Anonymisierung degradiert auf den deterministischen Regex-Pfad.
        Info "installiert spaCy (Extra nlp) + Modell de_core_news_lg ..."
        & python -m pip install --require-hashes --only-binary=:all: -r requirements-ner-lock.txt
        if ($LASTEXITCODE -eq 0) {
            & python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('de_core_news_lg') else 1)"
        }
        if ($LASTEXITCODE -eq 0) { & python -m pip check }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [warnung] NER-Provisionierung fehlgeschlagen - NER-Schicht inaktiv," -ForegroundColor Yellow
            Write-Host "  Engine laeuft regex-only weiter. Nachholen mit:" -ForegroundColor Yellow
            # Nachhol-Kommando mit ';' statt '&&' — Windows PowerShell 5.1 kennt kein '&&'.
            Write-Host '    pip install --require-hashes --only-binary=:all: -r requirements-ner-lock.txt' -ForegroundColor Yellow
        } else {
            OkMsg "NER-Schicht installiert (de_core_news_lg)"
        }
    }
} else { Info "NER uebersprungen (-OhneNer) - Anonymisierung laeuft regex-only" }

# --- Auth-Token ------------------------------------------------------------
Schritt "Auth-Token (.engine_token)"
$tokenPath = Join-Path $RepoRoot ".engine_token"
if (Test-Path -LiteralPath $tokenPath) {
    $tokenItem = Get-Item -LiteralPath $tokenPath -Force
    if ($tokenItem.PSIsContainer -or (($tokenItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
        throw ".engine_token muss eine regulaere Datei sein und darf kein Link sein."
    }
    # Auch Dateien aus aelteren Deploy-Laeufen auf die restriktive ACL heben.
    Set-EngineTokenAcl -Path $tokenPath
}
# Artefakt-Rebuild (-Force) und Credential-Rotation sind getrennte
# Betriebsereignisse. Nur -RotateToken sperrt bestehende Clients aus.
if ((-not (Test-Path -LiteralPath $tokenPath)) -or $RotateToken) {
    $tok = & python -c "import secrets; print(secrets.token_hex(32))"
    $tokenExitCode = $LASTEXITCODE
    $tok = ([string]$tok).Trim()
    if ($tokenExitCode -ne 0 -or $tok -notmatch "^[0-9a-f]{64}$") {
        throw "Auth-Token konnte nicht sicher erzeugt werden."
    }
    Write-EngineTokenAtomically -Path $tokenPath -Value $tok
    OkMsg "neues Token erzeugt (.engine_token, restriktive ACL, gitignored)"
} else { OkMsg ".engine_token vorhanden (wiederverwendet)" }
Set-EngineTokenAcl -Path $tokenPath
$token = (Get-Content -LiteralPath $tokenPath -Raw).Trim()
# Token NICHT auf die Konsole drucken (Scrollback/Logs) - nur Pfad und bewussten Abruf nennen.
Info "Token gespeichert: $tokenPath (nicht angezeigt)"
Info "Bewusst anzeigen: Get-Content -LiteralPath .engine_token -Raw"

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
        # Download-Fehler NICHT fatal (wie setup.sh): ohne llama-server laufen alle
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
        & python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', revision='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e')"
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
    Info 'NER inaktiv (-OhneNer) - Anonymisierung regex-only; nachholen: pip install --require-hashes --only-binary=:all: -r requirements-ner-lock.txt'
} else {
    Info 'NER inaktiv (Modell fehlt) - Anonymisierung regex-only; nachholen: pip install --require-hashes --only-binary=:all: -r requirements-ner-lock.txt'
}

Maybe-PromptDockerNativeAccess
$baseUrl = Client-BaseUrl
$serviceStateDir = Join-Path $RepoRoot ".signal"
$serviceStatePath = Join-Path $serviceStateDir "service.json"
if (-not (Test-Path -LiteralPath $serviceStateDir -PathType Container)) {
    [void][System.IO.Directory]::CreateDirectory($serviceStateDir)
}
$serviceState = [ordered]@{
    schema = "risk-layer-checkout-service/v1"
    host = $BindHost
    port = $Port
    scope = "windows-task"
}
Write-EngineTokenAtomically -Path $serviceStatePath `
    -Value (($serviceState | ConvertTo-Json -Compress) + "`n")
Write-Host "`nDeploy abgeschlossen." -ForegroundColor Green
Write-Host "Kanzleikonsole .env (Host-Prozess, nicht Docker):" -ForegroundColor Yellow
Write-Output "RISK_LAYER_URL=$baseUrl"
Write-Output "# RISK_LAYER_TOKEN aus .engine_token uebernehmen (nicht angezeigt)"
Write-Output "RISK_LAYER_HEALTH_PATH=/v1/health"
Write-Output "RISK_LAYER_ANALYSE_PATH=/v1/analyse"
Write-Host "Kanzleikonsole .env (Docker, Risk-Layer im selben Compose-Netz):" -ForegroundColor Yellow
Write-Output "RISK_LAYER_URL=http://risk-layer:8000"
Write-Output "# RISK_LAYER_TOKEN aus .engine_token uebernehmen (nicht angezeigt)"
Write-Output "RISK_LAYER_HEALTH_PATH=/v1/health"
Write-Output "RISK_LAYER_ANALYSE_PATH=/v1/analyse"
if ($script:DockerNativeUrl) {
    Write-Host "Kanzleikonsole .env (Docker-App -> nativer Host-Risk-Layer):" -ForegroundColor Yellow
    Write-Output "RISK_LAYER_URL=$script:DockerNativeUrl"
    Write-Output "# RISK_LAYER_TOKEN aus .engine_token uebernehmen (nicht angezeigt)"
    Write-Output "RISK_LAYER_HEALTH_PATH=/v1/health"
    Write-Output "RISK_LAYER_ANALYSE_PATH=/v1/analyse"
}
Write-Host "Token bei Bedarf bewusst anzeigen:" -ForegroundColor Yellow
Write-Host "  Get-Content -LiteralPath .engine_token -Raw"
Write-Host "Alltagsbetrieb (Engine + LLM als Benutzer-Task):" -ForegroundColor Yellow
Write-Host "  .\signal.ps1 start | .\signal.ps1 stop | .\signal.ps1 restart | .\signal.ps1 status"

if ($Service) {
    Schritt "Windows-Autostart einrichten (begrenzter Benutzer-Task, --llm-autostart)"
    & "$RepoRoot/deploy/install-service.ps1" -Port $Port -BindHost $BindHost `
        -EmbeddingDevice $env:RISK_LAYER_EMB_DEVICE
    if (-not $?) { throw "Windows-Autostart konnte nicht installiert werden." }
    $healthHost = if ($BindHost -in @("0.0.0.0", "::", "")) { "127.0.0.1" } else { $BindHost }
    if ($healthHost.Contains(":")) { $healthHost = "[$healthHost]" }
    & python -P -m risk_layer.ops_cli doctor `
        --url "http://${healthHost}:$Port" --token-file $tokenPath `
        --timeout 60 --require-layer matcher --allow-insecure-http
    if ($LASTEXITCODE -ne 0) { throw "Health/Status nach Autostart fehlgeschlagen." }
    Write-Host "`nAutostart eingerichtet - startet bei Benutzeranmeldung. (Foreground-Start uebersprungen.)" -ForegroundColor Green
}
elseif ($Start) {
    Schritt "Engine starten (Engine + LLM, Vordergrund)"
    $env:RISK_LAYER_TOKEN = $token
    & python -m risk_layer.web --host $BindHost --port $Port --llm-autostart
}
