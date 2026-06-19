#!/usr/bin/env bash
#
# Setup/Deploy des Risk-Layer (Linux) von "frischem Clone" zu lauffaehiger
# /v1-Engine: Abhaengigkeiten, LLM-Modell (GGUF) + passende llama.cpp-Engine
# herunterladen, Normkorpus/Graph/Trigger(/Embedding) bauen, Auth-Token. Idempotent.
#
# NER-Schicht (Anonymisierung, §203-Gate): standardmaessig werden spaCy +
# de_core_news_lg MIT installiert — NER staerkt das §203-Gate (faengt Personen
# ohne Anrede / Firmen ohne Rechtsform, die die Regexe nicht sehen). Opt-out
# fuer schlanke/Offline-Installationen: --ohne-ner; die Anonymisierung laeuft
# dann rein regex-basiert weiter (Laufzeit-Schalter: RISK_LAYER_NER=aus).
#
# Beim interaktiven Deploy fragt das Skript die Implementierung ab:
# NVIDIA/CUDA, AMD/ROCm, Intel/Vulkan oder CPU. Auf NVIDIA-Boxen (z. B.
# B300/Blackwell) wird zudem ein Blackwell-faehiges torch (--cuda-wheel, Default
# cu128) installiert, sonst nutzt sentence-transformers die GPU nicht. Falls kein
# CUDA-llama-server-Prebuilt: Discovery/LLM spricht jeden OpenAI-kompatiblen
# /v1-Endpoint an. Ohne llama-server laufen alle deterministischen Layer + /v1;
# nur 'mitLLM:true' ist inaktiv.
#
# Beispiele:
#   ./deploy.sh                         # Deps + Modell + Engine + Korpus/Graph/Trigger
#   ./deploy.sh --with-embedding --voll # zusaetzlich Embedding-Index + breiter Korpus
#   ./deploy.sh --with-quanten          # Quantenlos-Extras fuer /v1/los/* installieren
#   ./deploy.sh --skip-engine           # Engine separat bereitstellen
#   ./deploy.sh --ohne-ner              # ohne spaCy-NER (schlank/offline)
#   ./deploy.sh --start                 # Setup, dann Engine starten
#
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_DEPS=0; SKIP_MODEL=0; SKIP_ENGINE=0; SKIP_CORPUS=0; WITH_EMBEDDING=0; WITH_QUANTEN=0; OHNE_NER=0; VOLL=0; FORCE=0; START=0; SERVICE=0
BACKEND_EXPLICIT=0; HOST_EXPLICIT=0; QUANTEN_EXPLICIT=0
MODELL=""; BACKEND="auto"; BIND_HOST="127.0.0.1"; PORT=8000; CUDA_WHEEL="cu128"
DOCKER_NATIVE_URL=""
VENV_DIR="${RISK_LAYER_VENV:-.venv}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-deps)      SKIP_DEPS=1 ;;
    --skip-model)     SKIP_MODEL=1 ;;
    --skip-engine)    SKIP_ENGINE=1 ;;
    --skip-corpus)    SKIP_CORPUS=1 ;;
    --with-embedding) WITH_EMBEDDING=1 ;;
    --with-quanten)   WITH_QUANTEN=1; QUANTEN_EXPLICIT=1 ;;
    --ohne-quanten)   WITH_QUANTEN=0; QUANTEN_EXPLICIT=1 ;;
    --ohne-ner)       OHNE_NER=1 ;;
    --voll)           VOLL=1 ;;
    --force)          FORCE=1 ;;
    --start)          START=1 ;;
    --service)        SERVICE=1 ;;
    --modell)         MODELL="${2:?--modell braucht einen Wert}"; shift ;;
    --backend)        BACKEND="${2:?--backend braucht einen Wert}"; BACKEND_EXPLICIT=1; shift ;;
    --host)           BIND_HOST="${2:?--host braucht einen Wert}"; HOST_EXPLICIT=1; shift ;;
    --port)           PORT="${2:?--port braucht einen Wert}"; shift ;;
    --cuda-wheel)     CUDA_WHEEL="${2:?--cuda-wheel braucht einen Wert (z. B. cu128, cu124)}"; shift ;;
    -h|--help)        sed -n '3,27p' "$0"; exit 0 ;;
    *) echo "unbekanntes Flag: $1" >&2; exit 2 ;;
  esac
  shift
done

# Farben nur, wenn Terminal
if [[ -t 1 ]]; then C='\033[36m'; G='\033[32m'; D='\033[90m'; Y='\033[33m'; N='\033[0m'
else C=''; G=''; D=''; Y=''; N=''; fi
schritt() { printf "\n${C}=== %s ===${N}\n" "$1"; }
info()    { printf "${D}  %s${N}\n" "$1"; }
ok()      { printf "${G}  [ok] %s${N}\n" "$1"; }
can_prompt() { [[ -t 0 && -t 1 ]]; }

ask_yes_no() {
  local frage="$1" default="${2:-n}" prompt antwort
  if [[ "$default" == "j" ]]; then prompt="[J/n]"; else prompt="[j/N]"; fi
  while true; do
    printf "%s %s " "$frage" "$prompt"
    IFS= read -r antwort || return 1
    antwort="${antwort:-$default}"
    case "$(printf '%s' "$antwort" | tr '[:upper:]' '[:lower:]')" in
      j|ja|y|yes) return 0 ;;
      n|nein|no) return 1 ;;
      *) printf "Bitte j oder n eingeben.\n" ;;
    esac
  done
}

detected_impl() {
  if command -v nvidia-smi >/dev/null 2>&1; then printf "nvidia\n"; return; fi
  if command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1 || [[ -d /opt/rocm ]]; then
    printf "amd\n"; return
  fi
  if command -v lspci >/dev/null 2>&1; then
    local blob
    blob="$(lspci 2>/dev/null | tr '[:upper:]' '[:lower:]' || true)"
    if [[ "$blob" == *nvidia* || "$blob" == *geforce* || "$blob" == *" rtx "* ]]; then printf "nvidia\n"; return; fi
    if [[ "$blob" == *amd* || "$blob" == *radeon* || "$blob" == *instinct* ]]; then printf "amd\n"; return; fi
  fi
  printf "cpu\n"
}

apply_implementation_choice() {
  local choice="$1" emb="cuda" label=""
  case "$choice" in
    1|nvidia|cuda)
      BACKEND="cuda"; emb="cuda"; label="NVIDIA/CUDA"
      info "NVIDIA: Embeddings/Reranker laufen ueber torch CUDA. Unter Linux gibt es fuer llama.cpp oft kein CUDA-Prebuilt; das Skript nimmt dann Vulkan/CPU-Fallback oder nutzt eine separat bereitgestellte Engine."
      ;;
    2|amd|rocm|hip)
      BACKEND="rocm"; emb="cuda"; label="AMD/ROCm"
      info "AMD: llama.cpp nutzt ROCm/HIP, wenn ein passendes Release verfuegbar ist. PyTorch/ROCm meldet Geraete weiter als 'cuda', daher bleibt RISK_LAYER_EMB_DEVICE=cuda korrekt."
      ;;
    3|intel|vulkan)
      BACKEND="vulkan"; emb="cpu"; label="Intel/Vulkan"
      info "Intel/andere iGPU: llama.cpp laeuft ueber Vulkan. Embeddings bleiben standardmaessig auf CPU, weil torch-Intel-GPU je Host stark variiert."
      ;;
    4|cpu)
      BACKEND="cpu"; emb="cpu"; label="CPU"
      info "CPU: robust und ohne GPU-Treiber. Erwartbar langsamer, aber alle deterministischen Layer und /v1 bleiben nutzbar."
      ;;
    *)
      BACKEND="auto"; label="Auto"; emb="cuda"
      info "Auto: Vendor-Erkennung entscheidet beim Engine-Download. Bei unbekannter GPU faellt llama.cpp auf CPU zurueck."
      ;;
  esac
  if [[ -z "${RISK_LAYER_EMB_DEVICE+x}" ]]; then
    export RISK_LAYER_EMB_DEVICE="$emb"
  else
    info "RISK_LAYER_EMB_DEVICE ist bereits gesetzt: $RISK_LAYER_EMB_DEVICE"
  fi
  ok "Implementierung: $label (Engine-Backend: $BACKEND, Embedding-Geraet: ${RISK_LAYER_EMB_DEVICE:-$emb})"
}

choose_implementation() {
  local erkannt default choice
  erkannt="$(detected_impl)"
  case "$erkannt" in
    nvidia) default=1 ;;
    amd) default=2 ;;
    *) default=4 ;;
  esac

  schritt "Implementierung waehlen"
  if [[ $BACKEND_EXPLICIT -eq 1 ]]; then
    case "$BACKEND" in
      cuda) apply_implementation_choice 1 ;;
      rocm|hip) apply_implementation_choice 2 ;;
      vulkan) apply_implementation_choice 3 ;;
      cpu) apply_implementation_choice 4 ;;
      auto) apply_implementation_choice auto ;;
      *) info "Unbekanntes Backend '$BACKEND' - Auto-Erkennung wird genutzt"; apply_implementation_choice auto ;;
    esac
    return
  fi

  if can_prompt; then
    info "Erkannt: $erkannt. Diese Wahl betrifft llama.cpp-Engine und ML-Geraet."
    printf "  1) NVIDIA/CUDA  - CUDA fuer Embeddings, llama.cpp ggf. Vulkan-Fallback unter Linux\n"
    printf "  2) AMD/ROCm     - ROCm/HIP fuer llama.cpp, Embeddings ueber torch-ROCm\n"
    printf "  3) Intel/Vulkan - manuell waehlen, wenn wirklich Intel-GPU/Vulkan nutzbar ist\n"
    printf "  4) CPU          - kein GPU-Treiber noetig, langsamer\n"
    printf "Auswahl [%s]: " "$default"
    IFS= read -r choice || choice="$default"
    choice="${choice:-$default}"
    apply_implementation_choice "$choice"
  else
    info "Nicht interaktiv: Implementierung wird automatisch erkannt (erkannt: $erkannt)."
    case "$erkannt" in
      nvidia) apply_implementation_choice 1 ;;
      amd) apply_implementation_choice 2 ;;
      *) apply_implementation_choice 4 ;;
    esac
  fi
}

choose_quanten_extras() {
  [[ $SKIP_DEPS -eq 1 || $QUANTEN_EXPLICIT -eq 1 ]] && return 0
  if ! can_prompt; then
    info "Nicht interaktiv: Quantenlos-Extras werden nicht installiert (bei Bedarf --with-quanten)."
    return 0
  fi

  schritt "Quantenlos"
  info "/v1/los/* braucht fuer qpu/simulator die optionalen qiskit-Extras."
  info "Ohne Extras liefert die Engine eine 503-Betriebsvoraussetzung; csprng bleibt lokal moeglich."
  if ask_yes_no "Quantenlos-Extras jetzt installieren?" "j"; then
    WITH_QUANTEN=1
  else
    WITH_QUANTEN=0
  fi
}

client_base_url() {
  local host="$BIND_HOST"
  case "$host" in
    ""|"0.0.0.0"|"::") host="127.0.0.1" ;;
  esac
  if [[ "$host" == *:* && "$host" != \[*\] ]]; then host="[$host]"; fi
  printf "http://%s:%s\n" "$host" "$PORT"
}

docker_bridge_ip() {
  local ip_addr=""
  if command -v ip >/dev/null 2>&1; then
    ip_addr="$(ip -4 addr show docker0 2>/dev/null | awk '/inet / {split($2,a,"/"); print a[1]; exit}')"
  fi
  if [[ -z "$ip_addr" ]] && command -v docker >/dev/null 2>&1; then
    ip_addr="$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || true)"
  fi
  printf '%s\n' "$ip_addr"
}

docker_container_gateway_ip() {
  local container="$1"
  [[ -n "$container" ]] || return 0
  docker inspect "$container" \
    --format '{{range $name, $net := .NetworkSettings.Networks}}{{if $net.Gateway}}{{println $net.Gateway}}{{end}}{{end}}' \
    2>/dev/null | awk 'NF {print; exit}'
}

docker_bind_suggestion() {
  DOCKER_BIND_SUGGESTION=""
  if ! command -v docker >/dev/null 2>&1; then
    DOCKER_BIND_SUGGESTION="$(docker_bridge_ip)"
    return
  fi

  local default_container container gateway
  default_container="$(docker ps --format '{{.Names}}' 2>/dev/null | awk '$0=="taxtronik-app"{print; found=1; exit} END{if(!found) exit 0}')"
  if [[ -z "$default_container" ]]; then
    default_container="$(docker ps --format '{{.Names}}' 2>/dev/null | awk 'NF {print; exit}')"
  fi

  if [[ -n "$default_container" && -t 0 && -t 1 ]]; then
    printf "Container fuer IP-Vorschlag [%s]: " "$default_container"
    IFS= read -r container || container="$default_container"
    container="${container:-$default_container}"
    gateway="$(docker_container_gateway_ip "$container")"
    if [[ -n "$gateway" ]]; then
      info "Vorschlag aus Container '$container': Docker-Gateway $gateway"
      DOCKER_BIND_SUGGESTION="$gateway"
      return
    fi
    info "Container '$container' liefert kein Gateway - fallback auf Docker-Bridge."
  fi

  DOCKER_BIND_SUGGESTION="$(docker_bridge_ip)"
}

maybe_prompt_docker_native_bind() {
  [[ $HOST_EXPLICIT -eq 1 ]] && return 0
  if ! can_prompt; then return 0; fi

  schritt "Docker-Zugriff"
  info "127.0.0.1 ist nur vom Host erreichbar; im Container zeigt es auf den Container selbst."
  info "Am saubersten: Risk-Layer als Compose-Service starten und RISK_LAYER_URL=http://risk-layer:8000 setzen."
  if ! ask_yes_no "Soll eine Docker-App diese native Host-Engine erreichen koennen?" "n"; then
    return 0
  fi

  if [[ "$(uname -s 2>/dev/null || true)" != Linux* ]]; then
    DOCKER_NATIVE_URL="http://host.docker.internal:$PORT"
    info "Docker Desktop erreicht Host-Dienste ueber host.docker.internal; Bind bleibt $BIND_HOST."
    return 0
  fi

  local vorgeschlagen bind
  DOCKER_BIND_SUGGESTION=""
  docker_bind_suggestion
  vorgeschlagen="$DOCKER_BIND_SUGGESTION"
  if [[ -z "$vorgeschlagen" ]]; then
    info "Keine Docker-Bridge-IP gefunden; Bind bleibt $BIND_HOST."
    info "Starte alternativ Risk-Layer im selben Compose-Netz oder setze --host auf eine konkrete Docker-Bridge-IP."
    return 0
  fi

  while true; do
    printf "Bind-IP fuer Docker-Zugriff [%s]: " "$vorgeschlagen"
    IFS= read -r bind || bind="$vorgeschlagen"
    bind="${bind:-$vorgeschlagen}"
    case "$bind" in
      "0.0.0.0"|"::")
        printf "Bitte keine Wildcard-Adresse nutzen; eine konkrete Docker-Bridge-IP ist sicherer.\n"
        ;;
      *)
        BIND_HOST="$bind"
        DOCKER_NATIVE_URL="http://$BIND_HOST:$PORT"
        ok "Docker-Zugriff aktiviert: Risk-Layer bindet an $BIND_HOST, Container nutzen $DOCKER_NATIVE_URL"
        return 0
        ;;
    esac
  done
}

maybe_prompt_systemd() {
  [[ $SERVICE -eq 1 || $START -eq 1 ]] && return 0
  if ! command -v systemctl >/dev/null 2>&1; then
    info "systemd nicht erkannt - kein Autostart-Prompt."
    return 0
  fi
  if ! can_prompt; then
    info "systemd erkannt, aber kein interaktives Terminal - Autostart wird nicht eingerichtet."
    return 0
  fi
  schritt "Autostart"
  info "systemd startet Risk-Layer mit --llm-autostart und liest den Token aus einer chmod-600 EnvironmentFile."
  info "Mit root wird ein Systemdienst eingerichtet; ohne root ein User-Dienst."
  if ask_yes_no "Risk-Layer jetzt als systemd-Dienst einrichten?" "n"; then
    SERVICE=1
  else
    info "kein systemd-Dienst eingerichtet"
  fi
}

run_systemctl() {
  local beschreibung="$1"
  local rc=0
  shift
  info "$beschreibung"
  if command -v timeout >/dev/null 2>&1; then
    timeout 45s systemctl "$@" || rc=$?
  else
    systemctl "$@" || rc=$?
  fi
  if [[ $rc -eq 124 ]]; then
    echo "systemctl hat nach 45s nicht geantwortet: systemctl $*" >&2
  fi
  return "$rc"
}

PY=python3; command -v "$PY" >/dev/null 2>&1 || PY=python
cli()   { printf "${D}  > %s -m risk_layer.cli %s${N}\n" "$PY" "$*"; "$PY" -m risk_layer.cli "$@"; }
py_ok() { "$PY" -c "$1" >/dev/null 2>&1; }
spacy_ok() { "$PY" -c "import spacy; print(spacy.__version__)" >/dev/null; }
pip_ok() { "$PY" -m pip --version >/dev/null 2>&1; }
python_abs() {
  case "$PY" in
    /*) printf '%s\n' "$PY" ;;
    */*) printf '%s/%s\n' "$(pwd)" "$PY" ;;
    *) command -v "$PY" ;;
  esac
}
use_project_venv() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    info "aktive virtuelle Umgebung: $VIRTUAL_ENV"
    return 0
  fi
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    PY="$VENV_DIR/bin/python"
    ok "virtuelle Umgebung gefunden ($VENV_DIR)"
    return 0
  fi
  if [[ $SKIP_DEPS -eq 1 ]]; then
    info "keine virtuelle Umgebung gefunden; nutze System-Python (--skip-deps)"
    return 0
  fi
  info "erstelle virtuelle Umgebung ($VENV_DIR)"
  if "$PY" -m venv "$VENV_DIR"; then
    PY="$VENV_DIR/bin/python"
    ok "virtuelle Umgebung erstellt"
    return 0
  fi
  cat >&2 <<EOF
Virtuelle Umgebung konnte nicht erstellt werden.
Bitte installiere venv fuer diesen Python-Interpreter und starte deploy.sh erneut.

Debian/Ubuntu/WSL:
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-full
EOF
  exit 1
}
ensure_pip() {
  if pip_ok; then return 0; fi
  info "pip fehlt -> versuche Bootstrap via ensurepip"
  if "$PY" -m ensurepip --upgrade >/dev/null 2>&1 && pip_ok; then
    ok "pip bereitgestellt"
    return 0
  fi
  cat >&2 <<EOF
pip ist fuer $PY nicht installiert und konnte nicht via ensurepip bereitgestellt werden.
Bitte installiere pip fuer diesen Python-Interpreter und starte deploy.sh erneut.

Debian/Ubuntu/WSL:
  sudo apt-get update
  sudo apt-get install -y python3-pip python3-venv

Andere Distributionen: das Paket heisst meist python3-pip.
EOF
  exit 1
}

# --- Python ----------------------------------------------------------------
schritt "Python pruefen"
command -v "$PY" >/dev/null 2>&1 || { echo "python/python3 nicht gefunden – Python >= 3.11 installieren." >&2; exit 1; }
PYV="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)' \
  || { echo "Python >= 3.11 noetig (gefunden $PYV)." >&2; exit 1; }
ok "Python $PYV"
use_project_venv
choose_implementation
choose_quanten_extras

# --- Abhaengigkeiten -------------------------------------------------------
if [[ $SKIP_DEPS -eq 0 ]]; then
  schritt "Abhaengigkeiten installieren"
  ensure_pip
  # Kern (PyYAML + numpy), dev und Build-Backend exakt aus dem Hash-Lock
  # (deterministisch ueber die Flotte, identisch zu CI/Dockerfile).
  "$PY" -m pip install --require-hashes --only-binary=:all: -r requirements-lock.txt
  "$PY" -m pip install --no-build-isolation --no-deps -e .
  if [[ $WITH_EMBEDDING -eq 1 ]]; then
    # NVIDIA: Blackwell-faehiges torch ZUERST aus dem CUDA-Wheel (sonst zieht
    # sentence-transformers ggf. ein zu altes/CPU-torch -> GPU ungenutzt). Default
    # cu128 deckt Blackwell (sm_100/sm_120); per --cuda-wheel anpassbar.
    if command -v nvidia-smi >/dev/null 2>&1; then
      info "NVIDIA erkannt -> Blackwell-faehiges torch ($CUDA_WHEEL) vorab installieren"
      "$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_WHEEL" \
        || info "torch-CUDA-Install fehlgeschlagen -> sentence-transformers waehlt torch selbst"
    fi
    info "Embedding+Reranker gewuenscht -> sentence-transformers (+ torch, gross) + hf_transfer"
    # bewusst ungepinnt (GPU-/Plattform-Matrix) — ML-Extras sind nicht im Lock.
    "$PY" -m pip install "sentence-transformers>=2.7" hf_transfer
    ok "Abhaengigkeiten installiert (PyYAML + numpy + sentence-transformers + hf_transfer)"
  else
    ok "Abhaengigkeiten installiert (PyYAML + numpy)"
  fi
  if [[ $WITH_QUANTEN -eq 1 ]]; then
    info "Quantenlos-Extras gewuenscht -> qiskit + qiskit-aer + qiskit-ibm-runtime"
    "$PY" -m pip install --only-binary=:all: "qiskit>=1.0" "qiskit-ibm-runtime>=0.30" "qiskit-aer>=0.15"
    ok "Quantenlos-Extras installiert"
  else
    info "Quantenlos-Extras uebersprungen - /v1/los qpu/simulator meldet sonst fehlende Betriebsvoraussetzung"
  fi
else info "Deps uebersprungen (--skip-deps)"; fi

# --- NER-Schicht (Anonymisierung, §203-Gate) --------------------------------
if [[ $OHNE_NER -eq 0 ]]; then
  schritt "NER-Schicht der Anonymisierung (spaCy + de_core_news_lg)"
  # Idempotent: Modell-Paket schon importierbar (lg, Fallback sm) -> ueberspringen.
  # find_spec statt Modell-Laden: reiner Metadaten-Lookup (laedt die ~0,5 GB NICHT).
  if py_ok "import importlib.util,sys; sys.exit(0 if any(importlib.util.find_spec(m) for m in ('de_core_news_lg','de_core_news_sm')) else 1)" && [[ $FORCE -eq 0 ]]; then
    ok "spaCy-Modell bereits installiert - uebersprungen"
  else
    # spaCy + Modell kommen aus dem separaten Hash-Lock. Fehler hier sind NICHT
    # fatal (wie der Engine-Download):
    # die Anonymisierung degradiert auf den deterministischen Regex-Pfad.
    info "installiert spaCy (Extra nlp) + Modell de_core_news_lg ..."
    if "$PY" -m pip install --require-hashes --only-binary=:all: -r requirements-ner-lock.txt \
       && spacy_ok \
       && py_ok "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('de_core_news_lg') else 1)"; then
      ok "NER-Schicht installiert (de_core_news_lg)"
    else
      printf "${Y}  [warnung] NER-Provisionierung fehlgeschlagen - NER-Schicht inaktiv,${N}\n"
      printf "${Y}  Engine laeuft regex-only weiter. Nachholen mit:${N}\n"
      printf "${Y}    pip install --require-hashes --only-binary=:all: -r requirements-ner-lock.txt${N}\n"
    fi
  fi
else info "NER uebersprungen (--ohne-ner) - Anonymisierung laeuft regex-only"; fi

# --- Auth-Token ------------------------------------------------------------
schritt "Auth-Token (.engine_token)"
if [[ ! -f .engine_token || $FORCE -eq 1 ]]; then
  "$PY" -c "import secrets; open('.engine_token','w').write(secrets.token_hex(32))"
  ok "neues Token erzeugt (.engine_token, gitignored)"
else ok ".engine_token vorhanden (wiederverwendet)"; fi
TOKEN="$(tr -d '[:space:]' < .engine_token)"
export RISK_LAYER_TOKEN="$TOKEN"
ok "RISK_LAYER_TOKEN fuer diesen Deploy-Lauf gesetzt"

# --- LLM-Modell (GGUF) -----------------------------------------------------
if [[ $SKIP_MODEL -eq 0 ]]; then
  schritt "LLM-Modell (GGUF) herunterladen"
  if [[ -n "$MODELL" ]]; then MREF="M.MODELLE['$MODELL']"; else MREF="M.MODELLE[M.DEFAULT_MODELL]"; fi
  if py_ok "from risk_layer import modell as M; import sys; sys.exit(0 if M.modell_pfad($MREF) else 1)" && [[ $FORCE -eq 0 ]]; then
    ok "Modell bereits vorhanden (./models) - uebersprungen"
  else
    info "laedt das Default-GGUF (resumebar) nach ./models ..."
    if [[ -n "$MODELL" ]]; then cli modell download --modell "$MODELL"; else cli modell download; fi
    ok "Modell geladen"
  fi
else info "Modell uebersprungen (--skip-model)"; fi

# --- llama.cpp-Engine (llama-server, GPU-Backend) --------------------------
if [[ $SKIP_ENGINE -eq 0 ]]; then
  schritt "llama.cpp-Engine bereitstellen (Backend: $BACKEND)"
  if py_ok "from risk_layer import modell as M; import sys; sys.exit(0 if M.finde_llama_server() else 1)" && [[ $FORCE -eq 0 ]]; then
    ok "llama-server gefunden (./engine oder PATH) - uebersprungen"
  else
    info "laedt die passende llama.cpp-Linux-Release (AMD->rocm, sonst Vulkan/CPU) nach ./engine ..."
    info "Hinweis: fuer NVIDIA-CUDA gibt es KEIN Linux-Prebuilt (Auto faellt auf Vulkan) -"
    info "  fuer volle CUDA-Speed llama.cpp aus Source/Docker (server-cuda) bauen."
    if cli modell engine --backend "$BACKEND"; then
      ok "Engine bereitgestellt (./engine)"
    else
      info "Engine-Download fehlgeschlagen. llama-server separat auf den PATH legen."
      info "  Ohne ihn laufen alle deterministischen Layer + /v1; nur mitLLM:true ist inaktiv."
    fi
  fi
else info "Engine uebersprungen (--skip-engine)"; fi

# --- Normkorpus + Graph + Indizes ------------------------------------------
if [[ $SKIP_CORPUS -eq 0 ]]; then
  schritt "Normkorpus + Graph + Trigger-Index bauen"

  if [[ ! -f corpus/gesetze.yaml || $VOLL -eq 1 || $FORCE -eq 1 ]]; then
    if [[ $VOLL -eq 1 ]]; then cli korpus init --voll; else cli korpus init; fi
  else ok "corpus/gesetze.yaml vorhanden"; fi

  if compgen -G "corpus/gii/*.xml" >/dev/null && [[ $FORCE -eq 0 ]]; then
    ok "Gesetzes-XML vorhanden (corpus/gii/)"
  else
    info "laedt Gesetzes-XML von gesetze-im-internet.de ..."
    cli korpus fetch
  fi

  if [[ ! -f corpus/graph.sqlite || $FORCE -eq 1 ]]; then cli graph build; else ok "Normgraph vorhanden (corpus/graph.sqlite)"; fi
  # Trigger-Index neu bauen, wenn er fehlt, --force, ODER der Graph neuer ist
  # (Korpus-Aenderung, z. B. neue Gesetze/DVs -> der Index muss sie aufnehmen).
  if [[ ! -f corpus/trigger.sqlite || $FORCE -eq 1 ]] || [[ corpus/graph.sqlite -nt corpus/trigger.sqlite ]]; then cli trigger build; else ok "Trigger-Index aktuell (corpus/trigger.sqlite)"; fi

  if [[ $WITH_EMBEDDING -eq 1 ]]; then
    schritt "Embedding-Index (BGE-M3) + Reranker (Schicht 1.7 / 1.7b)"
    if "$PY" -c "import hf_transfer" 2>/dev/null; then export HF_HUB_ENABLE_HF_TRANSFER=1; fi
    # Bauen, wenn Index fehlt, --force, der Katalog NEUER ist (neue Paraphrasen-
    # Anker in catalog/begriffe.yaml) ODER der Graph neuer ist (Korpus-Aenderung,
    # z. B. neue Gesetze/DVs) — beides MUSS mit-eingebettet werden.
    if [[ ! -f corpus/embedding/vectors.npy || $FORCE -eq 1 ]] \
       || [[ catalog/begriffe.yaml -nt corpus/embedding/vectors.npy ]] \
       || [[ corpus/graph.sqlite -nt corpus/embedding/vectors.npy ]]; then
      info "bettet den Korpus ein (laedt BGE-M3 ~2,3 GB beim ersten Mal) ..."
      cli embedding build
    else ok "Embedding-Index aktuell (corpus/embedding/)"; fi
    # Reranker (Schicht 1.7b) ist in der Engine fest aktiv (GPU). Modell vorab
    # cachen, sonst laedt der ERSTE Request ~600 MB nach. Nur Download, kein GPU.
    info "cacht den Reranker (bge-reranker-v2-m3, ~600 MB) ..."
    "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3')" \
      && ok "Reranker gecacht" || info "Reranker-Cache fehlgeschlagen (laedt sonst beim 1. Request)"
  fi
else info "Korpus uebersprungen (--skip-corpus)"; fi

# --- Status + Abschluss ----------------------------------------------------
schritt "Status"
cli modell status
# NER-Status nur per find_spec (Metadaten-Lookup) — das Modell wird NICHT geladen
# (Laden waere langsam und unnoetig fuer eine reine Installations-Auskunft).
NER_MODELL="$("$PY" -c "import importlib.util; print(next((m for m in ('de_core_news_lg','de_core_news_sm') if importlib.util.find_spec(m)), ''))" 2>/dev/null || true)"
if [[ -n "$NER_MODELL" ]]; then
  ok "NER aktiv (Modell $NER_MODELL)"
  case "$(printf '%s' "${RISK_LAYER_NER:-}" | tr '[:upper:]' '[:lower:]')" in
    aus|off|none|0) info "Hinweis: RISK_LAYER_NER=aus ist gesetzt - Laufzeit nutzt den Regex-Pfad" ;;
  esac
elif [[ $OHNE_NER -eq 1 ]]; then
  info "NER inaktiv (--ohne-ner) - Anonymisierung regex-only; nachholen: pip install --require-hashes --only-binary=:all: -r requirements-ner-lock.txt"
else
  info "NER inaktiv (Modell fehlt) - Anonymisierung regex-only; nachholen: pip install --require-hashes --only-binary=:all: -r requirements-ner-lock.txt"
fi

maybe_prompt_docker_native_bind
BASE_URL="$(client_base_url)"
maybe_prompt_systemd

printf "\n${G}Deploy abgeschlossen.${N}\n"
printf "${Y}Kanzleikonsole .env (Host-Prozess, nicht Docker):${N}\n"
printf "RISK_LAYER_URL=%s\n" "$BASE_URL"
printf "RISK_LAYER_TOKEN=%s\n" "$TOKEN"
printf "RISK_LAYER_HEALTH_PATH=/v1/health\n"
printf "RISK_LAYER_ANALYSE_PATH=/v1/analyse\n"
printf "${Y}Kanzleikonsole .env (Docker, Risk-Layer im selben Compose-Netz):${N}\n"
printf "RISK_LAYER_URL=http://risk-layer:8000\n"
printf "RISK_LAYER_TOKEN=%s\n" "$TOKEN"
printf "RISK_LAYER_HEALTH_PATH=/v1/health\n"
printf "RISK_LAYER_ANALYSE_PATH=/v1/analyse\n"
if [[ -n "$DOCKER_NATIVE_URL" ]]; then
  printf "${Y}Kanzleikonsole .env (Docker-App -> nativer Host-Risk-Layer):${N}\n"
  printf "RISK_LAYER_URL=%s\n" "$DOCKER_NATIVE_URL"
  printf "RISK_LAYER_TOKEN=%s\n" "$TOKEN"
  printf "RISK_LAYER_HEALTH_PATH=/v1/health\n"
  printf "RISK_LAYER_ANALYSE_PATH=/v1/analyse\n"
fi
printf "${Y}Ein-Befehl-Start (Engine + LLM):${N}\n"
printf "  export RISK_LAYER_TOKEN=\"%s\"\n" "$TOKEN"
printf "  %s -m risk_layer.web --host %s --port %s --llm-autostart\n" "$PY" "$BIND_HOST" "$PORT"

# --- systemd-Dienst direkt einrichten (--service) --------------------------
if [[ $SERVICE -eq 1 ]]; then
  schritt "systemd-Dienst einrichten (--llm-autostart, Autostart bei Boot)"
  command -v systemctl >/dev/null 2>&1 || { echo "systemctl nicht gefunden – kein systemd-System." >&2; exit 1; }
  PYBIN="$(python_abs)"
  UNIT_SRC="deploy/risk-layer.service.in"
  [[ -f "$UNIT_SRC" ]] || { echo "Vorlage fehlt: $UNIT_SRC" >&2; exit 1; }
  # Token NICHT in die Unit rendern (/etc/systemd/system/* ist welt-lesbar, 644) —
  # stattdessen in eine separate EnvironmentFile mit chmod 600 (umask 077 deckt
  # auch die Neuanlage ab); die Unit referenziert sie nur per Pfad.
  if [[ $EUID -eq 0 ]]; then ENV_DIR="/etc/risk-layer"; else ENV_DIR="$HOME/.config/risk-layer"; fi
  ENV_FILE="$ENV_DIR/engine.env"
  mkdir -p "$ENV_DIR"
  (umask 077; printf 'RISK_LAYER_TOKEN=%s\n' "$TOKEN" > "$ENV_FILE")
  chmod 600 "$ENV_FILE"
  [[ $EUID -eq 0 ]] && chown root:root "$ENV_FILE"
  ok "Token-EnvironmentFile geschrieben: $ENV_FILE (chmod 600)"
  RENDER="$(sed -e "s|@USER@|$(id -un)|g" -e "s|@WORKDIR@|$(pwd)|g" \
               -e "s|@ENVFILE@|$ENV_FILE|g" -e "s|@EMB_DEVICE@|${RISK_LAYER_EMB_DEVICE:-cuda}|g" \
               -e "s|@PYTHON@|$PYBIN|g" -e "s|@HOST@|$BIND_HOST|g" -e "s|@PORT@|$PORT|g" "$UNIT_SRC")"
  if [[ $EUID -eq 0 ]]; then
    printf '%s\n' "$RENDER" > /etc/systemd/system/risk-layer.service
    run_systemctl "systemd Unit neu laden" daemon-reload
    run_systemctl "systemd Autostart aktivieren" enable risk-layer.service
    run_systemctl "systemd Dienst neu starten" restart risk-layer.service
    systemctl is-active --quiet risk-layer.service || { systemctl --no-pager status risk-layer.service; exit 1; }
    ok "System-Dienst aktiv: systemctl status risk-layer"
  else
    # Ohne root: User-Unit (läuft bei Login; 'loginctl enable-linger' für Boot ohne Login)
    mkdir -p "$HOME/.config/systemd/user"
    printf '%s\n' "$RENDER" | sed '/^User=/d' > "$HOME/.config/systemd/user/risk-layer.service"
    run_systemctl "systemd User-Unit neu laden" --user daemon-reload
    run_systemctl "systemd User-Autostart aktivieren" --user enable risk-layer.service
    run_systemctl "systemd User-Dienst neu starten" --user restart risk-layer.service
    systemctl --user is-active --quiet risk-layer.service || { systemctl --user --no-pager status risk-layer.service; exit 1; }
    ok "User-Dienst aktiv: systemctl --user status risk-layer"
    info "Boot ohne Login: sudo loginctl enable-linger $(id -un)"
  fi
elif [[ $START -eq 1 ]]; then
  schritt "Engine starten (Engine + LLM, Vordergrund)"
  export RISK_LAYER_TOKEN="$TOKEN"
  exec "$PY" -m risk_layer.web --host "$BIND_HOST" --port "$PORT" --llm-autostart
fi
