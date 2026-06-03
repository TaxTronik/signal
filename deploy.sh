#!/usr/bin/env bash
#
# Setup/Deploy des Risk-Layer (Linux) von "frischem Clone" zu lauffaehiger
# /v1-Engine: Abhaengigkeiten, LLM-Modell (GGUF) + passende llama.cpp-Engine
# herunterladen, Normkorpus/Graph/Trigger(/Embedding) bauen, Auth-Token. Idempotent.
#
# Engine-Backend (--backend, Default auto): AMD->rocm, sonst Vulkan/CPU. Fuer
# NVIDIA gibt es KEIN Linux-Prebuilt -> Auto faellt auf Vulkan; volle CUDA-Speed
# nur via llama.cpp aus Source/Docker (server-cuda). Ohne llama-server laufen alle
# deterministischen Layer + /v1; nur 'mitLLM:true' ist inaktiv.
#
# Beispiele:
#   ./deploy.sh                         # Deps + Modell + Engine + Korpus/Graph/Trigger
#   ./deploy.sh --backend rocm          # AMD/ROCm-Engine erzwingen
#   ./deploy.sh --with-embedding --voll # zusaetzlich Embedding-Index + breiter Korpus
#   ./deploy.sh --skip-engine           # Engine separat bereitstellen
#   ./deploy.sh --start                 # Setup, dann Engine starten
#
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_DEPS=0; SKIP_MODEL=0; SKIP_ENGINE=0; SKIP_CORPUS=0; WITH_EMBEDDING=0; VOLL=0; FORCE=0; START=0
MODELL=""; BACKEND="auto"; BIND_HOST="127.0.0.1"; PORT=8000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-deps)      SKIP_DEPS=1 ;;
    --skip-model)     SKIP_MODEL=1 ;;
    --skip-engine)    SKIP_ENGINE=1 ;;
    --skip-corpus)    SKIP_CORPUS=1 ;;
    --with-embedding) WITH_EMBEDDING=1 ;;
    --voll)           VOLL=1 ;;
    --force)          FORCE=1 ;;
    --start)          START=1 ;;
    --modell)         MODELL="${2:?--modell braucht einen Wert}"; shift ;;
    --backend)        BACKEND="${2:?--backend braucht einen Wert}"; shift ;;
    --host)           BIND_HOST="${2:?--host braucht einen Wert}"; shift ;;
    --port)           PORT="${2:?--port braucht einen Wert}"; shift ;;
    -h|--help)        sed -n '3,20p' "$0"; exit 0 ;;
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

PY=python3; command -v "$PY" >/dev/null 2>&1 || PY=python
cli()   { printf "${D}  > %s -m risk_layer.cli %s${N}\n" "$PY" "$*"; "$PY" -m risk_layer.cli "$@"; }
py_ok() { "$PY" -c "$1" >/dev/null 2>&1; }

# --- Python ----------------------------------------------------------------
schritt "Python pruefen"
command -v "$PY" >/dev/null 2>&1 || { echo "python/python3 nicht gefunden – Python >= 3.11 installieren." >&2; exit 1; }
PYV="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)' \
  || { echo "Python >= 3.11 noetig (gefunden $PYV)." >&2; exit 1; }
ok "Python $PYV"

# --- Abhaengigkeiten -------------------------------------------------------
if [[ $SKIP_DEPS -eq 0 ]]; then
  schritt "Abhaengigkeiten installieren"
  "$PY" -m pip install --upgrade pip >/dev/null
  "$PY" -m pip install -e . "numpy>=1.26"
  if [[ $WITH_EMBEDDING -eq 1 ]]; then
    info "Embedding gewuenscht -> sentence-transformers (+ torch, gross) installieren"
    "$PY" -m pip install "sentence-transformers>=2.7"
    ok "Abhaengigkeiten installiert (PyYAML + numpy + sentence-transformers)"
  else
    ok "Abhaengigkeiten installiert (PyYAML + numpy)"
  fi
else info "Deps uebersprungen (--skip-deps)"; fi

# --- Auth-Token ------------------------------------------------------------
schritt "Auth-Token (.engine_token)"
if [[ ! -f .engine_token || $FORCE -eq 1 ]]; then
  "$PY" -c "import secrets; open('.engine_token','w').write(secrets.token_hex(32))"
  ok "neues Token erzeugt (.engine_token, gitignored)"
else ok ".engine_token vorhanden (wiederverwendet)"; fi
TOKEN="$(tr -d '[:space:]' < .engine_token)"
info "RISK_LAYER_TOKEN = $TOKEN"

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
  if [[ ! -f corpus/trigger.sqlite || $FORCE -eq 1 ]]; then cli trigger build; else ok "Trigger-Index vorhanden (corpus/trigger.sqlite)"; fi

  if [[ $WITH_EMBEDDING -eq 1 ]]; then
    schritt "Semantischer Embedding-Index (BGE-M3)"
    if [[ ! -f corpus/embedding/vectors.npy || $FORCE -eq 1 ]]; then
      info "bettet den Korpus ein (laedt BGE-M3 ~2,3 GB beim ersten Mal) ..."
      cli embedding build
    else ok "Embedding-Index vorhanden (corpus/embedding/)"; fi
  fi
else info "Korpus uebersprungen (--skip-corpus)"; fi

# --- Status + Abschluss ----------------------------------------------------
schritt "Status"
cli modell status

printf "\n${G}Deploy abgeschlossen.${N}\n"
printf "${Y}Engine starten (headless /v1):${N}\n"
printf "  export RISK_LAYER_TOKEN=\"\$(cat .engine_token)\"\n"
printf "  %s -m risk_layer.web --host %s --port %s\n" "$PY" "$BIND_HOST" "$PORT"
printf "${Y}LLM ueber die API: POST /v1/llm/start  -  GET /v1/llm/status${N}\n"

if [[ $START -eq 1 ]]; then
  schritt "Engine starten"
  export RISK_LAYER_TOKEN="$TOKEN"
  exec "$PY" -m risk_layer.web --host "$BIND_HOST" --port "$PORT"
fi
