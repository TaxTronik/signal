#!/usr/bin/env bash
#
# Setup/Deploy des Risk-Layer (Linux) von "frischem Clone" zu lauffaehiger
# /v1-Engine: Abhaengigkeiten, LLM-Modell (GGUF) + passende llama.cpp-Engine
# herunterladen, Normkorpus/Graph/Trigger(/Embedding) bauen, Auth-Token. Idempotent.
#
# Engine-Backend (--backend, Default auto): NVIDIA->cuda, AMD->rocm, sonst Vulkan/CPU.
# Auf NVIDIA-Boxen (z. B. B300/Blackwell) wird zudem ein Blackwell-faehiges torch
# (--cuda-wheel, Default cu128) installiert, sonst nutzt sentence-transformers die GPU
# nicht. Falls kein CUDA-llama-server-Prebuilt: Discovery/LLM spricht jeden OpenAI-
# kompatiblen /v1-Endpoint an (z. B. vLLM) via --llm-url. Ohne llama-server laufen alle
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

SKIP_DEPS=0; SKIP_MODEL=0; SKIP_ENGINE=0; SKIP_CORPUS=0; WITH_EMBEDDING=0; VOLL=0; FORCE=0; START=0; SERVICE=0
MODELL=""; BACKEND="auto"; BIND_HOST="127.0.0.1"; PORT=8000; CUDA_WHEEL="cu128"

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
    --service)        SERVICE=1 ;;
    --modell)         MODELL="${2:?--modell braucht einen Wert}"; shift ;;
    --backend)        BACKEND="${2:?--backend braucht einen Wert}"; shift ;;
    --host)           BIND_HOST="${2:?--host braucht einen Wert}"; shift ;;
    --port)           PORT="${2:?--port braucht einen Wert}"; shift ;;
    --cuda-wheel)     CUDA_WHEEL="${2:?--cuda-wheel braucht einen Wert (z. B. cu128, cu124)}"; shift ;;
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
    # NVIDIA: Blackwell-faehiges torch ZUERST aus dem CUDA-Wheel (sonst zieht
    # sentence-transformers ggf. ein zu altes/CPU-torch -> GPU ungenutzt). Default
    # cu128 deckt Blackwell (sm_100/sm_120); per --cuda-wheel anpassbar.
    if command -v nvidia-smi >/dev/null 2>&1; then
      info "NVIDIA erkannt -> Blackwell-faehiges torch ($CUDA_WHEEL) vorab installieren"
      "$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_WHEEL" \
        || info "torch-CUDA-Install fehlgeschlagen -> sentence-transformers waehlt torch selbst"
    fi
    info "Embedding+Reranker gewuenscht -> sentence-transformers (+ torch, gross) + hf_transfer"
    "$PY" -m pip install "sentence-transformers>=2.7" hf_transfer
    ok "Abhaengigkeiten installiert (PyYAML + numpy + sentence-transformers + hf_transfer)"
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

printf "\n${G}Deploy abgeschlossen.${N}\n"
printf "${Y}Ein-Befehl-Start (Engine + LLM):${N}\n"
printf "  export RISK_LAYER_TOKEN=\"\$(cat .engine_token)\"\n"
printf "  %s -m risk_layer.web --host %s --port %s --llm-autostart\n" "$PY" "$BIND_HOST" "$PORT"
printf "${Y}Als systemd-Dienst (Autostart bei Boot): ./deploy.sh --service${N}\n"

# --- systemd-Dienst direkt einrichten (--service) --------------------------
if [[ $SERVICE -eq 1 ]]; then
  schritt "systemd-Dienst einrichten (--llm-autostart, Autostart bei Boot)"
  command -v systemctl >/dev/null 2>&1 || { echo "systemctl nicht gefunden – kein systemd-System." >&2; exit 1; }
  PYBIN="$(command -v "$PY")"
  UNIT_SRC="deploy/risk-layer.service.in"
  [[ -f "$UNIT_SRC" ]] || { echo "Vorlage fehlt: $UNIT_SRC" >&2; exit 1; }
  RENDER="$(sed -e "s|@USER@|$(id -un)|g" -e "s|@WORKDIR@|$(pwd)|g" \
               -e "s|@TOKEN@|$TOKEN|g" -e "s|@EMB_DEVICE@|${RISK_LAYER_EMB_DEVICE:-cuda}|g" \
               -e "s|@PYTHON@|$PYBIN|g" -e "s|@HOST@|$BIND_HOST|g" -e "s|@PORT@|$PORT|g" "$UNIT_SRC")"
  if [[ $EUID -eq 0 ]]; then
    printf '%s\n' "$RENDER" > /etc/systemd/system/risk-layer.service
    systemctl daemon-reload && systemctl enable --now risk-layer.service
    ok "System-Dienst aktiv: systemctl status risk-layer"
  else
    # Ohne root: User-Unit (läuft bei Login; 'loginctl enable-linger' für Boot ohne Login)
    mkdir -p "$HOME/.config/systemd/user"
    printf '%s\n' "$RENDER" | sed '/^User=/d' > "$HOME/.config/systemd/user/risk-layer.service"
    systemctl --user daemon-reload && systemctl --user enable --now risk-layer.service
    ok "User-Dienst aktiv: systemctl --user status risk-layer"
    info "Boot ohne Login: sudo loginctl enable-linger $(id -un)"
  fi
elif [[ $START -eq 1 ]]; then
  schritt "Engine starten (Engine + LLM, Vordergrund)"
  export RISK_LAYER_TOKEN="$TOKEN"
  exec "$PY" -m risk_layer.web --host "$BIND_HOST" --port "$PORT" --llm-autostart
fi
