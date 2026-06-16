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
#   ./deploy.sh --ohne-ner              # ohne spaCy-NER (schlank/offline)
#   ./deploy.sh --start                 # Setup, dann Engine starten
#
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_DEPS=0; SKIP_MODEL=0; SKIP_ENGINE=0; SKIP_CORPUS=0; WITH_EMBEDDING=0; OHNE_NER=0; VOLL=0; FORCE=0; START=0; SERVICE=0
MODELL=""; BACKEND="auto"; BIND_HOST="127.0.0.1"; PORT=8000; CUDA_WHEEL="cu128"
VENV_DIR="${RISK_LAYER_VENV:-.venv}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-deps)      SKIP_DEPS=1 ;;
    --skip-model)     SKIP_MODEL=1 ;;
    --skip-engine)    SKIP_ENGINE=1 ;;
    --skip-corpus)    SKIP_CORPUS=1 ;;
    --with-embedding) WITH_EMBEDDING=1 ;;
    --ohne-ner)       OHNE_NER=1 ;;
    --voll)           VOLL=1 ;;
    --force)          FORCE=1 ;;
    --start)          START=1 ;;
    --service)        SERVICE=1 ;;
    --modell)         MODELL="${2:?--modell braucht einen Wert}"; shift ;;
    --backend)        BACKEND="${2:?--backend braucht einen Wert}"; shift ;;
    --host)           BIND_HOST="${2:?--host braucht einen Wert}"; shift ;;
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

# --- Abhaengigkeiten -------------------------------------------------------
if [[ $SKIP_DEPS -eq 0 ]]; then
  schritt "Abhaengigkeiten installieren"
  ensure_pip
  "$PY" -m pip install --upgrade pip >/dev/null
  # Kern (PyYAML) + dev exakt aus dem Lock (deterministisch ueber die Flotte,
  # identisch zu CI/Dockerfile); das Paket selbst mit --no-deps, damit die
  # >=-Ranges aus pyproject.toml das Lock nicht ueberstimmen.
  # numpy bewusst ungepinnt (GPU-/Plattform-Matrix, ML-Extra).
  "$PY" -m pip install -r requirements-lock.txt "numpy>=1.26"
  "$PY" -m pip install --no-deps -e .
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
else info "Deps uebersprungen (--skip-deps)"; fi

# --- NER-Schicht (Anonymisierung, §203-Gate) --------------------------------
if [[ $OHNE_NER -eq 0 ]]; then
  schritt "NER-Schicht der Anonymisierung (spaCy + de_core_news_lg)"
  # Idempotent: Modell-Paket schon importierbar (lg, Fallback sm) -> ueberspringen.
  # find_spec statt Modell-Laden: reiner Metadaten-Lookup (laedt die ~0,5 GB NICHT).
  if py_ok "import importlib.util,sys; sys.exit(0 if any(importlib.util.find_spec(m) for m in ('de_core_news_lg','de_core_news_sm')) else 1)" && [[ $FORCE -eq 0 ]]; then
    ok "spaCy-Modell bereits installiert - uebersprungen"
  else
    # spaCy Range-basiert wie die anderen ML-Extras (bewusst nicht im Lock, siehe
    # requirements.in); -e haelt die editierbare Kern-Installation von oben intakt
    # (ein nicht-editierbares ".[nlp]" wuerde sie durch eine Kopie ersetzen).
    # `spacy download` loest die zur installierten spaCy-Version passende
    # Modellversion auf. Fehler hier sind NICHT fatal (wie der Engine-Download):
    # die Anonymisierung degradiert auf den deterministischen Regex-Pfad.
    info "installiert spaCy (Extra nlp) + Modell de_core_news_lg ..."
    if "$PY" -m pip install --upgrade --upgrade-strategy eager "click>=8.1" -e ".[nlp]" \
       && spacy_ok \
       && ( "$PY" -m spacy download de_core_news_lg \
            || "$PY" -m spacy download de_core_news_sm ); then
      if py_ok "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('de_core_news_lg') else 1)"; then
        ok "NER-Schicht installiert (de_core_news_lg)"
      else
        ok "NER-Schicht installiert (de_core_news_sm)"
      fi
    else
      printf "${Y}  [warnung] NER-Provisionierung fehlgeschlagen - NER-Schicht inaktiv,${N}\n"
      printf "${Y}  Engine laeuft regex-only weiter. Nachholen mit:${N}\n"
      printf "${Y}    pip install -e \".[nlp]\" && python -m spacy download de_core_news_lg${N}\n"
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
printf "${Y}Bearer-Token:${N} %s\n" "$TOKEN"
printf "${Y}Authorization-Header:${N} Authorization: Bearer %s\n" "$TOKEN"

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
  info "NER inaktiv (--ohne-ner) - Anonymisierung regex-only; nachholen: pip install -e \".[nlp]\" && python -m spacy download de_core_news_lg"
else
  info "NER inaktiv (Modell fehlt) - Anonymisierung regex-only; nachholen: pip install -e \".[nlp]\" && python -m spacy download de_core_news_lg"
fi

printf "\n${G}Deploy abgeschlossen.${N}\n"
printf "${Y}Ein-Befehl-Start (Engine + LLM):${N}\n"
printf "  export RISK_LAYER_TOKEN=\"%s\"\n" "$TOKEN"
printf "  %s -m risk_layer.web --host %s --port %s --llm-autostart\n" "$PY" "$BIND_HOST" "$PORT"
printf "${Y}Als systemd-Dienst (Autostart bei Boot): ./deploy.sh --service${N}\n"

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
    systemctl daemon-reload && systemctl enable --now risk-layer.service
    systemctl is-active --quiet risk-layer.service || { systemctl --no-pager status risk-layer.service; exit 1; }
    ok "System-Dienst aktiv: systemctl status risk-layer"
  else
    # Ohne root: User-Unit (läuft bei Login; 'loginctl enable-linger' für Boot ohne Login)
    mkdir -p "$HOME/.config/systemd/user"
    printf '%s\n' "$RENDER" | sed '/^User=/d' > "$HOME/.config/systemd/user/risk-layer.service"
    systemctl --user daemon-reload && systemctl --user enable --now risk-layer.service
    systemctl --user is-active --quiet risk-layer.service || { systemctl --user --no-pager status risk-layer.service; exit 1; }
    ok "User-Dienst aktiv: systemctl --user status risk-layer"
    info "Boot ohne Login: sudo loginctl enable-linger $(id -un)"
  fi
elif [[ $START -eq 1 ]]; then
  schritt "Engine starten (Engine + LLM, Vordergrund)"
  export RISK_LAYER_TOKEN="$TOKEN"
  exec "$PY" -m risk_layer.web --host "$BIND_HOST" --port "$PORT" --llm-autostart
fi
