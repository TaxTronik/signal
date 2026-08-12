#!/bin/sh
# =============================================================================
# Baut ein reproduzierbares, manifestiertes FESTWISSEN-Release. Der Default ist
# explizit OHNE Embedding; der Inhalt hängt nicht mehr von zufällig installierten
# Python-Paketen ab. Mandanten- und Beraterdaten werden niemals aufgenommen.
#
# Aufruf:
#   ./scripts/build-release-artifact.sh VERSION [--mit-embedding|--ohne-embedding]
# Für byte-identische Wiederholungen denselben SOURCE_DATE_EPOCH mitgeben.
# =============================================================================
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
cd "$ROOT"

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
VER="${1:-$(date +%Y-%m-%d)}"
MODUS="${2:---ohne-embedding}"
case "$MODUS" in
  --mit-embedding) MIT_EMBEDDING=1 ;;
  --ohne-embedding) MIT_EMBEDDING=0 ;;
  *) echo "Unbekannte Option: $MODUS" >&2; exit 2 ;;
esac

GIT_SHA="${GIT_SHA:-$(git rev-parse HEAD 2>/dev/null || printf unknown)}"
SOURCE_EPOCH="${SOURCE_DATE_EPOCH:-$(date +%s)}"
export SOURCE_DATE_EPOCH="$SOURCE_EPOCH"
OUT="taxtronik-festwissen-$VER.tar.gz"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/risk-layer-release.XXXXXXXX")"
cleanup() {
  case "$STAGE" in
    "${TMPDIR:-/tmp}"/risk-layer-release.*) rm -rf -- "$STAGE" ;;
    *) echo "Unsicheres Staging-Verzeichnis wird nicht entfernt: $STAGE" >&2 ;;
  esac
}
trap cleanup 0 1 2 15
CORPUS="$STAGE/corpus"
mkdir -p "$STAGE/catalog"
install -m 0644 catalog/begriffe.yaml "$STAGE/catalog/begriffe.yaml"

echo "=== Festwissen bauen (Version $VER, Embedding=$MIT_EMBEDDING) ==="
"$PY" -m risk_layer.cli korpus init --dir "$CORPUS"
"$PY" -m risk_layer.cli korpus fetch --dir "$CORPUS"
# `korpus fetch` war historisch schon bei einem einzigen erfolgreichen Download
# exit 0. Das separate Gate verlangt aus dem leeren Staging exakt jede
# konfigurierte, parsebare XML und schließt dadurch Teil-/Altbestände aus.
"$PY" -m tools.verify_corpus_fetch "$CORPUS"
"$PY" -m risk_layer.cli graph build \
  --dir "$CORPUS" --db "$CORPUS/graph.sqlite"
"$PY" -m risk_layer.cli trigger build \
  --db "$CORPUS/trigger.sqlite" --graph-db "$CORPUS/graph.sqlite"

if [ "$MIT_EMBEDDING" = "1" ]; then
  "$PY" -c "import sentence_transformers" >/dev/null 2>&1 || {
    echo "--mit-embedding verlangt sentence-transformers" >&2
    exit 1
  }
  MODEL="$STAGE/models/bge-m3"
  "$PY" -m tools.snapshot_embedding_model --output "$MODEL"
  export RISK_LAYER_EMBEDDING_MODEL="$MODEL"
  export RISK_LAYER_EMBEDDING_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  "$PY" -m risk_layer.cli embedding build \
    --encoder bge-m3 --model "$MODEL" --device cpu \
    --dir "$CORPUS/embedding" --graph-db "$CORPUS/graph.sqlite"
fi

if [ "$MIT_EMBEDDING" = "1" ]; then
  "$PY" -m tools.package_festwissen_release \
    --release-id "$VER" \
    --git-sha "$GIT_SHA" \
    --source-date-epoch "$SOURCE_EPOCH" \
    --output "$OUT" \
    --source-root "$STAGE" \
    --embedding
else
  "$PY" -m tools.package_festwissen_release \
    --release-id "$VER" \
    --git-sha "$GIT_SHA" \
    --source-date-epoch "$SOURCE_EPOCH" \
    --output "$OUT" \
    --source-root "$STAGE"
fi

echo "=== fertig ==="
echo "  $OUT"
echo "  $OUT.sha256"
echo "  $OUT.manifest.json"
echo "Installation ausschließlich über das profilbasierte signal; nicht live entpacken."
