#!/bin/sh
# Baut aus einem kontrollierten Signal-Checkout ein lokal getaggtes,
# self-contained CPU-Image fuer den von TaxTronik verwalteten Betrieb.
#
# Der Build installiert die Embedding- und Quantenlos-Runtime und snapshotet das
# lokale Modell, berechnet aber bewusst KEINEN Embedding-Index. Der
# rechenintensive Aufbau bleibt damit hinter der ausdruecklichen Bestaetigung in
# TaxTronik.
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
cd "$ROOT"

TARGET_IMAGE="${1:-}"
case "$TARGET_IMAGE" in
  ""|-*|*[!A-Za-z0-9._:/-]*)
    echo "Aufruf: $0 <lokales-image:tag>" >&2
    exit 2
    ;;
esac

command -v docker >/dev/null 2>&1 || {
  echo "docker fehlt" >&2
  exit 1
}
command -v git >/dev/null 2>&1 || {
  echo "git fehlt" >&2
  exit 1
}
command -v tar >/dev/null 2>&1 || {
  echo "tar fehlt" >&2
  exit 1
}

SHA="$(git rev-parse --verify HEAD 2>/dev/null || true)"
case "$SHA" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
  *) echo "Signal-Checkout hat keinen gueltigen HEAD-Commit" >&2; exit 1 ;;
esac

SHORT_SHA="$(printf '%s' "$SHA" | cut -c1-12)"
RELEASE_ID="source-$SHORT_SHA"
EPOCH="$(git show -s --format=%ct HEAD)"
RUNTIME_IMAGE="taxtronik/risk-layer-engine-runtime:$RELEASE_ID"
MEMORY_LIMIT="${SIGNAL_BUILD_MEMORY_LIMIT:-3g}"
CPU_LIMIT="${SIGNAL_BUILD_CPUS:-2}"
MEMORY_NUMBER="${MEMORY_LIMIT%?}"
MEMORY_UNIT="${MEMORY_LIMIT#"$MEMORY_NUMBER"}"
case "$MEMORY_NUMBER" in
  0|""|*[!0-9]*)
    echo "SIGNAL_BUILD_MEMORY_LIMIT muss z. B. 4096m oder 4g sein" >&2
    exit 2
    ;;
esac
case "$MEMORY_UNIT" in
  m|M|g|G) ;;
  *) echo "SIGNAL_BUILD_MEMORY_LIMIT muss z. B. 4096m oder 4g sein" >&2; exit 2 ;;
esac
case "$CPU_LIMIT" in
  [1-9]|[1-9][0-9]) ;;
  *) echo "SIGNAL_BUILD_CPUS muss eine ganze Zahl zwischen 1 und 99 sein" >&2; exit 2 ;;
esac

ARTIFACT="$ROOT/taxtronik-festwissen-$RELEASE_ID.tar.gz"
MANAGED_RELEASE="$ROOT/.managed-release"
CACHE="$ROOT/.signal/managed-build-cache"
cleanup() {
  rm -f -- "$ARTIFACT" "$ARTIFACT.sha256" "$ARTIFACT.manifest.json"
  rm -rf -- "$MANAGED_RELEASE"
}
trap cleanup 0 1 2 15
cleanup
install -d -m 0755 "$MANAGED_RELEASE" "$CACHE"

echo "=== Signal-Runtime aus Commit $SHORT_SHA bauen ==="
docker build \
  --resource "memory=$MEMORY_LIMIT" \
  --resource "memory-swap=$MEMORY_LIMIT" \
  --resource "cpu-period=100000" \
  --resource "cpu-quota=$((CPU_LIMIT * 100000))" \
  --build-arg WITH_EMBEDDING=1 \
  --build-arg WITH_QUANTEN=1 \
  -t "$RUNTIME_IMAGE" "$ROOT"

echo "=== Festwissen ohne automatischen Embedding-Index bauen ==="
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --cpus "$CPU_LIMIT" \
  --memory "$MEMORY_LIMIT" --memory-swap "$MEMORY_LIMIT" \
  -e HOME=/tmp -e "GIT_SHA=$SHA" -e "SOURCE_DATE_EPOCH=$EPOCH" \
  -v "$ROOT:/src" -w /src \
  "$RUNTIME_IMAGE" \
  sh ./scripts/build-release-artifact.sh "$RELEASE_ID" --ohne-embedding \
    --corpus-snapshot ./release-input/corpus/gii

tar -xzf "$ARTIFACT" -C "$MANAGED_RELEASE" --no-same-owner --no-same-permissions

echo "=== Manifestiertes BGE-M3-Modell lokal snapshotten (kein Index-Build) ==="
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --cpus "$CPU_LIMIT" \
  --memory "$MEMORY_LIMIT" --memory-swap "$MEMORY_LIMIT" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/src/.signal/managed-build-cache \
  -e HF_HOME=/src/.signal/managed-build-cache/huggingface \
  -e SENTENCE_TRANSFORMERS_HOME=/src/.signal/managed-build-cache/sentence-transformers \
  -e TORCH_HOME=/src/.signal/managed-build-cache/torch \
  -v "$ROOT:/src" -w /src \
  "$RUNTIME_IMAGE" \
  python -m tools.snapshot_embedding_model --output /src/.managed-release/models/bge-m3

echo "=== Self-contained Signal-Image $TARGET_IMAGE bauen ==="
docker build \
  --resource "memory=$MEMORY_LIMIT" \
  --resource "memory-swap=$MEMORY_LIMIT" \
  --resource "cpu-period=100000" \
  --resource "cpu-quota=$((CPU_LIMIT * 100000))" \
  -f Dockerfile.release \
  --build-arg "ENGINE_BASE_IMAGE=$RUNTIME_IMAGE" \
  --build-arg ALLOW_MISSING_EMBEDDING_INDEX=1 \
  -t "$TARGET_IMAGE" "$ROOT"

docker run --rm --entrypoint python "$TARGET_IMAGE" -c \
  "import importlib.util,pathlib,sys; required=['/release/catalog/begriffe.yaml','/release/corpus/graph.sqlite','/release/models/bge-m3/model-manifest.json']; modules=['sentence_transformers','qiskit','qiskit_aer','qiskit_ibm_runtime']; sys.exit(0 if all(pathlib.Path(p).is_file() for p in required) and all(importlib.util.find_spec(m) is not None for m in modules) else 1)"

echo "=== fertig: $TARGET_IMAGE ==="
echo "Der Embedding-Index wurde nicht automatisch aufgebaut."
