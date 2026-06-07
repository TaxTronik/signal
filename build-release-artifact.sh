#!/usr/bin/env bash
# =============================================================================
# Baut das FESTWISSEN-Release-Artefakt EINMAL zentral: Korpus + Normgraph +
# Trigger + Embedding-Index. Kanzleien ZIEHEN das Artefakt, statt es lokal aus
# gesetze-im-internet zu bauen (Minuten statt Stunden, deterministisch über die
# Flotte, keine Internet-Abhängigkeit beim Kunden).
#
# Mandantenfrei + regenerierbar — enthält NUR Festwissen, nie Mandantendaten.
# Aufruf:  ./build-release-artifact.sh [VERSION]   (Default: heutiges Datum)
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python3; command -v "$PY" >/dev/null 2>&1 || PY=python
VER="${1:-$(date +%Y-%m-%d)}"

echo "=== Festwissen bauen (Version $VER) ==="
"$PY" -m risk_layer.cli korpus init
"$PY" -m risk_layer.cli korpus fetch
"$PY" -m risk_layer.cli graph build
"$PY" -m risk_layer.cli trigger build
if "$PY" -c "import sentence_transformers" 2>/dev/null; then
  "$PY" -m risk_layer.cli embedding build
  EMB="corpus/embedding"
else
  echo "  (sentence-transformers fehlt -> Embedding-Index NICHT im Artefakt)"; EMB=""
fi

OUT="taxtronik-festwissen-$VER.tar.gz"
# corpus/gii (Roh-XML) bewusst NICHT mit ausliefern — nur die gebauten Artefakte.
tar -czf "$OUT" \
  corpus/gesetze.yaml corpus/graph.sqlite corpus/trigger.sqlite $EMB \
  catalog/begriffe.yaml
( command -v sha256sum >/dev/null && sha256sum "$OUT" || shasum -a 256 "$OUT" ) > "$OUT.sha256"
echo "=== fertig: $OUT (+ .sha256) ==="
echo "Auf dem Zielrechner:  tar -xzf $OUT   (überschreibt corpus/ + catalog/begriffe.yaml)"
