# =============================================================================
# Risk-Layer Engine – Container für die TaxTronik-Integration
#
# Liefert die ZUSTANDSLOSE Erkennungs-/Wissens-Engine über /v1/* (siehe
# INTEGRATION_TAXTRONIK.md). Persistiert KEINE Mandantendaten – nur Festwissen
# (Katalog, Normgraph, Embedding-Index) als gemountete Volumes.
#
# Default-Build = Kern (PyYAML + numpy: deterministische Schichten 1/1b/1.5,
# Normgraph, Zitat-Parser) + NER-Schicht der Anonymisierung (spaCy +
# de_core_news_lg, §203-Gate — exakt gepinnt, siehe Block unten). Die
# Embedding-Schicht (Schicht 1.7, ~2,3 GB BGE-M3) ist optional über das
# Build-Arg WITH_EMBEDDING=1 zuschaltbar – sie lädt das Modell lazy zur
# Laufzeit; ohne sie liefert /v1/embedding/suche schlicht leer.
# Schicht 2 (LLM) läuft als SEPARATER llama-server (nicht in diesem Image).
# =============================================================================
# Basis-Image per DIGEST gepinnt (reproduzierbare Builds): der Tag 3.12-slim
# dokumentiert die Intention, der Digest pinnt den multi-arch Manifest-List-Stand
# (OCI image index, Docker-Content-Digest der Registry-API, Stand 2026-06-10).
# Update: neuen Digest holen via
#   docker buildx imagetools inspect python:3.12-slim
# (oder Registry-API: HEAD registry-1.docker.io/v2/library/python/manifests/3.12-slim)
# und hier ersetzen.
FROM python:3.12-slim@sha256:090ba77e2958f6af52a5341f788b50b032dd4ca28377d2893dcf1ecbdfdfe203 AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_INPUT=1 \
    RISK_LAYER_EMB_DEVICE=cpu \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers

WORKDIR /app

# Nur die Projektmetadaten zuerst (Layer-Caching der Dependencies): die
# Dependencies werden VOR dem Quellcode installiert — eine reine Code-Änderung
# in risk_layer/ invalidiert diesen (teuren) Layer damit NICHT.
COPY pyproject.toml README.md LICENSE \
     requirements-runtime-lock.txt \
     requirements-embedding-linux-cpu-py312-lock.txt \
     requirements-ner-lock.txt ./

ARG WITH_EMBEDDING=0
# Kern exakt aus dem schlanken Runtime-Hashlock (keine Testwerkzeuge). Die
# optionale semantische Schicht hat einen eigenen, in Linux/Python 3.12
# erzeugten CPU-Lock; kein freies Aufloesen von torch/Transformers im Build.
RUN pip install --require-hashes --only-binary=:all: -r requirements-runtime-lock.txt \
 && if [ "$WITH_EMBEDDING" = "1" ]; then \
      pip install --require-hashes --only-binary=:all: \
        -r requirements-embedding-linux-cpu-py312-lock.txt; \
    fi

# --- NER-Schicht (Anonymisierung, §203-Gate): spaCy + de_core_news_lg -------
# EXAKT gepinnt + hash-geprüft in requirements-ner-lock.txt. Das Image soll
# deterministisch sein, und `spacy download` würde die Modellversion zur
# BUILDZEIT neu auflösen. Statt dessen das Modell-Wheel direkt aus den
# spacy-models-Releases. Versionspaar belegt über
#   https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json
# (spacy 3.8.x -> de_core_news_lg 3.8.0; geprüft 2026-06-10). Update: dort das
# neue Paar ablesen und BEIDE Zeilen gemeinsam anheben.
# Größe: Modell ~0,5 GB — vertretbar neben dem torch/sentence-transformers-
# Stack (WITH_EMBEDDING). Schlankes Image gewünscht? Diesen RUN-Block
# auskommentieren — die Engine degradiert auf den deterministischen Regex-Pfad
# (zur Laufzeit erzwingbar per RISK_LAYER_NER=aus).
# Eigener Layer (eigenes RUN): wird unabhängig vom Kern-Layer gecacht.
RUN pip install --require-hashes --only-binary=:all: -r requirements-ner-lock.txt \
 && pip check

# Quellcode erst NACH den Dependencies kopieren + Paket ohne Dep-Auflösung installieren.
COPY risk_layer ./risk_layer
COPY catalog ./catalog
RUN pip install --no-build-isolation --no-deps .

# Laufzeit strikt ohne root. UID/GID sind Build-Args, damit Linux-Betreiber sie
# an den Besitzer des schreibbaren definitionen/-Bind-Mounts anpassen können.
# Nur die drei ausdrücklich benötigten Laufzeitpfade gehören dem Dienstkonto;
# Anwendung und installierte Pakete bleiben root-owned und damit unveränderbar.
ARG RISK_LAYER_UID=1000
ARG RISK_LAYER_GID=1000
RUN groupadd --gid "$RISK_LAYER_GID" risklayer \
 && useradd --uid "$RISK_LAYER_UID" --gid "$RISK_LAYER_GID" \
      --home-dir /home/risklayer --create-home --shell /usr/sbin/nologin risklayer \
 && install -d -o risklayer -g risklayer -m 0750 \
      /app/corpus /app/definitionen /cache

# Festwissen-Volumes (regenerierbar, mandantenfrei). corpus/ wird gemountet:
#   corpus/graph.sqlite     – Normgraph (graph build)
#   corpus/embedding/       – Embedding-Index (embedding build), optional
# catalog/begriffe.yaml ist eingebacken, kann aber überlagert gemountet werden.
VOLUME ["/app/corpus"]

EXPOSE 8000

USER risklayer:risklayer

# Healthcheck gegen den Engine-Vertrag (nicht die GUI). /v1/* ist auth-pflichtig,
# sobald RISK_LAYER_TOKEN gesetzt ist -> der Probe MUSS den Bearer-Token mitsenden
# (sonst 401 -> faelschlich unhealthy). Ohne gesetztes Token (Loopback-Dev) entfaellt
# der Header und der Check laeuft ebenfalls.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request,sys; t=os.environ.get('RISK_LAYER_TOKEN',''); h={'Authorization':'Bearer '+t} if t else {}; sys.exit(0 if urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/v1/health',headers=h),timeout=4).status==200 else 1)"

# Bindet auf 0.0.0.0, damit der Compose-Netzzugriff funktioniert – der Dienst
# wird NICHT nach außen exponiert (nur TaxTronik im selben Netz spricht ihn an,
# mit RISK_LAYER_TOKEN). risk_layer.web ist headless (kein Browser-Start).
CMD ["python", "-m", "risk_layer.web", "--host", "0.0.0.0", "--port", "8000", \
     "--graph", "corpus/graph.sqlite"]
