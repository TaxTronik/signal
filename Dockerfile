# =============================================================================
# Risk-Layer Engine – Container für die TaxTronik-Integration
#
# Liefert die ZUSTANDSLOSE Erkennungs-/Wissens-Engine über /v1/* (siehe
# INTEGRATION_TAXTRONIK.md). Persistiert KEINE Mandantendaten – nur Festwissen
# (Katalog, Normgraph, Embedding-Index) als gemountete Volumes.
#
# Default-Build = schlank (PyYAML + numpy): deterministische Schichten 1/1b/1.5,
# Normgraph, Zitat-Parser. Die Embedding-Schicht (Schicht 1.7, ~2,3 GB BGE-M3)
# ist optional über das Build-Arg WITH_EMBEDDING=1 zuschaltbar – sie lädt das
# Modell lazy zur Laufzeit; ohne sie liefert /v1/embedding/suche schlicht leer.
# Schicht 2 (LLM) läuft als SEPARATER llama-server (nicht in diesem Image).
# =============================================================================
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RISK_LAYER_EMB_DEVICE=cpu

WORKDIR /app

# Nur die Projektmetadaten zuerst (Layer-Caching der Dependencies): die
# Dependencies werden VOR dem Quellcode installiert — eine reine Code-Änderung
# in risk_layer/ invalidiert diesen (teuren) Layer damit NICHT.
COPY pyproject.toml README.md ./

ARG WITH_EMBEDDING=0
# Basis immer; Embedding nur, wenn ausdrücklich gewünscht. "PyYAML>=6.0" spiegelt
# [project].dependencies aus pyproject.toml (Two-Step: Deps explizit, Paket später
# mit --no-deps — pip kann die Deps nicht ohne den Quellcode auflösen).
RUN pip install --upgrade pip \
 && pip install "numpy>=1.26" "PyYAML>=6.0" \
 && if [ "$WITH_EMBEDDING" = "1" ]; then pip install "sentence-transformers>=2.7"; fi

# Quellcode erst NACH den Dependencies kopieren + Paket ohne Dep-Auflösung installieren.
COPY risk_layer ./risk_layer
COPY catalog ./catalog
RUN pip install --no-deps .

# Festwissen-Volumes (regenerierbar, mandantenfrei). corpus/ wird gemountet:
#   corpus/graph.sqlite     – Normgraph (graph build)
#   corpus/embedding/       – Embedding-Index (embedding build), optional
# catalog/begriffe.yaml ist eingebacken, kann aber überlagert gemountet werden.
VOLUME ["/app/corpus"]

EXPOSE 8000

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
