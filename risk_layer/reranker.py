"""Schicht-1.7-Reranker – Cross-Encoder ordnet die Bi-Encoder-Top-N um.

Der Bi-Encoder (BGE-M3) liefert je Satz die k semantisch nächsten §§. Ein Cross-
Encoder bewertet danach ``(Satz, Norm-Doc)``-PAARE gemeinsam (volle Attention statt
vorberechneter Vektoren) und sortiert um. Entscheidend (gemessen): der Norm-Doc ist
ANKER-BEWUSST – Gesetzestext PLUS die kuratierten Laien-Paraphrasen des §. So sieht
der Reranker beide Register und kämpft NICHT gegen die Anker. Effekt auf der Gold-
Menge: held-out-Recall@5 0,17 -> 0,35 (naiv gegen reinen Gesetzestext: nur 0,19,
und Produktion -0,20 – darum anker-bewusst).

Bleibt Projekt-Prinzip treu: bewertet wird gegen KORPUS-Text (Gesetz + kuratierte
Paraphrasen), nicht gegen Modellwissen. Optional + degradierend wie der Encoder:
ohne sentence-transformers liefert ``reranker_oeffnen`` einfach ``None``.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable

from .idschema import para_level


@runtime_checkable
class Reranker(Protocol):
    name: str

    def score(self, paare: list[tuple[str, str]]) -> list[float]:
        ...


class CrossEncoderReranker:
    """BAAI/bge-reranker-v2-m3 (Apache-2.0, mehrsprachig). Lazy + thread-sicher
    (prozessweit geteilt im ThreadingHTTPServer); ROCm-Workaround wie beim Encoder."""

    name = "bge-reranker-v2-m3"

    def __init__(self, modell: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
        self._modell_name = modell
        self._device = device
        self._modell = None
        self._lock = threading.Lock()

    def _laden(self):
        if self._modell is None:
            with self._lock:
                if self._modell is None:
                    try:                       # ROCm: fehlendes is_initialized patchen
                        import torch.distributed as _dist
                        if not hasattr(_dist, "is_initialized"):
                            _dist.is_initialized = lambda: False
                    except ImportError:
                        pass
                    from sentence_transformers import CrossEncoder  # lazy
                    self._modell = CrossEncoder(self._modell_name, device=self._device)
        return self._modell

    def score(self, paare: list[tuple[str, str]]) -> list[float]:
        if not paare:
            return []
        m = self._laden()
        with self._lock:                       # forward-Pass serialisieren
            return [float(x) for x in m.predict(paare)]


def reranker_oeffnen(name: str = "auto", *, device: str | None = None) -> Reranker | None:
    """``auto``/``bge-reranker-v2-m3`` -> Cross-Encoder, falls verfügbar; sonst None.
    ``aus`` schaltet ab. Fehlt sentence-transformers, degradiert ``auto`` zu None."""
    if name in ("aus", "none", "off"):
        return None
    if name in ("auto", "bge-reranker-v2-m3", "bge-reranker"):
        try:
            import sentence_transformers  # noqa: F401
            return CrossEncoderReranker(device=device)
        except Exception:
            if name != "auto":
                raise
            return None
    return CrossEncoderReranker(modell=name, device=device)


def baue_norm_doc_fn(katalog, graph, *, max_text: int = 300):
    """Liefert ``norm_id -> ANKER-BEWUSSTEN Doc-Text``: Titel + Gesetzestext (gekappt)
    + die kuratierten Laien-Paraphrasen aller Katalog-Begriffe, deren Norm auf diesen
    § auflöst. Genau diese Doppel-Repräsentation hebt den Rerank-Gewinn."""
    from .zitate import parse_normzitat

    anker: dict[str, list[str]] = {}
    for e in katalog:
        phrasen = list(getattr(e, "paraphrasen", ()) or [])
        if not phrasen:
            continue
        quellen = list(getattr(e, "norm_anker", ()) or []) + \
            [k.primaer for k in (getattr(e, "normketten", ()) or []) if k.primaer]
        for z in quellen:
            for a in parse_normzitat(z):
                if a.norm_id:
                    anker.setdefault(para_level(a.norm_id.to_id()), []).extend(phrasen)
                    break

    cache: dict[str, str] = {}

    def doc_fn(norm_id: str) -> str:
        pid = para_level(norm_id)
        if pid not in cache:
            r = graph.con.execute(
                "SELECT titel, text FROM knoten WHERE id=?", (pid,)).fetchone()
            titel = (r["titel"] or "").strip() if r else ""
            text = (r["text"] or "")[:max_text] if r else ""
            doc = f"{titel}. {text}"
            if anker.get(pid):
                doc += " || " + " | ".join(anker[pid])
            cache[pid] = doc.strip() or pid
        return cache[pid]

    return doc_fn


def rerank_normen(reranker: Reranker | None, query: str,
                  normen: list[dict[str, Any]], doc_fn, k: int) -> list[dict[str, Any]]:
    """Sortiert die Norm-Kandidaten eines Satzes per Cross-Encoder um (gegen
    ``doc_fn(norm_id)``) und gibt die Top-k zurück. Ohne Reranker: einfach Top-k."""
    if reranker is None or len(normen) <= 1:
        return normen[:k]
    scores = reranker.score([(query, doc_fn(n["norm_id"])) for n in normen])
    rang = sorted(range(len(normen)), key=lambda i: scores[i], reverse=True)
    return [normen[i] for i in rang[:k]]
