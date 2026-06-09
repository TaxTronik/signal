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

import os
import threading
from typing import Any, Protocol, runtime_checkable

from .idschema import para_level
from .torch_compat import patch_rocm_dist

# Bi-Encoder-Top-N, die der Cross-Encoder umsortiert (mehr Headroom = mehr Chance,
# einen vom Bi-Encoder vergrabenen Treffer hochzuholen).
RERANK_TOPN = 20


def rerank_schwelle() -> float:
    """Mindest-Rerank-Score, ab dem ein Vorschlag überhaupt erscheint (Abstention).
    Gemessen (Gold): bei 0,10 fallen ALLE neutralen Sätze raus, 10/15 Positive
    bleiben. Per ``RISK_LAYER_RERANK_SCHWELLE`` tunebar."""
    try:
        return float(os.environ.get("RISK_LAYER_RERANK_SCHWELLE", "0.10"))
    except ValueError:
        return 0.10


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
                    patch_rocm_dist()          # ROCm: fehlendes is_initialized abfangen
                    try:
                        from sentence_transformers import CrossEncoder  # lazy
                        modell = CrossEncoder(self._modell_name, device=self._device)
                    except Exception as exc:
                        # Lazy-Load schlaegt erst beim ERSTEN Request fehl – dann
                        # braucht der Aufrufer eine handelbare Meldung statt eines
                        # nackten ImportError tief im Stack. self._modell bleibt
                        # None ⇒ der naechste Aufruf versucht es erneut (kein
                        # halb-initialisierter Zustand).
                        raise RuntimeError(
                            f"Reranker-Modell '{self._modell_name}' konnte nicht geladen werden: {exc} — "
                            "Abhilfe: pip install \".[embedding]\"; offline muss das "
                            "Modell bereits im HF-Cache liegen."
                        ) from exc
                    # fp16 auf der GPU: ~25x schneller (11,9 s -> 0,47 s je Dokument).
                    # fp32 ist auf RDNA4/ROCm eine Falle – dieselbe Lektion wie beim LLM.
                    # CPU bleibt fp32 (half() wird dort nicht unterstützt).
                    try:
                        if next(modell.model.parameters()).device.type == "cuda":
                            modell.model.half()
                    except Exception:
                        pass
                    self._modell = modell      # erst NACH Erfolg zuweisen
        return self._modell

    def score(self, paare: list[tuple[str, str]]) -> list[float]:
        if not paare:
            return []
        m = self._laden()
        with self._lock:                       # forward-Pass serialisieren
            return [float(x) for x in m.predict(paare)]


class Qwen3Reranker:
    """Qwen3-Reranker (0.6B/4B/8B) — KEIN CrossEncoder, sondern ein Causal-LM, das
    je (Query, Doc) die Wahrscheinlichkeit von 'yes' (relevant) liefert. Eigene
    Inferenz über transformers (yes/no-Logit am letzten Token). Multilingual; fp16
    auf der GPU. Bewertet weiterhin gegen KORPUS-Doc (anker-bewusst), nicht Modellwissen."""

    INSTRUCTION = ("Beurteile, ob die Rechtsnorm (Document) den im Sachverhalt (Query) "
                   "beschriebenen Rechtsbegriff/Sachverhalt trifft.")
    PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based '
              'on the Query and the Instruct provided. Note that the answer can only be '
              '"yes" or "no".<|im_end|>\n<|im_start|>user\n')
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, variante: str = "4B", device: str | None = None):
        self.name = f"qwen3-reranker-{variante.lower()}"
        self._modell_name = f"Qwen/Qwen3-Reranker-{variante}"
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    def _laden(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    patch_rocm_dist()
                    try:
                        import torch
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        tok = AutoTokenizer.from_pretrained(self._modell_name, padding_side="left")
                        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
                        dtype = torch.float16 if "cuda" in str(dev) else torch.float32
                        model = AutoModelForCausalLM.from_pretrained(
                            self._modell_name, torch_dtype=dtype).to(dev).eval()
                    except Exception as exc:
                        # s. CrossEncoderReranker._laden: handelbare Meldung, kein
                        # halb-initialisierter Zustand, Retry beim naechsten Aufruf.
                        raise RuntimeError(
                            f"Reranker-Modell '{self._modell_name}' konnte nicht geladen werden: {exc} — "
                            "Abhilfe: pip install \".[embedding]\" plus transformers/torch; "
                            "offline muss das Modell bereits im HF-Cache liegen."
                        ) from exc
                    self._tok, self._dev = tok, dev
                    self._yes = tok.convert_tokens_to_ids("yes")
                    self._no = tok.convert_tokens_to_ids("no")
                    self._pre = tok(self.PREFIX, add_special_tokens=False).input_ids
                    self._suf = tok(self.SUFFIX, add_special_tokens=False).input_ids
                    self._model = model
        return self._model

    def score(self, paare: list[tuple[str, str]]) -> list[float]:
        if not paare:
            return []
        self._laden()    # ZUERST: klare RuntimeError statt nacktem ImportError von torch
        import torch
        out: list[float] = []
        with self._lock:                            # forward-Pass serialisieren
            for i in range(0, len(paare), 16):      # Batches à 16
                batch = paare[i:i + 16]
                ids = []
                for q, d in batch:
                    t = f"<Instruct>: {self.INSTRUCTION}\n<Query>: {q}\n<Document>: {d}"
                    mid = self._tok(t, add_special_tokens=False, truncation=True, max_length=1024).input_ids
                    ids.append(self._pre + mid + self._suf)
                inp = self._tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(self._dev)
                with torch.no_grad():
                    logits = self._model(**inp).logits[:, -1, :]   # links gepaddet -> letztes Token real
                yn = logits[:, [self._no, self._yes]].float()
                out.extend(float(x) for x in torch.softmax(yn, dim=1)[:, 1])
        return out


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
    if name.startswith("qwen3-reranker"):
        var = "0.6B" if name.endswith("0.6b") else "8B" if name.endswith("8b") else "4B"
        return Qwen3Reranker(variante=var, device=device)
    return CrossEncoderReranker(modell=name, device=device)


def baue_norm_doc_fn(katalog, graph, *, max_text: int = 300):
    """Liefert ``norm_id -> ANKER-BEWUSSTEN Doc-Text``: Titel + Gesetzestext (gekappt)
    + die kuratierten Laien-Paraphrasen aller Katalog-Begriffe, deren Norm auf diesen
    § auflöst. Genau diese Doppel-Repräsentation hebt den Rerank-Gewinn."""
    from .zitate import parse_normzitat

    anker: dict[str, list[str]] = {}
    for e in katalog:
        phrasen = list(e.paraphrasen)
        if not phrasen:
            continue
        quellen = list(e.norm_anker) + [k.primaer for k in e.normketten if k.primaer]
        # norm_anker und Ketten-Kopf koennen auf DENSELBEN § aufloesen – ohne
        # Dedup landen die Phrasen dann doppelt im Doc-Text desselben §.
        gesehen: set[str] = set()
        for z in quellen:
            for a in parse_normzitat(z):
                if a.norm_id:
                    pid = para_level(a.norm_id.to_id())
                    if pid not in gesehen:
                        gesehen.add(pid)
                        anker.setdefault(pid, []).extend(phrasen)
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
    ``doc_fn(norm_id)``) und gibt die Top-k zurück; jede Marke trägt ihren
    ``rerank``-Score (für Abstention/Anzeige). Ohne Reranker: einfach Top-k.

    Auch EINZELNE Kandidaten werden gescort: der Rerank-Score ist die
    Abstention-Basis der Pipeline. Die fruehere ``len(normen) <= 1``-Abkuerzung
    sparte nur einen forward-Pass ("umsortieren unnoetig"), liess aber Einzel-
    Kandidaten OHNE Score passieren – mit dem abgesenkten Reranker-Floor (0,35)
    rutschten so gerade die unplausiblen Faelle (Score 0,35–0,49) durch.
    """
    if reranker is None or not normen:
        return normen[:k]
    scores = reranker.score([(query, doc_fn(n["norm_id"])) for n in normen])
    rang = sorted(range(len(normen)), key=lambda i: scores[i], reverse=True)[:k]
    out: list[dict[str, Any]] = []
    for i in rang:
        marke = dict(normen[i])
        marke["rerank"] = round(scores[i], 3)
        out.append(marke)
    return out
