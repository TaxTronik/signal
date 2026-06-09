"""Schicht-1.7-Reranker: Umsortierung, Score-Anhang, Abstention-Schwelle.

Der Cross-Encoder selbst wird über einen Mock geprüft (kein Modell-Download):
die Logik – nach rerank-Score sortieren, Score anhängen, Top-k – ist im Code.
"""

from risk_layer.reranker import rerank_normen, rerank_schwelle


class _MockReranker:
    name = "mock"

    def __init__(self, nach_doc: dict[str, float]):
        self._nach_doc = nach_doc

    def score(self, paare):
        return [self._nach_doc.get(doc, 0.0) for _q, doc in paare]


def test_rerank_sortiert_nach_score_und_haengt_ihn_an():
    normen = [{"norm_id": "A", "sim": 0.9}, {"norm_id": "B", "sim": 0.6},
              {"norm_id": "C", "sim": 0.7}]
    rr = _MockReranker({"A": 0.10, "B": 0.90, "C": 0.50})
    out = rerank_normen(rr, "frage", normen, doc_fn=lambda nid: nid, k=3)
    assert [n["norm_id"] for n in out] == ["B", "C", "A"]   # rerank-Reihenfolge
    assert out[0]["rerank"] == 0.9 and out[0]["sim"] == 0.6  # sim bleibt, rerank dazu


def test_rerank_ohne_reranker_ist_einfach_top_k():
    normen = [{"norm_id": "A"}, {"norm_id": "B"}, {"norm_id": "C"}]
    assert rerank_normen(None, "frage", normen, doc_fn=lambda x: x, k=2) == normen[:2]
    # Auch ein Einzelkandidat wird gescort: der rerank-Score ist die
    # Abstention-Basis der Pipeline (die frühere len<=1-Abkürzung ließ
    # Einzelkandidaten OHNE Score ungeprüft passieren).
    einzel = rerank_normen(_MockReranker({}), "f", normen[:1], lambda x: x, k=5)
    assert einzel == [{"norm_id": "A", "rerank": 0.0}]


def test_rerank_schwelle_default_und_env(monkeypatch):
    monkeypatch.delenv("RISK_LAYER_RERANK_SCHWELLE", raising=False)
    assert rerank_schwelle() == 0.10
    monkeypatch.setenv("RISK_LAYER_RERANK_SCHWELLE", "0.25")
    assert rerank_schwelle() == 0.25
    monkeypatch.setenv("RISK_LAYER_RERANK_SCHWELLE", "quatsch")
    assert rerank_schwelle() == 0.10                        # robust gegen Müll
