"""Schicht-1.7-Reranker: Umsortierung, Score-Anhang, Abstention-Schwelle.

Der Cross-Encoder selbst wird über einen Mock geprüft (kein Modell-Download):
die Logik – nach rerank-Score sortieren, Score anhängen, Top-k – ist im Code.
"""

import sys
from types import SimpleNamespace

import pytest

from risk_layer.reranker import (HF_MODELL_REVISIONEN, CrossEncoderReranker,
                                 Qwen3Reranker, rerank_normen, rerank_schwelle)


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


def test_bge_cross_encoder_laedt_exakten_commit(monkeypatch):
    aufrufe = []

    def fake_cross_encoder(name, **kwargs):
        aufrufe.append((name, kwargs))
        return SimpleNamespace()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=fake_cross_encoder),
    )
    reranker = CrossEncoderReranker(device="cuda")
    reranker._laden()
    revision = HF_MODELL_REVISIONEN["BAAI/bge-reranker-v2-m3"]
    assert reranker.revision == revision
    assert aufrufe == [("BAAI/bge-reranker-v2-m3", {
        "device": "cuda",
        "revision": revision,
    })]


def test_lokaler_cross_encoder_bleibt_ohne_hf_revision(monkeypatch, tmp_path):
    modellpfad = tmp_path / "lokaler-reranker"
    modellpfad.mkdir()
    aufrufe = []

    def fake_cross_encoder(name, **kwargs):
        aufrufe.append((name, kwargs))
        return SimpleNamespace()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=fake_cross_encoder),
    )
    reranker = CrossEncoderReranker(modell=str(modellpfad))
    reranker._laden()
    assert reranker.revision is None
    assert aufrufe[0][0] == str(modellpfad)
    assert aufrufe[0][1]["revision"] is None


def test_kuratierter_reranker_laesst_pin_nicht_ueberschreiben():
    with pytest.raises(ValueError, match="ausschließlich Revision"):
        CrossEncoderReranker(revision="0" * 40)


@pytest.mark.parametrize(("variante", "modell"), [
    ("0.6B", "Qwen/Qwen3-Reranker-0.6B"),
    ("4B", "Qwen/Qwen3-Reranker-4B"),
    ("8B", "Qwen/Qwen3-Reranker-8B"),
])
def test_qwen_reranker_pin_fuer_tokenizer_und_modell(
        monkeypatch, variante, modell):
    tokenizer_aufrufe = []
    modell_aufrufe = []

    class FakeTokenizer:
        @staticmethod
        def convert_tokens_to_ids(token):
            return {"yes": 1, "no": 2}[token]

        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_ids=[1, 2])

    class FakeModell:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def tokenizer_from_pretrained(name, **kwargs):
        tokenizer_aufrufe.append((name, kwargs))
        return FakeTokenizer()

    def modell_from_pretrained(name, **kwargs):
        modell_aufrufe.append((name, kwargs))
        return FakeModell()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float16=object(),
        float32=object(),
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_from_pretrained),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=modell_from_pretrained),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr("risk_layer.reranker.patch_rocm_dist", lambda: None)

    reranker = Qwen3Reranker(variante=variante, device="cpu")
    reranker._laden()
    revision = HF_MODELL_REVISIONEN[modell]
    assert reranker.revision == revision
    assert tokenizer_aufrufe == [(modell, {
        "padding_side": "left",
        "revision": revision,
    })]
    assert modell_aufrufe[0][0] == modell
    assert modell_aufrufe[0][1]["revision"] == revision
