"""Schicht-1.7-Reranker: Umsortierung, Score-Anhang, Abstention-Schwelle.

Der Cross-Encoder selbst wird über einen Mock geprüft (kein Modell-Download):
die Logik – nach rerank-Score sortieren, Score anhängen, Top-k – ist im Code.
"""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from risk_layer.reranker import (HF_MODELL_REVISIONEN, CascadedQwenReranker, CrossEncoderReranker,
                                 Qwen3Reranker, baue_norm_doc_fn, rerank_normen,
                                 rerank_schwelle, filtere_mit_rettung)


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


def test_rerank_behaelt_reihenfolge_bei_knapp_unterschiedlichen_scores():
    normen = [{"norm_id": "A", "sim": 0.9}, {"norm_id": "B", "sim": 0.6}]
    rr = _MockReranker({"A": 0.5011, "B": 0.5014})
    out = rerank_normen(rr, "frage", normen, doc_fn=lambda nid: nid, k=2)
    behalten = filtere_mit_rettung(out, 0.10, 0.55)
    # Gerundete Scores waeren beide 0.501: der sim-Tie-Breaker wuerde A
    # faelschlich vor B schieben und damit auch die Top-k-Auswahl veraendern.
    assert [n["norm_id"] for n in behalten] == ["B", "A"]
    assert behalten[0]["rerank"] == 0.5014


def test_rerank_ohne_reranker_ist_einfach_top_k():
    normen = [{"norm_id": "A"}, {"norm_id": "B"}, {"norm_id": "C"}]
    assert rerank_normen(None, "frage", normen, doc_fn=lambda x: x, k=2) == normen[:2]
    # Auch ein Einzelkandidat wird gescort: der rerank-Score ist die
    # Abstention-Basis der Pipeline (die frühere len<=1-Abkürzung ließ
    # Einzelkandidaten OHNE Score ungeprüft passieren).
    einzel = rerank_normen(_MockReranker({}), "f", normen[:1], lambda x: x, k=5)
    assert einzel == [{"norm_id": "A", "rerank": 0.0}]


def test_norm_doc_fn_nutzt_thread_sichere_graph_api():
    class FakeGraph:
        def __init__(self):
            self.calls = []

        @property
        def con(self):
            raise AssertionError("SQLite-Verbindung darf nicht direkt verwendet werden")

        def node(self, node_id):
            self.calls.append(node_id)
            return {"titel": "Schätzung", "text": "Besteuerungsgrundlagen"}

    graph = FakeGraph()
    doc_fn = baue_norm_doc_fn([], graph)
    assert doc_fn("norm:AO:162:1") == "Schätzung. Besteuerungsgrundlagen"
    assert graph.calls == ["norm:AO:162"]
    # Der Cache vermeidet auch unter dem sicheren API-Pfad weitere DB-Zugriffe.
    assert doc_fn("norm:AO:162:2") == "Schätzung. Besteuerungsgrundlagen"
    assert graph.calls == ["norm:AO:162"]


def test_rerank_schwelle_default_und_env(monkeypatch):
    monkeypatch.delenv("RISK_LAYER_RERANK_SCHWELLE", raising=False)
    assert rerank_schwelle() == 0.10
    monkeypatch.setenv("RISK_LAYER_RERANK_SCHWELLE", "0.25")
    assert rerank_schwelle() == 0.25
    monkeypatch.setenv("RISK_LAYER_RERANK_SCHWELLE", "quatsch")
    assert rerank_schwelle() == 0.10                        # robust gegen Müll


@pytest.mark.parametrize("rerank, sim, erwartet", [
    (0.009999, 0.55, False),
    (0.01, 0.55, True),
    (0.01, 0.549999, False),
    (None, 0.9, False),
    (0.09, None, False),
    (0.1, None, True),
    (0.1, 0.1, True),
])
def test_optionale_rettungsgrenze_nutzt_rohe_scores(rerank, sim, erwartet):
    kandidat = {"norm_id": "A", "rerank": rerank, "sim": sim}
    treffer = filtere_mit_rettung([kandidat], 0.1, 0.55, rettung_min_rerank=0.01)
    assert treffer == ([kandidat] if erwartet else [])


def test_rettungsgrenze_verschaerft_nicht_den_normalen_rerank_zweig():
    stark = {"norm_id": "stark", "rerank": 0.1, "sim": None}
    schwach = {"norm_id": "schwach", "rerank": 0.05, "sim": 0.9}
    assert filtere_mit_rettung(
        [schwach, stark], 0.1, 0.55, rettung_min_rerank=0.2) == [stark]


def test_rettungsgrenze_default_erhaelt_auch_alte_sim_rettung_ohne_rerank():
    stark = {"norm_id": "stark", "rerank": 0.5, "sim": None}
    schwach = {"norm_id": "schwach", "rerank": 0.001, "sim": 0.55}
    ohne_rerank = {"norm_id": "ohne", "sim": 0.9}
    pool = [ohne_rerank, schwach, stark]
    assert filtere_mit_rettung(pool, 0.1, 0.55) == [stark, schwach, ohne_rerank]
    assert filtere_mit_rettung(pool, 0.1, 0.55, rettung_min_rerank=None) == \
        filtere_mit_rettung(pool, 0.1, 0.55)


def test_rettung_bleibt_mit_optionaler_grenze_abschaltbar():
    kandidat = {"norm_id": "A", "rerank": 0.05, "sim": 0.9}
    assert filtere_mit_rettung([kandidat], 0.1, 0, rettung_min_rerank=0.01) == []


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


def test_kaskade_begrenzt_beide_modelle_und_erhaelt_paarreihenfolge(monkeypatch):
    ranker = CascadedQwenReranker(device="cuda")
    pairs = [("query", str(i)) for i in range(21)]
    forwarded = []

    def qwen_score(self, batch):
        assert len(batch) <= 2
        forwarded.extend(batch)
        return [float(doc) for _, doc in batch]

    def bge_predict(batch, *, batch_size):
        assert batch_size == 8
        assert batch == pairs
        return [float(doc) for _, doc in batch]

    monkeypatch.setattr(Qwen3Reranker, "score", qwen_score)
    ranker.vorauswahl._modell = SimpleNamespace(predict=bge_predict)
    assert ranker.vorauswahl.score(pairs) == list(map(float, range(21)))
    assert ranker.score(pairs) == list(map(float, range(21)))
    assert forwarded == pairs


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


def test_qwen_bewertet_nur_letzte_yes_no_logits_ohne_generation_cache(monkeypatch):
    class Tensor(np.ndarray):
        def float(self):
            return self.astype(np.float32)

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False, "truncation": True, "max_length": 1024}
            assert text.startswith(f"<Instruct>: {Qwen3Reranker.INSTRUCTION}\n<Query>: ")
            return SimpleNamespace(input_ids=[10, 11] if "lang" in text else [10])

        def pad(self, batch, **kwargs):
            assert kwargs == {"padding": True, "return_tensors": "pt"}
            assert batch["input_ids"] == [[7, 10, 8, 9], [7, 10, 11, 8, 9]]
            # Verschiedene Eingabelaengen werden links gepaddet; das letzte
            # Suffix-Token bleibt bei beiden Beispielen am rechten Rand.
            return Inputs(input_ids=np.array([[0, 7, 10, 8, 9], [7, 10, 11, 8, 9]]),
                          attention_mask=np.array([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]))

    def model(*, input_ids, attention_mask, logits_to_keep, use_cache):
        assert logits_to_keep == 1 and use_cache is False
        assert input_ids[:, -1].tolist() == [9, 9]
        assert attention_mask[:, -1].tolist() == [1, 1]
        # Fruehere Positionen und andere Vokabeln waeren absichtlich irrefuehrend.
        # Die letzten no/yes-Logits kodieren dagegen 0.8 bzw. 0.2 fuer yes.
        full = np.array([[[0, 30, 0, -30], [90, 0, 90, np.log(4)]],
                         [[0, -30, 0, 30], [90, np.log(4), 90, 0]]])
        return SimpleNamespace(logits=full[:, -logits_to_keep:, :].view(Tensor))

    def softmax(values, dim):
        assert dim == 1 and values.shape == (2, 2)
        exp = np.exp(values - values.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext, softmax=softmax))
    reranker = Qwen3Reranker(device="cpu")
    reranker._model = model
    reranker._tok = Tokenizer()
    reranker._dev = "cpu"
    reranker._pre, reranker._suf = [7], [8, 9]
    reranker._no, reranker._yes = 1, 3
    assert reranker.score([("kurz", "Dokument"), ("lang", "Dokument")]) == pytest.approx([0.8, 0.2])
